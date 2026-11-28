"""
The far side of a session: one model (or one tree of models) per process.

Started as `python -m evedesign.remote.worker` (images need only evedesign[remote]
and their Python on PATH). Speaks length-prefixed frames on stdin/stdout and exits
on EOF, or, where a dead orchestrator causes no EOF (Modal), once its lease runs out:
no message for `lease` seconds while idle (the orchestrator pings meanwhile).

A model is resolved here, so the image is authoritative about what it holds:

  1. a class path: imported, with the constructor's own defaults (the card does not
     apply, so a spec with a class path means the same in every image);
  2. otherwise the image's model card (README front matter: model_class,
     hyper_parameters), if it describes the model: no key given, or the card's key
     (else its name) given exactly;
  3. otherwise an error naming the card the image has.

Nested models come from the spec's `models`; `args` are passed on as given.
"""

import os
import random
import signal
import sys
import threading
import time
import traceback
from typing import Any

from . import protocol
from .build import instantiate_tree, printable, resolve
from .spec import children, validate
from .surface import PROTOCOL_VERSION, all_attribute_names, declared_surface, interfaces_of, surface_hash

def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    import importlib.util

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % 2**32)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self) -> None:
        self.instance: Any = None
        self.collection: list = []
        self.records: list = []
        self.seed: int | None = None
        self.cuda_visible: str | None = None
        self.cuda: dict = {}
        # the lease: with no message for this long while idle, the orchestrator is taken to be gone.
        # From hello; a launcher may set it in the environment, so it holds before hello too.
        self.lease: float | None = float(os.environ.get(protocol.LEASE, 0)) or None
        self.busy = False
        self.last = time.monotonic()

    def touch(self) -> None:
        self.last = time.monotonic()

    # --- ops --------------------------------------------------------------

    def op_hello(self, protocol_version: int, surface: str, seed: int | None = None,
                 launcher: dict | None = None, lease: float | None = None) -> dict:
        from .provenance import collect_environment

        if protocol_version != PROTOCOL_VERSION or surface != surface_hash():
            raise RuntimeError(
                f"incompatible worker: protocol {PROTOCOL_VERSION} surface {surface_hash()}, "
                f"orchestrator sent protocol {protocol_version} surface {surface}. Rebuild the "
                f"image against a compatible evedesign."
            )
        self.seed, self.lease = seed, lease
        seed_everything(seed)
        return {"environment": collect_environment(seed=seed), "launcher": launcher}

    def op_ping(self) -> None:
        return None

    def _devices(self, devices: dict | None, uses_gpu: bool) -> None:
        """Pick the offered CUDA_VISIBLE_DEVICES before anything initialises CUDA; only then
        ask torch about CUDA, which initialises it."""
        from .provenance import collect_cuda

        self.cuda_visible = (devices or {}).get("gpu" if uses_gpu else "cpu")
        if self.cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible
        self.cuda = collect_cuda()

    def op_open_model(self, spec: dict, devices: dict | None = None) -> dict:
        validate(spec)
        if spec.get("models"):
            raise ValueError("a spec with nested models is opened with open_built")
        if "serialized" in spec:
            raise ValueError("a stored model is opened with open_built or reload_model")
        cls, args, record = resolve(spec)
        self._devices(devices, _uses_gpu(cls))
        self.instance = cls(**args)
        self.records = [{**record, "args": printable(args), "loaded": False, "rebuilt": False,
                         "citations": getattr(self.instance, "citations", None), "stats": None}]
        self.collection = [(spec, self.instance)]
        return self.describe()

    def op_open_built(self, spec: dict, system=None, data: Any = None, devices: dict | None = None) -> dict:
        validate(spec)
        if system is None and not ("serialized" in spec and not spec.get("rebuild")):
            raise ValueError("open_built needs a system, unless it loads a stored model without rebuilding it")
        self._devices(devices, _tree_uses_gpu(spec))
        self.collection, self.records = [], []
        root_data = data if data is not None else spec.get("data")
        self.instance = instantiate_tree(spec, system, root_data, self.collection, self.records)
        return self.describe()

    def op_build(self, system, data: Any = None, build_args: dict | None = None, status_callback=None) -> dict:
        extra = {"status_callback": status_callback} if status_callback is not None else {}
        self.instance.build(system=system, data=data, **(build_args or {}), **extra)
        # the model's record follows its build, as for a tree built by open_built
        self.records[-1].update(citations=getattr(self.instance, "citations", None), stats=self.instance.stats())
        return self.describe()

    def op_call(self, method: str, kwargs: dict) -> Any:
        _, methods = declared_surface(interfaces_of(self.instance))
        if method not in methods:
            raise AttributeError(f"{method!r} is not declared by the model's interfaces")
        return getattr(self.instance, method)(**kwargs)

    def op_serialize(self) -> list[bytes]:
        """dill bytes for every model in the session, in instantiate_class collection order."""
        import dill

        return [dill.dumps(obj) for _, obj in self.collection]

    def describe(self) -> dict:
        from evedesign.model import BaseModel

        inst = self.instance
        attributes = {}
        for name in all_attribute_names():
            try:
                value = getattr(inst, name)
            except Exception:  # noqa: BLE001 -- e.g. a property that needs build()
                continue
            if not callable(value):
                attributes[name] = value
        ready = bool(getattr(inst, "ready", True)) if isinstance(inst, BaseModel) else True
        system = getattr(inst, "system", None) if ready else None
        return {
            "interfaces": list(interfaces_of(inst)),
            "class_name": type(inst).__name__,
            "class_path": f"{type(inst).__module__}:{type(inst).__qualname__}",
            "is_base_model": isinstance(inst, BaseModel),
            "attributes": attributes,
            "ready": ready,
            "system": system,
            "models": self.records,
            "cuda_visible_devices": self.cuda_visible,
            "environment": self.cuda,
        }


def _uses_gpu(cls) -> bool:
    """From the class's declarations; not a plain bool (e.g. a property): assume it may."""
    # wrappers (e.g. system_subset_model) delegate through properties: ask the wrapped class
    while "__wrapped__" in vars(cls):  # own attribute only: a subclass of a wrapper is not one
        cls = cls.__wrapped__
    declared = [getattr(cls, a, False) for a in ("requires_gpu", "supports_gpu")]
    return any(d is True or not isinstance(d, bool) for d in declared)


def _tree_uses_gpu(spec: dict) -> bool:
    """Whether any model of a tree may use a GPU; a stored model's class is unknown until loaded."""
    if "serialized" in spec:
        return True
    cls, _, _ = resolve(spec)
    return _uses_gpu(cls) or any(_tree_uses_gpu(child) for child in children(spec))


def _error(message: dict, exc: BaseException) -> dict:
    return {"id": message["id"], "type": "error", "error_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc()}


class _Out:
    """The worker's output: one reply per call, and status messages while a call runs."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.lock = threading.Lock()
        self.call_id = None  # the call whose status messages are sent
        self.sent = 0.0

    def status_callback(self, call_id):
        def report(status, progress=None, message=None) -> None:
            now = time.monotonic()
            with self.lock:
                # at most 5 a second, except the last; nothing once the reply is out
                if self.call_id != call_id or (status == "running" and now - self.sent < 0.2):
                    return
                self.sent = now
                try:
                    protocol.write_frame(self.stream, {"id": call_id, "type": "status", "status": status,
                                                       "progress": progress, "message": message})
                except Exception:  # noqa: BLE001 -- status is best effort
                    pass

        self.call_id, self.sent = call_id, 0.0
        return report

    def reply(self, message: dict, reply: dict) -> None:
        with self.lock:
            self.call_id = None
            try:
                protocol.write_frame(self.stream, reply)
            except Exception as exc:  # noqa: BLE001 -- e.g. a result the codec cannot encode
                protocol.write_frame(self.stream, _error(message, exc))


def handle(session: Session, message: dict, out: _Out | None = None) -> dict:
    """One call -> its reply. Everything raised goes back to the caller."""
    try:
        op = message["method"]
        if op == "close":
            return {"id": message["id"], "type": "result", "value": None}
        handler = getattr(session, f"op_{op}", None)
        if handler is None:
            raise ValueError(f"unknown operation {op!r}")
        kwargs = dict(message.get("kwargs", {}))
        if message.get("status") and out is not None and op in ("call", "build"):
            callback = out.status_callback(message["id"])
            if op == "call":
                kwargs["kwargs"] = {**kwargs["kwargs"], "status_callback": callback}
            else:
                kwargs["status_callback"] = callback
        return {"id": message["id"], "type": "result", "value": handler(**kwargs)}
    except Exception as exc:  # noqa: BLE001
        return _error(message, exc)


class _Watched:
    """stdin, noting when bytes last arrived: a large frame still arriving is not idleness."""

    def __init__(self, stream, session: Session) -> None:
        self._stream, self._session = stream, session

    def read(self, n: int) -> bytes:
        data = self._stream.read1(n)
        self._session.touch()
        return data


def _watch(session: Session) -> None:
    while True:
        time.sleep(1)
        lease = session.lease
        if lease and not session.busy and time.monotonic() - session.last > lease:
            _expire(lease)


def _expire(lease: float) -> None:
    sys.stderr.write(f"evedesign worker: no message for {lease:g}s; the orchestrator is gone, exiting\n")
    sys.stderr.flush()
    if os.environ.get(protocol.END_CONTAINER) == "1":
        os.kill(1, signal.SIGTERM)  # Modal: the Sandbox's init passes it on and exits, ending the Sandbox
    os._exit(0)


def serve(stdin=None, stdout=None) -> None:
    # The protocol owns the original stdout; anything a model prints goes to stderr.
    if stdout is None:
        stdout = os.fdopen(os.dup(1), "wb", buffering=0)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
    session = Session()
    stdin = _Watched(stdin or sys.stdin.buffer, session)
    out = _Out(stdout)

    stdout.write(protocol.READY)
    stdout.flush()
    threading.Thread(target=_watch, args=(session,), daemon=True).start()
    while True:
        try:
            message = protocol.read_frame(stdin)
        except EOFError:
            return
        if message.get("type") != "call":
            continue
        session.busy = True
        out.reply(message, handle(session, message, out))
        session.busy = False
        session.touch()
        if message["method"] == "close":
            return


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] != "serve":  # `serve` is accepted for images whose entrypoint passes it
        raise SystemExit(f"usage: python -m evedesign.remote.worker [serve]  (got {argv})")
    serve()


if __name__ == "__main__":
    main()
