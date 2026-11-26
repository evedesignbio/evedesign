"""
The far side of a session: one model (or one tree of models) per process.

Started as `python -m evedesign.remote.worker` (images need only evedesign[remote]
and their Python on PATH). Speaks length-prefixed frames on stdin/stdout and exits
on EOF, so a killed orchestrator never leaves it running.

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
import sys
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

    # --- ops --------------------------------------------------------------

    def op_hello(self, protocol_version: int, surface: str, seed: int | None = None,
                 launcher: dict | None = None) -> dict:
        from .provenance import collect_environment

        if protocol_version != PROTOCOL_VERSION or surface != surface_hash():
            raise RuntimeError(
                f"incompatible worker: protocol {PROTOCOL_VERSION} surface {surface_hash()}, "
                f"orchestrator sent protocol {protocol_version} surface {surface}. Rebuild the "
                f"image against a compatible evedesign."
            )
        self.seed = seed
        seed_everything(seed)
        return {"environment": collect_environment(seed=seed), "launcher": launcher}

    def _devices(self, devices: dict | None, uses_gpu: bool) -> None:
        """Pick the offered CUDA_VISIBLE_DEVICES before anything initialises CUDA."""
        self.cuda_visible = (devices or {}).get("gpu" if uses_gpu else "cpu")
        if self.cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible

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

    def op_build(self, system, data: Any = None, build_args: dict | None = None) -> dict:
        self.instance.build(system=system, data=data, **(build_args or {}))
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


def handle(session: Session, message: dict) -> dict:
    """One call -> its reply. Everything raised goes back to the caller."""
    try:
        op = message["method"]
        if op == "close":
            return {"id": message["id"], "type": "result", "value": None}
        handler = getattr(session, f"op_{op}", None)
        if handler is None:
            raise ValueError(f"unknown operation {op!r}")
        return {"id": message["id"], "type": "result", "value": handler(**message.get("kwargs", {}))}
    except Exception as exc:  # noqa: BLE001
        return _error(message, exc)


def serve(stdin=None, stdout=None) -> None:
    # The protocol owns the original stdout; anything a model prints goes to stderr.
    if stdout is None:
        stdout = os.fdopen(os.dup(1), "wb", buffering=0)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
    stdin = stdin or sys.stdin.buffer

    stdout.write(protocol.READY)
    stdout.flush()
    session = Session()
    while True:
        try:
            message = protocol.read_frame(stdin)
        except EOFError:
            return
        if message.get("type") != "call":
            continue
        reply = handle(session, message)
        try:
            protocol.write_frame(stdout, reply)
        except Exception as exc:  # noqa: BLE001 -- e.g. a result the codec cannot encode
            protocol.write_frame(stdout, _error(message, exc))
        if message["method"] == "close":
            return


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] != "serve":  # `serve` is accepted for images whose entrypoint passes it
        raise SystemExit(f"usage: python -m evedesign.remote.worker [serve]  (got {argv})")
    serve()


if __name__ == "__main__":
    main()
