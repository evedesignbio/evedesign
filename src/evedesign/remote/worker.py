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
from importlib import import_module
from pathlib import Path
from typing import Any

from . import protocol
from .spec import children, validate
from .surface import PROTOCOL_VERSION, all_attribute_names, declared_surface, interfaces_of, surface_hash

CARD_PATH = Path("/opt/program/README.md")  # proteingym-models layout


# ---------------------------------------------------------------------------
# spec resolution
# ---------------------------------------------------------------------------

def read_card() -> tuple[dict, str] | None:
    """(front matter, sha256 of the card file) for the image's model card, if any."""
    import hashlib

    import yaml

    if not CARD_PATH.is_file():
        return None
    text = CARD_PATH.read_text()
    if not text.startswith("---"):
        return None
    return yaml.safe_load(text.split("---", 2)[1]) or {}, hashlib.sha256(text.encode()).hexdigest()


def _import(path: str) -> type:
    module, _, name = path.partition(":")
    return getattr(import_module(module), name)


def card_describes(meta: dict, spec: dict) -> bool:
    """Whether the card is for the model a spec without a class path names: no key, or the
    card's key (else its name), exactly."""
    held = meta.get("key", meta.get("name"))
    return "model_class" in meta and spec.get("key", held) == held


def resolve(spec: dict) -> tuple[type, dict, dict]:
    """
    (class, constructor args without nested models, record for provenance).
    The card's hyper_parameters are the defaults when it describes the model (never
    for a class path); the spec's args override them. Then the spec's wrap factories are applied.
    """
    args = dict(spec.get("args") or {})
    card = None if "class" in spec else read_card()
    if "class" in spec:
        cls, record = _import(spec["class"]), {"via": "class", "class": spec["class"]}
    elif card is not None and card_describes(card[0], spec):
        meta, card_hash = card
        cls, record = _import(meta["model_class"]), {
            "via": "model_card", "class": meta["model_class"], "card_sha256": card_hash}
        args = {**(meta.get("hyper_parameters") or {}), **args}
    else:
        name = spec.get("key")
        held = card[0].get("key", card[0].get("name")) if card else None
        raise LookupError(
            f"cannot resolve model {name!r}: "
            + (f"the image's model card is for {held!r}" if card else "the image has no model card")
            + "; name the model by its class path, or by the card's key"
        )
    if spec.get("wrap"):
        from .wrapping import apply_wraps, normalize

        cls, record = apply_wraps(cls, spec["wrap"]), {**record, "wrap": normalize(spec["wrap"])}
    return cls, args, record


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


def instantiate_tree(spec: dict, system, data: Any, collection: list, records: list) -> Any:
    """
    Create (or load) and build the model spec names, its nested models first.
    `collection` gets (spec, object) and `records` one record per model, both in
    build order: children first, in `models` order, the parent last.
    """
    from evedesign.model import BaseModel

    build_args = spec.get("build_args") or {}
    if "serialized" in spec:
        import dill

        obj = dill.loads(spec["serialized"])
        if spec.get("rebuild"):
            obj.build(system=system, data=data, **build_args)
        elif system is not None and isinstance(obj, BaseModel) and obj.system != system:
            raise ValueError("the system of the loaded model does not agree with the given system")
        record = {"via": "serialized", "class": f"{type(obj).__module__}:{type(obj).__qualname__}",
                  "loaded": True, "rebuilt": bool(spec.get("rebuild"))}
    else:
        cls, args, record = resolve(spec)
        printable = _printable(args)
        for name, value in (spec.get("models") or {}).items():
            if isinstance(value, list):
                args[name] = [instantiate_tree(v, system, v.get("data"), collection, records) for v in value]
            else:
                args[name] = instantiate_tree(value, system, value.get("data"), collection, records)
        obj = cls(**args)
        if isinstance(obj, BaseModel):
            obj.build(system=system, data=data, **build_args)
        record = {**record, "args": printable, "loaded": False, "rebuilt": False}

    record["citations"] = getattr(obj, "citations", None)
    record["stats"] = obj.stats() if isinstance(obj, BaseModel) and obj.ready else None
    collection.append((spec, obj))
    records.append(record)
    return obj


def _printable(args: dict) -> dict:
    return {k: (v if isinstance(v, (str, int, float, bool, type(None))) else type(v).__name__) for k, v in args.items()}


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
        self.records = [{**record, "args": _printable(args), "loaded": False, "rebuilt": False}]
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
