"""
The far side of a session: one model (or one co-located tree) per process.

Started as `python -m evedesign.remote.worker [serve]` (in images, via
docker/entrypoint.sh `serve`). Speaks length-prefixed frames on stdin/stdout and
exits on EOF, so a killed orchestrator never leaves it running.

Construction is resolved here, so the far side is authoritative about what a
model key means:

  1. spec["class"] = "module:Class"                 explicit
  2. the model card in the image (README front matter: model_class,
     hyper_parameters), if the spec's key matches the card or names none
  3. installed resolver plugins, e.g. evedesign-server's model table

Plugins are entry points in the group "evedesign.remote", so this package never
imports a package that depends on it:

  resolve        (spec: dict, transform: bool) -> (class, args, record) | None
  pipeline_tree  (spec, system, schema, stored_models, server_version)
                     -> (instance, [(spec, obj), ...], records)
"""

import os
import random
import sys
import traceback
from importlib import import_module
from pathlib import Path
from typing import Any

from . import protocol
from .surface import PROTOCOL_VERSION, all_attribute_names, declared_surface, interfaces_of, surface_hash

SPEC_FIELDS = frozenset({"key", "class", "variant", "args", "data", "id", "source_id", "rebuild_on_load",
                         "stats", "citations", "wrap", "build_args"})
CARD_PATH = Path("/opt/program/README.md")  # proteingym-models layout


# ---------------------------------------------------------------------------
# spec resolution
# ---------------------------------------------------------------------------

def is_spec(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and ("class" in value or isinstance(value.get("key"), str))
        and set(value) <= SPEC_FIELDS
    )


def has_nested_specs(spec: dict) -> bool:
    args = spec.get("args") or {}
    return any(
        is_spec(v) or (isinstance(v, list) and any(is_spec(e) for e in v))
        for v in args.values()
    )


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


def resolve(spec: dict, *, transform: bool = True) -> tuple[type, dict, dict]:
    """_resolve, then the spec's wrap factories (see wrapping.py)."""
    cls, args, record = _resolve(spec, transform=transform)
    if spec.get("wrap"):
        from .wrapping import apply_wraps, normalize

        cls, record = apply_wraps(cls, spec["wrap"]), {**record, "wrap": normalize(spec["wrap"])}
    return cls, args, record


def _resolve(spec: dict, *, transform: bool = True) -> tuple[type, dict, dict]:
    """
    (class, constructor args without nested specs, resolution record for provenance).

    The card resolves a spec that names no key, or whose key the card claims
    (its `key`, `name` or class name). Other keys go to the model table; that
    way a placed sampler in a tree, or a key the image does not hold, cannot be
    silently constructed as the image's own model.
    """
    args = dict(spec.get("args") or {})

    if "class" in spec:
        return _import(spec["class"]), args, {"via": "class", "class": spec["class"]}

    card = read_card()
    if card is not None and "model_class" in card[0]:
        meta, card_hash = card
        if _card_claims(meta, spec.get("key")):
            merged = {**(meta.get("hyper_parameters") or {}), **args}
            return _import(meta["model_class"]), merged, {
                "via": "model_card", "class": meta["model_class"], "card_sha256": card_hash,
            }

    for plugin in _plugins("resolve"):
        resolved = plugin(spec, transform)
        if resolved is not None:
            return resolved

    card_name = card[0].get("key", card[0].get("name")) if card else None
    raise LookupError(
        f"cannot resolve model {spec.get('key')!r}: no 'class' given, "
        + (f"the image's model card is for {card_name!r} (add `key: {spec.get('key')}` to the "
           f"card if it is the same model), " if card else "there is no model card, ")
        + "and no installed resolver knows it (evedesign-server provides the model table)"
    )


def _plugins(name: str) -> list:
    from importlib.metadata import entry_points

    return [ep.load() for ep in entry_points(group="evedesign.remote", name=name)]


def _card_claims(meta: dict, key: str | None) -> bool:
    """A card claims a key equal to its `key`, its `name`, or its class name (case-insensitive)."""
    if key is None:
        return True
    if "key" in meta:
        return meta["key"] == key
    norm = lambda s: str(s).lower().replace("-", "").replace("_", "")  # noqa: E731
    return norm(key) in {norm(meta.get("name", "")), norm(meta["model_class"].rpartition(":")[2])}


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
    Core equivalent of evedesign_server's instantiate_class, minus the repository:
    children are instantiated (and built) first, then the parent is constructed
    with them and built if it is a BaseModel. `collection` gets (spec, object) in
    the same order instantiate_class appends them: children before parent.
    """
    from evedesign.model import BaseModel

    cls, args, record = resolve(spec)
    for name, value in list(args.items()):
        if is_spec(value):
            args[name] = instantiate_tree(value, system, value.get("data"), collection, records)
        elif isinstance(value, list) and value and is_spec(value[0]):
            args[name] = [instantiate_tree(v, system, v.get("data"), collection, records) for v in value]

    obj = cls(**args)
    if isinstance(obj, BaseModel):
        obj.build(system=system, data=data, **(spec.get("build_args") or {}))
    collection.append((spec, obj))
    records.append({**record, "key": spec.get("key"), "args": _printable(args)})
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
        if has_nested_specs(spec):
            raise ValueError("spec has nested model specs; use open_built")
        cls, args, record = resolve(spec)
        self._devices(devices, _uses_gpu(cls))
        self.instance = cls(**args)
        self.records = [{**record, "key": spec.get("key"), "args": _printable(args)}]
        self.collection = [(spec, self.instance)]
        return self.describe()

    def op_open_built(self, spec: dict, system, data: Any = None, schema: str | None = None,
                      devices: dict | None = None, stored_models: dict | None = None,
                      server_version: str | None = None, uses_gpu: bool | None = None) -> dict:
        if schema is not None and uses_gpu is None:
            # a pipeline tree is only interpreted by the pipeline's own validated code, never
            # walked here as raw dicts: its specs may carry user data that looks like specs
            raise ValueError("a pipeline tree needs uses_gpu from the orchestrator")
        self._devices(devices, _tree_uses_gpu(spec) if uses_gpu is None else uses_gpu)
        self.collection, self.records = [], []
        if schema is not None:
            self._open_pipeline_tree(spec, system, schema, stored_models or {}, server_version)
        else:
            root_data = data if data is not None else spec.get("data")
            self.instance = instantiate_tree(spec, system, root_data, self.collection, self.records)
        return self.describe()

    def _open_pipeline_tree(self, spec: dict, system, schema: str, stored_models: dict,
                            server_version: str | None) -> None:
        """A pipeline spec tree, instantiated by the pipeline's own code (a plugin)."""
        plugins = _plugins("pipeline_tree")
        if not plugins:
            raise RuntimeError(
                "placing a pipeline spec with nested models needs evedesign-server in the image "
                "(it runs the server's own instantiate_class). Use an image that has it, or place "
                "the nested models individually and keep the parent local."
            )
        self.instance, self.collection, self.records = plugins[0](
            spec, system, schema, stored_models, server_version)

    def op_reload(self, blob: bytes, devices: dict | None = None) -> dict:
        self._devices(devices, True)  # unknown until loaded, and loading may already place tensors
        import dill

        self.instance = dill.loads(blob)
        self.collection = [({}, self.instance)]
        self.records = [{"via": "reload", "class": f"{type(self.instance).__module__}:{type(self.instance).__qualname__}"}]
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
    wrapped = getattr(cls, "_evedesign_wrap", None)
    if wrapped is not None:  # wrappers delegate (as properties): the base class declares it
        cls = _import(wrapped["base"])
    declared = [getattr(cls, a, False) for a in ("requires_gpu", "supports_gpu")]
    return any(d is True or not isinstance(d, bool) for d in declared)


def _tree_uses_gpu(spec: dict) -> bool:
    """Whether any model of a tree may use a GPU. Pipeline keys a plugin cannot resolve: assume so."""
    try:
        cls, args, _ = resolve(spec, transform=False)  # transforms expect the pipeline's objects
    except LookupError:
        return True
    nested = [v for value in args.values() for v in (value if isinstance(value, list) else [value])]
    return _uses_gpu(cls) or any(_tree_uses_gpu(n) for n in nested if is_spec(n))


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
    if argv and argv[0] not in ("serve",):
        raise SystemExit(f"usage: python -m evedesign.remote.worker [serve]  (got {argv})")
    serve()


if __name__ == "__main__":
    main()
