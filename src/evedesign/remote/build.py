"""
Building a spec tree into models, in this process: the worker builds what it is sent with this, and
build_local() builds a tree (or the local part of one) in the orchestrator. Imports nothing from the
remote extras up front; specs naming a model by key read the card with pyyaml, and stored models
(`serialized`) need dill, both in the `remote` extra.

A model is resolved from its spec:

  1. a class path: imported, with the constructor's own defaults (the card does not apply, so a
     spec with a class path means the same in every image);
  2. otherwise the image's model card (README front matter: model_class, hyper_parameters), if it
     describes the model: no key given, or the card's key (else its name) given exactly;
  3. otherwise an error naming the card the image has.

Nested models come from the spec's `models`; `args` are passed on as given.
"""

from importlib import import_module
from pathlib import Path
from typing import Any

from .spec import children, validate

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
    """"package.module:Class", or "package.module.Class" as model cards may also write it."""
    module, _, name = path.partition(":") if ":" in path else path.rpartition(".")
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


def instantiate_tree(spec: dict, system, data: Any, collection: list, records: list,
                     prebuilt: dict[int, Any] | None = None) -> Any:
    """
    Create (or load) and build the model spec names, its nested models first.
    `collection` gets (spec, object) and `records` one record per model, both in
    build order: children first, in `models` order, the parent last.
    `prebuilt` maps id(spec) of nested specs to objects made elsewhere (e.g. models opened on
    other targets): they are used as given, neither created nor built, and recorded as such.
    """
    from evedesign.model import BaseModel

    if prebuilt is not None and id(spec) in prebuilt:
        obj = prebuilt[id(spec)]
        if system is not None and isinstance(obj, BaseModel) and obj.system != system:
            raise ValueError(f"prebuilt model for {spec.get('class') or spec.get('key')!r} was built "
                             f"for a different system")
        collection.append((spec, obj))
        records.append(_prebuilt_record(obj))
        return obj

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
        shown = printable(args)
        for name, value in (spec.get("models") or {}).items():
            if isinstance(value, list):
                args[name] = [instantiate_tree(v, system, v.get("data"), collection, records, prebuilt)
                              for v in value]
            else:
                args[name] = instantiate_tree(value, system, value.get("data"), collection, records, prebuilt)
        obj = cls(**args)
        if isinstance(obj, BaseModel):
            obj.build(system=system, data=data, **build_args)
        record = {**record, "args": shown, "loaded": False, "rebuilt": False}

    record["citations"] = getattr(obj, "citations", None)
    record["stats"] = obj.stats() if isinstance(obj, BaseModel) and obj.ready else None
    collection.append((spec, obj))
    records.append(record)
    return obj


def _prebuilt_record(obj: Any) -> dict:
    """A prebuilt model's record: a remote model's own (from the worker that built it), else from the
    object. Marked loaded, as it was neither created nor built here."""
    from evedesign.model import BaseModel

    remote = getattr(obj, "provenance", None)
    if isinstance(remote, dict) and remote.get("models"):
        own = remote["models"][-1]  # the root of what that worker built
        record = {"class": own["class"], "citations": own.get("citations"), "stats": own.get("stats")}
    else:
        record = {"class": f"{type(obj).__module__}:{type(obj).__qualname__}",
                  "citations": getattr(obj, "citations", None),
                  "stats": obj.stats() if isinstance(obj, BaseModel) and obj.ready else None}
    return {"via": "prebuilt", **record, "loaded": True, "rebuilt": False}


def printable(args: dict) -> dict:
    return {k: (v if isinstance(v, (str, int, float, bool, type(None))) else type(v).__name__) for k, v in args.items()}


def _reachable(spec: dict, prebuilt: dict) -> set[int]:
    """ids of the nodes instantiate_tree will visit: it does not descend into prebuilt nodes."""
    ids = {id(spec)}
    if id(spec) not in prebuilt:
        for child in children(spec):
            ids |= _reachable(child, prebuilt)
    return ids


def build_local(spec: dict, system=None, data: Any = None, prebuilt: dict[int, Any] | None = None):
    """
    Build a spec tree in this process, as a worker would: (model, [(spec, object), ...], records),
    children first. Nodes in `prebuilt` (id of a nested spec -> object) are used as given:

        proxy = open_built("esm2-gpu", esm2_spec, system)
        gibbs, collection, records = build_local(gibbs_spec, system, prebuilt={id(esm2_spec): proxy})
    """
    validate(spec)
    if prebuilt:
        # entries are matched by identity: a copy of the tree (JSON, deepcopy), or a node inside
        # another prebuilt one, would silently go unused, and those models be built here instead
        unused = set(prebuilt) - _reachable(spec, prebuilt)
        if unused:
            raise ValueError(f"{len(unused)} prebuilt entr{'y is' if len(unused) == 1 else 'ies are'} not a "
                             f"spec this tree builds; key them by id() of the tree's own nested spec dicts")
    collection, records = [], []
    model = instantiate_tree(spec, system, data if data is not None else spec.get("data"),
                             collection, records, prebuilt)
    return model, collection, records
