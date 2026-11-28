"""
Model specs: how a caller names a model, as plain data. Used on both sides.

    None                                  the image's own model (its card)
    "esm2"                                {"key": "esm2"}
    "evedesign.models.esm2:ESM2"          {"class": "evedesign.models.esm2:ESM2"}
    {"key" | "class": ..., "args": {...}, "models": {...}, "wrap": [...],
     "build_args": {...}, "data": ...}    a full spec
    {"serialized": <bytes>, "rebuild": bool, "data": ..., "build_args": {...}}
                                          a stored model, loaded instead of constructed

`args` is passed to the constructor as given and never searched for models.
Nested models go in `models`: {argument name: spec or [spec, ...]}.
"""

from collections.abc import Iterator, Mapping
from typing import Any

FIELDS = ("key", "class", "args", "models", "wrap", "build_args", "data", "serialized", "rebuild")
LOADED_FIELDS = frozenset({"serialized", "rebuild", "data", "build_args"})


def as_spec(model: Mapping[str, Any] | str | None) -> dict:
    """A spec from a string, a spec dictionary, or nothing; checked before anything starts."""
    if model is None:
        return {}
    if isinstance(model, str):
        prefix, sep, _ = model.partition(":")
        return {"class": model} if sep and "." in prefix else {"key": model}
    if isinstance(model, Mapping):
        spec = dict(model)
        validate(spec)
        return spec
    raise TypeError(f"name a model by a string, a spec dictionary or nothing, not a {type(model).__name__}")


def validate(spec: Any, where: str = "spec") -> None:
    """Raise if spec is not a valid spec; nested specs in `models` are checked too."""
    if not isinstance(spec, Mapping):
        raise TypeError(f"{where}: a spec is a dictionary, not a {type(spec).__name__}")
    unknown = set(spec) - set(FIELDS)
    if unknown:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}; a spec can have {list(FIELDS)}")
    if "key" in spec and "class" in spec:
        raise ValueError(f"{where}: give either 'key' or 'class', not both")
    for name in ("key", "class"):
        if name in spec and not isinstance(spec[name], str):
            raise TypeError(f"{where}.{name}: must be a string")
    for name in ("args", "build_args", "models"):
        if spec.get(name) is not None and not isinstance(spec[name], Mapping):
            raise TypeError(f"{where}.{name}: must be a dictionary")

    if "serialized" in spec:
        extra = set(spec) - LOADED_FIELDS
        if extra:
            raise ValueError(f"{where}: a model loaded from 'serialized' bytes takes only "
                             f"{sorted(LOADED_FIELDS)}, not {sorted(extra)}")
        if not isinstance(spec["serialized"], bytes):
            raise TypeError(f"{where}.serialized: must be bytes")
        return
    if "rebuild" in spec:
        raise ValueError(f"{where}: 'rebuild' only applies to a model loaded from 'serialized' bytes")

    models = spec.get("models") or {}
    overlap = set(models) & set(spec.get("args") or {})
    if overlap:
        raise ValueError(f"{where}: {sorted(overlap)} given in both 'args' and 'models'")
    for name, value in models.items():
        if isinstance(value, list):
            for i, child in enumerate(value):
                validate(child, f"{where}.models.{name}[{i}]")
        else:
            validate(value, f"{where}.models.{name}")


def children(spec: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """The nested specs of spec, in build order."""
    for value in (spec.get("models") or {}).values():
        yield from (value if isinstance(value, list) else [value])


def describe(spec: Mapping[str, Any]) -> str:
    """A short name for logs and provenance."""
    if "serialized" in spec:
        return "serialized"
    return spec.get("class") or spec.get("key") or "card"
