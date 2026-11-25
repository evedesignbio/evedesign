"""
Class factories named in a spec's "wrap": factory(cls, **args) -> cls, applied
in list order, first entry innermost. An entry is "module:factory" or
{"factory": "module:factory", "args": {...}}. Wrap args configure the wrapper
and must not depend on the system; those go in the spec's build_args.

A factory is an import and a call: where specs come from users, pass `allowed`,
which is checked before anything is imported.
"""

import functools
import inspect
import json
import threading
from collections.abc import Collection
from importlib import import_module

_LOCK = threading.Lock()


def normalize(wrap) -> list[dict]:
    """Entries as [{"factory": ..., "args": {...}}]."""
    return [{"factory": e, "args": {}} if isinstance(e, str) else {"factory": e["factory"], "args": e.get("args") or {}}
            for e in wrap or []]


def apply_wraps(cls: type, wrap, allowed: Collection[str] | None = None) -> type:
    """cls wrapped by each factory in turn. `allowed`: factories that may be used (None: any)."""
    entries = normalize(wrap)
    if allowed is not None:
        disallowed = sorted({e["factory"] for e in entries} - set(allowed))
        if disallowed:
            raise PermissionError(f"wrap factories not allowed: {disallowed}")
    for entry in entries:
        try:  # args as canonical JSON: plain data only, and argument order makes no difference
            args = json.dumps(entry["args"], sort_keys=True)
        except TypeError as exc:
            raise TypeError(f"wrap factory {entry['factory']}: args must be plain data ({exc})") from None
        with _LOCK:  # so equal specs give the same class
            cls = _wrap(cls, entry["factory"], args)
    return cls


@functools.lru_cache(maxsize=256)  # bounded, since args may come from users
def _wrap(cls: type, factory_path: str, args_json: str) -> type:
    module, _, name = factory_path.partition(":")
    factory = getattr(import_module(module), name)
    args = json.loads(args_json)  # a fresh copy for the factory
    try:
        inspect.signature(factory).bind(cls, **args)
    except TypeError as exc:
        raise TypeError(f"wrap factory {factory_path}: {exc}") from None
    wrapped = factory(cls, **args)
    if wrapped is not cls:
        # how this class was made, for as_spec(cls); from __dict__ since subclasses inherit it
        inner = vars(cls).get("_evedesign_wrap")
        wrapped._evedesign_wrap = {
            "base": inner["base"] if inner else f"{cls.__module__}:{cls.__qualname__}",
            "wrap": (inner["wrap"] if inner else []) + [{"factory": factory_path, "args": json.loads(args_json)}],
        }
    return wrapped
