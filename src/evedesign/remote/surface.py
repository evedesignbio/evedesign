"""
The interface surface a proxy exposes, derived from evedesign's ABCs.

Both sides compute it from the same code, so the proxy's forwarded methods and
the worker's callable methods cannot drift apart. Its hash, together with
PROTOCOL_VERSION, is the compatibility check at session open: an evedesign
release that changes nothing on the interfaces does not force rebuilding images.
"""

import functools
import hashlib
import inspect

PROTOCOL_VERSION = 1

INTERFACE_NAMES: tuple[str, ...] = (
    "_Core", "Generator", "Scorer", "ConditionalMutationScorer", "MutationScorer",
    "Transformer", "BaseModel", "SupervisedBaseModel", "Analyzer",
)

# Answered from state the orchestrator holds, never forwarded.
LOCAL_ONLY = frozenset({"system", "ready", "build"})


@functools.cache
def interfaces() -> dict[str, type]:
    from evedesign import analysis, model

    found = {}
    for name in INTERFACE_NAMES:
        cls = getattr(model, name, None) or getattr(analysis, name, None)
        if cls is not None:
            found[name] = cls
    return found


def interfaces_of(obj_or_cls) -> tuple[str, ...]:
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    known = interfaces()
    return tuple(name for name, iface in known.items() if issubclass(cls, iface))


@functools.cache
def declared_surface(names: tuple[str, ...]) -> tuple[frozenset[str], dict[str, inspect.Signature]]:
    """(attribute names, {method name: signature without self/cls}) for these interfaces."""
    attributes: set[str] = set()
    methods: dict[str, inspect.Signature] = {}
    known = interfaces()

    for name in names:
        for base in known[name].__mro__:
            if base is object or base.__module__ == "abc":
                continue
            for attr, raw in vars(base).items():
                if attr.startswith("_") or attr in LOCAL_ONLY:
                    continue
                if isinstance(raw, property):
                    attributes.add(attr)
                elif isinstance(raw, (classmethod, staticmethod)) or callable(raw):
                    func = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
                    sig = inspect.signature(func)
                    params = list(sig.parameters.values())
                    if not isinstance(raw, staticmethod):
                        params = params[1:]
                    methods.setdefault(attr, sig.replace(parameters=params))
                else:
                    attributes.add(attr)
            attributes.update(
                a for a in getattr(base, "__annotations__", {})
                if not a.startswith("_") and a not in LOCAL_ONLY
            )
    return frozenset(attributes - set(methods)), methods


def all_attribute_names() -> tuple[str, ...]:
    attrs, _ = declared_surface(tuple(interfaces()))
    return tuple(sorted(attrs))


@functools.cache
def surface_hash() -> str:
    """Stable hash of every interface's attributes and method signatures."""
    parts = []
    for name in sorted(interfaces()):
        attrs, methods = declared_surface((name,))
        parts.append(name)
        parts.extend(sorted(attrs))
        parts.extend(f"{m}{list(sig.parameters)}" for m, sig in sorted(methods.items()))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
