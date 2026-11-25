"""
A local stand-in for a model living in another process, container or region.

The orchestrator never imports the model. Interface conformance comes from ABC
registration (`Scorer.register(cls)`), so `isinstance(proxy, Scorer)` holds with
no concrete class present; the interfaces and attribute values come from the
descriptor the worker reports. Declared methods are forwarded by one generic
forwarder that binds arguments against the ABC's own signature, so positional
calls work exactly as they do on a local model.
"""

import functools
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .surface import declared_surface, interfaces
from .transport import Transport


@dataclass
class CallStats:
    calls: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def record(self, method: str, elapsed: float) -> None:
        self.calls[method] = self.calls.get(method, 0) + 1
        self.seconds[method] = self.seconds.get(method, 0.0) + elapsed

    def summary(self) -> str:
        if not self.calls:
            return "no calls"
        parts = [f"{m} x{self.calls[m]} {self.seconds[m]:.2f}s" for m in sorted(self.calls, key=lambda k: -self.seconds[k])]
        return f"{sum(self.seconds.values()):.2f}s over {sum(self.calls.values())} calls ({', '.join(parts)})"


class ProxyNotPicklable(TypeError):
    """Raised when something tries to pickle a proxy; the built state lives on the far side."""


class RemoteModel:
    """Inherits from nothing; conformance is by registration in `proxy_class`."""

    def __init__(self, transport: Transport, descriptor: dict, label: str, provenance: dict) -> None:
        self._transport = transport
        self._label = label
        self._stats = CallStats()
        self._closed = False
        self.provenance = provenance
        self._apply(descriptor)

    def _apply(self, descriptor: dict, system=None) -> None:
        self._descriptor = descriptor
        self._attributes = dict(descriptor["attributes"])
        self.provenance["models"] = descriptor.get("models", [])
        # `system` is local state: set by a successful build(), by open_built's
        # argument, or by a reload's reported System -- never fetched per access.
        if system is not None:
            self._system = system
        else:
            self._system = descriptor.get("system") if descriptor.get("ready") else None

    # -- the declared surface -------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        attributes = self.__dict__.get("_attributes", {})
        if name in attributes:
            return attributes[name]
        descriptor = self.__dict__.get("_descriptor")
        if descriptor is not None:
            _, methods = declared_surface(tuple(descriptor["interfaces"]))
            if name in methods:
                return functools.partial(self._forward, name, methods[name])
        raise AttributeError(
            f"{self.__dict__.get('_label', '<remote>')}: remote model does not expose {name!r}; it is "
            f"not declared by its interfaces ({', '.join(descriptor['interfaces']) if descriptor else '?'})"
        )

    def _forward(self, name: str, signature, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("status_callback", None)  # not serialisable; nobody watches a worker
        return self._call("call", method=name, kwargs=arguments)

    @property
    def system(self):
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None if self._descriptor.get("is_base_model") else True

    def build(self, system, data: Any = None, status_callback=None, **build_args: Any):
        """Forwarded every time, exactly like a local build(); the far side builds here.
        The spec's build_args are the defaults; keyword arguments here override them."""
        build_args = {**((self.provenance.get("spec") or {}).get("build_args") or {}), **build_args}
        descriptor = self._call("build", system=system, data=data, build_args=build_args)
        self._apply(descriptor, system=system)
        return self

    def ready_or_raise(self) -> None:
        if not self.ready:
            raise ValueError("Must call build() first to use model")

    # -- lifecycle ------------------------------------------------------------

    def fetch_serialized(self) -> list[bytes]:
        """dill bytes per model in the session (children first, then this model)."""
        return self._call("serialize")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info(f"{self._label}: transport {self._stats.summary()}")
        self._transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __reduce__(self):
        raise ProxyNotPicklable(f"{self._label}: a remote proxy cannot be pickled; use fetch_serialized()")

    def __repr__(self) -> str:
        return f"<remote {self._descriptor['class_name']} via {self._transport.description}>"

    def _call(self, op: str, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._transport.request(op, **kwargs)
        finally:
            self._stats.record(kwargs.get("method", op), time.perf_counter() - started)


@functools.cache
def proxy_class(interface_names: tuple[str, ...], class_name: str) -> type[RemoteModel]:
    """
    One class per (interfaces, remote class name), registered with each ABC.

    Named after the remote class, because consumers use type(x).__name__ (e.g.
    GibbsSampler keys its score metadata by it), so placed and local runs agree.
    """
    known = interfaces()
    unknown = [n for n in interface_names if n not in known]
    if unknown:
        raise ValueError(f"worker reported interfaces {unknown} unknown to this evedesign")
    cls = type(class_name, (RemoteModel,), {"__module__": "evedesign.remote.proxies"})
    for name in interface_names:
        known[name].register(cls)
    return cls


def make_proxy(transport: Transport, descriptor: dict, label: str, provenance: dict) -> RemoteModel:
    cls = proxy_class(tuple(descriptor["interfaces"]), descriptor["class_name"])
    return cls(transport, descriptor, label, provenance)
