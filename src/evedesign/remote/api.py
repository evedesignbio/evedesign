"""
Two entry points: an unbuilt model, as its constructor gives you, or a built one.

    open_model(target, model, **args)              == the constructor; build() is forwarded
    open_built(target, model, system, data, **args) == construct + build there, nested models included

Direct use:

    with session("infra.yaml", seed=1234) as s:
        evm = s.open_model("evmutation2", "evmutation2")
        esm = s.open_model("docker:esm2", "evedesign.models.esm2:ESM2", device="cuda")
        s.build_all(system)
        designs = GibbsSampler(scorers=[evm, esm], weights=[1, .5]).generate(16)
"""

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Self

from . import provenance as prov
from .config import Infra, Target, derive_seed
from .proxy import RemoteModel, make_proxy
from .surface import PROTOCOL_VERSION, surface_hash
from .transport import launch, offer_devices, settle_devices


InfraLike = Infra | Mapping[str, Any] | str | Path | None
TargetLike = Target | Mapping[str, Any] | str


def _as_infra(infra: InfraLike) -> Infra:
    return Infra.coerce(infra)


def _open(target: TargetLike, op: str, kwargs: dict, seed: int | None, infra, label: str | None) -> RemoteModel:
    infra = _as_infra(infra)
    target = infra.resolve(target)
    launcher = prov.pin(target)
    transport = launch(target, launcher.get("pinned_image"))
    try:
        hello = transport.request(
            "hello", protocol_version=PROTOCOL_VERSION, surface=surface_hash(), seed=seed, launcher=launcher,
        )
        devices = offer_devices(target)
        descriptor = transport.request(op, **kwargs, devices=devices.as_message())
        settle_devices(devices, descriptor["cuda_visible_devices"])
        launcher["cuda_visible_devices"] = descriptor["cuda_visible_devices"]
    except BaseException:
        transport.close()
        raise
    record = {"launcher": launcher, "environment": hello["environment"], "spec": kwargs.get("spec")}
    proxy = make_proxy(transport, descriptor, label or f"{descriptor['class_name']}@{target.name}", record)
    return proxy


ModelLike = Mapping[str, Any] | str | type | None


def as_spec(model: ModelLike, args: Mapping[str, Any] | None = None) -> dict:
    """
    Normalise how a model is named into a spec, merging constructor arguments.

        "esm2"                                   model key (the image's card or the model table)
        "evedesign.models.esm2:ESM2"             module:Class
        ESM2                                     a class, if importable here (only its path is used)
        {"key": ..., "variant": ..., "args": ...} a full spec; nested specs for open_built
        None                                     the image's own model (its card)
    """
    if isinstance(model, Mapping):
        spec = dict(model)
    elif isinstance(model, type) and "_evedesign_wrap" in vars(model):  # made by evedesign.remote.wrapping
        spec = {"class": model._evedesign_wrap["base"], "wrap": model._evedesign_wrap["wrap"]}
    elif isinstance(model, type):
        spec = {"class": f"{model.__module__}:{model.__qualname__}"}
    elif isinstance(model, str):
        prefix, sep, _ = model.partition(":")
        spec = {"class": model} if sep and "." in prefix else {"key": model}
    elif model is None:
        spec = {}
    else:
        raise TypeError(f"cannot name a model by a {type(model).__name__}")
    if args:
        spec["args"] = {**(spec.get("args") or {}), **args}
    return spec


def open_model(target: TargetLike, model: ModelLike = None, /, *, seed: int | None = None,
               infra: InfraLike = None, label: str | None = None,
               args: Mapping[str, Any] | None = None, **constructor_args: Any) -> RemoteModel:
    """
    Construct one model on the far side, like calling its constructor:

        esm = open_model("esm2-gpu", "esm2", device="cuda", batch_size=256)
        esm.build(system, data=None)          # not built until you say so, as locally

    Keyword arguments go to the constructor (over the image card's defaults).
    A constructor argument named like an option here (seed, infra, label, args)
    goes in `args={...}` instead.
    """
    spec = as_spec(model, {**(args or {}), **constructor_args})
    return _open(target, "open_model", {"spec": spec}, seed, infra, label)


def open_built(target: TargetLike, model: ModelLike, system, data: Any = None, /, *,
               seed: int | None = None, infra: InfraLike = None, label: str | None = None,
               schema: str | None = None, stored_models: Mapping[str, bytes] | None = None,
               server_version: str | None = None, uses_gpu: bool | None = None,
               args: Mapping[str, Any] | None = None, **constructor_args: Any) -> RemoteModel:
    """
    Construct and build a model on the far side, with any nested models, like
    instantiate_class: nested models are built against `system` first, then the
    model is constructed with them and built (if it is a BaseModel). Returns it
    ready, so the whole model, a sampling loop with its scorers say, runs there.

        sampler = open_built("full-gpu", GibbsSampler, system,
                             scorers=[{"key": "evmutation2"}, {"key": "esm2", "args": {"device": "cuda"}}],
                             weights=[1.0, 0.5], num_sweeps=100)

    Nested models are given as specs (mappings), not as proxies. `schema`,
    `stored_models`, `server_version` and `uses_gpu` are set by the pipeline
    integration (a worker does not interpret a pipeline tree's raw specs itself).
    """
    spec = as_spec(model, {**(args or {}), **constructor_args})
    kwargs = {"spec": spec, "system": system, "data": data, "schema": schema,
              "stored_models": dict(stored_models or {}), "server_version": server_version,
              "uses_gpu": uses_gpu}
    proxy = _open(target, "open_built", kwargs, seed, infra, label)
    proxy._apply(proxy._descriptor, system=system)
    return proxy


def reload_model(target: TargetLike, blob: bytes, *, seed: int | None = None,
                 infra: InfraLike = None, label: str | None = None) -> RemoteModel:
    """Deserialise a stored model inside the target environment (where its classes import)."""
    return _open(target, "reload", {"blob": blob}, seed, infra, label)


def probe(target: TargetLike, infra: InfraLike = None) -> dict:
    """Start a target's worker, handshake, stop it. What `evedesign-remote check` reports."""
    infra = _as_infra(infra)
    target = infra.resolve(target)
    launcher = prov.pin(target)
    transport = launch(target, launcher.get("pinned_image"))
    try:
        hello = transport.request("hello", protocol_version=PROTOCOL_VERSION, surface=surface_hash(), launcher=launcher)
    finally:
        transport.close()
    return {"launcher": launcher, "environment": hello["environment"]}


class Session:
    """Owns every model opened through it and closes them on exit, failure included."""

    def __init__(self, infra: InfraLike = None, seed: int | None = None) -> None:
        self.infra = _as_infra(infra)
        self.seed = seed
        self._stack = ExitStack()
        self.models: list[RemoteModel] = []
        self._counts: dict[str, int] = {}

    def _seed(self, model: ModelLike, explicit: int | None) -> int | None:
        """An explicit seed wins; otherwise derived from the session seed and opening order per model."""
        if explicit is not None or self.seed is None:
            return explicit
        spec = as_spec(model)
        name = spec.get("class") or spec.get("key") or "card"
        n = self._counts.get(name, 0)
        self._counts[name] = n + 1
        return derive_seed(self.seed, f"{name}#{n}")

    def _keep(self, proxy: RemoteModel) -> RemoteModel:
        self.models.append(proxy)
        self._stack.callback(proxy.close)
        return proxy

    def open_model(self, target: TargetLike, model: ModelLike = None, /, *, seed: int | None = None,
                   **kwargs: Any) -> RemoteModel:
        """As module-level open_model; closed with the session, seeded from the session seed."""
        return self._keep(open_model(target, model, seed=self._seed(model, seed), infra=self.infra, **kwargs))

    def open_built(self, target: TargetLike, model: ModelLike, system, data: Any = None, /, *,
                   seed: int | None = None, **kwargs: Any) -> RemoteModel:
        """As module-level open_built; closed with the session, seeded from the session seed."""
        return self._keep(open_built(target, model, system, data, seed=self._seed(model, seed),
                                     infra=self.infra, **kwargs))

    def build_all(self, system, data: Any = None) -> None:
        """Build every unbuilt model concurrently; isolated builds are independent."""
        from concurrent.futures import ThreadPoolExecutor

        pending = [m for m in self.models if not m.ready]
        with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
            for future in [pool.submit(m.build, system, data) for m in pending]:
                future.result()

    def provenance(self) -> list[dict]:
        return [m.provenance for m in self.models]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._stack.close()


def session(infra: InfraLike = None, seed: int | None = None) -> Session:
    return Session(infra, seed)
