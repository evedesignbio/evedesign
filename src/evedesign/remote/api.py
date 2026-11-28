"""
Opening models in other environments.

    open_model(target, model)                   created there, unbuilt (build() is forwarded)
    open_built(target, model, system, data)     created and built there, nested models included
    reload_model(target, blob)                  a stored model, loaded there
    session(infra, seed)                        several models: closed together, started and built in parallel

A model is named by a string (a key or a class path), a spec dictionary, or nothing
(the image's own model); see spec.py. A target is a name from the infra file, a
`<launcher>:<image>` shortcut, or a dictionary; see config.py.
"""

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Self

from . import provenance as prov
from .config import Infra, derive_seed
from .proxy import RemoteModel, make_proxy
from .spec import as_spec, describe
from .surface import PROTOCOL_VERSION, surface_hash
from .transport import launch, offer_devices, settle_devices


InfraLike = Infra | Mapping[str, Any] | str | Path | None
TargetLike = Mapping[str, Any] | str
ModelLike = Mapping[str, Any] | str | None


def _open(target: TargetLike, op: str, kwargs: dict, seed: int | None, infra: InfraLike) -> RemoteModel:
    target = Infra.coerce(infra).resolve(target)
    launcher = prov.pin(target)
    transport = launch(target, launcher.get("pinned_image"))
    try:
        hello = transport.request(
            "hello", protocol_version=PROTOCOL_VERSION, surface=surface_hash(), seed=seed, launcher=launcher,
            lease=target.lease or None, timeout=target.start_timeout,
        )
        transport.keep_alive(target.lease)
        devices = offer_devices(target)
        descriptor = transport.request(op, **kwargs, devices=devices.as_message())
        settle_devices(devices, descriptor["cuda_visible_devices"])
        launcher["cuda_visible_devices"] = descriptor["cuda_visible_devices"]
    except BaseException:
        transport.close()
        raise
    # CUDA facts come with the model, collected after the worker chose its devices
    environment = {**hello["environment"], **descriptor.get("environment", {})}
    record = {"launcher": launcher, "environment": environment, "spec": _recordable(kwargs["spec"])}
    return make_proxy(transport, descriptor, f"{describe(kwargs['spec'])}@{target.name}", record)


def _recordable(spec: dict) -> dict:
    """The spec for provenance, without the bytes of stored models."""
    if "serialized" in spec:
        return {**spec, "serialized": f"<{len(spec['serialized'])} bytes>"}
    models = spec.get("models")
    if not models:
        return spec
    return {**spec, "models": {name: [_recordable(v) for v in value] if isinstance(value, list) else _recordable(value)
                               for name, value in models.items()}}


def open_model(target: TargetLike, model: ModelLike = None, /, *, seed: int | None = None,
               infra: InfraLike = None) -> RemoteModel:
    """
    Create a model in the target's environment, unbuilt, like calling its constructor:

        esm = open_model("esm2-gpu", {"key": "esm2", "args": {"device": "cuda"}})
        esm.build(system)

    `seed` seeds the worker's global random generators (random, numpy, torch).
    """
    spec = as_spec(model)
    if spec.get("models"):
        raise ValueError("a spec with nested models is opened with open_built")
    if "serialized" in spec:
        raise ValueError("a stored model is opened with open_built or reload_model")
    return _open(target, "open_model", {"spec": spec}, seed, infra)


def open_built(target: TargetLike, model: ModelLike, system=None, data: Any = None, /, *,
               seed: int | None = None, infra: InfraLike = None) -> RemoteModel:
    """
    Create and build a model in the target's environment, with its nested models
    (`models` in the spec) built first, and return it ready:

        sampler = open_built("full-modal", {
            "class": "evedesign.samplers.gibbs:GibbsSampler",
            "args": {"num_sweeps": 100, "seed": 7},
            "models": {"scorers": [{"key": "esm2"}, {"key": "molweight"}]},
        }, system)

    `system` may be left out only for a stored model loaded without rebuilding.
    """
    spec = as_spec(model)
    if system is None and not ("serialized" in spec and not spec.get("rebuild")):
        raise ValueError("open_built needs a system, unless it loads a stored model without rebuilding it")
    proxy = _open(target, "open_built", {"spec": spec, "system": system, "data": data}, seed, infra)
    if system is not None:
        proxy._apply(proxy._descriptor, system=system)
    return proxy


def reload_model(target: TargetLike, blob: bytes, *, seed: int | None = None,
                 infra: InfraLike = None) -> RemoteModel:
    """A stored model (bytes from fetch_serialized()), loaded in the target's environment.
    Short for open_built(target, {"serialized": blob})."""
    return open_built(target, {"serialized": blob}, seed=seed, infra=infra)


def probe(target: TargetLike, infra: InfraLike = None) -> dict:
    """Start a target's worker, handshake, stop it. What `evedesign-remote check` reports."""
    target = Infra.coerce(infra).resolve(target)
    launcher = prov.pin(target)
    transport = launch(target, launcher.get("pinned_image"))
    try:
        hello = transport.request("hello", protocol_version=PROTOCOL_VERSION, surface=surface_hash(),
                                  launcher=launcher, timeout=target.start_timeout)
    finally:
        transport.close()
    return {"launcher": launcher, "environment": hello["environment"]}


class Session:
    """Owns every model opened through it and closes them on exit, failure included."""

    def __init__(self, infra: InfraLike = None, seed: int | None = None) -> None:
        self.infra = Infra.coerce(infra)
        self.seed = seed
        self._stack = ExitStack()
        self.models: list[RemoteModel] = []
        self._counts: dict[str, int] = {}

    def _seed(self, spec: dict, explicit: int | None) -> int | None:
        """An explicit seed wins; otherwise derived from the session seed and opening order per model."""
        if explicit is not None or self.seed is None:
            return explicit
        name = describe(spec)
        n = self._counts.get(name, 0)
        self._counts[name] = n + 1
        return derive_seed(self.seed, f"{name}#{n}")

    def _keep(self, proxy: RemoteModel) -> RemoteModel:
        self.models.append(proxy)
        self._stack.callback(proxy.close)
        return proxy

    def open_model(self, target: TargetLike, model: ModelLike = None, /, *,
                   seed: int | None = None) -> RemoteModel:
        """As module-level open_model; closed with the session, seeded from the session seed."""
        spec = as_spec(model)
        return self._keep(open_model(target, spec, seed=self._seed(spec, seed), infra=self.infra))

    def open_built(self, target: TargetLike, model: ModelLike, system=None, data: Any = None, /, *,
                   seed: int | None = None) -> RemoteModel:
        """As module-level open_built; closed with the session, seeded from the session seed."""
        spec = as_spec(model)
        return self._keep(open_built(target, spec, system, data, seed=self._seed(spec, seed), infra=self.infra))

    def open_all(self, models: list[tuple]) -> list[RemoteModel]:
        """
        Open several models unbuilt, starting their workers at the same time:

            esm, mw = s.open_all([("esm2-gpu", "esm2"), ("molweight-cpu", None)])

        Each entry is (target, model) or (target, model, {"seed": n}). Returns the
        models in the same order. If one fails to start, the others are closed again
        and its error is raised, with a note naming the entry.
        """
        entries = []
        for entry in models:
            target, model, *rest = entry
            options = dict(rest[0]) if rest else {}
            if set(options) - {"seed"}:
                raise ValueError(f"open_all: unknown options {sorted(set(options) - {'seed'})}; only 'seed'")
            spec = as_spec(model)  # every spec is checked before any worker starts
            entries.append((target, spec, self._seed(spec, options.get("seed"))))
        pool = ThreadPoolExecutor(max_workers=max(1, len(entries)))
        futures = [pool.submit(open_model, t, spec, seed=seed, infra=self.infra) for t, spec, seed in entries]
        try:
            wait(futures, return_when=FIRST_EXCEPTION)
            failed = next(((i, f.exception()) for i, f in enumerate(futures) if f.done() and f.exception()), None)
            if failed is not None:
                i, exc = failed
                target, spec, _ = entries[i]
                exc.add_note(f"open_all: entry {i} ({describe(spec)} on {target!r}) failed to start; "
                             f"the other models were closed")
                raise exc
        except BaseException:
            # failed or interrupted: close what opened, and what still opens, without waiting for it
            for future in futures:
                future.add_done_callback(_close_opened)
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return [self._keep(future.result()) for future in futures]

    def build_all(self, system, data: Any = None) -> None:
        """Build every unbuilt model concurrently; isolated builds are independent. If one fails
        (or this is interrupted), the builds still running are stopped, not waited for."""
        pending = [m for m in self.models if not m.ready]
        pool = ThreadPoolExecutor(max_workers=max(1, len(pending)))
        futures = {pool.submit(m.build, system, data): m for m in pending}
        try:
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            for future in done:
                future.result()
        except BaseException:
            for future, model in futures.items():
                if not future.done():
                    model.close()  # stops its worker; that build ends with SessionLost
            raise
        finally:
            pool.shutdown(wait=False)

    def provenance(self) -> list[dict]:
        return [m.provenance for m in self.models]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._stack.close()


def _close_opened(future) -> None:
    if not future.cancelled() and future.exception() is None:
        future.result().close()


def session(infra: InfraLike = None, seed: int | None = None) -> Session:
    return Session(infra, seed)
