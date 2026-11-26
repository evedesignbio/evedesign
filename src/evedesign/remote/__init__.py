"""
Isolated and remote execution of evedesign models: each model in its own
environment (container, Apptainer image, Modal sandbox), used through a local
proxy that behaves like the model.

    open_model(target, model)                  created there, unbuilt; build() is forwarded
    open_built(target, model, system, data)    created and built there, nested models included
    reload_model(target, blob)                 a stored model, loaded there
    session(infra, seed)                       owns models, starts and builds them in parallel
    build_local(spec, system, prebuilt=...)    a tree built here; given nodes (e.g. proxies) used as they are

    with session("infra.toml", seed=1234) as s:
        esm, mw = s.open_all([("esm2-gpu", {"key": "esm2", "args": {"device": "cuda"}}),
                              ("docker:molweight:latest", None)])
        s.build_all(system)
        designs = GibbsSampler(scorers=[esm, mw], weights=[1, .5]).generate(16)

Needs the `remote` extra (msgpack, pyyaml, dill); Modal targets also `modal`.
Names are imported on first use, so submodules without those dependencies
(e.g. evedesign.remote.wrapping) work without the extra.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "Session": "api", "open_model": "api", "open_built": "api", "reload_model": "api", "session": "api",
    "Infra": "config", "build_local": "build",
}
__all__ = ["Infra", "Session", "build_local", "open_built", "open_model", "reload_model", "session"]

if TYPE_CHECKING:
    from .api import Session, open_built, open_model, reload_model, session
    from .build import build_local
    from .config import Infra


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
