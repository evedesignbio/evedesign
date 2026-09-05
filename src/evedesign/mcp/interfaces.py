"""
Introspection of evedesign's core interfaces for the MCP server's list_interfaces tool.
"""
from __future__ import annotations

import inspect
from typing import Any

from evedesign import model as _model
from evedesign.analysis import Analyzer
from evedesign.nucleotides import ProteinToDnaOptimizer

# _Core is the base contract shared by every interface below (name, citations, requires_gpu,
# positions(), ...). Other interfaces are described by the members it adds on top of _Core
BASE_INTERFACE = _model._Core

INTERFACES: dict[str, type] = {
    "Generator": _model.Generator,
    "Scorer": _model.Scorer,
    "Transformer": _model.Transformer,
    "MutationScorer": _model.MutationScorer,
    "ConditionalMutationScorer": _model.ConditionalMutationScorer,
    "BaseModel": _model.BaseModel,
    "SupervisedBaseModel": _model.SupervisedBaseModel,
}

# standalones that don't extend _Core
STANDALONE_INTERFACES: dict[str, type] = {
    "Analyzer": Analyzer,
    "ProteinToDnaOptimizer": ProteinToDnaOptimizer,
}


def _member_info(cls: type, name: str) -> dict[str, Any]:
    member = cls.__dict__[name]
    # abstractmethod-decorated properties/classmethods are wrapped descriptors. Unwrap to get
    # real fxn
    if isinstance(member, property):
        func, kind = member.fget, "property"
    elif isinstance(member, classmethod):
        func, kind = member.__func__, "classmethod"
    elif isinstance(member, staticmethod):
        func, kind = member.__func__, "staticmethod"
    else:
        func, kind = member, "method"
    try:
        signature = str(inspect.signature(func)) if func is not None else None
    except (TypeError, ValueError):
        signature = None
    return {
        "name": name,
        "kind": kind,
        "signature": signature,
        "docstring": inspect.getdoc(func),
    }


def _own_abstract_members(cls: type) -> list[dict[str, Any]]:
    """
    Return abstract methods/properties introduced directly on cls (not inherited)

    Parameters
    ----------
    cls
        Interface class to inspect

    Returns
    -------
    Member descriptions (name, kind, signature, docstring)
    """
    return [
        _member_info(cls, name)
        for name in sorted(getattr(cls, "__abstractmethods__", ()))
        if name in cls.__dict__
    ]


def _describe(cls: type, *, extends: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": cls.__name__,
        "docstring": inspect.getdoc(cls),
        "extends": extends or [],
        "members": _own_abstract_members(cls),
    }


def describe_all() -> dict[str, Any]:
    """
    Describe every core interface: the shared base contract, then each interface's
    additional members, then the standalone (non-_Core) ABCs

    Returns
    -------
    Dict with keys "base", "interfaces" and "standalone_interfaces"
    """
    return {
        "base": _describe(BASE_INTERFACE),
        "interfaces": {
            name: _describe(cls, extends=[b.__name__ for b in cls.__bases__ if b is not object])
            for name, cls in INTERFACES.items()
        },
        "standalone_interfaces": {name: _describe(cls) for name, cls in STANDALONE_INTERFACES.items()},
    }
