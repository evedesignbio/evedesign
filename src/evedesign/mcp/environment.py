"""
Environment/installation check for the MCP server's check_environment tool.
"""
from __future__ import annotations

import importlib
import platform
import shutil
import sys
from typing import Any

import evedesign
from evedesign.mcp import registry

# The imports that presence of the extra is checked by. Same as [project.optional-dependencies] 
# in pyproject.toml. boltzgen is checked separately below
_EXTRA_CHECKS: dict[str, tuple[str, ...]] = {
    "eve": ("torch", "tqdm"),
    "evmutation2": ("torch", "evmutation2"),
    "evcouplings": ("evcouplings",),
    "esm2": ("torch", "transformers"),
    "mpnn": ("torch", "prody"),
    "promb": ("promb",),
    "boltz2fold": ("torch", "boltz", "yaml"),
    "umap": ("umap",),
    "gpytorch": ("torch", "gpytorch"),
}


def _importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _extra_status(imports: tuple[str, ...]) -> dict[str, Any]:
    per_import = {name: _importable(name) for name in imports}
    return {"installed": all(per_import.values()), "imports": per_import}


def _torch_devices() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    return {
        "cpu": True,
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }


def check() -> dict[str, Any]:
    """
    Report which optional extras are importable in this process and which torch devices
    (cpu/cuda/mps) are visible

    Returns
    -------
    evedesign/python version, platform, per-extra availability, torch device availability,
    and any errors encountered while building the model registry
    """
    extras_status = {name: _extra_status(imports) for name, imports in _EXTRA_CHECKS.items()}
    extras_status["boltzgen"] = {
        "installed": shutil.which("boltzgen") is not None,
        "note": "checked via the boltzgen CLI binary on PATH, not a Python import",
    }

    _, import_errors = registry.discover()

    return {
        "ok": True,
        "evedesign_version": evedesign.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "extras": extras_status,
        "torch_devices": _torch_devices(),
        "registry_import_errors": import_errors,
    }
