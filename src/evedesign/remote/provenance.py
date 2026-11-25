"""
What ran, recorded from both sides.

  launcher side  what was asked for, pinned to a content hash before launch
  worker side    what actually ran, reported from inside the environment

A run's provenance file is also a valid lock file (see config.read_lock).
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from .surface import PROTOCOL_VERSION, surface_hash


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def collect_environment(seed: int | None = None) -> dict[str, Any]:
    """Run inside the worker (and locally for in-process models)."""
    packages = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]
    )
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "evedesign": _version("evedesign"),
        "protocol": PROTOCOL_VERSION,
        "surface": surface_hash(),
        "packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
        "packages": packages,
        "seed": seed,
    }
    if "torch" in sys.modules or _version("torch"):
        env["torch"] = _version("torch")
        try:
            import torch

            env["cuda"] = torch.version.cuda
            env["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            env["cuda_available"] = torch.cuda.is_available()
        except Exception:  # noqa: BLE001 -- provenance must never break a session
            pass
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        env["gpus"] = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return env


# ---------------------------------------------------------------------------
# launcher side
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _file_sha256(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    st = os.stat(path)
    return _file_sha256(str(path), st.st_size, st.st_mtime)


def pin(target) -> dict[str, Any]:
    """
    Resolve a target to a content hash, before anything is launched.

    Returns the record that goes into provenance; `pinned_image` is what the
    launcher must actually run, so a tag moving mid-run cannot mix two images.
    """
    record: dict[str, Any] = {"target": target.name, "launcher": target.launcher, "requested": target.image}

    record.update(target.launcher_impl.pin(target))
    if target.mounts:
        record["mounts"] = list(target.mounts)
        record["outside_image"] = True  # files not pinned by the image digest

    if target.pinned:
        # Under a lock the image *is* the pin; inspect above already refused it if missing.
        record["pinned_image"] = target.image
        record["from_lock"] = True
    return record


def write_lock(infra, names: list[str], path: str | Path) -> dict[str, Any]:
    """Pin the given named targets without running anything; output is provenance-shaped."""
    unnamed = [name for name in names if name not in infra.targets]
    if unnamed:
        raise ValueError(f"a lock pins named targets of the infra file only, not {unnamed}")
    records = [{"launcher": pin(infra.resolve(name))} for name in names]
    data = {"kind": "evedesign-remote-lock", "models": records}
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def write_run(records: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"kind": "evedesign-remote-run", "models": records}, indent=2, default=str))


def compare_build_environment(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Differences that matter when a stored model is reloaded somewhere else."""
    issues = []
    s_l, c_l = stored.get("launcher") or {}, current.get("launcher") or {}
    if s_l.get("pinned_image") and s_l.get("pinned_image") != c_l.get("pinned_image"):
        issues.append(f"image {s_l.get('pinned_image')} -> {c_l.get('pinned_image')}")
    s_e, c_e = stored.get("environment") or {}, current.get("environment") or {}
    if s_e.get("packages_sha256") != c_e.get("packages_sha256"):
        issues.append("installed packages differ")
    if (s_e.get("evedesign") or "").split(".")[:2] != (c_e.get("evedesign") or "").split(".")[:2]:
        issues.append(f"evedesign {s_e.get('evedesign')} -> {c_e.get('evedesign')}")
    return issues
