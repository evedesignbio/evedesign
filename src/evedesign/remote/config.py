"""
Site infrastructure and locks -- as plain data.

An infra config is a mapping of plain values (str, int, float, bool, list, dict;
never None), so it can live in Python, YAML, TOML or JSON, and round-trips:

    Infra.from_dict(d).to_dict() == d
    Infra.coerce(x)      x: Infra | mapping | path(.toml/.yaml/.yml/.json) | None

    {"targets": {
         "esm2":       {"image": "ghcr.io/proteingym/esm2:2026.11"},
         "eve":        {"image": "eve:latest", "gpus": [1]},
         "esmc":       {"launcher": "apptainer", "image": "/srv/sif/esmc.sif"},
         "esm2-modal": {"launcher": "modal", "image": "ghcr.io/proteingym/esm2:2026.11", "gpu_type": "L40S"}},
     "lock": {"esm2": "ghcr.io/proteingym/esm2@sha256:..."}}      # optional pins

A target is named one of three ways:

    "esm2"                               a target of the infra file (no colon)
    "docker:ghcr.io/org/esm2:2026.11"    <launcher>:<image>; the image exactly as written
    {"launcher": "docker", "image": "esm2:1", "gpus": "none"}     a one-off

Under a lock only named targets the lock lists are accepted.
"""

import hashlib
import json
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Self

@dataclass(frozen=True)
class Target:
    """
    Where and how one worker runs. The common fields below apply to every
    launcher; anything else a launcher accepts (Modal's gpu, volumes, ...) is in
    `options`, written flat next to them in a config file.
    """

    name: str
    launcher: str = "docker"
    image: str | None = None
    # the full command line; command launcher only (containers start the worker themselves)
    command: tuple[str, ...] = ()
    # auto: each model (or tree) that can use a GPU gets one of this process's GPUs
    #       (CUDA_VISIBLE_DEVICES, else all on the host), round-robin; CPU-only models get none
    # none: no GPU     ("0", "1"): exactly these devices (host numbering)
    gpus: str | tuple[str, ...] = "auto"
    # thread counts of OpenMP / MKL / OpenBLAS / numexpr inside, with every launcher;
    # also a hard limit for docker/podman (--cpus) and a reservation on Modal
    cpus: float | None = None
    env: tuple[tuple[str, str], ...] = ()
    # host folders, "source:dest[:ro]" (docker/podman bind mounts, apptainer --bind)
    mounts: tuple[str, ...] = ()
    # seconds to wait for the worker to start, and for one call; None waits for ever.
    # On expiry the worker is stopped and TimeoutError raised.
    start_timeout: float | None = None
    call_timeout: float | None = None
    # seconds an idle worker waits for a message before taking its orchestrator to be gone and
    # exiting (on Modal, with its Sandbox); the orchestrator pings meanwhile. 0 turns it off.
    lease: float = 300
    options: Mapping[str, Any] = field(default_factory=dict, hash=False)
    # set when resolved through a lock
    pinned: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.gpus, str) or not isinstance(self.gpus, (list, tuple)):
            if self.gpus not in ("auto", "none"):
                raise ValueError(f"target {self.name!r}: gpus is \"auto\", \"none\" or a list of devices "
                                 f"such as [0, 1], not {self.gpus!r}")
        else:
            gpus = tuple(str(d) for d in self.gpus)
            if not gpus or not all(d.strip().isdigit() for d in gpus):
                raise ValueError(f"target {self.name!r}: gpus must list device numbers, e.g. [0, 1]")
            object.__setattr__(self, "gpus", gpus)
        for key in ("start_timeout", "call_timeout"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"target {self.name!r}: {key} must be a positive number of seconds")
        if isinstance(self.lease, bool) or not isinstance(self.lease, (int, float)) or self.lease < 0:
            raise ValueError(f"target {self.name!r}: lease must be a number of seconds (0 turns it off)")
        launcher = self.launcher_impl
        unknown = set(self.options) - set(launcher.options)
        if unknown:
            raise ValueError(f"target {self.name!r}: unknown options {sorted(unknown)} for launcher "
                             f"{launcher.name!r}; it accepts {sorted(launcher.options)}")
        object.__setattr__(self, "options", {**launcher.options, **self.options})
        launcher.validate(self)

    @property
    def launcher_impl(self):
        from . import launchers

        return launchers.get(self.launcher)

    @property
    def is_network(self) -> bool:
        return self.launcher_impl.is_network(self)


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Read a config file by suffix: .toml, .yaml/.yml, .json (or .lock, which is JSON)."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".toml":
        return tomllib.loads(text)
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    if path.suffix in (".json", ".lock"):  # lock and provenance files are JSON
        return json.loads(text)
    raise ValueError(f"{path}: unknown config format; use .toml, .yaml, .yml, .json or .lock")


COMMON_FIELDS = {f.name for f in fields(Target)} - {"name", "options", "pinned"}


def _plain(value: Any) -> Any:
    """Options as plain data: lists and mappings stay what they are in the config."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _target(name: str, values: Mapping[str, Any]) -> Target:
    """A Target from config values; every field must be known to the target's launcher."""
    from . import launchers

    launcher = launchers.get(values.get("launcher", "docker"))
    allowed = COMMON_FIELDS | set(launcher.options)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"target {name!r}: unknown fields {sorted(unknown)} for launcher {launcher.name!r}; "
                         f"allowed: {sorted(allowed)}")
    common = {k: v for k, v in values.items() if k in COMMON_FIELDS} | {"launcher": launcher.name}
    for key in ("command", "mounts"):
        if key in common:
            common[key] = tuple(common[key])
    if isinstance(common.get("gpus"), (list, tuple)):
        common["gpus"] = tuple(common["gpus"])
    if "env" in common:
        common["env"] = tuple(sorted(dict(common["env"]).items()))
    options = {k: _plain(v) for k, v in values.items() if k in launcher.options}
    return Target(name=name, options=options, **common)


@dataclass
class Infra:
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        unknown = set(data) - {"targets", "lock"}
        if unknown:
            raise ValueError(f"infra: unknown keys {sorted(unknown)}; expected targets, lock")
        targets = {name: dict(values) for name, values in (data.get("targets") or {}).items()}
        for name, values in targets.items():
            if ":" in name:
                raise ValueError(f"infra target {name!r}: names may not contain a colon "
                                 f"(a colon marks a <launcher>:<image> shortcut)")
            _target(name, values)  # validate early, not at first use
        lock = dict(data["lock"]) if data.get("lock") else None
        return cls(targets, lock)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.targets:
            out["targets"] = {name: dict(values) for name, values in self.targets.items()}
        if self.lock:
            out["lock"] = dict(self.lock)
        return out

    @classmethod
    def load(cls, path: str | Path | None = None, lock: str | Path | Mapping | None = None) -> Self:
        infra = cls.from_dict(load_mapping(path)) if path else cls()
        return infra.with_lock(lock) if lock is not None else infra

    @classmethod
    def coerce(cls, value: "Infra | Mapping | str | Path | None",
               lock: str | Path | Mapping | None = None) -> Self:
        if isinstance(value, Infra):
            infra = value
        elif isinstance(value, Mapping):
            infra = cls.from_dict(value)
        else:
            infra = cls.load(value)
        return infra.with_lock(lock) if lock is not None else infra

    def with_lock(self, lock: str | Path | Mapping) -> Self:
        """Pins from a lock or provenance file (or the mapping read from one)."""
        return replace(self, lock=read_lock(lock))

    def resolve(self, target: str | Mapping[str, Any] | Target) -> Target:
        """A target from its name, a <launcher>:<image> shortcut, or a one-off mapping."""
        from . import launchers

        if isinstance(target, Target):
            return target
        if self.lock is not None and not (isinstance(target, str) and ":" not in target):
            raise ValueError(f"target {target!r}: a locked run uses only named targets of the infra file")
        if isinstance(target, Mapping):
            values = dict(target)
            launcher = values.get("launcher", "docker")
            name = values.pop("name", None) or (f"{launcher}:{values['image']}" if values.get("image") else launcher)
            return _target(name, values)
        if ":" in target:
            launcher, image = target.split(":", 1)
            if launcher not in launchers.names():
                raise ValueError(f"target {target!r}: unknown launcher {launcher!r}; available: {launchers.names()}")
            if not launchers.get(launcher).needs_image:
                raise ValueError(f"target {target!r}: launcher {launcher!r} has no shortcut; define the target")
            return _target(target, {"launcher": launcher, "image": image})
        if target not in self.targets:
            raise KeyError(f"unknown target {target!r}. Defined: {sorted(self.targets) or '(none)'}; "
                           f"or use a shortcut like 'docker:<image>'")
        resolved = _target(target, self.targets[target])
        if self.lock is not None:
            if target not in self.lock:
                raise LookupError(f"target {target!r} is not in the lock; re-run `evedesign-remote lock`")
            resolved = replace(resolved, image=self.lock[target] or resolved.image, pinned=True)
        return resolved

    def names(self) -> list[str]:
        return sorted(self.targets)


def derive_seed(run_seed: int | None, label: str) -> int | None:
    """Per-model seed from the run seed and the model's position in the spec."""
    if run_seed is None:
        return None
    return int(hashlib.sha256(f"{run_seed}:{label}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# locks: a provenance-shaped file; a run's provenance can be used as a lock
# ---------------------------------------------------------------------------

def read_lock(lock: str | Path | Mapping[str, Any]) -> dict[str, str]:
    """
    Pins (target name -> image) from a lock file, a run's provenance file, or the
    mapping read from either. A flat {name: image} mapping is taken as-is.
    """
    data = lock if isinstance(lock, Mapping) else load_mapping(lock)
    if "models" not in data:
        return {str(k): str(v) for k, v in data.items()}
    return {
        m["launcher"]["target"]: m["launcher"]["pinned_image"]
        for m in data["models"] if (m.get("launcher") or {}).get("pinned_image")
    }
