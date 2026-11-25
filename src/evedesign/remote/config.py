"""
Site infrastructure and locks -- as plain data.

Every config is a mapping of plain values (str, int, float, bool, list, dict;
never None), so it can live in Python, YAML, TOML or JSON, and round-trips:

    Infra.from_dict(d).to_dict() == d
    Infra.coerce(x)      x: Infra | mapping | path(.toml/.yaml/.yml/.json) | None

Infra (describes a machine or site):

    {"defaults": {"launcher": "docker", "gpus": "inherit"},
     "targets": {
         "esm2":       {"image": "ghcr.io/proteingym/esm2:2026.11"},
         "eve":        {"image": "eve:latest", "gpus": "device=1"},
         "esmc":       {"launcher": "apptainer", "image": "/srv/sif/esmc.sif"},
         "esm2-modal": {"launcher": "modal", "image": "ghcr.io/proteingym/esm2:2026.11", "gpu": "L40S"}},
     "lock": {"esm2": "ghcr.io/proteingym/esm2@sha256:..."}}      # optional pins

Without any targets, `docker:esm2`, `podman:esm2`, `apptainer:esm2` and
`modal:esm2` are valid names by convention, and anywhere a target name is
accepted, a target mapping is too: {"launcher": "docker", "image": "esm2:latest"}.

Placement (which model of a pipeline runs on which target) lives with the
pipeline code, in evedesign_server.placement.
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
    command: tuple[str, ...] = ("serve",)
    # inherit: each GPU model gets one of this process's GPUs (CUDA_VISIBLE_DEVICES,
    #          else all on the host), round-robin; CPU-only models get none
    # share:   every model sees all of them    none: no GPU    "device=1": pinned
    gpus: str = "inherit"
    # thread counts of OpenMP / MKL / OpenBLAS / numexpr inside, with every launcher;
    # also a hard limit for docker/podman (--cpus) and a reservation on Modal
    cpus: float | None = None
    env: tuple[tuple[str, str], ...] = ()
    # host folders, "source:dest[:ro]" (docker/podman bind mounts, apptainer --bind)
    mounts: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict, hash=False)
    # set when resolved through a lock
    pinned: bool = False

    def __post_init__(self) -> None:
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


def _target(name: str, values: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> Target:
    """
    A Target from config values. Defaults apply where the target's launcher has
    the field (a Modal default like `gpu` does not reach docker targets); the
    target's own fields must all be known to its launcher.
    """
    from . import launchers

    defaults = dict(defaults or {})
    launcher = launchers.get(values.get("launcher", defaults.get("launcher", "docker")))
    allowed = COMMON_FIELDS | set(launcher.options)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"target {name!r}: unknown fields {sorted(unknown)} for launcher {launcher.name!r}; "
                         f"allowed: {sorted(allowed)}")
    merged = {**{k: v for k, v in defaults.items() if k in allowed}, **values, "launcher": launcher.name}
    common = {k: v for k, v in merged.items() if k in COMMON_FIELDS}
    for key in ("command", "mounts"):
        if key in common:
            common[key] = tuple(common[key])
    if "env" in common:
        common["env"] = tuple(sorted(dict(common["env"]).items()))
    options = {k: _plain(v) for k, v in merged.items() if k in launcher.options}
    return Target(name=name, options=options, **common)


def _check_defaults(defaults: Mapping[str, Any]) -> None:
    from . import launchers

    known = COMMON_FIELDS | {o for n in launchers.names() for o in launchers.get(n).options}
    unknown = set(defaults) - known
    if unknown:
        raise ValueError(f"infra defaults: unknown fields {sorted(unknown)}; allowed: {sorted(known)}")


@dataclass
class Infra:
    defaults: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    sif_dir: Path = Path(".")
    lock: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        unknown = set(data) - {"defaults", "targets", "lock"}
        if unknown:
            raise ValueError(f"infra: unknown keys {sorted(unknown)}; expected defaults, targets, lock")
        defaults = dict(data.get("defaults") or {})
        sif_dir = Path(defaults.pop("sif_dir", "."))
        _check_defaults(defaults)
        targets = {name: dict(values) for name, values in (data.get("targets") or {}).items()}
        for name, values in targets.items():
            _target(name, values, defaults)  # validate early, not at first use
        lock = dict(data["lock"]) if data.get("lock") else None
        return cls(defaults, targets, sif_dir, lock)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        defaults = dict(self.defaults)
        if self.sif_dir != Path("."):
            defaults["sif_dir"] = str(self.sif_dir)
        if defaults:
            out["defaults"] = defaults
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
        from . import launchers

        if isinstance(target, Target):
            return target
        if isinstance(target, Mapping):
            values = dict(target)
            name = values.pop("name", f"inline:{values.get('image') or values.get('launcher', '?')}")
            return _target(name, values, self.defaults)
        name = target
        if name in self.targets:
            resolved = _target(name, self.targets[name], self.defaults)
        elif ":" in name and name.split(":", 1)[0] in launchers.names():
            launcher, model = name.split(":", 1)
            image = launchers.get(launcher).image_for(model, self.sif_dir)
            resolved = _target(name, {"launcher": launcher, "image": image}, self.defaults)
        else:
            raise KeyError(
                f"unknown target {name!r}. Defined: {sorted(self.targets) or '(none)'}; "
                f"or use a convention name like 'docker:<model>'"
            )
        if self.lock is not None:
            if name not in self.lock:
                raise LookupError(f"target {name!r} is not in the lock; re-run `evedesign-remote lock`")
            resolved = replace(resolved, image=self.lock[name] or resolved.image, pinned=True)
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
