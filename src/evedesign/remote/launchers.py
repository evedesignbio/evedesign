"""
Launchers: how a target's worker is started, and how its image is pinned.

Built in: docker, podman, apptainer, command (any argv: venvs, `uv run`, ssh,
srun, kubectl, ...), and modal. Other providers register through entry points in
the group "evedesign.remote.launchers", so each can be its own package:

    [project.entry-points."evedesign.remote.launchers"]
    myprovider = "evedesign_myprovider:MyProvider"      # a Launcher subclass

A provider needs two things: one container (or process) kept for the whole
session, since the built model lives in the worker, and a byte stream to the
worker in it. A launcher declares the target options it accepts (name ->
default); they are written flat next to the common target fields and arrive in
`target.options`.
"""

import functools
import json
import os
import subprocess
from typing import Any, Mapping

from . import transport
from .config import Target
from .transport import POOL, StdioTransport, Transport, thread_env


class Launcher:
    name: str = ""
    # target options this launcher accepts, with their defaults
    options: Mapping[str, Any] = {}
    needs_image: bool = True
    # docker --gpus renumbers the devices it passes in from 0; others see host numbering
    renumbers_gpus: bool = False

    def is_network(self, target: Target) -> bool:
        """Calls cross a network: preflight warns about per-step round trips, and the
        worker gets no device from this host's GPU pool."""
        return False

    def validate(self, target: Target) -> None:
        if self.needs_image and not target.image:
            raise ValueError(f"target {target.name!r}: launcher {self.name!r} needs an image")

    def image_for(self, model: str, sif_dir) -> str:
        """The image a convention name `<launcher>:<model>` means."""
        return model if ":" in model or "@" in model else f"{model}:latest"

    def pin(self, target: Target) -> dict[str, Any]:
        """Provenance fields, resolved before launch; `pinned_image` is what start() gets."""
        return {"pinned_image": target.image}

    def start(self, target: Target, image: str | None) -> Transport:
        raise NotImplementedError


def _worker_command(target: Target) -> list[str]:
    # images built from the template have the worker as entrypoint (`serve`);
    # other images name the whole command
    return list(target.command)


# ---------------------------------------------------------------------------
# containers on this host
# ---------------------------------------------------------------------------

def gpu_flags(target: Target) -> list[str]:
    """Docker/podman --gpus: which devices the container can see at all."""
    if target.gpus in ("inherit", "share"):
        devices = POOL.devices()
        return ["--gpus", f'"device={",".join(devices)}"'] if devices else []
    return [] if target.gpus == "none" else ["--gpus", target.gpus]


def container_argv(target: Target, image: str) -> list[str]:
    argv = [
        target.launcher, "run", "-i", "--rm", "--init",
        "--label", "evedesign-session",
        "--user", f"{os.getuid()}:{os.getgid()}",
        # a worker needs no privileges: no capabilities, no setuid escalation
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--shm-size", target.options.get("shm_size", "2g"),
        "--env", "HOME=/tmp", "--env", "UV_NO_SYNC=1",
    ]
    if target.options.get("network_mode"):
        argv += ["--network", target.options["network_mode"]]  # e.g. "none": no network at all
    if target.cpus is not None:
        argv += ["--cpus", str(target.cpus)]
    for key, value in thread_env(target.cpus).items():
        argv += ["--env", f"{key}={value}"]
    for key, _ in target.env:
        # by name: the value comes from the docker CLI's environment (see Docker.start),
        # so values such as tokens never appear in argv (ps, docker inspect)
        argv += ["--env", key]
    for mount in target.mounts:
        source, _, rest = mount.partition(":")
        dest, _, mode = rest.partition(":")
        spec = f"type=bind,source={os.path.abspath(source)},target={dest}"
        argv += ["--mount", spec + (",readonly" if mode == "ro" else "")]
    return argv + gpu_flags(target) + [image, *_worker_command(target)]


class Docker(Launcher):
    name = "docker"
    # network_mode: --network for the container, e.g. "none" for models that need no network
    options = {"shm_size": "2g", "network_mode": None}
    renumbers_gpus = True

    def pin(self, target: Target) -> dict[str, Any]:
        out = subprocess.run(
            [target.launcher, "image", "inspect", target.image, "--format", "{{json .}}"],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            raise LookupError(f"target {target.name!r}: image {target.image!r} not found locally "
                              f"({out.stderr.strip()}). Pull or build it first (`evedesign-remote build`).")
        info = json.loads(out.stdout)
        digests = info.get("RepoDigests") or []
        if digests:
            return {"image_id": info["Id"], "pinned_image": digests[0], "local_only": False}
        # reproducible on this machine until pushed
        return {"image_id": info["Id"], "pinned_image": info["Id"], "local_only": True}

    def start(self, target: Target, image: str | None) -> Transport:
        env = {**os.environ, **dict(target.env)}
        return StdioTransport(container_argv(target, image), f"{target.launcher} {image}", env=env)


class Podman(Docker):
    name = "podman"


class Apptainer(Launcher):
    name = "apptainer"

    def image_for(self, model: str, sif_dir) -> str:
        return str(sif_dir / f"{model}.sif")

    def pin(self, target: Target) -> dict[str, Any]:
        from .provenance import file_sha256

        return {"sha256": file_sha256(target.image), "pinned_image": target.image}

    def start(self, target: Target, image: str | None) -> Transport:
        argv = ["apptainer", "run", "--containall"]
        if target.gpus != "none" and transport._has_nvidia():
            argv.append("--nv")
        for mount in target.mounts:
            argv += ["--bind", mount]
        # APPTAINERENV_KEY sets KEY inside the container; values stay out of argv
        values = {"UV_NO_SYNC": "1", **thread_env(target.cpus), **dict(target.env)}
        env = {**os.environ, **{f"APPTAINERENV_{k}": v for k, v in values.items()}}
        return StdioTransport(argv + [image, *_worker_command(target)], f"apptainer {image}", env=env)


class Command(Launcher):
    """
    Any argv whose stdin/stdout reach a worker: a venv's python, `uv run`, or a
    worker on another machine (`ssh -T host ...`, `srun ...`, `kubectl run -i ...`).
    Set `network = true` for the latter: this host's GPUs then mean nothing there.
    Output before the worker's ready marker (login banners, scheduler messages) is
    skipped and logged. Not hermetic: provenance records the argv and the
    environment the worker reports.
    """

    name = "command"
    options = {"network": False}
    needs_image = False

    def is_network(self, target: Target) -> bool:
        return bool(target.options.get("network"))

    def validate(self, target: Target) -> None:
        if not target.command or target.command == ("serve",):
            raise ValueError(f"target {target.name!r}: launcher 'command' needs a command")
        if target.mounts:
            raise ValueError(f"target {target.name!r}: launcher 'command' has no mounts; use the paths directly")

    def pin(self, target: Target) -> dict[str, Any]:
        return {"argv": list(target.command), "not_hermetic": True}

    def start(self, target: Target, image: str | None) -> Transport:
        env = {**os.environ, **thread_env(target.cpus), **dict(target.env)}
        return StdioTransport(list(target.command), f"command {target.name}", env=env)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _modal() -> Launcher:
    from .modal_launcher import Modal

    return Modal()


_BUILTIN = {"docker": Docker, "podman": Podman, "apptainer": Apptainer, "command": Command, "modal": _modal}


@functools.cache
def _plugins() -> dict[str, Any]:
    from importlib.metadata import entry_points

    return {ep.name: ep for ep in entry_points(group="evedesign.remote.launchers")}


def names() -> list[str]:
    return sorted({*_BUILTIN, *_plugins()})


@functools.cache
def get(name: str) -> Launcher:
    if name in _BUILTIN:
        return _BUILTIN[name]()
    if name in _plugins():
        launcher = _plugins()[name].load()
        return launcher() if isinstance(launcher, type) else launcher
    raise ValueError(f"unknown launcher {name!r}; available: {names()}")
