"""
Modal (modal.com): each session is a Sandbox from the target's registry image,
running the worker, with frames over the exec's stdio.

A Sandbox keeps one container for the whole session, which a Modal function or
Cls does not guarantee, so build state is never lost between calls. Needs the
`modal` extra and Modal credentials (`modal token new`).
"""

import sys
import threading
from typing import Any

from . import protocol
from .config import Target
from .launchers import Launcher
from .transport import Transport, thread_env

# The Sandbox's main process. It must outlive the session: run as is, the image's own
# command may exit at once (a worker reading no stdin, a shell, --help), ending the
# Sandbox and the worker exec'd into it before any work is done. It blocks without
# consuming anything; coreutils is present in the Debian-based images Modal supports.
IDLE_ENTRYPOINT = ("sleep", "infinity")


class Modal(Launcher):
    name = "modal"
    options = {
        "gpu": None,                 # e.g. "L40S", "H100"
        "timeout": 24 * 3600,        # upper bound on a session's lifetime (s)
        "app": "evedesign-remote",
        # Modal secret for a private image: REGISTRY_USERNAME / REGISTRY_PASSWORD, or for
        # AWS ECR (*.dkr.ecr.*.amazonaws.com) AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
        "registry_secret": None,
        "secrets": [],               # further Modal secrets by name, e.g. "huggingface"
        "volumes": {},               # Modal volumes by mount path, e.g. {"/models": "esm2-weights"}
    }

    def is_network(self, target: Target) -> bool:
        return True

    def validate(self, target: Target) -> None:
        super().validate(target)
        if target.mounts:
            raise ValueError(f"target {target.name!r}: Modal cannot bind host folders; use volumes")

    def image_for(self, model: str, sif_dir) -> str:
        return model

    def pin(self, target: Target) -> dict[str, Any]:
        # a registry digest pins exactly; a tag is recorded as given and flagged
        record: dict[str, Any] = {"pinned_image": target.image, "pinned_by_digest": "@sha256:" in target.image}
        if target.options["volumes"]:
            record["volumes"] = dict(target.options["volumes"])
            record["outside_image"] = True  # not pinned by the image digest
        return record

    def start(self, target: Target, image: str | None) -> Transport:
        return ModalTransport(target, image)


def registry_image(modal, image: str, secret_name: str | None):
    """The Modal image for a registry reference; ECR gets Modal's own ECR pull (short-lived tokens)."""
    secret = modal.Secret.from_name(secret_name) if secret_name else None
    host = image.split("/", 1)[0]
    if ".dkr.ecr." in host and host.endswith(".amazonaws.com"):
        if secret is None:
            raise ValueError(f"{image}: a private ECR image needs registry_secret, a Modal secret with "
                             f"AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_REGION")
        built = modal.Image.from_aws_ecr(image, secret=secret)
    else:
        built = modal.Image.from_registry(image, secret=secret)
    # A Sandbox's command replaces only the image's CMD: without an ENTRYPOINT the idle
    # command runs as is, whatever entrypoint the image has (e.g. one that does not exec
    # its arguments). The worker is exec'd by name, never through the entrypoint.
    return built.dockerfile_commands(["ENTRYPOINT []"])


class ModalTransport(Transport):
    def __init__(self, target: Target, image: str) -> None:
        super().__init__(f"modal {image}")
        import modal

        options = target.options
        app = modal.App.lookup(options["app"], create_if_missing=True)
        secrets = [modal.Secret.from_name(name) for name in options["secrets"]]
        if target.env:
            secrets.append(modal.Secret.from_dict(dict(target.env)))
        if target.cpus is not None:
            secrets.append(modal.Secret.from_dict(thread_env(target.cpus)))
        self._sandbox = modal.Sandbox.create(
            *IDLE_ENTRYPOINT,
            app=app, image=registry_image(modal, image, options["registry_secret"]),
            gpu=options["gpu"], cpu=target.cpus, timeout=options["timeout"], secrets=secrets,
            volumes={path: modal.Volume.from_name(name) for path, name in options["volumes"].items()},
        )
        # The worker runs as an exec'd process, started by name: the image puts its
        # environment's python first on PATH. Set `command` on the target if the image's
        # python is elsewhere.
        command = list(target.command) if target.command != ("serve",) else ["python", "-m", "evedesign.remote.worker"]
        self._proc = self._sandbox.exec(*command, text=False)
        self._chunks = iter(self._proc.stdout)
        self._buffer = protocol.FrameBuffer(self.description)
        self._pending: list[dict[str, Any]] = []
        threading.Thread(target=self._forward_stderr, daemon=True).start()

    def _forward_stderr(self) -> None:
        try:
            for chunk in self._proc.stderr:
                sys.stderr.buffer.write(chunk)
                sys.stderr.flush()
        except Exception:  # noqa: BLE001 -- logs are best effort; the result path is unaffected
            pass

    def _send(self, data: bytes) -> None:
        self._proc.stdin.write(data)
        self._proc.stdin.drain()

    def _receive(self) -> dict[str, Any]:
        while not self._pending:
            chunk = next(self._chunks, None)
            if chunk is None:
                raise EOFError("modal sandbox stream closed")
            self._pending.extend(self._buffer.feed(chunk))
        return self._pending.pop(0)

    def _terminate(self) -> None:
        self._sandbox.terminate()
