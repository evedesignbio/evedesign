"""
Modal (modal.com): each session is a Sandbox from the target's registry image,
running the worker, with frames over the exec's stdio.

A Sandbox keeps one container for the whole session, which a Modal function or
Cls does not guarantee, so build state is never lost between calls. A dead
orchestrator gives the worker no EOF here: its lease ends it, and the Sandbox with it. Needs the
`modal` extra and Modal credentials (`modal token new`).
"""

import sys
import threading
from typing import Any

from . import protocol
from .config import Target
from .launchers import WORKER, Launcher
from .transport import Transport, thread_env

# The Sandbox's main process. It must outlive the session: run as is, the image's own
# command may exit at once (a worker reading no stdin, a shell, --help), ending the
# Sandbox and the worker exec'd into it before any work is done. It blocks without
# consuming anything; coreutils is present in the Debian-based images Modal supports.
IDLE_ENTRYPOINT = ("sleep", "infinity")

DEFAULT_CPUS = 1     # without `cpus`, Modal would reserve an eighth of a core
CHUNK = 4 << 20      # Modal refuses a single stdin write somewhere between 16 and 32 MiB


class Modal(Launcher):
    name = "modal"
    options = {
        "gpu_type": None,            # e.g. "L40S", "H100", "A100:2"
        "memory": None,              # MiB reserved, e.g. 16384; or [request, limit] for a hard limit
        "allow_tags": False,         # allow an image by tag, not digest (development: a tag can move mid-run)
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
        if not isinstance(target.gpus, str):
            raise ValueError(f"target {target.name!r}: Modal has no local devices; choose GPUs with gpu_type")
        memory = target.options["memory"]
        mib = lambda m: type(m) is int and m > 0  # noqa: E731 -- type(), so True is no size
        if not (memory is None or mib(memory) or (isinstance(memory, (list, tuple)) and len(memory) == 2
                                                 and all(map(mib, memory)) and memory[0] <= memory[1])):
            raise ValueError(f"target {target.name!r}: memory is MiB (e.g. 16384) or [request, limit], "
                             f"not {memory!r}")

    def pin(self, target: Target) -> dict[str, Any]:
        # Every Sandbox pulls its image again, so a tag that moves mid-run would mix two images
        if "@sha256:" not in target.image and not target.options["allow_tags"]:
            raise ValueError(f"target {target.name!r}: Modal images are pinned by digest, not {target.image!r}. "
                             f"Get it with `docker buildx imagetools inspect {target.image}` and write "
                             f"image@sha256:..., or set allow_tags = true for development")
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
        cpus = DEFAULT_CPUS if target.cpus is None else target.cpus
        memory = options["memory"]
        limit = memory[1] if isinstance(memory, (list, tuple)) else None
        self._hint = f"; it may have run out of its memory limit of {limit} MiB (raise `memory`)" if limit else ""
        app = modal.App.lookup(options["app"], create_if_missing=True)
        secrets = [modal.Secret.from_name(name) for name in options["secrets"]]
        if target.env:
            secrets.append(modal.Secret.from_dict(dict(target.env)))
        # the lease from the start: a worker whose orchestrator dies before hello still ends
        secrets.append(modal.Secret.from_dict({**thread_env(cpus), protocol.END_CONTAINER: "1",
                                               protocol.LEASE: str(target.lease)}))
        self._sandbox = modal.Sandbox.create(
            *IDLE_ENTRYPOINT,
            app=app, image=registry_image(modal, image, options["registry_secret"]),
            gpu=None if target.gpus == "none" else options["gpu_type"], cpu=cpus,
            memory=tuple(memory) if isinstance(memory, (list, tuple)) else memory,
            timeout=options["timeout"], secrets=secrets,
            volumes={path: modal.Volume.from_name(name) for path, name in options["volumes"].items()},
        )
        try:
            # The worker runs as an exec'd process, started by name: the image puts its
            # environment's python first on PATH.
            self._proc = self._sandbox.exec(*WORKER, text=False)
            self._chunks = iter(self._proc.stdout)
            self._buffer = protocol.FrameBuffer(self.description)
            self._pending: list[dict[str, Any]] = []
            threading.Thread(target=self._forward_stderr, daemon=True).start()
        except BaseException:
            self._terminate()  # else the Sandbox idles until its timeout
            raise

    def _forward_stderr(self) -> None:
        try:
            for chunk in self._proc.stderr:
                sys.stderr.buffer.write(chunk)
                sys.stderr.flush()
        except Exception:  # noqa: BLE001 -- logs are best effort; the result path is unaffected
            pass

    def _lost(self, exc: BaseException | str) -> EOFError:
        """Modal's own errors (the Sandbox gone, a full buffer) end the session like a closed stream."""
        return EOFError(f"modal sandbox ended ({exc}){self._hint}")

    def _send(self, data: bytes) -> None:
        view = memoryview(data)
        try:
            for start in range(0, len(view), CHUNK):
                self._proc.stdin.write(view[start:start + CHUNK])
                self._proc.stdin.drain()
        except Exception as exc:  # noqa: BLE001
            raise self._lost(exc) from exc

    def _receive(self) -> dict[str, Any]:
        while not self._pending:
            try:
                chunk = next(self._chunks, None)
            except Exception as exc:  # noqa: BLE001
                raise self._lost(exc) from exc
            if chunk is None:
                raise self._lost("stream closed")
            self._pending.extend(self._buffer.feed(chunk))
        return self._pending.pop(0)

    def _terminate(self, kill: bool = False) -> None:
        try:
            self._sandbox.terminate()
        except Exception:  # noqa: BLE001 -- an ended Sandbox needs no stopping
            pass
