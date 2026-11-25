"""
Connections to workers, and this host's GPUs and CPUs.

A Transport carries frames to one worker: subclasses implement _send, _receive
and _terminate. StdioTransport is a child process whose stdin/stdout carry them
(every built-in launcher but Modal). How a worker is started is a launcher's
business (launchers.py).
"""

import itertools
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

from . import protocol
from .config import Target


class Transport:
    """A connection to one worker. Thread-safe: one request in flight at a time."""

    def __init__(self, description: str) -> None:
        self.description = description
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._closed = False

    # subclasses implement these three
    def _send(self, data: bytes) -> None: ...
    def _receive(self) -> dict[str, Any]: ...
    def _terminate(self) -> None: ...

    def request(self, op: str, /, **kwargs: Any) -> Any:
        if self._closed:
            raise protocol.SessionLost(f"{self.description}: session already closed")
        with self._lock:
            msg_id = next(self._ids)
            # encoded first: a call too large to send fails here and leaves the session usable
            data = protocol.encode({"id": msg_id, "type": "call", "method": op, "kwargs": kwargs})
            try:
                self._send(data)
                reply = self._receive()
            except (EOFError, BrokenPipeError, OSError) as exc:
                self._closed = True
                raise protocol.SessionLost(f"{self.description}: worker went away ({exc})") from exc
        if reply.get("id") != msg_id:
            raise protocol.SessionLost(f"{self.description}: out-of-order reply {reply.get('id')} != {msg_id}")
        if reply["type"] == "error":
            raise protocol.RemoteError(reply["error_type"], reply["message"], reply.get("traceback", ""))
        return reply["value"]

    def close(self) -> None:
        """Idempotent; cleanup paths call it more than once."""
        if self._closed:
            return
        try:
            self.request("close")
        except Exception:  # noqa: BLE001 -- closing a dead worker is fine
            pass
        self._closed = True
        self._terminate()


class StdioTransport(Transport):
    def __init__(self, argv: list[str], description: str, env: dict[str, str] | None = None) -> None:
        super().__init__(description)
        self.argv = argv
        self._ready = False
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env, bufsize=0,
        )

    def _send(self, data: bytes) -> None:
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _receive(self) -> dict[str, Any]:
        if not self._ready:
            protocol.read_ready(self._proc.stdout, self.description)
            self._ready = True
        return protocol.read_frame(self._proc.stdout)

    def _terminate(self) -> None:
        for stream in (self._proc.stdin, self._proc.stdout):
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()


# ---------------------------------------------------------------------------
# GPUs and CPUs on this host
# ---------------------------------------------------------------------------

def _has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


class DevicePool:
    """
    The GPUs this process may use: CUDA_VISIBLE_DEVICES if a scheduler set it,
    else every GPU on the host. Each GPU model takes the next one, round-robin,
    so two models in one job land on two devices instead of both on cuda:0.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0

    def devices(self) -> list[str]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            return [d for d in visible.split(",") if d.strip()]
        if not _has_nvidia():
            return []
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return [str(i) for i, line in enumerate(out.stdout.splitlines()) if line.startswith("GPU")]

    def take(self) -> int | None:
        """A ticket for the next device (its position is ticket % len(devices())), or None."""
        if not self.devices():
            return None
        with self._lock:
            self._next += 1
            return self._next - 1

    def give_back(self, ticket: int) -> None:
        """Return an unused ticket, if no other model took one since."""
        with self._lock:
            if self._next == ticket + 1:
                self._next = ticket


POOL = DevicePool()


@dataclass(frozen=True)
class Devices:
    """
    CUDA_VISIBLE_DEVICES offered to a worker before it constructs its model: `gpu`
    for a model that can use a GPU, `cpu` for one that cannot. None leaves
    visibility as launched; "" hides all GPUs. The worker picks one, so the
    orchestrator need not know the model's class.
    """

    gpu: str | None
    cpu: str | None
    ticket: int | None = None

    def as_message(self) -> dict:
        return {"gpu": self.gpu, "cpu": self.cpu}


def offer_devices(target: Target) -> Devices:
    if target.is_network or target.gpus not in ("inherit", "none"):
        return Devices(None, None)  # remote: its own GPUs; "share" and "device=N" are as launched
    if target.gpus == "none":
        return Devices("", "")
    ticket = POOL.take()
    if ticket is None:
        return Devices(None, "")
    devices = POOL.devices()
    position = ticket % len(devices)
    # docker/podman renumber the GPUs they pass in from 0; apptainer and plain
    # processes see host numbering
    gpu = str(position) if target.launcher_impl.renumbers_gpus else devices[position]
    return Devices(gpu, "", ticket)


def settle_devices(devices: Devices, applied: str | None) -> None:
    """After the worker chose: a GPU slot the model did not take goes back to the pool."""
    if devices.ticket is not None and applied != devices.gpu:
        POOL.give_back(devices.ticket)


THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def thread_env(cpus: float | None) -> dict[str, str]:
    """Thread counts for numeric libraries (torch reads OMP_NUM_THREADS); {} without a limit."""
    if cpus is None:
        return {}
    return dict.fromkeys(THREAD_VARIABLES, str(max(1, int(cpus))))


def launch(target: Target, pinned_image: str | None) -> Transport:
    return target.launcher_impl.start(target, pinned_image or target.image)
