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
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from . import protocol
from .config import Target


_DEFAULT = object()


class Transport:
    """A connection to one worker. Thread-safe: one request in flight at a time."""

    # seconds one call may take (the target's call_timeout); None waits for ever
    call_timeout: float | None = None

    def __init__(self, description: str) -> None:
        self.description = description
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._closed = False

    # subclasses implement these three; _terminate(kill=True) must not wait for the worker
    def _send(self, data: bytes) -> None: ...
    def _receive(self) -> dict[str, Any]: ...
    def _terminate(self, kill: bool = False) -> None: ...

    def _receive_within(self, deadline: float | None, op: str, timeout: float | None) -> dict[str, Any]:
        """_receive, giving up at `deadline` (time.monotonic()): the worker is then stopped, since a
        stream cannot be resynchronised mid-frame."""
        if deadline is None:
            return self._receive()
        outcome: dict[str, Any] = {}

        def receive() -> None:
            try:
                outcome["reply"] = self._receive()
            except BaseException as exc:  # noqa: BLE001 -- re-raised in the caller
                outcome["error"] = exc

        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        reader.join(max(0.0, deadline - time.monotonic()))
        if reader.is_alive():
            self._closed = True
            self._terminate(kill=True)
            raise TimeoutError(f"{self.description}: {op} took longer than {timeout:g}s; the worker was stopped")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["reply"]

    def _exchange(self, op: str, timeout: float | None, kwargs: dict, status_callback=None) -> Any:
        """One call and its reply; the caller holds the lock. Status messages the worker sends
        while the call runs go to status_callback."""
        msg_id = next(self._ids)
        message = {"id": msg_id, "type": "call", "method": op, "kwargs": kwargs}
        if status_callback is not None:
            message["status"] = True
        # encoded first: a call too large to send fails here and leaves the session usable
        data = protocol.encode(message)
        label = kwargs.get("method", op)
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            self._send(data)
            reply = self._receive_within(deadline, label, timeout)
            while reply.get("type") == "status":
                if reply.get("id") == msg_id:
                    _report(status_callback, reply, self.description)
                reply = self._receive_within(deadline, label, timeout)
        except TimeoutError:
            raise
        except (EOFError, BrokenPipeError, OSError) as exc:
            self._closed = True
            raise protocol.SessionLost(f"{self.description}: worker went away ({exc})") from exc
        if reply.get("id") != msg_id:
            raise protocol.SessionLost(f"{self.description}: out-of-order reply {reply.get('id')} != {msg_id}")
        if reply["type"] == "error":
            raise protocol.RemoteError(reply["error_type"], reply["message"], reply.get("traceback", ""))
        return reply["value"]

    def request(self, op: str, /, *, timeout: Any = _DEFAULT, status_callback=None, **kwargs: Any) -> Any:
        """Send one call and wait for its reply, for at most `timeout` seconds
        (default: call_timeout)."""
        if self._closed:
            raise protocol.SessionLost(f"{self.description}: session already closed")
        if timeout is _DEFAULT:
            timeout = self.call_timeout
        with self._lock:
            return self._exchange(op, timeout, kwargs, status_callback)

    def keep_alive(self, lease: float | None) -> None:
        """Ping the worker every lease/3 seconds while no call runs, so its lease does not run out."""
        if lease:
            threading.Thread(target=self._ping, args=(lease / 3,), daemon=True).start()

    def _ping(self, every: float) -> None:
        while not self._closed:
            time.sleep(every)
            if not self._lock.acquire(blocking=False):
                continue  # a call is running: the worker knows it is in use
            try:
                self._exchange("ping", 30, {})
            except Exception:  # noqa: BLE001 -- a lost worker shows on the next call
                pass
            finally:
                self._lock.release()

    def close(self) -> None:
        """Idempotent; cleanup paths call it more than once. Never waits for a call running in
        another thread: that worker is stopped, and the call ends with SessionLost."""
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            self._closed = True
            self._terminate(kill=True)
            return
        try:
            self._exchange("close", 30 if self.call_timeout is None else min(30, self.call_timeout), {})
        except Exception:  # noqa: BLE001 -- closing a dead worker is fine
            pass
        finally:
            self._lock.release()
        self._closed = True
        self._terminate()


def _report(status_callback, message: dict, description: str) -> None:
    try:
        status_callback(message["status"], message.get("progress"), message.get("message"))
    except Exception as exc:  # noqa: BLE001 -- the stream must still be read to the reply
        logger.warning(f"{description}: status_callback raised {exc!r}")


class StdioTransport(Transport):
    """
    A child process. `kill_argv`, if given, is run when the worker must be stopped at
    once (a timeout): e.g. `docker kill <name>`, since killing the docker client alone
    leaves a busy container running.
    """

    def __init__(self, argv: list[str], description: str, env: dict[str, str] | None = None,
                 kill_argv: list[str] | None = None) -> None:
        super().__init__(description)
        self.argv = argv
        self._kill_argv = kill_argv
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

    def _terminate(self, kill: bool = False) -> None:
        if kill:
            if self._kill_argv:
                try:
                    subprocess.run(self._kill_argv, capture_output=True, timeout=60, check=False)
                except (OSError, subprocess.TimeoutExpired):  # killing the client below still ends the stream
                    pass
            self._proc.kill()
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
    if target.is_network or not isinstance(target.gpus, str):
        return Devices(None, None)  # remote: its own GPUs; a device list is as launched
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
    transport = target.launcher_impl.start(target, pinned_image or target.image)
    transport.call_timeout = target.call_timeout
    return transport
