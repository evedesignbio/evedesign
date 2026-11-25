"""
Framing shared by every launcher: 4-byte big-endian length + msgpack body.

Every message is a dict with an `id` and a `type` (call | result | error). Either
side may send `call`; v1 only uses orchestrator -> worker, but keeping the frame
symmetric means callbacks can be added later without breaking the protocol.
"""

import os
import struct
from typing import Any, BinaryIO

from . import codec

_HEADER = struct.Struct(">I")

# Largest frame accepted, so a misbehaving peer cannot make this side allocate
# arbitrary memory. Serialized models travel as one frame: raise it for large ones.
MAX_FRAME = int(os.environ.get("EVEDESIGN_REMOTE_MAX_FRAME_MB", "2048")) << 20

# The worker writes this before its first frame. Whatever arrives before it on the
# same stream (login banners, scheduler or container-runtime messages) is skipped
# and logged, so a worker can be reached through ssh, srun, kubectl, ...
READY = b"\x00evedesign.remote:ready\x00"
_SKIP_LIMIT = 1 << 20


class RemoteError(RuntimeError):
    """
    An exception raised on the far side. Its message is the remote type and message
    only; the remote traceback (paths, environment details) is in `remote_traceback`,
    so it does not travel wherever error messages are shown.
    """

    def __init__(self, type_name: str, message: str, traceback: str = "") -> None:
        super().__init__(f"{type_name}: {message}")
        self.type_name = type_name
        self.remote_traceback = traceback


class FrameTooLarge(EOFError):
    """A frame over MAX_FRAME: the stream cannot be trusted any more, so the session ends."""


class SessionLost(RuntimeError):
    """The worker went away. Not retried: build() may dominate the cost, so the caller decides."""


def write_frame(stream: BinaryIO, message: dict[str, Any]) -> int:
    body = codec.dumps(message)
    _check_length(len(body))  # fail here, where the error can still be reported
    stream.write(_HEADER.pack(len(body)) + body)
    stream.flush()
    return len(body)


def read_exact(stream: BinaryIO, n: int) -> bytes:
    chunks, remaining = [], n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("stream closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _check_length(length: int) -> int:
    if length > MAX_FRAME:
        raise FrameTooLarge(f"frame of {length >> 20} MB exceeds the limit of {MAX_FRAME >> 20} MB "
                            f"(EVEDESIGN_REMOTE_MAX_FRAME_MB)")
    return length


def read_frame(stream: BinaryIO) -> dict[str, Any]:
    (length,) = _HEADER.unpack(read_exact(stream, _HEADER.size))
    return codec.loads(read_exact(stream, _check_length(length)))


def _skipped(description: str, data: bytes) -> None:
    if data.strip():
        from loguru import logger

        logger.warning(f"{description}: skipped output before the worker started: "
                       f"{data.decode(errors='replace').strip()[:500]!r}")


def _no_worker(description: str, data: bytes) -> Exception:
    return EOFError(f"{description}: no worker answered; output was "
                    f"{data.decode(errors='replace').strip()[-500:]!r}")


def read_ready(stream: BinaryIO, description: str) -> None:
    """Consume the stream up to and including READY."""
    seen = bytearray()
    while not seen.endswith(READY):
        byte = stream.read(1)
        if not byte or len(seen) > _SKIP_LIMIT:
            raise _no_worker(description, bytes(seen))
        seen += byte
    _skipped(description, bytes(seen[:-len(READY)]))


def encode(message: dict[str, Any]) -> bytes:
    body = codec.dumps(message)
    _check_length(len(body))
    return _HEADER.pack(len(body)) + body


class FrameBuffer:
    """
    Reassembles frames from arbitrary byte chunks (for stream APIs without exact
    reads), after skipping everything up to READY.
    """

    def __init__(self, description: str = "worker") -> None:
        self._buf = bytearray()
        self._ready = False
        self._description = description

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self._buf.extend(chunk)
        if not self._ready:
            at = self._buf.find(READY)
            if at < 0:
                if len(self._buf) > _SKIP_LIMIT:
                    raise _no_worker(self._description, bytes(self._buf))
                return []
            _skipped(self._description, bytes(self._buf[:at]))
            del self._buf[:at + len(READY)]
            self._ready = True
        out = []
        while len(self._buf) >= _HEADER.size:
            (length,) = _HEADER.unpack_from(self._buf)
            _check_length(length)
            if len(self._buf) < _HEADER.size + length:
                break
            body = bytes(self._buf[_HEADER.size:_HEADER.size + length])
            del self._buf[:_HEADER.size + length]
            out.append(codec.loads(body))
        return out
