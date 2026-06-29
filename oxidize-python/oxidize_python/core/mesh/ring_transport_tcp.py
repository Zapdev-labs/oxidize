"""TCP ring transport mirroring oxidize-golang/core/mesh/ring_transport_tcp.go.

Implements a full-duplex (dual-socket) ring transport with little-endian
4-byte length-prefixed framing and a localhost bootstrap helper with
exponential backoff + jitter.
"""

from __future__ import annotations

import random
import socket
import struct
import threading
import time

from .ring_collective import RingBackend, RingError, RingNodeTransport

# Upper bound on a single ring frame (256 MiB) to reject corrupt length prefixes.
MAX_RING_FRAME = 256 << 20


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock`` or raise on EOF."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RingError("io", "unexpected EOF")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class FullDuplexTcpTransport(RingNodeTransport):
    """Dual-socket send/recv ring transport (mirrors DualTcpTransport).

    Uses little-endian 4-byte length-prefixed framing, matching the Rust ring
    TCP wire format.
    """

    def __init__(self, send: socket.socket, recv: socket.socket) -> None:
        self._send = send
        self._recv = recv
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send_to_right(self, data: bytes) -> None:
        with self._send_lock:
            header = struct.pack("<I", len(data))
            try:
                self._send.sendall(header)
                self._send.sendall(data)
            except OSError as exc:
                raise RingError("io", str(exc)) from exc

    def recv_from_left(self) -> bytes:
        with self._recv_lock:
            try:
                header = _recv_exact(self._recv, 4)
                (n,) = struct.unpack("<I", header)
                if n > MAX_RING_FRAME:
                    raise RingError("io", "ring frame too large")
                return _recv_exact(self._recv, n)
            except OSError as exc:
                raise RingError("io", str(exc)) from exc

    def close(self) -> None:
        for sock in (self._send, self._recv):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass


class TcpRing:
    """Bundle of backends and their transports so callers can clean up."""

    def __init__(
        self, backends: list[RingBackend], transports: list[FullDuplexTcpTransport]
    ) -> None:
        self.backends = backends
        self._transports = transports

    def close(self) -> None:
        for t in self._transports:
            t.close()


def dial_with_backoff(
    addr: tuple[str, int], attempts: int = 5, base: float = 0.005
) -> socket.socket:
    """Dial ``addr`` with exponential backoff + jitter (mirrors dialWithBackoff)."""
    last_err: Exception | None = None
    delay = base
    for _ in range(attempts):
        try:
            return socket.create_connection(addr, timeout=5.0)
        except OSError as exc:
            last_err = exc
            jitter = delay * random.uniform(0.0, 0.25)
            time.sleep(delay + jitter)
            delay *= 2
    raise RingError("io", str(last_err) if last_err else "dial failed")


def create_tcp_ring(num_ranks: int) -> TcpRing:
    """Spawn a TCP ring of ``num_ranks`` backends on localhost ephemeral ports.

    Each rank opens an outbound connection to its right neighbour and accepts an
    inbound connection from its left neighbour, wrapping both in a
    ``FullDuplexTcpTransport``. Mirrors create_tcp_ring.
    """
    listeners: list[socket.socket] = []
    addrs: list[tuple[str, int]] = []
    try:
        for _ in range(num_ranks):
            ln = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ln.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            ln.bind(("127.0.0.1", 0))
            ln.listen(1)
            listeners.append(ln)
            addrs.append(ln.getsockname())
    except OSError as exc:
        for ln in listeners:
            ln.close()
        raise RingError("io", str(exc)) from exc

    recv_streams: list[socket.socket | None] = [None] * num_ranks
    send_streams: list[socket.socket | None] = [None] * num_ranks
    errors: list[Exception] = []
    errors_lock = threading.Lock()
    threads: list[threading.Thread] = []

    def _accept(rank: int) -> None:
        try:
            conn, _ = listeners[rank].accept()
            recv_streams[rank] = conn
        except OSError as exc:
            with errors_lock:
                errors.append(RingError("io", str(exc)))
        finally:
            listeners[rank].close()

    def _dial(rank: int) -> None:
        right = addrs[(rank + 1) % num_ranks]
        try:
            send_streams[rank] = dial_with_backoff(right)
        except Exception as exc:  # noqa: BLE001
            with errors_lock:
                errors.append(exc)

    for rank in range(num_ranks):
        t = threading.Thread(target=_accept, args=(rank,))
        t.start()
        threads.append(t)
    for rank in range(num_ranks):
        t = threading.Thread(target=_dial, args=(rank,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if errors:
        for c in recv_streams + send_streams:
            if c is not None:
                c.close()
        raise errors[0]

    backends: list[RingBackend] = []
    transports: list[FullDuplexTcpTransport] = []
    for rank in range(num_ranks):
        transport = FullDuplexTcpTransport(send_streams[rank], recv_streams[rank])  # type: ignore[arg-type]
        transports.append(transport)
        backends.append(RingBackend(rank, num_ranks, transport))
    return TcpRing(backends, transports)
