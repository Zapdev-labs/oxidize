"""Ring collective operations mirroring oxidize-golang/core/mesh/ring_collective.go.

Implements ring all-reduce (``all_sum``) and ring all-gather (``all_gather``)
over an abstract ring transport, plus pipeline / tensor-parallel helpers and
little-endian f32 framing matching the Rust/Go wire format.
"""

from __future__ import annotations

import queue
import struct
import threading
from abc import ABC, abstractmethod


class RingError(Exception):
    """Failure modes of a ring collective operation (mirrors RingError)."""

    def __init__(self, kind: str, message: str = "") -> None:
        self.kind = kind
        self.message = message
        if message:
            super().__init__(f"ring: {kind}: {message}")
        else:
            super().__init__(f"ring: {kind}")


def _err_wrong_chunk_size(expected: int, actual: int) -> RingError:
    return RingError(
        "wrong_chunk_size",
        f"expected multiple of {expected}, got remainder {actual}",
    )


def _err_byte_length_mismatch(expected: int, actual: int) -> RingError:
    return RingError("byte_length_mismatch", f"expected {expected} bytes, got {actual}")


class RingNodeTransport(ABC):
    """Abstract ring transport (mirrors the Rust RingTransport trait).

    Each rank sends to its right neighbour and receives from its left
    neighbour.
    """

    @abstractmethod
    def send_to_right(self, data: bytes) -> None: ...

    @abstractmethod
    def recv_from_left(self) -> bytes: ...


class RingBackend:
    """Ring all-reduce / all-gather backend (mirrors RingBackend)."""

    def __init__(self, rank: int, num_ranks: int, transport: RingNodeTransport) -> None:
        self.rank = rank
        self.num_ranks = num_ranks
        self.transport = transport

    def all_sum(self, data: list[float]) -> None:
        """Ring all-reduce, summing ``data`` in place.

        ``len(data)`` must be divisible by ``num_ranks``.
        """
        if self.num_ranks <= 1:
            return
        if len(data) % self.num_ranks != 0:
            raise _err_wrong_chunk_size(self.num_ranks, len(data) % self.num_ranks)

        chunk_size = len(data) // self.num_ranks

        # Scatter-reduce: N-1 steps.
        send_chunk = self.rank
        recv_chunk = (self.rank + self.num_ranks - 1) % self.num_ranks
        for _ in range(self.num_ranks - 1):
            send_off = send_chunk * chunk_size
            send_buf = data[send_off : send_off + chunk_size]
            recv_bytes = self._exchange(f32_to_bytes(send_buf))
            recv_f32 = bytes_to_f32(recv_bytes, chunk_size)
            recv_off = recv_chunk * chunk_size
            for i in range(chunk_size):
                data[recv_off + i] += recv_f32[i]
            send_chunk = (send_chunk + self.num_ranks - 1) % self.num_ranks
            recv_chunk = (recv_chunk + self.num_ranks - 1) % self.num_ranks

        # All-gather: N-1 steps.
        send_chunk = (self.rank + 1) % self.num_ranks
        recv_chunk = self.rank
        for _ in range(self.num_ranks - 1):
            send_off = send_chunk * chunk_size
            send_buf = data[send_off : send_off + chunk_size]
            recv_bytes = self._exchange(f32_to_bytes(send_buf))
            recv_f32 = bytes_to_f32(recv_bytes, chunk_size)
            recv_off = recv_chunk * chunk_size
            data[recv_off : recv_off + chunk_size] = recv_f32
            send_chunk = (send_chunk + self.num_ranks - 1) % self.num_ranks
            recv_chunk = (recv_chunk + self.num_ranks - 1) % self.num_ranks

    def all_gather(self, data: list[float]) -> list[float]:
        """Ring all-gather: contribute ``data`` and receive all chunks concatenated."""
        chunk_size = len(data)
        if self.num_ranks <= 1:
            return list(data)

        out = [0.0] * (chunk_size * self.num_ranks)
        local_offset = self.rank * chunk_size
        out[local_offset : local_offset + chunk_size] = data

        send_chunk = self.rank
        recv_chunk = (self.rank + self.num_ranks - 1) % self.num_ranks
        for _ in range(self.num_ranks - 1):
            send_off = send_chunk * chunk_size
            send_buf = out[send_off : send_off + chunk_size]
            recv_bytes = self._exchange(f32_to_bytes(send_buf))
            recv_f32 = bytes_to_f32(recv_bytes, chunk_size)
            recv_off = recv_chunk * chunk_size
            out[recv_off : recv_off + chunk_size] = recv_f32
            send_chunk = (send_chunk + self.num_ranks - 1) % self.num_ranks
            recv_chunk = (recv_chunk + self.num_ranks - 1) % self.num_ranks

        return out

    def _exchange(self, send: bytes) -> bytes:
        """Concurrent send-to-right / recv-from-left to avoid ring deadlock.

        Mirrors the Rust ``tokio::join!`` pattern with a receiver thread.
        """
        result: dict[str, object] = {}

        def _recv() -> None:
            try:
                result["data"] = self.transport.recv_from_left()
            except Exception as exc:  # noqa: BLE001
                result["err"] = exc

        recv_thread = threading.Thread(target=_recv)
        recv_thread.start()
        send_err: Exception | None = None
        try:
            self.transport.send_to_right(send)
        except Exception as exc:  # noqa: BLE001
            send_err = exc
        recv_thread.join()
        if send_err is not None:
            raise send_err
        if "err" in result:
            raise result["err"]  # type: ignore[misc]
        return result["data"]  # type: ignore[return-value]


def pipeline_send(ring: RingBackend, activations: list[float]) -> None:
    """Forward activations to the next pipeline stage (mirrors pipeline_send)."""
    ring.transport.send_to_right(f32_to_bytes(activations))


def pipeline_recv(ring: RingBackend, num_floats: int) -> list[float]:
    """Receive activations from the previous pipeline stage (mirrors pipeline_recv)."""
    data = ring.transport.recv_from_left()
    return bytes_to_f32(data, num_floats)


def tensor_parallel_all_sum(ring: RingBackend, partial: list[float]) -> None:
    """Tensor-parallel all_sum over the ring (mirrors tensor_parallel_all_sum)."""
    ring.all_sum(partial)


def tensor_parallel_all_gather(ring: RingBackend, partial: list[float]) -> list[float]:
    """Gather outputs from all ranks (mirrors tensor_parallel_all_gather)."""
    return ring.all_gather(partial)


def f32_to_bytes(data: list[float]) -> bytes:
    """Convert a list of float32 into little-endian bytes (matches Rust framing)."""
    return struct.pack(f"<{len(data)}f", *data)


def bytes_to_f32(data: bytes, count: int) -> list[float]:
    """Convert little-endian bytes into a list of ``count`` float32 values."""
    if len(data) != count * 4:
        raise _err_byte_length_mismatch(count * 4, len(data))
    return list(struct.unpack(f"<{count}f", data))


def bytes_to_f32_slice(data: bytes) -> list[float]:
    """Convert little-endian bytes into a freshly sized float32 list."""
    if len(data) % 4 != 0:
        raise _err_byte_length_mismatch((len(data) // 4 + 1) * 4, len(data))
    return bytes_to_f32(data, len(data) // 4)


class ChannelRingTransport(RingNodeTransport):
    """In-memory ring transport for tests (mirrors ChannelRingTransport)."""

    def __init__(self, right_tx: "queue.Queue[bytes]", left_rx: "queue.Queue[bytes]") -> None:
        self.right_tx = right_tx
        self.left_rx = left_rx

    def send_to_right(self, data: bytes) -> None:
        self.right_tx.put(data)

    def recv_from_left(self) -> bytes:
        return self.left_rx.get()


def create_mock_ring(num_ranks: int) -> list[RingBackend]:
    """Build a mock ring of ``num_ranks`` backends via in-memory channels.

    Mirrors create_mock_ring.
    """
    channels: list[queue.Queue[bytes]] = [queue.Queue(maxsize=64) for _ in range(num_ranks)]
    backends: list[RingBackend] = []
    for rank in range(num_ranks):
        transport = ChannelRingTransport(
            # rank sends to its right neighbour's inbound channel.
            right_tx=channels[(rank + 1) % num_ranks],
            # rank receives from its own inbound channel (fed by its left).
            left_rx=channels[rank],
        )
        backends.append(RingBackend(rank, num_ranks, transport))
    return backends
