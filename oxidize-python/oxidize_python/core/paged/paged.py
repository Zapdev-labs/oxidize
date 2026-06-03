"""vLLM-style paged attention scheduler mirroring oxidize-golang/core/paged."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    block_size: int = 16
    total_blocks: int = 1024
    max_requests: int = 32
    max_tokens_per_req: int = 2048
    preemption: str = "recompute"


def default_scheduler_config() -> SchedulerConfig:
    return SchedulerConfig()


BlockHash = tuple[int, ...]


def block_hash_string(h: BlockHash) -> str:
    hex_digits = "0123456789abcdef"
    out = []
    for b in h:
        out.append(hex_digits[(b >> 4) & 0xF])
        out.append(hex_digits[b & 0xF])
    return "".join(out)


@dataclass
class PhysicalBlock:
    id: int
    ref: int = 0
    dirty: bool = False


class BlockPool:
    def __init__(self, n: int, block_size: int) -> None:
        if n <= 0:
            n = 1
        if block_size <= 0:
            block_size = 16
        self._mu = threading.Lock()
        self.blocks = [PhysicalBlock(id=i) for i in range(n)]
        self.free = list(range(n - 1, -1, -1))
        self.block_size = block_size

    def allocate(self) -> int:
        with self._mu:
            if not self.free:
                raise RuntimeError("paged: no free blocks")
            block_id = self.free.pop()
            self.blocks[block_id].ref = 1
            return block_id

    def release(self, block_id: int) -> None:
        with self._mu:
            if block_id < 0 or block_id >= len(self.blocks):
                return
            self.blocks[block_id].ref -= 1
            if self.blocks[block_id].ref <= 0:
                self.blocks[block_id].ref = 0
                self.blocks[block_id].dirty = False
                self.free.append(block_id)

    def free_count(self) -> int:
        with self._mu:
            return len(self.free)

    def total_count(self) -> int:
        return len(self.blocks)


@dataclass
class BlockTable:
    request_id: int
    blocks: list[int] = field(default_factory=list)


@dataclass
class Request:
    id: int
    tokens: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    max_tokens: int = 0
    priority: int = 0
    preempted: bool = False


class SchedulerError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"paged: {message}")


class Scheduler:
    def __init__(self, cfg: SchedulerConfig) -> None:
        self._mu = threading.Lock()
        self.cfg = cfg
        self.pool = BlockPool(cfg.total_blocks, cfg.block_size)
        self.queue: list[Request] = []
        self.active: dict[int, Request] = {}
        self._sequence = 0
        self.hash_table: dict[BlockHash, int] = {}

    def add_request(self, tokens: list[int], max_tokens: int) -> Request:
        with self._mu:
            if len(self.active) >= self.cfg.max_requests and self.queue:
                raise SchedulerError("max requests reached")
            self._sequence += 1
            r = Request(
                id=self._sequence,
                tokens=list(tokens),
                max_tokens=max_tokens,
            )
            blocks = max(1, (len(tokens) + self.cfg.block_size - 1) // self.cfg.block_size)
            for _ in range(blocks):
                r.block_table.append(self.pool.allocate())
            self.queue.append(r)
            return r

    def step(self) -> list[Request] | None:
        with self._mu:
            if not self.queue:
                return None
            if self.cfg.preemption == "recompute" and self.pool.free_count() == 0:
                victim: Request | None = None
                for r in self.active.values():
                    if victim is None or r.priority < victim.priority:
                        victim = r
                if victim is not None:
                    for b in victim.block_table:
                        self.pool.release(b)
                    del self.active[victim.id]
                    victim.preempted = True
                    return None
            scheduled: list[Request] = []
            for r in self.queue:
                if len(self.active) >= self.cfg.max_requests:
                    break
                self.active[r.id] = r
                scheduled.append(r)
            self.queue = self.queue[len(scheduled) :]
            return scheduled or None

    def finish(self, req_id: int) -> None:
        with self._mu:
            r = self.active.pop(req_id, None)
            if r is None:
                return
            for b in r.block_table:
                self.pool.release(b)

    def stats(self) -> tuple[int, int, int]:
        with self._mu:
            return len(self.queue), len(self.active), self.pool.free_count()

    def register_block_hash(self, h: BlockHash, block_id: int) -> None:
        with self._mu:
            self.hash_table[h] = block_id

    def lookup_block_hash(self, h: BlockHash) -> tuple[int, bool]:
        with self._mu:
            if h in self.hash_table:
                return self.hash_table[h], True
            return 0, False


def compute_block_hash(tokens: list[int]) -> BlockHash:
    h = [0] * 16
    for i, t in enumerate(tokens):
        h[i % 16] ^= t & 0xFF
    return tuple(h)
