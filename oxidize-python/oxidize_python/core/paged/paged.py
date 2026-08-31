"""vLLM-style paged attention scheduler mirroring oxidize-golang/core/paged."""

from __future__ import annotations

import enum
import threading
from collections import deque
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
    # hash is the prefix-cache hash for this block, or None when the block is
    # not currently part of the prefix cache.
    hash: BlockHash | None = None
    # last_accessed is the monotonic access counter used for LRU eviction.
    last_accessed: int = 0


class BlockPool:
    def __init__(self, n: int, block_size: int) -> None:
        if n <= 0:
            n = 1
        if block_size <= 0:
            block_size = 16
        self._mu = threading.RLock()
        self.blocks = [PhysicalBlock(id=i) for i in range(n)]
        self.free = list(range(n - 1, -1, -1))
        self.block_size = block_size
        # prefix_cache maps a block hash to the physical block id holding that
        # prefix.
        self.prefix_cache: dict[BlockHash, int] = {}
        # access_counter is a monotonically increasing LRU clock.
        self.access_counter = 0

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

    def allocated_count(self) -> int:
        with self._mu:
            return len(self.blocks) - len(self.free)

    def _allocate_locked(self) -> int:
        """Pop a free block and mark it allocated (ref=1). Caller holds _mu."""
        if not self.free:
            raise RuntimeError("paged: no free blocks")
        block_id = self.free.pop()
        blk = self.blocks[block_id]
        blk.ref = 1
        blk.hash = None
        blk.dirty = False
        return block_id

    def allocate_blocks(self, n: int) -> list[int]:
        """Allocate ``n`` blocks with all-or-nothing semantics.

        If fewer than ``n`` blocks are free, no blocks are allocated and a
        RuntimeError is raised. Mirrors BlockPool::allocate_blocks.
        """
        with self._mu:
            if n <= 0:
                return []
            if len(self.free) < n:
                raise RuntimeError("paged: no free blocks")
            return [self._allocate_locked() for _ in range(n)]

    def inc_ref(self, block_id: int) -> None:
        """Increment the reference count of an allocated block.

        Raises if the block was not allocated (ref==0). Mirrors
        BlockPool::inc_ref.
        """
        with self._mu:
            if block_id < 0 or block_id >= len(self.blocks):
                raise RuntimeError("paged: invalid block id")
            if self.blocks[block_id].ref == 0:
                raise RuntimeError("paged: block not allocated")
            self.blocks[block_id].ref += 1

    def dec_ref(self, block_id: int) -> None:
        """Decrement the reference count, returning the block to the free list
        when it reaches zero. Mirrors BlockPool::dec_ref.
        """
        with self._mu:
            if block_id < 0 or block_id >= len(self.blocks):
                raise RuntimeError("paged: invalid block id")
            if self.blocks[block_id].ref == 0:
                raise RuntimeError("paged: block not allocated")
            self.blocks[block_id].ref -= 1
            if self.blocks[block_id].ref == 0:
                self.blocks[block_id].dirty = False
                if block_id not in self.free:
                    self.free.append(block_id)

    def ref_count(self, block_id: int) -> int:
        with self._mu:
            if block_id < 0 or block_id >= len(self.blocks):
                return 0
            return self.blocks[block_id].ref

    def lookup_prefix_cache(self, h: BlockHash) -> tuple[int, bool]:
        """Return the physical block id for a cached prefix hash, updating its
        LRU access time. Stale entries (block freed) are pruned and reported as
        not found. Mirrors BlockPool::lookup_prefix_cache.
        """
        with self._mu:
            block_id = self.prefix_cache.get(h)
            if block_id is None:
                return 0, False
            if (
                block_id < 0
                or block_id >= len(self.blocks)
                or self.blocks[block_id].ref == 0
            ):
                del self.prefix_cache[h]
                return 0, False
            self.access_counter += 1
            self.blocks[block_id].last_accessed = self.access_counter
            return block_id, True

    def insert_prefix_cache(self, h: BlockHash, block_id: int) -> None:
        """Record a hash -> block mapping for prefix reuse. The block must be
        allocated. If the hash already exists the existing mapping wins
        (first-seen). Mirrors BlockPool::insert_prefix_cache.
        """
        with self._mu:
            if (
                block_id < 0
                or block_id >= len(self.blocks)
                or self.blocks[block_id].ref == 0
            ):
                return
            self.blocks[block_id].hash = h
            self.access_counter += 1
            self.blocks[block_id].last_accessed = self.access_counter
            if h not in self.prefix_cache:
                self.prefix_cache[h] = block_id

    def evict_lru_prefix_cache_entry(self) -> bool:
        """Remove the least-recently-used cache entry whose block ref count is
        zero. Returns True if an entry was evicted. Mirrors
        BlockPool::evict_lru_prefix_cache_entry.
        """
        with self._mu:
            victim_hash: BlockHash | None = None
            victim_id = -1
            victim_clock = 0
            for h, block_id in self.prefix_cache.items():
                if block_id < 0 or block_id >= len(self.blocks):
                    continue
                if self.blocks[block_id].ref != 0:
                    continue
                if victim_id == -1 or self.blocks[block_id].last_accessed < victim_clock:
                    victim_hash = h
                    victim_id = block_id
                    victim_clock = self.blocks[block_id].last_accessed
            if victim_id == -1 or victim_hash is None:
                return False
            del self.prefix_cache[victim_hash]
            self.blocks[victim_id].hash = None
            return True

    def clear_prefix_cache(self) -> None:
        """Remove all prefix-cache entries (e.g. on model switch). Mirrors
        BlockPool::clear_prefix_cache.
        """
        with self._mu:
            for h, block_id in self.prefix_cache.items():
                if 0 <= block_id < len(self.blocks) and self.blocks[block_id].hash == h:
                    self.blocks[block_id].hash = None
            self.prefix_cache = {}

    def prefix_cache_len(self) -> int:
        with self._mu:
            return len(self.prefix_cache)

    def copy_on_write(self, block_id: int) -> int | None:
        """Implement copy-on-write for a shared block. If the block's ref count
        is > 1, a new block is allocated, the original's ref is decremented, and
        the new block id is returned. If the block is not shared, None is
        returned. Mirrors BlockPool::copy_on_write.
        """
        with self._mu:
            if block_id < 0 or block_id >= len(self.blocks):
                raise RuntimeError("paged: invalid block id")
            if self.blocks[block_id].ref <= 1:
                return None
            new_id = self._allocate_locked()
            self.blocks[block_id].ref -= 1
            return new_id


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
    """Compute a deterministic FNV-1a block hash from tokens, mirroring the Rust
    ``compute_block_hash`` (64-bit FNV-1a). The 64-bit digest is stored in the
    leading 8 bytes of the 16-byte hash so existing string/map behaviour is
    preserved while collisions are far less likely than the old XOR-fold scheme.
    """
    fnv_offset = 0xCBF29CE484222325
    fnv_prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    h = fnv_offset
    for t in tokens:
        h = (h * fnv_prime) & mask
        h ^= t & 0xFFFFFFFF
    out = [0] * 16
    for i in range(8):
        out[i] = (h >> (8 * (7 - i))) & 0xFF
    return tuple(out)


# v2 budgeted three-phase scheduler.
#
# Ports the vLLM-style scheduler from oxidize-core/src/paged_attention/scheduler
# (core.rs, config.rs, prefix_cache.rs, lifecycle.rs, sequence.rs) and mirrors
# oxidize-golang/core/paged/scheduler_v2.go. Additive: the legacy ``Scheduler``
# and ``BlockPool`` API above are left untouched so existing callers keep
# working, while ``SchedulerV2`` provides budgeted batching, prefill chunking,
# prefix caching, and copy-on-write.

SeqID = int


class SequenceStatus(enum.Enum):
    WAITING = 1
    RUNNING = 2
    FINISHED = 3


@dataclass
class SchedulerV2Config:
    # max_batched_tokens is the token budget per scheduler step.
    max_batched_tokens: int = 512
    # prefill_chunk_size is the default tokens per prefill chunk.
    prefill_chunk_size: int = 16
    # max_running_seqs is the maximum number of simultaneously running sequences.
    max_running_seqs: int = 8


def default_scheduler_v2_config() -> SchedulerV2Config:
    return SchedulerV2Config()


class SeqBlockTable:
    """Per-sequence logical->physical block mapping. Mirrors
    block_pool.rs::BlockTable.
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            block_size = 16
        self.logical_to_physical: list[int] = []
        self.num_tokens = 0
        self.block_size = block_size

    def num_blocks(self) -> int:
        return len(self.logical_to_physical)

    def physical_blocks(self) -> list[int]:
        return self.logical_to_physical

    def get_physical_block(self, logical: int) -> tuple[int, bool]:
        if logical < 0 or logical >= len(self.logical_to_physical):
            return 0, False
        return self.logical_to_physical[logical], True

    def append_block(self, block_id: int) -> None:
        self.logical_to_physical.append(block_id)

    def set_physical_block(self, logical: int, block_id: int) -> None:
        if 0 <= logical < len(self.logical_to_physical):
            self.logical_to_physical[logical] = block_id

    def append_token(self) -> bool:
        """Advance the token count, returning True if a new block is required.
        Mirrors BlockTable::append_token.
        """
        idx = self.num_tokens % self.block_size
        needs = idx == 0 and self.num_tokens > 0
        self.num_tokens += 1
        return needs

    def blocks_needed_for_tokens(self, n: int) -> int:
        """Return how many additional blocks are needed to store ``n`` more
        tokens. Mirrors BlockTable::blocks_needed_for_tokens.
        """
        future = self.num_tokens + n
        future_blocks = (future + self.block_size - 1) // self.block_size
        need = future_blocks - len(self.logical_to_physical)
        return max(0, need)


class Sequence:
    """A single generation request managed by the v2 scheduler. Mirrors
    scheduler/sequence.rs::Sequence.
    """

    def __init__(
        self,
        seq_id: SeqID,
        prompt_tokens: list[int],
        block_size: int,
        max_new_tokens: int,
        stop_token: int = 0,
        has_stop_token: bool = False,
    ) -> None:
        self.id = seq_id
        self.status = SequenceStatus.WAITING
        self.prompt_tokens = list(prompt_tokens)
        self.generated_tokens: list[int] = []
        self.block_table = SeqBlockTable(block_size)
        self.arrival_order = 0
        self.max_new_tokens = max_new_tokens
        self.stop_token = stop_token
        self.has_stop_token = has_stop_token
        self.num_prefilled = 0

    def num_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    def remaining_prefill_tokens(self) -> int:
        return max(0, len(self.prompt_tokens) - self.num_prefilled)

    def record_prefilled(self, count: int) -> None:
        self.num_prefilled += count
        if self.num_prefilled > len(self.prompt_tokens):
            self.num_prefilled = len(self.prompt_tokens)

    def append_token(self, token: int) -> None:
        self.generated_tokens.append(token)

    def is_finished(self) -> bool:
        if len(self.generated_tokens) >= self.max_new_tokens:
            return True
        if (
            self.has_stop_token
            and self.generated_tokens
            and self.generated_tokens[-1] == self.stop_token
        ):
            return True
        return False


@dataclass
class InputBatch:
    """Flattened batch for a single forward pass. Mirrors
    scheduler/config.rs::InputBatch.
    """

    batch_size: int = 0
    seq_ids: list[SeqID] = field(default_factory=list)
    token_ids: list[list[int]] = field(default_factory=list)
    positions: list[list[int]] = field(default_factory=list)
    block_tables: list[list[int]] = field(default_factory=list)
    num_tokens: list[int] = field(default_factory=list)
    total_tokens: int = 0
    is_prefill: list[bool] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)


@dataclass
class SchedulerStepResult:
    """Returned by SchedulerV2.step. Mirrors
    scheduler/config.rs::SchedulerStepResult.
    """

    scheduled_seq_ids: list[SeqID] = field(default_factory=list)
    prefill_tokens: int = 0
    decode_tokens: int = 0
    seq_prefill_tokens: dict[SeqID, int] = field(default_factory=dict)
    seq_decode_tokens: dict[SeqID, int] = field(default_factory=dict)


class SchedulerV2:
    """Budgeted three-phase paged-attention scheduler. Mirrors
    scheduler/core.rs::Scheduler.
    """

    def __init__(
        self,
        config: SchedulerV2Config,
        total_blocks: int,
        block_size: int,
        pool: BlockPool | None = None,
    ) -> None:
        if config.max_batched_tokens <= 0:
            config.max_batched_tokens = 512
        if config.prefill_chunk_size <= 0:
            config.prefill_chunk_size = 16
        if config.max_running_seqs <= 0:
            config.max_running_seqs = 8
        self._mu = threading.RLock()
        self.config = config
        self.pool = pool if pool is not None else BlockPool(total_blocks, block_size)
        self.sequences: dict[SeqID, Sequence] = {}
        self.waiting: deque[SeqID] = deque()
        self.running: list[SeqID] = []
        self.next_arrival = 0
        self.next_seq_id = 0
        self.use_prefix_cache = True
        self._last_cow_triggered = False

    @classmethod
    def with_pool(cls, config: SchedulerV2Config, pool: BlockPool) -> "SchedulerV2":
        return cls(config, pool.total_count(), pool.block_size, pool=pool)

    def set_prefix_cache_enabled(self, enabled: bool) -> None:
        with self._mu:
            self.use_prefix_cache = enabled

    def add_sequence(self, seq: Sequence) -> None:
        with self._mu:
            seq.arrival_order = self.next_arrival
            self.next_arrival += 1
            if seq.id <= 0:
                self.next_seq_id += 1
                seq.id = self.next_seq_id
            elif seq.id > self.next_seq_id:
                self.next_seq_id = seq.id
            self.sequences[seq.id] = seq
            self.waiting.append(seq.id)

    def add_request(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        stop_token: int = 0,
        has_stop_token: bool = False,
    ) -> SeqID:
        with self._mu:
            block_size = self.pool.block_size
            self.next_seq_id += 1
            seq_id = self.next_seq_id
        seq = Sequence(
            seq_id, prompt_tokens, block_size, max_new_tokens, stop_token, has_stop_token
        )
        self.add_sequence(seq)
        return seq_id

    def get_sequence(self, seq_id: SeqID) -> Sequence | None:
        with self._mu:
            return self.sequences.get(seq_id)

    def waiting_count(self) -> int:
        with self._mu:
            return len(self.waiting)

    def running_count(self) -> int:
        with self._mu:
            return len(self.running)

    def _apply_prefill_chunk(self, seq: Sequence, chunk_size: int) -> None:
        """Allocate blocks and advance token counters for a chunk. Caller holds
        _mu. Mirrors core.rs::apply_prefill_chunk.
        """
        blocks_needed = seq.block_table.blocks_needed_for_tokens(chunk_size)
        if blocks_needed > 0:
            ids = self.pool.allocate_blocks(blocks_needed)
            for block_id in ids:
                seq.block_table.append_block(block_id)
        for _ in range(chunk_size):
            seq.block_table.append_token()
        seq.record_prefilled(chunk_size)

    def _prefill_chunk(self, seq: Sequence, chunk_size: int) -> None:
        """Dispatch to the prefix-cache-aware path when enabled. Caller holds
        _mu.
        """
        if self.use_prefix_cache:
            self._apply_prefill_chunk_with_prefix_cache(seq, chunk_size)
        else:
            self._apply_prefill_chunk(seq, chunk_size)

    def step(self) -> SchedulerStepResult:
        """Perform one scheduler step using the three-phase policy:

        Phase 1: decode for fully-prefilled running sequences (1 token each)
        Phase 2: continue prefill for partially-prefilled running sequences
        Phase 3: prefill from the waiting queue (FCFS)

        The token budget (max_batched_tokens) is enforced across all phases.
        Mirrors scheduler/core.rs::step.
        """
        with self._mu:
            budget = self.config.max_batched_tokens
            scheduled: list[SeqID] = []
            prefill_tokens = 0
            decode_tokens = 0
            seq_prefill: dict[SeqID, int] = {}
            seq_decode: dict[SeqID, int] = {}

            running_ids = list(self.running)

            # --- Phase 1: decode for fully-prefilled running sequences ---
            for seq_id in running_ids:
                seq = self.sequences.get(seq_id)
                if seq is None:
                    raise SchedulerError(f"sequence {seq_id} not found")
                if seq.is_finished() or seq.remaining_prefill_tokens() > 0:
                    continue
                if budget == 0:
                    break
                needs_block = seq.block_table.append_token()
                if needs_block:
                    ids = self.pool.allocate_blocks(1)
                    seq.block_table.append_block(ids[0])
                else:
                    self._cow_decode_block(seq)
                scheduled.append(seq_id)
                budget -= 1
                decode_tokens += 1
                seq_decode[seq_id] = 1

            # --- Phase 2: continue prefill for partially-prefilled running ---
            for seq_id in running_ids:
                seq = self.sequences.get(seq_id)
                if seq is None or seq.is_finished() or seq.remaining_prefill_tokens() == 0:
                    continue
                chunk_size = min(
                    seq.remaining_prefill_tokens(),
                    self.config.prefill_chunk_size,
                    budget,
                )
                if chunk_size == 0:
                    continue
                self._prefill_chunk(seq, chunk_size)
                scheduled.append(seq_id)
                budget -= chunk_size
                prefill_tokens += chunk_size
                seq_prefill[seq_id] = seq_prefill.get(seq_id, 0) + chunk_size

            # --- Phase 3: prefill chunks from the waiting queue (FCFS) ---
            still_waiting: deque[SeqID] = deque()
            running_count = len(scheduled)
            for seq_id in self.waiting:
                seq = self.sequences.get(seq_id)
                if seq is None:
                    raise SchedulerError(f"sequence {seq_id} not found")
                if seq.status != SequenceStatus.WAITING:
                    still_waiting.append(seq_id)
                    continue
                if running_count >= self.config.max_running_seqs:
                    still_waiting.append(seq_id)
                    continue
                remaining = seq.remaining_prefill_tokens()
                if remaining == 0:
                    seq.status = SequenceStatus.RUNNING
                    scheduled.append(seq_id)
                    running_count += 1
                    continue
                chunk_size = min(remaining, self.config.prefill_chunk_size, budget)
                if chunk_size == 0:
                    still_waiting.append(seq_id)
                    continue
                self._prefill_chunk(seq, chunk_size)
                seq.status = SequenceStatus.RUNNING
                scheduled.append(seq_id)
                running_count += 1
                budget -= chunk_size
                prefill_tokens += chunk_size
                seq_prefill[seq_id] = seq_prefill.get(seq_id, 0) + chunk_size

            self.waiting = still_waiting
            self.running = list(scheduled)

            return SchedulerStepResult(
                scheduled_seq_ids=scheduled,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                seq_prefill_tokens=seq_prefill,
                seq_decode_tokens=seq_decode,
            )

    def build_input_batch(self, res: SchedulerStepResult) -> InputBatch:
        """Flatten a step result into a single batch. Mirrors
        core.rs::build_input_batch.
        """
        with self._mu:
            batch = InputBatch()
            for seq_id in res.scheduled_seq_ids:
                seq = self.sequences.get(seq_id)
                if seq is None:
                    continue
                prefill_count = res.seq_prefill_tokens.get(seq_id, 0)
                decode_count = res.seq_decode_tokens.get(seq_id, 0)

                if prefill_count > 0:
                    start = max(0, seq.num_prefilled - prefill_count)
                    end = min(start + prefill_count, len(seq.prompt_tokens))
                    chunk = list(seq.prompt_tokens[start:end])
                    pos = list(range(start, end))
                    batch.seq_ids.append(seq_id)
                    batch.token_ids.append(chunk)
                    batch.positions.append(pos)
                    batch.block_tables.append(list(seq.block_table.physical_blocks()))
                    batch.num_tokens.append(prefill_count)
                    batch.is_prefill.append(True)
                    batch.context_lens.append(seq.num_prefilled)
                    batch.total_tokens += prefill_count
                elif decode_count > 0 and not seq.is_finished():
                    decode_pos = max(0, seq.num_tokens() - 1)
                    tok = [seq.generated_tokens[-1]] if seq.generated_tokens else []
                    batch.seq_ids.append(seq_id)
                    batch.token_ids.append(tok)
                    batch.positions.append([decode_pos])
                    batch.block_tables.append(list(seq.block_table.physical_blocks()))
                    batch.num_tokens.append(1)
                    batch.is_prefill.append(False)
                    batch.context_lens.append(seq.num_tokens())
                    batch.total_tokens += 1
            batch.batch_size = len(batch.seq_ids)
            return batch

    def find_prefix_cache_hits(self, seq_id: SeqID) -> int:
        """Return how many prompt tokens of a sequence can be served from the
        prefix cache. Mirrors prefix_cache.rs::find_prefix_cache_hits.
        """
        with self._mu:
            seq = self.sequences.get(seq_id)
            if seq is None:
                raise SchedulerError(f"sequence {seq_id} not found")
            return self._find_prefix_cache_hits(seq)

    def _find_prefix_cache_hits(self, seq: Sequence) -> int:
        prompt = seq.prompt_tokens
        if not prompt:
            return 0
        block_size = seq.block_table.block_size
        cached = 0
        num_blocks = (len(prompt) + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            block_end = min((block_idx + 1) * block_size, len(prompt))
            h = compute_block_hash(prompt[:block_end])
            _, found = self.pool.lookup_prefix_cache(h)
            if found:
                cached = block_end
            else:
                break
        return cached

    def _apply_prefill_chunk_with_prefix_cache(self, seq: Sequence, chunk_size: int) -> int:
        """Allocate/reuse blocks for a chunk, sharing cached prefix blocks and
        returning the count of newly-computed tokens. Mirrors
        prefix_cache.rs::apply_prefill_chunk_with_prefix_cache.
        """
        prompt = seq.prompt_tokens
        block_size = seq.block_table.block_size
        already_prefilled = seq.num_prefilled
        this_chunk = min(seq.remaining_prefill_tokens(), chunk_size)
        if this_chunk == 0:
            return 0

        # Compute total cached prefix length for this prompt.
        cached_total = 0
        if prompt:
            num_blocks = (len(prompt) + block_size - 1) // block_size
            for block_idx in range(num_blocks):
                block_end = min((block_idx + 1) * block_size, len(prompt))
                h = compute_block_hash(prompt[:block_end])
                _, found = self.pool.lookup_prefix_cache(h)
                if found:
                    cached_total = block_end
                else:
                    break

        chunk_end = already_prefilled + this_chunk
        cached_in_chunk = 0
        if cached_total > already_prefilled:
            cached_in_chunk = max(0, min(cached_total, chunk_end) - already_prefilled)
        new_tokens = this_chunk - cached_in_chunk

        # Ensure block table has physical blocks for all tokens up to chunk_end.
        target_blocks = (chunk_end + block_size - 1) // block_size
        current_blocks = seq.block_table.num_blocks()
        for block_idx in range(current_blocks, target_blocks):
            block_end = min((block_idx + 1) * block_size, len(prompt))
            h = compute_block_hash(prompt[:block_end])
            if block_end <= cached_total:
                cid, found = self.pool.lookup_prefix_cache(h)
                if found:
                    self.pool.inc_ref(cid)
                    block_id = cid
                else:
                    block_id = self.pool.allocate_blocks(1)[0]
            else:
                block_id = self.pool.allocate_blocks(1)[0]
            seq.block_table.append_block(block_id)

        # Advance token counters.
        for _ in range(this_chunk):
            seq.block_table.append_token()
        seq.record_prefilled(this_chunk)

        # Insert newly-computed blocks into the prefix cache.
        for block_idx in range(target_blocks):
            block_end = min((block_idx + 1) * block_size, len(prompt))
            if block_end > cached_total:
                h = compute_block_hash(prompt[:block_end])
                pid, found = seq.block_table.get_physical_block(block_idx)
                if found:
                    self.pool.insert_prefix_cache(h, pid)

        return new_tokens

    def cow_decode_block(self, seq_id: SeqID) -> bool:
        """Trigger copy-on-write on a sequence's last block if shared. Mirrors
        prefix_cache.rs::cow_decode_block.
        """
        with self._mu:
            seq = self.sequences.get(seq_id)
            if seq is None:
                raise SchedulerError(f"sequence {seq_id} not found")
            self._cow_decode_block(seq)
            return self._last_cow_triggered

    def _cow_decode_block(self, seq: Sequence) -> None:
        self._last_cow_triggered = False
        last_logical = seq.block_table.num_blocks() - 1
        if last_logical < 0:
            return
        original_id, ok = seq.block_table.get_physical_block(last_logical)
        if not ok:
            return
        new_id = self.pool.copy_on_write(original_id)
        if new_id is not None:
            seq.block_table.set_physical_block(last_logical, new_id)
            self._last_cow_triggered = True

    def postprocess_step(self, sampled: dict[SeqID, int]) -> None:
        """Append sampled tokens, detect finished sequences, and reclaim their
        blocks. Mirrors lifecycle.rs::postprocess_step.
        """
        with self._mu:
            for seq_id, token in sampled.items():
                seq = self.sequences.get(seq_id)
                if seq is None or seq.status != SequenceStatus.RUNNING:
                    continue
                seq.append_token(token)

            finished: list[SeqID] = []
            for seq_id in self.running:
                seq = self.sequences.get(seq_id)
                if seq is None:
                    continue
                if seq.is_finished():
                    finished.append(seq_id)
            for seq_id in finished:
                self._finish_sequence(seq_id)

    def _finish_sequence(self, seq_id: SeqID) -> None:
        seq = self.sequences.get(seq_id)
        if seq is None:
            raise SchedulerError(f"sequence {seq_id} not found")
        seq.status = SequenceStatus.FINISHED
        for b in seq.block_table.physical_blocks():
            self.pool.dec_ref(b)
        self.running = [s for s in self.running if s != seq_id]

    def finish_sequence(self, seq_id: SeqID) -> None:
        with self._mu:
            self._finish_sequence(seq_id)

    def preempt_sequence(self, seq_id: SeqID) -> None:
        """Free a sequence's blocks, reset its prefill state, and return it to
        the front of the waiting queue. Mirrors
        prefix_cache.rs::preempt_sequence.
        """
        with self._mu:
            seq = self.sequences.get(seq_id)
            if seq is None:
                raise SchedulerError(f"sequence {seq_id} not found")
            for b in seq.block_table.physical_blocks():
                self.pool.dec_ref(b)
            seq.block_table = SeqBlockTable(seq.block_table.block_size)
            seq.num_prefilled = 0
            seq.status = SequenceStatus.WAITING
            self.running = [s for s in self.running if s != seq_id]
            if seq_id not in self.waiting:
                self.waiting.appendleft(seq_id)

    def remove_sequence(self, seq_id: SeqID) -> None:
        """Remove a sequence entirely, freeing blocks if still running. Mirrors
        lifecycle.rs::remove_sequence.
        """
        with self._mu:
            seq = self.sequences.get(seq_id)
            if seq is None:
                raise SchedulerError(f"sequence {seq_id} not found")
            del self.sequences[seq_id]
            if seq.status == SequenceStatus.RUNNING:
                for b in seq.block_table.physical_blocks():
                    self.pool.dec_ref(b)
                self.running = [s for s in self.running if s != seq_id]
            self.waiting = deque(s for s in self.waiting if s != seq_id)

    def invalidate_prefix_cache(self) -> None:
        with self._mu:
            self.pool.clear_prefix_cache()

    def drain_and_reinitialize(self) -> None:
        """Free all blocks, clear the prefix cache, and reset the scheduler so it
        can accept a new backend/model. Mirrors
        lifecycle.rs::drain_and_reinitialize.
        """
        with self._mu:
            for seq in self.sequences.values():
                for b in seq.block_table.physical_blocks():
                    try:
                        self.pool.dec_ref(b)
                    except RuntimeError:
                        # Blocks may already be free.
                        pass
            self.pool.clear_prefix_cache()
            self.sequences = {}
            self.waiting = deque()
            self.running = []
            self.next_arrival = 0
