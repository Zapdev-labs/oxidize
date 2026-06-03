# oxidize-core/src/paged_attention/

**Generated:** 2026-06-03
**Domain:** vLLM-style PagedAttention memory management and scheduling

## OVERVIEW
vLLM-style paged attention implementation: block-based KV cache memory pool, physical block allocation/deallocation, and request scheduler with sequence state machine.

## STRUCTURE
```
paged_attention/
├── mod.rs           # Module exports, BlockId, BlockTable, PhysicalTokenBlock
├── block_pool.rs    # BlockPool allocation, block hash for copy-on-write, reference counting
└── scheduler.rs     # Sequence state machine, scheduling loop, token budget management
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Change scheduling policy | `scheduler.rs` | `Scheduler::schedule()` dispatches to strategy |
| Add sequence state | `scheduler.rs` | `SequenceStatus` enum + `Sequence` struct |
| Block allocation | `block_pool.rs` | `BlockPool::allocate()`, `free_unused_blocks()` |
| Copy-on-write pages | `block_pool.rs` | `compute_block_hash()`, ref counting |
| Token budget tuning | `scheduler.rs` | `max_num_seqs`, `max_num_batched_tokens` |
| Prefill vs decode | `scheduler.rs` | Prefill sequences go first, then running decode |

## CONVENTIONS
- **Block size is fixed**: Typically 16 tokens per block (`BLOCK_SIZE`). Tune in `mod.rs`.
- **FCFS scheduling**: Waiting queue is strictly first-come-first-served based on `arrival_order`.
- **Copy-on-write sharing**: Blocks with identical KV contents share physical memory via hash dedup and reference counting.
- **State machine**: `Waiting → Running → Finished` with no transitions back.

## ANTI-PATTERNS
- `scheduler.rs` is 2,291 lines — mixes sequence management, scheduling logic, and budget heuristics. Refactor candidate.
- Hardcoded `BLOCK_SIZE = 16` in multiple places — should be a single `BlockPoolConfig` field.
- `unwrap()` on block allocation failure — should return `SchedulerError::OutOfMemory` and preempt sequences.
- Token budget heuristics are not benchmarked against vLLM reference — need regression tests.
