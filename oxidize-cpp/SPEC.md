# SPEC: Async Layer-Ahead Prefetch for GGUF Mmap'd Weights

## Goal
Build on PR #28 (SSD-aware mmap policies) by adding **asynchronous layer-ahead prefetch** and completing the **transparent hugepages** hint that the autotune plan already recommended but did not apply. Together these reduce cold page-fault latency, TLB pressure, and decode jitter for large mmap'd models.

## Research basis
- **LLM in a Flash** (Apple, 2024): flash/SSD bandwidth is high for large sequential reads but latency-penalized for small random reads. The key win is reading larger, contiguous chunks ahead of need.
- **llama.cpp prefetch PRs**: overlapping weight fetch for layer *n+1* while computing layer *n* is the standard memory-hiding technique for CPU/GPU offloading.
- Linux `readahead(2)` gives the kernel an explicit, fd-offset based prefetch hint that is more aggressive and better targeted than `madvise(MADV_WILLNEED)` on an anonymous mapping.
- Transparent hugepages reduce the number of page-table entries and page faults for large read-mostly mappings, improving both cold-start and steady-state memory bandwidth.

## Design

### 1. Layer range map in `GgufModel`
After parsing, group tensors whose names match `blk.<N>.` into per-layer byte ranges. Store:
```cpp
struct ShardRange { size_t shard_index; off_t offset; size_t length; };
std::map<size_t, std::vector<ShardRange>> layer_ranges_;
```
Ranges are clipped to the shard's data section bounds and merged when adjacent.

### 2. `LayerPrefetcher` class (`include/oxidize/prefetcher.hpp`, `src/prefetcher.cpp`)
- Owns a background thread and a request queue.
- `request(layer_index)` enqueues the layer; the worker calls `readahead(fd, offset, len)` for every `ShardRange` of that layer.
- `prefetch_sync(layer_index)` performs the same operation on the caller thread (useful for startup).
- `stop()` drains the queue and joins the thread.
- All operations are no-ops on non-Linux or when built without layer ranges.

### 3. `mmap_hugepages` completion
- PR #28 already emitted `mmap_hugepages` in `TuningPlan` but never applied it.
- Add `bool mmap_hugepages` to `GgufLoadOptions`.
- On Linux, `advise_mmap()` now issues `madvise(MADV_HUGEPAGE)` when the flag is set.
- New CLI flag `--mmap-hugepages`; autotune turns it on for models >= 200 GiB with free 2 MiB hugepages.

### 4. CLI / autotune
- New CLI flags: `--prefetch-layers <n>` and `--mmap-hugepages`.
- `TuningPlan` gains `int prefetch_layers`.
- Autotune rule:
  - `0` if model fits comfortably in RAM (`size <= 0.5 * ram`).
  - `1` if model is large (`size > 0.8 * ram` or `size > 192 GiB`).
  - `2` if model is much larger than RAM (`size > 1.5 * ram`) to keep more I/O in flight.
- `--print-plan` emits `mmap_hugepages` and `prefetch_layers`.

### 5. `LlamaModel` integration
- At the end of construction, build the layer range map from the `GgufModel` and instantiate `LayerPrefetcher` if `prefetch_layers > 0`.
- In `forward_single`, at the top of the layer loop for layer `l`, call `prefetcher_.request(l + prefetch_layers_)`.
- The first few layers are prefetched synchronously during model load so the prefill is not stalled.

### 6. Testing
- `tests/prefetch_layer_map_test.cpp`: load a synthetic GGUF with `blk.0.*` and `blk.1.*` tensors and verify per-layer ranges are disjoint and sorted.
- `tests/gguf_mmap_policy_test.cpp`: load a fixture with `mmap_hugepages=true` and ensure it succeeds.
- `tests/autotune_test.cpp`: assert huge-model plans set `prefetch_layers > 0` and `mmap_hugepages=true` when applicable, and small-model plans disable them.
- Remote QA on `ai@192.168.1.132`: build and run the focused tests; run single-token decode on a 14 GiB model with `--prefetch-layers 0/1/2` to verify no crash and inspect timing.

## Known limitation
- Batched prefill (`forward_batched`, triggered by `--tokens "1,2,3"` with batch > 1) segfaults on this codebase independent of this PR. Single-token prefill/decode works and is used for the prefetch smoke tests.

## Success criteria
- Focused tests pass locally and on the remote box.
- No regressions for existing `autotune_test` and `gguf_mmap_policy_test`.
- Single-token decode on a real model succeeds with prefetch enabled.
- Cold-cache decode on a model larger than RAM is expected to show reduced jitter vs `--prefetch-layers 0` (validated where cache can be cleared; otherwise accepted as the established Linux readahead/hugepage behavior).

## Out of scope
- GPU async copy engines (CUDA/HIP).
- Predictive expert prefetching for MoE.
- Reordering layers or windowing (future work).
- Fixing the unrelated `forward_batched` segfault.
