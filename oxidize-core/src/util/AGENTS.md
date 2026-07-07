# oxidize-core/src/util

**Domain:** Low-level utilities — mmap, debug dumps, benchmark discovery, WASM bridge

## OVERVIEW
Grab-bag of low-level utilities: centralized `unsafe` mmap/byte reads, an env-gated attention-debug dump, an env-driven benchmark/perplexity case discovery layer, and the WASM web-worker inference bridge.

## STRUCTURE
```
util/
├── bytes.rs            # mmap + byte-read primitives (the one place platform unsafe mmap lives)
├── attn_dump.rs        # OX_ATTN_DUMP debug tap for attention vectors
├── benchmark_suite.rs  # env-driven benchmark/perplexity case discovery
└── web_worker.rs       # WASM worker protocol + embedded TS interface contracts
```

## KEY FUNCTIONS
| Module | Key API | Role |
|--------|---------|------|
| `bytes.rs` | `map_readonly`, `read_le_i16`, `read_q8_k_bsum`, `read_volatile_byte` | GGUF/SafeTensors mmap loading, quant kernels, page prefault |
| `attn_dump.rs` | `should_dump()`, `write_block(...)` | One-shot dump of labeled attention vectors; inert unless `OX_ATTN_DUMP` set |
| `benchmark_suite.rs` | `BenchmarkCase`, `PerplexityDatasetCase` | Build benchmark suites from env (`;`/`:`-separated) |
| `web_worker.rs` | `WorkerModelConfig`, `WorkerInferenceRequest/Response`, `WorkerStreamChunk` | WASM inference bridge; 60+ line embedded TypeScript interface contracts |

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add mmap primitive | `bytes.rs` | Keep all platform `unsafe` mmap here |
| Debug CPU vs GPU attention | `attn_dump.rs` | Wired into CPU island (`model/inference/layers.rs`) and CUDA fused path (`backends/cuda/gpu_native_forward.rs`) |
| WASM worker protocol | `web_worker.rs` | `#[cfg(all(target_arch = "wasm32", feature = "wasm"))]` |
| Benchmark case discovery | `benchmark_suite.rs` | Linux-aware fs probing |

## NOTES
- `attn_dump.rs` output is identical between CPU and CUDA paths so operators can `diff` them.
- `web_worker.rs` bridges `generation`, `llama`, `model::{Session, Token}`, and `sampling` for the WASM target.
