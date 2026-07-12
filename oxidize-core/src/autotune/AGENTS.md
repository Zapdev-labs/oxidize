# oxidize-core/src/autotune

**Domain:** Hardware detection + GGUF fingerprinting + inference tuning plan

## OVERVIEW
Detects host hardware, fingerprints a GGUF model, and produces a fully-resolved `TuningPlan` (threads, ctx, KV dtype, GPU offload, pipeline mode, ISA, speculative, etc.) that the CLI/server apply only for flags the user didn't set. Backing logic for `--auto`, `--no-auto`, `--print-plan` across Rust/Go/Python/C++ ports. Design doc: `plans/auto-detect-and-tune-inference.md`.

## STRUCTURE
```
autotune/
├── mod.rs           # module root + re-exports
├── detect.rs        # hardware probes → HardwareInventory
├── fingerprint.rs   # GGUF → ModelFingerprint + memory-size helpers
├── rules.rs         # tiered pure-function planner → TuningPlan
└── apply.rs         # TuningPlan → PlanOverrides for CLI/server
```

## KEY TYPES
| Type | Role |
|------|------|
| `HardwareInventory` | OS, CPU vendor, SIMD backend, cores, NUMA nodes + per-node RAM, GPU family/VRAM, Metal/CUDA/ROCm/RDMA, WSL, cgroup mem, hugepages |
| `ModelFingerprint` | architecture, layer/head/kv/hidden/intermediate/vocab sizes, file size, dominant quant, MoE/MTP flags |
| `TuningPlan` | Fully-resolved plan; every decision recorded in `plan.rationale` |
| `PlanOverrides` | Per-flag `Option`s so CLI/server apply "explicit beats implicit" |
| `PipelineMode` | Sequential / Continuous / Paged / Asymmetric |
| `SpeculativeSpec` | None / DFlash / Mtp |
| `OxkIsa` / `OxkTile` | Scalar / Avx2 / Avx512; T1/T4/T8/T16 |

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add hardware probe | `detect.rs` | Keep cheap (<50 ms) |
| Add model fingerprint field | `fingerprint.rs` | `fingerprint(&MappedGgufFile)` |
| Add tuning rule | `rules.rs` | Pure function; tiers 0–8; no I/O |
| Wire to CLI/server | `apply.rs` | `overrides_from_plan(&TuningPlan)` |
| Skylake-SP AVX-512 opt-out | `rules.rs` | Uses `oxidize_kernels::cpu::is_skylake_sp()` |

## NOTES
- `plan()` is a **pure function** — no I/O, no clocks.
- Heavily cross-cutting: pulls from `gpu_cluster`, `numa`, `simd`, `spinpool`, `oxidize_kernels`, `gguf`, `inference::InferenceConfig`, `kv_cache::KvQuantization`, backend build-info.
- NUMA rules (dual-socket): dense models ≤192 GB → `--numa single --threads 16`; >192 GB → `--numa interleave --threads 48`.
