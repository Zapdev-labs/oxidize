# Plan: Auto-detect hardware and auto-tune inference for max tok/s

## Goal

When the user runs `oxidize run <model>` (or `oxidize serve`), the
binary should:

1. **Detect** the host hardware (CPU, ISA, RAM, NUMA, GPUs, OS, disk).
2. **Plan** the optimal inference config for that exact machine +
   model — thread count, batch size, context size, KV-cache dtype,
   GPU layer offload, mlock vs mmap, NUMA replication, GEMV backend,
   speculative decoding eligibility, layer cache size, etc.
3. **Apply** the plan (override flags) and **log** it so the user
   can see what was decided and why.
4. **Bypass** cleanly: any explicit flag the user passed wins over
   the auto plan. `--no-auto` disables it entirely.

Target: a single binary that gives an unconfigured user the
"as-good-as-it-gets-on-this-machine" tok/s without them reading the
docs. Explicit tuning still wins, and the user always sees a clear
print of what was chosen.

---

## What already exists (and what we're not re-implementing)

| Capability | Where it lives | What we'll reuse |
|---|---|---|
| GPU detection (`nvidia-smi` → `DetectedGpu`) | `oxidize-core/src/cluster/gpu_cluster.rs:504` | `detect_gpus()` |
| SIMD backend probe (AVX2/AVX-512/NEON) | `oxidize-core/src/compute/simd.rs:34` | `preferred_backend()` |
| Physical-core count + thread-pinning | `oxidize-core/src/compute/spinpool.rs:130` | `physical_core_count()`, `pin_to_slot()` |
| NUMA node count + min-node RAM | `oxidize-core/src/compute/numa.rs:18` | `node_count()`, `min_node_total_bytes()` |
| `linux_mem_available_bytes` | `oxidize-core/src/format/gguf.rs:17` | for KV-cap calc |
| Per-architecture CPU heuristics (AVX-512 use, prefetch distance) | `oxidize-kernels/src/cpu.rs:18` | `tune()` returns `&OxkTune` |
| Memory-mapped GGUF with advise hints | `oxidize-core/src/format/gguf.rs:39` | `MappedGgufFile::advise_*` |
| Inferred KV-cache cap (auto-shrink ctx) | `oxidize-cli/src/main.rs:2258-2280` | the math; we'll generalize it |
| GPU layer offload planning | `oxidize-core/src/model/offload.rs:64` | `plan_layer_offload()` |
| Multi-GPU planning | `oxidize-core/src/model/offload.rs:90` | `plan_multi_gpu_offload()` |
| Paged attention | `oxidize-core/src/paged_attention/` | wired into server via `BatchMode::Paged` |
| Speculative decoding (DFlash + native MTP) | `oxidize-core/src/model/dflash.rs`, `generation.rs` | `--draft-model`, `--no-mtp` flags |
| Continuous batching | `oxidize-server/src/runtime/model.rs` | `ContinuousBatcher` |
| Layer-wise streaming | `oxidize-core/src/model/layer_wise.rs:534` | `LayerWiseModel` |

**The auto-tuner is the orchestrator that ties these together.**
It does not invent new kernels, schedulers, or quantization formats.

---

## Design: a new module `oxidize_core::autotune`

### File: `oxidize-core/src/autotune/mod.rs`

The autotuner is **stateless** — it's a pure function over
(hardware detection, model GGUF) that produces a `TuningPlan`. This
makes it trivially testable (table-driven) and easy to extend.

```rust
pub struct HardwareInventory {
    pub os: OsKind,                       // Linux | Macos | Windows
    pub cpu_vendor: CpuVendor,            // Intel | Amd | Apple | Other
    pub simd: SimdBackend,                // preferred SIMD
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub numa_nodes: usize,
    pub min_node_ram_bytes: u64,
    pub total_ram_bytes: u64,
    pub has_gpu: bool,
    pub gpu_family: Option<GpuFamily>,
    pub gpu_vram_bytes: u64,              // sum across GPUs
    pub has_metal: bool,                  // macOS
    pub has_cuda: bool,                   // libcuda visible
    pub is_wsl: bool,
    pub container_mem_limit: Option<u64>, // cgroup v2 max, if any
    pub hugepages_2mib_avail: bool,
}

pub struct ModelFingerprint {
    pub architecture: String,             // "llama", "qwen2", ...
    pub layer_count: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub file_size_bytes: u64,
    pub quant: GgufQuantizationType,      // most common qtype
    pub is_moe: bool,
    pub expert_count: usize,
}

pub struct TuningPlan {
    pub threads: usize,
    pub ctx_size: usize,
    pub kv_cache_dtype: KvCacheDType,     // F16 | Q8 | Q4 | F32
    pub n_gpu_layers: usize,
    pub gpu_split: Vec<f32>,              // tensor-split per GPU
    pub mmap: bool,
    pub mlock: bool,
    pub mmap_hugepages: bool,
    pub mmap_prefetch: bool,
    pub numa_replicate_dense: bool,       // NUMA-replicate `*weight` ranges
    pub layer_wise: bool,                 // use LayerWiseModel
    pub layer_cache: usize,               // # layers to keep resident
    pub pipeline: PipelineMode,           // Sequential | Continuous | Paged | Asymmetric
    pub speculative: Option<SpeculativeSpec>, // DFlash | Mtp | None
    pub decode_tile_tokens: usize,        // split-K tile size
    pub oxk_isa: OxkIsa,                  // scalar|avx2|avx512|...
    pub oxk_tile: OxkTile,                // 1|4|8|16
    pub expected_prompt_tps: f32,         // estimate for "should you trust this plan" log
    pub expected_decode_tps: f32,
    pub rationale: Vec<String>,           // human-readable decisions
}

pub fn detect() -> HardwareInventory { ... }
pub fn fingerprint(mapped: &MappedGgufFile) -> ModelFingerprint { ... }
pub fn plan(inv: &HardwareInventory, model: &ModelFingerprint) -> TuningPlan { ... }
```

### File: `oxidize-core/src/autotune/detect.rs`

Hardware detection. Pure functions + a few `cfg(target_os)`-gated
probes.

- `cpu_vendor()` / `simd::preferred_backend()` reused from
  `oxidize_core::compute::cpu` (the kernels crate re-exports).
- `physical_cores` / `logical_cores` from
  `oxidize_core::compute::spinpool`.
- `numa_nodes` / `min_node_ram_bytes` from
  `oxidize_core::compute::numa`.
- `total_ram_bytes` from `linux_mem_available_bytes` is the
  available figure; total RAM from `/proc/meminfo` `MemTotal`
  (Linux) or `sysctlbyname("hw.memsize")` (macOS) or
  `GlobalMemoryStatusEx` (Windows).
- `gpu_vram_bytes` from `cluster::gpu_cluster::detect_gpus()`
  summed.
- `has_metal` from `oxidize_core::metal::metal_build_info()`.
- `has_cuda` from `oxidize_core::cuda::cuda_build_info()` + try
  `cuda::initialize_cuda` with ignore-on-error.
- `is_wsl` from `/proc/version` substring "microsoft" or
  `/proc/sys/kernel/osrelease` "Microsoft".
- `container_mem_limit` from `/sys/fs/cgroup/memory.max`
  (cgroup v2) or `/sys/fs/cgroup/memory/memory.limit_in_bytes`
  (v1).
- `hugepages_2mib_avail` from
  `/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages`.

All of these are cheap (single file reads / one nvidia-smi
shellout that we already have). Probe cost < 50 ms on a typical
box.

### File: `oxidize-core/src/autotune/fingerprint.rs`

Reads the GGUF once (already mmap'd by the caller) and extracts
the arch-specific fields from `metadata`. Counts `*_exps` tensors
to detect MoE. Picks the dominant qtype by byte-size histogram
across all weight tensors.

### File: `oxidize-core/src/autotune/rules.rs` — the actual planner

The planner is a **rule table** — ordered, mutually exclusive,
with `rationale` strings attached. Each rule returns
`Option<TuningPlan>` (or a partial plan to be merged).

Order matters. We pick from a curated set of named "profiles"
first, then refine.

#### Tier 0: hard rules (always apply)

1. If `inv.total_ram_bytes < model.file_size_bytes * 1.2` →
   **enable mmap, disable mlock, force layer_wise=true** with
   `layer_cache = max(1, physical_cores / 4)`. Rationale:
   "model is too big for RAM, streaming layers from disk".
2. If MoE + `inv.physical_cores <= 8` → **disable NUMA
   replication** (overhead exceeds benefit).
3. If `inv.os == Macos && inv.has_metal` → **prefer Metal
   backend** (the kernel has a real impl; the build's `metal`
   feature exposes `metal::should_use_mps_gemv`).

#### Tier 1: backend + ISA

4. If `inv.simd == SimdBackend::Avx512f` and not Skylake-SP →
   `oxk_isa = Avx512`, `oxk_tile = 8`.
5. If `inv.simd == SimdBackend::Avx2` →
   `oxk_isa = Avx2`, `oxk_tile = physical_cores >= 16 ? 8 : 4`.
6. Otherwise `oxk_isa = Scalar`, `oxk_tile = 1`.

(Skylake-SP detection reuses the heuristic in
`oxidize-kernels/src/cpu.rs:128` — we'll lift it into a public
helper there.)

#### Tier 2: GPU offload

7. If `inv.has_gpu && model.quant.is_k_quant()`:
   - `n_gpu_layers = floor(gpu_vram_bytes * 0.85 / per_layer_bytes)`
   - `pipeline = Paged` (default)
   - if `inv.gpu_vram_bytes < model.file_size_bytes * 0.25` →
     `n_gpu_layers = 0` (overhead would dominate)
8. If `inv.gpu_vram_bytes >= model.file_size_bytes` →
   `n_gpu_layers = layer_count` (whole model on GPU),
   `mmap = false`, `mlock = false` (the file is fully resident
   so the mlock is redundant).
9. If multi-GPU: `gpu_split = equal_split(inv.gpu_count)` — using
   the same math as `plan_multi_gpu_offload`.

#### Tier 3: KV cache dtype + ctx size

10. If `inv.gpu_vram_bytes >= 16 GiB` → `kv_cache_dtype = F16`
    (lossless at this precision; the existing `KvCacheDType` enum
    already supports it).
11. If `inv.gpu_vram_bytes in [8, 16) GiB` or
    `model.layer_count * ctx >= 64k tokens equivalent` →
    `kv_cache_dtype = Q8` (asymmetric INT8 — already implemented
    in `KvQuantization::Asymmetric`).
12. If `inv.gpu_vram_bytes < 8 GiB` or `model.layer_count >= 80` →
    `kv_cache_dtype = Q4` (TurboQuant — already implemented).
13. Context cap: `ctx_size = min(model_default_ctx, kv_budget / kv_bytes_per_token)`
    where `kv_budget = total_ram * 0.6` (the existing
    `optimize_mapped_model_memory` code uses a different factor;
    we keep the existing factor for that path and use 0.6 here,
    since the auto-tuner is allowed to be a bit more aggressive
    when deciding than the conservative runtime cap).

#### Tier 4: layer cache + NUMA

14. If `inv.numa_nodes >= 2 && physical_cores >= 16 &&
    !model.is_moe`:
    `numa_replicate_dense = true` (the existing
    `OXIDIZE_NUMA_REPLICATE=dense` behavior).
15. `layer_cache = clamp(physical_cores, 2, 8)`. Rationale: 1
    layer per ~2 cores for steady-state decode. Capped at 8
    because beyond 8 the LRU working set stops being a win (cf.
    FlexGen's zigzag block schedule).

#### Tier 5: speculative

16. If `inv.has_gpu` and the model is in a known DFlash-supported
    list (Qwen3, Llama-3.x) → `speculative = Some(Mtp)` and
    `pipeline = Paged` (the native MTP path needs the paged
    runtime).
17. If the user has set `OXIDIZE_DRAFT_MODEL` env → prefer that
    over auto-suggest.

#### Tier 6: thread count

18. `threads = physical_cores` for pure CPU decode.
19. If `inv.has_gpu && n_gpu_layers == layer_count` →
    `threads = 4` (CPU is only doing scheduling + sampling;
    over-subscribing CPU hurts).
20. If `inv.container_mem_limit.is_some()` →
    `threads = clamp(physical_cores, 2, 8)` (containers often
    share a host; over-pinning makes the scheduler sad).

#### Tier 7: decode tile (split-K attention)

21. If `ctx_size > 4096` AND `inv.simd == Avx2` →
    `decode_tile_tokens = 512`.
22. Else if `ctx_size > 8192` →
    `decode_tile_tokens = 1024`.
23. Else `decode_tile_tokens = 0` (split-K off; existing path).

(Heuristic from the FlashDecoding paper: split-K only pays off
above ~1024 KV tokens for SIMD/AVX2; on AVX-512 or GPU we never
need it because per-head parallelism is already high.)

#### Tier 8: paged vs continuous vs sequential

24. If the model is being served (`serve_api` flag) →
    `pipeline = Paged`.
25. If `inv.has_gpu` → `pipeline = Paged` (continuous batching
    + paged attention are gated on a GPU because CPU paged
    attention has no kernel yet — though we're about to add
    that).
26. If `inv.physical_cores >= 8 && inv.total_ram_bytes >= 64
    GiB` → `pipeline = Continuous`.
27. Otherwise `pipeline = Sequential`.

#### Estimates

For `expected_decode_tps` and `expected_prompt_tps`, we use a
heuristic derived from the FlexGen/NEO cost models:

```
decode_tps = min(
    model.file_size_bytes / (inv.gpu_vram_bytes.max(inv.total_ram_bytes) * 0.7),
    physical_cores * per_core_decode_tps(model)
)
```

`per_core_decode_tps(model)` is a simple lookup table calibrated
against the existing `results/bench/`:

| model.quant | per-core decode t/s (DDR4-3200) |
|---|---|
| Q4_K_M (small, ≤8B) | 1.2 |
| Q4_K_M (medium, 8–30B) | 0.6 |
| Q4_K_M (large, ≥30B) | 0.25 |
| Q2_K (medium) | 1.4 |
| Q2_K (large) | 0.5 |
| F16 (any) | 0.4 |
| Q8_0 (any) | 0.8 |

GPU families get a multiplier: A100 4×, H100 6×, RTX Pro 6000
4×, B200 10×. (These are crude — the goal is "is the plan
self-consistent?" not "is it perfect?")

The estimate is only used to print a confidence-style line in the
rationale ("expected ≈ 8.4 t/s decode on this box"); if real perf
differs by >2× the user has something to investigate.

---

## CLI integration

### New flag surface (`oxidize run`, `oxidize serve`)

- `--auto` (default `true` for `run`, `false` for `serve`):
  enable auto-tuning.
- `--no-auto`: explicit opt-out.
- `--print-plan` (default `true` when `--auto` and stdout is a
  tty): print the `TuningPlan` summary before generation starts.
  Output format is plain text, one `key: value` per line, with
  `rationale` indented under each decision. JSON output via
  `--print-plan=json` for tooling.
- `--auto-profile <name>`: pin to a specific named profile
  (`desktop-llama-3-8b`, `server-llama-3-70b`,
  `h100-qwen2-72b`, `macbook-air-qwen3-4b`, etc.). Each profile
  is a pre-computed `TuningPlan` template the user can copy from
  `--print-plan=json` after a good run.

### Resolution order in `oxidize run <model>`

For every flag the autotuner would set:

1. CLI flag (e.g. `--threads 16`) — wins.
2. Env var (e.g. `OXIDIZE_THREADS=16`) — wins.
3. Auto-plan — applied.
4. Hard-coded default — applied.

This is the "explicit beats implicit" rule the existing
`physical_core_count()` fallback at `main.rs:2037` already
follows. The autotuner just extends that pattern to *all* the
relevant flags, with a `rationale` for each.

### Where the autotuner runs

In `main()` of `oxidize-cli/src/main.rs`, between line 2148
(where `model_path` is detected) and line 2164 (where
`plan_layer_offload` runs):

```rust
let inv = oxidize_core::autotune::detect();
let mapped = loader.load(&model_path)?;
let model = oxidize_core::autotune::fingerprint(&mapped);
let mut plan = if args.auto { Some(oxidize_core::autotune::plan(&inv, &model)) } else { None };
if let Some(plan) = plan.as_ref() {
    eprintln!("oxidize auto-tune plan:\n{}", plan.summary());
    apply_plan(args, &mut config, &inv, plan);  // mutates args + config
}
// ... existing layer_offload / model build follows
```

`apply_plan` is a small function that fills in any `args.*` /
`config.*` field that the user didn't already set.

### Server

`oxidize-server/src/cli.rs` gets the same flags. The server
defaults `--auto=true` (you almost always want it). The same
`apply_plan` is called.

---

## What we'll build (file list)

1. `oxidize-core/src/autotune/mod.rs` — module root, re-exports.
2. `oxidize-core/src/autotune/detect.rs` — `HardwareInventory`,
   `detect()`.
3. `oxidize-core/src/autotune/fingerprint.rs` — `ModelFingerprint`,
   `fingerprint()`.
4. `oxidize-core/src/autotune/rules.rs` — `TuningPlan`, `plan()`,
   the rule table.
5. `oxidize-core/src/autotune/apply.rs` — `apply_plan(args, config, plan)`
   helpers used by the CLI and the server. Lives here so it's
   testable independent of clap.
6. `oxidize-core/src/lib.rs` — register the module.
7. `oxidize-kernels/src/cpu.rs` — lift the Skylake-SP detection
   into a `pub fn is_skylake_sp() -> bool` so the autotuner can
   reuse it.
8. `oxidize-cli/src/main.rs` — wire `--auto`, `--no-auto`,
   `--print-plan`, `--auto-profile`; call `detect` → `fingerprint`
   → `plan` → `apply_plan`; print summary.
9. `oxidize-server/src/cli.rs` — same flags.
10. `scripts/auto_tune_report.sh` — a small shell script that
    runs `oxidize run` on a few model sizes, parses
    `--print-plan=json`, and emits a Markdown table of the plans
    for documentation. Used in the AGENTS.md.
11. `AGENTS.md` — new "WHERE TO LOOK" row for autotune.

---

## Test plan

### Unit tests (table-driven)

For each (hardware, model) pair, the planner must produce a
deterministic `TuningPlan` with `rationale` populated. The
fixtures live in `oxidize-core/src/autotune/tests_fixtures.rs` and
cover:

| Fixture | Hardware | Model | Expected plan highlight |
|---|---|---|---|
| `desktop_no_gpu` | 16c/32T, 64 GiB, no GPU | Qwen3-4B Q4_K_M | n_gpu_layers=0, ctx=4096, kv=f16 |
| `desktop_big_model` | 16c/32T, 64 GiB, no GPU | Gemma4 31B Q2_K | layer_wise=true, layer_cache=4, mmap=true |
| `workstation_a100` | 32c/128T, 256 GiB, 1×A100 80G | Qwen3-32B Q4_K_M | n_gpu_layers=all, mmap=false, paged |
| `server_2xh100` | 64c/256T, 1 TiB, 2×H100 | Llama-3-70B Q4_K_M | n_gpu_layers=all, multi-gpu split, continuous batching |
| `macbook_air` | 8c Apple Silicon, 16 GiB unified | Qwen3-4B Q4_K_M | metal backend, kv=q4, ctx=2048 |
| `wsl_laptop` | 8c/16T, 16 GiB, no GPU, WSL | Llama-3-8B Q4_K_M | layer_wise=true, mlock=false (cgroup), kv=q4 |
| `tiny_box` | 4c/8T, 8 GiB, no GPU | Qwen3-0.5B Q8_0 | layer_wise=false (model fits), ctx=2048 |

The rules-as-data design makes it trivial to add a new fixture
when a user reports a bad plan on their hardware.

### Integration test (smoke)

`scripts/auto_tune_report.sh` runs `oxidize run --no-api
--auto --print-plan=json --max-tokens 1` on the existing
Qwen3-4B Q4_K_M fixture and verifies the plan includes
`n_gpu_layers`, `kv_cache_dtype`, and at least one `rationale`
entry per set field. No actual model loading — uses the GGUF
header only.

### End-to-end on the K3 cluster

`scripts/auto_tune_report.sh --node ai-2` (CPU-only) and
`--node ai@192.168.1.68` (CPU-only) prints a side-by-side plan
for each. Output goes to
`results/bench/auto_tune_ai2_<date>.txt` and
`results/bench/auto_tune_ai_<date>.txt` for the AGENTS.md
"autotune evidence" section.

---

## What this is *not*

- **Not** a new GEMV kernel. We pick among the existing
  `oxk_isa` / `oxk_tile` values. The kernel crate's `tune()`
  already does ISA-level tuning.
- **Not** a new scheduler. The pipeline pick is from
  `{Sequential, Continuous, Paged, Asymmetric}` which the server
  already supports.
- **Not** a new quantization path. We pick from the existing
  `KvCacheDType` enum and the existing `KvQuantization` enum.
- **Not** a new speculative decoder. We pick from
  `{None, DFlash, Mtp}`.
- **Not** a new core abstraction. The autotuner is a pure
  function over the existing detection helpers, producing a plan
  that the existing CLI / server consume via small `apply_*`
  helpers.

The constraint: **the autotuner must not require a new
`ComputeBackend` trait, a new runtime, or a new public type**,
because the user's preference is "extend what exists". All the
detection primitives we need are already in the workspace.

---

## Rollout (3 steps, each one ships)

1. **Detection only**: ship `HardwareInventory` +
   `ModelFingerprint` + a `--print-hardware` subcommand that just
   prints them. No changes to inference behavior. Lets us
   validate the detection on real K3 nodes before we trust it.
2. **Planner + apply**: add `TuningPlan` + `plan()` +
   `apply_plan()` and the `--auto` flag in CLI and server.
   Default `--auto=true` for `run`; the user can opt out. The
   `print-plan` summary is on by default. Stage 1 is unchanged.
3. **Profiles + benchmarks**: ship
   `scripts/auto_tune_report.sh`, gather plans on the K3 nodes,
   write up the results in `AGENTS.md`. Optional
   `~/.config/oxidize/auto-profile.json` file that lets the
   user pin a profile by name.

Each step ends with `make build && make test && make lint` green,
and a fresh entry in `results/bench/auto_tune_*.txt`.

---

## Summary of changes

- New module `oxidize-core/src/autotune/` (~600 lines + tests).
- New public functions on `oxidize-kernels::cpu`:
  `pub fn is_skylake_sp() -> bool`.
- CLI: ~120 new lines in `oxidize-cli/src/main.rs` for the new
  flags + the `apply_plan` call.
- Server: ~30 new lines in `oxidize-server/src/cli.rs`.
- `scripts/auto_tune_report.sh` (~80 lines).
- AGENTS.md update.
- All existing tests must continue to pass; the new module ships
  with at least 12 unit tests covering the table above.

Net: 1 new module + 1 small function lift + CLI/server plumbing +
scripts. No new runtime, no new kernel, no new public type.
