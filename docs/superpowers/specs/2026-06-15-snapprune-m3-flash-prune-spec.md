# Spec: Accelerate MiniMax-M3 via SnapPrune Flash-Prune → Q4_K_M GGUF

**Date:** 2026-06-15
**Status:** Draft
**Owner:** oxidize / M3 perf
**Target host:** `ai@192.168.1.68` (dual-socket Xeon Silver 4110, 32 logical cores, 310 GB RAM, 2 NUMA nodes, no GPU)

---

## 1. Problem

MiniMax-M3 (427B total / ~26B active VL-MoE) runs correctly on oxidize but is impractically slow on CPU: **~0.20 tok/s (~5 s/token)** measured on the merged IQ4_XS GGUF, even after NUMA tuning (`numactl --interleave=all` + 32 threads, which only bought ~13% over the unpinned baseline).

Root cause: the IQ3_S/IQ4_XS expert weights run through oxidize's **scalar dequant-and-dot** path. oxidize has *fused* AVX2 integer kernels for Q4_K/Q6_K (`gemv_q4_k_q8_k_fused`) but **not** for IQ types, so every token re-dequantizes ~26B active params to f32 and does float dot-products. Runtime knobs (NUMA, threads, page-cache) are exhausted.

## 2. Goal

Produce a **smaller, faster M3** that runs on oxidize's fused Q4_K path, by:
1. **Pruning** a fraction of the 128 experts per layer (reduces total size / RAM pressure), and
2. **Requantizing** the pruned weights to **Q4_K_M** (moves decode onto the fused AVX2 kernel),

in a **single SnapPrune pass**, then benchmarking the result in oxidize.

### Success metric
- **Primary:** M3 decode throughput **≥ 3× the 0.20 tok/s baseline (≥ 0.6 tok/s)**, measured the same way (32-token completion, warm cache, `--interleave=all`, 32 threads).
- **Secondary:** output remains coherent on a fixed smoke set (e.g. "The capital of France is" → "Paris"; a 3-sentence prose prompt produces grammatical text).
- **Footprint:** pruned Q4_K_M GGUF materially smaller than the 207 GB IQ4_XS GGUF.

## 3. Background: what SnapPrune provides

Source: `Zapdev-labs/snapprune`, `python/snapprune/{cli,flash,gguf,model,config}.py`.

Three modes (all accept `--gguf --quant Q4_K_M` to emit a quantized GGUF directly):

| Mode  | Cost    | Expert saliency                         | Calibration                          |
|-------|---------|-----------------------------------------|--------------------------------------|
| flash | seconds | router-bias magnitude (weight-only)     | none                                 |
| swift | minutes | weight-norm × router-bias               | 128 **simulated** samples            |
| deep  | hours   | simulated REAP                          | 1024 **simulated** (hash-based) gates |

Key properties confirmed from source:
- **Streams layer-by-layer** via `model.safetensors.index.json` (loads/writes one shard at a time) → the 854 GB BF16 model prunes within 310 GB RAM. **No whole-model load.**
- **Prune + requantize in one command** (`--gguf --quant Q4_K_M`).
- **No real calibration corpus is consumed** — even `deep` uses simulated/hash-based gate values, not real activations. Therefore supplying external calibration data (e.g. the oxidize repo) would **not** change results.
- Arch detection is **tensor-name-pattern based**, currently covering **Mixtral, DeepSeek MoE, Qwen MoE**, and dense variants. **MiniMax-M3 is not yet recognized.**

### Mode decision
Use **`flash`**. Rationale: it is data-free and fast, and because `deep`'s "calibration" is simulated anyway, the slower modes offer no real quality advantage here. `swift` is an optional fallback if `flash` quality is unacceptable.

## 4. Scope

### In scope
1. Add **MiniMax-M3 architecture detection** to SnapPrune (expert/router tensor-name patterns).
2. Run **flash prune** on `~/models/MiniMax-M3-bf16` → pruned model + **Q4_K_M GGUF**.
3. Validate the GGUF loads and generates coherently in oxidize.
4. Benchmark decode TPS and compare to the 0.20 tok/s baseline.
5. Record results and the M3-detection patch.

### Out of scope (separate tracks)
- Fused IQ4_XS/IQ3_S AVX-512 kernels in oxidize.
- EAGLE3 speculative decoding (`Inferact/MiniMax-M3-EAGLE3`) — stacks *after* this, separately specced.
- Tile-based GPU inference (already landed for the CUDA path; CPU-irrelevant here).
- True activation-based REAP / real calibration data.
- MiniMax Sparse Attention (only matters at long context).

## 5. Requirements

### R1 — M3 architecture support in SnapPrune
SnapPrune must recognize M3's MoE structure from the BF16 checkpoint:
- Config: `model_type` is `minimax_m3_vl`; MoE params may be nested under `text_config` (`num_local_experts`, `num_experts_per_tok`, leading-dense-layer count).
- Expert tensors named `language_model.…block_sparse_moe.experts.{E}.w{1,2,3}` (gate/up/down).
- Router bias tensor `e_score_correction_bias` (sigmoid-gated routing with bias).
- Must correctly enumerate **per-layer expert count (128)**, skip the **3 leading dense layers**, and leave the **shared expert** intact (prune only routed experts).
- Detection must not misclassify or corrupt non-expert tensors (attention, norms, embeddings, lm_head, vision tower if present).

### R2 — Flash prune execution
- Input: `~/models/MiniMax-M3-bf16` (59-shard BF16, index present).
- Command shape:
  ```bash
  python -m snapprune flash ~/models/MiniMax-M3-bf16 \
    -o ~/models/MiniMax-M3-pruned -r 0.5 --gguf --quant Q4_K_M
  ```
- `-r 0.5` = drop ~50% of routed experts per layer by router-bias saliency. If quality fails (R4), re-run at `-r 0.25`.
- Output: pruned safetensors **and** a single Q4_K_M GGUF (or split set; if split, merge with the existing `~/merge_gguf.py`, since oxidize lacks a split-GGUF loader).

### R3 — Disk / memory budget
- Box has ~1.1 TB free. BF16 input 854 GB (read-only). Pruned Q4_K_M GGUF est. < 120 GB. Pruned intermediate safetensors must not co-exist at full BF16 size — verify SnapPrune writes pruned (smaller) shards, not full copies. Abort if projected usage exceeds free disk.
- Pruning must stay within 310 GB RAM (layer-by-layer streaming; verify peak RSS during a dry first layer).

### R4 — Correctness / quality gate
- Pruned GGUF loads in oxidize with the M3 arch path (no tensor-count/shape errors).
- Smoke prompts produce coherent output (factual recall + grammatical prose). A pruned model that emits garbage at `-r 0.5` → retry `-r 0.25`; if still broken, fall back to `swift`.

### R5 — Performance validation
- Benchmark identically to the baseline: warm cache, `numactl --interleave=all`, `--threads 32`, `--layer-wise --cpu-optimized --kv-cache-dtype q8`, 32-token completion, report tok/s.
- Record: model size, expert count/layer before/after, tok/s before/after, output samples.

## 6. Implementation plan

1. **Clone + inspect** `Zapdev-labs/snapprune` on the ai box; read `flash.py`/`model.py` arch-detection to find the extension point.
2. **Add M3 detection** (R1): a tensor-name/`config.json` matcher for `minimax_m3_vl` mirroring the Qwen/DeepSeek MoE handlers; unit-check expert enumeration on M3's `index.json` (names only, no payload load).
3. **Dry-run guard:** prune layer 3 (first MoE layer) only / `--ratio` smoke, confirm peak RSS < 310 GB and pruned shard sizes shrink (R3).
4. **Full flash prune** → Q4_K_M GGUF (R2). Merge if split.
5. **Load + smoke** in oxidize (R4).
6. **Benchmark** TPS vs baseline (R5); if quality fails, drop ratio and repeat.
7. **Record** results + patch in project memory; update task #9.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Flash (router-bias-only) pruning degrades quality at `-r 0.5` | Fall back to `-r 0.25`, then `swift`. Quality gate R4 catches it before benchmarking. |
| M3 tensor naming differs from assumption / vision tower interferes | Verify against actual `index.json` before coding; prune only routed-expert tensors, pass everything else through untouched. |
| Box thrashes/OOMs during prune (happened during NUMA test) | Stop the running M3 server first to free RAM; dry-run RSS check (R3) before the full pass. |
| SnapPrune writes full-size intermediates → disk blowout | Verify incremental pruned-shard writes on the dry run; abort on projected overflow. |
| SnapPrune GGUF writer doesn't support M3 / Q4_K_M expert layout | Fall back: prune to safetensors, then convert with oxidize's existing `safetensors_to_gguf` (M3 arch already supported). |
| Pruned expert count breaks oxidize's M3 router (expects 128) | oxidize must read expert count from GGUF metadata, not hardcode 128 — verify/adjust the M3 loader. |

## 8. Acceptance criteria

- [ ] SnapPrune recognizes and prunes M3 routed experts (3 leading dense layers + shared expert preserved).
- [ ] Flash prune completes within RAM/disk budget, emits a loadable Q4_K_M GGUF.
- [ ] Pruned model generates coherent output on the smoke set in oxidize.
- [ ] Decode throughput **≥ 0.6 tok/s** (≥ 3× baseline), measured under the standard harness.
- [ ] Results + M3-detection patch recorded; follow-on EAGLE3 stacking noted.

## 9. Open questions

1. Does SnapPrune's GGUF writer emit M3-compatible MoE tensor names/metadata, or must we route through oxidize's `safetensors_to_gguf`?
2. Does oxidize's M3 loader read per-layer expert count from metadata, or assume 128? (Determines whether a pruned model loads without a code change.)
3. Acceptable quality floor for the use case (general vs code) — sets the max safe prune ratio.
