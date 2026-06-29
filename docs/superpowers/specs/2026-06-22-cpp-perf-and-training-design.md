# oxidize-cpp: Max-speed + Training design

**Date:** 2026-06-22 · **Branch:** cpp-port · **Target box:** ai@192.168.1.68 (pw `machine`)

## Target hardware (probed)
- Dual-socket Intel Xeon Silver 4110 (Skylake-SP). 16 physical / 32 logical cores. **2 NUMA nodes** (node distance 10 local / 21 remote → remote RAM ~2x slower). CPU numbering interleaved: node0 = even cpus, node1 = odd.
- 123 GB RAM (≈63 GB per node). 3.4 TB disk (2.9 TB free).
- AVX-512 (f/dq/cd/bw/vl) but **no VNNI, no AVX512-BF16**. Skylake-SP down-clocks under AVX-512 → must measure, gate at runtime.
- **No GPU.** CUDA path is irrelevant for this box.

## Models on box
- `~/models/glm-5.2/target/UD-IQ1_M/` — GLM-5.2, 6-shard GGUF, **213 GB**, unsloth IQ1_M (~1.5-bit). MoE. EAGLE3 draft at `~/models/glm-5.2/eagle3/draft/model.safetensors` (2.3 GB).
- `~/models/qwen2.5-0.5b-instruct-q4_0.gguf` — small.
- `~/rl-finetune/rl-all-sft.jsonl` — 55 MB chat-format SFT dataset.

## Hard constraints
1. **213 GB model > 123 GB RAM.** Only feasible because MoE (few active experts/token). Strategy: mmap + page-cache-resident hot experts + NUMA placement. Cold-expert disk stalls are inherent.
2. **IQ1_M not supported in C++** (only Q4_0/5_0/8_0/Q4_K/Q5_K). Must implement IQ-quant dequant from scratch — gating item for GLM.
3. **GLM-5.2 arch (MLA?)** — C++ currently `throw`s on MLA. Architecture port needed (MoE FFN exists).
4. **Full FT of GLM on CPU is not realistic.** Full-FT scoped to small models only; GLM → LoRA best-effort.

## Workstreams
- **Phase 0 — Deep research (parallel):** IQ1_M format+dequant; GLM-5.2 exact arch from GGUF metadata; EAGLE3 mechanics; CPU LoRA/full backprop design; NUMA inference best practices. Each returns a spec, verified before build.
- **A — CPU/NUMA speed:** NUMA-aware alloc + first-touch + thread pinning per socket; AVX-512-vs-AVX2 measurement w/ runtime gate; mmap + huge pages; per-socket replication for small models.
- **B — GLM-5.2 enablement:** IQ1_M dequant → GLM/MLA arch → mmap MoE streaming → EAGLE3 speculative decode. Sequential, research-gated.
- **C — Training:** autograd-lite (only needed ops) → AdamW → LoRA → full-FT (small models) → SFT loop over rl-all-sft.jsonl. Verify by overfitting a tiny batch (loss→0) + forward parity.
- **D — Integration/benchmarks:** real before/after tok/s on the box (Qwen-0.5B + GLM-5.2); token-exact correctness gates; single PR to cpp-port.

## Verification
- Existing `ctest` parity suite must stay green; add tests per feature.
- Real benchmarks on the box (not the dev laptop). Correctness = token-exact vs current C++ baseline.
- One PR on `cpp-port` (standing user preference). Code reaches box via rsync from local repo.

## Risk register
- IQ1_M dequant correctness (codebook subtlety) — HIGH.
- GLM MLA arch port — HIGH.
- GLM decode speed bottlenecked by disk for cold experts — MEDIUM (mitigate w/ page cache warmth + interleave).
- AVX-512 net-negative on Skylake — MEDIUM (gate at runtime).
- Full-FT memory blowup — handled by scoping to small models.
