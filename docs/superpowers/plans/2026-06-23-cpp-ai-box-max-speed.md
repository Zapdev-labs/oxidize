# oxidize-cpp: Max-speed on ai@192.168.1.68

**Date:** 2026-06-23 · **Branch:** cpp-port · **Target:** ai@192.168.1.68 (pw `machine`)

## Global Constraints

- Target box: dual-socket Xeon Silver 4110, 16 physical / 32 logical cores, **2 NUMA nodes**, ~123 GB RAM, **no GPU**. AVX-512 present but Skylake-SP may down-clock — measure before enabling AVX-512 kernels at runtime.
- Models: GLM-5.2 (213 GB IQ1_M MoE, 6 shards), Qwen2.5-0.5B Q4_0, MiniMax-M3 (IQ4_XS, prune target).
- 213 GB model > 123 GB RAM → MoE + mmap + page cache + NUMA interleave required. Cold expert disk stalls are inherent.
- All changes on `cpp-port`. Existing `ctest` parity suite must stay green. Token-exact correctness vs baseline.
- Code reaches box via rsync; benchmarks run **on the box**, not the dev laptop.
- Do not start dev servers unless benchmarking requires it.

---

## Task 1: ai-box deploy + benchmark harness

Create `scripts/ai-box-bench.sh` that:

1. Accepts `--host ai@192.168.1.68` (default), `--password` via `SSHPASS` env (never hardcode).
2. Rsyncs `oxidize-cpp/` to `~/oxidize-cpp-build/` on the remote (exclude `build/`).
3. Remote build: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`.
4. Runs a benchmark matrix on `~/models/qwen2.5-0.5b-instruct-q4_0.gguf`:
   - Baseline: `--threads 32`
   - NUMA single: `--numa single --threads 16`
   - NUMA interleave: `--numa interleave --threads 32`
5. Emits JSON lines (`--json`) to `bench-results-$(date +%Y%m%d-%H%M%S).jsonl`.
6. Prints a summary table (prefill_tps, decode_tps, device).

**Acceptance:** Script runs end-to-end when the box is reachable; exits non-zero on SSH/build/bench failure.

---

## Task 2: IQ1 block-wise fused GEMV (decode-once per block)

Port Rust `gemv_iq1_m_f32_fused` / `gemv_iq1_s_f32_fused` pattern to C++:

1. In `tensor_cpu.cpp`, add `dot_iq1_s` and `dot_iq1_m` that dequantize one `QK_K` block into a stack scratch `[QK_K]` then `dot_f32` against the matching `x` slice — **not** full-row dequant.
2. Wire `QuantType::IQ1_S` and `QuantType::IQ1_M` fast paths in `gemv_quantized` before the slow per-row dequant fallback.
3. Add parity test in `tests/parity_test.cpp`: random IQ1_M block vs `dequantize_row` + `dot_f32`.

**Acceptance:** `ctest` green; IQ1 fast path produces bit-identical sums to slow path on test vectors.

---

## Task 3: mmap huge pages (2 MiB)

In `GgufModel::load` (`gguf.cpp`):

1. After successful `mmap`, if `/proc/meminfo` reports `Hugepagesize: 2048 kB` and `HugePages_Free > 0`, call `madvise(map, size, MADV_HUGEPAGE)`.
2. Add CLI flag `--hugepages` to `oxidize-cpp` that sets an env var or load option (default off; autotune can enable later).
3. Document in script help text only (no new .md file).

**Acceptance:** Compiles; when huge pages unavailable, silently no-ops. Unit test not required (kernel-dependent).

---

## Task 4: C++ autotune for Skylake-SP NUMA box

Add minimal autotune to `oxidize-cpp` CLI (not server yet):

1. `--auto` flag: detect NUMA nodes via existing `discover_numa_nodes()`, physical cores, model file size via `stat`.
2. Rules (mirror Rust `oxidize-core/src/autotune/plan.rs` for CPU-only):
   - `model_size > 0.8 * total_ram` → `--numa interleave`, threads = logical cores
   - else dual NUMA → `--numa single`, threads = physical cores on node 0
   - Q4_K / Q8_0 / F16 weights → no extra flags
3. `--print-plan` prints JSON plan without running inference.
4. `--no-auto` disables (default off for now to preserve backward compat).

**Acceptance:** `--print-plan --model <gguf> --auto` prints sensible JSON on any host; existing tests green.

---

## Task 5: Model prep playbook script (prune + quantize)

Create `scripts/prepare-fast-model.sh`:

1. Wraps `oxidize-prune` and `oxidize-quantize` for the common path: magnitude prune 50% + joint Q4_K_M.
2. Documents SnapPrune flash path for MoE (MiniMax-M3) as comments in script header.
3. Example usage for Qwen and generic dense GGUF.

**Acceptance:** Script is valid bash; dry-run mode prints commands without executing.

---

## Task 6: Integration benchmark on ai box

After Tasks 1–4 land, run `scripts/ai-box-bench.sh` on the box and record:

- Qwen 0.5B before/after autotune
- GLM-5.2 shard-0 smoke (if arch loads) or document blocker

**Acceptance:** Results appended to `.superpowers/sdd/bench-results.md` on the box run.
