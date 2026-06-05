# Perf & Features: VNNI matmul, SafeTensors→GGUF, Continuous batching

**Date:** 2026-06-05
**Status:** Approved, implementing

## Scope
Three independently shippable units in `oxidize-core`/`oxidize-server`, then ported to
`oxidize-golang` and `oxidize-python` where applicable. No Rust crate restructuring.

## Component 1 — AVX-512 VNNI quantized matmul (Speed)
- **File:** `oxidize-core/src/compute/tensor.rs`
- **Target kernel:** `q4_k_q8_k_row_dot_avx2` (int8 domain Q4_K×Q8_K row dot).
- **Change:** Add `q4_k_q8_k_row_dot_vnni` using `_mm512_dpbusd_epi32`. Each `gp`
  iteration concatenates groups g1|g2 (64 q4 bytes, 64 q8 bytes) into 512-bit
  registers; one `dpbusd` yields 16 int32 lanes (lanes 0..8 = g1 scaled by s1,
  lanes 8..16 = g2 scaled by s2). Replaces the `maddubs`+`madd(ones)` pair.
- **Dispatch:** `q4_k_q8_k_vnni_available()` = `avx512f && avx512bw && avx512vnni`,
  checked before the existing avx2 branch in both call sites (line ~1217 fused
  path and the gemv path). AVX2 remains the fallback.
- **Ports:** Go/Python have no hand-written SIMD; this component is Rust-only.
  Go uses compiler autovectorization; Python is scalar/numpy. Mark N/A in ports.
- **Validation:** Unit test asserts VNNI output == AVX2 output (exact i32 acc, f32
  within 1e-4) on random Q4_K/Q8_K blocks. Criterion bench before/after.

## Component 2 — Finish SafeTensors→GGUF auto-conversion (Feature)
- **Files:** `oxidize-core/src/format/safetensors_to_gguf.rs`, `conversion.rs`, `gguf.rs`.
- **Requirements (per memory):** tokenizer embedding emitted, BF16→F32/F16 dequant,
  QKV bias tensors, `*.attention.key_length` metadata, skip LoRA adapter tensors.
- **Wiring:** `oxidize run <hf-model>` auto-converts when given SafeTensors input.
- **Ports:** Mirror into `oxidize-golang` (conversion package) and `oxidize-python`.
- **Validation:** Round-trip test — convert fixture, load via normal loader, one forward pass.

## Component 3 — Continuous batching (Feature/Speed) — DEFERRED
**Finding (2026-06-05):** Not implementable at the server layer. `Model::forward`
drives a single contiguous KV cache; `forward_many` is sequential single-sequence
forwards; the paged scheduler only does block accounting, not real paged KV in the
forward pass. Genuine batching needs a paged KV cache threaded through every
architecture's attention in `inference.rs` (~3456 lines) + backends, plus a batched
multi-sequence decode step. Large, high-risk prerequisite — deferred, not faked.
See memory `continuous-batching-blocked-on-paged-kv`.

### Original plan (pending prerequisite)
- **Files:** `oxidize-server/src/runtime/`, paged-attention `scheduler.rs`.
- **Change:** Decode-step request merging so concurrent generations share a forward
  pass; admission + step-merge loop threaded through the OpenAI streaming handler.
- **Ports:** Mirror server runtime change into Go (`internal/`) and Python server.
- **Validation:** Integration test, 4 concurrent completions, correct independent
  outputs + throughput assertion vs serial.

## Order
1 → 2 → 3, tests/benches committed per step. Ports follow each Rust component.
