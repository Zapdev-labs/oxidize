# Oxidize Compute Kernels & Performance Analysis

**Date:** 2026-05-20  
**Branch:** `deepflash-safetensors-perf`  
**Scope:** `oxidize-core/src/compute/*`, `oxidize-core/src/model/inference.rs`, `oxidize-core/src/model/layer_wise.rs`, `oxidize-core/src/lib.rs`, `oxidize-core/Cargo.toml`, `oxidize-core/build.rs`

---

## Executive Summary

The oxidize codebase is a Rust-based LLM inference engine with a functional CPU-centric design, nascent GPU backends (CUDA/Metal/WebGPU stubs), and support for GGUF quantization formats. While the architecture is clean and modular, the compute kernels are **not yet competitive with state-of-the-art inference engines** (llama.cpp, vLLM, TensorRT-LLM, candle). The primary gaps are:

1. **No AVX-512 or NEON kernels** — only AVX2 is implemented in hot paths.
2. **No operator fusion** beyond basic SwiGLU — attention, norm, and activation are separate passes with many temporary allocations.
3. **No custom memory allocator / arena** — inference allocates dozens of temporary `Vec<f32>` per token.
4. **No batched / continuous batching in compute kernels** — decode is strictly single-sequence, single-token.
5. **GEMM is naive CPU triple-loop** — no BLAS, no tiling, no cache blocking, no SIMD in GEMM.
6. **Quantized GEMV is scalar dequantization** — Q4_K/Q6_K/Q8_0 GEMV extracts bits one at a time in scalar loops.
7. **No kernel-level prefill optimization** — flash attention prefill is O(q_seq × kv_seq) scalar loops.
8. **KV cache is not page-based / paged attention** — sliding window only, no vLLM-style paging.

---

## 1. SIMD / Vectorization Analysis

### 1.1 What IS Implemented

| File | SIMD | Details |
|------|------|---------|
| `compute/flash_attention.rs:12-46` | **AVX2 + FMA** | `dot_product_f32_avx2` — 8-wide `_mm256_loadu_ps` + `_mm256_fmadd_ps` + horizontal sum via stack array. Used in both decode and prefill. |
| `compute/tensor.rs:358-420` | **AVX2 + FMA** | `accumulate_f32_row_avx2` — transposed GEMV (output += row × factor). 8-wide FMA. Used when `rows×cols ≥ PARALLEL_GEMV_MIN_OPS` (1 MiB ops). |
| `compute/tensor.rs:490-607` | **AVX2 (int8 unpack)** | `accumulate_q4_block_avx2` — Q4_K dequant nibble unpack via SSE `_mm_unpacklo_epi8` + AVX2 `_mm256_cvtepi8_epi32` + `_mm256_add_ps`. Only fires for Q4_K transposed GEMV with full 256-element blocks. |
| `compute/simd.rs` | **Detection only** | `SimdBackend` enum lists AVX512F/AVX2/AVX/SSE2/NEON, but **no kernels actually dispatch to AVX-512 or NEON**. |

### 1.2 What Is NOT Implemented

- **AVX-512**: Detected in `simd.rs` but **zero kernels use it**. The `dot_product_f32_avx2` could trivially be widened to 16 floats with `_mm512_loadu_ps` + `_mm512_fmadd_ps` + `_mm512_reduce_add_ps`.
- **NEON / ARM**: No ARM NEON kernels anywhere. On Apple Silicon or ARM servers, everything falls back to scalar loops.
- **SSE2 fallback**: No explicit SSE2 kernels; scalar is the only fallback below AVX2.
- **No SIMD in GEMM**: `gemm_f32_cpu` (`tensor.rs:1048-1080`) is pure scalar triple-loop with manual transpose.
- **No SIMD in RoPE, RMSNorm, LayerNorm, Softmax, SwiGLU**: All scalar elementwise loops.
- **No SIMD in quantization encode/decode**: `quantization.rs` scalar loops for all Q4/Q5/Q6/Q8/K-quants.

### 1.3 Code References

```rust
// flash_attention.rs:12 — AVX2 dot product gate
if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    return unsafe { dot_product_f32_avx2(a, b) };
}

// tensor.rs:406 — AVX2 availability gate for transposed GEMV
fn f32_fma_avx2_available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

// simd.rs:66 — AVX-512 is detected but never called in compute
if has_avx512f() { return SimdBackend::Avx512f; }
```

---

## 2. Threading Model

### 2.1 What Is Used

| Mechanism | Where | Purpose |
|-----------|-------|---------|
| **rayon `par_iter_mut`** | `tensor.rs` (GEMV, transposed GEMV, Q8_0 GEMV) | Parallelize over output rows / chunks when `rows×cols ≥ 1_048_576` (1 MiB ops). |
| **rayon `par_chunks_mut`** | `tensor.rs:360` | Transposed GEMV parallelizes over column chunks of size `TRANSPOSED_GEMV_COL_CHUNK` (4096). |
| **rayon `ThreadPoolBuilder`** | `quantization.rs:342` | Mixed quantization spawns a custom thread pool with `thread_count = min(available_parallelism, plans.len())`. |
| **`std::thread::scope`** | `tensor.rs:863` | Tensor-parallel GEMM shards over K dimension when `shared_dim ≥ 1024`. |

### 2.2 Threading Gaps

- **No intra-kernel threading in flash attention**: Decode and prefill are strictly single-threaded. For long sequences this is a massive bottleneck.
- **No batch parallelism**: Even though `InferenceModel::forward` loops over tokens, there is no batching of heads or sequences.
- **Tensor-parallel GEMM is naive**: Spawns `std::thread` per shard, allocates `rows×cols` partial per shard, then serial reduction. No work-stealing, no cache-aware tiling.
- ** rayon threshold is coarse**: `PARALLEL_GEMV_MIN_OPS = 1 << 20` means small-ish matrices (e.g., 4096×256 = 1M) barely trigger parallelism, and the overhead may not be worth it.

### 2.3 Code References

```rust
// tensor.rs:15
const PARALLEL_GEMV_MIN_OPS: usize = 1 << 20;

// tensor.rs:282-286 — GEMV row parallelism via rayon
if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
    output.par_iter_mut().enumerate().for_each(|(row_idx, out)| {
        *out = compute_row(row_idx);
    });
}

// tensor.rs:863 — Tensor-parallel GEMM via std::thread::scope
let partials = std::thread::scope(|scope| { ... });
```

---

## 3. Memory Allocation Patterns

### 3.1 Temporary Allocations Per Token (Inference Hot Path)

In `inference.rs::forward_single`, **every token allocates many temporary vectors**:

| Allocation | Size | Line |
|-----------|------|------|
| `x` (hidden state) | `hidden_size` | 439 |
| `normed` (attn norm) | `hidden_size` | 494 |
| `gate` (mamba) | `gate_len` | 499 |
| `x_proj` (mamba qkv) | `qkv_out_len` | 513 |
| `conv_out` (mamba) | `qkv_out_len` | 519 |
| `mamba_out` | `half` | 553 |
| `x_ssm` | `half` | 554 |
| `z_gate` | `half` | 556 |
| `normed_group` (SSM) | `group_size` | 569 |
| `bx` | `state_dim` | 593 |
| `y_ssm` | `y_len` | 620 |
| `gate_to_use` | `mamba_out.len()` | 654 |
| `residual` | `hidden_size` | 664 |
| `projected` | `out_len` | 668 |
| `attn_out` | `hidden_size` | 694 |
| `normed` (attn) | `hidden_size` | 696 |
| `q_full` | `q_len` | 713 |
| `q` | `q_len_actual` | 730 |
| `k_vec` | `kv_len` | 732 |
| `v_vec` | `kv_len` | 733 |
| `normed_head` (Q norm) | `q_head_dim` | 777 |
| `normed_head` (K norm) | `kv_head_dim` | 796 |
| `rotated` (Q RoPE) | `q_head_dim` | 817 |
| `rotated` (K RoPE) | `kv_head_dim` | 835 |
| `key_cache` / `value_cache` (copy fallback) | `seq_len × kv_len` | 875-876 |
| `attn_result` | `q_len_used` | 897 |
| `out_head` | `kv_head_dim` | 918 |
| `padded` (attn result) | `attn_output_input_len` | 947 |
| `ffn_out` | `hidden_size` | 985 |
| `normed` (FFN) | `hidden_size` | 987 |
| `gate` (FFN) | `intermediate_size` | 991 |
| `up` (FFN) | `intermediate_size` | 992 |
| `swiglu` | `intermediate_size` | 1004 |
| `normed` (final) | `hidden_size` | 1033 |
| `logits` | `vocab_size` | 1037 |

**Total temporary bytes per token ≈**  
`~4 × (hidden_size × layers + intermediate_size × layers + seq_len × kv_len × layers + vocab_size)`  
For a 7B model (hidden=4096, intermediate=11008, layers=32, vocab=32000): **~2.5 GB of temporary allocations per token** (many repeated every layer).

### 3.2 Layer-Wise Model Adds MMAP + Cache

`layer_wise.rs` avoids loading all layers into RAM simultaneously. It uses:
- `Arc<MappedGgufFile>` for memory-mapped weights.
- `LayerCache` (LRU eviction by generation counter) with configurable capacity.
- Each layer is dequantized on demand into `Vec<f32>` or kept as `Vec<u8>` quantized blobs.

This is **good for memory footprint** but adds per-layer dequantization overhead and still allocates layer weights fresh on cache miss.

### 3.3 No Arena / Bump Allocator

There is **no custom allocator** or scratch buffer reuse. Every `vec![0.0_f32; N]` goes to the global allocator. For inference this means:
- Frequent `malloc/free` churn.
- Poor cache locality between layers.
- No way to pre-allocate a workspace and reuse it across tokens.

### 3.4 Code References

```rust
// inference.rs:439 — typical pattern repeated dozens of times
let mut x = vec![0.0_f32; h];

// inference.rs:875-876 — KV cache copy fallback allocates full layer cache every token
let mut key_cache = vec![0.0_f32; seq_len * kv_len];
let mut value_cache = vec![0.0_f32; seq_len * kv_len];
```

---

## 4. Cache Efficiency & Memory Layout

### 4.1 Flash Attention Decode

- **Online softmax** (Milakov & Gimelshein 2018) is implemented correctly — avoids full materialization of the attention score matrix.
- **Block size = 64 tokens** (`FLASH_BLOCK_SIZE`). This is reasonable for L1/L2 cache but not tuned for modern CPUs (128 or 256 may be better).
- **Dot product is AVX2** — good, but the rest of the loop (exp rescaling, value accumulation) is scalar.
- **No head-level parallelism** — heads are processed sequentially inside the layer loop.

### 4.2 GEMM

- `gemm_f32_cpu` transposes the right matrix into a new `Vec<f32>` (`right_transposed`). This is **cache-friendly for the inner loop** (sequential access) but:
  - The transpose itself is O(N) and unvectorized.
  - The inner loop is scalar, no tiling, no register blocking.
  - The "prefetch" hint uses `std::hint::black_box` which is a no-op for actual prefetching.

### 4.3 Transposed GEMV (GGUF Weights)

- GGUF stores weights as `[input_dim, output_dim]` (transposed vs PyTorch).
- `gemv_f32_transposed_cpu` iterates over input rows and accumulates into output columns.
- For large matrices it parallelizes over output column chunks (`TRANSPOSED_GEMV_COL_CHUNK = 4096`).
- AVX2 is used for the accumulation (`accumulate_f32_row_avx2`).
- **Good layout for weights** (row-major, sequential access), but the chunk size is fixed and not tuned to L1 cache.

### 4.4 KV Cache Layout

- `KvCache` uses **layer-major layout**: `[layer][position][head][head_dim]`.
- This allows `f32_layer_key_prefix` to borrow a contiguous slice when the sliding window has not wrapped.
- Supports F32, F16, Q8, Q4 storage.
- **Not page-based**: No vLLM-style paging, no block tables. Eviction is simple sliding window or stop-at-capacity.

### 4.5 Quantized Weight Access

- Q4_K/Q6_K/Q8_0 GEMV reads scales (`f16`) and bitstreams per block.
- The bitstream is accessed via `extract_bits` which does **4-byte unaligned loads and bit shifts per element** — extremely expensive.
- No lookup tables, no pre-unpacked caches.

---

## 5. Performance Bottlenecks & Low-Hanging Fruit

### 5.1 Critical Bottlenecks (Fix First)

| Rank | Bottleneck | Impact | Fix Complexity |
|------|-----------|--------|--------------|
| 1 | **Temporary allocations in inference.rs** | Massive malloc churn, GC pressure, poor locality | Medium — add workspace buffers |
| 2 | **Scalar GEMM** | Every matmul in FFN/attention projection is naive O(N³) scalar | High — integrate BLIS/OpenBLAS/MKL or write tiled SIMD GEMM |
| 3 | **Scalar bit extraction in quantized GEMV** | Q4_K/Q6_K dequant per element is ~10× slower than llama.cpp | High — write block-unpack SIMD kernels |
| 4 | **Single-threaded flash attention decode** | Long-context decode is CPU-bound and not parallelized | Medium — parallelize over heads or blocks |
| 5 | **No operator fusion** | Norm → GEMV → activation are separate passes with intermediate buffers | Medium — fuse norm+gemv+silu, rope+q/k projection |
| 6 | **No AVX-512 / NEON** | Missing 2× SIMD width on x86 and all SIMD on ARM | Low-Medium — extend existing AVX2 kernels |
| 7 | **No batched decode** | Continuous batching exists in KV cache API but compute is still per-token | High — rewrite kernels for batch dims |

### 5.2 Low-Hanging Fruit

1. **Replace `vec!` allocations with reusable scratch buffers in `InferenceModel`**  
   Add a `Workspace` struct with pre-allocated `Vec<f32>` slots for `normed`, `gate`, `up`, `q`, `k`, `v`, `attn_result`, etc. Reset lengths rather than reallocating.

2. **Widen `dot_product_f32_avx2` to AVX-512**  
   Copy-paste the AVX2 kernel, replace `_mm256_*` with `_mm512_*`, add `target_feature(enable = "avx512f")`. Gate with `has_avx512f()`.

3. **Add NEON `dot_product_f32` and `accumulate_f32_row`**  
   Use `vld1q_f32`, `vfmaq_f32`, `vaddvq_f32` (or pairwise reduction). Essential for Apple Silicon.

4. **Fuse RMSNorm + GEMV in FFN**  
   Instead of `rms_norm_f32` → `gemv_weight` (which reads `normed` from RAM), compute `normed` on the fly inside the GEMV row loop if the weight layout allows.

5. **Use `black_box` prefetch correctly or remove it**  
   `std::hint::black_box(*next_left + *next_right)` does not prefetch. Replace with `_mm_prefetch` on x86 or remove.

6. **Pre-allocate `block_acc` and `block_scores` in `scaled_dot_product_attention_f32`**  
   Currently allocated fresh on every call. These can be thread-local scratch buffers.

---

## 6. Benchmarks

### 6.1 What Exists

File: `oxidize-core/benches/criterion.rs`

| Benchmark | What It Measures |
|-----------|-----------------|
| `loader/mapped_gguf/*` | GGUF parsing speed vs llama.cpp baseline |
| `memory/loader/*` | Memory delta during GGUF load |
| `perplexity/dataset/*` | Text perplexity (toy metric, not real model eval) |
| `flash_attention/decode/{64,256,512,1024,2048}` | Decode flash attention with head_dim=128, kv_heads=8 |
| `flash_attention/prefill/{64x64,128x128,256x256,512x512}` | Prefill flash attention with head_dim=128 |

### 6.2 What Is Missing

- **No GEMV/GEMM microbenchmarks** for quantized vs f32.
- **No end-to-end token generation throughput benchmark** (tok/sec).
- **No memory bandwidth benchmark** (crucial for decode-bound inference).
- **No multi-threaded scaling benchmark** (how does throughput change with cores?).
- **No batch throughput benchmark**.
- **No comparison against llama.cpp / candle for actual model inference**.

---

## 7. Operator / Kernel Fusion

### 7.1 What Is Fused

- **SwiGLU**: `apply_swiglu_f32` fuses SiLU(gate) × up in a single loop. Good.
- **Q8_0 GEMV**: `gemv_q8_0_f32_fused` dequantizes and multiplies on-the-fly without full dequant buffer. Good.
- **Q4_K / Q6_K GEMV**: `gemv_qk_f32_fused` similarly fuses dequant + dot product, but the inner loop is scalar bit extraction.

### 7.2 What Is NOT Fused

| Unfused Sequence | Where | Cost |
|-----------------|-------|------|
| RMSNorm → GEMV (attn_q/k/v) | `inference.rs:696-733` | Writes `normed` to RAM, then reads it 3× for Q/K/V projections |
| RoPE → KV Cache write | `inference.rs:817-835` | Writes rotated Q/K to RAM, then reads back for attention |
| Flash Attention → GEMV (attn_output) | `inference.rs:897-945` | Writes `attn_result` to RAM, then reads for `attn_output` GEMV |
| RMSNorm → GEMV (ffn_gate/up) | `inference.rs:987-992` | Writes `normed` to RAM, then reads twice |
| SwiGLU → GEMV (ffn_down) | `inference.rs:1004-1010` | Writes `swiglu` to RAM, then reads for `ffn_down` |
| Final RMSNorm → Output GEMV | `inference.rs:1033-1037` | Separate passes |

**Fusion opportunity**: In decode (single token), the intermediate activations are small (hidden_size ~ 4-8 KB). Fusing norm+gemv+activation into a single pass would eliminate most of these writes and significantly improve cache locality.

---

## 8. Custom Memory Allocator

### 8.1 What Exists

- **No global custom allocator**.
- **Metal backend has `UnifiedBufferManager`** (`metal.rs:170-260`): page-aligned bump allocator with LRU cache eviction. This is **only for Metal**, not CPU.
- **CUDA backend has `DeviceBuffer`** (`cuda.rs:35-95`): simple host + device buffer wrapper. No pooling.

### 8.2 What Is Needed

- A **CPU workspace allocator** (bump allocator or ring buffer) for inference scratch space.
- A **quantized weight cache** for layer-wise loading (already partially done in `layer_wise.rs` but not allocator-based).
- **Page-based KV cache allocator** (vLLM-style) for efficient continuous batching and prefix sharing.

---

## 9. KV Cache Management

### 9.1 Architecture

File: `compute/kv_cache.rs`

- **Storage layouts**: F32, F16, Q8 (per-token scale+min), Q4 (nibble-packed, per-token scale+min).
- **Logical layout**: Layer-major `[layer][position][head][head_dim]`.
- **Eviction strategies**: `SlidingWindow` (overwrites old physical slots) and `StopAtCapacity`.
- **Contiguous borrow**: `f32_layer_key_prefix` returns `&[f32]` when the logical prefix `[0, seq_len)` is physically contiguous (no wrap). This avoids copies for flash attention.
- **Continuous batching API**: `ContinuousBatchKvCache` tracks multiple sequences with position vectors, but the underlying cache is still a single flat buffer.

### 9.2 Strengths

- Quantized KV cache reduces memory footprint (2× for F16, 4× for Q4).
- Borrow path avoids copies for the common case (no wrap).
- Supports sliding window for long contexts.

### 9.3 Weaknesses

- **No paged attention**: The cache is a flat circular buffer per layer. No block tables, no prefix sharing across sequences, no vLLM-style memory efficiency.
- **No KV cache compression / eviction policies**: No H2O, Scissorhands, or FastGen-style eviction.
- **Q4/Q8 KV cache read is scalar**: `read_storage` dequantizes element-by-element. No SIMD unpacking.
- **Copy fallback is expensive**: When the sliding window wraps, `copy_layer_keys` / `copy_layer_values` do full `seq_len × kv_len` copies every token.

### 9.4 Code References

```rust
// kv_cache.rs:175-185 — Q4 read is scalar nibble extraction
KvStorage::Q4 { data, scales, mins } => {
    let scale = scales[token_index];
    let min = mins[token_index];
    let packed_start = range.start / 2;
    for (pair_index, pair) in dst.chunks_mut(2).enumerate() {
        let byte = data[packed_start + pair_index];
        pair[0] = ((byte & 0x0F) as f32) * scale + min;
        ...
    }
}
```

---

## 10. Quantization Kernels

### 10.1 Supported Formats

File: `compute/quantization.rs`

| Format | Block Size | Encode | Decode | GEMV |
|--------|-----------|--------|--------|------|
| F32 | 1 | ✅ scalar | ✅ scalar | N/A |
| F16 | 1 | ✅ scalar (custom f32↔f16) | ✅ scalar | N/A |
| Q4_0 | 32 | ✅ scalar | ✅ scalar | N/A |
| Q4_1 | 32 | ✅ scalar | ✅ scalar | N�A |
| Q5_0 | 32 | ✅ scalar | ✅ scalar | N/A |
| Q5_1 | 32 | ✅ scalar | ✅ scalar | N/A |
| Q8_0 | 32 | ✅ scalar | ✅ scalar | ✅ fused scalar |
| Q2_K | 256 | ✅ scalar | ✅ scalar | N/A |
| Q3_K_* | 256 | ✅ scalar | ✅ scalar | N/A |
| Q4_K_S/M | 256 | ✅ scalar | ✅ scalar | ✅ fused scalar (+ AVX2 for full blocks) |
| Q5_K_S/M | 256 | ✅ scalar | ✅ scalar | N/A |
| Q6_K | 256 | ✅ scalar | ✅ scalar | ✅ fused scalar |

### 10.2 Optimization Level

- **Encode (quantize)**: All scalar loops. Acceptable for one-time model loading.
- **Decode (dequantize)**: All scalar loops. Used when weights are dequantized for unsupported formats or small tensors. Slow.
- **GEMV (fused)**: Q8_0 and Q4_K have fused GEMV kernels, but the inner loop is scalar bit extraction except for Q4_K transposed GEMV which has an AVX2 path for full 256-element blocks.

### 10.3 Comparison to llama.cpp

llama.cpp uses:
- Hand-written AVX2/AVX-512/NEON assembly for all Q4/Q5/Q6/Q8/K-quant GEMV/GEMM.
- Lookup tables for Q4 nibble unpacking.
- Block interleaving and super-block optimization.

Oxidize is **orders of magnitude slower** on quantized inference because every quantized weight is unpacked with `extract_bits` (bit shifts and masks) per scalar element.

### 10.4 Code References

```rust
// tensor.rs:271-279 — scalar bit extraction per element in Q4_K/Q6_K GEMV
fn compute_row(row_idx: usize) {
    for (block_idx, block) in row_blocks.chunks_exact(block_size).enumerate() {
        let d = f16_le_to_f32([block[0], block[1]]);
        let bitstream = &block[2..];
        for idx in 0..QK_K {
            let q = extract_bits(bitstream, idx, bits) as f32;
            sum += (q - zero_point) * d * vector[vector_offset + idx];
        }
    }
}
```

---

## 11. What's Missing vs. State-of-the-Art

| Feature | Oxidize Status | SOTA (llama.cpp / vLLM / TensorRT-LLM) |
|--------|----------------|----------------------------------------|
| **AVX-512 kernels** | ❌ Detected only | ✅ Full GEMV/GEMM/attention |
| **NEON kernels** | ❌ None | ✅ Full on Apple Silicon / ARM |
| **BLAS / tuned GEMM** | ❌ Naive scalar | ✅ OpenBLAS/BLIS/MKL/cuBLAS |
| **Operator fusion** | ⚠️ SwiGLU only | ✅ Fused attention, fused MLP, fused norm+proj |
| **Custom allocator** | ❌ None (CPU) | ✅ Arena, bump pool, weight cache |
| **PagedAttention** | ❌ Sliding window only | ✅ Block tables, prefix sharing |
| **Continuous batching (compute)** | ❌ API only | ✅ Batched GEMM, batched attention |
| **Speculative decoding** | ❌ Not present | ✅ Draft model, tree attention |
| **FP8 / BF16** | ❌ Not present | ✅ NVIDIA FP8, AMD BF16 |
| **Tensor parallelism (distributed)** | ⚠️ CPU-only K-shard | ✅ NCCL, pipeline + tensor parallel |
| **Pipeline parallelism** | ❌ Not present | ✅ Multi-GPU pipeline |
| **FlashAttention-2/3** | ⚠️ Basic online softmax | ✅ Tiled, sequence-parallel, warp-specialized |
| **KV cache quantization (inference)** | ⚠️ Q8/Q4 storage, scalar read | ✅ Q4_0/Q8_0 with SIMD dequant |
| **Prefix caching** | ❌ Not present | ✅ RadixAttention (vLLM) |
| **LoRA / adapter serving** | ⚠️ Module exists (`lora.rs`) | ✅ PEFT adapter swapping |
| **GPU kernels (CUDA)** | ⚠️ PTX for GEMV/GEMM only | ✅ Full model CUDA graphs |
| **GPU kernels (Metal)** | ⚠️ MSL source included, stub dispatch | ✅ Full MPS / custom Metal kernels |
| **WebGPU kernels** | ❌ Validation only | ✅ Compute shaders for WASM |

---

## 12. Recommendations (Priority Order)

1. **Add workspace buffers to `InferenceModel`** to eliminate per-token `vec!` allocations. This is the easiest win with the biggest latency improvement for decode.
2. **Write SIMD unpack kernels for Q4_K and Q8_0** in `tensor.rs`. Target AVX2 first, then NEON. Use lookup tables or SSE nibble unpack (like the existing `accumulate_q4_block_avx2`) for all block sizes, not just 256.
3. **Parallelize flash attention decode over heads** using rayon. For multi-head attention this is embarrassingly parallel.
4. **Integrate a BLAS library** (OpenBLAS, BLIS, or Intel MKL) for `gemm_f32_cpu`. Even a simple C FFI to `cblas_sgemm` would outperform the naive triple-loop by 10-50×.
5. **Add AVX-512 and NEON dispatch** to `dot_product_f32`, `accumulate_f32_row`, and `rms_norm_f32`.
6. **Fuse RMSNorm + GEMV** for the FFN and attention projections. For single-token decode, the intermediate vectors fit in L1 cache.
7. **Implement page-based KV cache** with block tables to enable true continuous batching and prefix sharing.
8. **Add end-to-end throughput benchmarks** (tokens/sec) for real models, not just microbenchmarks.

---

*Report generated by automated codebase analysis.*
