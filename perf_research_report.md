# Performance Research Report: llama.cpp vs oxidize

**Date:** 2026-05-20
**Branch:** `deepflash-safetensors-perf`
**Scope:** Compare llama.cpp's key performance optimizations against oxidize's current implementation, identify gaps, and recommend top improvements.

---

## 1. Executive Summary

llama.cpp (via GGML) is one of the fastest CPU-first LLM inference engines in existence. Its performance comes from a deep stack of optimizations: hand-written SIMD kernels for every quantization format, a computation graph with backend-specific operator fusion, mmap-based model loading, NUMA-aware threading, continuous batching, speculative decoding, and fused Flash Attention on GPU backends. oxidize has made solid progress—AVX2 GEMV, online softmax Flash Attention, quantized GEMV for Q4_K/Q6_K/Q8_0, layer-wise caching, and multi-backend stubs (CUDA/Metal/WebGPU)—but it is missing many of the architectural and micro-architectural optimizations that make llama.cpp fast. This report catalogs those gaps and ranks the highest-impact fixes.

---

## 2. Feature Comparison Table

| # | Performance Feature | llama.cpp Status | oxidize Status | Gap Severity |
|---|---------------------|------------------|----------------|--------------|
| 1 | **Hand-optimized SIMD GEMM/GEMV kernels** | ✅ Extensive: AVX2, AVX512, AMX, ARM NEON, SVE, KleidiAI | ⚠️ Partial: AVX2 transposed GEMV, Q4 AVX2 unpack, scalar fallback | **High** |
| 2 | **Quantization format coverage** | ✅ Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q8_K, IQ quants | ⚠️ Q8_0, Q4_K_S/M, Q6_K + TurboQuant INT4/8 | **Medium** |
| 3 | **Computation graph + operator fusion** | ✅ GGML graph fuses norm→mul, add→silu, attention patterns | ❌ No graph; ops called sequentially in Rust | **High** |
| 4 | **Kernel fusion (Flash Attention + quant KV)** | ✅ CUDA/HIP fused FA with quantized KV (symmetric) | ⚠️ CPU Flash Attention (f32 only); no fused quant FA | **High** |
| 5 | **mmap model loading / zero-copy weights** | ✅ `mmap` by default; weights stay on disk until touched | ⚠️ `memmap2` for GGUF header; weights copied into `Vec<u8>` | **Medium** |
| 6 | **Memory pool / scratch buffer allocator** | ✅ `ggml_context` scratch buffers; no per-op allocations | ❌ `Vec` allocations every layer (key/value copies, temps) | **High** |
| 7 | **Threading / NUMA-aware scheduling** | ✅ `ggml_threadpool`, pinned threads, NUMA node affinity | ⚠️ `rayon` `par_iter` for GEMV above threshold only | **Medium** |
| 8 | **Continuous batching (server)** | ✅ `llama-server` with `--cont-batching` | ⚠️ `ContinuousBatchKvCache` exists but not wired to server | **Medium** |
| 9 | **Speculative decoding** | ✅ Full implementation with draft model + tree verification | ❌ Not implemented | **High** |
| 10 | **KV cache quantization (runtime)** | ✅ Q4_0, Q5_0, Q8_0 KV cache; TurboQuant-style research | ⚠️ Q8/Q4 storage types exist but decode path dequantizes to f32 | **Medium** |
| 11 | **Multi-GPU tensor/pipeline parallelism** | ✅ Tensor + pipeline parallel, RPC backend | ⚠️ Offload plan structs only; no actual GPU dispatch | **Medium** |
| 12 | **GPU backend maturity (CUDA/Metal/Vulkan)** | ✅ Mature: cuBLAS, custom CUDA kernels, Metal MPS, Vulkan | ⚠️ Stubs: CUDA/Metal/WebGPU have validation + CPU fallback only | **High** |
| 13 | **Prefill batching / prompt chunking** | ✅ Batched matmul for prompt processing | ⚠️ Prefill batch size config exists; no batched GEMM | **Medium** |
| 14 | **RoPE / RMSNorm fused kernels** | ✅ Fused RoPE, fused RMSNorm×Mul in backends | ❌ Scalar Rust loops for RoPE, RMSNorm, SwiGLU | **Medium** |
| 15 | **Auto-tuned tile sizes / micro-kernels** | ✅ Architecture-specific GEMM micro-kernels (e.g. 4×24) | ❌ Fixed `TRANSPOSED_GEMV_COL_CHUNK = 4096` | **Medium** |
| 16 | **DeepSeek MLA / shared KV compression** | ✅ MLA optimization (K-cache only, 47% saving) | ❌ Not implemented | **Medium** |
| 17 | **MTP / Multi-Token Prediction** | ✅ Draft tokens via MTP heads (ik_llama.cpp fork) | ❌ Not implemented | **Low** |
| 18 | **Build-time CPU feature dispatch** | ✅ Runtime dispatch for every ISA variant | ⚠️ Runtime AVX2 detection; no AVX512/AMX/NEON GEMV | **High** |

---

## 3. Specific Gaps Where oxidize Is Likely Slower

### 3.1 Missing Operator Fusion (Biggest Decode-Phase Cost)
llama.cpp's GGML graph executor fuses common patterns:
- `RMSNorm → Mul(weights)` into one kernel
- `Add → SiLU → Mul` (SwiGLU gate) into one kernel
- `Q×K^T → Softmax → ×V` into fused Flash Attention on GPU

oxidize calls each operation as a separate Rust function with intermediate `Vec<f32>` allocations. In the decode phase (single token), memory bandwidth is the bottleneck; every extra read/write of activations cuts throughput.

**Evidence:** `inference.rs` shows `rms_norm_f32` → `gemv_weight` (Q, K, V) → `apply_rope_f32` → `flash_attention_decode_f32` → `gemv_weight` (O) → `rms_norm_f32` → `gemv_weight` (gate/up) → `apply_swiglu_f32` → `gemv_weight` (down). No fusion between any pair.

### 3.2 No True GEMM Kernel (Prefill Is Slow)
llama.cpp uses optimized GEMM for prompt processing (batched Q, K, V projections). oxidize has `gemm_f32_cpu` and `gemm_f32_tensor_parallel`, but the CPU implementation is a naive triple loop with no blocking/tiling:
```rust
for row in 0..rows {
    for k in start_k..end_k {
        for col in 0..cols {
            partial[row * cols + col] += left * right_row[col];
        }
    }
}
```
This is cache-unfriendly and lacks SIMD vectorization. Prefill throughput will be an order of magnitude slower than llama.cpp on the same hardware.

### 3.3 Memory Allocation Overhead Per Layer
oxidize allocates temporary vectors inside the layer loop:
- `InferenceModel::forward_layer` copies KV cache rows into contiguous `Vec<f32>` for attention.
- `LayerWiseModel` copies layer weights from mmap into `LayerWeights` on every access (unless cached).

llama.cpp uses scratch buffers pre-allocated in `ggml_context` and reuses them across layers. oxidize's per-layer `Vec` allocations cause heap pressure and cache pollution.

### 3.4 Quantized GEMV Is Scalar for Most Formats
oxidize supports on-the-fly quantized GEMV for `Q8_0`, `Q4_K_S/M`, and `Q6_K`. However:
- Only `Q4_K` transposed GEMV has an AVX2 fast path (`accumulate_q4_block_avx2`).
- `Q8_0` transposed GEMV is scalar.
- `Q6_K` is entirely scalar.
- No ARM NEON path exists.

llama.cpp has hand-optimized SIMD kernels for **every** quantization format on x86 (AVX2/AVX512/AMX) and ARM (NEON/SVE/KleidiAI).

### 3.5 Flash Attention Is CPU-Only and f32
oxidize's `flash_attention_decode_f32` uses online softmax with 64-token blocks and AVX2 dot product—good for CPU. But:
- No GPU implementation (CUDA/Metal kernels are stubs that fall back to CPU).
- No quantized KV cache path; KV is dequantized to f32 before attention.
- No causal masking optimization for prefill.

llama.cpp has fused Flash Attention on CUDA/HIP/Metal/Vulkan with quantized KV support, delivering 2–4× speedups on long contexts.

### 3.6 No Speculative Decoding
Speculative decoding is one of the highest-impact single features for latency reduction (25–40% speedup on consumer GPUs). oxidize has no draft model or tree-verification logic.

### 3.7 Backend Stubs (CUDA/Metal/WebGPU)
The `cuda.rs`, `metal.rs`, and `webgpu.rs` backends contain dimension validation and build-time feature flags, but the actual compute kernels fall back to `gemv_f32_cpu` / `gemm_f32_cpu`. There is no actual GPU dispatch. This means:
- `--n-gpu-layers` in the CLI is a no-op for performance.
- All inference runs on CPU regardless of flags.

### 3.8 Missing AVX512 and AMX Kernels
oxidize detects AVX512F at runtime (`simd.rs`) but never uses it in compute kernels. llama.cpp has AVX512 GEMV/GEMM kernels and Intel AMX tile-based matmul, which can be 2–4× faster than AVX2 for large matrices.

---

## 4. Top 10 Recommendations (Ranked by Impact)

| Rank | Recommendation | Estimated Impact | Effort | Notes |
|------|----------------|------------------|--------|-------|
| 1 | **Implement operator fusion for decode path** | 20–40% decode speedup | Medium | Fuse `RMSNorm+Mul`, `Add+SiLU+Mul`, `RoPE` into single kernels. Eliminate intermediate activation buffers. |
| 2 | **Add blocked + SIMD GEMM for prefill** | 5–10× prefill speedup | High | Implement cache-blocked GEMM with AVX2/AVX512/NEON. Use `matrixmultiply` or `blis` crate, or write custom micro-kernels. |
| 3 | **Wire up actual CUDA/Metal kernels** | 5–20× GPU speedup | High | Replace CPU fallback in `cuda.rs` / `metal.rs` with real cuBLAS/cuDNN or MPS calls. Add custom FA kernels. |
| 4 | **Add speculative decoding** | 25–40% latency reduction | Medium | Use a small draft model (same architecture, fewer layers) to predict 2–4 tokens, verify with target model. |
| 5 | **Pre-allocate scratch buffers + eliminate per-layer Vec allocs** | 10–20% decode speedup | Low | Reuse a single `Vec<f32>` workspace across layers; copy KV cache only when layout requires it. |
| 6 | **Write SIMD kernels for remaining quant formats (Q6_K, Q8_0 transposed, NEON)** | 2–4× quant inference speedup | Medium | Extend `accumulate_q4_block_avx2` pattern to Q6_K and Q8_0. Add ARM NEON unpack+mul kernels. |
| 7 | **Add AVX512 and AMX GEMV/GEMM paths** | 1.5–3× on supported CPUs | Medium | Use `std::arch::x86_64::_mm512_*` for GEMV. For AMX, use inline assembly or `libc` syscall wrapper. |
| 8 | **Implement fused quantized Flash Attention** | 2–4× attention speedup on long context | High | Fuse dequantization into attention dot products; keep KV cache quantized during decode. |
| 9 | **Add true continuous batching in server** | 2–5× server throughput | Medium | Wire `ContinuousBatchKvCache` into `oxidize-server`; batch independent sequences into single forward pass. |
| 10 | **Use mmap for weight tensors (zero-copy)** | 10–30% memory reduction, faster load | Low | Keep quantized weights as `&[u8]` slices into `Mmap` instead of `Vec<u8>` copies in `WeightStorage`. |

---

## 5. Detailed Analysis

### 5.1 llama.cpp's Performance Architecture

llama.cpp's speed is not from any single trick but from a vertically integrated stack:

1. **GGML Tensor Library:** A computation graph (`ggml_cgraph`) where each node is an operator. The graph is built once per model architecture and reused. Backends (CPU, CUDA, Metal, Vulkan, etc.) register implementations for each operator.
2. **Backend Operator Fusion:** The CUDA backend fuses `RMS_NORM + MUL`, `RMS_NORM + MUL + ADD`, and attention patterns into single kernels. CPU backend fusion is an active area (Discussion #22315).
3. **SIMD Kernel Matrix:** For every quantization type (`Q4_0` through `Q8_K` and IQ variants), there are x86 (SSE2/AVX/AVX2/AVX512/AMX) and ARM (NEON/SVE/KleidiAI) kernels. Kernels are selected at runtime via CPU feature detection.
4. **Memory Management:** `ggml_context` pre-allocates a scratch buffer. All intermediate tensors are views into this buffer. Model weights are `mmap`ped and stay page-cached by the OS.
5. **Threading:** `ggml_threadpool` creates a pool of pinned threads. Work is split into tasks with a work-stealing queue. NUMA awareness ensures memory is allocated on the node that will access it.
6. **Advanced Features:** Continuous batching (`--cont-batching`), speculative decoding (`--draft`), and multi-GPU tensor/pipeline parallelism are all production-ready.

### 5.2 oxidize's Current Strengths

- **Clean Rust architecture:** Strong type safety, good error handling, modular backends.
- **Flash Attention:** Online softmax decode kernel with AVX2 dot product is a solid foundation.
- **Quantization support:** GGUF parsing, Q4_K/Q6_K/Q8_0 GEMV, TurboQuant custom format.
- **Multi-backend stubs:** CUDA, Metal, WebGPU feature flags and validation are in place.
- **Layer-wise loading:** `LayerWiseModel` with LRU cache reduces memory for large models.
- **Continuous batching data structures:** `ContinuousBatchKvCache` and `KvCache` with sliding window eviction exist.

### 5.3 oxidize's Critical Weaknesses

- **No computation graph:** The inference loop is imperative Rust code. This makes operator fusion and backend optimization nearly impossible without a major refactor.
- **Naive GEMM:** Prefill performance is severely limited by the triple-loop GEMM.
- **Backend stubs:** GPU acceleration is currently a no-op.
- **Allocation churn:** Per-layer `Vec` allocations hurt decode latency.
- **Incomplete SIMD coverage:** Only AVX2 transposed GEMV and Q4 unpack are optimized.

---

## 6. Implementation Roadmap (Suggested Order)

### Phase 1: Low-Hanging Fruit (1–2 weeks)
1. Replace per-layer `Vec` allocations with a reusable scratch buffer.
2. Use `mmap` slices for `WeightStorage::Quantized` instead of `Vec<u8>` copies.
3. Add `RMSNorm+Mul` and `SwiGLU` fused scalar kernels as a proof of concept.

### Phase 2: CPU Kernel Improvements (2–4 weeks)
4. Integrate `matrixmultiply` crate (or `blis` bindings) for blocked GEMM.
5. Write AVX512 GEMV/GEMM paths (extend existing AVX2 patterns).
6. Write NEON GEMV paths for ARM support.
7. Add AVX2/NEON kernels for Q6_K and Q8_0 transposed GEMV.

### Phase 3: Architectural Improvements (4–8 weeks)
8. Design a lightweight computation graph (even a simple DAG of ops) to enable fusion.
9. Implement fused `RMSNorm→Mul`, `Add→SiLU→Mul`, and `RoPE` kernels.
10. Wire `ContinuousBatchKvCache` into the server for true continuous batching.

### Phase 4: GPU and Advanced Features (4–8 weeks)
11. Implement real CUDA kernels: cuBLAS GEMM/GEMV, custom Flash Attention.
12. Implement Metal MPS kernels for Apple Silicon.
13. Add speculative decoding with a draft model.
14. Implement quantized KV Flash Attention (fuse dequant into attention).

---

## 7. Conclusion

oxidize has a strong foundation in Rust with good architectural decisions (layer-wise loading, Flash Attention, quantization support, multi-backend stubs). However, it is currently missing the two biggest performance levers that make llama.cpp fast: **operator fusion** and **hand-optimized SIMD kernels for every quantization format**. Until these are addressed, oxidize will likely be 2–5× slower than llama.cpp on CPU and orders of magnitude slower on GPU because the backend stubs do not dispatch to actual GPU kernels.

The highest-ROI next steps are:
1. **Fuse the decode-path operators** to reduce memory bandwidth.
2. **Replace naive GEMM** with a blocked, SIMD-accelerated implementation.
3. **Turn backend stubs into real GPU dispatch** (cuBLAS / MPS).

These three changes alone could close 50–70% of the performance gap to llama.cpp.

---

*Report generated by research subagent. All findings based on source code analysis of oxidize (`deepflash-safetensors-perf` branch) and public documentation of llama.cpp/GGML as of 2026-05-20.*
