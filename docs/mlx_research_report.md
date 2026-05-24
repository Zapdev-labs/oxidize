# MLX Integration Research Report for Oxidize

## Executive Summary

Apple's MLX framework has a **official C API (`mlx-c`)** that can be called from Rust. There are multiple Rust binding crates available, with `mlx-rs` (by OxideAI/oxiglade) being the most mature unofficial option. MLX provides a **unified memory model** on Apple Silicon that eliminates CPU↔GPU copies, supports **native quantization** (2-8 bits, mxfp4/8), and offers **dedicated fast ops** for LLM primitives (attention, RMS norm, RoPE). MLX is measurably faster than PyTorch MPS and is now the default backend for Ollama on Apple Silicon.

---

## 1. MLX C API: Exists and Is Official

**Repository:** https://github.com/ml-explore/mlx-c  
**Docs:** https://ml-explore.github.io/mlx-c/build/html/index.html  
**Version:** 0.4.1 (as of docs snapshot)

MLX C is Apple's **official** C API wrapper around the C++ MLX core. It is actively maintained and is the same bridge used by the official **MLX Swift** bindings. This means:
- It tracks MLX core releases reliably
- It exposes the full operation surface (ops, linalg, FFT, random, IO, transforms, distributed, fast custom ops, Metal backend)
- It provides a stable ABI for FFI binding generation

### How to call from Rust
The C API exports plain C functions with `mlx_` prefix. Example pattern:
```c
int mlx_matmul(mlx_array* res, const mlx_array a, const mlx_array b,
               const mlx_stream s);
int mlx_array_new_data(mlx_array* arr, const void* data,
                       const int* shape, size_t ndim, mlx_dtype dtype);
```

You can bind to it directly with `bindgen` or use the existing `mlx-sys` crate (see Section 3).

---

## 2. Key C API Functions for LLM Inference

### 2.1 Matrix Multiplication
- **`mlx_matmul()`** — standard matrix multiply
- **`mlx_addmm()`** — fused `alpha*matmul(a,b) + beta*c`
- **`mlx_quantized_matmul()`** — matrix multiply with a quantized weight matrix `w` using per-group scales/biases
- **`mlx_block_masked_mm()`** — block-sparse masked matmul (useful for MoE/structured sparsity)

### 2.2 Attention
MLX provides a **single, optimized fused attention kernel** in the C API:
- **`mlx_fast_scaled_dot_product_attention()`**
  - Signature includes `q`, `k`, `v`, `scale`, `mask` string (e.g., `"causal"`), plus optional `key_mask` and `value_mask`
  - This is MLX's native FlashAttention-style fused op

### 2.3 Normalization & Embeddings (Fast Ops)
Located in the **Fast Custom Ops** API (`fast.h`):
- **`mlx_fast_rms_norm()`** — fused RMSNorm (used by Llama/Qwen/etc.)
- **`mlx_fast_layer_norm()`** — fused LayerNorm
- **`mlx_fast_rope()`** — fused Rotary Position Embedding
- **`mlx_fast_rope_dynamic()`** — dynamic RoPE (for variable-length sequences)

### 2.4 Quantization
Located in `ops.h`:
- **`mlx_quantize()`** — quantize an array to specified bits/group_size
- **`mlx_dequantize()`** — dequantize back to full precision
- **`mlx_quantized_matmul()`** — perform matmul directly on quantized weights

### 2.5 Buffer / Array Management
Core object lifetime functions in `array.h` / `device.h` / `stream.h`:
- **`mlx_array_new_data()`** — create array from host data pointer
- **`mlx_array_set()`** / **`mlx_array_free()`** — retain/release
- **`mlx_device_new()`** — create CPU or GPU device handle
- **`mlx_stream_new()`** — create async execution stream
- **`mlx_synchronize()`** — barrier / flush streams

### 2.6 Linear Algebra
`linalg.h` provides:
- `mlx_linalg_svd()`, `mlx_linalg_qr()`, `mlx_linalg_inv()`, `mlx_linalg_solve()`, `mlx_linalg_cholesky()`

### 2.7 I/O
- **`mlx_load()`** / **`mlx_save()`** — load/save arrays (`.safetensors`-compatible, `.npz`, `.gguf` via the ecosystem)

---

## 3. Existing Rust Bindings

### Primary Crate: `mlx-rs`
- **Crates.io:** https://crates.io/crates/mlx-rs
- **GitHub:** https://github.com/oxiglade/mlx-rs (formerly oxideai/mlx-rs)
- **Latest:** 0.25.3
- **MSRV:** Rust 1.82.0+
- **Status:** Active development, ~328 stars, 53 forks

**Architecture:**
- `mlx-rs` — safe, idiomatic Rust API (tensors, ops, nn modules)
- `mlx-sys` (pinned at `=0.2.0`) — low-level `bindgen`-generated FFI to `mlx-c`
- `mlx-macros` / `mlx-internal-macros` — derive macros

**Key dependencies:** `half`, `num-complex`, `num-traits`, `safetensors` (optional), `bytemuck` (optional), `libc`, `mach-sys`, `parking_lot`

**Features:**
- `metal` — enables GPU usage
- `accelerate` — enables Apple's Accelerate framework on CPU
- `safetensors` — load/save SafeTensors

**Important caveat:** `docs.rs` builds for `mlx-rs` currently **fail** because the docs.rs builders are Linux-only and cannot link the macOS-only `mlx-c` library. Documentation is hosted externally at https://oxideai.github.io/mlx-rs/mlx_rs/.

### Alternative / Fork Crates
| Crate | Purpose |
|---|---|
| `mlx-sys` | Low-level FFI only; follows `mlx-c` versioning |
| `burn-mlx` | MLX backend for the **Burn** deep learning framework |
| `mlx-rs-burn` | Fork of `mlx-rs` with extra ops needed by `burn-mlx` |
| `pmetal-mlx-rs` | Fork maintained by pmetal; may have Metal-specific patches |
| `quill-mlx` | Bindings with explicit Metal memory management |
| `apple-mlx` | Another set of Rust wrappers (earlier/experimental) |

**Recommendation:** Start with **`mlx-rs`** for a safe Rust API. If you need operations it hasn't wrapped yet, fall back to `mlx-sys` raw FFI calls. If you use the **Burn** framework, use `burn-mlx`.

---

## 4. Unified Memory on Apple Silicon

### How It Works
MLX is architected specifically for Apple Silicon's **unified memory architecture (UMA)**:
- CPU and GPU share the **same physical memory pool**
- An `mlx_array` has **no fixed device location** — it lives in unified memory
- You do not "move" arrays to GPU; you specify the **stream/device at operation time**

### Performance Implications
1. **Zero-copy CPU↔GPU:** Since both processors access the same memory, there are no `cudaMemcpy`-style transfers. This eliminates a major bottleneck seen in discrete-GPU systems.
2. **Lazy evaluation:** MLX builds a deferred computation graph. Arrays are only materialized when `.eval()` is called or their values are read. This enables:
   - Automatic operation fusion
   - Reduced intermediate memory allocations
   - Dynamic graph reshaping without recompilation
3. **CPU+GPU parallelism:** Because memory is shared, the CPU can preprocess (e.g., tokenization, KV cache updates) while the GPU runs matmuls, with automatic stream dependencies managed by the MLX scheduler.
4. **Quantified example:** MLX docs show a mixed CPU/GPU workload running **2× faster** than GPU-only on an M1 Max, because small elementwise ops stay on CPU while large matmuls stay on GPU.

### Streams & Devices
- **`mlx_device`** — `cpu` or `gpu`
- **`mlx_stream`** — async queue; the scheduler inserts dependencies automatically
- **`mlx.synchronize()`** — global barrier when you need host-visible results

---

## 5. Native Quantization Formats

MLX's quantization is **flexible and native** — not limited to a single format like GGUF.

### Supported Bit Widths
MLX supports quantization to **2, 3, 4, 5, 6, and 8 bits** per element, packed into `uint32` arrays.

### Additional Formats
- **MXFP8 / MXFP4** — micro-scaling floating point (hardware-friendly on recent Apple Silicon)
- **FP8** (via `mlx_to_fp8()`)

### Quantization API
```c
int mlx_quantize(mlx_array* res, const mlx_array w, int group_size,
                 int bits, const char* mode, mlx_stream s);
```
- `mode`: `"affine"` (default), `"normal"`, `"lite"`, `"nf4"`, `"fp4"`
- `group_size`: elements sharing one scale/bias
- `bits`: 2–8 (or `None` for default per mode)

### Dequantization
```c
int mlx_dequantize(mlx_array* res, const mlx_array w,
                   const mlx_array scales, const mlx_array biases,
                   mlx_optional_int group_size, mlx_optional_int bits,
                   const char* mode, mlx_optional_dtype dtype, mlx_stream s);
```

### Quantized Matmul
```c
int mlx_quantized_matmul(mlx_array* res, const mlx_array x, const mlx_array w,
                         const mlx_array scales, const mlx_array biases,
                         bool transpose, mlx_optional_int group_size,
                         mlx_optional_int bits, const char* mode, mlx_stream s);
```

### Practical Note
MLX quantization is **not GGUF-compatible out of the box**, but the ecosystem has converters. The `mlx-community` HuggingFace org distributes models in MLX-native format (e.g., `Qwen3.6-27B` in `bf16`, `8bit`, `6bit`, `5bit`, `4bit`, `mxfp8`, `mxfp4`).

---

## 6. MLX vs Metal Performance Shaders (MPS)

### MLX
- **Purpose-built for ML** on Apple Silicon
- **Unified memory native:** zero-copy, lazy graphs, automatic scheduling
- **LLM-optimized fused kernels:** attention, RMS norm, RoPE, quantized matmul
- **Higher-level API:** NumPy-like array ops + fast custom ops
- **Ecosystem:** `mlx-lm` (LLM inference server), `mlx-vlm`, `mlx-whisper`, etc.
- **Performance:** Ollama switched from llama.cpp to MLX and saw **15–30% throughput gains** with **~10% lower memory usage**. vLLM-MLX and Rapid-MLX report even larger wins for batch serving.

### Metal Performance Shaders (MPS / PyTorch MPS backend)
- **General-purpose GPU compute** via Metal
- **Requires explicit copies** from CPU tensors to MPS tensors (PyTorch MPS does this under the hood)
- **No unified memory awareness at framework level:** PyTorch treats MPS as a separate device
- **Less optimized for LLM-specific primitives:** no fused attention kernel, no native quantized matmul
- **Overhead:** PyTorch's eager execution + MPS runtime overhead adds latency, especially for small decode steps

### Concrete Comparison
| Aspect | MLX | PyTorch MPS |
|---|---|---|
| Memory model | Unified (zero-copy) | Separate device buffers |
| Attention | Native fused `scaled_dot_product_attention` | No native fused op |
| Quantized matmul | Native | Not supported |
| Lazy evaluation | Yes (graph fusion) | No (eager) |
| CPU+GPU parallel | Automatic stream scheduling | Manual |
| Decode perf | 15–30% faster than llama.cpp (Apple Silicon) | Slower than MLX for decode |
| Prefill perf | Competitive, fused kernels | Good but higher overhead |

### Gotchas with MLX
1. **macOS-only:** MLX is Apple Silicon-only. There is a CUDA backend experimentally, but the primary value proposition is UMA on Apple Silicon.
2. **Lazy eval surprises:** Values are not computed until `.eval()` or a readback. Debugging requires explicit evaluation.
3. **Autodiff + closures in Rust:** The `mlx-rs` crate has documented differences from Python MLX around closure variable capture and graph tracing — you must explicitly pass arrays to the inputs slice (see `mlx-rs` docs).
4. **Build environment:** `mlx-c` must be installed/built on macOS. Cross-compilation from Linux is not practical because it links Metal.framework.
5. **C API surface is large:** `mlx-c` auto-generates from the C++ API; function signatures can be verbose (e.g., many optional args passed as structs).

---

## 7. Integration Recommendations for Oxidize

### Option A: Use `mlx-rs` (Recommended)
Add to `Cargo.toml`:
```toml
[target.'cfg(target_os = "macos")'.dependencies]
mlx-rs = "0.25"
```

- Wrap MLX tensors in your existing Rust model abstractions
- Use MLX's `fast_scaled_dot_product_attention`, `fast_rms_norm`, `fast_rope` for transformer layers
- Use `quantized_matmul` for weight loading if you support MLX-native quantized checkpoints
- Keep a CPU fallback path for non-macOS builds

### Option B: Raw FFI via `mlx-sys` (If `mlx-rs` is insufficient)
```toml
[target.'cfg(target_os = "macos")'.dependencies]
mlx-sys = "0.2"
```

Use `unsafe` blocks to call `mlx_matmul`, `mlx_fast_scaled_dot_product_attention`, etc. directly. This gives access to every C API function immediately, at the cost of manual memory management.

### Option C: Conditional Compilation
```rust
#[cfg(target_os = "macos")]
mod mlx_backend;
#[cfg(not(target_os = "macos"))]
mod cpu_backend;
```

This mirrors how MLX-aware projects (e.g., `burn-mlx`) are structured.

### Build/System Requirements
- macOS 14+ (Sonoma / Sequoia)
- Xcode Command Line Tools (for Metal compiler)
- Rust 1.82.0+ (MSRV of `mlx-rs`)
- `mlx-c` library must be available at link time (homebrew or built from source)

---

## 8. Concrete API Quick Reference

```c
// Array lifecycle
int mlx_array_new_data(mlx_array* arr, const void* data,
                       const int* shape, size_t ndim, mlx_dtype dtype);
void mlx_array_free(mlx_array arr);

// Device / Stream
int mlx_device_new(mlx_device* dev, mlx_device_type type, int index);
int mlx_stream_new(mlx_stream* stream, mlx_device dev);
void mlx_synchronize(void);

// Matmul
int mlx_matmul(mlx_array* res, const mlx_array a, const mlx_array b,
               const mlx_stream s);
int mlx_addmm(mlx_array* res, const mlx_array c,
              const mlx_array a, const mlx_array b,
              float alpha, float beta, const mlx_stream s);

// Quantized matmul
int mlx_quantized_matmul(mlx_array* res, const mlx_array x, const mlx_array w,
                         const mlx_array scales, const mlx_array biases,
                         bool transpose, mlx_optional_int group_size,
                         mlx_optional_int bits, const char* mode,
                         const mlx_stream s);

// Attention (fused)
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res, const mlx_array q, const mlx_array k, const mlx_array v,
    float scale, const char* mask, const mlx_array key_mask,
    const mlx_array value_mask, const mlx_stream s);

// Fast LLM primitives
int mlx_fast_rms_norm(mlx_array* res, const mlx_array x, const mlx_array w,
                      float eps, const mlx_stream s);
int mlx_fast_rope(mlx_array* res, const mlx_array x, int dims, bool traditional,
                  mlx_optional_float base, int scale,
                  const mlx_array offset, const mlx_stream s);

// Quantization
int mlx_quantize(mlx_array* res, const mlx_array w, int group_size,
                 int bits, const char* mode, mlx_stream s);
int mlx_dequantize(mlx_array* res, const mlx_array w,
                   const mlx_array scales, const mlx_array biases,
                   mlx_optional_int group_size, mlx_optional_int bits,
                   const char* mode, mlx_optional_dtype dtype,
                   const mlx_stream s);
```

---

## 9. Sources & References

1. **MLX C API Docs** — https://ml-explore.github.io/mlx-c/build/html/index.html
2. **MLX C GitHub** — https://github.com/ml-explore/mlx-c
3. **MLX Core GitHub** — https://github.com/ml-explore/mlx
4. **mlx-rs Crate** — https://crates.io/crates/mlx-rs / https://github.com/oxiglade/mlx-rs
5. **mlx-sys Crate** — https://crates.io/crates/mlx-sys
6. **burn-mlx Crate** — https://crates.io/crates/burn-mlx
7. **MLX Unified Memory Docs** — https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
8. **MLX Quantized Matmul Docs** — https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
9. **MLX vs Ollama Benchmarks** — https://willitrunai.com/blog/mlx-vs-ollama-apple-silicon-benchmarks
10. **Ollama MLX Switch** — https://yage.ai/share/mlx-apple-silicon-en-20260331.html
11. **Apple MLX Research (M5 Neural Accelerator)** — https://machinelearning.apple.com/research/exploring-llms-mlx-m5
12. **MLX Swift bindings** (uses mlx-c) — https://github.com/ml-explore/mlx-swift

---

*Report generated 2026-05-22 for the Oxidize Rust LLM inference engine project.*
