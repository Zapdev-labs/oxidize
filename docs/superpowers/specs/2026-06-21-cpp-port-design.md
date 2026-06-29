# oxidize-cpp — C++/CUDA port design

**Date:** 2026-06-21
**Branch:** `cpp-port`
**Author:** Claude (Opus 4.8) + user (Jackson57279 / bryan-wheeler)

## Goal

Port the `oxidize` LLM inference engine to C++/CUDA in `oxidize-cpp/` and benchmark it
against the existing Rust engine on **Modal H100 + A100** to determine whether a C++ host
changes single-GPU token throughput. The Rust workspace stays intact; the port is additive.

> Premise caveat (recorded for honesty): On H100/A100, throughput is dominated by the CUDA
> kernels (GEMM, flash-attention, dequant), not the host language. The benchmark exists to
> *measure* this rather than assume it. Real wins, if any, come from FP8 (H100), CUDA graphs,
> fused kernels, cuBLASLt/CUTLASS GEMM, and reduced per-token sync — which this port enables.

## Scope

Full parity, **staged over many runs**. This run delivers Phase 1.

### Phase 1 — a real, working, benchmarkable Llama hot path
1. `oxidize-cpp/` CMake (C++20 + CUDA), CPU-only build works without nvcc for parity tests.
2. **GGUF loader** — port of `format/gguf.rs`: mmap, metadata KV, tensor table, arch detect.
3. **Quantization / dequant** — Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, F16, BF16 → F32 (port of
   `compute/quantization.rs`), byte-for-byte matching block layouts.
4. **Tensor + CPU backend** — subset of `compute/tensor.rs`: gemv/gemm (quantized + f32),
   rms_norm, rope, swiglu/geglu, softmax. Correctness reference for parity.
5. **CUDA backend** — cuBLASLt/CUTLASS GEMM, fused flash-attention (decode + prefill),
   RoPE, RMSNorm, dequant, KV cache, sampling. FP16 everywhere; **FP8 path gated for H100**.
6. **Llama inference loop** — port of `model/inference.rs` hot path (dense Llama/Mistral/Qwen
   without MoE/shortconv/MLA for Phase 1); produces logits matching Rust.
7. **CLI** `oxidize-cpp run --model X --prompt Y [--cuda]` for the benchmark.
8. **Modal app** (`oxidize-cpp/modal/`) — builds the CUDA image, mounts the model, runs the
   **same model + prompt + token count** through both C++ and Rust `oxidize` on H100 and A100;
   reports tokens/sec (prefill + decode) and latency side by side as JSON + table.

### Later phases (scaffolded, not claimed done)
MoE / Mixtral / DeepSeek-MLA / LFM2 shortconv, paged-attention, server (HTTP/WS),
remaining backends (Metal/Vulkan/ROCm), vision/video, training/prune/merge.

## Architecture / module map (`oxidize-cpp/`)
```
include/oxidize/   config.hpp model.hpp gguf.hpp tensor.hpp quant.hpp backend.hpp
                   cuda_backend.hpp sampler.hpp
src/               gguf.cpp quant.cpp tensor_cpu.cpp model_llama.cpp sampler.cpp main.cpp
src/cuda/          gemm.cu flash_attn.cu rope.cu rmsnorm.cu dequant.cu sampling.cu backend.cu
tests/             parity_test.cpp   (C++ CPU logits vs golden from Rust)
modal/             benchmark.py image.py
CMakeLists.txt
```
Interfaces mirror Rust: `DType`, `ModelArchitecture`, `InferenceConfig`, `GgufQuantizationType`,
`Model{forward, vocab_size, context_size, layer_count, rewind_to}`, `Session`, `Token=uint32`.

## Data flow
GGUF file → mmap → `GgufModel` (metadata + tensor views) → `LlamaModel` (weights, optionally
dequant/upload to GPU) → per-token `forward()` → logits → `Sampler` → next token. Backend
(`CpuBackend` | `CudaBackend`) is injected; same model code drives both.

## Correctness / testing
- **Parity gate (blocks GPU spend):** C++ CPU backend must reproduce Rust logits for the
  benchmark model within tolerance (argmax match + top-k cosine ≥ 0.999) before any Modal run.
- Golden logits captured from the Rust engine on the chosen model/prompt.
- Unit tests per kernel (dequant blocks, rms_norm, rope) vs Rust reference values.

## Error handling
`OxidizeError` enum mirroring `ModelError`/`GgufError`; `Result`-style via `expected<T>`-like
return or exceptions at the CLI boundary only. CUDA errors checked via `CUDA_CHECK` macro.

## Benchmark model
User supplies a GGUF they already have (path/HF repo) — captured at Modal-wiring step.

## Build / run
```
cmake -B build -DOXIDIZE_CUDA=ON && cmake --build build -j     # GPU (on Modal)
cmake -B build -DOXIDIZE_CUDA=OFF && cmake --build build -j    # CPU parity (local)
modal run oxidize-cpp/modal/benchmark.py --model <gguf>        # H100+A100 A/B vs Rust
```
