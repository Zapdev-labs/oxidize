# Product Requirements Document (PRD)

## Project: oxidize — A Rust-Based LLM Inference Engine

**Date:** April 30, 2026
**Status:** Draft v0.1
**Target Language:** Rust
**Inspiration:** llama.cpp by Georgi Gerganov

---

## 1. Executive Summary

Build a high-performance, dependency-light LLM inference engine in Rust that runs large language models on commodity hardware (CPUs, GPUs, Apple Silicon) using quantization and modern system programming techniques.

**Key Differentiators:**
- Zero-cost abstractions via Rust's ownership model
- Memory safety without GC overhead
- First-class async/concurrency support
- Native WebAssembly support for browser inference
- Modern crate ecosystem (burn, candle, etc.)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│  CLI │ HTTP Server │ Python Bindings │ WASM │ FFI          │
├─────────────────────────────────────────────────────────────┤
│                    API LAYER                                │
│  Session Management │ Sampling │ Tokenization │ Scheduling  │
├─────────────────────────────────────────────────────────────┤
│                    COMPUTE LAYER                            │
│  CPU Kernels (AVX2/AVX512/NEON) │ GPU (CUDA/HIP/Vulkan)    │
│  Quantization │ Dequantization │ Matrix Multiplication     │
├─────────────────────────────────────────────────────────────┤
│                    MODEL LAYER                              │
│  GGUF Loader │ Model Graph │ Weight Storage │ KV Cache      │
├─────────────────────────────────────────────────────────────┤
│                    HARDWARE ABSTRACTION                     │
│  CPU │ NVIDIA (CUDA) │ AMD (HIP) │ Apple (Metal) │ WASM     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Modules & TODOs

### MODULE 1: Project Foundation & Build System

**Objective:** Establish Rust project structure with cross-platform compilation support

- [x] **TODO-1.1:** Initialize Cargo workspace with workspace-level dependencies
  ```toml
  [workspace]
  members = ["oxidize-core", "oxidize-cli", "oxidize-server", "oxidize-quantize"]
  resolver = "3"
  ```
- [x] **TODO-1.2:** Set up CI/CD (GitHub Actions) for Linux, macOS, Windows builds
- [x] **TODO-1.3:** Configure cross-compilation for ARM64, WASM32 targets
- [x] **TODO-1.4:** Set up benchmark harness with `criterion.rs`
- [x] **TODO-1.5:** Create Docker images for deployment
- [x] **TODO-1.6:** Add `justfile`/`Makefile` for common tasks
- [x] **TODO-1.7:** Set up `cargo deny` for license/security auditing
- [x] **TODO-1.8:** Configure `release` profile with LTO and panic=abort

**Estimated Effort:** 2-3 days
**Priority:** P0 (Blocking)

---

### MODULE 2: GGUF Format & Model Loader

**Objective:** Parse and load GGUF (Georgi Gerganov Universal Format) files

- [x] **TODO-2.1:** Implement GGUF file format parser
  - Magic number validation (`GGUF`)
  - Version handling (v2, v3)
  - Tensor info metadata parsing
  - Alignment and padding handling
- [x] **TODO-2.2:** Create `Tensor` struct with shape, strides, dtype
- [x] **TODO-2.3:** Implement memory-mapped file loading (`memmap2` crate)
- [x] **TODO-2.4:** Support tensor name mapping for different architectures
- [x] **TODO-2.5:** Add quantization type detection (Q4_0, Q4_K_M, Q5_K_M, Q8_0, etc.)
- [x] **TODO-2.6:** Implement `ModelLoader` trait for extensibility
- [x] **TODO-2.7:** Add progress callbacks for large model loading
- [x] **TODO-2.8:** Create comprehensive unit tests with fixture files
- [x] **TODO-2.9:** Benchmark loader against llama.cpp baseline

**Key Crates:** `memmap2`, `bytemuck`, `half`
**Estimated Effort:** 5-7 days
**Priority:** P0 (Blocking)

---

### MODULE 3: Quantization Engine

**Objective:** Implement quantization/dequantization schemes matching llama.cpp

- [x] **TODO-3.1:** Implement scalar dequantization kernels:
  - Q4_0, Q4_1 (4-bit with/without offsets)
  - Q5_0, Q5_1 (5-bit variants)
  - Q8_0 (8-bit)
  - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (K-quants)
- [x] **TODO-3.2:** Implement dequantization to f16 and f32
- [x] **TODO-3.3:** Add quantization from f16/f32 to all supported formats
- [x] **TODO-3.4:** Implement block-wise quantization with per-block scales
- [x] **TODO-3.5:** Add importance matrix support (IMatrix) for better quality
- [x] **TODO-3.6:** Create quantization CLI tool (`oxidize-quantize`)
- [x] **TODO-3.7:** Add mixed quantization support (different types per layer)
- [x] **TODO-3.8:** Validate output against llama.cpp reference implementation

**Key Traits:**
```rust
pub trait Quantization {
    fn quantize(&self, data: &[f32], output: &mut [u8]) -> Result<()>;
    fn dequantize(&self, data: &[u8], output: &mut [f32]) -> Result<()>;
    fn block_size(&self) -> usize;
    fn type_size(&self) -> usize;
}
```

**Estimated Effort:** 10-14 days
**Priority:** P0 (Blocking)

---

### MODULE 4: Compute Kernels — CPU

**Objective:** High-performance CPU inference kernels with SIMD optimization

- [x] **TODO-4.1:** Set up SIMD abstraction layer
  - x86: SSE2, AVX, AVX2, AVX512 (via `std::arch`)
  - ARM: NEON (via `std::arch`)
  - Fallback: scalar implementations
- [x] **TODO-4.2:** Implement matrix-vector multiplication (GEMV)
  - F32, F16 input types
  - Quantized weights with on-the-fly dequantization
- [x] **TODO-4.3:** Implement matrix-matrix multiplication (GEMM) for batching
- [x] **TODO-4.4:** Implement attention mechanisms:
  - Multi-head attention (MHA)
  - Grouped-query attention (GQA)
  - Multi-query attention (MQA)
- [x] **TODO-4.5:** Implement RoPE (Rotary Position Embedding)
- [x] **TODO-4.6:** Implement SwiGLU activation
- [x] **TODO-4.7:** Implement RMSNorm and LayerNorm
- [x] **TODO-4.8:** Implement Softmax (stable, numerically accurate)
- [x] **TODO-4.9:** Add thread pool for parallel layer execution
- [x] **TODO-4.10:** Optimize cache locality and prefetching
- [x] **TODO-4.11:** Add runtime CPU feature detection

**Performance Target:** Within 10% of llama.cpp CPU performance
**Estimated Effort:** 15-20 days
**Priority:** P0 (Blocking)

---

### MODULE 5: Compute Kernels — GPU (CUDA)

**Objective:** CUDA kernels for NVIDIA GPU acceleration

- [x] **TODO-5.1:** Set up CUDA build pipeline with `rust-cuda` or `cust`
- [x] **TODO-5.2:** Implement memory management (device allocation, H2D/D2H transfers)
- [x] **TODO-5.3:** Port GEMV kernels to CUDA
- [x] **TODO-5.4:** Port GEMM kernels using cuBLAS
- [x] **TODO-5.5:** Implement attention kernels (flash attention style)
- [x] **TODO-5.6:** Implement quantization-aware kernels (dequantize on GPU)
- [x] **TODO-5.7:** Add kernel fusion (combine multiple ops into single kernel)
- [x] **TODO-5.8:** Implement layer offloading (--n-gpu-layers equivalent)
- [x] **TODO-5.9:** Add multi-GPU support (tensor/pipeline parallelism)
- [x] **TODO-5.10:** Optimize memory usage with Flash Attention

**Key Crates:** `cust`, `cudarc`
**Estimated Effort:** 20-25 days
**Priority:** P1 (High)

---

### MODULE 6: Compute Kernels — Apple Metal

**Objective:** Metal Performance Shaders for Apple Silicon

- [x] **TODO-6.1:** Set up Metal build with `metal-rs`
- [x] **TODO-6.2:** Implement buffer management for unified memory
- [x] **TODO-6.3:** Port compute kernels to Metal Shading Language
- [x] **TODO-6.4:** Optimize for Apple Silicon unified memory architecture
- [x] **TODO-6.5:** Add Metal Performance Shaders integration where beneficial

**Estimated Effort:** 10-12 days
**Priority:** P1 (High)

---

### MODULE 7: Model Architectures

**Objective:** Support multiple transformer architectures

- [x] **TODO-7.1:** Define `Model` trait:
  ```rust
  pub trait Model {
      fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits>;
      fn vocab_size(&self) -> usize;
      fn context_size(&self) -> usize;
      fn layer_count(&self) -> usize;
  }
  ```
- [x] **TODO-7.2:** Implement LLaMA architecture (LLaMA 2, LLaMA 3)
- [x] **TODO-7.3:** Implement Mistral architecture
- [x] **TODO-7.4:** Implement Mixtral MoE architecture
- [x] **TODO-7.5:** Implement Qwen architecture
- [x] **TODO-7.6:** Implement Gemma architecture
- [x] **TODO-7.7:** Implement Falcon architecture
- [x] **TODO-7.8:** Implement GPT architecture (GPT-2, GPT-J, GPT-NeoX)
- [x] **TODO-7.9:** Implement Phi architecture
- [x] **TODO-7.10:** Implement architecture auto-detection from GGUF metadata
- [x] **TODO-7.11:** Add LoRA/QLoRA support

**Estimated Effort:** 15-20 days
**Priority:** P0 (Blocking)

---

### MODULE 8: KV Cache Management

**Objective:** Efficient key-value cache for attention

- [x] **TODO-8.1:** Implement KV cache storage with configurable dtype
- [x] **TODO-8.2:** Add sliding window attention cache management
- [x] **TODO-8.3:** Implement cache eviction strategies
- [x] **TODO-8.4:** Add cache quantization (8-bit, 4-bit KV cache)
- [x] **TODO-8.5:** Support continuous batching with cache management
- [x] **TODO-8.6:** Add cache persistence across sessions
- [x] **TODO-8.7:** Optimize memory layout for cache access patterns

**Estimated Effort:** 5-7 days
**Priority:** P1 (High)

---

### MODULE 9: Tokenization

**Objective:** Text-to-tokens and tokens-to-text conversion

- [x] **TODO-9.1:** Implement Byte-Pair Encoding (BPE)
- [x] **TODO-9.2:** Implement SentencePiece (Unigram)
- [x] **TODO-9.3:** Implement WordPiece
- [x] **TODO-9.4:** Add Tiktoken (GPT-4/Claude style)
- [x] **TODO-9.5:** Create tokenizer loader from GGUF metadata
- [x] **TODO-9.6:** Add special token handling (BOS, EOS, PAD, etc.)
- [x] **TODO-9.7:** Implement streaming detokenization
- [x] **TODO-9.8:** Add chat template processing
- [x] **TODO-9.9:** Support token healing (merge incomplete tokens)

**Key Crates:** `tokenizers` (Hugging Face), `tiktoken-rs`
**Estimated Effort:** 5-7 days
**Priority:** P0 (Blocking)

---

### MODULE 10: Sampling & Generation

**Objective:** Text generation with various sampling strategies

- [x] **TODO-10.1:** Implement basic sampling:
  - Greedy
  - Temperature scaling
  - Top-k
  - Top-p (nucleus)
  - Min-p
- [x] **TODO-10.2:** Implement advanced sampling:
  - Mirostat
  - Typical sampling
  - Tail-free sampling
  - Locally typical sampling
- [x] **TODO-10.3:** Implement repetition penalties:
  - Frequency penalty
  - Presence penalty
  - Penalize newlines
- [x] **TODO-10.4:** Add grammar-based constrained generation
- [x] **TODO-10.5:** Implement speculative decoding
- [x] **TODO-10.6:** Add beam search
- [x] **TODO-10.7:** Implement streaming generation (async iterator)

**Estimated Effort:** 7-10 days
**Priority:** P1 (High)

---

### MODULE 11: CLI Application

**Objective:** Command-line interface for inference

- [x] **TODO-11.1:** Create `oxidize-cli` binary
- [x] **TODO-11.2:** Implement argument parsing (`clap`)
  - Model path
  - Prompt (interactive, file, stdin)
  - Context size
  - Thread count
  - GPU layers
  - Sampling parameters
  - System prompt
- [x] **TODO-11.3:** Add interactive chat mode (REPL)
- [x] **TODO-11.4:** Implement single-shot inference mode
- [x] **TODO-11.5:** Add conversation history management
- [x] **TODO-11.6:** Implement progress indicators for loading/generation
- [x] **TODO-11.7:** Add token/speed reporting
- [x] **TODO-11.8:** Support prompt caching
- [x] **TODO-11.9:** Add multi-line input support
- [x] **TODO-11.10:** Implement reverse prompt (stop sequences)

**Estimated Effort:** 5-7 days
**Priority:** P1 (High)

---

### MODULE 12: HTTP Server & API

**Objective:** OpenAI-compatible HTTP API server

- [x] **TODO-12.1:** Create `oxidize-server` binary with `axum` or `actix-web`
- [x] **TODO-12.2:** Implement OpenAI-compatible endpoints:
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `GET /v1/models`
  - `POST /v1/embeddings`
- [x] **TODO-12.3:** Add Server-Sent Events (SSE) for streaming
- [x] **TODO-12.4:** Implement JSON mode and structured output
- [x] **TODO-12.5:** Add request/response logging
- [x] **TODO-12.6:** Implement rate limiting and request queuing
- [x] **TODO-12.7:** Add health check endpoints
- [x] **TODO-12.8:** Support concurrent request handling
- [x] **TODO-12.9:** Add authentication middleware (API keys)
- [x] **TODO-12.10:** Create OpenAPI documentation

**Estimated Effort:** 7-10 days
**Priority:** P1 (High)

---

### MODULE 13: Python Bindings

**Objective:** Python interface via PyO3

- [x] **TODO-13.1:** Set up `pyo3` workspace
- [x] **TODO-13.2:** Create `oxidize` Python package
- [x] **TODO-13.3:** Implement `Llama` class with methods:
  - `__init__`
  - `generate`
  - `create_chat_completion`
  - `embed`
- [x] **TODO-13.4:** Add async support with `asyncio`
- [x] **TODO-13.5:** Support `numpy` and `torch` tensor interop
- [x] **TODO-13.6:** Create `pip` installable wheels (maturin)
- [x] **TODO-13.7:** Add Python type stubs
- [x] **TODO-13.8:** Match `llama-cpp-python` API for compatibility

**Estimated Effort:** 7-10 days
**Priority:** P2 (Medium)

---

### MODULE 14: WebAssembly Support

**Objective:** Browser-based inference

- [x] **TODO-14.1:** Set up `wasm-bindgen` build
- [x] **TODO-14.2:** Implement WebGPU compute backend
- [x] **TODO-14.3:** Add Web Worker support for background inference
- [x] **TODO-14.4:** Create JavaScript/TypeScript bindings
- [x] **TODO-14.5:** Implement streaming generation in browser
- [x] **TODO-14.6:** Add model download/cache management in browser
- [x] **TODO-14.7:** Create demo web application

**Estimated Effort:** 10-14 days
**Priority:** P2 (Medium)

---

### MODULE 15: Performance Optimization

**Objective:** Achieve llama.cpp-level performance

- [x] **TODO-15.1:** Profile CPU inference with `perf`/`samply`
- [x] **TODO-15.2:** Optimize memory access patterns
- [x] **TODO-15.3:** Implement operator fusion (combine linear + activation)
- [x] **TODO-15.4:** Add INT8/INT4 GEMM via `gemm` crate or custom kernels
- [x] **TODO-15.5:** Implement Flash Attention for long contexts
- [x] **TODO-15.6:** Add continuous batching for server throughput
- [x] **TODO-15.7:** Implement pipeline parallelism for multi-GPU
- [x] **TODO-15.8:** Add tensor parallelism for large models
- [x] **TODO-15.9:** Optimize prompt processing (prefill) with batching
- [x] **TODO-15.10:** Add memory pool allocator to reduce allocations

**Performance Targets:**
- CPU: Within 15% of llama.cpp
- GPU: Within 20% of llama.cpp
- Memory usage: Comparable to llama.cpp

**Estimated Effort:** Ongoing (20+ days)
**Priority:** P1 (High)

---

### MODULE 16: Testing & Quality Assurance

**Objective:** Comprehensive test coverage and benchmarking

- [x] **TODO-16.1:** Unit tests for all quantization schemes
- [x] **TODO-16.2:** Numerical accuracy tests (vs. PyTorch reference)
- [x] **TODO-16.3:** Integration tests with real GGUF models
- [x] **TODO-16.4:** Benchmark suite comparing to llama.cpp
- [x] **TODO-16.5:** Perplexity benchmarks on standard datasets
- [x] **TODO-16.6:** Memory usage benchmarks
- [x] **TODO-16.7:** Create CI benchmarks with regression detection
- [x] **TODO-16.8:** Add fuzzing for parser and tokenizer
- [x] **TODO-16.9:** Create benchmark dashboard
- [x] **TODO-16.10:** Add model compatibility tests (run 100+ models)

**Estimated Effort:** Ongoing (10+ days)
**Priority:** P1 (High)

---

### MODULE 17: Documentation & Examples

**Objective:** Excellent developer and user experience

- [x] **TODO-17.1:** Write comprehensive README with quick start
- [x] **TODO-17.2:** Create API documentation with `rustdoc`
- [x] **TODO-17.3:** Write architecture documentation
- [x] **TODO-17.4:** Create quantization guide
- [x] **TODO-17.5:** Write performance tuning guide
- [x] **TODO-17.6:** Create examples:
  - Basic inference
  - Chat completion
  - Streaming generation
  - Batch processing
  - Custom sampling
  - Embedding extraction
- [x] **TODO-17.7:** Add troubleshooting guide
- [x] **TODO-17.8:** Create contribution guidelines
- [x] **TODO-17.9:** Write blog post announcing release

**Estimated Effort:** 5-7 days
**Priority:** P2 (Medium)

---

## 4. Technology Stack

| Component | Primary Choice | Alternatives |
|-----------|---------------|--------------|
| Build System | Cargo | Bazel |
| Async Runtime | Tokio | async-std |
| CLI Framework | clap | structopt |
| Web Server | axum | actix-web, rocket |
| Serialization | serde | - |
| CUDA Bindings | cudarc | rustacuda, cust |
| Metal Bindings | metal-rs | - |
| Python Bindings | pyo3 + maturin | - |
| WASM | wasm-bindgen | - |
| Logging | tracing | log |
| Error Handling | thiserror + anyhow | - |
| Testing | built-in + criterion | - |
| Quantization | custom | - |
| BLAS | intel-mkl-src, openblas-src | gemm |

---

## 5. Development Phases

### Phase 1: Foundation (Weeks 1-3)
- TODO-1.x: Project setup
- TODO-2.x: GGUF loader
- TODO-3.x: Basic quantization
- TODO-7.1, 7.2: LLaMA architecture

**Deliverable:** Load and run LLaMA 2/3 models on CPU

### Phase 2: Core Inference (Weeks 4-6)
- TODO-4.x: CPU kernels
- TODO-8.x: KV cache
- TODO-9.x: Tokenization
- TODO-10.x: Sampling
- TODO-11.x: CLI

**Deliverable:** Full CLI with chat mode, competitive CPU performance

### Phase 3: GPU Acceleration (Weeks 7-9)
- TODO-5.x: CUDA support
- TODO-6.x: Metal support
- TODO-15.x: Performance optimization

**Deliverable:** GPU inference matching llama.cpp speeds

### Phase 4: Production Features (Weeks 10-12)
- TODO-12.x: HTTP server
- TODO-13.x: Python bindings
- TODO-7.3-7.9: More architectures
- TODO-16.x: Testing

**Deliverable:** Production-ready with server and Python API

### Phase 5: Advanced Features (Weeks 13-16)
- TODO-14.x: WASM
- TODO-10.5: Speculative decoding
- TODO-15.6-15.8: Advanced parallelism
- TODO-7.10-7.11: LoRA, more models

**Deliverable:** Full feature parity with llama.cpp + Rust advantages

---

## 6. Success Metrics

| Metric | Target |
|--------|--------|
| Models Supported | 50+ GGUF architectures |
| CPU Performance | Within 15% of llama.cpp |
| GPU Performance | Within 20% of llama.cpp |
| Memory Safety | Zero memory leaks (verified by valgrind/MIRI) |
| Test Coverage | >80% line coverage |
| Binary Size | <50MB for CLI (release) |
| Startup Time | <2s for 7B model |
| Token Throughput | Match or exceed llama.cpp per watt |

---

## 7. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CUDA kernel performance gap | Medium | High | Use cuBLAS where possible, profile extensively |
| Quantization accuracy loss | Low | High | Validate against reference, use IMatrix |
| Memory overhead vs C++ | Medium | Medium | Zero-copy design, careful allocation |
| Build complexity (CUDA deps) | High | Medium | Feature flags, optional GPU backends |
| Compilation time | High | Low | Workspace organization, sccache |

---

## 8. Open Questions

1. Should we use `candle` or `burn` crates for tensor operations, or implement custom?
2. How to handle CUDA build in CI? (GitHub Actions has limited GPU runners)
3. Should we support GGML format legacy loading?
4. What's the minimum Rust version to support?
5. How to handle model downloads and Hugging Face integration?

---

## 9. References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Reference implementation
- [GGUF Format Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [The Rust Programming Language](https://doc.rust-lang.org/book/)
- [Rust SIMD Guide](https://rust-lang.github.io/packed_simd/perf-guide/)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

---

**Next Steps:**
1. Create GitHub repository and initialize workspace
2. Start with TODO-1.1 (workspace setup)
3. Implement TODO-2.1 (GGUF parser) as first milestone
4. Set up benchmark harness to compare against llama.cpp baseline

**Estimated Total Effort:** 4-5 months for MVP, 6-8 months for full feature parity
**Team Size:** 2-3 developers recommended
