# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-26
**Commit:** d9fb076
**Branch:** master

## OVERVIEW
Local-first LLM inference workspace in Rust. Core compute (`oxidize-core`) with CLI, server, Python bindings, and quantization utility frontends.

## STRUCTURE
```
.
├── oxidize-core/     # GGUF parsing, tensors, quantization, generation, backends
│   ├── src/backends/ # CUDA, Metal, Vulkan, MLX, WASM, WebGPU
│   ├── src/compute/  # Tensor ops, KV cache, flash attention, quantization
│   ├── src/format/   # GGUF, SafeTensors, tokenizer
│   ├── src/mesh/     # Distributed inference (submodule hierarchy)
│   ├── src/model/    # Inference engine, sampling, DFlash speculative decoding
│   ├── src/paged_attention/ # vLLM-style paging scheduler
│   └── src/vision/   # Vision encoder / multimodal
├── oxidize-cli/      # Prompt/chat CLI, profiling, pipeline modes
├── oxidize-server/   # OpenAI-compatible HTTP API (axum)
├── oxidize-quantize/ # Offline weight conversion
├── oxidize-py/       # Python bindings (pyo3 + maturin)
├── oxidize-train/    # CSV classifier training
└── scripts/          # CI benchmark regression + dashboard
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add model architecture | `oxidize-core/src/model/inference.rs` | Extend `ModelArchitecture` enum |
| Add backend | `oxidize-core/src/backends/` | Implement `ComputeBackend` trait, add `XxxBuildInfo` |
| Add quantization type | `oxidize-core/src/compute/quantization.rs` | Also update `GgufQuantizationType` in `format/gguf.rs` |
| Tokenizer change | `oxidize-core/src/format/tokenizer.rs` | 4 formats: SP, WordPiece, BPE, Tiktoken |
| Server route | `oxidize-server/src/routes/` | OpenAI-compatible endpoints |
| CLI subcommand | `oxidize-cli/src/main.rs` | Also check `src/bin/` for aux tools |
| Distributed logic | `oxidize-core/src/mesh/` | Only dir with real `mod.rs` + privacy boundaries |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `ComputeBackend` | trait | `oxidize-core/src/backend.rs` | Abstraction all backends implement |
| `Model` | trait | `oxidize-core/src/model.rs` | Implemented by 5 structs (Inference, Llama, LayerWise, MLX, DFlash) |
| `GgufQuantizationType` | enum | `oxidize-core/src/format/gguf.rs` | Central type hub; 20+ cross-module refs |
| `tensor.rs` | module | `oxidize-core/src/compute/` | 5,153 lines; 135 unsafe blocks; SIMD kernels |
| `scheduler.rs` | module | `oxidize-core/src/paged_attention/` | vLLM-style request scheduling |
| `app.rs` | module | `oxidize-server/src/` | Axum route assembly |

## CONVENTIONS
- **Flat module system**: `lib.rs` uses `#[path = "..."]` to flatten all modules into crate root. Only `mesh/`, `paged_attention/`, `vision/` have real `mod.rs` files.
- **Config + Error + Trait trinity**: Every subsystem has `XxxConfig`, `XxxError`, and core trait/struct.
- **Error chaining**: All errors wrap lower-level errors via `From` impls.
- **Backend dual-file**: `vulkan.rs` + `vulkan_stub.rs` pair (only backend with this pattern).
- **Build info micro-pattern**: Every backend exposes `XxxBuildInfo` + `xxx_build_info()` for compile-time detection.
- **Test co-location**: Every `.rs` file has `#[cfg(test)]` module at bottom; no separate `tests/` inside `src/`.

## ANTI-PATTERNS (THIS PROJECT)
- `StdMutex` in async context (`oxidize-server/src/runtime/paged.rs`) — should be `tokio::sync::Mutex`.
- `tensor.rs` monolith — 5,153 lines mixing kernels, types, and ops. Refactor candidate.
- Quantization constants shadowed in `tensor.rs` and `cuda.rs` — should be shared.
- `unwrap()/expect()` proliferation — 1000+ instances in non-test code.

## UNIQUE STYLES
- **Bottom-up file organization** (`tensor.rs`): constants → errors → low-level kernels → high-level functions → `Tensor` struct (inverse of typical Rust).
- **WASM worker type embedding**: `util/web_worker.rs` embeds complete TypeScript interface contracts as 60+ line string literals.
- **MLX macOS fortress**: `mlx.rs` and `mlx_inference.rs` are heavily `#[cfg(target_os = "macos")]` gated.

## COMMANDS
```bash
# Build / test / lint
make build    # release build
make test     # workspace tests
make lint     # clippy -D warnings
make fmt      # format check
make ci       # full CI equivalent

# Run
sfw cargo run -p oxidize-cli -- --prompt "hello"
sfw cargo run -p oxidize-server -- --host 127.0.0.1 --port 8080
sfw cargo run -p oxidize-quantize -- --input in.bin --output out.bin --source F32 --target Q4_0

# WASM
make wasm     # outputs to dist/wasm
```

## NOTES
- Rust edition 2024, resolver "3".
- Release profile: `lto = true`, `panic = "abort"`.
- `cargo-deny` audits licenses + security (see `deny.toml`).
- `.cargo/config.toml` sets custom linker for `aarch64-unknown-linux-gnu` and WASM runner.
- `oxidize-core/fuzz/` exists but is NOT in workspace members/exclude.
- `models/` is gitignored but contains tracked files.
- GGUF/SafeTensors draft-model loading + speculative generation (DFlash) is active development area.
