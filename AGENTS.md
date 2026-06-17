# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-03
**Commit:** d9fb076 (master)
**Workspace:** Local-first LLM inference in Rust with Go and Python ports

## OVERVIEW
This workspace contains the core Rust LLM inference engine (`oxidize-core`) and multiple frontends/bindings (CLI, server, Python bindings), plus parallel language ports in Go and pure Python for cross-platform deployment.

## STRUCTURE
```
.
├── oxidize-core/      # Rust core: GGUF, tensors, quantization, generation, backends
│   ├── src/backends/  # CUDA, Metal, Vulkan, MLX, WebGPU (see backends/AGENTS.md)
│   ├── src/compute/   # Tensor ops, KV cache, flash attention, quantization (see compute/AGENTS.md)
│   ├── src/format/    # GGUF, SafeTensors, tokenizer (see format/AGENTS.md)
│   ├── src/mesh/      # Distributed inference (see mesh/AGENTS.md)
│   ├── src/model/     # Inference engine, sampling, DFlash (see model/AGENTS.md)
│   ├── src/paged_attention/ # vLLM-style paging scheduler (see paged_attention/AGENTS.md)
│   └── src/vision/    # Vision encoder / multimodal
├── oxidize-cli/       # Prompt/chat CLI, profiling, pipeline modes (see AGENTS.md)
├── oxidize-server/    # OpenAI-compatible HTTP API (axum) (see src/AGENTS.md)
├── oxidize-quantize/  # Offline weight conversion (see AGENTS.md)
├── oxidize-py/        # Python bindings (pyo3 + maturin) (see AGENTS.md)
├── oxidize-train/     # CSV classifier training
├── oxidize-golang/    # Go port of oxidize-core (see AGENTS.md)
├── oxidize-python/    # Pure-Python port (see AGENTS.md)
└── scripts/           # CI benchmark regression + dashboard
```

## SUBDIRECTORY AGENTS.md MAP
| Directory | File | Domain |
|-----------|------|--------|
| `oxidize-core/src/compute/` | `compute/AGENTS.md` | CPU tensor ops, quantization, KV cache, flash attention |
| `oxidize-core/src/model/` | `model/AGENTS.md` | Inference engine, model loading, speculative decoding |
| `oxidize-core/src/mesh/` | `mesh/AGENTS.md` | Distributed inference (libp2p mesh) |
| `oxidize-core/src/backends/` | `backends/AGENTS.md` | Hardware compute backends |
| `oxidize-core/src/format/` | `format/AGENTS.md` | GGUF, SafeTensors, tokenizer |
| `oxidize-core/src/paged_attention/` | `paged_attention/AGENTS.md` | vLLM-style PagedAttention scheduler |
| `oxidize-server/src/` | `src/AGENTS.md` | OpenAI-compatible HTTP API (Axum) |
| `oxidize-cli/` | `AGENTS.md` | CLI for prompt/chat, benchmarking |
| `oxidize-quantize/` | `AGENTS.md` | Offline weight quantization utility |
| `oxidize-py/` | `AGENTS.md` | PyO3 Python bindings |
| `oxidize-golang/` | `AGENTS.md` | Go port of oxidize-core |
| `oxidize-python/` | `AGENTS.md` | Pure-Python port |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `ComputeBackend` | trait | `oxidize-core/src/backend.rs` | Abstraction all backends implement |
| `Model` | trait | `oxidize-core/src/model.rs` | Implemented by 5 structs (Inference, Llama, LayerWise, MLX, DFlash) |
| `GgufQuantizationType` | enum | `oxidize-core/src/format/gguf.rs` | Central type hub; 20+ cross-module refs |
| `tensor.rs` | module | `oxidize-core/src/compute/` | 5,153 lines; 135 unsafe blocks; SIMD kernels |
| `scheduler.rs` | module | `oxidize-core/src/paged_attention/` | vLLM-style request scheduling |
| `app.rs` | module | `oxidize-server/src/` | Axum route assembly |

## WHERE TO LOOK (High-Level)
| Task | Location | Notes |
|------|----------|-------|
| Add model architecture | `oxidize-core/src/model/inference.rs` | Extend `ModelArchitecture` enum |
| Add backend | `oxidize-core/src/backends/` | Implement `ComputeBackend` trait, add `XxxBuildInfo` |
| Add quantization type | `oxidize-core/src/compute/quantization.rs` | Also update `GgufQuantizationType` in `format/gguf.rs` |
| Tokenizer change | `oxidize-core/src/format/tokenizer.rs` | 4 formats: SP, WordPiece, BPE, Tiktoken |
| Server route | `oxidize-server/src/routes/` | OpenAI-compatible endpoints |
| CLI subcommand | `oxidize-cli/src/main.rs` | Also check `src/bin/` for aux tools |
| Distributed logic | `oxidize-core/src/mesh/` | Only dir with real `mod.rs` + privacy boundaries |
| Port to Go | `oxidize-golang/` | Mirror Rust structure; see `oxidize-golang/AGENTS.md` |
| Port to Python | `oxidize-python/` | Mirror Go structure; see `oxidize-python/AGENTS.md` |
| Wanda pruning | `oxidize-prune/src/wanda.rs` | Per-output-row `|W| · ‖X‖_2`; see `oxidize-prune/AGENTS.md` |
| Magnitude pruning | `oxidize-prune/src/mask.rs` + `wanda.rs` | Per-output-row `|W|`; per Wanda paper, the right default for LLMs |
| Activation L2 norms (Wanda calibration) | `oxidize-core/src/compute/activation_stats.rs` | `ActivationStats` + `CalibrationRunner`; consumed by `oxidize-prune` |
| Auto-detect + auto-tune | `oxidize-core/src/autotune/` | `detect()` (CPU/RAM/NUMA/GPU/ISA) + `fingerprint()` + `plan()` rule table; CLI flags `--auto --no-auto --print-plan` |
| Skylake-SP detection (AVX-512 regression gate) | `oxidize-kernels/src/cpu.rs` | `pub fn is_skylake_sp() -> bool` |

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
- Stop trying to use C for tasks that can be done in rust (Claude)
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
- GGUF/SafeTensors draft-model loading + speculative generation summarizing is active development area.

## Learned User Preferences
- When adding `oxidize-python` or expanding `oxidize-golang`, keep all Rust crates and features; do not delete or replace the Rust workspace.
- Parallel language ports should reach feature parity with `oxidize-core` (user asked for every Rust feature in Python/Go, with Python targeting similar CLOC to Rust).
- Keep `oxidize-py` (PyO3/maturin bindings) alongside the pure-Python `oxidize-python` package.
- When extending Go/Python ports, implement in `oxidize-golang` first, mirror to `oxidize-python`, and sync new `master` Rust features rather than leaving ports stale.
- For Go/Python GPU backends, use pure native implementations (no Rust FFI/CGO at runtime); CUDA first, then Vulkan/Metal/WebGPU.
- Avoid creating extra markdown documentation files unless asked; update README when needed.
- On feature branches, stage and commit only files related to the task; exclude unrelated workspace changes.
- `oxidize run <model>` should start the OpenAI-compatible HTTP/WebSocket server by default; use `--no-api` for local inference only.
- Contributions should keep tests passing and use clear, ethical PR/markdown descriptions; include benchmarks when claiming performance changes.
- When a user asks to run a model. It means run it using oxidize 
- Prefer building and testing over starting development servers unless the user explicitly asks to run or serve.
## Learned Workspace Facts
- `oxidize-golang/` is the active Go port of `oxidize-core`; CLI lives in `internal/cli/` (`run`, `chat`, `bench`, `inspect`, `list`, `serve`); HF GGUF resolver in `hf/`.
- `oxidize-python/` is a pure-Python implementation (`oxidize_python`, `pyproject.toml`, uv/pytest); CLI mirrors Go subcommands; HF resolver in `oxidize_python/hf/hub.py` with cache `~/.cache/oxidize/hf`.
- Do not modify Rust crates when extending `oxidize-python`; port from `oxidize-golang` or Rust sources.
- `oxidize-py/` is the PyO3 bindings crate, separate from `oxidize-python`.
- Go and Python port tests reuse GGUF fixtures under `oxidize-core/tests/fixtures/` (e.g. `valid-v3.gguf`).
- DFlash speculative decoding in `oxidize-core/src/model/dflash.rs` is an active port target for `oxidize-golang` (and downstream Python); inference needs a compatible target GGUF paired with the draft (hidden-size mismatch falls back to target-only).
- Rust `oxidize run` rewrites to `--serve-api` by default (background in-process server on `--api-host`/`--api-port`); realtime WebSocket at `ws://HOST:PORT/v1/realtime` (`oxidize-server/tests/realtime_ws.rs`).
- `oxidize-convert` converts HuggingFace SafeTensors (file or model directory with `config.json`) to GGUF; core logic in `oxidize-core/src/format/safetensors_to_gguf.rs`.
- Git installs must name `oxidize-cli` explicitly (`cargo install --git … oxidize-cli --bin oxidize`) because the workspace ships multiple binary crates.
- `oxidize-prune` depends on `oxidize-kernels` for SIMD magnitude/Wanda masks (`prune.rs`), Q4_K dequant (`q4k_dequant.rs`), and rayon-parallel tensor processing in `wanda.rs`.
- Both Go and Python ports include `core/autotune/` with `--auto`, `--no-auto`, and `--print-plan` CLI flags.
- Run Go port tests with `CGO_ENABLED=0` (exclude `scripts` package); Python tests via `uv run pytest` (`OXIDIZE_SLOW_TESTS=1` for slow GGUF integrations).
