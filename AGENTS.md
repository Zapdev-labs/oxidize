# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-07
**Commit:** 00fb96e (master)
**Workspace:** Local-first LLM inference in Rust with Go, Python, C, and C++ ports

## OVERVIEW
This workspace contains the core Rust LLM inference engine (`oxidize-core`) and multiple frontends/bindings (CLI, server, Python bindings), plus parallel language ports in Go, pure Python, C, and C++ for cross-platform deployment. Supporting crates cover quantization, conversion, pruning, merging, finetuning, kernels, and FFI.

## STRUCTURE
```
.
├── oxidize-core/          # Rust core: GGUF, tensors, quantization, generation, backends
│   ├── src/backends/      # CUDA, Metal, Vulkan, MLX, WebGPU (see backends/AGENTS.md)
│   ├── src/compute/       # Tensor ops, KV cache, flash attention, quantization (see compute/AGENTS.md)
│   ├── src/format/        # GGUF, SafeTensors, tokenizer (see format/AGENTS.md)
│   ├── src/mesh/          # Distributed inference (see mesh/AGENTS.md)
│   ├── src/model/         # Inference engine, sampling, DFlash (see model/AGENTS.md)
│   ├── src/paged_attention/ # vLLM-style paging scheduler (see paged_attention/AGENTS.md)
│   ├── src/vision/        # CLIP-style vision encoder (see vision/AGENTS.md)
│   ├── src/video/         # Video multimodal path (see video/AGENTS.md)
│   ├── src/autotune/      # Hardware detect + tuning plan (see autotune/AGENTS.md)
│   └── src/util/          # mmap, attn dump, WASM bridge (see util/AGENTS.md)
├── oxidize-cli/           # Prompt/chat CLI, profiling, pipeline modes (see AGENTS.md)
├── oxidize-server/        # OpenAI-compatible HTTP API (axum) (see src/AGENTS.md)
├── oxidize-quantize/      # Offline weight conversion (see AGENTS.md)
├── oxidize-convert/       # SafeTensors → GGUF conversion (see AGENTS.md)
├── oxidize-prune/         # Wanda/magnitude pruning (see AGENTS.md)
├── oxidize-merge/         # SafeTensors checkpoint merging (see AGENTS.md)
├── oxidize-finetuning/    # LoRA / SFT / self-train (see AGENTS.md)
├── oxidize-kernels/       # OXK hand-tuned CPU GEMV kernels (see AGENTS.md)
├── oxidize-ffi/           # C-ABI FFI over oxidize-core (see AGENTS.md)
├── oxidize-py/            # Python bindings (pyo3 + maturin) (see AGENTS.md)
├── oxidize-train/         # CSV classifier + video training (see AGENTS.md)
├── oxidize-golang/        # Go port of oxidize-core (see AGENTS.md)
├── oxidize-python/        # Pure-Python port (see AGENTS.md)
├── oxidize-c/             # Dependency-free C11 port (see AGENTS.md)
├── oxidize-cpp/           # C++20 Llama-family inference (see AGENTS.md)
└── scripts/               # CI benchmark regression + remote bench recipes (see AGENTS.md)
```

## SUBDIRECTORY AGENTS.md MAP
| Directory | File | Domain |
|-----------|------|--------|
| **oxidize-core** | | |
| `oxidize-core/src/compute/` | `compute/AGENTS.md` | CPU tensor ops, quantization, KV cache, flash attention |
| `oxidize-core/src/model/` | `model/AGENTS.md` | Inference engine, model loading, speculative decoding |
| `oxidize-core/src/mesh/` | `mesh/AGENTS.md` | Distributed inference (libp2p mesh) |
| `oxidize-core/src/backends/` | `backends/AGENTS.md` | Hardware compute backends |
| `oxidize-core/src/format/` | `format/AGENTS.md` | GGUF, SafeTensors, tokenizer |
| `oxidize-core/src/paged_attention/` | `paged_attention/AGENTS.md` | vLLM-style PagedAttention scheduler |
| `oxidize-core/src/vision/` | `vision/AGENTS.md` | CLIP-style vision encoder for multimodal |
| `oxidize-core/src/video/` | `video/AGENTS.md` | Video multimodal frame sampling + encoding |
| `oxidize-core/src/autotune/` | `autotune/AGENTS.md` | Hardware detect + GGUF fingerprint + tuning plan |
| `oxidize-core/src/util/` | `util/AGENTS.md` | mmap, attn dump, benchmark suite, WASM bridge |
| **Rust crates** | | |
| `oxidize-cli/` | `AGENTS.md` | CLI for prompt/chat, benchmarking |
| `oxidize-server/src/` | `src/AGENTS.md` | OpenAI-compatible HTTP API (Axum) |
| `oxidize-quantize/` | `AGENTS.md` | Offline weight quantization utility |
| `oxidize-convert/` | `AGENTS.md` | SafeTensors → GGUF conversion + optional prune |
| `oxidize-prune/` | `AGENTS.md` | Wanda/magnitude pruning |
| `oxidize-merge/` | `AGENTS.md` | SafeTensors checkpoint merging (linear/SLERP) |
| `oxidize-finetuning/` | `AGENTS.md` | LoRA / SFT / self-train fine-tuning |
| `oxidize-kernels/` | `AGENTS.md` | OXK hand-tuned CPU GEMV kernels |
| `oxidize-ffi/` | `AGENTS.md` | C-ABI FFI over oxidize-core |
| `oxidize-train/` | `AGENTS.md` | CSV classifier + video training |
| **Language ports** | | |
| `oxidize-py/` | `AGENTS.md` | PyO3 Python bindings |
| `oxidize-golang/` | `AGENTS.md` | Go port of oxidize-core |
| `oxidize-python/` | `AGENTS.md` | Pure-Python port |
| `oxidize-c/` | `AGENTS.md` | Dependency-free C11 port |
| `oxidize-cpp/` | `AGENTS.md` | C++20 Llama-family inference |
| **Tooling** | | |
| `scripts/` | `AGENTS.md` | CI benchmark gating, remote NUMA bench, publish recipes |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `ComputeBackend` | trait | `oxidize-core/src/backend.rs` | Abstraction all backends implement |
| `Model` | trait | `oxidize-core/src/model.rs` | Implemented by 5 structs (Inference, Llama, LayerWise, MLX, DFlash) |
| `GgufQuantizationType` | enum | `oxidize-core/src/format/gguf.rs` | Central type hub; 20+ cross-module refs |
| `tensor.rs` | module | `oxidize-core/src/compute/` | 5,153 lines; 135 unsafe blocks; SIMD kernels |
| `scheduler.rs` | module | `oxidize-core/src/paged_attention/` | vLLM-style request scheduling |
| `app.rs` | module | `oxidize-server/src/` | Axum route assembly |
| `TuningPlan` | struct | `oxidize-core/src/autotune/rules.rs` | Fully-resolved autotune plan |
| `q4k_q8k_row_dot_avx2` | fn | `oxidize-kernels/src/q4k_avx2.rs` | Bit-exact OXK GEMV kernel |
| `oc_forward` | fn | `oxidize-c/model.c` | C port forward pass |
| `LlamaModel` | struct | `oxidize-cpp/include/oxidize/model_llama.hpp` | C++ Llama inference |

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
| Port to C | `oxidize-c/` | Intentional standalone C11 port; see `oxidize-c/AGENTS.md` |
| Port to C++ | `oxidize-cpp/` | llama.cpp parity focus; see `oxidize-cpp/AGENTS.md` |
| SafeTensors → GGUF | `oxidize-convert/` | Core logic in `oxidize-core/src/format/safetensors_to_gguf.rs` |
| Wanda pruning | `oxidize-prune/src/wanda.rs` | Per-output-row `|W| · ‖X‖_2`; see `oxidize-prune/AGENTS.md` |
| Magnitude pruning | `oxidize-prune/src/mask.rs` + `wanda.rs` | Per-output-row `|W|`; per Wanda paper, the right default for LLMs |
| Activation L2 norms (Wanda calibration) | `oxidize-core/src/compute/activation_stats.rs` | `ActivationStats` + `CalibrationRunner`; consumed by `oxidize-prune` |
| Checkpoint merging | `oxidize-merge/` | Linear/SLERP blend of SafeTensors checkpoints |
| LoRA / SFT / self-train | `oxidize-finetuning/` | `sft`, `self-train`, `merge` wired; `dpo`/`ppo` are stubs |
| OXK CPU kernels | `oxidize-kernels/` | Bit-exact Q4_K×Q8_K GEMV; consumed via `oxk` feature |
| C FFI surface | `oxidize-ffi/src/lib.rs` | `cdylib`/`staticlib` over oxidize-core |
| Auto-detect + auto-tune | `oxidize-core/src/autotune/` | `detect()` + `fingerprint()` + `plan()`; CLI flags `--auto --no-auto --print-plan` |
| Vision / multimodal | `oxidize-core/src/vision/` | CLIP-style encoder + `MultimodalPrompt` |
| Video multimodal | `oxidize-core/src/video/` | Frame sampling + temporal aggregation |
| Skylake-SP detection (AVX-512 regression gate) | `oxidize-kernels/src/cpu.rs` | `pub fn is_skylake_sp() -> bool` |
| CI benchmark regression | `scripts/ci_benchmark_regression.py` | Perf gate + dashboard |
| Remote NUMA benchmark | `scripts/bench-ai-box.sh` | Defaults to `ai@192.168.1.132` |

## CONVENTIONS
- **Flat module system**: `lib.rs` uses `#[path = "..."]` to flatten all modules into crate root. Only `mesh/`, `paged_attention/`, `vision/`, `video/` have real `mod.rs` files.
- **Config + Error + Trait trinity**: Every subsystem has `XxxConfig`, `XxxError`, and core trait/struct.
- **Error chaining**: All errors wrap lower-level errors via `From` impls.
- **Backend dual-file**: `vulkan.rs` + `vulkan_stub.rs` pair (only backend with this pattern).
- **Build info micro-pattern**: Every backend exposes `XxxBuildInfo` + `xxx_build_info()` for compile-time detection.
- **Test co-location**: Every `.rs` file has `#[cfg(test)]` module at bottom; no separate `tests/` inside `src/`.
- **AGENTS.md per domain**: Every crate and major `oxidize-core/src/` subdirectory has an `AGENTS.md` — check the map above before exploring blindly.
- **Port order**: Rust → Go (`oxidize-golang`) → Python (`oxidize-python`); C (`oxidize-c`) and C++ (`oxidize-cpp`) are independent ports.

## ANTI-PATTERNS (THIS PROJECT)
- `StdMutex` in async context (`oxidize-server/src/runtime/paged.rs`) — should be `tokio::sync::Mutex`.
- `tensor.rs` monolith — 5,153 lines mixing kernels, types, and ops. Refactor candidate.
- Quantization constants shadowed in `tensor.rs` and `cuda.rs` — should be shared.
- `unwrap()/expect()` proliferation — 1000+ instances in non-test code.
- Stop trying to use C for tasks that can be done in rust (Claude) — except `oxidize-c/` which is an intentional standalone C port.
- `oxidize-finetuning` breaks whole-workspace `cargo build --workspace` (pre-existing `qlora.rs` borrow-check error) — build per-crate.

## UNIQUE STYLES
- **Bottom-up file organization** (`tensor.rs`): constants → errors → low-level kernels → high-level functions → `Tensor` struct (inverse of typical Rust).
- **WASM worker type embedding**: `util/web_worker.rs` embeds complete TypeScript interface contracts as 60+ line string literals.
- **MLX macOS fortress**: `mlx.rs` and `mlx_inference.rs` are heavily `#[cfg(target_os = "macos")]` gated.
- **OXK bit-exact parity**: `oxidize-kernels` kernels must match scalar reference exactly — parity is a hard invariant.
- **Autotune pure planner**: `autotune/rules.rs::plan()` is a pure function with every decision in `plan.rationale`.

## COMMANDS
```bash
# Build / test / lint (core product — avoids oxidize-finetuning borrow error)
cargo build -p oxidize-cli -p oxidize-server -p oxidize-quantize -p oxidize-convert
cargo test  -p oxidize-core -p oxidize-cli -p oxidize-server -p oxidize-kernels
make build    # release build (may fail on oxidize-finetuning)
make test     # workspace tests
make lint     # clippy -D warnings
make fmt      # format check
make ci       # full CI equivalent

# Run
sfw cargo run -p oxidize-cli -- --prompt "hello"
sfw cargo run -p oxidize-server -- --host 127.0.0.1 --port 8080
sfw cargo run -p oxidize-quantize -- --input in.bin --output out.bin --source F32 --target Q4_0
sfw cargo run -p oxidize-convert -- --input model/ --output model.gguf --target Q4_K_M
sfw cargo run -p oxidize-finetuning -- self-train --model base.gguf --dataset data.jsonl

# C / C++ ports
make -C oxidize-c && ./oxidize-c/oxidize-c --model model.gguf --prompt "hi"
cmake -B oxidize-cpp/build -S oxidize-cpp && cmake --build oxidize-cpp/build -j

# Go / Python ports
cd oxidize-golang && CGO_ENABLED=0 go test ./...
cd oxidize-python && uv run pytest

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
- Git installs must name `oxidize-cli` explicitly (`cargo install --git … oxidize-cli --bin oxidize`) because the workspace ships multiple binary crates.

## Learned User Preferences
- When adding `oxidize-python` or expanding `oxidize-golang`, keep all Rust crates and features; do not delete or replace the Rust workspace.
- Parallel Go/Python ports should reach `oxidize-core` feature parity (Python targeting similar CLOC to Rust); implement in `oxidize-golang` first, mirror to `oxidize-python`, and sync new `master` Rust features.
- Keep `oxidize-py` (PyO3/maturin bindings) alongside the pure-Python `oxidize-python` package.
- Prioritize oxidize-c/oxidize-cpp CPU inference speed; use `oxidize-c` for Qwen 3.5/GDN hybrid models (`oxidize-cpp` cannot load them); benchmark C/C++ tuning on `ai@192.168.1.122` or `ai@192.168.1.132`.
- For Go/Python GPU backends, use pure native implementations (no Rust FFI at runtime; CGO permitted for native GPU bindings); CUDA first, then Vulkan/Metal/WebGPU.
- Avoid creating extra markdown documentation files unless asked; update README when needed.
- On feature branches, stage and commit only files related to the task; exclude unrelated workspace changes.
- Running a model means running it with oxidize; `oxidize run <model>` starts the OpenAI-compatible HTTP/WebSocket server by default (`--no-api` for local inference only), otherwise prefer building/testing over starting dev servers unless explicitly asked to run/serve.
- Contributions should keep tests passing and use clear, ethical PR/markdown descriptions; include benchmarks when claiming performance changes.
- Publish quant/finetuned models to private HF repos unless the user requests public; strip all branding from produced GGUFs, and for finetuning convert from the original SafeTensors rather than reusing a published GGUF.
- Run GPU finetuning via the Prime Intellect CLI on a cheap but powerful rented GPU (not CPU); SSH in and commit changes as needed (`scripts/prime-qwen35-self-train`).
- Provide quick-running Google Colab notebooks for quant/finetuning workflows; ensure notebooks/README use the correct oxidize GitHub URL and that changes land on `master`.

## Learned Workspace Facts
- `oxidize-golang/` is the active Go port of `oxidize-core`; CLI lives in `internal/cli/` (`run`, `chat`, `bench`, `inspect`, `list`, `serve`); HF GGUF resolver in `hf/`.
- `oxidize-python/` is a pure-Python implementation (`oxidize_python`, `pyproject.toml`, uv/pytest); CLI mirrors Go subcommands; HF resolver in `oxidize_python/hf/hub.py` with cache `~/.cache/oxidize/hf`; `oxidize-py/` is the separate PyO3/maturin bindings crate.
- Do not modify Rust crates when extending `oxidize-python`; port from `oxidize-golang` or Rust sources.
- `oxidize-cpp/` is the C++ Llama-family inference port (CPU + optional `OXIDIZE_CUDA` or `OXIDIZE_ROCM`); CLI `--auto`/`--print-plan` autotune NUMA/threads from model file size; CUDA fast path is `resident_forward` (~1 sync/token); llama.cpp parity is an active focus; cannot load Qwen 3.5/GDN hybrids.
- `oxidize-c/` is a dependency-free C11 port with optional `OC_CUDA` fast path; shares AL-family quant types with Rust/C++; supports Qwen 3.5 Gated-DeltaNet hybrids and is the fastest CPU serve path for self-train GDN GGUFs.
- Remote hosts: `ai@192.168.1.132` (primary NUMA bench: 2× Xeon Gold 5220R, 96 logical, 376 GB RAM); `ai@192.168.1.122` (dual-Xeon CPU, no GPU, ~376 GB RAM — C/C++ inference benchmarks, self-train serve, model merge staging; hosts `oxidize-tps` with `--numa replicate` for peak CPU TPS); `ai@192.168.1.121` (~20 TB storage for large-model quant + HF publish); legacy `ai@192.168.1.68`; `scripts/bench-ai-box.sh` defaults to `.132`; `oxidize-cpp-glm/` is a separate GLM fork (MLA/IQ1/MoE).
- Custom AL-family quants (`AL5`, `AL5_XS`, `AL6`, `AL8`; ggml types 240–243) live in `oxidize-core`, `oxidize-cpp`, and `oxidize-c`; AL5 is MSE-optimized 4-bit; prioritize speed without quality loss when tuning them; validated against Gemma 4 31B and Qwen 35 base models via the `oxidize-c` runner.
- `oxidize-finetuning` exposes `self-train` CLI (`cargo run -p oxidize-finetuning -- self-train`): iterative LoRA SFT with per-round checkpoints, self-dialogue synthetic data (`synthetic.jsonl`), and optional self-critique; resume via `--resume-from`; GPU self-train on rented Prime Intellect boxes uses `scripts/prime-qwen35-self-train/` (run-gpu-finetune / unsloth-gpu, merge-and-upload, HF upload).
- DFlash speculative decoding in `oxidize-core/src/model/dflash.rs` is an active port target for `oxidize-golang` (and downstream Python); inference needs a compatible target GGUF paired with the draft (hidden-size mismatch falls back to target-only).
- Rust `oxidize run` rewrites to `--serve-api` by default (background in-process server on `--api-host`/`--api-port`); realtime WebSocket at `ws://HOST:PORT/v1/realtime` (`oxidize-server/tests/realtime_ws.rs`).
- `oxidize-convert` converts HuggingFace SafeTensors (file or model directory with `config.json`) to GGUF; core logic in `oxidize-core/src/format/safetensors_to_gguf.rs`.
- Go/Python ports and `oxidize-cpp` expose `--auto`, `--no-auto`, `--print-plan` autotune; on dual-socket CPU, dense models ≤192 GB use `--numa single --threads 16`, models >192 GB use `--numa interleave --threads 48`; test Go with `CGO_ENABLED=0`, Python with `uv run pytest` (`OXIDIZE_SLOW_TESTS=1` for slow GGUF).

## Cursor Cloud specific instructions
- The startup update script ensures the Rust `stable` toolchain (edition 2024 needs >= 1.85; the base image ships 1.83 which is too old), `cargo fetch`, Go module deps, and the Python port's `uv sync`. Standard build/test/run commands live in `Makefile`, `QUICKSTART.md`, and `HOW_TO_INSTALL.md`.
- Non-obvious gotcha: `make build` / `cargo build --workspace` currently FAILS to compile the optional `oxidize-finetuning` crate (`src/qlora.rs` borrow-check error, pre-existing). Build/test the core product per-crate instead, e.g. `cargo build -p oxidize-cli -p oxidize-server -p oxidize-quantize -p oxidize-convert` and `cargo test -p oxidize-core -p oxidize-cli -p oxidize-server -p oxidize-kernels`. The MUST product (CLI + server) is unaffected.
- `make lint` (clippy `-D warnings`), `make audit` (needs `cargo install cargo-deny`), and the Python `ruff check` all currently report pre-existing warnings/errors; these are code-quality debts, not environment breakage.
- CLI/server run with placeholder weights when no `--model` is given: `oxidize-cli --prompt ...` echoes the prompt and `/v1/chat/completions` returns an empty `chatcmpl-placeholder`. This is expected; real token generation requires a real GGUF (`--model path.gguf`, or an HF id via the resolver). Committed `oxidize-core/tests/fixtures/*.gguf` are tiny parser fixtures, not runnable models.
- Go port auto-downloads the `go1.26.2` toolchain via `GOTOOLCHAIN=auto` on first `go build`/`go test` (base image has Go 1.22); no manual Go upgrade needed.
- `uv` is installed to `~/.local/bin`; if not on PATH, invoke as `~/.local/bin/uv`. Run Python port commands from `oxidize-python/` (or pass `--directory oxidize-python`).
