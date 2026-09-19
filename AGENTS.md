# PROJECT KNOWLEDGE BASE

**Workspace:** Local-first LLM inference in Rust, with a standalone C11 port

## OVERVIEW
This workspace is the Rust LLM inference engine (`oxidize-core`) plus CLI, server, quantization, conversion, pruning, merge, finetune, kernels, and C-ABI FFI. `oxidize-c` is the only non-Rust product: a dependency-free C11 port of the same inference surface. Duplicate language ports (Go, Python, C++, TypeScript TUI, PyO3) were removed so features live in Rust and C only.

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
├── oxidize-train/         # CSV classifier + video training (see AGENTS.md)
├── oxidize-c/             # Dependency-free C11 port (see AGENTS.md)
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
| **C port** | | |
| `oxidize-c/` | `AGENTS.md` | Dependency-free C11 port |
| **Tooling** | | |
| `scripts/` | `AGENTS.md` | CI benchmark gating, remote NUMA bench, publish recipes |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `ComputeBackend` | trait | `oxidize-core/src/backend.rs` | Abstraction all backends implement |
| `Model` | trait | `oxidize-core/src/model.rs` | Implemented by 5 structs (Inference, Llama, LayerWise, MLX, DFlash) |
| `GgufQuantizationType` | enum | `oxidize-core/src/format/gguf.rs` | Central type hub; 20+ cross-module refs |
| `tensor.rs` | module | `oxidize-core/src/compute/` | SIMD kernels and tensor ops |
| `scheduler.rs` | module | `oxidize-core/src/paged_attention/` | vLLM-style request scheduling |
| `app.rs` | module | `oxidize-server/src/` | Axum route assembly |
| `TuningPlan` | struct | `oxidize-core/src/autotune/rules.rs` | Fully-resolved autotune plan |
| `q4k_q8k_row_dot_avx2` | fn | `oxidize-kernels/src/q4k_avx2.rs` | Bit-exact OXK GEMV kernel |
| `oc_forward` | fn | `oxidize-c` model path | C port forward pass |

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
| Port to C | `oxidize-c/` | Intentional standalone C11 port; see `oxidize-c/AGENTS.md` |
| SafeTensors → GGUF | `oxidize-convert/` | Core logic in `oxidize-core/src/format/safetensors_to_gguf.rs` |
| Wanda pruning | `oxidize-prune/src/wanda.rs` | Per-output-row `|W| · ‖X‖_2` |
| Magnitude pruning | `oxidize-prune/src/mask.rs` + `wanda.rs` | Per-output-row `|W|` |
| Activation L2 norms (Wanda calibration) | `oxidize-core/src/compute/activation_stats.rs` | Consumed by `oxidize-prune` |
| Checkpoint merging | `oxidize-merge/` | Linear/SLERP blend of SafeTensors checkpoints |
| LoRA / SFT / self-train | `oxidize-finetuning/` | `sft`, `self-train`, `merge` wired; `dpo`/`ppo` are stubs |
| OXK CPU kernels | `oxidize-kernels/` | Bit-exact Q4_K×Q8_K GEMV; consumed via `oxk` feature |
| C FFI surface | `oxidize-ffi/src/lib.rs` | `cdylib`/`staticlib` over oxidize-core |
| Auto-detect + auto-tune | `oxidize-core/src/autotune/` | `detect()` + `fingerprint()` + `plan()`; `--auto --no-auto --print-plan` |
| Vision / multimodal | `oxidize-core/src/vision/` | CLIP-style encoder + `MultimodalPrompt` |
| Video multimodal | `oxidize-core/src/video/` | Frame sampling + temporal aggregation |
| Skylake-SP detection | `oxidize-kernels/src/cpu.rs` | `pub fn is_skylake_sp() -> bool` |
| CI benchmark regression | `scripts/ci_benchmark_regression.py` | Perf gate + dashboard |
| Remote NUMA benchmark | `scripts/bench-ai-box.sh` | Defaults to `ai@192.168.1.132`; uses oxidize-c |

## CONVENTIONS
- **Two product languages only:** Rust workspace crates and `oxidize-c`. Do not reintroduce Go, Python, C++, or TypeScript product ports.
- **Flat module system:** `lib.rs` uses `#[path = "..."]` to flatten modules. Only `mesh/`, `paged_attention/`, `vision/`, `video/` have real `mod.rs` files.
- **Config + Error + Trait trinity:** Every subsystem has `XxxConfig`, `XxxError`, and core trait/struct.
- **Error chaining:** All errors wrap lower-level errors via `From` impls.
- **Backend dual-file:** `vulkan.rs` + `vulkan_stub.rs` pair.
- **Build info micro-pattern:** Every backend exposes `XxxBuildInfo` + `xxx_build_info()`.
- **Test co-location:** Every `.rs` file has `#[cfg(test)]` at bottom; no separate `tests/` inside `src/`.
- Embed C callers through `oxidize-ffi` when they need the Rust engine; use `oxidize-c` when they need a libc-only binary.

Good vs bad (this repo):
- Good: one implementation of sampling/quant in `oxidize-core`, C port mirrors behavior instead of inventing a third API.
- Bad: parallel engines in extra languages that drift (the deleted Go/Python/C++ trees).
- Good: `plan()` in autotune is a pure function with every decision in `plan.rationale`.
- Bad: `unwrap()` in non-test inference paths; `StdMutex` in async (`oxidize-server` paged runtime).

## ANTI-PATTERNS (THIS PROJECT)
- `StdMutex` in async context (`oxidize-server/src/runtime/paged.rs`) — should be `tokio::sync::Mutex`.
- `tensor.rs` monolith mixing kernels, types, and ops.
- Quantization constants shadowed in `tensor.rs` and `cuda.rs`.
- `unwrap()/expect()` proliferation in non-test code.
- Use C only in `oxidize-c/` or FFI consumers — not for tasks that belong in Rust crates.
- `oxidize-finetuning` breaks whole-workspace `cargo build --workspace` (pre-existing `qlora.rs` borrow-check error) — build per-crate.

## UNIQUE STYLES
- **Bottom-up file organization** (`tensor.rs`): constants → errors → low-level kernels → high-level functions → `Tensor` struct.
- **WASM worker type embedding:** `util/web_worker.rs` embeds TypeScript interface contracts as string literals (WASM glue, not a product TUI).
- **MLX macOS fortress:** `mlx.rs` and `mlx_inference.rs` are `#[cfg(target_os = "macos")]` gated.
- **OXK bit-exact parity:** kernels must match scalar reference exactly.
- **Autotune pure planner:** `autotune/rules.rs::plan()` is a pure function.

## COMMANDS
```bash
# Build / test / lint (core product — avoids oxidize-finetuning borrow error)
cargo build -p oxidize-cli -p oxidize-server -p oxidize-quantize -p oxidize-convert
cargo test  -p oxidize-core -p oxidize-cli -p oxidize-server -p oxidize-kernels
make build
make test
make lint
make fmt
make ci
make c-build
make c-test

# Run
sfw cargo run -p oxidize-cli -- --prompt "hello"
sfw cargo run -p oxidize-server -- --host 127.0.0.1 --port 8080
sfw cargo run -p oxidize-quantize -- --input in.bin --output out.bin --source F32 --target Q4_0
sfw cargo run -p oxidize-convert -- --input model/ --output model.gguf --target Q4_K_M
sfw cargo run -p oxidize-finetuning -- self-train --model base.gguf --dataset data.jsonl

# C port
make -C oxidize-c && ./oxidize-c/oxidize-c --model model.gguf --prompt "hi"

# WASM
make wasm
```

## NOTES
- Rust edition 2024, resolver "3".
- Release profile: `lto = true`, `panic = "abort"`.
- `cargo-deny` audits licenses + security (see `deny.toml`).
- `.cargo/config.toml` sets custom linker for `aarch64-unknown-linux-gnu` and WASM runner.
- `oxidize-core/fuzz/` exists but is NOT in workspace members/exclude.
- `models/` is gitignored but contains tracked files.
- GGUF/SafeTensors draft-model loading + speculative generation is an active development area.
- Git installs must name `oxidize-cli` explicitly (`cargo install --git … oxidize-cli --bin oxidize`).

## Learned User Preferences
- Keep all Rust crates and the C port; do not add language ports besides C and Rust.
- Avoid extra markdown files unless asked; update README when needed.
- On feature branches, stage and commit only files related to the task.
- `oxidize run <model>` should start the OpenAI-compatible HTTP/WebSocket server by default; use `--no-api` for local inference only.
- Contributions should keep tests passing and use clear PR descriptions; include benchmarks when claiming performance changes.
- When a user asks to run a model, run it using oxidize.
- Prefer building and testing over starting development servers unless asked to serve.
- Custom Hugging Face repos for quant/model publishing should be private unless the user explicitly requests public.

## Learned Workspace Facts
- `oxidize-c/` is a dependency-free C11 port with optional `OC_CUDA` fast path; shares AL-family quant types with Rust.
- Remote hosts: `ai@192.168.1.132` (primary NUMA bench: 2× Xeon Gold 5220R, 96 logical, 376 GB RAM); `ai@192.168.1.121` (~20 TB storage for large-model quant + HF publish); legacy `ai@192.168.1.68`; `scripts/bench-ai-box.sh` defaults to `.132`.
- Custom AL-family quants (`AL5`, `AL5_XS`, `AL6`, `AL8`; ggml types 240–243) live in `oxidize-core` and `oxidize-c`. AL5 is MSE-optimized 4-bit.
- `oxidize-finetuning` exposes `self-train` CLI: iterative LoRA SFT with per-round checkpoints and optional self-critique; resume via `--resume-from`.
- DFlash speculative decoding lives in `oxidize-core/src/model/dflash.rs`; inference needs a compatible target GGUF paired with the draft (hidden-size mismatch falls back to target-only).
- Rust `oxidize run` rewrites to `--serve-api` by default (in-process server on `--api-host`/`--api-port`); realtime WebSocket at `ws://HOST:PORT/v1/realtime`.
- `oxidize-convert` converts HuggingFace SafeTensors (file or model directory with `config.json`) to GGUF; core logic in `oxidize-core/src/format/safetensors_to_gguf.rs`.
- Autotune flags `--auto`, `--no-auto`, `--print-plan` exist on Rust CLI and oxidize-c. On dual-socket CPU, dense models ≤192 GB use `--numa single --threads 16`, models >192 GB use `--numa interleave --threads 48`.

## Cursor Cloud specific instructions
- The startup update script ensures the Rust `stable` toolchain (edition 2024 needs >= 1.85; the base image ships 1.83 which is too old) and `cargo fetch`. Standard commands live in `Makefile`, `QUICKSTART.md`, and `HOW_TO_INSTALL.md`.
- Non-obvious gotcha: `make build` / `cargo build --workspace` currently FAILS to compile the optional `oxidize-finetuning` crate (`src/qlora.rs` borrow-check error, pre-existing). Build/test the core product per-crate instead, e.g. `cargo build -p oxidize-cli -p oxidize-server -p oxidize-quantize -p oxidize-convert` and `cargo test -p oxidize-core -p oxidize-cli -p oxidize-server -p oxidize-kernels`. The MUST product (CLI + server) is unaffected.
- `make lint` (clippy `-D warnings`) and `make audit` currently report pre-existing warnings/errors; these are code-quality debts, not environment breakage.
- CLI/server run with placeholder weights when no `--model` is given: `oxidize-cli --prompt ...` echoes the prompt and `/v1/chat/completions` returns an empty `chatcmpl-placeholder`. Real token generation requires a real GGUF. Committed `oxidize-core/tests/fixtures/*.gguf` are tiny parser fixtures, not runnable models.
