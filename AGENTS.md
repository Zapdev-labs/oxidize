# PROJECT KNOWLEDGE BASE

**Updated:** 2026-09-23
**Commit:** d6505c4e (feat/distill-training-kaggle-k2)
**Workspace:** Local-first LLM inference — C11 fast path (`oxidize-c`) plus Rust core, Go, Python, and C++ ports

## ⚡ TOP PRIORITY: `oxidize-c/`
`oxidize-c` is the **fastest** implementation in this workspace and the **#1 priority** for inference work.
- New inference features, model architectures (incl. from-scratch MoE / Edge0 / A3B–A4B day-0 support), quant types, and speed work land in **`oxidize-c` first**.
- When benchmarking or running a model for real throughput, default to `oxidize-c` (CPU, `make avx512`, or `make cuda` / `OC_CUDA`).
- Never regress `oxidize-c` speed. Any change touching the forward pass, SIMD kernels, or quant paths must be benchmarked before/after (local `dih@192.168.1.15`, NUMA box `ai@192.168.1.132`).
- Other implementations (Rust, C++, Go, Python) follow `oxidize-c` for speed; when they disagree on performance, `oxidize-c` is the reference to beat.
- Conventions, build targets, and test rules live in `oxidize-c/CONTRIBUTING.md` — read it before editing `oxidize-c/`.

## OVERVIEW
This workspace contains a dependency-free C11 inference engine (`oxidize-c`, the fastest and top-priority path), the Rust LLM inference engine (`oxidize-core`) with its frontends/bindings (CLI, server, Python bindings, TUI), and parallel ports in Go, pure Python, and C++. Supporting Rust crates cover quantization, conversion, pruning, merging, finetuning, kernels, and FFI.

## STRUCTURE
```
.
├── oxidize-c/             # ⚡ TOP PRIORITY — dependency-free C11 engine, fastest path (see CONTRIBUTING.md)
│   ├── src/               # core/ (SIMD dispatch), compute/, model/, format/, backends/, server/, cli/, ...
│   ├── include/oxidize/   # Public headers
│   └── tests/             # Criterion test suite
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
├── oxidize-cpp/           # C++20 Llama-family inference (see AGENTS.md)
├── oxidize-tui/           # OpenTUI/Bun terminal UI driving `oxidize serve` (see AGENTS.md)
└── scripts/               # CI benchmark regression + remote bench recipes (see AGENTS.md)
```

## SUBDIRECTORY AGENTS.md MAP
| Directory | File | Domain |
|-----------|------|--------|
| **⚡ Top priority** | | |
| `oxidize-c/` | `CONTRIBUTING.md` | Dependency-free C11 engine — fastest path, build/test/C conventions |
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
| `oxidize-cpp/` | `AGENTS.md` | C++20 Llama-family inference |
| `oxidize-tui/` | `AGENTS.md` | Terminal UI (OpenTUI + React, Bun) |
| **Tooling** | | |
| `scripts/` | `AGENTS.md` | CI benchmark gating, remote NUMA bench, publish recipes |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `oc_llama_forward` / `oc_qwen_forward` / `oc_moe_ffn_forward` | fn | `oxidize-c/src/model/` | ⚡ C forward passes — the hot path to optimize first |
| `simd.c` / `simd_avx2.c` / `simd_avx512.c` | module | `oxidize-c/src/core/` | C SIMD dispatch; intrinsics only in `simd_*.c` |
| `ComputeBackend` | trait | `oxidize-core/src/backend.rs` | Abstraction all backends implement |
| `Model` | trait | `oxidize-core/src/model.rs` | Implemented by 5 structs (Inference, Llama, LayerWise, MLX, DFlash) |
| `GgufQuantizationType` | enum | `oxidize-core/src/format/gguf.rs` | Central type hub; 20+ cross-module refs |
| `tensor.rs` | module | `oxidize-core/src/compute/` | 5,153 lines; 135 unsafe blocks; SIMD kernels |
| `scheduler.rs` | module | `oxidize-core/src/paged_attention/` | vLLM-style request scheduling |
| `app.rs` | module | `oxidize-server/src/` | Axum route assembly |
| `TuningPlan` | struct | `oxidize-core/src/autotune/rules.rs` | Fully-resolved autotune plan |
| `q4k_q8k_row_dot_avx2` | fn | `oxidize-kernels/src/q4k_avx2.rs` | Bit-exact OXK GEMV kernel |
| `LlamaModel` | struct | `oxidize-cpp/include/oxidize/model_llama.hpp` | C++ Llama inference |

## WHERE TO LOOK (High-Level)
| Task | Location | Notes |
|------|----------|-------|
| **Speed / inference work (default)** | `oxidize-c/` | ⚡ Start here; benchmark before/after; see `oxidize-c/CONTRIBUTING.md` |
| New arch / MoE day-0 support | `oxidize-c/src/model/` | Land in C first, then Rust/C++ |
| New quant type (incl. AL-family) | `oxidize-c/src/compute/` + `oxidize-core` + `oxidize-cpp` | Keep ggml type ids in sync (AL: 240–243) |
| Add model architecture (Rust) | `oxidize-core/src/model/inference.rs` | Extend `ModelArchitecture` enum |
| Add backend | `oxidize-core/src/backends/` | Implement `ComputeBackend` trait, add `XxxBuildInfo` |
| Add quantization type | `oxidize-core/src/compute/quantization.rs` | Also update `GgufQuantizationType` in `format/gguf.rs` |
| Tokenizer change | `oxidize-core/src/format/tokenizer.rs` | 4 formats: SP, WordPiece, BPE, Tiktoken |
| Server route | `oxidize-server/src/routes/` | OpenAI-compatible endpoints |
| CLI subcommand | `oxidize-cli/src/main.rs` | Also check `src/bin/` for aux tools |
| Distributed logic | `oxidize-core/src/mesh/` | Only dir with real `mod.rs` + privacy boundaries |
| Port to Go | `oxidize-golang/` | Mirror Rust structure; see `oxidize-golang/AGENTS.md` |
| Port to Python | `oxidize-python/` | Mirror Go structure; see `oxidize-python/AGENTS.md` |
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
- **Priority order**: `oxidize-c` (fastest, top priority) → Rust `oxidize-core` → C++ `oxidize-cpp`. Go/Python ports follow Rust: Rust → Go (`oxidize-golang`) → Python (`oxidize-python`).
- **oxidize-c conventions**: C11, libc/libm/pthread only in the default build; optional backends behind `OC_CUDA`/`OC_VULKAN`/`OC_METAL`/`OC_WEBGPU`/`OC_ROCM`/`OC_AVX512`; `OcPascalCase` types, `oc_snake_case` fns; return `OcError`; `OcArena` + mmap for weights; no `printf` in library code (use `oc_log`). Full rules in `oxidize-c/CONTRIBUTING.md`.

## ANTI-PATTERNS (THIS PROJECT)
- `StdMutex` in async context (`oxidize-server/src/runtime/paged.rs`) — should be `tokio::sync::Mutex`.
- `tensor.rs` monolith — 5,153 lines mixing kernels, types, and ops. Refactor candidate.
- Quantization constants shadowed in `tensor.rs` and `cuda.rs` — should be shared.
- `unwrap()/expect()` proliferation — 1000+ instances in non-test code.
- Don't add C code inside the Rust crates for tasks Rust can do — C belongs in `oxidize-c/`, which is the top-priority engine.
- Don't regress `oxidize-c` tokens/sec — unbenchmarked changes to its hot path are not acceptable.
- `oxidize-finetuning` breaks whole-workspace `cargo build --workspace` (pre-existing `qlora.rs` borrow-check error) — build per-crate.

## UNIQUE STYLES
- **Bottom-up file organization** (`tensor.rs`): constants → errors → low-level kernels → high-level functions → `Tensor` struct (inverse of typical Rust).
- **WASM worker type embedding**: `util/web_worker.rs` embeds complete TypeScript interface contracts as 60+ line string literals.
- **MLX macOS fortress**: `mlx.rs` and `mlx_inference.rs` are heavily `#[cfg(target_os = "macos")]` gated.
- **OXK bit-exact parity**: `oxidize-kernels` kernels must match scalar reference exactly — parity is a hard invariant.
- **Autotune pure planner**: `autotune/rules.rs::plan()` is a pure function with every decision in `plan.rationale`.

## COMMANDS
```bash
# ⚡ oxidize-c (top priority — fastest)
make -C oxidize-c                 # CPU build (libc only) → ./oxidize-c/oxidize-c
make -C oxidize-c test            # Criterion suite
make -C oxidize-c avx512          # AVX-512 BW+VNNI build
make -C oxidize-c cuda            # OC_CUDA build
make -C oxidize-c lint            # clang-tidy
./oxidize-c/oxidize-c --model model.gguf --prompt "hi"

# Build / test / lint (Rust core product — avoids oxidize-finetuning borrow error)
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

# C++ port
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

## Server deployment
- Servers are LAN-reachable by design, but auth is **off** unless `OXIDIZE_API_KEY` / `OXIDIZE_API_KEYS` (comma-separated) is set. With auth off on a non-loopback bind (`--host 0.0.0.0`, a LAN IP) the server logs a startup warning; pass `--require-auth-on-public-bind` (both `oxidize-server` and `oxidize run`) to refuse to start instead (exit 2). Enable it for any shared-LAN deployment. Loopback binds never warn.
- With auth on, `/v1/*` and `/metrics` need `x-api-key` or `Authorization: Bearer`; the `?api_key=` query fallback is accepted only on `/v1/realtime` (browser WS) and is redacted in access logs. `/healthz` `/livez` `/readyz` stay open and bypass the rate limiter.
- Audit log: every request gets an `x-request-id`; API keys are logged only as HMAC-SHA256 fingerprints (set `OXIDIZE_AUDIT_HMAC_KEY` to keep them stable across restarts).
- `--max-tokens-cap` (`--api-max-tokens-cap` on `oxidize run`) clamps per-request `max_tokens`; default unlimited. Realtime sessions cap transcripts at 1024 items / 8 MiB (oldest evicted).
- `oxidize-c` server (`--serve-api`, `serve`, `serve-realtime`) follows the same posture: startup `oc_log` WARN on an unauthenticated non-loopback bind, `--require-auth-on-public-bind` refuses with exit 2; Bearer keys compared in constant time. `serve-realtime` has no auth layer, so it always counts as unauthenticated.

## Learned User Preferences
- **`oxidize-c` is the top priority — it is the fastest implementation.** Default to it for inference, speed, and new-model work; other ports follow.
- When adding `oxidize-python` or expanding `oxidize-golang`, keep all Rust crates and features; do not delete or replace the Rust workspace.
- Parallel Go/Python ports should reach `oxidize-core` feature parity (Python targeting similar CLOC to Rust); implement in `oxidize-golang` first, mirror to `oxidize-python`, and sync new `master` Rust features.
- Keep `oxidize-py` (PyO3/maturin bindings) alongside the pure-Python `oxidize-python` package.
- Prioritize oxidize-cpp speed and llama.cpp feature parity for large-model inference; benchmark CPU deployments on the remote NUMA box `ai@192.168.1.132` when tuning; treat llama.cpp as the speculative-decoding speed baseline for DFlash.
- For Go/Python GPU backends, use pure native implementations (no Rust FFI at runtime; CGO permitted for native GPU bindings); CUDA first, then Vulkan/Metal/WebGPU.
- Avoid extra markdown docs unless asked (update README when needed); put one-off Modal/training/helper scripts under `scripts/` and gitignore ephemeral artifacts there.
- On feature branches, stage and commit only files related to the task; exclude unrelated workspace changes.
- `oxidize run <model>` should start the OpenAI-compatible HTTP/WebSocket server by default (`--no-api` for local inference only); when the user asks to run a model, run it via oxidize.
- Prefer building and testing over starting development servers unless the user explicitly asks to run or serve; for notebook Unsloth/HF finetunes prefer Google Colab via the Colab CLI when available, do not exhaust Colab compute credits on one run, and end Colab/cloud sessions when the job finishes; finish those finetunes with LoRA merge + GGUF export when publishing; for long-running remote/Kaggle/Colab/cloud jobs, keep active tabs with periodic status checks (about every 5 minutes via `/loop` or equivalent) rather than fire-and-forget, and do not leave failing jobs unattended.
- For new MoE-style models, prefer from-scratch custom architectures with day-0 oxidize support (llama.cpp later), not only finetuning third-party bases; finish training with oxidize's custom training stack when possible and push throughput without quality collapse.
- Custom Hugging Face repos for quant/model publishing should be private unless the user explicitly requests public; uploads must include full weight artifacts (safetensors/GGUF) and merged LoRA adapters when publishing finetunes, not only config/tokenizer files; finetune notebooks should accept a multi-dataset input list and an HF token for faster downloads, then upload finished weights; when sweeping GGUF bit-widths prefer IQ-family / Unsloth Dynamic (IQ3_M for 3-bit; cover roughly 1–8 bit with extra 4-bit variants as needed) over MXFP4-only uploads; smoke-test on `dih@192.168.1.15` when asked (Pi/OpenCode/Droid are OK harnesses — edit those apps' configs on the machine where they run, and prefer ~262K context for K2 Horizon agent use; do not rewrite agent system prompts to chase TTFT — reuse the server prompt cache / matching prefix instead), compare K2 Horizon finetunes against the DERISKED/base checkpoint not an unrelated baseline, then delete local quant copies after a successful upload to free disk.

## Learned Workspace Facts
- `oxidize-golang/` is the active Go port of `oxidize-core`; CLI lives in `internal/cli/` (`run`, `chat`, `bench`, `inspect`, `list`, `serve`); HF GGUF resolver in `hf/`.
- `oxidize-python/` is a pure-Python implementation (`oxidize_python`, `pyproject.toml`, uv/pytest); CLI mirrors Go subcommands; HF resolver in `oxidize_python/hf/hub.py` with cache `~/.cache/oxidize/hf`; `oxidize-py/` is the separate PyO3/maturin bindings crate.
- Do not modify Rust crates when extending `oxidize-python`; port from `oxidize-golang` or Rust sources.
- `oxidize-cpp/` is the C++ Llama-family inference port (CPU + optional `OXIDIZE_CUDA` or `OXIDIZE_ROCM`); CLI `--auto`/`--print-plan` autotune NUMA/threads from model file size; CUDA fast path is `resident_forward` (~1 sync/token); llama.cpp parity is an active focus.
- `oxidize-c/` is a dependency-free C11 port with optional `OC_CUDA` fast path; shares AL-family quant types with Rust/C++; MoE-style (Edge0 / custom A3B–A4B) day-0 support for from-scratch custom arches is an active focus.
- Remote hosts: `ai@192.168.1.132` (primary NUMA bench: 2× Xeon Gold 5220R, 96 logical, 376 GB RAM); `ai@192.168.1.121` (~20 TB storage for large-model quant + HF publish); `dih@192.168.1.15` (local Omarchy: Ryzen/680M for oxidize-c, downloads, and local training before cloud scale-up); notebook Unsloth finetunes often run on Google Colab (Colab CLI); heavy training may move to RunPod L4-class GPUs; legacy `ai@192.168.1.68`; `scripts/bench-ai-box.sh` defaults to `.132`; `oxidize-cpp-glm/` is a separate GLM fork (MLA/IQ1/MoE).
- Custom AL-family quants (`AL5`, `AL5_XS`, `AL6`, `AL8`; ggml types 240–243) live in `oxidize-core`, `oxidize-cpp`, and `oxidize-c`; AL5 is MSE-optimized 4-bit; prioritize speed without quality loss when tuning them.
- `oxidize-finetuning` exposes `self-train` CLI (`cargo run -p oxidize-finetuning -- self-train`): iterative LoRA SFT with per-round checkpoints, self-dialogue synthetic data (`synthetic.jsonl`), and optional self-critique; resume via `--resume-from`.
- DFlash speculative decoding in `oxidize-core/src/model/dflash.rs` is an active port target for `oxidize-golang` (and downstream Python); draft train/export lives under `scripts/dflash/`; inference needs a compatible target GGUF paired with the draft (hidden-size mismatch falls back to target-only).
- Rust `oxidize run` rewrites to `--serve-api` by default (background in-process server on `--api-host`/`--api-port`); realtime WebSocket at `ws://HOST:PORT/v1/realtime` (`oxidize-server/tests/realtime_ws.rs`).
- `oxidize-convert` converts HuggingFace SafeTensors (file or model directory with `config.json`) to GGUF; core logic in `oxidize-core/src/format/safetensors_to_gguf.rs`; K2 Horizon MoVA LoRA merge helper is `scripts/k2_merge_lora.py` (BF16 base: `Blackfrost-AI/K2-Horizon-MoVA-36B-A4B-DERISKED-BF16`; published Forge LoRA/GGUF: `freakyskittle/K2-Horizon-MoVA-36B-A4B-Forge` and `freakyskittle/K2-Horizon-MoVA-36B-A4B-Forge-GGUF`); K2 GGUFs need the `ifm-ai/llama.cpp` `model/K2Horizon` fork — stock llama.cpp cannot load them; day-to-day agent use on `dih@192.168.1.15` is often `oxidize-c` with ~262K context and compressed KV (Vulkan prefill is faster but can diverge from the CPU token stream); GLM 5.3 Flash uncensored (`BoldingBuilds/GLM-5.3-Flash-Uncensored-IQ1S`, model id `glm-5.3-flash`) needs `llama.cpp-bigmoe` on that host because `oxidize-c` cannot load `glm5next` yet — typically ~131K context on port 8083, which is not LAN-reachable so local Pi/OpenCode/Droid use an SSH tunnel; TTFT is seconds only when the prompt prefix matches the server cache.
- Go/Python ports and `oxidize-cpp` expose `--auto`, `--no-auto`, `--print-plan` autotune; on dual-socket CPU, dense models ≤192 GB use `--numa single --threads 16`, models >192 GB use `--numa interleave --threads 48`; test Go with `CGO_ENABLED=0`, Python with `uv run pytest` (`OXIDIZE_SLOW_TESTS=1` for slow GGUF).

## Cursor Cloud specific instructions
- The startup update script ensures the Rust `stable` toolchain (edition 2024 needs >= 1.85; the base image ships 1.83 which is too old), `cargo fetch`, Go module deps, and the Python port's `uv sync`. Standard build/test/run commands live in `Makefile`, `QUICKSTART.md`, and `HOW_TO_INSTALL.md`.
- Non-obvious gotcha: `make build` / `cargo build --workspace` currently FAILS to compile the optional `oxidize-finetuning` crate (`src/qlora.rs` borrow-check error, pre-existing). Build/test the core product per-crate instead, e.g. `cargo build -p oxidize-cli -p oxidize-server -p oxidize-quantize -p oxidize-convert` and `cargo test -p oxidize-core -p oxidize-cli -p oxidize-server -p oxidize-kernels`. The MUST product (CLI + server) is unaffected.
- `make lint` (clippy `-D warnings`), `make audit` (needs `cargo install cargo-deny`), and the Python `ruff check` all currently report pre-existing warnings/errors; these are code-quality debts, not environment breakage.
- CLI/server run with placeholder weights when no `--model` is given: `oxidize-cli --prompt ...` echoes the prompt and `/v1/chat/completions` returns an empty `chatcmpl-placeholder`. This is expected; real token generation requires a real GGUF (`--model path.gguf`, or an HF id via the resolver). Committed `oxidize-core/tests/fixtures/*.gguf` are tiny parser fixtures, not runnable models.
- Go port auto-downloads the `go1.26.2` toolchain via `GOTOOLCHAIN=auto` on first `go build`/`go test` (base image has Go 1.22); no manual Go upgrade needed.
- `uv` is installed to `~/.local/bin`; if not on PATH, invoke as `~/.local/bin/uv`. Run Python port commands from `oxidize-python/` (or pass `--directory oxidize-python`).
