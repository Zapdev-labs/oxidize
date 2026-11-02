# oxidize

`oxidize` is a local-first LLM inference stack. A Rust workspace holds the core
engine and every interface around it; sibling directories carry ports of the
engine to Go, Python, C, and C++.

## Workspace layout

### Rust crates

| Crate | Purpose |
|-------|---------|
| `oxidize-core` | Engine: GGUF/SafeTensors loading, tokenizers, tensor + quantization kernels, KV cache, sampling, generation, GPU backends, paged attention, distributed mesh, vision/video multimodal, autotune, speculative decoding |
| `oxidize-cli` | Ollama-style CLI (`oxidize run`, `serve`, `list`, `gpu-cluster`) plus benchmarking/profiling and diagnostic binaries |
| `oxidize-server` | OpenAI-compatible HTTP/WebSocket API (Axum) |
| `oxidize-quantize` | Offline weight quantization utility |
| `oxidize-convert` | SafeTensors → GGUF conversion (with optional pruning) |
| `oxidize-prune` | Wanda/magnitude pruning |
| `oxidize-merge` | Checkpoint merging (linear/SLERP) |
| `oxidize-finetuning` | LoRA / SFT / self-train / DPO / PPO |
| `oxidize-kernels` | OXK hand-tuned CPU GEMV kernels (bit-exact Q4_K×Q8_K) |
| `oxidize-ffi` | C-ABI FFI over `oxidize-core` |
| `oxidize-py` | Python bindings (PyO3 + maturin) |
| `oxidize-train` | CSV classifier + video training |

### Ports and front ends

| Directory | What it is |
|-----------|------------|
| `oxidize-tui` | OpenTUI/Bun terminal UI — a pure client over the HTTP/SSE API |
| `oxidize-golang` | Go port of `oxidize-core` (`run`, `chat`, `bench`, `inspect`, `list`, `serve`) |
| `oxidize-python` | Pure-Python port (uv/pytest), mirrors `oxidize-golang` |
| `oxidize-c` | Dependency-free C11 port, including native EXL3 inference |
| `oxidize-cpp` | C++20 Llama-family engine; llama.cpp-parity CPU path + optional CUDA/ROCm |
| `oxidize-training` | Standalone C training stack (GPT trainer + corpus/eval tools) |

## Install with your AI agent

Copy and paste the prompt below into your agent (Claude, Cursor, Codex, etc.) and it will install `oxidize` for you.

```text
Install the oxidize LLM inference workspace from https://github.com/Zapdev-labs/oxidize.git.

Follow the exact steps in HOW_TO_INSTALL.md in the repo root. Before building, verify that rustup, cargo, make, and git are available and that the Rust toolchain supports edition 2024. Build the workspace from source with `make build` (prefix cargo commands with `sfw` if your environment uses Socket Firewall Free). After building, run a smoke test with `target/release/oxidize run <path-or-hf-repo> "What is 2+2?"` and confirm there is no panic. Report back the install method used, the path to the `oxidize` binary, and the verification output. If anything fails, pause and ask before proceeding.
```

## Quick start

### Prerequisites

- Rust toolchain (`rustup`, `cargo`) with edition 2024 support
- `make`
- Optional for WASM builds: `wasm-bindgen-cli`
- Optional for the TUI: `bun`

### Clone and build

```bash
git clone https://github.com/Zapdev-labs/oxidize.git oxidize
cd oxidize
make build
```

Or install the `oxidize` binary directly:

```bash
cargo install --git https://github.com/Zapdev-labs/oxidize.git oxidize-cli --bin oxidize
```

### Run a model

```bash
# Local GGUF file, interactive chat + local API server
oxidize run ./models/Qwen3-4B-Q4_K_M.gguf

# One-shot prompt (no server)
oxidize run ./models/model.gguf "What is 2+2?"

# Hugging Face GGUF repo (downloads to the local HF cache)
oxidize run Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --file qwen2.5-0.5b-instruct-q4_k_m.gguf --chat
```

`oxidize run` accepts a local `.gguf` path or a `user/repo` Hugging Face spec.
HF repos that only ship SafeTensors weights are converted to GGUF on the fly.
With no prompt it enters chat mode and starts the OpenAI-compatible API server
(`--no-api` disables the server). Autotune is on by default — inspect the
resolved plan with `--print-plan yes` (or `json`) or opt out with `--no-auto`.

Other commands: `oxidize list` (models in `./models`), `oxidize serve [model]`
(server only), `oxidize gpu-cluster`, and `oxidize --help`.

### Run tests and lint

```bash
make test
make lint
```

### Fast local validation

```bash
make fmt
make check
```

## Features

- **Model formats:** GGUF v3 and SafeTensors (memory-mapped); four tokenizer
  families (SentencePiece, WordPiece, BPE, Tiktoken)
- **Architectures:** Llama, Mistral, Mixtral, DeepSeek (V2/V3/MoE), Qwen
  (incl. Qwen3/Qwen3.5 dense + MoE), Gemma, Phi, Falcon, GPT-2/GPT-J/GPT-NeoX,
  MiniMax, LFM2 (+MoE), GLM MoE (DSA), Hunyuan MoE
- **Quantization:** F32/F16/BF16, legacy quants (`Q4_0`–`Q8_0`), k-quants
  (`Q2_K`–`Q6_K`), i-quants (`IQ*`), `AL*`, NVFP4
- **Backends:** CPU (scalar/AVX2/AVX-512 via OXK kernels, NEON SIMD in core),
  CUDA, ROCm, Metal, MLX (macOS), Vulkan, Intel Arc; WebGPU for WASM/browser targets
- **KV cache:** TurboQuant block quantization by default for q4/q8 caches
  (`--no-turboquant` for the legacy asymmetric quantizer); `f32`/`f16`/`q8`/`q4`
  dtypes via `--kv-cache-dtype`
- **Speculative decoding:** DFlash draft models (`--draft-model`,
  `--draft-tokens`), native in-GGUF MTP/nextn (`--no-mtp` disables), QuantSpec
  self-speculation (`--quantspec`)
- **Autotune:** hardware detection + tuning plan resolves threads, context
  size, KV dtype, GPU offload, and mmap strategy (`--auto`, `--no-auto`,
  `--print-plan`)
- **Serving:** OpenAI-compatible endpoints + realtime WebSocket, continuous
  batching, a vLLM-style paged-attention scheduler, Prometheus metrics
- **Scale-out:** libp2p mesh inference (`--mesh`), two-node pipeline-parallel
  decode (`--pipe-head`/`--pipe-tail`), multi-GPU tensor/pipeline parallelism
- **Multimodal:** CLIP-style vision encoder (`--vision --image <path>`), plus a
  video frame-sampling path in `oxidize-core`

## Common usage

### CLI single prompt

```bash
cargo run -p oxidize-cli -- --model /path/to/model.gguf --prompt "hello"
```

### CLI chat mode

```bash
cargo run -p oxidize-cli -- --model /path/to/model.gguf --chat
```

### CLI with model loading + GPU planning

```bash
cargo run -p oxidize-cli -- --model /path/to/model.gguf --backend cuda \
  --n-gpu-layers 20 --gpus 2 --parallelism pipeline
```

### Speculative decoding with a DFlash draft model

```bash
cargo run -p oxidize-cli -- --model /path/to/target.gguf \
  --draft-model /path/to/draft.gguf --draft-tokens 4 \
  --prompt "Write a haiku about borrow checkers"
```

### Terminal UI

`oxidize-tui` is a full-screen console over the engine: chat with streaming, a
GGUF model browser, live server metrics, and the server log — with a `ctrl+k`
command palette. It drives `oxidize serve` for you, or attaches to a server you
already have running.

```bash
cd oxidize-tui && bun install
bun run start                                   # browse and load a model
bun run start -- /path/to/model.gguf --backend cuda
bun run start -- --api http://127.0.0.1:8080    # attach to a running server
```

### Server (OpenAI-compatible endpoints)

```bash
# via the CLI
oxidize serve /path/to/model.gguf --host 127.0.0.1 --port 8080

# or the dedicated binary
cargo run -p oxidize-server -- --model /path/to/model.gguf --host 127.0.0.1 --port 8080
```

Endpoints:

- `POST /v1/chat/completions` (SSE streaming supported)
- `POST /v1/completions`
- `GET  /v1/models`
- `POST /v1/embeddings`
- `GET  /v1/realtime` (WebSocket)
- `POST /v1/mesh/chat/completions` (mesh master nodes)
- `GET  /healthz` `/livez` `/readyz` `/metrics` `/openapi.json`

Health checks:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/openapi.json
```

### Quantization utility

```bash
cargo run -p oxidize-quantize -- \
  --input /path/to/input.bin \
  --output /path/to/output.bin \
  --source F32 \
  --target F16
```

#### Quantization guide

1. Start from a floating-point model file (`F32` or `F16`) and pick a target that matches your latency/quality tradeoff.
2. Use `F16` for a low-risk size reduction, or lower-bit targets (`Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`) for stronger memory savings.
3. Quantize to a new output path and keep the source model unchanged so you can benchmark both variants.
4. Run inference/perplexity checks on representative prompts before promoting the quantized model.

Example:

```bash
cargo run -p oxidize-quantize -- \
  --input /models/model-f32.bin \
  --output /models/model-q4_0.bin \
  --source F32 \
  --target Q4_0
```

## Examples

### Basic inference

```bash
cargo run -p oxidize-cli -- \
  --model /path/to/model.gguf \
  --prompt "Summarize Rust ownership in one paragraph."
```

### Chat completion

```bash
cargo run -p oxidize-cli -- \
  --model /path/to/model.gguf \
  --chat
```

### Streaming generation

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role":"user","content":"Write a short poem about systems programming."}],
    "stream": true
  }'
```

### SafeTensors → GGUF conversion

```bash
cargo run -p oxidize-convert -- \
  --input /path/to/model-dir \
  --output /path/to/model.gguf \
  --target Q4_K_M
```

### Custom sampling

```bash
cargo run -p oxidize-cli -- \
  --model /path/to/model.gguf \
  --prompt "Generate a release note title." \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 40
```

### Embedding extraction

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": ["Rust is fast.", "Rust is memory-safe."]
  }'
```

### WASM build

```bash
make wasm
```

WASM artifacts are written to `dist/wasm`.

## Performance tuning guide

Use this loop for predictable, low-risk tuning:

1. Start with a stable baseline and record tokens/sec and latency:
   ```bash
   cargo run -p oxidize-cli -- --model /path/to/model.gguf --prompt "benchmark prompt"
   ```
2. Let the autotuner pick a plan, then inspect what it chose:
   ```bash
   cargo run -p oxidize-cli -- --model /path/to/model.gguf --prompt "benchmark prompt" --print-plan yes
   ```
3. Profile one run to find bottlenecks:
   ```bash
   cargo run -p oxidize-cli -- --model /path/to/model.gguf --prompt "benchmark prompt" --profile perf
   ```
4. Increase GPU offload gradually (`--n-gpu-layers`) and compare throughput after each step.
5. If using multiple GPUs, test both parallel strategies and keep the faster one for your hardware:
   ```bash
   cargo run -p oxidize-cli -- --model /path/to/model.gguf --gpus 2 --n-gpu-layers 20 --parallelism pipeline
   cargo run -p oxidize-cli -- --model /path/to/model.gguf --gpus 2 --n-gpu-layers 20 --parallelism tensor
   ```
6. Try speculative decoding (`--draft-model`, `--quantspec`) and a quantized KV
   cache (`--kv-cache-dtype f16` or `q8`) before reaching for weight
   quantization; then re-run the same benchmark prompts and check quality.

Practical tuning priorities:

- **Largest speed gains first:** GPU layer offload, multi-GPU strategy, and speculative decoding.
- **Memory pressure next:** KV cache dtype (`f16`, `q8`, `q4` with TurboQuant), then weight quantization.
- **Stability before peak speed:** benchmark with representative prompts, not one short prompt.
- **Measure every change:** keep a small log of config -> tokens/sec -> latency -> quality notes.

## Troubleshooting guide

- **Model path errors:** verify the model file exists and is readable, then rerun with an absolute path to avoid shell-relative path mistakes. HF specs must be `user/repo`; use `--file` when the repo ships several GGUF variants.
- **Slow or no GPU acceleration:** check that the matching backend feature was compiled in, increase `--n-gpu-layers` gradually, and compare throughput after each change; if speed does not improve, fall back to `--backend cpu` and confirm baseline correctness.
- **Server auth failures (`401`):** set `OXIDIZE_API_KEY` before starting `oxidize-server`, then send the same value with `x-api-key` or `Authorization: Bearer <key>`.
- **WASM build failures:** install the wasm target (`rustup target add wasm32-unknown-unknown`) and ensure `wasm-bindgen` is available, then run `make wasm` again.
- **Unexpected output quality after quantization:** keep the original model, benchmark both variants on representative prompts, and move to a less aggressive target if quality regresses.
- **Whole-workspace build issues:** `cargo build --workspace` covers every crate; if one crate fails you can still build the core product crates individually (`cargo build -p oxidize-cli -p oxidize-server`).

## Workspace commands

- Build workspace crates (release): `make build`
- Test all targets: `make test`
- Lint with denied warnings: `make lint`
- Format check: `make fmt`
- License/security audit (cargo-deny): `make audit`
- Unused dependency check (nightly + cargo-udeps): `make udeps`
- fmt + lint + audit + test: `make check`
- Full CI-equivalent check + build: `make ci`
- WASM build to `dist/wasm`: `make wasm`

## Environment variables

- `OXIDIZE_API_KEY` / `OXIDIZE_API_KEYS`: optional API key(s) for `oxidize-server` `/v1/*` routes. Supports `x-api-key` or `Authorization: Bearer <key>`.
- `OXIDIZE_PROFILE_CHILD`: internal flag used by `oxidize-cli` profiling flow.

Hugging Face downloads and converted models are cached under
`~/.cache/oxidize/hf`.

## Architecture

`oxidize` is organized as a layered workspace:

- **Core compute layer (`oxidize-core`)**: owns GGUF/SafeTensors parsing, tensor + quantization primitives, model loading, token generation loop, KV cache and paged attention, speculative decoding, multimodal encoders, the distributed mesh, and backend-specific execution paths (CPU, CUDA, ROCm, Metal, MLX, Vulkan, WebGPU, WASM).
- **Interface layer (`oxidize-cli`, `oxidize-server`, `oxidize-py`, `oxidize-ffi`)**: exposes core capabilities through the `oxidize` CLI, OpenAI-compatible HTTP/WebSocket routes, Python bindings, and a C ABI without duplicating inference logic.
- **Utility layer (`oxidize-quantize`, `oxidize-convert`, `oxidize-prune`, `oxidize-merge`, `oxidize-finetuning`, `oxidize-train`, `oxidize-kernels`)**: offline weight workflows (conversion, quantization, pruning, merging), training tools, and the OXK CPU kernel crate.
- **Terminal UI (`oxidize-tui`)**: an OpenTUI/Bun front end that talks to `oxidize-server` over HTTP/SSE rather than linking the engine, so it stays a pure client.
- **Language ports (`oxidize-golang`, `oxidize-python`, `oxidize-c`, `oxidize-cpp`, `oxidize-training`)**: independent reimplementations that mirror the Rust structure for cross-platform deployment; they do not link the Rust crates at runtime.

At runtime, request flow is: input prompt -> interface crate -> `oxidize-core` model/session setup -> token generation + sampling -> streamed or buffered output to the caller.

Design goals:

- Keep inference and model logic centralized in `oxidize-core` so all frontends share the same behavior.
- Keep transport/UI concerns at the edge crates (`oxidize-cli`, `oxidize-server`, `oxidize-py`) for maintainability.
- Support multiple acceleration targets behind stable core APIs to keep feature parity across platforms.

### EXL3 on CPU in oxidize-c

Build with `make -C oxidize-c`, then load the native EXL3 checkpoint directory:

```bash
./oxidize-c/oxidize-c --model /path/to/exl3/ \
  --prompt "Hello" --n-predict 64 --threads 16
```

The directory supplies `config.json`, `tokenizer.json`, optional tokenizer
configuration, and the `.safetensors` weight shards. A path to a shard is also
accepted; companion assets and other shards are resolved in its directory.
No companion GGUF or weight conversion is required. Packed EXL3 projections,
ordinary embeddings, norms and unquantized projections come from the same
checkpoint.

Native tokenizer import supports byte-level BPE with recognized GPT-2,
Llama 3 and Qwen2 split rules, including `ignore_merges`, and raw Unigram
tokenizers. Unsupported normalization/Metaspace pipelines and RoPE scaling
configurations are rejected explicitly.

The earlier `--model matching.gguf --exl3 /path/to/exl3/` overlay form remains
available for explicit mixed-checkpoint use. Those inputs must come from the
same model and revision.

The CPU path keeps native trellis weights compressed, applies 128-point
Hadamard transforms to activations, and decodes tiles during multiplication.
It supports the 3inst, MCG and mul1 codebooks, integer rates 1–8 bits, and
mul1 rates 1.5/2.5/3.5 bits, including FP16 channel scales and legacy packed
signs. Initial model integration supports dense Llama/Mistral/Qwen2 projections
and the output head, plus qwen4_exp checkpoints natively — those are hybrid
MoE models (EXL3 experts, delta-net and full-attention layers,
hyper-connections, and the MTP draft head), not dense. Unsupported
architectures, tensor groups and execution modes fail explicitly. GPU
execution, expert streaming, speculative decoding
and serving with EXL3 are unavailable. Speed and model-quality claims
require full-checkpoint benchmarks; fewer stored bits do not guarantee faster
CPU decoding.

Format reference: [ExLlamaV3 CPU oracle](https://github.com/turboderp-org/exllamav3/blob/058b8630fd081f3ff46fce6246955b157c3a2e3b/tests/testlib/trellis.py).

## Security

See [SECURITY_MODEL.md](SECURITY_MODEL.md) for the threat model, trust boundaries, and existing controls.

## License

Zapdev labs Fair Use License
