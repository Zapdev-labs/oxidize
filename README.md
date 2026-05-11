# llamas-cpp

`llamas-cpp` is a Rust workspace for local LLM tooling:

- `llamas-core`: model loading, quantization, tensor/sampling primitives, and optional WASM support
- `llamas-cli`: local CLI for prompt runs, chat mode, model planning, and profiling hooks
- `llamas-server`: OpenAI-compatible HTTP API surface
- `llamas-quantize`: file quantization utility
- `llamas-py`: Python bindings built with `pyo3`

## Quick start

### Prerequisites

- Rust toolchain (`rustup`, `cargo`) with edition 2024 support
- `make`
- Optional for WASM builds: `wasm-bindgen-cli`

### Clone and build

```bash
git clone https://github.com/Zapdev-labs/llamas-cpp llamas-cpp
cd llamas-cpp
make build
```

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

## Release announcement: llamas-cpp 0.1.0

Today we are announcing `llamas-cpp` `0.1.0`, the first stable workspace release for local-first LLM workflows in Rust.

This release brings together a complete core-to-interface stack:

- `llamas-core` for model loading, quantization primitives, and generation
- `llamas-cli` for prompt and chat runs with profiling hooks
- `llamas-server` for OpenAI-compatible HTTP endpoints
- `llamas-py` for Python integration
- `llamas-quantize` for offline model conversion

What this means for early users:

- Start quickly with one workspace and consistent commands (`make build`, `make test`, `make lint`)
- Deploy the same inference behavior across CLI, server, and Python surfaces
- Tune memory and latency tradeoffs using quantization targets that fit your hardware

Thank you to everyone testing early builds and sharing feedback. `0.1.0` is our stability baseline, and future releases will focus on performance, platform parity, and better developer ergonomics.

## Common usage

### CLI single prompt

```bash
cargo run -p llamas-cli -- --prompt "hello"
```

### CLI chat mode

```bash
cargo run -p llamas-cli -- --chat
```

### CLI with model loading + GPU planning

```bash
cargo run -p llamas-cli -- --model /path/to/model.gguf --n-gpu-layers 20 --gpus 2 --parallelism pipeline
```

### Server (OpenAI-compatible endpoints)

```bash
cargo run -p llamas-server -- --host 127.0.0.1 --port 8080
```

Health checks:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/openapi.json
```

### Quantization utility

```bash
cargo run -p llamas-quantize -- \
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
cargo run -p llamas-quantize -- \
  --input /models/model-f32.bin \
  --output /models/model-q4_0.bin \
  --source F32 \
  --target Q4_0
```

## Examples

### Basic inference

```bash
cargo run -p llamas-cli -- \
  --model /path/to/model.gguf \
  --prompt "Summarize Rust ownership in one paragraph."
```

### Chat completion

```bash
cargo run -p llamas-cli -- \
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

### Batch processing

```bash
cargo run -p llamas-cli -- \
  --model /path/to/model.gguf \
  --batch-size 4 \
  --prompt "Classify: Rust is memory-safe."
```

### Custom sampling

```bash
cargo run -p llamas-cli -- \
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
   cargo run -p llamas-cli -- --model /path/to/model.gguf --prompt "benchmark prompt"
   ```
2. Profile one run to find bottlenecks:
   ```bash
   cargo run -p llamas-cli -- --model /path/to/model.gguf --prompt "benchmark prompt" --profile perf
   ```
3. Increase GPU offload gradually (`--n-gpu-layers`) and compare throughput after each step.
4. If using multiple GPUs, test both parallel strategies and keep the faster one for your hardware:
   ```bash
   cargo run -p llamas-cli -- --model /path/to/model.gguf --gpus 2 --n-gpu-layers 20 --parallelism pipeline
   cargo run -p llamas-cli -- --model /path/to/model.gguf --gpus 2 --n-gpu-layers 20 --parallelism tensor
   ```
5. Quantize only after offload strategy is stable; then re-run the same benchmark prompts and check quality.

Practical tuning priorities:

- **Largest speed gains first:** GPU layer offload and multi-GPU strategy.
- **Memory pressure next:** quantization target (`F16`, `Q8_0`, `Q5_*`, `Q4_*`).
- **Stability before peak speed:** benchmark with representative prompts, not one short prompt.
- **Measure every change:** keep a small log of config -> tokens/sec -> latency -> quality notes.

## Troubleshooting guide

- **Model path errors:** verify the model file exists and is readable, then rerun with an absolute path to avoid shell-relative path mistakes.
- **Slow or no GPU acceleration:** increase `--n-gpu-layers` gradually and compare throughput after each change; if speed does not improve, fall back to CPU and confirm baseline correctness.
- **Server auth failures (`401`):** set `LLAMAS_API_KEY` before starting `llamas-server`, then send the same value with `x-api-key` or `Authorization: Bearer <key>`.
- **WASM build failures:** install the wasm target (`rustup target add wasm32-unknown-unknown`) and ensure `wasm-bindgen` is available, then run `make wasm` again.
- **Unexpected output quality after quantization:** keep the original model, benchmark both variants on representative prompts, and move to a less aggressive target if quality regresses.

## Workspace commands

- Build all targets (release): `make build`
- Test all targets: `make test`
- Lint with denied warnings: `make lint`
- Format check: `make fmt`
- Full CI-equivalent check + build: `make ci`

## Environment variables

- `LLAMAS_API_KEY`: optional API key for `llamas-server` `/v1/*` routes. Supports `x-api-key` or `Authorization: Bearer <key>`.
- `LLAMAS_PROFILE_CHILD`: internal flag used by `llamas-cli` profiling flow.

## Architecture

`llamas-cpp` is organized as a layered workspace:

- **Core compute layer (`llamas-core`)**: owns GGUF parsing, tensor + quantization primitives, model loading, token generation loop, and backend-specific execution paths (CPU, CUDA, Metal, WASM).
- **Interface layer (`llamas-cli`, `llamas-server`, `llamas-py`)**: exposes core capabilities through a CLI, OpenAI-compatible HTTP routes, and Python bindings without duplicating inference logic.
- **Utility layer (`llamas-quantize`)**: handles offline model weight conversion and quantization workflows.

At runtime, request flow is: input prompt -> interface crate -> `llamas-core` model/session setup -> token generation + sampling -> streamed or buffered output to the caller.

Design goals:

- Keep inference and model logic centralized in `llamas-core` so all frontends share the same behavior.
- Keep transport/UI concerns at the edge crates (`llamas-cli`, `llamas-server`, `llamas-py`) for maintainability.
- Support multiple acceleration targets behind stable core APIs to keep feature parity across platforms.

## License

MIT
