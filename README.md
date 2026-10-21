# oxidize

`oxidize` is a local LLM inference workspace. The primary engine is the dependency-free C11 port in `oxidize-c`. The Rust crates live under `oxidize-rust/` and remain members of the Cargo workspace at the repo root. Go, Python, and C++ ports sit beside them.

## Install with your AI agent

```text
Install oxidize from https://github.com/Zapdev-labs/oxidize.git.

Follow HOW_TO_INSTALL.md. The primary build is the C engine: `make` (or `make -C oxidize-c build`) and `make -C oxidize-c test`. Smoke-test with `./oxidize-c/oxidize-c --help` and `./oxidize-c/oxidize-c eagle4 --bench-iters 4`. The Rust workspace is optional: `make rust-build` needs a Rust toolchain that supports edition 2024. Report the binary path and the eagle4 microbench output. If a step fails, pause and ask before proceeding.
```

## Quick start

### Prerequisites

- A C11 compiler (`gcc` or `clang`), `make`
- Optional: Rust edition 2024 for `make rust-build`

### Clone and build

```bash
git clone https://github.com/Zapdev-labs/oxidize.git oxidize
cd oxidize
make
make -C oxidize-c test
./oxidize-c/oxidize-c --help
```

## EAGLE-4

EAGLE-4 speculative decoding lives in `oxidize-c`. The drafter is one decoder layer over fused target hidden states. Three things make a draft step cheap:

- Sliding-window attention (released shape: the last 4,096 positions) so draft attention stays flat as context grows. The key/value cache is append-only and rolls back to the accepted prefix.
- A low-rank candidate head. A rank-128 projection scores the vocabulary, then the full LM head runs only on the top 128 rows.
- A cost harness. At batch 1 it builds a tree (depth 12, branch 4, verify cap 16), branching widest on the first token. From batch 64 it switches to a chain of 4, and from batch 384 the chain cap is 3. A level is dropped when the expected acceptance gain per microsecond falls below the threshold.

Training reads SafeTensors and writes a SafeTensors draft. Example shards contain `features` `[N, extract_width]`, `tokens` `[N]`, and `targets` `[N]`. A path ending in `.gguf` is rejected by the trainer and by the loader.

```bash
./oxidize-c/oxidize-c eagle4-train \
  --model draft.safetensors \
  --dataset examples.safetensors \
  --output draft.safetensors \
  --lr 0.01 --epochs 1

./oxidize-c/oxidize-c eagle4-export \
  --model draft.safetensors \
  --output draft.gguf
```

`eagle4-export` packages a draft that was already trained. It is the step that produces the GGUF you publish. This repository does not ship draft weights. The SafeTensors checkpoint and the GGUF package will be published together when that drafter is ready.

`oxidize-c eagle4` times the head and the attention window on the machine you are on. On the build machine used for this change, hidden 1024 / rank 64 / vocab 16384 scored the low-rank head at about 7.6× a full head, and a 64-token window scored about 41× a 2,048-token attention scan. Those are kernel timings, not end-to-end tokens per second for a 27B model.

## Rust workspace

```bash
make rust-build
make rust-test
cargo run -p oxidize-cli -- --prompt "hello"
```

Crates: `oxidize-rust/oxidize-core`, `oxidize-cli`, `oxidize-server`, `oxidize-quantize`, `oxidize-convert`, `oxidize-prune`, `oxidize-merge`, `oxidize-finetuning`, `oxidize-kernels`, `oxidize-ffi`, `oxidize-py`, `oxidize-train`.

## Release announcement: oxidize 0.1.0

Today we are announcing `oxidize` `0.1.0`, the first stable workspace release for local-first LLM workflows in Rust.

This release brings together a complete core-to-interface stack:

- `oxidize-core` for model loading, quantization primitives, and generation
- `oxidize-cli` for prompt and chat runs with profiling hooks
- `oxidize-server` for OpenAI-compatible HTTP endpoints
- `oxidize-py` for Python integration
- `oxidize-quantize` for offline model conversion

What this means for early users:

- Start with the C engine (`make`, `make test`) or the Rust crates (`make rust-build`, `make rust-test`)
- Deploy the same inference behavior across CLI, server, and Python surfaces
- Tune memory and latency tradeoffs using quantization targets that fit your hardware

Thank you to everyone testing early builds and sharing feedback. `0.1.0` is our stability baseline, and future releases will focus on performance, platform parity, and better developer ergonomics.

## Common usage

### CLI single prompt

```bash
cargo run -p oxidize-cli -- --prompt "hello"
```

### CLI chat mode

```bash
cargo run -p oxidize-cli -- --chat
```

### CLI with model loading + GPU planning

```bash
cargo run -p oxidize-cli -- --model /path/to/model.gguf --n-gpu-layers 20 --gpus 2 --parallelism pipeline
```

### Terminal UI

`oxidize-tui` is a full-screen console over the engine: chat with streaming, a GGUF model
browser, live server metrics, and the server log — with a `ctrl+k` command palette. It
drives `oxidize serve` for you, or attaches to a server you already have running.

```bash
cd oxidize-tui && bun install
bun run start                                   # browse and load a model
bun run start -- /path/to/model.gguf --backend cuda
bun run start -- --api http://127.0.0.1:8080    # attach to a running server
```

### Server (OpenAI-compatible endpoints)

```bash
cargo run -p oxidize-server -- --host 127.0.0.1 --port 8080
```

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

### Batch processing

```bash
cargo run -p oxidize-cli -- \
  --model /path/to/model.gguf \
  --batch-size 4 \
  --prompt "Classify: Rust is memory-safe."
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
2. Profile one run to find bottlenecks:
   ```bash
   cargo run -p oxidize-cli -- --model /path/to/model.gguf --prompt "benchmark prompt" --profile perf
   ```
3. Increase GPU offload gradually (`--n-gpu-layers`) and compare throughput after each step.
4. If using multiple GPUs, test both parallel strategies and keep the faster one for your hardware:
   ```bash
   cargo run -p oxidize-cli -- --model /path/to/model.gguf --gpus 2 --n-gpu-layers 20 --parallelism pipeline
   cargo run -p oxidize-cli -- --model /path/to/model.gguf --gpus 2 --n-gpu-layers 20 --parallelism tensor
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
- **Server auth failures (`401`):** set `OXIDIZE_API_KEY` before starting `oxidize-server`, then send the same value with `x-api-key` or `Authorization: Bearer <key>`.
- **WASM build failures:** install the wasm target (`rustup target add wasm32-unknown-unknown`) and ensure `wasm-bindgen` is available, then run `make rust-wasm` again.
- **Unexpected output quality after quantization:** keep the original model, benchmark both variants on representative prompts, and move to a less aggressive target if quality regresses.

## Workspace commands

- Build the C engine: `make`
- Test the C engine: `make test`
- Build Rust crates (release): `make rust-build`
- Test Rust crates: `make rust-test`
- Clippy: `make rust-lint`
- Format check: `make rust-fmt`
- Rust CI-equivalent: `make rust-ci`

## Environment variables

- `OXIDIZE_API_KEY`: optional API key for `oxidize-server` `/v1/*` routes. Supports `x-api-key` or `Authorization: Bearer <key>`.
- `OXIDIZE_PROFILE_CHILD`: internal flag used by `oxidize-cli` profiling flow.

## Architecture

`oxidize` is organized as a layered workspace:

- **Core compute layer (`oxidize-core`)**: owns GGUF parsing, tensor + quantization primitives, model loading, token generation loop, and backend-specific execution paths (CPU, CUDA, Metal, WASM).
- **Interface layer (`oxidize-cli`, `oxidize-server`, `oxidize-py`)**: exposes core capabilities through a CLI, OpenAI-compatible HTTP routes, and Python bindings without duplicating inference logic.
- **Utility layer (`oxidize-quantize`)**: handles offline model weight conversion and quantization workflows.
- **Terminal UI (`oxidize-tui`)**: an OpenTUI/Bun front end that talks to `oxidize-server` over HTTP/SSE rather than linking the engine, so it stays a pure client.

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

## License

Zapdev labs Fair Use License, in `LICENSE`. Individuals, non-commercial entities, and academic institutions may use the Software without charge. A commercial entity with more than 10 employees or more than $1,000,000 USD annual gross revenue needs a separate commercial license from Zapdev-labs (`jacksonwheeler@zapdev.link`). Vendored Criterion under `oxidize-c/tests/criterion/` remains MIT. Model weights are not part of this grant.
