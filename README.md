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
git clone <your-fork-or-remote> llamas-cpp
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

### WASM build

```bash
make wasm
```

WASM artifacts are written to `dist/wasm`.

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
