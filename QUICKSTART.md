# Quick Start

## Prerequisites

- Rust toolchain (`rustup`, `cargo`) with edition 2024 support
- `make`
- Optional for WASM builds: `wasm-bindgen-cli`

## Build

```bash
git clone <your-fork-or-remote> oxidize
cd oxidize
make build
```

## Run

### CLI single prompt

```bash
cargo run -p oxidize-cli -- --prompt "hello"
```

### CLI with model + TPS tracking

```bash
cargo run -p oxidize-cli --release -- \
  --model /path/to/model.gguf \
  --prompt "Your prompt here" \
  --max-tokens 512 \
  --temperature 0.7
```

**Example with your model:**

```bash
cargo run -p oxidize-cli --release -- \
  --model "/run/media/dih/8CEDA5F938E73A48/AI/models/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf" \
  --prompt "Write a 3 page essay on why llama cpp is better than LM Studio" \
  --max-tokens 1500 \
  --temperature 0.7
```

**Generation parameters:**
- `--max-tokens N` - Maximum tokens to generate (default: 512)
- `--temperature T` - Sampling temperature 0.0-2.0 (default: 0.8)
- `--top-p P` - Nucleus sampling threshold (optional)
- `--top-k K` - Top-k sampling limit (optional)

**Output includes TPS tracking:**
```
generation stats: tokens=200 speed=4540.92 tok/s
```

### CLI chat mode

```bash
cargo run -p oxidize-cli -- --chat
```

### Server (OpenAI-compatible API)

```bash
cargo run -p oxidize-server -- --host 127.0.0.1 --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/healthz
```

### Quantization utility

```bash
cargo run -p oxidize-quantize -- \
  --input /path/to/input.bin \
  --output /path/to/output.bin \
  --source F32 \
  --target F16
```

## Test and lint

```bash
make test
make lint
```

## Available make targets

- `make fmt` - Check Rust formatting
- `make lint` - Run clippy with warnings denied
- `make audit` - Run cargo-deny license/security audit
- `make test` - Run workspace tests
- `make build` - Build release binaries for all targets
- `make wasm` - Build oxidize-core with wasm-bindgen output
- `make check` - Run fmt + lint + test
- `make ci` - Run check + build
