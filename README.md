# oxidize

Local-first LLM inference. The **runtime is C11** (`oxidize-c`). The **terminal UI is TypeScript** (`oxidize-tui`, Bun + OpenTUI). Other language ports are not part of this product.

## Quick start

```bash
git clone https://github.com/Zapdev-labs/oxidize.git oxidize
cd oxidize
make build
./oxidize-c/oxidize-c --prompt "hello"
```

OpenAI-compatible HTTP API:

```bash
./oxidize-c/oxidize-c serve /path/to/model.gguf --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/healthz
```

Terminal UI (spawns `oxidize-c serve` for you):

```bash
cd oxidize-tui && bun install
bun run start
bun run start -- /path/to/model.gguf --backend cpu
bun run start -- --api http://127.0.0.1:8080
```

## Layout

| Path | Role |
|------|------|
| `oxidize-c/` | Inference engine, GGUF, CLI, HTTP/SSE, CUDA optional |
| `oxidize-tui/` | Full-screen TUI; HTTP client only, does not link the engine |

## Build / test

```bash
make build          # gcc -std=c11, libc + libm + pthread
make test           # C tests with ASan/UBSan
make tui-test       # bun test
make check          # both
make -C oxidize-c cuda   # optional NVIDIA build
```

Binary: `oxidize-c/oxidize-c`. Point the TUI at it with `OXIDIZE_BIN` if it is installed elsewhere.

## CLI

```bash
./oxidize-c/oxidize-c --model model.gguf --prompt "Summarize this." --n-predict 128
./oxidize-c/oxidize-c serve model.gguf --host 127.0.0.1 --port 8080
./oxidize-c/oxidize-c inspect model.gguf
./oxidize-c/oxidize-c --help
```

`serve` accepts a positional model path (what the TUI passes) or `--model`.
