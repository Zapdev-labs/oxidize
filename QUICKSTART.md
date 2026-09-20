# Quick Start

## Prerequisites

- C11 compiler (`gcc` or `clang`) and `make`
- Optional TUI: [Bun](https://bun.sh)

## Build

```bash
git clone <your-fork-or-remote> oxidize
cd oxidize
make build
```

## Run

```bash
./oxidize-c/oxidize-c --prompt "hello"

./oxidize-c/oxidize-c --model /path/to/model.gguf \
  --prompt "Your prompt here" \
  --n-predict 512 \
  --temperature 0.7

./oxidize-c/oxidize-c serve /path/to/model.gguf --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/healthz
```

## Terminal UI

```bash
cd oxidize-tui && bun install
bun run start -- /path/to/model.gguf
```
