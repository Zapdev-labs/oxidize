# How to Install `oxidize` (Agent Guide)

This guide is for agents installing the oxidize **C runtime** and optional TypeScript TUI.

## 1. Preconditions

```bash
cc --version    # or gcc / clang
make --version
git --version
```

For the TUI:

```bash
bun --version
```

Install Bun from https://bun.sh if missing. Do **not** require Rust, Go, Python, or C++ toolchains.

## 2. Build from source

```bash
git clone https://github.com/Zapdev-labs/oxidize.git oxidize
cd oxidize
make build
ls -la oxidize-c/oxidize-c
./oxidize-c/oxidize-c --version
./oxidize-c/oxidize-c --prompt "hello"
```

With a GGUF:

```bash
./oxidize-c/oxidize-c --model <path.gguf> --prompt "Summarize in one paragraph." --n-predict 128
```

TUI:

```bash
cd oxidize-tui && bun install
bun run typecheck
bun test
```

## 3. Verify

`make test` runs the C suite (ASan/UBSan). `make tui-test` runs the TUI tests. Report the path to `oxidize-c/oxidize-c` and the `--version` output.
