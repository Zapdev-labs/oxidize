# How to Install `oxidize` (Agent Guide)

This guide is written for AI agents that need to install, build, and verify the `oxidize` workspace on a user's machine.

## 1. Preconditions (check before building)

Run these checks and stop with a clear error message if any fail:

```bash
# Rust toolchain (edition 2024 required)
rustup --version
cargo --version
rustc --version
```

Required minimums: stable Rust that supports edition 2024. If `rustup` is missing, install it from <https://rustup.rs/> before continuing.

```bash
# make
make --version
```

```bash
# git
git --version
```

## 2. Choose an install path

The user usually wants one of two outcomes:

| Goal | Command |
|------|---------|
| Build from source for development / local runs | `git clone` + `make build` |
| Install the `oxidize` binary globally | `cargo install --git` |

Ask the user which they prefer if it is unclear. Default to **build from source**.

## 3. Build from source

```bash
# Clone
git clone https://github.com/Zapdev-labs/oxidize.git oxidize
cd oxidize

# Build the workspace in release mode
make build
```

This runs `cargo build --workspace --release` and produces release binaries under `target/release/`.

> **Agent note:** `make build` is the workspace's documented build entry point. For direct `cargo` commands (run, test, etc.), prefix with `sfw` if your environment uses Socket Firewall Free.

### Verify the build

```bash
# Confirm the main CLI binary exists
ls -la target/release/oxidize

# Quick smoke test
sfw cargo run -p oxidize-cli --release -- --prompt "hello"
```

Expected output includes a generated response and no panic.

### Optional: run the full validation suite

```bash
make check
```

This runs formatting, linting, audit, and tests. It is slower but proves the workspace is healthy.

If your environment requires every `cargo` invocation to go through Socket Firewall Free, run the individual commands from the `Makefile` instead of `make check`.

## 4. Install the binary globally (alternative)

If the user wants a system-wide `oxidize` command:

```bash
sfw cargo install --git https://github.com/Zapdev-labs/oxidize.git oxidize-cli --bin oxidize --locked
```

Important: the workspace ships multiple binary crates, so you **must** name `oxidize-cli` explicitly and request the `oxidize` binary.

Verify:

```bash
which oxidize
oxidize --help
```

## 5. Verify the installation

Run the smallest possible end-to-end check:

```bash
sfw cargo run -p oxidize-cli --release -- --prompt "What is 2+2?"
```

If a model path is available, run with it:

```bash
sfw cargo run -p oxidize-cli --release -- \
  --model /path/to/model.gguf \
  --prompt "Summarize Rust ownership in one paragraph." \
  --max-tokens 128
```

## 6. Common failures and recovery

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cargo: command not found` | Rust not installed | Install `rustup` from <https://rustup.rs/> |
| `error: package ... requires edition 2024` | Rust toolchain too old | `rustup update` |
| `make: command not found` | `make` missing | Install `build-essential` (Linux) or Xcode Command Line Tools (macOS) |
| Build is very slow | Release build with LTO | Normal; first build can take several minutes |
| `cargo install` picks wrong binary | Missing `--bin oxidize` | Re-run with `oxidize-cli --bin oxidize` |
| WASM build fails | Missing `wasm32-unknown-unknown` target or `wasm-bindgen` | Run `rustup target add wasm32-unknown-unknown` and `cargo install --locked wasm-bindgen-cli --version 0.2.120` |

## 7. What to report back

After installation succeeds, tell the user:

- Whether you built from source or installed globally
- The path to the `oxidize` binary (`target/release/oxidize` or `~/.cargo/bin/oxidize`)
- The output of the verification command
- Any warnings or skipped optional steps

## 8. Next steps

Point the user to:

- [`QUICKSTART.md`](./QUICKSTART.md) for run/chat/server examples
- [`README.md`](./README.md) for the full feature overview and tuning guide
- `oxidize --help` for CLI flags
