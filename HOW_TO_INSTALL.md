# How to Install `oxidize` (Agent Guide)

This guide is written for AI agents that need to install, build, and verify the `oxidize` workspace on a user's machine.

## 1. Preconditions (check before building)

The C engine needs a C compiler, `make`, and `git`. Stop if any of these fail:

```bash
cc --version
make --version
git --version
```

The Rust workspace additionally needs a toolchain that supports edition 2024 (`rustup update` if `cargo` reports that `edition2024` is missing). Skip the Rust checks when the user only asked for the C engine.

## 2. Choose an install path

| Goal | Command |
|------|---------|
| Build the C engine (primary) | `git clone` + `make` |
| Build the Rust workspace as well | `make rust-build` |
| Install the Rust `oxidize` binary globally | `cargo install --git` |

Default to the C engine.

## 3. Build the C engine

```bash
git clone https://github.com/Zapdev-labs/oxidize.git oxidize
cd oxidize
make
make -C oxidize-c test
```

`make` builds `oxidize-c/oxidize-c`. On a host whose `cc` is Clang without compiler-rt, run the tests with `CC=gcc make -C oxidize-c test`.

### Verify the build

```bash
ls -la oxidize-c/oxidize-c
./oxidize-c/oxidize-c --help
./oxidize-c/oxidize-c eagle4 --bench-iters 4
```

The eagle4 line prints low-rank head and sliding-window speedups plus the harness caps at batch 1, 64, and 384. It does not load a checkpoint.

## 3b. Optional Rust workspace

The Rust crates are in `oxidize-rust/`. The workspace manifest stays at the repo root.

```bash
make rust-build
sfw cargo run -p oxidize-cli --release -- --prompt "hello"
```

This needs a Rust toolchain that supports edition 2024. Prefix `cargo` with `sfw` when the environment uses Socket Firewall Free.

### Optional: run the Rust validation suite

```bash
make rust-check
```

This runs formatting, clippy, `cargo deny`, and the Rust tests.

If your environment requires every `cargo` invocation to go through Socket Firewall Free, run the individual `rust-*` commands from the `Makefile` instead of `make rust-check`.

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

```bash
./oxidize-c/oxidize-c eagle4 --bench-iters 4
```

If a Rust build was requested and a GGUF is available:

```bash
sfw cargo run -p oxidize-cli --release -- \
  --model /path/to/model.gguf \
  --prompt "Summarize ownership in one paragraph." \
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

- Whether you built the C engine, the Rust workspace, or both
- The path to `oxidize-c/oxidize-c` (and `target/release/oxidize` if Rust was built)
- The eagle4 microbench output
- Any warnings or skipped optional steps

## 8. Next steps

Point the user to:

- [`README.md`](./README.md) for EAGLE-4, the C engine, and the Rust crates
- `./oxidize-c/oxidize-c --help` for the C CLI
- [`QUICKSTART.md`](./QUICKSTART.md) for the Rust run/chat/server examples
