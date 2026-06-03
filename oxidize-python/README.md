# Oxidize Python

Pure-Python port of the Oxidize workspace. Mirrors `oxidize-core`, `oxidize-cli`, `oxidize-server`, `oxidize-quantize`, and `oxidize-train` without modifying the Rust crates.

`oxidize-py` remains the PyO3 bindings to the Rust core; this package is the standalone Python implementation.

## Quick start

```bash
cd oxidize-python
uv sync --extra dev
uv run pytest
uv run oxidize --help
```

## Layout

```
oxidize_python/
  core/          # inference engine (mirrors oxidize-core)
  internal/      # CLI, HTTP server, GGUF I/O
  quantize/      # offline weight conversion CLI
  train/         # CSV classifier training
```
