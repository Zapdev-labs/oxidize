# oxidize-py

**Generated:** 2026-06-03
**Domain:** Python bindings via PyO3 + maturin (Rust-in-Python)

## OVERVIEW
PyO3-based Python extension module exposing `oxidize-core` types to Python. Provides a `Llama` class with `generate()`, `generate_async()`, and `create_chat_completion()` methods. Built with `maturin`.

## STRUCTURE
```
oxidize-py/
├── Cargo.toml
├── pyproject.toml      # Maturin build config
├── src/
│   └── lib.rs          # PyO3 module: Llama class, method impls, async bridging (743 lines)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add Python method | `src/lib.rs` | `#[pymethods]` impl block for `Llama` |
| Async generation | `src/lib.rs` | `generate_async()` bridges to `tokio` runtime |
| Chat completion | `src/lib.rs` | `create_chat_completion()` handles message list |
| Error conversion | `src/lib.rs` | `PyValueError` for validation, `PyO3Error` for runtime |
| Build/release | `pyproject.toml` | `maturin` config, version sync with `Cargo.toml` |

## RUN
```bash
# Build wheel
sfw cargo run -p oxidize-py -- maturin develop

# Or directly
sfw maturin develop -m oxidize-py/Cargo.toml

# Use in Python
python -c "from oxidize_py import Llama; print(Llama)"
```

## CONVENTIONS
- **Class-centric API**: Single `Llama` class with constructor + methods. Minimal surface area.
- **Validation at boundary**: All Python-facing methods validate inputs before touching Rust state.
- **Async via thread pool**: `generate_async()` uses `resolve_to_thread()` to offload to tokio runtime.
- **Version sync**: `PYTHON_PACKAGE_VERSION` pulls from `CARGO_PKG_VERSION`.

## ANTI-PATTERNS
- `lib.rs` at 743 lines with only one `#[pyclass]` — could be split into submodules per feature.
- `generate()` uses `prompt.bytes().map(u32::from)` for tokenization — this is NOT real tokenization, just a placeholder. Should delegate to `oxidize-core` tokenizer.
- Hardcoded defaults (`vocab_size=32000`, `context_size=4096`, `layer_count=32`) — should come from model file.
- No type stubs (`.pyi`) — Python IDE users get no autocomplete.

## NOTE
This is the **PyO3 bindings crate** (`oxidize-py`). For the pure-Python implementation, see `oxidize-python/` (separate directory).
