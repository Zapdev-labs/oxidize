# oxidize-python

**Generated:** 2026-06-03
**Domain:** Pure-Python implementation of oxidize-core (mirror to oxidize-golang)

## OVERVIEW
Pure-Python reimplementation of `oxidize-core` and `oxidize-golang` functionality. Uses `uv` for package management, `pytest` for testing, and `ruff` for linting. Target feature parity with Rust, similar CLOC per module.

## STRUCTURE
```
oxidize-python/
├── pyproject.toml               # Hatchling build, scripts, pytest, ruff config
├── uv.lock                      # uv lockfile
├── README.md
└── oxidize_python/              # Main package
    ├── __init__.py
    ├── cli.py                   # CLI entry point (`oxidize` command)
    ├── core/                    # Core library (ports oxidize-core/src/)
    │   ├── __init__.py
    │   ├── tensor/              # Tensor ops, dtype, constants, GEMV, parallel
    │   │   ├── __init__.py
    │   │   ├── constants.py
    │   │   ├── dtype.py
    │   │   ├── ops.py
    │   │   ├── gemv.py
    │   │   ├── kernels_extended.py
    │   │   ├── parallel.py
    │   │   └── errors.py
    │   ├── simd/                # SIMD wrappers (Numba/ctypes)
    │   ├── quantization/        # Quantize/dequantize, imatrix, k-quant
    │   ├── model/               # Inference, DFlash, sampling, generation, LoRA, offload
    │   │   ├── inference.py
    │   │   ├── inference_config.py
    │   │   ├── generation.py
    │   │   ├── sampling.py
    │   │   ├── speculative.py
    │   │   ├── dflash.py
    │   │   ├── dflash_forward.py
    │   │   ├── dflash_heuristic.py
    │   │   ├── dflash_weights.py
    │   │   ├── llama.py
    │   │   ├── llama_decoder.py
    │   │   ├── llama_decoder_forward.py
    │   │   ├── layer_wise.py
    │   │   ├── prefix_cache.py
    │   │   ├── offload.py
    │   │   ├── lora.py
    │   │   ├── loader.py
    │   │   ├── model.py
    │   │   └── advanced_features.py
    │   ├── flash_attention/     # Flash attention, ALiBi, sliding window
    │   ├── kv_cache.py          # KV cache management
    │   ├── tokenizer/           # SP, WordPiece, BPE, Tiktoken
    │   ├── ggufcore/            # GGUF v3 parser
    │   ├── safetensors/         # SafeTensors format
    │   ├── mesh/                # Distributed inference mesh
    │   ├── vision/              # Vision encoder / multimodal
    │   ├── paged/               # Paged attention
    │   ├── backends/            # WebGPU, Vulkan, MLX, Strix stubs
    │   ├── cpu_kernels/         # CPU kernel implementations
    │   ├── workspace.py         # Training workspace utilities
    │   ├── validation.py        # Validation utilities
    │   ├── util.py              # Shared utilities
    │   └── backend.py           # Backend abstraction
    ├── internal/                 # Private package
    │   ├── __init__.py
    │   ├── api/                 # OpenAI-compatible schema
    │   ├── auth.py              # API key authentication
    │   ├── server.py            # HTTP server
    │   ├── gguf/                # GGUF writer, reader, metadata, tensor_size
    │   └── generate/            # Completion runtime
    ├── quantize/                # Offline quantization CLI (`oxidize-quantize`)
    │   ├── __init__.py
    │   └── cli.py
    └── train/                   # Training CLI (`oxidize-train`)
        ├── __init__.py
        ├── cli.py
        └── lib.py
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Port GGUF parser | `core/ggufcore/` | Mirror Go `internal/gguf/` and Rust `gguf.rs` |
| Port tensor ops | `core/tensor/` | Mirror `tensor.rs`, `ops.py` ≈ `ops.rs` |
| Port quantization | `core/quantization/` | Mirror Go `core/quantization/` |
| Port inference | `core/model/` | Mirror `inference.rs`, `generation.rs` |
| Port DFlash | `core/model/dflash*.py` | Mirror Go `dflash*.go` |
| Port tokenizer | `core/tokenizer/` | Mirror `tokenizer.rs` (4 formats) |
| Port server | `internal/server.py` | Mirror `oxidize-server` routes |
| Add backend | `core/backends/` | Implement Python `ComputeBackend` protocol |

## RUN
```bash
# Install dependencies
sfw uv sync

# Run tests
sfw uv run pytest

# Run linting
sfw uv run ruff check .

# CLI
sfw uv run python -m oxidize_python.cli --help
sfw uv run oxidize --help
sfw uv run oxidize-quantize --help
sfw uv run oxidize-train --help
```

## CONVENTIONS
- **Mirror Go structure first**: Python packages map to Go packages, which map to Rust modules.
- **Type hints everywhere**: All functions have type annotations; use `from __future__ import annotations`.
- **Protocols for traits**: Rust traits become `typing.Protocol` classes (e.g., `ComputeBackend` protocol).
- **Dataclasses for config**: Rust `XxxConfig` structs become `@dataclass` or `pydantic.BaseModel`.
- **Test fixtures**: Reuse `oxidize-core/tests/fixtures/valid-v3.gguf` for parity tests.
- **Error wrapping**: Use `raise XxxError(...) from err` to preserve cause chains.

## ANTI-PATTERNS
- Do NOT modify Rust crates when porting to Python — copy from `oxidize-golang` or Rust sources.
- Some modules are stubs — check for `TODO(port)` or `NotImplementedError` before relying on them.
- `core/tensor/ops.py` may become large — consider splitting into `ops_blas.py`, `ops_elementwise.py` early.
- Avoid runtime type checking (`isinstance` guards) in hot paths — use protocols/ABC for structure, rely on duck typing for speed.
- Missing `__all__` exports in many `__init__.py` — add for public API clarity.

## SYNC NOTES
- When `oxidize-golang` master gets new features, bring them into `oxidize-python` next.
- DFlash speculative decoding is a current priority port target.
- Feature branch rule: stage and commit only files related to the task; exclude unrelated changes.
