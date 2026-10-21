# oxidize-ffi

**Domain:** C-ABI FFI layer over `oxidize-core`

## OVERVIEW
Thin C-ABI wrapper exposing `oxidize-core` model load/forward, a fused quantized GEMV, and session management to non-Rust callers (e.g. the C/C++ ports). Builds as both `cdylib` and `staticlib`.

## STRUCTURE
```
oxidize-ffi/
├── Cargo.toml   # lib name "oxidize_ffi"; crate-type = ["cdylib", "staticlib"]; deps: oxidize-core, rayon
└── src/
    └── lib.rs   # entire FFI surface + Rayon pool init + quant-type mapping
```

## PUBLIC API (C ABI, `#[no_mangle]`)
| Function | Role |
|----------|------|
| `oxidize_ffi_version()` | Version string |
| `oxidize_gemv_quantized(quant_type, qbytes, len, rows, cols, vector, output)` | AVX2+FMA fused quantized GEMV |
| `oxidize_model_load(path)` / `oxidize_model_free(handle)` | Model lifecycle |
| `oxidize_model_vocab_size(handle)` | Vocab size |
| `oxidize_session_new()` / `oxidize_session_reset(session)` / `oxidize_session_free(session)` | Session lifecycle |
| `oxidize_model_forward(handle, session, tokens, n_tokens, logits_out, vocab_size)` | Forward pass |
| `oxidize_sample_argmax(logits, vocab_size)` | Greedy sample |

Returns `0`/`-1` int status codes; handles are opaque `*mut c_void`.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add exported function | `src/lib.rs` | Keep signatures C-compatible; no panics across the boundary |
| Thread pool tuning | `src/lib.rs` | `OXIDIZE_THREADS` env override; heuristic clamps to `[4, 12]`, halves on SMT |

## BUILD / RUN
```bash
cargo build -p oxidize-ffi     # -> liboxidize_ffi.so / liboxidize_ffi.a
```

## NOTES
- Runtime env var `OXIDIZE_THREADS` overrides the Rayon pool size.
- Linux-only `advise_huge_pages()` (THP) is applied on model load.
- Consumers (C/C++) link the `cdylib`/`staticlib` and call the exported symbols.
