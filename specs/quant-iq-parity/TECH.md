# IQ Quantization Parity — Technical Spec

**Commit:** `7b163b2fdc299556446dc85991495752a7326edb`

## Context

Oxidize mirrors llama.cpp GGML block layouts. IQ formats use lookup grids + sign masks from `ggml-common.h`. Existing ports implement IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS dequant; **IQ2_XS, IQ2_S, IQ4_NL** are enum-only.

Reference dequant (llama.cpp `ggml-quants.c`):

- **IQ2_XS** (74 B): `d:f16` + `qs:u16[32]` + `scales:u8[8]` — grid index in low 9 bits of each u16, signs in bits 9+.
- **IQ2_S** (82 B): `d:f16` + `qs:u8[64]` + `qh:u8[8]` + `scales:u8[8]` — signs at `qs+32`.
- **IQ4_NL** (18 B, `QK4_NL=32`): `d:f16` + `qs:u8[16]` — nonlinear codebook `kvalues_iq4nl`.

Key files today:

- `oxidize-core/src/compute/quantization/quant_iq_series.rs` — IQ2_XXS dequant pattern
- `oxidize-core/src/compute/quantization/iq_grids.rs` — grid tables (missing iq2xs/iq2s)
- `oxidize-core/src/compute/quantization/quant_dispatch.rs` — dispatch (missing 3 arms)
- `oxidize-cpp/src/quant.cpp` + `iq_grids.inc` — same gap
- `oxidize-c/quant.c` — no IQ types beyond IQ4_XS

**Bug:** `quantization.rs` maps `IQ4_NL → (QK_K, BLOCK_Q4_K_SIZE)` — must be `(32, 18)`.

## Proposed changes

### 1. Shared constants

| Symbol | Value |
|--------|-------|
| `QK4_NL` | 32 |
| `BLOCK_IQ4_NL_SIZE` | 18 |
| `BLOCK_IQ2_S_SIZE` | 82 |

Fix `quant_block_layout` for `IQ2_S` and `IQ4_NL`.

### 2. Grid tables

Extract `iq2xs_grid[512]` and `iq2s_grid[1024]` from upstream `ggml-common.h` into:

- `oxidize-core/.../iq_grids.rs`
- `oxidize-cpp/.../iq_grids.inc`

### 3. Dequant implementations

Port verbatim from llama.cpp into `quant_iq_series.rs` (Rust), `quant.cpp` (C++), `quant.c` (C).

### 4. IQ4_NL encoder

Add `quantize_iq4_nl_scalar` in `quant_k_blocks.rs`: per 32-el block, `d = max|x|/max(codebook)`, nearest codebook index per value (reuse `best_index_iq4nl`).

Wire into `quantize_from_f32_scalar` and `oxidize-quantize` CLI `--target IQ4_NL`.

### 5. oxidize-c scope

Add `OC_IQ2_XXS`, `OC_IQ2_XS`, `OC_IQ2_S`, `OC_IQ3_XXS`, `OC_IQ3_S`, `OC_IQ4_NL` to `oc_quant` (dequant + ggml mapping). GEMV uses existing dequant fallback.

## Testing and validation

| Invariant | Test |
|-----------|------|
| Block sizes | `iq_block_sizes_match_ggml_layout` extended |
| IQ2_XS dequant | Golden block bytes → expected f32 (hand-computed from llama.cpp walk) |
| IQ4_NL round-trip | Gaussian sample: encode→decode MSE < Q4_0 MSE |
| IQ4_NL vs Q4_0 | Comparative MSE test on same weights |
| Cross-port | Shared hex fixture in Rust + C++ parity_test + test_oc |
| Regression | `cargo test -p oxidize-core`, `make -C oxidize-cpp test`, `make -C oxidize-c test` |

## Parallelization

| Agent | Owns | Mode |
|-------|------|------|
| Parent | Spec, Rust core + grids + tests | local |
| cpp-agent | oxidize-cpp quant.cpp, iq_grids.inc, parity_test | local |
| c-agent | oxidize-c quant.c, oc.h, test_oc.c | local |

Rust lands first (reference); C++/C agents port from Rust + llama.cpp.
