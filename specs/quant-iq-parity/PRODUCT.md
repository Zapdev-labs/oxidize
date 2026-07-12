# IQ Quantization Parity — Product Spec

**Commit:** `7b163b2fdc299556446dc85991495752a7326edb`

## Problem

Several widely-used GGUF IQ quantization formats cannot be loaded in Oxidize today. Models quantized as **IQ2_XS**, **IQ2_S**, or **IQ4_NL** fail at weight dequantization even though the enum variants exist. This blocks inference on aggressive-compression checkpoints (e.g. Gemma IQ4_XS/IQ4_NL families, IQ2-tier research quants).

Additionally, **IQ4_NL** block sizing is wrong in oxidize-core (`QK_K`/`BLOCK_Q4_K_SIZE` instead of `QK4_NL=32` / 18-byte blocks), which would corrupt any future implementation.

## User-visible behavior

1. **Load**: Any GGUF weight tensor typed `IQ2_XS` (ggml 17), `IQ2_S` (22), or `IQ4_NL` (20) dequantizes to finite f32 values identical to llama.cpp `dequantize_row_*`.
2. **Infer**: Dequantized weights participate in matmul via the existing dequant→dot fallback path (no new fused kernel required for v1).
3. **Quantize (IQ4_NL only)**: `oxidize-quantize --target IQ4_NL` produces byte-compatible blocks for F32/F16 source tensors (32-element blocks, nonlinear codebook).
4. **Cross-port parity**: oxidize-core, oxidize-cpp, and oxidize-c produce bitwise-identical dequant output for shared golden blocks.

## Behavior invariants

1. `quantized_size(IQ4_NL, n)` uses `QK4_NL=32` and `BLOCK_IQ4_NL=18` bytes.
2. `quantized_size(IQ2_S, n)` uses `QK_K=256` and `BLOCK_IQ2_S=82` bytes (not `BLOCK_Q2_K`).
3. `dequantize_scalar` never returns `UnsupportedQuantizationType` for IQ2_XS, IQ2_S, IQ4_NL.
4. IQ2_XS/IQ2_S grid tables (`iq2xs_grid` 512 entries, `iq2s_grid` 1024 entries) are verbatim from ggml-common.h.
5. IQ4_NL decode: `x[j] = d × kvalues_iq4nl[nibble]` (same codebook as IQ4_XS).
6. IQ4_NL encode round-trip: encode→decode→encode reproduces identical bytes on codebook-grid values.

## Success criteria

- All three ports pass new unit tests for the three formats.
- IQ4_NL MSE on Gaussian weights is ≤ Q4_0 MSE at equal nominal bpw (4.5).
- No regression in existing quant tests (`cargo test -p oxidize-core`, oxidize-cpp `parity_test`, oxidize-c `test_oc`).

## Out of scope (v1)

- IQ1_S full 2048-entry grid fix (separate correctness issue).
- Fused GEMV kernels for IQ2_XS/IQ2_S.
- IQ2_XS/IQ2_S encoders (load-only).
- MXFP4.
