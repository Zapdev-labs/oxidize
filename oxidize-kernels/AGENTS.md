# oxidize-kernels (OXK)

**Domain:** Hand-tuned, zero-dependency CPU kernels for quantized GEMV

## OVERVIEW
OXK provides bit-exact Q4_K × Q8_K row-dot / GEMV kernels (scalar reference + AVX2 ×1/×4/×8/×16 + AVX-512) plus pruning-mask helpers. The per-row math is **bit-identical** to the legacy kernels in `oxidize-core/src/compute/tensor.rs` (same integer op sequence, same per-block f32 accumulation order) so parity tests assert exact equality. OXK's wins are structural: more independent DRAM streams in flight (multi-row variants) and a wider software-prefetch window tuned for Xeon. The crate is **self-contained (no deps, not even `oxidize-core`)** so it can be benchmarked/tested in isolation.

## STRUCTURE
```
oxidize-kernels/
├── Cargo.toml       # no deps; benches: oxk_q4k_bench, oxk_token_bench
├── benches/
└── src/
    ├── lib.rs         # public API, constants, feature-detect helpers
    ├── cpu.rs         # CpuInfo/CpuVendor detection, OxkTune, is_skylake_sp()
    ├── prune.rs       # magnitude_mask, wanda_mask, apply_mask_inplace
    ├── q4k_scalar.rs  # scalar reference row dot
    ├── q4k_avx2.rs    # AVX2 ×1/×4/×8/×16 row dots
    ├── q4k_avx512.rs  # AVX-512F+BW row dots
    ├── q4k_dequant.rs # dequantize_q4_k_into
    └── q8k.rs         # quantize_q8_k_into
```

## PUBLIC API (selected)
- `q4k_q8k_row_dot_scalar`, `q4k_q8k_row_dot_avx2` (+ `_x4/_x8/_x16`), AVX-512 variants.
- `dequantize_q4_k_into`, `quantize_q8_k_into`.
- `magnitude_mask`, `wanda_mask`, `apply_mask_inplace`.
- `cpuinfo`, `cpu_vendor`, `oxk_cpu_summary`, `OxkTune`, `is_skylake_sp()`.
- `oxk_avx2_available()`, `oxk_avx512_available()`.
- Constants: `QK_K = 256`, `BLOCK_Q4_K_SIZE = 144`, `BLOCK_Q8_K_BYTES = 292`.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add ISA variant | `q4k_avx2.rs` / `q4k_avx512.rs` | Keep bit-exact vs `q4k_scalar.rs` |
| CPU detection | `cpu.rs` | Skylake-SP gate for AVX-512 regression |
| Pruning masks | `prune.rs` | mirrors `oxidize-prune` |

## BUILD / TEST / BENCH
```bash
cargo test  -p oxidize-kernels                 # includes bit-exact parity tests
cargo bench -p oxidize-kernels                 # oxk_q4k_bench, oxk_token_bench
```

## NOTES
- Consumed by `oxidize-core` behind the optional `oxk` cargo feature; runtime kernel selection via `OXIDIZE_GEMV` (and related `OXIDIZE_OXK_*` env vars).
- Any new kernel MUST match the scalar reference bit-for-bit — parity is a hard invariant.
