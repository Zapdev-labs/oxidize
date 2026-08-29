# oxidize-core/src/compute/

**Generated:** 2026-05-26
**Domain:** CPU tensor ops, quantization, KV cache, flash attention

## OVERVIEW
CPU tensor ops, quantization, KV cache, and flash attention. 7 files, 10,000+ lines. Heavy SIMD, 135+ unsafe blocks.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add dtype | `tensor.rs` `DType` enum | Update `size_in_bytes()`, GEMV/GEMM dispatch |
| Add quantization type | `quantization.rs` + `tensor.rs` | Block size constants duplicated in both; keep in sync |
| KV cache eviction | `kv_cache.rs` | `KvCacheEvictionStrategy`: SlidingWindow or StopAtCapacity |
| KV cache quantization | `kv_cache.rs` | `KvQuantization::Asymmetric` (per-token) or `TurboQuant` (per-block, 32-el) |
| Flash attention | `flash_attention.rs` | `flash_attention_prefill_f32`, `flash_attention_decode_f32` |
| SIMD backend | `simd.rs` | Runtime detection: AVX-512 > AVX2 > NEON > scalar |
| Fused ops | `cpu_kernels.rs` | `dot_product_avx2_or_scalar`; quantized GEMV lives in `tensor/kernels` |
| TurboQuant weights | `turboquant.rs` | Block-wise INT4/INT8 for GEMV, 32-element blocks |

## CONVENTIONS
- **Bottom-up file organization** (`tensor.rs`): constants → errors → low-level kernels → high-level functions → `Tensor` struct (inverse of typical Rust)
- **Backend dispatch in every entrypoint**: `tensor.rs` functions check `#[cfg(feature = "cuda")]` / `metal` / `webgpu` before falling back to CPU
- **Config + Error + Trait trinity**: `KvCacheConfig`/`KvCacheError`, `QuantizationError`, etc.
- **Block size constants shadowed**: `QK_K`, `QK8_0`, `BLOCK_Q*_SIZE` defined in both `tensor.rs` and `quantization.rs` — refactor candidate
- **Test co-location**: Every `.rs` file has `#[cfg(test)]` module at bottom

## ANTI-PATTERNS
- `tensor.rs` monolith — 5,153 lines mixing kernels, types, and ops. Refactor candidate.
- Quantization constants shadowed in `tensor.rs` and `quantization.rs` — should be shared.
- `unwrap()/expect()` proliferation — 1000+ instances in non-test code (workspace-wide).
