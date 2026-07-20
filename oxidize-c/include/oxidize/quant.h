/*
 * quant.h — GGUF quantization types + dequant/pack API.
 *
 * Port of oxidize-core/src/compute/quantization.rs (block-size constants,
 * `quant_block_layout`, `dequantize_scalar`, `quantize_from_f32_scalar`) to
 * C11. Block layouts and bit-stream layouts are bit-exact ports of the Rust
 * reference; numerical parity is a hard invariant (VAL-QUANT-001..015).
 *
 * Scope of the `quant-standard-types` feature: the "standard" GGUF quant
 * types (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K_S/M/L/Q4_K_S/M/
 * Q5_K_S/M/Q6_K/I8/I16/I32/I64/F64). Custom AL-family, IQ-family, and NVFP4
 * variants are added by the `quant-al-iq-nvfp4` feature; their enum slots
 * exist here so the dispatch table is stable.
 *
 * SIMD dispatch (scalar vs AVX2 vs AVX-512) is layered on top by the
 * `quant-simd-dispatch` feature; this file provides only the scalar reference
 * path. SIMD kernels must produce output byte-identical to the scalar path.
 */
#ifndef OXIDIZE_QUANT_H
#define OXIDIZE_QUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Block-size constants (port of quantization.rs) ────────────────────
 *
 * Values MUST match the Rust `QK*` / `BLOCK_*_SIZE` constants bit-exactly
 * (VAL-QUANT-001). Do not change without updating both sides.
 */
#define OC_QK4_0      32u
#define OC_QK4_1      32u
#define OC_QK5_0      32u
#define OC_QK5_1      32u
#define OC_QK8_0      32u
#define OC_QK4_NL     32u
#define OC_QK_K       256u
#define OC_QK_NVFP4   64u
#define OC_QK_NVFP4_SUB 16u
/* AL-family block element count (shared by AL5_XS). */
#define OC_QK_AL      32u

/* Block byte sizes (port of BLOCK_*_SIZE constants). */
#define OC_BLOCK_Q4_0_SIZE   18u   /* 2 + 16                                  */
#define OC_BLOCK_Q4_1_SIZE   20u   /* 2 + 2 + 16                              */
#define OC_BLOCK_Q5_0_SIZE   22u   /* 2 + 4 + 16  (port of Rust BLOCK_Q5_0_SIZE) */
#define OC_BLOCK_Q5_1_SIZE   24u   /* 2 + 2 + 4 + 16 (port of Rust BLOCK_Q5_1_SIZE) */
#define OC_BLOCK_Q8_0_SIZE   34u   /* 2 + 32                                  */
#define OC_BLOCK_Q2_K_SIZE   84u   /* 2*f16 + QK_K/16 + QK_K/4  = 4+16+64     */
#define OC_BLOCK_Q3_K_SIZE   110u  /* f16 + QK_K/4 + QK_K/8 + 12 = 2+64+32+12 */
#define OC_BLOCK_Q4_K_SIZE   144u  /* 2*f16 + 12 + QK_K/2       = 4+12+128     */
#define OC_BLOCK_Q5_K_SIZE   176u  /* 2*f16 + 12 + QK_K/2 + QK_K/8 = 4+12+128+32 */
#define OC_BLOCK_Q6_K_SIZE   210u  /* f16 + QK_K/16 + 3*QK_K/4  = 2+16+192    */
#define OC_BLOCK_Q8_K_SIZE   292u  /* f32 + QK_K + QK_K/16*i16  = 4+256+32    */
/* AL-family + IQ-family + NVFP4 block sizes (added by al-iq-nvfp4 feature).
 * Ported bit-exact from oxidize-core/src/compute/quantization.rs:
 *   BLOCK_AL5_XS_SIZE = 2 + QK_AL*3/8 = 2 + 12 = 14
 *   BLOCK_IQ1_S_SIZE  = sizeof_of_f16() + QK_K/8  + QK_K/16  = 2 + 32 + 16 = 50
 *   BLOCK_IQ1_M_SIZE  = QK_K/8 + QK_K/16 + QK_K/32           = 32 + 16 + 8 = 56
 *   BLOCK_IQ2_XXS_SIZE= sizeof_of_f16() + QK_K/4             = 2 + 64      = 66
 *   BLOCK_IQ2_XS_SIZE = sizeof_of_f16() + QK_K/8*2 + QK_K/32 = 2 + 64 + 8  = 74
 *   BLOCK_IQ2_S_SIZE  = sizeof_of_f16() + QK_K/4 + QK_K/32 + QK_K/32 = 2+64+8+8 = 82
 *   BLOCK_IQ3_XXS_SIZE= sizeof_of_f16() + 3*(QK_K/8)         = 2 + 96      = 98
 *   BLOCK_IQ3_S_SIZE  = sizeof_of_f16() + QK_K/4 + QK_K/32 + QK_K/8 + QK_K/64
 *                                                                  = 2+64+8+32+4 = 110
 *   BLOCK_IQ4_NL_SIZE = sizeof_of_f16() + QK4_NL/2            = 2 + 16      = 18
 *   BLOCK_IQ4_XS_SIZE = sizeof_of_f16() + 2 + QK_K/64 + QK_K/2 = 2+2+4+128 = 136
 *   BLOCK_NVFP4_SIZE  = QK_NVFP4/QK_NVFP4_SUB + QK_NVFP4/2    = 4 + 32      = 36
 */
#define OC_BLOCK_AL5_XS_SIZE   14u
#define OC_BLOCK_IQ1_S_SIZE   50u
#define OC_BLOCK_IQ1_M_SIZE   56u
#define OC_BLOCK_IQ2_XXS_SIZE 66u
#define OC_BLOCK_IQ2_XS_SIZE  74u
#define OC_BLOCK_IQ2_S_SIZE   82u
#define OC_BLOCK_IQ3_XXS_SIZE 98u
#define OC_BLOCK_IQ3_S_SIZE  110u
#define OC_BLOCK_IQ4_NL_SIZE  18u
#define OC_BLOCK_IQ4_XS_SIZE 136u
#define OC_BLOCK_NVFP4_SIZE   36u

/* ─── Quantization type enum (port of GgufQuantizationType) ──────────────
 *
 * Order matches architecture.md §3.2. Values are stable and MUST NOT change
 * (GGUF on-disk dtype ids are mapped through `oc_quant_type_from_ggml_id`). */
typedef enum {
    OC_QUANT_F32      = 0,
    OC_QUANT_F16      = 1,
    OC_QUANT_BF16     = 2,
    OC_QUANT_Q4_0     = 3,
    OC_QUANT_Q4_1     = 4,
    OC_QUANT_Q5_0     = 5,
    OC_QUANT_Q5_1     = 6,
    OC_QUANT_Q8_0     = 7,
    OC_QUANT_Q2_K     = 8,
    OC_QUANT_Q3_K_S   = 9,
    OC_QUANT_Q3_K_M   = 10,
    OC_QUANT_Q3_K_L   = 11,
    OC_QUANT_Q4_K_S   = 12,
    OC_QUANT_Q4_K_M   = 13,
    OC_QUANT_Q5_K_S   = 14,
    OC_QUANT_Q5_K_M   = 15,
    OC_QUANT_Q6_K     = 16,
    /* AL-family (custom oxidize types, ggml ids 240-243). */
    OC_QUANT_AL5      = 17,
    OC_QUANT_AL5_XS   = 18,
    OC_QUANT_AL6      = 19,
    OC_QUANT_AL8      = 20,
    /* IQ-family. */
    OC_QUANT_IQ2_XXS  = 21,
    OC_QUANT_IQ2_XS   = 22,
    OC_QUANT_IQ2_S    = 23,
    OC_QUANT_IQ3_XXS  = 24,
    OC_QUANT_IQ3_S    = 25,
    OC_QUANT_IQ4_NL   = 26,
    OC_QUANT_IQ4_XS   = 27,
    OC_QUANT_IQ1_S    = 28,
    OC_QUANT_IQ1_M    = 29,
    OC_QUANT_NVFP4    = 30,
    /* Integer / wide-float plain storage (no quantization). */
    OC_QUANT_I8       = 31,
    OC_QUANT_I16      = 32,
    OC_QUANT_I32      = 33,
    OC_QUANT_I64      = 34,
    OC_QUANT_F64      = 35,
    OC_QUANT__COUNT,
    OC_QUANT_UNKNOWN  = 0xffffffffu,
} OcGgufQuantizationType;

/* ─── Public API ──────────────────────────────────────────────────────── */

/* `(elements_per_block, bytes_per_block)` for the given quant type, matching
 * Rust `quant_block_layout` bit-exactly. For unknown types returns (0, 0)
 * (VAL-QUANT-013). */
typedef struct OcQuantBlockLayout {
    size_t elements_per_block;
    size_t bytes_per_block;
} OcQuantBlockLayout;

OcQuantBlockLayout oc_quant_block_size(OcGgufQuantizationType qtype);

/* Byte length needed to encode `value_count` source values into the given
 * quant type. Returns 0 if `value_count` is not a multiple of
 * elements_per_block (port of Rust `quantized_size`). */
size_t oc_quantized_size(OcGgufQuantizationType qtype, size_t value_count);

/* Dequantize `src_len` bytes of `src` (a packed quant buffer of type `qtype`)
 * into `dst` (an f32 array of length `value_count`). `value_count` must equal
 * `src_len / bytes_per_block * elements_per_block`; otherwise returns
 * OC_ERR_INVALID_ARG. Unknown types return OC_ERR_QUANT (no crash).
 *
 * Mirrors Rust `dequantize_scalar(qtype, input, output)`. Output is
 * bit-exact with the Rust reference on integer paths; FP paths match
 * within f32 rounding tolerance (≤1e-7 relative). */
OcError oc_quant_dequant_row(OcGgufQuantizationType qtype,
                             const uint8_t *src, size_t src_len,
                             float *dst, size_t value_count);

/* Quantize `src` (an f32 array of length `value_count`) into `dst` (a packed
 * quant buffer). `dst_len` must equal `oc_quantized_size(qtype, value_count)`.
 * Returns OC_OK, OC_ERR_QUANT (unknown type), OC_ERR_INVALID_ARG (length
 * mismatch), or OC_ERR_OOM. Mirrors Rust `quantize_from_f32_scalar`. */
OcError oc_quant_pack_row(OcGgufQuantizationType qtype,
                          const float *src, size_t value_count,
                          uint8_t *dst, size_t dst_len);

/* Convenience: pack one block. `src` points to `elements_per_block` f32 values;
 * `dst` points to `bytes_per_block` bytes. Returns OC_OK or an error code. */
OcError oc_quant_pack_block(OcGgufQuantizationType qtype,
                            const float *src, uint8_t *dst);

/* Human-readable name ("F32", "Q4_K_M", ...). Returns "?" for unknown. */
const char *oc_quant_type_name(OcGgufQuantizationType qtype);

/* Map an on-disk ggml dtype id (from the GGUF tensor table) to the
 * OcGgufQuantizationType enum. Returns OC_QUANT_UNKNOWN for unrecognized ids
 * (VAL-QUANT-013). */
OcGgufQuantizationType oc_quant_type_from_ggml_id(uint32_t ggml_type);

/* Inverse of oc_quant_type_from_ggml_id: the canonical on-disk ggml dtype
 * id for the given quant type. Returns 0xffffffff for OC_QUANT_UNKNOWN. */
uint32_t oc_quant_type_to_ggml_id(OcGgufQuantizationType qtype);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_QUANT_H */
