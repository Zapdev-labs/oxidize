/* oxk.h — OXK CPU GEMV kernels. Bit-exact with oxidize-kernels (VAL-OXK-001). */
#ifndef OXIDIZE_OXK_H
#define OXIDIZE_OXK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OC_OXK_QK4_0      32u
#define OC_OXK_QK4_1      32u
#define OC_OXK_QK8_0      32u
#define OC_OXK_QK_K       256u

#define OC_OXK_BLOCK_Q4_0_SIZE   18u   /* f16 d + 16 packed 4-bit values      */
#define OC_OXK_BLOCK_Q4_1_SIZE   20u   /* f16 d + f16 m + 16 packed 4-bit     */
#define OC_OXK_BLOCK_Q8_0_SIZE   34u   /* f16 d + 32 int8 values              */
#define OC_OXK_BLOCK_Q2_K_SIZE   84u   /* 16 scale/min bytes + 64 packed 2-bit + f16 d + f16 dmin */
#define OC_OXK_BLOCK_Q3_K_SIZE  110u  /* 32 hmask + 64 packed 2-bit + 12 packed 6-bit scales + f16 d */
#define OC_OXK_BLOCK_Q4_K_SIZE  144u  /* f16 d + f16 dmin + 12 scales + 128 nibbles */
#define OC_OXK_BLOCK_Q5_K_SIZE  176u  /* f16 d + f16 dmin + 12 scales + 128 nibbles + 32qh */
#define OC_OXK_BLOCK_Q6_K_SIZE  210u  /* f16 d + 16 ql_scales + 192 packed (4+4-bit) */
#define OC_OXK_BLOCK_Q8_K_SIZE  292u  /* f32 d + 256 int8 + 16 i16 bsums     */

typedef enum {
    OC_OXK_SCALAR  = 0,   /* no usable SIMD acceleration                */
    OC_OXK_AVX2    = 1,   /* AVX2 + FMA + F16C (Haswell+, Zen2+)        */
    OC_OXK_AVX512  = 2,   /* AVX-512 BW + DQ + VNNI (Cascade Lake+, Zen4) */
    OC_OXK_NEON    = 3,   /* AArch64 Advanced SIMD — see oxk_neon.h. Not
                           * ordered against the x86 tiers; compare for
                           * equality, never `>=`. */
} OcOxkLevel;

typedef struct OcOxkCaps {
    OcOxkLevel level;
    bool has_f16c;    /* f16 → f32 hardware conversion (needed by AVX2+)   */
    bool has_fma;    /* FMA3 (available on every AVX2 host in practice)   */
    bool has_vnni;   /* AVX-512 VNNI (VPDPBUSD)                           */
    bool has_neon;   /* AArch64 Advanced SIMD                             */
    const char *name; /* "scalar" | "avx2" | "avx512" | "neon"            */
} OcOxkCaps;

/* Function-pointer types for each kernel family. */
typedef float (*OcOxkDotQ4_0Q8_0_fn)(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8);
typedef float (*OcOxkDotQ4_1Q8_0_fn)(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8);
typedef float (*OcOxkDotQ4_KQ8_K_fn)(const uint8_t *row, size_t blocks,
                                    const uint8_t *q8);
typedef float (*OcOxkDotQ5_KQ8_K_fn)(const uint8_t *row, size_t blocks,
                                    const uint8_t *q8);
typedef float (*OcOxkDotQ6_KQ8_K_fn)(const uint8_t *row, size_t blocks,
                                    const uint8_t *q8);
typedef float (*OcOxkDotQ8_0Q8_0_fn)(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8);
typedef float (*OcOxkDotQ2_KQ8_K_fn)(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8);
typedef float (*OcOxkDotQ3_KQ8_K_fn)(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8);

typedef void (*OcOxkMatvecQ4_0F32_fn)(const uint8_t *w, size_t n_rows,
                                     size_t row_bytes, const float *x,
                                     float *out);
typedef void (*OcOxkMatvecQ4_KF32_fn)(const uint8_t *w, size_t n_rows,
                                     size_t row_bytes, const float *x,
                                     float *out);
typedef void (*OcOxkMatvecQ8_0F32_fn)(const uint8_t *w, size_t n_rows,
                                     size_t row_bytes, const float *x,
                                     float *out);

/* One prepared Q4_K row dotted against `n_act` Q8_K activations spaced
 * `act_stride` bytes apart; `out` receives `n_act` f32 results. */
/* One prepared row dotted against a single Q8_K activation. */
typedef float (*OcOxkDotPrepped1_fn)(const void *prep, size_t blocks,
                                     const uint8_t *act);

typedef void (*OcOxkDotQ4_KPreppedMulti_fn)(const void *prep, size_t blocks,
                                            const uint8_t *acts,
                                            size_t act_stride, size_t n_act,
                                            float *out);

typedef struct OcOxkContext {
    OcOxkCaps caps;
    /* Dot-product dispatch table — one slot per quant pair. */
    OcOxkDotQ4_0Q8_0_fn dot_q4_0_q8_0;
    OcOxkDotQ4_1Q8_0_fn dot_q4_1_q8_0;
    OcOxkDotQ4_KQ8_K_fn dot_q4_k_q8_k;
    OcOxkDotQ5_KQ8_K_fn dot_q5_k_q8_k;
    OcOxkDotQ6_KQ8_K_fn dot_q6_k_q8_k;
    OcOxkDotQ8_0Q8_0_fn dot_q8_0_q8_0;
    OcOxkDotQ2_KQ8_K_fn dot_q2_k_q8_k;
    OcOxkDotQ3_KQ8_K_fn dot_q3_k_q8_k;
    OcOxkDotQ4_KPreppedMulti_fn dot_q4_k_prepped_multi;
    /* Same signature; prep layout is the Q6_K one (oc_oxk_q6_k_prep_row). */
    OcOxkDotQ4_KPreppedMulti_fn dot_q6_k_prepped_multi;
    /* Q3_K shares the Q6_K prep layout — both decode to unsigned codes with because it carries a per-group minimum as well. */
    OcOxkDotQ4_KPreppedMulti_fn dot_q3_k_prepped_multi;
    OcOxkDotQ4_KPreppedMulti_fn dot_q2_k_prepped_multi;
    /* Single-activation form of the two above, for decode. */
    OcOxkDotPrepped1_fn dot_q2_k_prepped_1;
    OcOxkDotPrepped1_fn dot_q3_k_prepped_1;
    /* Matvec dispatch table — one slot per quant type. */
    OcOxkMatvecQ4_0F32_fn matvec_q4_0_f32;
    OcOxkMatvecQ4_KF32_fn matvec_q4_k_f32;
    OcOxkMatvecQ8_0F32_fn matvec_q8_0_f32;
} OcOxkContext;

/* Detect CPU features once and populate the global context's dispatch table
 * with the best kernel for the host. Thread-safe via C11 one-time init.
 * Returns a pointer to the (now-initialized) global context. */
const OcOxkContext *oc_oxk_init(void);

/* Returns the cached capability struct (triggers init on first call). */
const OcOxkCaps *oc_oxk_caps(void);

float oc_oxk_dot_q4_0_q8_0(const uint8_t *row, size_t blocks_per_row,
                            const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0(const uint8_t *row, size_t blocks_per_row,
                            const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k(const uint8_t *row, size_t blocks_per_row,
                           const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k(const uint8_t *row, size_t blocks_per_row,
                           const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k(const uint8_t *row, size_t blocks_per_row,
                           const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0(const uint8_t *row, size_t blocks_per_row,
                            const uint8_t *q8);

/* oc_oxk_dot_q4_k_prepped() is bit-exact with oc_oxk_dot_q4_k_q8_k(). */

/* Scratch bytes oc_oxk_q4_k_prep_row() needs for a row of `blocks`. */
size_t oc_oxk_q4_k_prep_bytes(size_t blocks);

/* Decode one packed Q4_K row into `scratch` (>= oc_oxk_q4_k_prep_bytes()).
 * `scratch` must be suitably aligned for float — malloc'd memory is. */
void oc_oxk_q4_k_prep_row(const uint8_t *row, size_t blocks, void *scratch);

/* Dot a prepared row against one packed Q8_K activation. */
float oc_oxk_dot_q4_k_prepped(const void *scratch, size_t blocks,
                              const uint8_t *q8);

/* Dispatched multi-activation form: dot one prepared row against `n_act` activations spaced `act_stride` bytes apart. */
void oc_oxk_dot_q4_k_prepped_multi(const void *scratch, size_t blocks,
                                   const uint8_t *acts, size_t act_stride,
                                   size_t n_act, float *out);

/* Q5_K prep into the SAME layout as Q4_K (codes 0..31 fit unsigned bytes; Q4_K prepped dot/multi kernels unchanged — bit-exact against */
void oc_oxk_q5_k_prep_row(const uint8_t *row, size_t blocks, void *scratch);

/* Prepared Q6_K rows: codes stay unsigned 0..63 with 16 signed per-group
 * scales; the -32 offset is folded out through the activation block sums.
 * Bit-exact against oc_oxk_dot_q6_k_q8_k(). */
size_t oc_oxk_q6_k_prep_bytes(size_t blocks);
void oc_oxk_q6_k_prep_row(const uint8_t *row, size_t blocks, void *scratch);

float oc_oxk_dot_q2_k_q8_k(const uint8_t *row, size_t blocks,
                           const uint8_t *q8);
float oc_oxk_dot_q3_k_q8_k(const uint8_t *row, size_t blocks,
                           const uint8_t *q8);
float oc_oxk_dot_q2_k_q8_k_scalar(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8);
float oc_oxk_dot_q3_k_q8_k_scalar(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8);

size_t oc_oxk_q2_k_prep_bytes(size_t blocks);
void oc_oxk_q2_k_prep_row(const uint8_t *row, size_t blocks, void *scratch);
void oc_oxk_q3_k_prep_row(const uint8_t *row, size_t blocks, void *scratch);
float oc_oxk_dot_q2_k_prepped(const void *scratch, size_t blocks,
                              const uint8_t *q8);
float oc_oxk_dot_q3_k_prepped(const void *scratch, size_t blocks,
                              const uint8_t *q8);
void oc_oxk_dot_q2_k_prepped_multi(const void *scratch, size_t blocks,
                                   const uint8_t *acts, size_t act_stride,
                                   size_t n_act, float *out);
void oc_oxk_dot_q3_k_prepped_multi(const void *scratch, size_t blocks,
                                   const uint8_t *acts, size_t act_stride,
                                   size_t n_act, float *out);
/* Single-activation dots over the prepared rows: what decode uses. Same
 * results as the _prepped functions, dispatched to SIMD where available. */
float oc_oxk_dot_q2_k_prepped_1(const void *prep, size_t blocks,
                                const uint8_t *act);
float oc_oxk_dot_q3_k_prepped_1(const void *prep, size_t blocks,
                                const uint8_t *act);
float oc_oxk_dot_q6_k_prepped(const void *scratch, size_t blocks,
                              const uint8_t *q8);
void oc_oxk_dot_q6_k_prepped_multi(const void *scratch, size_t blocks,
                                   const uint8_t *acts, size_t act_stride,
                                   size_t n_act, float *out);

/* Quantized weight × f32-input matvec. `w` is `n_rows` rows of `row_bytes` */
void oc_oxk_matvec_q4_0_f32(const uint8_t *w, size_t n_rows, size_t row_bytes,
                            const float *x, float *out);
void oc_oxk_matvec_q4_k_f32(const uint8_t *w, size_t n_rows, size_t row_bytes,
                            const float *x, float *out);
void oc_oxk_matvec_q8_0_f32(const uint8_t *w, size_t n_rows, size_t row_bytes,
                            const float *x, float *out);

float oc_oxk_dot_q4_0_q8_0_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);

void oc_oxk_matvec_q4_0_f32_scalar(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q4_k_f32_scalar(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q8_0_f32_scalar(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);

float oc_oxk_dot_q4_0_q8_0_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
void oc_oxk_matvec_q4_0_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q4_k_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q8_0_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out);

float oc_oxk_dot_q4_0_q8_0_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
void oc_oxk_matvec_q4_0_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q4_k_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q8_0_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);

/* Bit-exact f16 → f32 conversion (no libm). */
float oc_oxk_f16_le_to_f32(const uint8_t p[2]);

/* Decode (scale, min) for sub-group `j` (0..7) from a Q4_K/Q5_K 12-byte
 * scale field. Identical to llama.cpp's `get_scale_min_k4`. */
void oc_oxk_get_scale_min_k4(unsigned j, const uint8_t scales[12],
                             uint8_t *scale, uint8_t *min);

/* Read the i16 bsum at `index` (0..15) from a Q8_K block's bsums field. */
int16_t oc_oxk_read_q8_k_bsum(const uint8_t *bsums, size_t index);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OXK_H */
