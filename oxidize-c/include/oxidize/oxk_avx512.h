/* oxk_avx512.h — AVX-512BW + AVX-512VNNI OXK kernel declarations. Bit-exactness invariants (VAL-OXK-007..011): they inherit the same invariants. */
#ifndef OXIDIZE_OXK_AVX512_H
#define OXIDIZE_OXK_AVX512_H

#include <stddef.h>
#include <stdint.h>

/* target("avx512...") is an x86-only function attribute; gcc rejects it when
 * cross-compiling for aarch64. Off-x86 these symbols are scalar-forwarding
 * stubs (see oxk_avx512.c), so the attribute must be empty there. */
#if defined(__x86_64__) || defined(__i386__)
#define OC_OXK_AVX512_TARGET __attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
#else
#define OC_OXK_AVX512_TARGET
#endif

#ifdef __cplusplus
extern "C" {
#endif

OC_OXK_AVX512_TARGET
float oc_oxk_dot_q8_0_q8_0_avx512_vnni(const uint8_t *row, size_t blocks,
                                        const uint8_t *q8);

OC_OXK_AVX512_TARGET
float oc_oxk_dot_q4_0_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

OC_OXK_AVX512_TARGET
float oc_oxk_dot_q4_1_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

OC_OXK_AVX512_TARGET
void oc_oxk_matvec_q8_0_f32_avx512_vnni(const uint8_t *w, size_t n_rows,
                                        size_t row_bytes, const float *x,
                                        float *out);

OC_OXK_AVX512_TARGET
void oc_oxk_matvec_q4_0_f32_avx512_bw(const uint8_t *w, size_t n_rows,
                                      size_t row_bytes, const float *x,
                                      float *out);

/* Bit-exact with oc_oxk_dot_q4_k_prepped(): all sub-group products stay in int32. */
OC_OXK_AVX512_TARGET
void oc_oxk_dot_q4_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out);

OC_OXK_AVX512_TARGET
float oc_oxk_dot_q6_k_q8_k_avx512vnni(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

/* Multi-activation dot over a prepared Q6_K row (oc_oxk_q6_k_prep_row layout). Same structure as the Q4_K prepped multi kernel, with 16 signed 16-element scale groups per block and the -32 offset folded out through the activation block sums. */
OC_OXK_AVX512_TARGET
/* Multi-activation dots over prepared Q3_K / Q2_K rows. */
/* Single-activation VNNI dots over the prepared rows — what decode uses,
 * where there is no activation tile to amortize a wider kernel over. */
float oc_oxk_dot_q2_k_prepped_avx512(const void *prep, size_t blocks,
                                     const uint8_t *act);
float oc_oxk_dot_q3_k_prepped_avx512(const void *prep, size_t blocks,
                                     const uint8_t *act);
float oc_oxk_dot_q6_k_prepped_avx512(const void *prep, size_t blocks,
                                     const uint8_t *act);

void oc_oxk_dot_q3_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out);
void oc_oxk_dot_q2_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out);

void oc_oxk_dot_q6_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out);

OC_OXK_AVX512_TARGET
float oc_oxk_dot_q5_k_q8_k_avx512vnni(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OXK_AVX512_H */
