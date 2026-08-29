/* oxk_neon.h — AArch64 Advanced SIMD (NEON) OXK kernel declarations.
 * Bit-exactness VAL-OXK-NEON-001..006: Q4_0/Q4_1/Q8_0/Q4_K/Q5_K are bit-exact
 * (int mul-accum); Q6_K is NOT bit-exact (f32 per 16-element scale-group) and
 * parity tests use a relative tolerance. */
#ifndef OXIDIZE_OXK_NEON_H
#define OXIDIZE_OXK_NEON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__aarch64__)

/* Quantized-weight × Q8-activation row dot products, identical to the scalar/AVX2 variants in oxk.h; return the f32 dot product. */
float oc_oxk_dot_q4_0_q8_0_neon(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0_neon(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0_neon(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k_neon(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k_neon(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k_neon(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);

#endif /* __aarch64__ */

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OXK_NEON_H */
