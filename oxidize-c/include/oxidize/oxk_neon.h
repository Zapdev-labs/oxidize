/* oxk_neon.h — AArch64 Advanced SIMD (NEON) OXK kernel declarations. */
#ifndef OXIDIZE_OXK_NEON_H
#define OXIDIZE_OXK_NEON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__aarch64__)

/* Quantized-weight × Q8-activation row dot products. Signatures are */
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
