/*
 * oxk_neon.h — AArch64 Advanced SIMD (NEON) OXK kernel declarations.
 *
 * NEON is architecturally mandatory on AArch64, so there is no runtime
 * feature test for the baseline kernels declared here: they are compiled
 * (and selected by the dispatcher in oxk.c) whenever `__aarch64__` is
 * defined, and do not exist at all on x86 builds. This mirrors how
 * oxk_avx2.c / oxk_avx512.c are x86-only, except that no
 * `__attribute__((target(...)))` is needed — the ISA is the baseline.
 *
 * Optional AArch64 extensions (dotprod / i8mm / fp16) are deliberately NOT
 * used: they would require a `getauxval(AT_HWCAP)` gate plus a second copy
 * of every kernel, and the widening-multiply path below already saturates
 * the load ports for these block layouts.
 *
 * Bit-exactness invariants (VAL-OXK-NEON-001..006), matching the contract in
 * oxk.h — the scalar reference in oxk.c is the ground truth:
 *   - Q4_0, Q4_1, Q8_0, Q4_K, Q5_K: every multiply-accumulate that the
 *     scalar path performs in *integer* arithmetic is performed in integer
 *     arithmetic here too (int8 → int16 widening multiply, pairwise
 *     accumulate into int32). Integer addition is associative, so lane
 *     reordering cannot change the result. The per-block / per-sub-group
 *     f32 expressions (`dw * dq * (float)isum`, `dw*dq*sc*sum - dw*dmin*dq*m*bs`)
 *     are then evaluated in scalar C in exactly the source order used by the
 *     scalar reference. => bit-exact.
 *   - Q6_K: NOT bit-exact. The scalar reference accumulates one f32 term per
 *     *element* (`sum += dw*dq*sc*(float)(q*q8)`), which cannot be vectorized
 *     without regrouping. The NEON kernel accumulates each 16-element
 *     scale-group in int32 and performs one f32 multiply-add per group
 *     (64 f32 adds per super-block instead of 1024). This is strictly *more*
 *     accurate than the scalar path but differs in the last ulp; parity tests
 *     use a relative tolerance for Q6_K only.
 *   - f16 → f32 uses the shared scalar `oc_oxk_f16_le_to_f32` bit-twiddle
 *     (once per block; not a hot path), so it is trivially identical.
 *
 * The matvec kernels (`oc_oxk_matvec_*_f32`) are intentionally NOT
 * implemented in NEON: the scalar reference accumulates f32 per element in a
 * fixed order, and any vectorization reassociates that sum. Rather than break
 * the bit-exactness invariant for a path that is not the inference hot spot,
 * the dispatcher keeps them on the scalar reference (same choice the AVX2
 * file makes).
 */
#ifndef OXIDIZE_OXK_NEON_H
#define OXIDIZE_OXK_NEON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__aarch64__)

/* Quantized-weight × Q8-activation row dot products. Signatures are
 * identical to the scalar / AVX2 variants in oxk.h:
 *   `row`    — packed weight row, `blocks_per_row` blocks back-to-back
 *   `q8`     — packed Q8_0 (or Q8_K) activation row, same block count
 * Returns the f32 dot product. */
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
