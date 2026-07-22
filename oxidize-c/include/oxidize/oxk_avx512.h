/*
 * oxk_avx512.h — AVX-512BW + AVX-512VNNI OXK kernel declarations.
 *
 * These are the *new* AVX-512 entry points that complement the existing
 * `oc_oxk_dot_*_avx512` / `oc_oxk_matvec_*_avx512` stubs in oxk.h (which
 * currently forward to scalar). They use the VNNI `_mm512_dpbusd_epi32`
 * instruction (uint8 × int8 → int32 multiply-add) and AVX-512BW nibble
 * unpacking for genuine speed-up over AVX2.
 *
 * All functions are guarded by `__attribute__((target("avx512bw,avx512dq,
 * avx512vnni")))` so they compile on any CPU but only execute correctly on
 * Cascade Lake+ / Zen4+. Callers MUST check `oc_oxk_caps()->level >=
 * OC_OXK_AVX512 && oc_oxk_caps()->has_vnni` before invoking.
 *
 * Bit-exactness invariants (VAL-OXK-007..011):
 *   - Q8_0×Q8_0 VNNI: VNNI computes int32 sum-of-products exactly; the only
 *     floating-point operation is the final `dw * dq * (float)isum` which
 *     matches the scalar path one-to-one.
 *   - Q4_0/Q4_1 BW: nibble unpacking uses integer mask+shift (no rounding);
 *     the subsequent VNNI accumulation is integer-exact, and the f32 scale
 *     multiply matches the scalar `dw * dq * (float)isum`.
 *   - Matvec variants iterate rows and reuse the dot-product kernels, so
 *     they inherit the same invariants.
 */
#ifndef OXIDIZE_OXK_AVX512_H
#define OXIDIZE_OXK_AVX512_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── AVX-512VNNI Q8_0 × Q8_0 dot product ────────────────────────────────
 *
 * Uses `_mm512_dpbusd_epi32` (uint8 × int8 → int32 accumulate) to process
 * 64 element-pairs per VNNI call. Since Q8_0 weights are *signed* int8 and
 * VNNI expects one operand as unsigned uint8, the sign of the weight is
 * handled by splitting into positive/negative lanes: the positive weights are
 * fed directly to VNNI while negative weights are negated, fed to VNNI, and
 * their contribution subtracted. The final f32 scale multiply (`dw * dq`)
 * matches the scalar path exactly. */
__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q8_0_q8_0_avx512_vnni(const uint8_t *row, size_t blocks,
                                        const uint8_t *q8);

/* ─── AVX-512BW Q4_0 × Q8_0 dot product ───────────────────────────────────
 *
 * Unpacks 4-bit nibbles into bytes using AVX-512BW mask + shift, applies the
 * Q4_0 bias (-8) in the int8 domain, then accumulates with the Q8_0 values
 * via AVX-512VNNI. The final f32 scale multiply matches the scalar path. */
__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_0_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

/* ─── AVX-512BW Q4_1 × Q8_0 dot product ───────────────────────────────────
 *
 * Same as Q4_0 BW but includes the Q4_1 min offset (m) contribution.
 * Uses `_mm512_dpbusd_epi32` for the nibble × q8 accumulation and a separate
 * integer sum of q8 values for the `mw * dq * q8_sum` term. */
__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_1_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

/* ─── AVX-512VNNI Q8_0 × f32 matvec ───────────────────────────────────────
 *
 * Iterates `n_rows` weight rows; each row is `row_bytes` of Q8_0 blocks. For
 * each row, the int8 weight × f32 input dot product is accumulated in f32.
 * Uses VNNI for the int8 × int8 portion when the f32 input is first
 * quantized to Q8_0 (caller passes a pre-quantized Q8 vector via `x_q8`; if
 * `x_q8` is NULL, the function falls back to a scalar f32 inner loop). */
__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q8_0_f32_avx512_vnni(const uint8_t *w, size_t n_rows,
                                        size_t row_bytes, const float *x,
                                        float *out);

/* ─── AVX-512BW Q4_0 × f32 matvec ─────────────────────────────────────────
 *
 * Iterates `n_rows` weight rows; each row is `row_bytes` of Q4_0 blocks.
 * Unpacks nibbles with AVX-512BW and multiplies by the f32 input directly
 * (no VNNI for the f32 path — the nibble unpack is the speedup). */
__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q4_0_f32_avx512_bw(const uint8_t *w, size_t n_rows,
                                      size_t row_bytes, const float *x,
                                      float *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OXK_AVX512_H */
