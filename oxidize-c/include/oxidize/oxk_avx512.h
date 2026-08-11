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

/* ─── AVX-512VNNI Q8_0 × Q8_0 dot product ────────────────────────────────
 *
 * Uses `_mm512_dpbusd_epi32` (uint8 × int8 → int32 accumulate) to process
 * 64 element-pairs per VNNI call. Since Q8_0 weights are *signed* int8 and
 * VNNI expects one operand as unsigned uint8, the sign of the weight is
 * handled by splitting into positive/negative lanes: the positive weights are
 * fed directly to VNNI while negative weights are negated, fed to VNNI, and
 * their contribution subtracted. The final f32 scale multiply (`dw * dq`)
 * matches the scalar path exactly. */
OC_OXK_AVX512_TARGET
float oc_oxk_dot_q8_0_q8_0_avx512_vnni(const uint8_t *row, size_t blocks,
                                        const uint8_t *q8);

/* ─── AVX-512BW Q4_0 × Q8_0 dot product ───────────────────────────────────
 *
 * Unpacks 4-bit nibbles into bytes using AVX-512BW mask + shift, applies the
 * Q4_0 bias (-8) in the int8 domain, then accumulates with the Q8_0 values
 * via AVX-512VNNI. The final f32 scale multiply matches the scalar path. */
OC_OXK_AVX512_TARGET
float oc_oxk_dot_q4_0_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

/* ─── AVX-512BW Q4_1 × Q8_0 dot product ───────────────────────────────────
 *
 * Same as Q4_0 BW but includes the Q4_1 min offset (m) contribution.
 * Uses `_mm512_dpbusd_epi32` for the nibble × q8 accumulation and a separate
 * integer sum of q8 values for the `mw * dq * q8_sum` term. */
OC_OXK_AVX512_TARGET
float oc_oxk_dot_q4_1_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

/* ─── AVX-512VNNI Q8_0 × f32 matvec ───────────────────────────────────────
 *
 * Iterates `n_rows` weight rows; each row is `row_bytes` of Q8_0 blocks. For
 * each row, the int8 weight × f32 input dot product is accumulated in f32.
 * Uses VNNI for the int8 × int8 portion when the f32 input is first
 * quantized to Q8_0 (caller passes a pre-quantized Q8 vector via `x_q8`; if
 * `x_q8` is NULL, the function falls back to a scalar f32 inner loop). */
OC_OXK_AVX512_TARGET
void oc_oxk_matvec_q8_0_f32_avx512_vnni(const uint8_t *w, size_t n_rows,
                                        size_t row_bytes, const float *x,
                                        float *out);

/* ─── AVX-512BW Q4_0 × f32 matvec ─────────────────────────────────────────
 *
 * Iterates `n_rows` weight rows; each row is `row_bytes` of Q4_0 blocks.
 * Unpacks nibbles with AVX-512BW and multiplies by the f32 input directly
 * (no VNNI for the f32 path — the nibble unpack is the speedup). */
OC_OXK_AVX512_TARGET
void oc_oxk_matvec_q4_0_f32_avx512_bw(const uint8_t *w, size_t n_rows,
                                      size_t row_bytes, const float *x,
                                      float *out);

/* ─── AVX-512VNNI prepared-Q4_K × Q8_K multi-activation dot ──────────────
 *
 * Dots ONE prepared Q4_K row (oc_oxk_q4_k_prep_row layout) against `n_act`
 * packed Q8_K activations spaced `act_stride` bytes apart, writing `n_act`
 * f32 results to `out`. The prepared codes are already unsigned bytes
 * 0..15, which is exactly the shape `_mm512_dpbusd_epi32` wants, so each
 * 64-element chunk is one load + one VNNI op per activation. Activations
 * are processed four at a time so the row's codes and scales are loaded
 * once per four dots.
 *
 * Bit-exact with oc_oxk_dot_q4_k_prepped(): all sub-group products and
 * scale multiplies stay in int32 (reassociation of integer adds is exact),
 * and the per-block float accumulation order is unchanged. */
OC_OXK_AVX512_TARGET
void oc_oxk_dot_q4_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out);

/* ─── AVX-512VNNI Q6_K × Q8_K dot ────────────────────────────────────────
 *
 * Unpacks the 6-bit values as unsigned 0..63 (low nibble | high 2 bits),
 * VNNI-accumulates against the Q8_K int8s per 16-element scale group, and
 * folds the implicit -32 offset out through the activation's stored block
 * sums: sum((q-32)*a) = sum(q*a) - 32*bsum. Bit-exact with the (integer-
 * accumulating) scalar reference. */
OC_OXK_AVX512_TARGET
float oc_oxk_dot_q6_k_q8_k_avx512vnni(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

/* Multi-activation dot over a prepared Q6_K row (oc_oxk_q6_k_prep_row
 * layout). Same structure as the Q4_K prepped multi kernel, with 16
 * signed 16-element scale groups per block and the -32 offset folded out
 * through the activation block sums. */
OC_OXK_AVX512_TARGET
/* Multi-activation dots over prepared Q3_K / Q2_K rows. Q3_K consumes the
 * Q6_K prepared layout (oc_oxk_q3_k_prep_row writes it); Q2_K consumes its
 * own (oc_oxk_q2_k_prep_row). Both fall back to the scalar prepped dot when
 * built without VNNI. */
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

/* ─── AVX-512VNNI Q5_K × Q8_K dot ────────────────────────────────────────
 *
 * Unpacks nibble | (qh group-bit << 4) as unsigned 0..31 (qh[l] holds one
 * bit per 64-element group), VNNI-accumulates per 32-element sub-group,
 * and applies the per-sub-group scale/min exactly as the scalar reference
 * does (integer until the per-block float multiply). */
OC_OXK_AVX512_TARGET
float oc_oxk_dot_q5_k_q8_k_avx512vnni(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OXK_AVX512_H */
