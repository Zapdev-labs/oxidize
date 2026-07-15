/* Private seam between quant.c (scalar + dispatch) and the per-ISA kernel
 * translation units (quant_avx2.c, quant_avx512.c, quant_vnni.c).
 *
 * Why separate TUs and not __attribute__((target(...))) per function: the
 * kernels below all inline these helpers, and GCC refuses to inline a
 * function without a target attribute into one that has it ("target specific
 * option mismatch"). Attributing the helpers too would mean one copy of each
 * per ISA anyway — which is exactly what a per-TU -m flag gives for free, and
 * it also lets the compiler use the ISA everywhere in the kernel (F16C for the
 * block scales, FMA in the reduction) instead of only where an intrinsic says
 * so. The cost is that these helpers are compiled once per TU; they are all
 * `static inline` and tiny.
 *
 * Every kernel here assumes its ISA is present. Nothing may call one directly:
 * go through the dispatch table in quant.c, which is resolved from cpuid. */
#ifndef OC_QUANT_IMPL_H
#define OC_QUANT_IMPL_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "quant.h"

static inline uint16_t rd16(const uint8_t* p) {
  return (uint16_t)(p[0] | (uint16_t)p[1] << 8);
}

static inline uint32_t rd32(const uint8_t* p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

/* f16 -> f32. The hardware instruction when this TU is built with F16C (every
 * SIMD TU is), the software path in the portable baseline TU. */
#if defined(__F16C__)
#include <immintrin.h>
#define oc_f16_fast(h) _cvtsh_ss(h)
#else
#define oc_f16_fast(h) oc_f16_to_f32(h)
#endif

/* Some AL5 files carry corrupt blocks with inf/huge f16 scales (seen:
 * blk.16.ffn_down rows 0-367 of gemma-4-31B-it-AL5_XS, scale 0x7C00). Sane
 * scales are < 0.3 model-wide; zero out anything non-finite or > 1 so a bad
 * block contributes nothing instead of NaN-poisoning the residual stream. */
static inline float al5xs_scale(const uint8_t* blk) {
  float s = oc_f16_fast(rd16(blk));
  return (isfinite(s) && fabsf(s) <= 1.0f) ? s : 0.0f;
}

static inline void get_scale_min_k4(size_t j, const uint8_t* scales, uint8_t* sc,
                                    uint8_t* m) {
  if (j < 4) {
    *sc = scales[j] & 63;
    *m = scales[j + 4] & 63;
  } else {
    *sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
    *m = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
  }
}

#if defined(__x86_64__) || defined(__i386__)
/* quant_avx2.c   — AVX2 + FMA + F16C */
float oc_dot_q4_k_avx2(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q5_k_avx2(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q6_k_avx2(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q2_k_avx2(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q3_k_avx2(const uint8_t* row, const float* x, size_t cols);
float oc_dot_iq4_xs_avx2(const uint8_t* row, const float* x, size_t cols);
float oc_dot_bf16_avx2(const uint8_t* row, const float* x, size_t cols);
float oc_dot_al5xs_avx2(const uint8_t* row, const float* x, size_t cols);
void oc_gemm_row_avx2(float* acc, const float* w, const float* xp, size_t kb,
                      size_t n);
void oc_gemm_row4_avx2(float* acc, const float* w, size_t ws, const float* xp,
                       size_t kb, size_t n);
void oc_dequant_al5xs_avx2(const uint8_t* row, float* out, size_t n);
void oc_dequant_q4_k_avx2(const uint8_t* row, float* out, size_t n);
void oc_dequant_q5_k_avx2(const uint8_t* row, float* out, size_t n);
void oc_dequant_q6_k_avx2(const uint8_t* row, float* out, size_t n);

/* quant_avx512.c — AVX512F/BW/VL (no VNNI: a Skylake-X has the former and not
 * the latter, and a vpdpbusd emitted into one of these would SIGILL there) */
float oc_dot_q4_k_avx512(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q5_k_avx512(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q6_k_avx512(const uint8_t* row, const float* x, size_t cols);
float oc_dot_al5xs_avx512(const uint8_t* row, const float* x, size_t cols);
void oc_q8_quantize_avx512(const float* x, size_t n, int8_t* q, float* d,
                           int32_t* bsum);
void oc_gemm_row_avx512(float* acc, const float* w, const float* xp, size_t kb,
                        size_t n);
void oc_gemm_row4_avx512(float* acc, const float* w, size_t ws, const float* xp,
                         size_t kb, size_t n);

/* quant_vnni.c   — the above + AVX512-VNNI */
float oc_dot_q4_k_vnni(const uint8_t* row, const OcQ8Act* a, size_t cols);
float oc_dot_q5_k_vnni(const uint8_t* row, const OcQ8Act* a, size_t cols);
float oc_dot_q6_k_vnni(const uint8_t* row, const OcQ8Act* a, size_t cols);
#endif

#if defined(__aarch64__)
/* quant_neon.c — ARM NEON / ASIMD. Baseline armv8-a (no dotprod needed): the
 * math is the same float-FMA decode as the AVX2 kernels, just 128-bit wide, so
 * every kernel is provably == the scalar reference the differential checks. */
float oc_dot_q4_0_neon(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q8_0_neon(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q4_k_neon(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q5_k_neon(const uint8_t* row, const float* x, size_t cols);
float oc_dot_q6_k_neon(const uint8_t* row, const float* x, size_t cols);
float oc_dot_bf16_neon(const uint8_t* row, const float* x, size_t cols);
void oc_dequant_q4_k_neon(const uint8_t* row, float* out, size_t n);
void oc_dequant_q5_k_neon(const uint8_t* row, float* out, size_t n);
void oc_dequant_q6_k_neon(const uint8_t* row, float* out, size_t n);
void oc_gemm_row_neon(float* acc, const float* w, const float* xp, size_t kb,
                      size_t n);
void oc_gemm_row4_neon(float* acc, const float* w, size_t ws, const float* xp,
                       size_t kb, size_t n);
#endif

#endif
