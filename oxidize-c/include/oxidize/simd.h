/*
 * simd.h — runtime SIMD dispatch for quantization hot paths.
 *
 * Implements the `quant-simd-dispatch` feature: a single binary detects the
 * host CPU level at first use and routes the hottest dequant kernels to the
 * best available SIMD implementation (AVX-512 BW+VNNI > AVX2+FMA+F16C >
 * scalar). All SIMD kernels are bit-exact with the scalar reference in
 * src/compute/quantization.c (VAL-SIMD-001..004); the dispatch never changes
 * numerical results, only throughput.
 *
 * Per CONTRIBUTING.md: intrinsics MUST NOT be called outside src/core/simd_*.c.
 * Callers go through `oc_simd_try_dequant`, which returns false if no SIMD
 * kernel is available for the given (qtype, caps) pair — the caller then falls
 * back to the scalar reference.
 *
 * Compile model: kernels live in simd_avx2.c / simd_avx512.c and are guarded
 * by `__attribute__((target(...)))` so the SAME binary runs on SSE2-only
 * hosts (kernels are unused there; cpuid reports OC_SIMD_SCALAR and the
 * scalar fallback handles every type). The `make avx512` target additionally
 * defines OC_AVX512 to make AVX-512 the compile baseline (still dispatched at
 * runtime; the attribute is idempotent).
 */
#ifndef OXIDIZE_SIMD_H
#define OXIDIZE_SIMD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/quant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Capability detection ──────────────────────────────────────────────
 *
 * Detected once via `__builtin_cpu_supports` (gcc/clang) on first call and
 * cached. The level is monotonic: AVX-512 implies AVX2 implies SCALAR.
 */
typedef enum {
    OC_SIMD_SCALAR  = 0,   /* no usable SIMD acceleration                */
    OC_SIMD_AVX2    = 1,   /* AVX2 + FMA + F16C (Haswell+, Zen2+)        */
    OC_SIMD_AVX512  = 2,   /* AVX-512 BW + DQ + VNNI (Cascade Lake+, Zen4) */
} OcSimdLevel;

typedef struct OcSimdCaps {
    OcSimdLevel level;
    bool has_f16c;    /* f16 → f32 hardware conversion (needed by AVX2+)   */
    bool has_fma;    /* FMA3 (available on every AVX2 host in practice)   */
    bool has_vnni;   /* AVX-512 VNNI (DP4A / VPDPBUSD)                    */
    const char *name; /* "scalar" | "avx2" | "avx512"                     */
} OcSimdCaps;

/* Returns a pointer to the cached capability struct. Triggers detection on
 * first call (thread-safe via C11 atomic flag, one-time init). */
const OcSimdCaps *oc_simd_caps(void);

/* ─── Dispatched dequant entry ──────────────────────────────────────────
 *
 * If a SIMD kernel is available for `qtype` on the detected host, dequants
 * `src` → `dst` and returns true. If no SIMD kernel exists for this type or
 * level, returns false WITHOUT touching `dst` (caller falls back to the
 * scalar reference in src/compute/quantization.c).
 *
 * Argument semantics match `oc_quant_dequant_row`: `src_len` is the packed
 * byte count, `value_count` is the f32 output count. The SIMD kernels
 * perform the same layout validation as the scalar path and return false
 * (rather than an error) on layout mismatch — callers' scalar fallback will
 * produce the canonical OC_ERR_INVALID_ARG.
 *
 * Bit-exactness (VAL-SIMD-001..004):
 *   - Q4_0, Q8_0: single multiply per element → identical rounding to scalar.
 *   - Q4_1, Q4_K: separate vmulps + vaddps/vsubps (NO FMA) so the two FP
 *     roundings match the scalar `a*b ± c` exactly.
 *   - f16 → f32: uses vcvtph2ps, the canonical f16→f32 conversion (f16 is a
 *     strict subset of f32; result is bit-identical to the scalar bit-twiddle
 *     in f16_le_to_f32).
 */
bool oc_simd_try_dequant(OcGgufQuantizationType qtype,
                         const uint8_t *src, size_t src_len,
                         float *dst, size_t value_count);

/* ─── Direct kernel entry points (for testing/benchmarking) ─────────────
 *
 * Exposed so tests can exercise each kernel directly without going through
 * the runtime dispatch. Each returns true on success, false on layout error
 * or if the kernel was not compiled in. The AVX2 kernels are guarded by
 * `__attribute__((target("avx2,f16c")))` and are present in every build;
 * the AVX-512 kernels are guarded by `target("avx512bw,avx512dq,avx512vnni")`
 * and likewise present in every build (the dispatcher selects them only on
 * capable hosts). These MUST NOT be called outside src/core/simd*.c — callers
 * go through oc_simd_try_dequant.
 */
bool oc_simd_dequant_q4_0_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count);
bool oc_simd_dequant_q4_1_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count);
bool oc_simd_dequant_q8_0_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count);
bool oc_simd_dequant_q4_k_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count);

bool oc_simd_dequant_q4_0_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count);
bool oc_simd_dequant_q4_1_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count);
bool oc_simd_dequant_q8_0_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count);
bool oc_simd_dequant_q4_k_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SIMD_H */
