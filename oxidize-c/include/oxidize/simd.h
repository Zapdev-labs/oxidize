/* simd.h — runtime SIMD dispatch for quantization hot paths. */
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

typedef enum {
    OC_SIMD_SCALAR  = 0,   /* no usable SIMD acceleration                */
    OC_SIMD_AVX2    = 1,   /* AVX2 + FMA + F16C (Haswell+, Zen2+)        */
    OC_SIMD_AVX512  = 2,   /* AVX-512 BW + DQ + VNNI (Cascade Lake+, Zen4) */
    OC_SIMD_NEON    = 3,   /* AArch64 Advanced SIMD (Apple Silicon, Graviton,
                            * Ampere). Disjoint from the x86 tiers — the
                            * ordering is NOT monotonic across ISAs, so
                            * callers must compare for equality, never `>=`. */
} OcSimdLevel;

typedef struct OcSimdCaps {
    OcSimdLevel level;
    bool has_f16c;    /* f16 → f32 hardware conversion (needed by AVX2+)   */
    bool has_fma;    /* FMA3 (available on every AVX2 host in practice)   */
    bool has_vnni;   /* AVX-512 VNNI (DP4A / VPDPBUSD)                    */
    bool has_neon;   /* AArch64 Advanced SIMD (architecturally guaranteed) */
    const char *name; /* "scalar" | "avx2" | "avx512" | "neon"            */
} OcSimdCaps;

/* Returns a pointer to the cached capability struct. Triggers detection on
 * first call (thread-safe via C11 atomic flag, one-time init). */
const OcSimdCaps *oc_simd_caps(void);

/* Dispatched dequant entry. Returns false WITHOUT touching `dst` so the caller can fall back to scalar. Bit-exactness VAL-SIMD-001..004. */
bool oc_simd_try_dequant(OcGgufQuantizationType qtype,
                         const uint8_t *src, size_t src_len,
                         float *dst, size_t value_count);

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
