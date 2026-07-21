/*
 * simd.c — runtime SIMD dispatch hub.
 *
 * Detects host CPU capabilities once (via __builtin_cpu_supports on gcc/clang,
 * falling back to SCALAR on other compilers) and routes dequant kernels to
 * the best available SIMD implementation. Intrinsics live ONLY in
 * simd_avx2.c / simd_avx512.c (per CONTRIBUTING.md); this file contains no
 * intrinsics, only function-pointer routing.
 *
 * Bit-exactness contract: the dispatched kernel MUST produce output
 * byte-identical to the scalar reference in src/compute/quantization.c.
 * Tests/test_simd.c asserts this on randomized inputs for every accelerated
 * type (VAL-SIMD-001..004).
 */
#include "oxidize/simd.h"

#include <stdatomic.h>
#include <string.h>

/* ─── Capability detection ────────────────────────────────────────────── */

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#  define OC_HAVE_BUILTIN_CPU 1
#else
#  define OC_HAVE_BUILTIN_CPU 0
#endif

static OcSimdCaps detect_caps(void)
{
    OcSimdCaps c;
    memset(&c, 0, sizeof(c));

#if OC_HAVE_BUILTIN_CPU
    c.has_f16c = __builtin_cpu_supports("f16c");
    c.has_fma  = __builtin_cpu_supports("fma");
    /* AVX-512 BW + DQ + VNNI together define the "useful for quant" tier
     * (skylake-x without VNNI is intentionally not preferred over AVX2
     * because VNNI is the win for int8 dot products; plain AVX-512 F is
     * rarely worth the frequency penalty). */
    bool avx512bw   = __builtin_cpu_supports("avx512bw");
    bool avx512dq   = __builtin_cpu_supports("avx512dq");
    bool avx512vnni = __builtin_cpu_supports("avx512vnni");
    bool avx512f    = __builtin_cpu_supports("avx512f");
    bool avx2       = __builtin_cpu_supports("avx2");

    if (avx512f && avx512bw && avx512dq && avx512vnni) {
        c.level    = OC_SIMD_AVX512;
        c.has_vnni = true;
        c.name     = "avx512";
    } else if (avx2 && c.has_fma && c.has_f16c) {
        c.level = OC_SIMD_AVX2;
        c.name  = "avx2";
    } else {
        c.level = OC_SIMD_SCALAR;
        c.name  = "scalar";
    }
#else
    c.level = OC_SIMD_SCALAR;
    c.name  = "scalar";
#endif
    return c;
}

const OcSimdCaps *oc_simd_caps(void)
{
    static _Atomic int s_inited = 0;
    static OcSimdCaps s_caps;
    int expected = 0;
    if (atomic_compare_exchange_strong(&s_inited, &expected, 1)) {
        s_caps = detect_caps();
    }
    /* Subsequent callers see the fully-written struct. atomic_compare_exchange
     * with seq_cst (default) provides the needed release/acquire fence; the
     * `expected == 1` path below only reads a stable pointer. */
    return &s_caps;
}

/* ─── Dispatch entry ──────────────────────────────────────────────────── */

bool oc_simd_try_dequant(OcGgufQuantizationType qtype,
                         const uint8_t *src, size_t src_len,
                         float *dst, size_t value_count)
{
    /* Validate up front — mirrors the scalar `validate_layout` contract.
     * On failure return false so the scalar fallback produces the canonical
     * OC_ERR_INVALID_ARG (SIMD path must not synthesize different errors). */
    if (src == NULL || dst == NULL || src_len == 0 || value_count == 0) {
        return false;
    }

    const OcSimdCaps *caps = oc_simd_caps();
    switch (caps->level) {
    case OC_SIMD_AVX512:
        switch (qtype) {
        case OC_QUANT_Q4_0: return oc_simd_dequant_q4_0_avx512(src, src_len, dst, value_count);
        case OC_QUANT_Q4_1: return oc_simd_dequant_q4_1_avx512(src, src_len, dst, value_count);
        case OC_QUANT_Q8_0: return oc_simd_dequant_q8_0_avx512(src, src_len, dst, value_count);
        case OC_QUANT_Q4_K_S:
        case OC_QUANT_Q4_K_M: return oc_simd_dequant_q4_k_avx512(src, src_len, dst, value_count);
        default: return false;
        }
        break;
    case OC_SIMD_AVX2:
        switch (qtype) {
        case OC_QUANT_Q4_0: return oc_simd_dequant_q4_0_avx2(src, src_len, dst, value_count);
        case OC_QUANT_Q4_1: return oc_simd_dequant_q4_1_avx2(src, src_len, dst, value_count);
        case OC_QUANT_Q8_0: return oc_simd_dequant_q8_0_avx2(src, src_len, dst, value_count);
        case OC_QUANT_Q4_K_S:
        case OC_QUANT_Q4_K_M: return oc_simd_dequant_q4_k_avx2(src, src_len, dst, value_count);
        default: return false;
        }
        break;
    case OC_SIMD_SCALAR:
    default:
        return false;
    }
}
