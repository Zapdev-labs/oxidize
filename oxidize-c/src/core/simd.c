/* simd.c — runtime SIMD dispatch hub. Bit-exactness contract: the dispatched kernel MUST produce output type (VAL-SIMD-001..004). */
#include "oxidize/simd.h"

#include <stdatomic.h>
#include <string.h>


/* Raw cpuid instead of __builtin_cpu_supports: older clang rejects feature
 * strings like "f16c" and "avx512vnni", so builtin-based detection does not
 * compile portably. <cpuid.h> ships with both gcc and clang. */
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#  define OC_HAVE_BUILTIN_CPU 1
#  include <cpuid.h>

static uint64_t oc_xgetbv0(void)
{
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}
#else
#  define OC_HAVE_BUILTIN_CPU 0
#endif

static OcSimdCaps detect_caps(void)
{
    OcSimdCaps c;
    memset(&c, 0, sizeof(c));

#if OC_HAVE_BUILTIN_CPU
    uint32_t eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        c.level = OC_SIMD_SCALAR;
        c.name  = "scalar";
        return c;
    }
    bool osxsave = (ecx >> 27) & 1;
    /* XCR0: bits 1|2 = XMM+YMM state; bits 5|6|7 = AVX-512 opmask/ZMM. */
    uint64_t xcr0 = osxsave ? oc_xgetbv0() : 0;
    bool os_avx    = (xcr0 & 0x06) == 0x06;
    bool os_avx512 = (xcr0 & 0xE6) == 0xE6;

    c.has_f16c = os_avx && ((ecx >> 29) & 1);
    c.has_fma  = os_avx && ((ecx >> 12) & 1);

    uint32_t b7 = 0, c7 = 0, d7 = 0, a7 = 0;
    if (__get_cpuid_count(7, 0, &a7, &b7, &c7, &d7) == 0) { b7 = 0; c7 = 0; }

    /* AVX-512 BW + DQ + VNNI together define the "useful for quant" tier (skylake-x without VNNI is intentionally not preferred over AVX2 because VNNI is the win for int8 dot products; plain AVX-512 F is rarely worth the frequency penalty). */
    bool avx512bw   = os_avx512 && ((b7 >> 30) & 1);
    bool avx512dq   = os_avx512 && ((b7 >> 17) & 1);
    bool avx512vnni = os_avx512 && ((c7 >> 11) & 1);
    bool avx512f    = os_avx512 && ((b7 >> 16) & 1);
    bool avx2       = os_avx && ((b7 >> 5) & 1);

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
#elif defined(__aarch64__)
    /* NEON (Advanced SIMD) is mandatory in the AArch64 base architecture — the OXK NEON kernels deliberately do not use them. */
    c.level    = OC_SIMD_NEON;
    c.has_neon = true;
    c.name     = "neon";
#else
    c.level = OC_SIMD_SCALAR;
    c.name  = "scalar";
#endif
    return c;
}

const OcSimdCaps *oc_simd_caps(void)
{
    static _Atomic int s_state = 0;
    static OcSimdCaps s_caps;
    int expected = 0;
    if (atomic_compare_exchange_strong_explicit(&s_state, &expected, 1,
                                                memory_order_acquire,
                                                memory_order_relaxed)) {
        s_caps = detect_caps();
        atomic_store_explicit(&s_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&s_state, memory_order_acquire) != 2) {}
    }
    return &s_caps;
}


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

#if !defined(__x86_64__) && !defined(__i386__)
    /* Off x86 there is no SIMD dequant kernel, so qtype goes unread and the
     * caller always takes the scalar path. */
    (void)qtype;
#endif

    const OcSimdCaps *caps = oc_simd_caps();
    switch (caps->level) {
/* The AVX kernels are only *defined* on x86 (simd_avx2.c / simd_avx512.c are */
#if defined(__x86_64__) || defined(__i386__)
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
#endif /* x86 */
    case OC_SIMD_SCALAR:
    default:
        return false;
    }
}
