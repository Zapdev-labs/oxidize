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

#if defined(__x86_64__) || defined(__i386__)
float oc_simd_dflash2_dot_f32_avx2(const float *a, const float *b, size_t n);
float oc_simd_dflash2_dot_bf16_avx2(const uint16_t *w, const float *x,
                                    size_t n);
void oc_simd_dflash2_gemv_rows_f32_avx2(const float *w, size_t rows,
                                        size_t cols, const float *x,
                                        float *out);
void oc_simd_dflash2_dot_bf16_batch_avx2(const uint16_t *w, const float *x,
                                         size_t width, size_t batch,
                                         float *out);
void oc_simd_dflash2_conv_mac_avx2(float *out, const float *kernel,
                                   const float *input, float dynamic_value,
                                   size_t n);
#endif

typedef float (*OcDFlash2DotF32Fn)(const float *, const float *, size_t);
typedef float (*OcDFlash2DotBf16Fn)(const uint16_t *, const float *, size_t);
typedef void (*OcDFlash2GemvRowsF32Fn)(const float *, size_t, size_t,
                                      const float *, float *);
typedef void (*OcDFlash2DotBf16BatchFn)(const uint16_t *, const float *,
                                        size_t, size_t, float *);
typedef void (*OcDFlash2ConvMacFn)(float *, const float *, const float *,
                                   float, size_t);

typedef struct OcDFlash2SimdDispatch {
    OcDFlash2DotF32Fn dot_f32;
    OcDFlash2DotBf16Fn dot_bf16;
    OcDFlash2GemvRowsF32Fn gemv_rows_f32;
    OcDFlash2DotBf16BatchFn dot_bf16_batch;
    OcDFlash2ConvMacFn conv_mac;
} OcDFlash2SimdDispatch;

#if defined(__x86_64__) || defined(__i386__)
static const OcDFlash2SimdDispatch s_dflash2_avx2 = {
    oc_simd_dflash2_dot_f32_avx2,
    oc_simd_dflash2_dot_bf16_avx2,
    oc_simd_dflash2_gemv_rows_f32_avx2,
    oc_simd_dflash2_dot_bf16_batch_avx2,
    oc_simd_dflash2_conv_mac_avx2,
};
#endif

/* ─── Capability detection ────────────────────────────────────────────── */

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

    /* AVX-512 BW + DQ + VNNI together define the "useful for quant" tier
     * (skylake-x without VNNI is intentionally not preferred over AVX2
     * because VNNI is the win for int8 dot products; plain AVX-512 F is
     * rarely worth the frequency penalty). */
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
    /* NEON (Advanced SIMD) is mandatory in the AArch64 base architecture —
     * there is no HWCAP bit worth testing, so detection is compile-time.
     * Optional extensions (dotprod / i8mm) would need getauxval(AT_HWCAP);
     * the OXK NEON kernels deliberately do not use them. */
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

static const OcDFlash2SimdDispatch *dflash2_dispatch(void)
{
#if defined(__x86_64__) || defined(__i386__)
    const OcSimdLevel level = oc_simd_caps()->level;
    if (level == OC_SIMD_AVX2 || level == OC_SIMD_AVX512)
        return &s_dflash2_avx2;
#endif
    return NULL;
}

bool oc_simd_try_dflash2_dot_f32(const float *a, const float *b, size_t n,
                                 float *result)
{
    if (!a || !b || !result || n == 0) return false;
    const OcDFlash2SimdDispatch *dispatch = dflash2_dispatch();
    if (!dispatch) return false;
    *result = dispatch->dot_f32(a, b, n);
    return true;
}

bool oc_simd_try_dflash2_dot_bf16(const uint16_t *w, const float *x, size_t n,
                                  float *result)
{
    if (!w || !x || !result || n == 0) return false;
    const OcDFlash2SimdDispatch *dispatch = dflash2_dispatch();
    if (!dispatch) return false;
    *result = dispatch->dot_bf16(w, x, n);
    return true;
}

bool oc_simd_try_dflash2_gemv_rows_f32(const float *w, size_t rows,
                                       size_t cols, const float *x,
                                       float *out)
{
    if (!w || !x || !out || rows == 0 || cols == 0) return false;
    const OcDFlash2SimdDispatch *dispatch = dflash2_dispatch();
    if (!dispatch) return false;
    dispatch->gemv_rows_f32(w, rows, cols, x, out);
    return true;
}

bool oc_simd_try_dflash2_dot_bf16_batch(const uint16_t *w, const float *x,
                                        size_t width, size_t batch,
                                        float *out)
{
    if (!w || !x || !out || width == 0 || batch == 0 || batch > 32)
        return false;
    const OcDFlash2SimdDispatch *dispatch = dflash2_dispatch();
    if (!dispatch) return false;
    dispatch->dot_bf16_batch(w, x, width, batch, out);
    return true;
}

bool oc_simd_try_dflash2_conv_mac(float *out, const float *kernel_values,
                                  const float *input, float dynamic_value,
                                  size_t n)
{
    if (!out || !kernel_values || !input || n == 0) return false;
    const OcDFlash2SimdDispatch *dispatch = dflash2_dispatch();
    if (!dispatch) return false;
    dispatch->conv_mac(out, kernel_values, input, dynamic_value, n);
    return true;
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

#if !defined(__x86_64__) && !defined(__i386__)
    /* Off x86 there is no SIMD dequant kernel, so qtype goes unread and the
     * caller always takes the scalar path. */
    (void)qtype;
#endif

    const OcSimdCaps *caps = oc_simd_caps();
    switch (caps->level) {
/* The AVX kernels are only *defined* on x86 (simd_avx2.c / simd_avx512.c are
 * wholly inside an `#if defined(__x86_64__) || defined(__i386__)`). These
 * branches are unreachable elsewhere — caps->level can never report an x86
 * tier on another architecture — but an unguarded call still needs the symbol
 * at link time, which is what broke the aarch64 link. */
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
