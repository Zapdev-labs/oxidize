/*
 * cpu_kernels.c — CPU kernel dispatch implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/cpu_kernels.h"

#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif

OcError oc_cpu_kernels_detect(OcCpuKernelCaps *caps)
{
    if (!caps) return OC_ERR_INVALID_ARG;
    memset(caps, 0, sizeof(*caps));

#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;

    /* CPUID 1 for feature flags. */
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        caps->has_sse42 = (ecx & bit_SSE4_2) != 0;
        caps->has_avx2 = false;
        caps->has_avx512 = false;
        caps->has_fma = (ecx & bit_FMA) != 0;
    }

    /* CPUID 7 for AVX2 and AVX-512. */
    unsigned int eax7, ebx7, ecx7, edx7;
    if (__get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7)) {
        caps->has_avx2 = (ebx7 & bit_AVX2) != 0;
        caps->has_avx512 = (ebx7 & bit_AVX512F) != 0;
        caps->has_avx512_vnni = (ecx7 & bit_AVX512VNNI) != 0;
    }

    /* Determine best level. */
    if (caps->has_avx512_vnni)
        caps->level = OC_CPU_KERNEL_AVX512_VNNI;
    else if (caps->has_avx512)
        caps->level = OC_CPU_KERNEL_AVX512;
    else if (caps->has_avx2)
        caps->level = OC_CPU_KERNEL_AVX2;
    else if (caps->has_sse42)
        caps->level = OC_CPU_KERNEL_SCALAR;
    else
        caps->level = OC_CPU_KERNEL_SCALAR;

    /* CPU model name from CPUID 0x80000002-0x80000004. */
    char name[49] = {0};
    unsigned int *name_int = (unsigned int *)name;
    if (__get_cpuid(0x80000002, &eax, &ebx, &ecx, &edx)) {
        name_int[0] = eax; name_int[1] = ebx; name_int[2] = ecx; name_int[3] = edx;
    }
    if (__get_cpuid(0x80000003, &eax, &ebx, &ecx, &edx)) {
        name_int[4] = eax; name_int[5] = ebx; name_int[6] = ecx; name_int[7] = edx;
    }
    if (__get_cpuid(0x80000004, &eax, &ebx, &ecx, &edx)) {
        name_int[8] = eax; name_int[9] = ebx; name_int[10] = ecx; name_int[11] = edx;
    }
    name[48] = '\0';
    /* Trim leading spaces. */
    char *p = name;
    while (*p == ' ') p++;
    caps->cpu_name = strdup(p);
#else
    caps->level = OC_CPU_KERNEL_SCALAR;
    caps->cpu_name = strdup("unknown");
#endif

    caps->has_neon = false;
    return OC_OK;
}

OcCpuKernelLevel oc_cpu_kernels_best_level(void)
{
    OcCpuKernelCaps caps;
    if (oc_cpu_kernels_detect(&caps) != OC_OK)
        return OC_CPU_KERNEL_SCALAR;
    OcCpuKernelLevel level = caps.level;
    free((void *)caps.cpu_name);
    return level;
}

float oc_cpu_dot_f32_scalar(const float *a, const float *b, size_t n)
{
    if (!a || !b) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

void oc_cpu_matvec_f32_scalar(const float *w, const float *x, float *out,
                               size_t n_rows, size_t n_cols)
{
    if (!w || !x || !out) return;
    for (size_t r = 0; r < n_rows; r++) {
        float sum = 0.0f;
        const float *row = w + r * n_cols;
        for (size_t c = 0; c < n_cols; c++)
            sum += row[c] * x[c];
        out[r] = sum;
    }
}


#if defined(OC_CPU_KERNELS_HAVE_AVX2)

__attribute__((target("avx2,fma")))
float oc_cpu_dot_f32_avx2(const float *a, const float *b, size_t n)
{
    if (!a || !b) return 0.0f;
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    /* Horizontal sum of the 8 lanes. */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx2,fma")))
void oc_cpu_matvec_f32_avx2(const float *w, const float *x, float *out,
                             size_t n_rows, size_t n_cols)
{
    if (!w || !x || !out) return;
    for (size_t r = 0; r < n_rows; r++)
        out[r] = oc_cpu_dot_f32_avx2(w + r * n_cols, x, n_cols);
}

#endif /* OC_CPU_KERNELS_HAVE_AVX2 */

#if defined(OC_CPU_KERNELS_HAVE_AVX512)

__attribute__((target("avx512f")))
float oc_cpu_dot_f32_avx512(const float *a, const float *b, size_t n)
{
    if (!a || !b) return 0.0f;
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx512f")))
void oc_cpu_matvec_f32_avx512(const float *w, const float *x, float *out,
                               size_t n_rows, size_t n_cols)
{
    if (!w || !x || !out) return;
    for (size_t r = 0; r < n_rows; r++)
        out[r] = oc_cpu_dot_f32_avx512(w + r * n_cols, x, n_cols);
}

#endif /* OC_CPU_KERNELS_HAVE_AVX512 */

OcError oc_cpu_kernels_init_scalar(OcCpuKernels *kernels)
{
    if (!kernels) return OC_ERR_INVALID_ARG;
    memset(kernels, 0, sizeof(*kernels));
    kernels->level = OC_CPU_KERNEL_SCALAR;
    kernels->dot_f32 = oc_cpu_dot_f32_scalar;
    kernels->matvec_f32 = oc_cpu_matvec_f32_scalar;
    return OC_OK;
}

OcError oc_cpu_kernels_init(OcCpuKernels *kernels, OcCpuKernelLevel level)
{
    if (!kernels) return OC_ERR_INVALID_ARG;
    /* Clamp the requested level to what this CPU can actually execute:
     * running an AVX-512 kernel on a machine without it is SIGILL, and
     * callers routinely ask for a level from config rather than CPUID. */
    OcCpuKernelLevel best = oc_cpu_kernels_best_level();
    if (level > best) level = best;
    /* VNNI adds integer kernels only; the f32 path is the AVX-512 one. */
    if (level == OC_CPU_KERNEL_AVX512_VNNI) level = OC_CPU_KERNEL_AVX512;

    OcError e = oc_cpu_kernels_init_scalar(kernels);
    if (e != OC_OK) return e;

#if defined(OC_CPU_KERNELS_HAVE_AVX512)
    if (level == OC_CPU_KERNEL_AVX512) {
        kernels->level = OC_CPU_KERNEL_AVX512;
        kernels->dot_f32 = oc_cpu_dot_f32_avx512;
        kernels->matvec_f32 = oc_cpu_matvec_f32_avx512;
        return OC_OK;
    }
#endif
#if defined(OC_CPU_KERNELS_HAVE_AVX2)
    if (level == OC_CPU_KERNEL_AVX2) {
        kernels->level = OC_CPU_KERNEL_AVX2;
        kernels->dot_f32 = oc_cpu_dot_f32_avx2;
        kernels->matvec_f32 = oc_cpu_matvec_f32_avx2;
        return OC_OK;
    }
#endif
    return OC_OK;
}

OcError oc_cpu_kernels_init_best(OcCpuKernels *kernels)
{
    if (!kernels) return OC_ERR_INVALID_ARG;
    return oc_cpu_kernels_init(kernels, oc_cpu_kernels_best_level());
}

const char *oc_cpu_kernel_level_name(OcCpuKernelLevel level)
{
    switch (level) {
    case OC_CPU_KERNEL_SCALAR:     return "scalar";
    case OC_CPU_KERNEL_AVX2:       return "avx2";
    case OC_CPU_KERNEL_AVX512:    return "avx512";
    case OC_CPU_KERNEL_AVX512_VNNI: return "avx512_vnni";
    case OC_CPU_KERNEL_NEON:      return "neon";
    default: return "unknown";
    }
}

bool oc_cpu_kernels_has_vnni(void)
{
    OcCpuKernelCaps caps;
    if (oc_cpu_kernels_detect(&caps) != OC_OK) return false;
    bool vnni = caps.has_avx512_vnni;
    free((void *)caps.cpu_name);
    return vnni;
}

bool oc_cpu_kernels_is_skylake_sp(void)
{
    OcCpuKernelCaps caps;
    if (oc_cpu_kernels_detect(&caps) != OC_OK) return false;
    bool is_skl = false;
    if (caps.cpu_name) {
        /* Check for Skylake-SP / Xeon Scalable. */
        if (strstr(caps.cpu_name, "Skylake") ||
            (strstr(caps.cpu_name, "Xeon") && strstr(caps.cpu_name, "Gold")))
            is_skl = true;
    }
    free((void *)caps.cpu_name);
    return is_skl;
}
