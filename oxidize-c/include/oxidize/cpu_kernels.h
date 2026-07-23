/*
 * cpu_kernels.h — CPU kernel dispatch.
 *
 * Provides function pointers that dispatch to the best available SIMD
 * implementation (scalar, AVX2, AVX-512) for each kernel.
 * Port from oxidize-core/src/compute/ and oxidize-kernels/.
 */
#ifndef OXIDIZE_CPU_KERNELS_H
#define OXIDIZE_CPU_KERNELS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_CPU_KERNEL_SCALAR = 0,
    OC_CPU_KERNEL_AVX2 = 1,
    OC_CPU_KERNEL_AVX512 = 2,
    OC_CPU_KERNEL_AVX512_VNNI = 3,
    OC_CPU_KERNEL_NEON = 4,
} OcCpuKernelLevel;

typedef struct {
    OcCpuKernelLevel level;
    bool has_fma;
    bool has_avx2;
    bool has_avx512;
    bool has_avx512_vnni;
    bool has_neon;
    bool has_sse42;
    const char *cpu_name;
} OcCpuKernelCaps;

/* Kernel function pointer types. */
typedef float (*OcDotProductFn)(const float *a, const float *b, size_t n);
typedef void (*OcMatVecFn)(const float *w, const float *x, float *out,
                           size_t n_rows, size_t n_cols);
typedef void (*OcQuantizeFn)(const float *src, void *dst, size_t n);
typedef float (*OcDequantizeRowFn)(const void *src, float *dst, size_t n);

typedef struct {
    OcDotProductFn dot_f32;
    OcMatVecFn matvec_f32;
    OcQuantizeFn quantize_q8_0;
    OcQuantizeFn quantize_q4_0;
    OcDequantizeRowFn dequantize_q8_0;
    OcDequantizeRowFn dequantize_q4_0;
    OcCpuKernelLevel level;
} OcCpuKernels;

OcError oc_cpu_kernels_detect(OcCpuKernelCaps *caps);
OcError oc_cpu_kernels_init(OcCpuKernels *kernels, OcCpuKernelLevel level);
OcError oc_cpu_kernels_init_best(OcCpuKernels *kernels);
OcError oc_cpu_kernels_init_scalar(OcCpuKernels *kernels);
OcCpuKernelLevel oc_cpu_kernels_best_level(void);
const char *oc_cpu_kernel_level_name(OcCpuKernelLevel level);
bool oc_cpu_kernels_has_vnni(void);
bool oc_cpu_kernels_is_skylake_sp(void);

/* Scalar reference implementations (always available). */
float oc_cpu_dot_f32_scalar(const float *a, const float *b, size_t n);
void oc_cpu_matvec_f32_scalar(const float *w, const float *x, float *out,
                               size_t n_rows, size_t n_cols);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CPU_KERNELS_H */
