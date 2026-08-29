/* test_cpu_kernels.c — CPU kernel dispatch tests. */
#include "framework.h"
#include "oxidize/cpu_kernels.h"
#include <string.h>

Test(ck, detect)
{
    OcCpuKernelCaps caps;
    cr_assert_eq(oc_cpu_kernels_detect(&caps), OC_OK);
    cr_assert(caps.cpu_name != NULL);
    /* Level should be at least scalar. */
    cr_assert_geq(caps.level, OC_CPU_KERNEL_SCALAR);
    free((void *)caps.cpu_name);
}

OC_TEST_NULL_SAFE(ck, detect_null,
        cr_assert_neq(oc_cpu_kernels_detect(NULL), OC_OK);)

Test(ck, best_level)
{
    OcCpuKernelLevel level = oc_cpu_kernels_best_level();
    cr_assert_geq(level, OC_CPU_KERNEL_SCALAR);
}

Test(ck, level_name)
{
    cr_assert_str_eq(oc_cpu_kernel_level_name(OC_CPU_KERNEL_SCALAR), "scalar");
    cr_assert_str_eq(oc_cpu_kernel_level_name(OC_CPU_KERNEL_AVX2), "avx2");
    cr_assert_str_eq(oc_cpu_kernel_level_name(OC_CPU_KERNEL_AVX512), "avx512");
    cr_assert_str_eq(oc_cpu_kernel_level_name(OC_CPU_KERNEL_AVX512_VNNI), "avx512_vnni");
    cr_assert_str_eq(oc_cpu_kernel_level_name(OC_CPU_KERNEL_NEON), "neon");
}

Test(ck, init_scalar)
{
    OcCpuKernels k;
    cr_assert_eq(oc_cpu_kernels_init_scalar(&k), OC_OK);
    cr_assert_eq(k.level, OC_CPU_KERNEL_SCALAR);
    cr_assert_not_null(k.dot_f32);
    cr_assert_not_null(k.matvec_f32);
}

OC_TEST_NULL_SAFE(ck, init_null,
        cr_assert_neq(oc_cpu_kernels_init_scalar(NULL), OC_OK);)

Test(ck, init_best)
{
    OcCpuKernels k;
    cr_assert_eq(oc_cpu_kernels_init_best(&k), OC_OK);
    cr_assert_not_null(k.dot_f32);
}

Test(ck, init_level)
{
    OcCpuKernels k;
    cr_assert_eq(oc_cpu_kernels_init(&k, OC_CPU_KERNEL_AVX512), OC_OK);
    cr_assert_not_null(k.dot_f32);
}

Test(ck, dot_scalar)
{
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result = oc_cpu_dot_f32_scalar(a, b, 4);
    cr_assert_float_eq(result, 70.0f, 0.001f); /* 5+12+21+32 */
}

OC_TEST_NULL_SAFE(ck, dot_scalar_null,
        cr_assert_float_eq(oc_cpu_dot_f32_scalar(NULL, NULL, 0), 0.0f, 0.001f);)

Test(ck, dot_scalar_empty)
{
    float a[] = {1.0f};
    cr_assert_float_eq(oc_cpu_dot_f32_scalar(a, a, 0), 0.0f, 0.001f);
}

Test(ck, matvec_scalar)
{
    float w[] = {1, 2, 3, 4, 5, 6}; /* 2x3 matrix */
    float x[] = {1.0f, 1.0f, 1.0f};
    float out[2];
    oc_cpu_matvec_f32_scalar(w, x, out, 2, 3);
    cr_assert_float_eq(out[0], 6.0f, 0.001f); /* 1+2+3 */
    cr_assert_float_eq(out[1], 15.0f, 0.001f); /* 4+5+6 */
}

OC_TEST_NULL_SAFE(ck, matvec_scalar_null,
        oc_cpu_matvec_f32_scalar(NULL, NULL, NULL, 0, 0);)

Test(ck, has_vnni)
{
    /* Just check it doesn't crash. */
    (void)oc_cpu_kernels_has_vnni();
}

Test(ck, is_skylake_sp)
{
    (void)oc_cpu_kernels_is_skylake_sp();
}

Test(ck, dot_via_function_ptr)
{
    OcCpuKernels k;
    oc_cpu_kernels_init_scalar(&k);
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float result = k.dot_f32(a, b, 3);
    cr_assert_float_eq(result, 32.0f, 0.001f); /* 4+10+18 */
}

Test(ck, matvec_via_function_ptr)
{
    OcCpuKernels k;
    oc_cpu_kernels_init_scalar(&k);
    float w[] = {1, 0, 0, 1}; /* identity 2x2 */
    float x[] = {3.0f, 4.0f};
    float out[2];
    k.matvec_f32(w, x, out, 2, 2);
    cr_assert_float_eq(out[0], 3.0f, 0.001f);
    cr_assert_float_eq(out[1], 4.0f, 0.001f);
}

/* SIMD dispatch parity: whatever level this CPU dispatches to must agree
 * with the scalar reference, including the non-multiple-of-16 tail. */
Test(ck, dispatch_matches_scalar)
{
    enum { N = 37, ROWS = 5 };
    float a[N], b[N], w[ROWS * N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)(i % 7) - 3.0f;
        b[i] = (float)(i % 5) * 0.5f;
    }
    for (int i = 0; i < ROWS * N; i++) w[i] = (float)(i % 11) * 0.25f - 1.0f;

    OcCpuKernels k;
    cr_assert_eq(oc_cpu_kernels_init_best(&k), OC_OK);

    cr_assert_float_eq(k.dot_f32(a, b, N),
                       oc_cpu_dot_f32_scalar(a, b, N), 1e-4f);

    float got[ROWS], want[ROWS];
    k.matvec_f32(w, a, got, ROWS, N);
    oc_cpu_matvec_f32_scalar(w, a, want, ROWS, N);
    for (int r = 0; r < ROWS; r++)
        cr_assert_float_eq(got[r], want[r], 1e-4f);
}

/* A requested level above what the CPU supports must be clamped, not
 * dispatched to an illegal instruction. */
Test(ck, init_clamps_unsupported_level)
{
    OcCpuKernels k;
    cr_assert_eq(oc_cpu_kernels_init(&k, OC_CPU_KERNEL_AVX512_VNNI), OC_OK);
    cr_assert_leq(k.level, oc_cpu_kernels_best_level());
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cr_assert_float_eq(k.dot_f32(a, a, 4), 30.0f, 1e-4f);
}
