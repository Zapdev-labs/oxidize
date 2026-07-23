/* test_detect.c — Hardware detection tests. */
#include <criterion/criterion.h>
#include "oxidize/detect.h"
#include <string.h>

Test(det, cpu)
{
    OcDetectInfo info;
    cr_assert_eq(oc_detect_cpu(&info), OC_OK);
    cr_assert(strlen(info.model_name) > 0);
}

Test(det, cpu_null)
{
    cr_assert_neq(oc_detect_cpu(NULL), OC_OK);
}

Test(det, all)
{
    OcDetectInfo info;
    cr_assert_eq(oc_detect_all(&info), OC_OK);
    /* Should have some defaults. */
    cr_assert(info.n_logical > 0 || info.n_cores > 0);
}

Test(det, all_null)
{
    cr_assert_neq(oc_detect_all(NULL), OC_OK);
}

Test(det, simd_level)
{
    OcDetectInfo info;
    oc_detect_cpu(&info);
    const char *level = oc_detect_simd_level(&info);
    cr_assert_not_null(level);
    cr_assert(strlen(level) > 0);
}

Test(det, simd_level_null)
{
    cr_assert_str_eq(oc_detect_simd_level(NULL), "none");
}

Test(det, supports_vnni)
{
    OcDetectInfo info;
    oc_detect_cpu(&info);
    /* Just check it doesn't crash. */
    (void)oc_detect_supports_vnni(&info);
}

Test(det, supports_vnni_null)
{
    cr_assert(!oc_detect_supports_vnni(NULL));
}

Test(det, is_server)
{
    OcDetectInfo info;
    oc_detect_all(&info);
    (void)oc_detect_is_server(&info);
}

Test(det, is_server_null)
{
    cr_assert(!oc_detect_is_server(NULL));
}

Test(det, recommended_threads)
{
    OcDetectInfo info;
    oc_detect_all(&info);
    uint32_t threads = oc_detect_recommended_threads(&info);
    cr_assert(threads > 0);
}

Test(det, recommended_threads_null)
{
    cr_assert_eq(oc_detect_recommended_threads(NULL), 4);
}

Test(det, print)
{
    OcDetectInfo info;
    oc_detect_all(&info);
    char out[4096];
    oc_detect_print(&info, out, sizeof(out));
    cr_assert(strstr(out, "Hardware Detection") != NULL);
    cr_assert(strstr(out, "CPU:") != NULL);
}

Test(det, print_null)
{
    oc_detect_print(NULL, NULL, 0);
    /* should not crash */
}

Test(det, numa)
{
    OcDetectInfo info;
    oc_detect_all(&info);
    cr_assert(info.n_numa_nodes > 0);
}

Test(det, has_avx2_or_neon)
{
    OcDetectInfo info;
    oc_detect_cpu(&info);
    /* At least one SIMD should be available. */
    cr_assert(info.has_sse42 || info.has_neon || info.has_avx || info.has_avx2);
}
