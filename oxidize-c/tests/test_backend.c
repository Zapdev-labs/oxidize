/* test_backend.c — Backend interface tests. */
#include <criterion/criterion.h>
#include "oxidize/backend.h"
#include <string.h>

Test(backend, type_name_all)
{
    cr_assert_str_eq(oc_backend_type_name(OC_BACKEND_CPU), "cpu");
    cr_assert_str_eq(oc_backend_type_name(OC_BACKEND_CUDA), "cuda");
    cr_assert_str_eq(oc_backend_type_name(OC_BACKEND_VULKAN), "vulkan");
    cr_assert_str_eq(oc_backend_type_name(OC_BACKEND_METAL), "metal");
    cr_assert_str_eq(oc_backend_type_name(OC_BACKEND_WEBGPU), "webgpu");
}

Test(backend, type_name_unknown)
{
    cr_assert_str_eq(oc_backend_type_name((OcBackendType)999), "unknown");
}

Test(backend, type_name_count)
{
    cr_assert_str_eq(oc_backend_type_name(OC_BACKEND__COUNT), "unknown");
}

Test(backend, detect_cpu)
{
    OcBackendInfo info;
    cr_assert_eq(oc_backend_detect(OC_BACKEND_CPU, &info), OC_OK);
    cr_assert(info.available);
    cr_assert_eq(info.device_count, 1);
    cr_assert_str_eq(info.name, "cpu");
}

Test(backend, detect_null)
{
    cr_assert_neq(oc_backend_detect(OC_BACKEND_CPU, NULL), OC_OK);
}

Test(backend, detect_invalid_type)
{
    OcBackendInfo info;
    cr_assert_neq(oc_backend_detect((OcBackendType)999, &info), OC_OK);
}

Test(backend, detect_cuda)
{
    OcBackendInfo info;
    cr_assert_eq(oc_backend_detect(OC_BACKEND_CUDA, &info), OC_OK);
    /* In the dependency-free C11 port (no OC_CUDA), CUDA is unavailable. */
    cr_assert_str_eq(info.name, "cuda");
}

Test(backend, detect_all)
{
    OcBackendInfo infos[OC_BACKEND__COUNT];
    uint32_t n = 0;
    cr_assert_eq(oc_backend_detect_all(infos, &n, OC_BACKEND__COUNT), OC_OK);
    cr_assert_eq(n, (uint32_t)OC_BACKEND__COUNT);
    cr_assert_str_eq(infos[0].name, "cpu");
}

Test(backend, detect_all_limited)
{
    OcBackendInfo infos[2];
    uint32_t n = 0;
    cr_assert_eq(oc_backend_detect_all(infos, &n, 2), OC_OK);
    cr_assert_eq(n, 2u);
}

Test(backend, detect_all_null)
{
    uint32_t n;
    cr_assert_neq(oc_backend_detect_all(NULL, &n, 1), OC_OK);
    cr_assert_neq(oc_backend_detect_all((OcBackendInfo[1]){0}, NULL, 1), OC_OK);
}

Test(backend, init_cpu)
{
    OcBackend b;
    cr_assert_eq(oc_backend_init(&b, OC_BACKEND_CPU), OC_OK);
    cr_assert(b.initialized);
    cr_assert_eq(b.type, OC_BACKEND_CPU);
    cr_assert(b.info.available);
    oc_backend_free(&b);
}

Test(backend, init_null)
{
    cr_assert_neq(oc_backend_init(NULL, OC_BACKEND_CPU), OC_OK);
}

Test(backend, init_invalid_type)
{
    OcBackend b;
    cr_assert_neq(oc_backend_init(&b, (OcBackendType)999), OC_OK);
}

Test(backend, free_null_safe)
{
    oc_backend_free(NULL);
}

Test(backend, free_after_init)
{
    OcBackend b;
    oc_backend_init(&b, OC_BACKEND_CPU);
    oc_backend_free(&b);
    cr_assert(!b.initialized);
}

Test(backend, is_available_cpu)
{
    cr_assert(oc_backend_is_available(OC_BACKEND_CPU));
}

Test(backend, is_available_invalid)
{
    cr_assert(!oc_backend_is_available((OcBackendType)999));
}

Test(backend, best_available)
{
    /* CPU is always available; best_available returns at least CPU. */
    OcBackendType best = oc_backend_best_available();
    cr_assert(best < OC_BACKEND__COUNT);
    cr_assert(oc_backend_is_available(best));
}

Test(backend, info_print)
{
    OcBackendInfo info;
    oc_backend_detect(OC_BACKEND_CPU, &info);
    char buf[256];
    size_t n = oc_backend_info_print(&info, buf, sizeof(buf));
    cr_assert(n > 0);
    cr_assert(strstr(buf, "cpu") != NULL);
    cr_assert(strstr(buf, "available=yes") != NULL);
}

Test(backend, info_print_null)
{
    char buf[64];
    size_t n = oc_backend_info_print(NULL, buf, sizeof(buf));
    cr_assert_eq(n, 0);
    cr_assert_eq(buf[0], '\0');
}

Test(backend, info_print_small_buf)
{
    OcBackendInfo info;
    oc_backend_detect(OC_BACKEND_CPU, &info);
    char buf[4];
    size_t n = oc_backend_info_print(&info, buf, sizeof(buf));
    cr_assert(n > 0);
    /* Output is truncated but NUL-terminated. */
    cr_assert_eq(buf[3], '\0');
}

Test(backend, info_print_zero_size)
{
    OcBackendInfo info;
    oc_backend_detect(OC_BACKEND_CPU, &info);
    size_t n = oc_backend_info_print(&info, NULL, 0);
    cr_assert(n > 0);
}
