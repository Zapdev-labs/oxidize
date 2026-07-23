/* test_loader.c — Universal loader tests. */
#include <criterion/criterion.h>
#include "oxidize/loader.h"
#include <string.h>

Test(loader, init)
{
    cr_assert_eq(oc_loader_init(), OC_OK);
}

Test(loader, load_null_path)
{
    OcLoaderResult r;
    cr_assert_neq(oc_loader_load(NULL, &r), OC_OK);
}

Test(loader, load_null_result)
{
    cr_assert_neq(oc_loader_load("/tmp/x.gguf", NULL), OC_OK);
}

Test(loader, load_nonexistent)
{
    OcLoaderResult r;
    OcError e = oc_loader_load("/nonexistent/path/model.gguf", &r);
    cr_assert_neq(e, OC_OK);
    cr_assert(!r.loaded);
    cr_assert(strlen(r.error_msg) > 0);
    oc_loader_unload(&r);
}

Test(loader, detect_arch_null_path)
{
    char arch[64];
    cr_assert_neq(oc_loader_detect_arch(NULL, arch, sizeof(arch)), OC_OK);
}

Test(loader, detect_arch_null_out)
{
    cr_assert_neq(oc_loader_detect_arch("/tmp/x.gguf", NULL, 64), OC_OK);
}

Test(loader, detect_arch_nonexistent)
{
    char arch[64];
    OcError e = oc_loader_detect_arch("/nonexistent/model.gguf", arch, sizeof(arch));
    cr_assert_neq(e, OC_OK);
}

Test(loader, detect_arch_zero_size)
{
    char arch[64];
    cr_assert_neq(oc_loader_detect_arch("/tmp/x.gguf", arch, 0), OC_OK);
}

Test(loader, supported_archs)
{
    const char *archs[32];
    uint32_t n = 0;
    cr_assert_eq(oc_loader_supported_archs(archs, &n), OC_OK);
    cr_assert(n > 0);
    /* Every entry is non-NULL and non-empty. */
    for (uint32_t i = 0; i < n; i++) {
        cr_assert_not_null(archs[i]);
        cr_assert(strlen(archs[i]) > 0);
    }
}

Test(loader, supported_archs_count_only)
{
    uint32_t n = 0;
    cr_assert_eq(oc_loader_supported_archs(NULL, &n), OC_OK);
    cr_assert(n > 0);
}

Test(loader, supported_archs_null_n)
{
    const char *archs[32];
    cr_assert_neq(oc_loader_supported_archs(archs, NULL), OC_OK);
}

Test(loader, is_supported)
{
    cr_assert(oc_loader_is_supported("llama"));
    cr_assert(oc_loader_is_supported("qwen"));
    cr_assert(oc_loader_is_supported("gpt2"));
    cr_assert(oc_loader_is_supported("falcon"));
}

Test(loader, is_supported_unsupported)
{
    cr_assert(!oc_loader_is_supported("nonexistent_arch"));
    cr_assert(!oc_loader_is_supported(""));
    cr_assert(!oc_loader_is_supported(NULL));
}

Test(loader, is_supported_case_insensitive)
{
    /* Normalization lowercases. */
    cr_assert(oc_loader_is_supported("LLAMA"));
    cr_assert(oc_loader_is_supported("Qwen"));
}

Test(loader, is_supported_dash_normalize)
{
    /* '-' → '_' normalization. */
    cr_assert(oc_loader_is_supported("glm-moe"));
}

Test(loader, arch_name_by_index)
{
    const char *name0 = oc_loader_arch_name(0);
    cr_assert_not_null(name0);
    cr_assert(strlen(name0) > 0);
}

Test(loader, arch_name_out_of_range)
{
    cr_assert_eq(oc_loader_arch_name(99999), NULL);
}

Test(loader, arch_name_roundtrip)
{
    uint32_t n = 0;
    oc_loader_supported_archs(NULL, &n);
    for (uint32_t i = 0; i < n; i++) {
        const char *name = oc_loader_arch_name(i);
        cr_assert_not_null(name);
        cr_assert(oc_loader_is_supported(name));
    }
}

Test(loader, load_with_arch_null_args)
{
    OcLoaderResult r;
    cr_assert_neq(oc_loader_load_with_arch(NULL, "llama", &r), OC_OK);
    cr_assert_neq(oc_loader_load_with_arch("/tmp/x.gguf", NULL, &r), OC_OK);
    cr_assert_neq(oc_loader_load_with_arch("/tmp/x.gguf", "llama", NULL), OC_OK);
}

Test(loader, load_with_arch_unsupported)
{
    OcLoaderResult r;
    OcError e = oc_loader_load_with_arch("/tmp/x.gguf", "bogus_arch", &r);
    cr_assert_eq(e, OC_ERR_MODEL);
    cr_assert(!r.loaded);
    oc_loader_unload(&r);
}

Test(loader, load_with_arch_nonexistent)
{
    OcLoaderResult r;
    OcError e = oc_loader_load_with_arch("/nonexistent.gguf", "llama", &r);
    cr_assert_neq(e, OC_OK);
    cr_assert(!r.loaded);
    oc_loader_unload(&r);
}

Test(loader, unload_null_safe)
{
    oc_loader_unload(NULL);
}

Test(loader, unload_zeros_result)
{
    OcLoaderResult r;
    memset(&r, 0xFF, sizeof(r));
    oc_loader_unload(&r);
    cr_assert(!r.loaded);
    cr_assert_eq(r.n_tensors, 0);
    cr_assert_eq(r.arch_name[0], '\0');
}
