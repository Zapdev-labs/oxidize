/* test_model_loader.c — Model loader tests. */
#include <criterion/criterion.h>
#include "oxidize/model_loader.h"
#include "oxidize/model.h"
#include <string.h>

Test(loader, init_free)
{
    OcModelLoader loader;
    cr_assert_eq(oc_model_loader_init(&loader, "/tmp/test.gguf"), OC_OK);
    cr_assert_str_eq(loader.path, "/tmp/test.gguf");
    cr_assert(!loader.loaded);
    oc_model_loader_free(&loader);
}

Test(loader, init_null)
{
    cr_assert_neq(oc_model_loader_init(NULL, "/tmp/t.gguf"), OC_OK);
    cr_assert_neq(oc_model_loader_init((OcModelLoader[]){0}, NULL), OC_OK);
}

Test(loader, arch_name)
{
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_LLAMA), "llama");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_GPT2), "gpt2");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_FALCON), "falcon");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_QWEN), "qwen");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_UNKNOWN), "unknown");
}

Test(loader, arch_parse)
{
    cr_assert_eq(oc_model_arch_from_str("llama"), OC_ARCH_LLAMA);
    cr_assert_eq(oc_model_arch_from_str("gpt2"), OC_ARCH_GPT2);
    cr_assert_eq(oc_model_arch_from_str("falcon"), OC_ARCH_FALCON);
    cr_assert_eq(oc_model_arch_from_str("unknown"), OC_ARCH_UNKNOWN);
}

Test(loader, load_nonexistent)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/nonexistent/path/model.gguf");
    cr_assert_neq(oc_model_loader_load(&loader), OC_OK);
    oc_model_loader_free(&loader);
}

Test(loader, load_empty_path)
{
    OcModelLoader loader;
    memset(&loader, 0, sizeof(loader));
    cr_assert_neq(oc_model_loader_load(&loader), OC_OK);
}

Test(loader, get_tensor_empty)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/tmp/test.gguf");
    const OcLoadedTensor *t;
    cr_assert_neq(oc_model_loader_get_tensor(&loader, "test", &t), OC_OK);
    oc_model_loader_free(&loader);
}

Test(loader, get_tensor_null)
{
    cr_assert_neq(oc_model_loader_get_tensor(NULL, "x", NULL), OC_OK);
}

Test(loader, list_tensors_empty)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/tmp/test.gguf");
    const OcLoadedTensor *arr;
    size_t count;
    cr_assert_eq(oc_model_loader_list_tensors(&loader, &arr, &count), OC_OK);
    cr_assert_eq(count, 0);
    oc_model_loader_free(&loader);
}

Test(loader, list_tensors_null)
{
    cr_assert_neq(oc_model_loader_list_tensors(NULL, NULL, NULL), OC_OK);
}

Test(loader, param_count_empty)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/tmp/test.gguf");
    cr_assert_eq(oc_model_loader_param_count(&loader), 0);
    oc_model_loader_free(&loader);
}

Test(loader, param_count_null)
{
    cr_assert_eq(oc_model_loader_param_count(NULL), 0);
}

Test(loader, double_load)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/nonexistent.gguf");
    /* First load fails, second should also fail gracefully. */
    oc_model_loader_load(&loader);
    cr_assert_neq(oc_model_loader_load(&loader), OC_OK);
    oc_model_loader_free(&loader);
}

Test(loader, arch_name_all)
{
    /* Test a subset of architectures. */
    const OcModelArchitecture archs[] = {OC_ARCH_LLAMA, OC_ARCH_GPT2, OC_ARCH_FALCON, OC_ARCH_QWEN};
    for (size_t i = 0; i < sizeof(archs)/sizeof(archs[0]); i++) {
        const char *name = oc_model_arch_name(archs[i]);
        cr_assert(name != NULL);
        cr_assert(strlen(name) > 0);
    }
}

Test(loader, arch_roundtrip)
{
    const char *names[] = {"llama", "gpt2", "falcon", "qwen"};
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        OcModelArchitecture arch = oc_model_arch_from_str(names[i]);
        cr_assert_str_eq(oc_model_arch_name(arch), names[i], "roundtrip failed for %s", names[i]);
    }
}
