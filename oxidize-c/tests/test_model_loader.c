/* test_model_loader.c — Model loader tests. */
#include "framework.h"
#include "oxidize/gguf_writer.h"
#include "oxidize/model_loader.h"
#include "oxidize/model.h"
#include <string.h>
#include <unistd.h>

#define MODEL_LOADER_FIXTURE "/tmp/oxidize-c-model-loader-view.gguf"

Test(mloader, init_free)
{
    OcModelLoader loader;
    cr_assert_eq(oc_model_loader_init(&loader, "/tmp/test.gguf"), OC_OK);
    cr_assert_str_eq(loader.path, "/tmp/test.gguf");
    cr_assert(!loader.loaded);
    oc_model_loader_free(&loader);
}

OC_TEST_NULL_SAFE(mloader, init_null,
        cr_assert_neq(oc_model_loader_init(NULL, "/tmp/t.gguf"), OC_OK);
        cr_assert_neq(oc_model_loader_init((OcModelLoader[]){0}, NULL), OC_OK);)

Test(mloader, arch_name)
{
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_LLAMA), "llama");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_GPT2), "gpt2");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_FALCON), "falcon");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_QWEN), "qwen");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_UNKNOWN), "unknown");
}

Test(mloader, arch_parse)
{
    cr_assert_eq(oc_model_arch_from_str("llama"), OC_ARCH_LLAMA);
    cr_assert_eq(oc_model_arch_from_str("gpt2"), OC_ARCH_GPT2);
    cr_assert_eq(oc_model_arch_from_str("falcon"), OC_ARCH_FALCON);
    cr_assert_eq(oc_model_arch_from_str("unknown"), OC_ARCH_UNKNOWN);
}

Test(mloader, load_nonexistent)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/nonexistent/path/model.gguf");
    cr_assert_neq(oc_model_loader_load(&loader), OC_OK);
    oc_model_loader_free(&loader);
}

Test(mloader, load_empty_path)
{
    OcModelLoader loader;
    memset(&loader, 0, sizeof(loader));
    cr_assert_neq(oc_model_loader_load(&loader), OC_OK);
}

Test(mloader, get_tensor_empty)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/tmp/test.gguf");
    const OcLoadedTensor *t;
    cr_assert_neq(oc_model_loader_get_tensor(&loader, "test", &t), OC_OK);
    oc_model_loader_free(&loader);
}

OC_TEST_NULL_SAFE(mloader, get_tensor_null,
        cr_assert_neq(oc_model_loader_get_tensor(NULL, "x", NULL), OC_OK);)

Test(mloader, list_tensors_empty)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/tmp/test.gguf");
    const OcLoadedTensor *arr;
    size_t count;
    cr_assert_eq(oc_model_loader_list_tensors(&loader, &arr, &count), OC_OK);
    cr_assert_eq(count, 0);
    oc_model_loader_free(&loader);
}

OC_TEST_NULL_SAFE(mloader, list_tensors_null,
        cr_assert_neq(oc_model_loader_list_tensors(NULL, NULL, NULL), OC_OK);)

Test(mloader, param_count_empty)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/tmp/test.gguf");
    cr_assert_eq(oc_model_loader_param_count(&loader), 0);
    oc_model_loader_free(&loader);
}

OC_TEST_NULL_SAFE(mloader, param_count_null,
        cr_assert_eq(oc_model_loader_param_count(NULL), 0);)

Test(mloader, double_load)
{
    OcModelLoader loader;
    oc_model_loader_init(&loader, "/nonexistent.gguf");
    /* First load fails, second should also fail gracefully. */
    oc_model_loader_load(&loader);
    cr_assert_neq(oc_model_loader_load(&loader), OC_OK);
    oc_model_loader_free(&loader);
}

Test(mloader, arch_name_all)
{
    /* Test a subset of architectures. */
    const OcModelArchitecture archs[] = {OC_ARCH_LLAMA, OC_ARCH_GPT2, OC_ARCH_FALCON, OC_ARCH_QWEN};
    for (size_t i = 0; i < sizeof(archs)/sizeof(archs[0]); i++) {
        const char *name = oc_model_arch_name(archs[i]);
        cr_assert(name != NULL);
        cr_assert(strlen(name) > 0);
    }
}

Test(mloader, arch_roundtrip)
{
    const char *names[] = {"llama", "gpt2", "falcon", "qwen"};
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        OcModelArchitecture arch = oc_model_arch_from_str(names[i]);
        cr_assert_str_eq(oc_model_arch_name(arch), names[i], "roundtrip failed for %s", names[i]);
    }
}

Test(mloader, reports_exact_tensor_byte_sizes)
{
    unlink(MODEL_LOADER_FIXTURE);
    OcGgufWriter writer;
    cr_assert_eq(oc_gguf_writer_init(MODEL_LOADER_FIXTURE, "llama",
                                     &writer), OC_OK);
    const uint64_t first_dims[] = {4};
    const float first[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const uint64_t second_dims[] = {2};
    const float second[] = {5.0f, 6.0f};
    cr_assert_eq(oc_gguf_writer_add_tensor(&writer, "first", 1, first_dims,
                                           0, first, sizeof(first)), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_tensor(&writer, "second", 1, second_dims,
                                           0, second, sizeof(second)), OC_OK);
    cr_assert_eq(oc_gguf_writer_finalize(&writer), OC_OK);
    oc_gguf_writer_free(&writer);

    OcModelLoader loader;
    cr_assert_eq(oc_model_loader_init(&loader, MODEL_LOADER_FIXTURE), OC_OK);
    cr_assert_eq(oc_model_loader_load(&loader), OC_OK);
    cr_assert_eq(unlink(MODEL_LOADER_FIXTURE), 0);
    const OcLoadedTensor *first_info = NULL;
    const OcLoadedTensor *second_info = NULL;
    cr_assert_eq(oc_model_loader_get_tensor(&loader, "first", &first_info),
                 OC_OK);
    cr_assert_eq(oc_model_loader_get_tensor(&loader, "second", &second_info),
                 OC_OK);
    cr_assert_eq(first_info->size, sizeof(first),
                 "first tensor size must exclude alignment padding");
    cr_assert_eq(second_info->size, sizeof(second),
                 "last tensor size must exclude trailing file bytes");

    const uint8_t *first_data = NULL;
    const uint8_t *second_data = NULL;
    size_t first_size = 0;
    size_t second_size = 0;
    cr_assert_eq(oc_model_loader_get_tensor_data(&loader, "first",
                                                  &first_data, &first_size),
                 OC_OK);
    cr_assert_eq(oc_model_loader_get_tensor_data(&loader, "second",
                                                  &second_data, &second_size),
                 OC_OK);
    cr_assert_eq(first_size, sizeof(first));
    cr_assert_eq(second_size, sizeof(second));
    cr_assert_arr_eq(first_data, first, sizeof(first),
                     "first mapped tensor bytes changed after unlink");
    cr_assert_arr_eq(second_data, second, sizeof(second),
                     "second mapped tensor bytes changed after unlink");

    oc_model_loader_free(&loader);
}
