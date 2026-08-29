/* test_llama_swap.c — Model swap tests. */
#include "framework.h"
#include "oxidize/llama_swap.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Helper: create a temp file with some content. */
static const char *make_temp_file(size_t size)
{
    static char path[256];
    snprintf(path, sizeof(path), "/tmp/oxidize_swap_test_%d.gguf", (int)getpid());
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;
    /* Write 'size' bytes of pattern data. */
    for (size_t i = 0; i < size; i++) {
        char c = (char)(i & 0xFF);
        fwrite(&c, 1, 1, f);
    }
    fclose(f);
    return path;
}

static void cleanup_temp_file(const char *path)
{
    if (path) unlink(path);
}

Test(swap, init_free)
{
    OcModelSwap sw;
    cr_assert_eq(oc_model_swap_init(&sw), OC_OK);
    cr_assert_eq(sw.active_idx, -1);
    cr_assert_eq(sw.n_models, 0);
    oc_model_swap_free(&sw);
}

Test(swap, init_null)
{
    cr_assert_neq(oc_model_swap_init(NULL), OC_OK);
}

Test(swap, register)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    cr_assert_eq(oc_model_swap_register(&sw, "/path/model.gguf", "q4km", &idx), OC_OK);
    cr_assert_eq(idx, 0);
    cr_assert_eq(sw.n_models, 1);
    cr_assert_str_eq(sw.models[0].path, "/path/model.gguf");
    cr_assert_str_eq(sw.models[0].name, "q4km");
    oc_model_swap_free(&sw);
}

Test(swap, register_multiple)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx1, idx2;
    oc_model_swap_register(&sw, "/path/m1.gguf", "model1", &idx1);
    oc_model_swap_register(&sw, "/path/m2.gguf", "model2", &idx2);
    cr_assert_eq(idx1, 0);
    cr_assert_eq(idx2, 1);
    cr_assert_eq(sw.n_models, 2);
    oc_model_swap_free(&sw);
}

Test(swap, register_null)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    cr_assert_neq(oc_model_swap_register(NULL, "p", "n", NULL), OC_OK);
    cr_assert_neq(oc_model_swap_register(&sw, NULL, "n", NULL), OC_OK);
    oc_model_swap_free(&sw);
}

Test(swap, load)
{
    const char *path = make_temp_file(4096);
    cr_assert_not_null(path);
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, path, "test", &idx);
    cr_assert_eq(oc_model_swap_load(&sw, idx), OC_OK);
    cr_assert(sw.models[0].loaded);
    cr_assert(sw.total_loaded_bytes > 0);
    cr_assert_eq(sw.models[0].model_data_size, 4096);
    /* Verify data is readable (mmap). */
    const uint8_t *data = (const uint8_t *)sw.models[0].model_data;
    cr_assert_eq(data[0], 0);  /* first byte = 0 & 0xFF */
    cr_assert_eq(data[255], 255);
    oc_model_swap_free(&sw);
    cleanup_temp_file(path);
}

Test(swap, unload)
{
    const char *path = make_temp_file(1024);
    cr_assert_not_null(path);
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, path, "test", &idx);
    oc_model_swap_load(&sw, idx);
    cr_assert_eq(oc_model_swap_unload(&sw, idx), OC_OK);
    cr_assert(!sw.models[0].loaded);
    oc_model_swap_free(&sw);
    cleanup_temp_file(path);
}

Test(swap, activate)
{
    const char *path = make_temp_file(2048);
    cr_assert_not_null(path);
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, path, "test", &idx);
    cr_assert_eq(oc_model_swap_activate(&sw, idx), OC_OK);
    cr_assert_eq(sw.active_idx, idx);
    cr_assert(sw.models[idx].loaded);
    oc_model_swap_free(&sw);
    cleanup_temp_file(path);
}

Test(swap, active)
{
    const char *path = make_temp_file(512);
    cr_assert_not_null(path);
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    cr_assert_eq(oc_model_swap_active(&sw), -1);
    int idx;
    oc_model_swap_register(&sw, path, "t", &idx);
    oc_model_swap_activate(&sw, idx);
    cr_assert_eq(oc_model_swap_active(&sw), idx);
    oc_model_swap_free(&sw);
    cleanup_temp_file(path);
}

Test(swap, info)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/path/m.gguf", "name", &idx);
    const OcSwapModelEntry *info;
    cr_assert_eq(oc_model_swap_info(&sw, idx, &info), OC_OK);
    cr_assert_str_eq(info->path, "/path/m.gguf");
    cr_assert_str_eq(info->name, "name");
    oc_model_swap_free(&sw);
}

Test(swap, info_null)
{
    cr_assert_neq(oc_model_swap_info(NULL, 0, NULL), OC_OK);
}

Test(swap, info_out_of_range)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    const OcSwapModelEntry *info;
    cr_assert_neq(oc_model_swap_info(&sw, 99, &info), OC_OK);
    oc_model_swap_free(&sw);
}

Test(swap, list)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    oc_model_swap_register(&sw, "/m1.gguf", "a", NULL);
    oc_model_swap_register(&sw, "/m2.gguf", "b", NULL);
    const OcSwapModelEntry *arr;
    size_t n = oc_model_swap_list(&sw, &arr);
    cr_assert_eq(n, 2);
    cr_assert_str_eq(arr[0].name, "a");
    cr_assert_str_eq(arr[1].name, "b");
    oc_model_swap_free(&sw);
}

Test(swap, loaded_bytes)
{
    const char *path = make_temp_file(4096);
    cr_assert_not_null(path);
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, path, "t", &idx);
    cr_assert_eq(oc_model_swap_loaded_bytes(&sw), 0);
    oc_model_swap_load(&sw, idx);
    cr_assert_eq(oc_model_swap_loaded_bytes(&sw), 4096);
    oc_model_swap_unload(&sw, idx);
    cr_assert_eq(oc_model_swap_loaded_bytes(&sw), 0);
    oc_model_swap_free(&sw);
    cleanup_temp_file(path);
}

Test(swap, double_load)
{
    const char *path = make_temp_file(256);
    cr_assert_not_null(path);
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, path, "t", &idx);
    oc_model_swap_load(&sw, idx);
    cr_assert_eq(oc_model_swap_load(&sw, idx), OC_OK); /* idempotent */
    oc_model_swap_free(&sw);
    cleanup_temp_file(path);
}

Test(swap, double_unload)
{
    const char *path = make_temp_file(256);
    cr_assert_not_null(path);
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, path, "t", &idx);
    oc_model_swap_load(&sw, idx);
    oc_model_swap_unload(&sw, idx);
    cr_assert_eq(oc_model_swap_unload(&sw, idx), OC_OK); /* idempotent */
    oc_model_swap_free(&sw);
    cleanup_temp_file(path);
}

Test(swap, load_nonexistent_file)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/nonexistent/path/model.gguf", "t", &idx);
    cr_assert_neq(oc_model_swap_load(&sw, idx), OC_OK);
    cr_assert(!sw.models[0].loaded);
    oc_model_swap_free(&sw);
}
