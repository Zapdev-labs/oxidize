/* test_hf_hub.c — HuggingFace Hub downloader tests. */
#include <criterion/criterion.h>
#include "oxidize/hf_hub.h"
#include <string.h>

Test(hf, is_gguf)
{
    cr_assert(oc_hf_is_gguf("model.gguf"));
    cr_assert(oc_hf_is_gguf("model.GGUF"));
    cr_assert(!oc_hf_is_gguf("model.bin"));
    cr_assert(!oc_hf_is_gguf("model.safetensors"));
    cr_assert(!oc_hf_is_gguf(""));
    cr_assert(!oc_hf_is_gguf(NULL));
}

Test(hf, parse_quant_type)
{
    char qt[32];
    cr_assert(oc_hf_parse_quant_type("model-Q4_K_M.gguf", qt, sizeof(qt)));
    cr_assert_str_eq(qt, "Q4_K_M");

    cr_assert(oc_hf_parse_quant_type("qwen2-7b-instruct-q8_0.gguf", qt, sizeof(qt)));
    cr_assert_str_eq(qt, "Q8_0");

    cr_assert(!oc_hf_parse_quant_type("model.bin", qt, sizeof(qt)));
    cr_assert_eq(qt[0], '\0');
}

Test(hf, parse_quant_type_null)
{
    char qt[32];
    cr_assert(!oc_hf_parse_quant_type(NULL, qt, sizeof(qt)));
    cr_assert(!oc_hf_parse_quant_type("model.gguf", NULL, 0));
}

Test(hf, sanitize_repo_id)
{
    char out[256];
    cr_assert_eq(oc_hf_sanitize_repo_id("Qwen/Qwen2-7B", out, sizeof(out)), OC_OK);
    cr_assert_str_eq(out, "Qwen_Qwen2-7B");

    cr_assert_eq(oc_hf_sanitize_repo_id("a/b/c", out, sizeof(out)), OC_OK);
    cr_assert_str_eq(out, "a_b_c");
}

Test(hf, sanitize_repo_id_null)
{
    char out[32];
    cr_assert_neq(oc_hf_sanitize_repo_id(NULL, out, sizeof(out)), OC_OK);
    cr_assert_neq(oc_hf_sanitize_repo_id("test", NULL, 0), OC_OK);
}

Test(hf, default_cache_dir)
{
    char dir[512];
    cr_assert_eq(oc_hf_default_cache_dir(dir, sizeof(dir)), OC_OK);
    cr_assert(strstr(dir, ".cache/oxidize/hf") != NULL);
}

Test(hf, config_init)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, NULL), OC_OK);
    cr_assert(cfg.cache_dir[0] != '\0');
    cr_assert_str_eq(cfg.revision, "main");
    cr_assert(cfg.api_base[0] != '\0');
}

Test(hf, config_init_custom_cache)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, "/tmp/test_cache"), OC_OK);
    cr_assert_str_eq(cfg.cache_dir, "/tmp/test_cache");
}

Test(hf, config_init_null)
{
    cr_assert_neq(oc_hf_config_init(NULL, NULL), OC_OK);
}

Test(hf, cache_path)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, "/tmp/test_cache");
    char path[512];
    cr_assert_eq(oc_hf_cache_path(&cfg, "Qwen/Qwen2-7B", "model.gguf", path, sizeof(path)), OC_OK);
    cr_assert_str_eq(path, "/tmp/test_cache/Qwen_Qwen2-7B/model.gguf");
}

Test(hf, cache_path_null)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, "/tmp/test_cache");
    char path[64];
    cr_assert_neq(oc_hf_cache_path(NULL, "test", "f", path, sizeof(path)), OC_OK);
    cr_assert_neq(oc_hf_cache_path(&cfg, NULL, "f", path, sizeof(path)), OC_OK);
    cr_assert_neq(oc_hf_cache_path(&cfg, "test", NULL, path, sizeof(path)), OC_OK);
    cr_assert_neq(oc_hf_cache_path(&cfg, "test", "f", NULL, 0), OC_OK);
}

Test(hf, cache_size)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, "/nonexistent_cache_dir_xyz");
    uint64_t size;
    cr_assert_eq(oc_hf_cache_size(&cfg, &size), OC_OK);
    cr_assert_eq(size, 0);
}

Test(hf, cache_list_empty)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, "/nonexistent_cache_dir_xyz");
    OcHfModel models[10];
    size_t count = 10;
    cr_assert_eq(oc_hf_cache_list(&cfg, models, &count), OC_OK);
    cr_assert_eq(count, 0);
}

Test(hf, cache_clean)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, "/nonexistent_cache_dir_xyz");
    size_t removed;
    cr_assert_eq(oc_hf_cache_clean(&cfg, 0, &removed), OC_OK);
    cr_assert_eq(removed, 0);
}

Test(hf, list_models_null)
{
    cr_assert_neq(oc_hf_list_models(NULL, NULL, NULL), OC_OK);
}

Test(hf, resolve_null)
{
    cr_assert_neq(oc_hf_resolve(NULL, NULL), OC_OK);
}

Test(hf, download_null)
{
    cr_assert_neq(oc_hf_download(NULL, NULL, NULL, NULL), OC_OK);
}
