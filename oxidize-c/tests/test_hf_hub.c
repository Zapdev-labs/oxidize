/*
 * test_hf_hub.c — HuggingFace Hub downloader tests.
 *
 * Covers the offline-testable surface: config defaults, cache path
 * construction, model filename parsing, null handling, empty-cache
 * listing, and progress struct initialization. Network-dependent
 * functions (oc_hf_list_models, oc_hf_download, oc_hf_resolve against
 * the real Hub) are not exercised here — they require live connectivity
 * and are validated manually.
 */
#define _GNU_SOURCE 1
#include "framework.h"

#include "oxidize/hf_hub.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ─── Config defaults ────────────────────────────────────────────────── */

Test(hf_hub, config_init_defaults)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, NULL), OC_OK);
    /* cache_dir should be non-empty (default ~/.cache/oxidize/hf or /tmp). */
    cr_assert(cfg.cache_dir[0] != '\0', "cache_dir should default");
    cr_assert_str_eq(cfg.revision, "main", "default revision is main");
    cr_assert(cfg.api_token[0] == '\0', "no token by default");
    cr_assert(cfg.repo_id[0] == '\0', "no repo by default");
    cr_assert(cfg.quant_type[0] == '\0', "no quant filter by default");
}

Test(hf_hub, config_init_explicit_cache)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, "/tmp/oxidize-test-hf"), OC_OK);
    cr_assert_str_eq(cfg.cache_dir, "/tmp/oxidize-test-hf");
}

OC_TEST_REJECTS_NULL(hf_hub, config_init_rejects_null, oc_hf_config_init(NULL, NULL))

Test(hf_hub, config_init_preserves_token)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, NULL), OC_OK);
    /* Manually set a token; this is the expected usage pattern. */
    snprintf(cfg.api_token, sizeof(cfg.api_token), "hf_testtoken");
    cr_assert_str_eq(cfg.api_token, "hf_testtoken");
}

/* ─── Cache path construction ────────────────────────────────────────── */

Test(hf_hub, cache_path_basic)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, "/tmp/oxidize-test-hf"), OC_OK);
    char path[1024];
    cr_assert_eq(oc_hf_cache_path(&cfg, "Qwen/Qwen2-7B-Instruct",
                                  "model.q4_k_m.gguf",
                                  path, sizeof(path)), OC_OK);
    cr_assert_str_eq(path,
                     "/tmp/oxidize-test-hf/Qwen_Qwen2-7B-Instruct/model.q4_k_m.gguf",
                     "slashes in repo_id become underscores");
}

Test(hf_hub, cache_path_nested_repo)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, "/tmp/oxidize-test-hf"), OC_OK);
    char path[1024];
    cr_assert_eq(oc_hf_cache_path(&cfg, "org/sub/model",
                                  "weights.gguf", path, sizeof(path)), OC_OK);
    cr_assert_str_eq(path, "/tmp/oxidize-test-hf/org_sub_model/weights.gguf");
}

Test(hf_hub, cache_path_uses_default_when_empty)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, NULL), OC_OK);
    /* Force cache_dir to empty to exercise the fallback path. */
    cfg.cache_dir[0] = '\0';
    char path[2048];
    cr_assert_eq(oc_hf_cache_path(&cfg, "test/repo", "f.gguf",
                                  path, sizeof(path)), OC_OK);
    /* Should contain the sanitized repo somewhere. */
    cr_assert(strstr(path, "test_repo") != NULL, "path contains sanitized repo");
    cr_assert(strstr(path, "f.gguf") != NULL, "path contains filename");
}

Test(hf_hub, cache_path_rejects_null_args)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, "/tmp/oxidize-test-hf"), OC_OK);
    char path[256];
    cr_assert_eq(oc_hf_cache_path(NULL, "a/b", "c.gguf", path, sizeof(path)),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_hf_cache_path(&cfg, NULL, "c.gguf", path, sizeof(path)),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_hf_cache_path(&cfg, "a/b", NULL, path, sizeof(path)),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_hf_cache_path(&cfg, "a/b", "c.gguf", NULL, sizeof(path)),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_hf_cache_path(&cfg, "a/b", "c.gguf", path, 0),
                 OC_ERR_INVALID_ARG);
}

Test(hf_hub, cache_path_truncates_long_input)
{
    OcHfConfig cfg;
    cr_assert_eq(oc_hf_config_init(&cfg, "/tmp/oxidize-test-hf"), OC_OK);
    char path[16];  /* too small */
    cr_assert_eq(oc_hf_cache_path(&cfg, "a/b", "c.gguf", path, sizeof(path)),
                 OC_ERR_INVALID_ARG);
}

/* ─── Model name parsing ────────────────────────────────────────────── */

Test(hf_hub, is_gguf_case_insensitive)
{
    cr_assert(oc_hf_is_gguf("model.gguf"));
    cr_assert(oc_hf_is_gguf("model.GGUF"));
    cr_assert(oc_hf_is_gguf("path/to/model.Gguf"));
    cr_assert(!oc_hf_is_gguf("model.bin"));
    cr_assert(!oc_hf_is_gguf("model.gguf.bak"));
    cr_assert(!oc_hf_is_gguf(""));
    cr_assert(!oc_hf_is_gguf(NULL));
}

Test(hf_hub, parse_quant_q4_k_m)
{
    char qt[OC_HF_MAX_QUANT_TYPE];
    cr_assert(oc_hf_parse_quant_type("model.q4_k_m.gguf", qt, sizeof(qt)));
    cr_assert_str_eq(qt, "Q4_K_M");
}

Test(hf_hub, parse_quant_q8_0)
{
    char qt[OC_HF_MAX_QUANT_TYPE];
    cr_assert(oc_hf_parse_quant_type("model-Q8_0.gguf", qt, sizeof(qt)));
    cr_assert_str_eq(qt, "Q8_0");
}

Test(hf_hub, parse_quant_f16)
{
    char qt[OC_HF_MAX_QUANT_TYPE];
    cr_assert(oc_hf_parse_quant_type("model.F16.gguf", qt, sizeof(qt)));
    cr_assert_str_eq(qt, "F16");
}

Test(hf_hub, parse_quant_iq2_xxs)
{
    char qt[OC_HF_MAX_QUANT_TYPE];
    cr_assert(oc_hf_parse_quant_type("model.IQ2_XXS.gguf", qt, sizeof(qt)));
    cr_assert_str_eq(qt, "IQ2_XXS");
}

Test(hf_hub, parse_quant_with_path)
{
    char qt[OC_HF_MAX_QUANT_TYPE];
    cr_assert(oc_hf_parse_quant_type("subdir/model.Q6_K.gguf", qt, sizeof(qt)));
    cr_assert_str_eq(qt, "Q6_K");
}

Test(hf_hub, parse_quant_none_for_no_quant)
{
    char qt[OC_HF_MAX_QUANT_TYPE];
    /* A filename with no quant tag. */
    cr_assert(!oc_hf_parse_quant_type("tokenizer.gguf", qt, sizeof(qt)));
    cr_assert(!oc_hf_parse_quant_type("model.gguf", qt, sizeof(qt)));
    cr_assert(!oc_hf_parse_quant_type("", qt, sizeof(qt)));
    cr_assert(!oc_hf_parse_quant_type(NULL, qt, sizeof(qt)));
}

OC_TEST_NULL_SAFE(hf_hub, parse_quant_null_out,
        cr_assert(!oc_hf_parse_quant_type("model.Q4_K_M.gguf", NULL, 0));)

/* ─── Repo ID sanitization ───────────────────────────────────────────── */

Test(hf_hub, sanitize_repo_id_replaces_slashes)
{
    char out[256];
    cr_assert_eq(oc_hf_sanitize_repo_id("Qwen/Qwen2-7B-Instruct", out, sizeof(out)),
                 OC_OK);
    cr_assert_str_eq(out, "Qwen_Qwen2-7B-Instruct");
}

Test(hf_hub, sanitize_repo_id_nested)
{
    char out[256];
    cr_assert_eq(oc_hf_sanitize_repo_id("a/b/c", out, sizeof(out)), OC_OK);
    cr_assert_str_eq(out, "a_b_c");
}

OC_TEST_REJECTS_NULL(hf_hub, sanitize_repo_id_rejects_null, oc_hf_sanitize_repo_id(NULL, NULL, 0))

Test(hf_hub, sanitize_repo_id_rejects_too_long)
{
    char out[8];
    cr_assert_eq(oc_hf_sanitize_repo_id("very/long/repo/id", out, sizeof(out)),
                 OC_ERR_INVALID_ARG);
}

Test(hf_hub, sanitize_repo_id_rejects_empty)
{
    char out[16];
    cr_assert_eq(oc_hf_sanitize_repo_id("", out, sizeof(out)),
                 OC_ERR_INVALID_ARG);
}

/* ─── Default cache dir ──────────────────────────────────────────────── */

Test(hf_hub, default_cache_dir_nonempty)
{
    char dir[1024];
    cr_assert_eq(oc_hf_default_cache_dir(dir, sizeof(dir)), OC_OK);
    cr_assert(dir[0] != '\0', "default cache dir is non-empty");
    cr_assert(strlen(dir) > 0, "default cache dir has content");
}

Test(hf_hub, default_cache_dir_rejects_bad_args)
{
    cr_assert_eq(oc_hf_default_cache_dir(NULL, 0), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_hf_default_cache_dir(NULL, 64), OC_ERR_INVALID_ARG);
    char dir[1];
    cr_assert_eq(oc_hf_default_cache_dir(dir, 0), OC_ERR_INVALID_ARG);
}

/* ─── Null handling for list/resolve/download ────────────────────────── */

Test(hf_hub, list_models_rejects_null)
{
    OcHfModel models[1];
    size_t count = 1;
    cr_assert_eq(oc_hf_list_models(NULL, models, &count), OC_ERR_INVALID_ARG);
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, NULL);
    cr_assert_eq(oc_hf_list_models(&cfg, NULL, &count), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_hf_list_models(&cfg, models, NULL), OC_ERR_INVALID_ARG);
}

Test(hf_hub, list_models_rejects_empty_repo)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, NULL);
    cfg.repo_id[0] = '\0';
    OcHfModel models[1];
    size_t count = 1;
    cr_assert_eq(oc_hf_list_models(&cfg, models, &count), OC_ERR_INVALID_ARG);
}

Test(hf_hub, resolve_rejects_null)
{
    cr_assert_eq(oc_hf_resolve(NULL, NULL), OC_ERR_INVALID_ARG);
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, NULL);
    cr_assert_eq(oc_hf_resolve(&cfg, NULL), OC_ERR_INVALID_ARG);
}

Test(hf_hub, resolve_rejects_empty_repo)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, NULL);
    OcHfModel m;
    memset(&m, 0, sizeof(m));
    cr_assert_eq(oc_hf_resolve(&cfg, &m), OC_ERR_INVALID_ARG);
}

Test(hf_hub, download_rejects_null)
{
    cr_assert_eq(oc_hf_download(NULL, NULL, NULL, NULL), OC_ERR_INVALID_ARG);
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, NULL);
    cr_assert_eq(oc_hf_download(&cfg, NULL, NULL, NULL), OC_ERR_INVALID_ARG);
}

Test(hf_hub, download_rejects_empty_filename)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, NULL);
    OcHfModel m;
    memset(&m, 0, sizeof(m));
    cr_assert_eq(oc_hf_download(&cfg, &m, NULL, NULL), OC_ERR_INVALID_ARG);
}

Test(hf_hub, cache_list_rejects_null)
{
    OcHfModel models[1];
    size_t count = 1;
    cr_assert_eq(oc_hf_cache_list(NULL, models, &count), OC_ERR_INVALID_ARG);
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, NULL);
    cr_assert_eq(oc_hf_cache_list(&cfg, NULL, &count), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_hf_cache_list(&cfg, models, NULL), OC_ERR_INVALID_ARG);
}

Test(hf_hub, cache_size_rejects_null)
{
    cr_assert_eq(oc_hf_cache_size(NULL, NULL), OC_ERR_INVALID_ARG);
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, NULL);
    uint64_t sz;
    cr_assert_eq(oc_hf_cache_size(&cfg, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_hf_cache_size(NULL, &sz), OC_ERR_INVALID_ARG);
}

OC_TEST_REJECTS_NULL(hf_hub, cache_clean_rejects_null, oc_hf_cache_clean(NULL, 0, NULL))

/* ─── Cache listing (empty cache) ────────────────────────────────────── */

Test(hf_hub, cache_list_empty_cache_returns_zero)
{
    /* Use a non-existent directory. */
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, "/tmp/oxidize-hf-nonexistent-12345");
    OcHfModel models[8];
    size_t count = 8;
    cr_assert_eq(oc_hf_cache_list(&cfg, models, &count), OC_OK);
    cr_assert_eq(count, 0u, "empty cache returns zero entries");
}

Test(hf_hub, cache_size_empty_cache_returns_zero)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, "/tmp/oxidize-hf-nonexistent-12345");
    uint64_t sz = 999;
    cr_assert_eq(oc_hf_cache_size(&cfg, &sz), OC_OK);
    cr_assert_eq(sz, 0u, "empty cache size is zero");
}

Test(hf_hub, cache_clean_empty_cache_removes_zero)
{
    OcHfConfig cfg;
    oc_hf_config_init(&cfg, "/tmp/oxidize-hf-nonexistent-12345");
    size_t removed = 999;
    cr_assert_eq(oc_hf_cache_clean(&cfg, 0, &removed), OC_OK);
    cr_assert_eq(removed, 0u, "empty cache clean removes nothing");
}

/* ─── Cache listing with a real (temporary) cache ────────────────────── */

Test(hf_hub, cache_list_finds_gguf_files)
{
    /* Build a tiny fake cache. */
    char tmpl[] = "/tmp/oxidize-hf-test-XXXXXX";
    cr_assert(mkdtemp(tmpl) != NULL, "mkdtemp failed");

    char repo_dir[1024];
    snprintf(repo_dir, sizeof(repo_dir), "%s/Org_Model-7B", tmpl);
    cr_assert_eq(mkdir(repo_dir, 0755), 0, "mkdir repo dir");

    char file1[2048], file2[2048], file3[2048];
    snprintf(file1, sizeof(file1), "%s/model.Q4_K_M.gguf", repo_dir);
    snprintf(file2, sizeof(file2), "%s/model.Q8_0.gguf", repo_dir);
    snprintf(file3, sizeof(file3), "%s/tokenizer.json", repo_dir);
    FILE *f1 = fopen(file1, "w"); cr_assert_not_null(f1);
    fputs("dummy", f1); fclose(f1);
    FILE *f2 = fopen(file2, "w"); cr_assert_not_null(f2);
    fputs("dummy2", f2); fclose(f2);
    FILE *f3 = fopen(file3, "w"); cr_assert_not_null(f3);
    fputs("not-gguf", f3); fclose(f3);

    OcHfConfig cfg;
    oc_hf_config_init(&cfg, tmpl);
    OcHfModel models[8];
    size_t count = 8;
    cr_assert_eq(oc_hf_cache_list(&cfg, models, &count), OC_OK);
    cr_assert_eq(count, 2u, "should find exactly 2 gguf files");

    /* Sort by filename for deterministic comparison. */
    bool found_q4 = false, found_q8 = false;
    for (size_t i = 0; i < count; i++) {
        if (strcmp(models[i].filename, "model.Q4_K_M.gguf") == 0) {
            found_q4 = true;
            cr_assert_str_eq(models[i].repo_id, "Org/Model-7B",
                             "repo_id un-sanitized");
            cr_assert_str_eq(models[i].quant_type, "Q4_K_M");
            cr_assert(models[i].size_bytes > 0, "size detected");
        }
        if (strcmp(models[i].filename, "model.Q8_0.gguf") == 0) {
            found_q8 = true;
            cr_assert_str_eq(models[i].quant_type, "Q8_0");
        }
    }
    cr_assert(found_q4, "Q4_K_M found");
    cr_assert(found_q8, "Q8_0 found");

    /* Cleanup. */
    unlink(file1); unlink(file2); unlink(file3);
    rmdir(repo_dir); rmdir(tmpl);
}

Test(hf_hub, cache_size_counts_all_files)
{
    char tmpl[] = "/tmp/oxidize-hf-test-XXXXXX";
    cr_assert(mkdtemp(tmpl) != NULL);

    char repo_dir[1024];
    snprintf(repo_dir, sizeof(repo_dir), "%s/repo", tmpl);
    cr_assert_eq(mkdir(repo_dir, 0755), 0);

    char file1[2048], file2[2048];
    snprintf(file1, sizeof(file1), "%s/a.gguf", repo_dir);
    snprintf(file2, sizeof(file2), "%s/b.gguf", repo_dir);
    FILE *f1 = fopen(file1, "w"); cr_assert_not_null(f1);
    fputs("12345", f1); fclose(f1);  /* 5 bytes */
    FILE *f2 = fopen(file2, "w"); cr_assert_not_null(f2);
    fputs("1234567", f2); fclose(f2);  /* 7 bytes */

    OcHfConfig cfg;
    oc_hf_config_init(&cfg, tmpl);
    uint64_t sz = 0;
    cr_assert_eq(oc_hf_cache_size(&cfg, &sz), OC_OK);
    cr_assert_eq(sz, 12u, "5 + 7 = 12 bytes");

    unlink(file1); unlink(file2);
    rmdir(repo_dir); rmdir(tmpl);
}

Test(hf_hub, cache_clean_removes_all_when_max_age_zero)
{
    char tmpl[] = "/tmp/oxidize-hf-test-XXXXXX";
    cr_assert(mkdtemp(tmpl) != NULL);

    char repo_dir[1024];
    snprintf(repo_dir, sizeof(repo_dir), "%s/repo", tmpl);
    cr_assert_eq(mkdir(repo_dir, 0755), 0);

    char file1[2048];
    snprintf(file1, sizeof(file1), "%s/a.gguf", repo_dir);
    FILE *f1 = fopen(file1, "w"); cr_assert_not_null(f1);
    fputs("x", f1); fclose(f1);

    OcHfConfig cfg;
    oc_hf_config_init(&cfg, tmpl);
    size_t removed = 0;
    cr_assert_eq(oc_hf_cache_clean(&cfg, 0, &removed), OC_OK);
    cr_assert_eq(removed, 1u, "should remove 1 file");
    struct stat st;
    cr_assert(stat(file1, &st) != 0, "file should be gone");

    rmdir(repo_dir); rmdir(tmpl);
}

/* ─── Progress struct initialization ────────────────────────────────── */

Test(hf_hub, progress_struct_zero_init)
{
    OcHfDownloadProgress prog = {0};
    cr_assert_eq(prog.downloaded_bytes, 0u);
    cr_assert_eq(prog.total_bytes, 0u);
    cr_assert_eq(prog.speed, 0.0);
    cr_assert_eq(prog.eta, 0.0);
}

Test(hf_hub, progress_struct_field_assignment)
{
    OcHfDownloadProgress prog = {
        .downloaded_bytes = 1024,
        .total_bytes = 4096,
        .speed = 128.0,
        .eta = 24.0,
    };
    cr_assert_eq(prog.downloaded_bytes, 1024u);
    cr_assert_eq(prog.total_bytes, 4096u);
    cr_assert_eq(prog.speed, 128.0);
    cr_assert_eq(prog.eta, 24.0);
}

/* ─── Model struct initialization ───────────────────────────────────── */

Test(hf_hub, model_struct_zero_init)
{
    OcHfModel m = {0};
    cr_assert_eq(m.repo_id[0], '\0');
    cr_assert_eq(m.filename[0], '\0');
    cr_assert_eq(m.size_bytes, 0u);
    cr_assert_eq(m.quant_type[0], '\0');
    cr_assert_eq(m.sha256[0], '\0');
    cr_assert_eq(m.download_url[0], '\0');
}

Test(hf_hub, config_struct_size_constants)
{
    /* Sanity: the buffer-size constants are large enough for realistic IDs. */
    cr_assert(OC_HF_MAX_REPO_ID >= 64u, "repo id buffer is reasonably large");
    cr_assert(OC_HF_MAX_FILENAME >= 128u, "filename buffer is reasonably large");
    cr_assert(OC_HF_MAX_SHA256 >= 65u, "sha256 buffer holds 64 hex + NUL");
    cr_assert(OC_HF_MAX_URL >= 256u, "url buffer is reasonably large");
    cr_assert(OC_HF_MAX_MODELS >= 8u, "model list capacity is reasonable");
}
