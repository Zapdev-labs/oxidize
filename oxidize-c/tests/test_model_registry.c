#define _POSIX_C_SOURCE 200809L

#include <criterion/criterion.h>
#include "oxidize/model_registry.h"

#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>


/* Create a temp directory and a couple of fake .gguf files (zero-byte, so
 * the GGUF parse will fail gracefully and entries get arch=UNKNOWN). */
static char tmpdir[256];

static void make_tmpdir(void)
{
    snprintf(tmpdir, sizeof(tmpdir), "/tmp/oc_reg_test_%d", (int)getpid());
    mkdir(tmpdir, 0700);
}

static void cleanup_tmpdir(void)
{
    /* Best-effort cleanup; ignore errors. */
    DIR *d = opendir(tmpdir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            if (de->d_name[0] == '.') continue;
            char p[512];
            snprintf(p, sizeof(p), "%s/%s", tmpdir, de->d_name);
            unlink(p);
        }
        closedir(d);
    }
    rmdir(tmpdir);
}

static void touch_gguf(const char *name)
{
    char p[512];
    snprintf(p, sizeof(p), "%s/%s", tmpdir, name);
    FILE *fp = fopen(p, "wb");
    if (fp) fclose(fp);
}


Test(model_registry, init_free)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, "/tmp/oc_test", 16), OC_OK);
    cr_assert_eq(reg.count, 0);
    cr_assert_eq(reg.max_entries, 16);
    cr_assert_str_eq(reg.cache_dir, "/tmp/oc_test");
    oc_model_registry_free(&reg);
    cr_assert_eq(reg.count, 0);
}

Test(model_registry, init_default_max)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 0), OC_OK);
    cr_assert_eq(reg.max_entries, OC_MODEL_REGISTRY_MAX_ENTRIES);
    cr_assert_eq(reg.cache_dir[0], '\0');
    oc_model_registry_free(&reg);
}

Test(model_registry, init_clamps_max)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 99999), OC_OK);
    cr_assert_eq(reg.max_entries, OC_MODEL_REGISTRY_MAX_ENTRIES);
    oc_model_registry_free(&reg);
}

Test(model_registry, init_null)
{
    cr_assert_eq(oc_model_registry_init(NULL, "/tmp", 4), OC_ERR_INVALID_ARG);
}

Test(model_registry, free_null)
{
    /* Should not crash. */
    oc_model_registry_free(NULL);
}


Test(model_registry, add_single)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/foo.gguf"), OC_OK);
    cr_assert_eq(reg.count, 1);
    cr_assert_str_eq(reg.entries[0].path, "/tmp/foo.gguf");
    cr_assert_str_eq(reg.entries[0].name, "foo");
    oc_model_registry_free(&reg);
}

Test(model_registry, add_dedup_by_path)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/foo.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/foo.gguf"), OC_OK);
    cr_assert_eq(reg.count, 1);
    oc_model_registry_free(&reg);
}

Test(model_registry, add_null_args)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_model_registry_add(NULL, "/tmp/x.gguf"), OC_ERR_INVALID_ARG);
    oc_model_registry_free(&reg);
}

Test(model_registry, remove)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/a.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/b.gguf"), OC_OK);
    cr_assert_eq(reg.count, 2);
    cr_assert_eq(oc_model_registry_remove(&reg, "/tmp/a.gguf"), OC_OK);
    cr_assert_eq(reg.count, 1);
    cr_assert_str_eq(reg.entries[0].path, "/tmp/b.gguf");
    oc_model_registry_free(&reg);
}

Test(model_registry, remove_not_found)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/a.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_remove(&reg, "/tmp/missing.gguf"),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(reg.count, 1);
    oc_model_registry_free(&reg);
}

Test(model_registry, remove_null)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_remove(&reg, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_model_registry_remove(NULL, "/tmp/x.gguf"),
                 OC_ERR_INVALID_ARG);
    oc_model_registry_free(&reg);
}


Test(model_registry, scan_empty_dir)
{
    make_tmpdir();
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, tmpdir, 16), OC_OK);
    cr_assert_eq(oc_model_registry_scan(&reg, tmpdir), OC_OK);
    cr_assert_eq(reg.count, 0);
    oc_model_registry_free(&reg);
    cleanup_tmpdir();
}

Test(model_registry, scan_finds_gguf)
{
    make_tmpdir();
    touch_gguf("alpha.gguf");
    touch_gguf("beta.gguf");
    /* Non-.gguf file should be ignored. */
    FILE *fp = fopen("/tmp/oc_reg_test_x.txt", "w");
    if (fp) { fclose(fp); unlink("/tmp/oc_reg_test_x.txt"); }

    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, tmpdir, 16), OC_OK);
    cr_assert_eq(oc_model_registry_scan(&reg, tmpdir), OC_OK);
    cr_assert_eq(reg.count, 2);
    oc_model_registry_free(&reg);
    cleanup_tmpdir();
}

Test(model_registry, scan_ignores_non_gguf)
{
    make_tmpdir();
    /* Create a non-gguf file in the tmpdir. */
    char p[512];
    snprintf(p, sizeof(p), "%s/notes.txt", tmpdir);
    FILE *fp = fopen(p, "w");
    if (fp) fclose(fp);

    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, tmpdir, 16), OC_OK);
    cr_assert_eq(oc_model_registry_scan(&reg, tmpdir), OC_OK);
    cr_assert_eq(reg.count, 0);
    oc_model_registry_free(&reg);
    cleanup_tmpdir();
}

Test(model_registry, scan_nonexistent_dir)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 16), OC_OK);
    cr_assert_eq(oc_model_registry_scan(&reg, "/tmp/does_not_exist_xyz"),
                 OC_ERR_IO);
    oc_model_registry_free(&reg);
}

Test(model_registry, scan_null_args)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 16), OC_OK);
    cr_assert_eq(oc_model_registry_scan(&reg, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_model_registry_scan(NULL, "/tmp"), OC_ERR_INVALID_ARG);
    oc_model_registry_free(&reg);
}


Test(model_registry, find_substring)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 16), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/llama-7b.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/qwen-14b.gguf"), OC_OK);

    const OcModelEntry *e = oc_model_registry_find(&reg, "llama");
    cr_assert_not_null(e);
    cr_assert_str_eq(e->name, "llama-7b");

    e = oc_model_registry_find(&reg, "QWEN");
    cr_assert_not_null(e);
    cr_assert_str_eq(e->name, "qwen-14b");

    oc_model_registry_free(&reg);
}

Test(model_registry, find_no_match)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 16), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/llama.gguf"), OC_OK);
    cr_assert_null(oc_model_registry_find(&reg, "completely-unrelated-name"));
    oc_model_registry_free(&reg);
}

Test(model_registry, find_null)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 16), OC_OK);
    cr_assert_null(oc_model_registry_find(NULL, "x"));
    cr_assert_null(oc_model_registry_find(&reg, NULL));
    cr_assert_null(oc_model_registry_find(&reg, ""));
    oc_model_registry_free(&reg);
}


Test(model_registry, list_by_name)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 16), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/zebra.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/apple.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/mango.gguf"), OC_OK);

    const OcModelEntry *out[3];
    size_t n = oc_model_registry_list(&reg, OC_MODEL_SORT_BY_NAME, out, 3);
    cr_assert_eq(n, 3);
    cr_assert_str_eq(out[0]->name, "apple");
    cr_assert_str_eq(out[1]->name, "mango");
    cr_assert_str_eq(out[2]->name, "zebra");
    oc_model_registry_free(&reg);
}

Test(model_registry, list_by_size)
{
    /* Sizes are 0 (no real files), so ties break arbitrarily but count is right. */
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 16), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/a.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/b.gguf"), OC_OK);
    const OcModelEntry *out[2];
    size_t n = oc_model_registry_list(&reg, OC_MODEL_SORT_BY_SIZE, out, 2);
    cr_assert_eq(n, 2);
    oc_model_registry_free(&reg);
}

Test(model_registry, list_null_registry)
{
    cr_assert_eq(oc_model_registry_list(NULL, OC_MODEL_SORT_BY_NAME, NULL, 0),
                 0);
}


Test(model_registry, format_empty)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    char buf[64];
    size_t n = oc_model_registry_format(&reg, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert_str_eq(buf, "[]");
    oc_model_registry_free(&reg);
}

Test(model_registry, format_single)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/foo.gguf"), OC_OK);
    char buf[1024];
    size_t n = oc_model_registry_format(&reg, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "\"path\":\"/tmp/foo.gguf\"") != NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "\"name\":\"foo\"") != NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "\"arch\":\"unknown\"") != NULL, "buf=%s", buf);
    oc_model_registry_free(&reg);
}

Test(model_registry, format_multiple)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/a.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/b.gguf"), OC_OK);
    char buf[2048];
    size_t n = oc_model_registry_format(&reg, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert_eq(buf[0], '[');
    cr_assert_eq(buf[n - 1], ']');
    /* Two objects separated by a comma. */
    cr_assert(strstr(buf, "},{") != NULL, "buf=%s", buf);
    oc_model_registry_free(&reg);
}

Test(model_registry, format_null)
{
    char buf[64];
    cr_assert_eq(oc_model_registry_format(NULL, buf, sizeof(buf)), 0);
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_format(&reg, NULL, 0), 0);
    oc_model_registry_free(&reg);
}


Test(model_registry, stats_empty)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    OcModelRegistryStats st;
    oc_model_registry_stats(&reg, &st);
    cr_assert_eq(st.total_models, 0);
    cr_assert_eq(st.total_size, 0);
    oc_model_registry_free(&reg);
}

Test(model_registry, stats_populated)
{
    OcModelRegistry reg;
    cr_assert_eq(oc_model_registry_init(&reg, NULL, 8), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/a.gguf"), OC_OK);
    cr_assert_eq(oc_model_registry_add(&reg, "/tmp/b.gguf"), OC_OK);
    OcModelRegistryStats st;
    oc_model_registry_stats(&reg, &st);
    cr_assert_eq(st.total_models, 2);
    /* All entries have arch=UNKNOWN since the files don't exist. */
    cr_assert_eq(st.by_arch[OC_ARCH_UNKNOWN], 2);
    cr_assert_gt(st.n_quant_types, 0);
    oc_model_registry_free(&reg);
}

Test(model_registry, stats_null)
{
    OcModelRegistryStats st;
    oc_model_registry_stats(NULL, &st);
    cr_assert_eq(st.total_models, 0);
}
