/* test_mlx_inference.c — OcMlxEngine stub tests.
 *
 * Unique suite name "mlx_inference". On non-macOS every operation is a
 * stub returning OC_ERR_BACKEND; these tests assert that contract.
 */
#include "framework.h"
#include <string.h>

#include "oxidize/mlx_inference.h"

/* ─── config init ──────────────────────────────────────────────────── */

Test(mlx_inference, config_init_defaults)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    cr_assert_eq(cfg.hidden_size, 4096u, "");
    cr_assert_eq(cfg.vocab_size, 32000u, "");
    cr_assert_eq(cfg.n_layers, 32u, "");
    cr_assert(cfg.use_metal, "");
    cr_assert_eq(cfg.model_path[0], '\0', "empty path");
}

Test(mlx_inference, config_init_null_is_noop)
{
    oc_mlx_config_init(NULL);
    cr_assert(true, "");
}

/* ─── engine init ──────────────────────────────────────────────────── */

Test(mlx_inference, engine_init_good)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    cr_assert_eq(oc_mlx_engine_init(&eng, &cfg), OC_OK, "");
    cr_assert_eq(eng.config.hidden_size, 4096u, "");
    cr_assert_not(eng.loaded, "not loaded after init");
    cr_assert_eq(eng.available, oc_mlx_is_available(), "available matches host");
    oc_mlx_engine_free(&eng);
}

Test(mlx_inference, engine_init_null_args)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    cr_assert_eq(oc_mlx_engine_init(NULL, &cfg), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_mlx_engine_init(&eng, NULL), OC_ERR_INVALID_ARG, "");
}

Test(mlx_inference, engine_free_null_is_safe)
{
    oc_mlx_engine_free(NULL);
    cr_assert(true, "");
}

Test(mlx_inference, engine_free_resets_flags)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    oc_mlx_engine_init(&eng, &cfg);
    oc_mlx_engine_free(&eng);
    cr_assert_not(eng.loaded, "");
    cr_assert_not(eng.available, "");
}

/* ─── load ─────────────────────────────────────────────────────────── */

Test(mlx_inference, load_returns_backend_error)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    oc_mlx_engine_init(&eng, &cfg);
    cr_assert_eq(oc_mlx_engine_load(&eng, "/tmp/fake.gguf"), OC_ERR_BACKEND, "");
    cr_assert_not(eng.loaded, "load did not set loaded");
    oc_mlx_engine_free(&eng);
}

Test(mlx_inference, load_null_path)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    oc_mlx_engine_init(&eng, &cfg);
    cr_assert_eq(oc_mlx_engine_load(&eng, NULL), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_mlx_engine_load(&eng, ""), OC_ERR_INVALID_ARG, "");
    oc_mlx_engine_free(&eng);
}

OC_TEST_NULL_SAFE(mlx_inference, load_null_engine,
        cr_assert_eq(oc_mlx_engine_load(NULL, "/tmp/x"), OC_ERR_INVALID_ARG, "");)

/* ─── generate ─────────────────────────────────────────────────────── */

Test(mlx_inference, generate_returns_error)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    oc_mlx_engine_init(&eng, &cfg);
    uint32_t in[2] = {1, 2};
    uint32_t out[4] = {0};
    size_t out_n = 0;
    cr_assert_eq(oc_mlx_engine_generate(&eng, in, 2, 4, out, &out_n),
                 OC_ERR_BACKEND, "");
    cr_assert_eq(out_n, 0u, "");
    oc_mlx_engine_free(&eng);
}

Test(mlx_inference, generate_null_args)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    oc_mlx_engine_init(&eng, &cfg);
    uint32_t in[1] = {1};
    uint32_t out[1] = {0};
    size_t out_n = 0;
    cr_assert_eq(oc_mlx_engine_generate(NULL, in, 1, 1, out, &out_n),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_mlx_engine_generate(&eng, NULL, 1, 1, out, &out_n),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_mlx_engine_generate(&eng, in, 1, 1, NULL, &out_n),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_mlx_engine_generate(&eng, in, 1, 1, out, NULL),
                 OC_ERR_INVALID_ARG, "");
    oc_mlx_engine_free(&eng);
}

Test(mlx_inference, generate_zero_tokens_or_max)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    oc_mlx_engine_init(&eng, &cfg);
    uint32_t in[1] = {1};
    uint32_t out[1] = {0};
    size_t out_n = 0;
    cr_assert_eq(oc_mlx_engine_generate(&eng, in, 0, 1, out, &out_n),
                 OC_ERR_INVALID_ARG, "n_tokens==0");
    cr_assert_eq(oc_mlx_engine_generate(&eng, in, 1, 0, out, &out_n),
                 OC_ERR_INVALID_ARG, "max_new==0");
    oc_mlx_engine_free(&eng);
}

/* ─── platform info ───────────────────────────────────────────────── */

Test(mlx_inference, is_available_is_false_on_non_macos)
{
    /* On non-macOS this MUST be false. On macOS the stub also reports
     * false, so the assertion holds unconditionally. */
    cr_assert_not(oc_mlx_is_available(), "");
}

Test(mlx_inference, backend_name_is_mlx)
{
    cr_assert_str_eq(oc_mlx_backend_name(), "mlx", "");
}

/* ─── available flag ──────────────────────────────────────────────── */

Test(mlx_inference, available_flag_matches_host)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    oc_mlx_engine_init(&eng, &cfg);
    cr_assert_eq(eng.available, oc_mlx_is_available(), "");
    oc_mlx_engine_free(&eng);
}

/* ─── loaded flag ─────────────────────────────────────────────────── */

Test(mlx_inference, loaded_flag_false_after_init)
{
    OcMlxConfig cfg;
    oc_mlx_config_init(&cfg);
    OcMlxEngine eng;
    oc_mlx_engine_init(&eng, &cfg);
    cr_assert_not(eng.loaded, "");
    oc_mlx_engine_free(&eng);
}
