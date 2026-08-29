/* test_config.c — Criterion tests for the model config module. */
#define _POSIX_C_SOURCE 200809L

#include <criterion/criterion.h>

#include "oxidize/config.h"
#include "oxidize/gguf.h"
#include "oxidize/gguf_writer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


static const char *tmp_gguf(const char *suffix)
{
    static char path[256];
    snprintf(path, sizeof(path), "/tmp/oc_cfg_test_%d_%s.gguf",
             (int)getpid(), suffix);
    return path;
}

/* Write a llama-style GGUF with the given config fields populated. */
static OcError write_llama_gguf(const char *path,
                                uint32_t n_layers, uint32_t n_heads,
                                uint32_t n_kv_heads, uint32_t embedding,
                                uint32_t ffn, uint32_t vocab, uint32_t ctx,
                                float rope_base, float norm_eps)
{
    OcGgufWriter w;
    OcError e = oc_gguf_writer_init(path, "llama", &w);
    if (e != OC_OK) return e;

    char key[128];
    snprintf(key, sizeof(key), "%s", "llama.block_count");
    oc_gguf_writer_add_uint32(&w, key, n_layers);
    snprintf(key, sizeof(key), "%s", "llama.embedding_length");
    oc_gguf_writer_add_uint32(&w, key, embedding);
    snprintf(key, sizeof(key), "%s", "llama.feed_forward_length");
    oc_gguf_writer_add_uint32(&w, key, ffn);
    snprintf(key, sizeof(key), "%s", "llama.attention.head_count");
    oc_gguf_writer_add_uint32(&w, key, n_heads);
    snprintf(key, sizeof(key), "%s", "llama.attention.head_count_kv");
    oc_gguf_writer_add_uint32(&w, key, n_kv_heads);
    snprintf(key, sizeof(key), "%s", "llama.context_length");
    oc_gguf_writer_add_uint32(&w, key, ctx);
    snprintf(key, sizeof(key), "%s", "llama.vocab_size");
    oc_gguf_writer_add_uint32(&w, key, vocab);
    snprintf(key, sizeof(key), "%s", "llama.rope.freq_base");
    oc_gguf_writer_add_float32(&w, key, rope_base);
    snprintf(key, sizeof(key), "%s", "llama.attention.layer_norm_rms_epsilon");
    oc_gguf_writer_add_float32(&w, key, norm_eps);

    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    return OC_OK;
}


Test(config, init_defaults)
{
    OcModelConfig cfg;
    cr_assert_eq(oc_model_config_init(&cfg), OC_OK);
    cr_assert_str_eq(cfg.arch, "");
    cr_assert_eq(cfg.n_layers, 0);
    cr_assert_eq(cfg.n_heads, 0);
    cr_assert_eq(cfg.rope_theta, 10000.0);
    cr_assert_eq(cfg.rope_scaling_factor, 1.0);
    cr_assert_float_eq((float)cfg.norm_eps, 1e-5f, 1e-12f);
    cr_assert_str_eq(cfg.ffn_type, "swiglu");
    cr_assert_eq(cfg.n_expert, 0);
    cr_assert_eq(cfg.sliding_window, 0);
}

Test(config, init_null)
{
    cr_assert_eq(oc_model_config_init(NULL), OC_ERR_INVALID_ARG);
}

Test(config, from_gguf_llama_dense)
{
    const char *path = tmp_gguf("dense");
    cr_assert_eq(write_llama_gguf(path, 32, 32, 32, 4096, 11008, 32000, 4096,
                                  10000.0f, 1e-5f),
                 OC_OK);

    OcGgufFile gf;
    cr_assert_eq(oc_gguf_open(path, &gf), OC_OK);

    OcModelConfig cfg;
    cr_assert_eq(oc_model_config_from_gguf(&gf, &cfg), OC_OK);
    cr_assert_str_eq(cfg.arch, "llama");
    cr_assert_eq(cfg.n_layers, 32);
    cr_assert_eq(cfg.n_heads, 32);
    cr_assert_eq(cfg.n_kv_heads, 32);
    cr_assert_eq(cfg.hidden_dim, 4096);
    cr_assert_eq(cfg.intermediate_dim, 11008);
    cr_assert_eq(cfg.vocab_size, 32000);
    cr_assert_eq(cfg.n_ctx, 4096);
    cr_assert_eq(cfg.rope_theta, 10000.0);
    cr_assert_float_eq((float)cfg.norm_eps, 1e-5f, 1e-12f);
    /* MHA: n_kv_heads == n_heads, so not GQA. */
    cr_assert(!oc_model_config_has_gqa(&cfg));
    cr_assert(!oc_model_config_is_moe(&cfg));

    oc_gguf_free(&gf);
    unlink(path);
}

Test(config, from_gguf_gqa)
{
    const char *path = tmp_gguf("gqa");
    /* 8 KV heads < 32 query heads => GQA. */
    cr_assert_eq(write_llama_gguf(path, 32, 32, 8, 4096, 11008, 32000, 4096,
                                  10000.0f, 1e-5f),
                 OC_OK);

    OcGgufFile gf;
    cr_assert_eq(oc_gguf_open(path, &gf), OC_OK);
    OcModelConfig cfg;
    cr_assert_eq(oc_model_config_from_gguf(&gf, &cfg), OC_OK);
    cr_assert_eq(cfg.n_kv_heads, 8);
    cr_assert_eq(cfg.n_heads, 32);
    cr_assert(oc_model_config_has_gqa(&cfg));
    cr_assert(!oc_model_config_is_moe(&cfg));
    oc_gguf_free(&gf);
    unlink(path);
}

Test(config, from_gguf_head_dim_derived)
{
    const char *path = tmp_gguf("headdim");
    /* No attention.key_length; head_dim should derive from hidden/n_heads. */
    cr_assert_eq(write_llama_gguf(path, 4, 8, 8, 256, 512, 100, 512,
                                  10000.0f, 1e-5f),
                 OC_OK);
    OcGgufFile gf;
    cr_assert_eq(oc_gguf_open(path, &gf), OC_OK);
    OcModelConfig cfg;
    cr_assert_eq(oc_model_config_from_gguf(&gf, &cfg), OC_OK);
    cr_assert_eq(cfg.head_dim, 32); /* 256 / 8 */
    oc_gguf_free(&gf);
    unlink(path);
}

Test(config, from_gguf_null_args)
{
    OcModelConfig cfg;
    cr_assert_eq(oc_model_config_from_gguf(NULL, &cfg), OC_ERR_INVALID_ARG);
    OcGgufFile gf = {0};
    cr_assert_eq(oc_model_config_from_gguf(&gf, NULL), OC_ERR_INVALID_ARG);
}

Test(config, from_gguf_no_arch)
{
    const char *path = tmp_gguf("noarch");
    /* Write a GGUF with architecture then... we need to omit it. The writer
     * always writes general.architecture. Instead, test the path by passing a
     * zeroed OcGgufFile (no metadata). */
    (void)path;
    OcGgufFile gf;
    memset(&gf, 0, sizeof(gf));
    OcModelConfig cfg;
    /* general.architecture lookup returns NULL -> OC_ERR_FORMAT. */
    cr_assert_eq(oc_model_config_from_gguf(&gf, &cfg), OC_ERR_FORMAT);
}

Test(config, validate_ok)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cfg.n_layers = 32;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 32;
    cfg.head_dim = 128;
    cfg.hidden_dim = 4096;
    cfg.intermediate_dim = 11008;
    cfg.vocab_size = 32000;
    cr_assert_eq(oc_model_config_validate(&cfg), OC_OK);
}

Test(config, validate_null)
{
    cr_assert_eq(oc_model_config_validate(NULL), OC_ERR_INVALID_ARG);
}

Test(config, validate_missing_layers)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cfg.n_heads = 32;
    cfg.hidden_dim = 4096;
    cfg.vocab_size = 32000;
    cfg.n_kv_heads = 32;
    cr_assert_neq(oc_model_config_validate(&cfg), OC_OK);
}

Test(config, validate_kv_heads_exceed_heads)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cfg.n_layers = 4;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 16; /* > n_heads */
    cfg.head_dim = 32;
    cfg.hidden_dim = 256;
    cfg.vocab_size = 100;
    cr_assert_neq(oc_model_config_validate(&cfg), OC_OK);
}

Test(config, validate_moe_no_used)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cfg.n_layers = 4;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 32;
    cfg.hidden_dim = 256;
    cfg.vocab_size = 100;
    cfg.n_expert = 8;
    cfg.n_expert_used = 0; /* invalid */
    cr_assert_neq(oc_model_config_validate(&cfg), OC_OK);
}

Test(config, validate_moe_used_exceeds)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cfg.n_layers = 4;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 32;
    cfg.hidden_dim = 256;
    cfg.vocab_size = 100;
    cfg.n_expert = 4;
    cfg.n_expert_used = 8; /* > n_expert */
    cr_assert_neq(oc_model_config_validate(&cfg), OC_OK);
}

Test(config, validate_moe_ok)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cfg.n_layers = 4;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 32;
    cfg.hidden_dim = 256;
    cfg.vocab_size = 100;
    cfg.n_expert = 8;
    cfg.n_expert_used = 2;
    cr_assert_eq(oc_model_config_validate(&cfg), OC_OK);
}

Test(config, arch_name)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cr_assert_str_eq(oc_model_config_arch_name(&cfg), "");
    strncpy(cfg.arch, "qwen2", OC_CONFIG_ARCH_LEN - 1);
    cr_assert_str_eq(oc_model_config_arch_name(&cfg), "qwen2");
    cr_assert_str_eq(oc_model_config_arch_name(NULL), "");
}

Test(config, is_moe)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cr_assert(!oc_model_config_is_moe(&cfg));
    cfg.n_expert = 8;
    cr_assert(oc_model_config_is_moe(&cfg));
    cr_assert(!oc_model_config_is_moe(NULL));
}

Test(config, has_gqa)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    cfg.n_heads = 32;
    cfg.n_kv_heads = 32;
    cr_assert(!oc_model_config_has_gqa(&cfg));
    cfg.n_kv_heads = 8;
    cr_assert(oc_model_config_has_gqa(&cfg));
    cr_assert(!oc_model_config_has_gqa(NULL));
}

Test(config, print_writes_summary)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    strncpy(cfg.arch, "llama", OC_CONFIG_ARCH_LEN - 1);
    cfg.n_layers = 32;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_dim = 4096;
    cfg.intermediate_dim = 11008;
    cfg.vocab_size = 32000;
    cfg.n_ctx = 4096;
    cfg.n_expert = 0;
    cfg.n_expert_used = 0;
    char buf[1024];
    size_t n = oc_model_config_print(&cfg, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(n < sizeof(buf));
    cr_assert(strstr(buf, "arch=llama") != NULL);
    cr_assert(strstr(buf, "n_layers=32") != NULL);
    cr_assert(strstr(buf, "gqa=yes") != NULL);
    cr_assert(strstr(buf, "moe=no") != NULL);
}

Test(config, print_null)
{
    char buf[64];
    cr_assert_eq(oc_model_config_print(NULL, buf, sizeof(buf)), 0);
    cr_assert_eq(buf[0], '\0');
}

Test(config, print_small_buffer)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    strncpy(cfg.arch, "llama", OC_CONFIG_ARCH_LEN - 1);
    cfg.n_layers = 32;
    char buf[4];
    size_t n = oc_model_config_print(&cfg, buf, sizeof(buf));
    /* Length reported is the full length, but buf is truncated + NUL. */
    cr_assert_gt(n, 3);
    cr_assert_eq(buf[3], '\0');
}

Test(config, n_params_dense)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    strncpy(cfg.arch, "llama", OC_CONFIG_ARCH_LEN - 1);
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 4;
    cfg.head_dim = 8;        /* hidden = 32 */
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    cfg.vocab_size = 100;
    uint64_t n = oc_model_config_n_params(&cfg);
    cr_assert_gt(n, 0);
    /* Sanity: embedding (100*32=3200) + lm_head (3200) + per-layer pieces
     * must all be present. */
    cr_assert_gt(n, 2 * 100 * 32);
}

Test(config, n_params_null)
{
    cr_assert_eq(oc_model_config_n_params(NULL), 0);
}

Test(config, n_params_invalid)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    /* Zero fields -> invalid -> 0. */
    cr_assert_eq(oc_model_config_n_params(&cfg), 0);
}

Test(config, n_params_moe_scales_by_experts)
{
    OcModelConfig cfg;
    oc_model_config_init(&cfg);
    strncpy(cfg.arch, "mixtral", OC_CONFIG_ARCH_LEN - 1);
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 4;
    cfg.head_dim = 8;
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    cfg.vocab_size = 100;
    cfg.n_expert = 4;
    cfg.n_expert_used = 2;

    uint64_t moe_params = oc_model_config_n_params(&cfg);

    /* Same config but dense (n_expert=0). */
    OcModelConfig dense = cfg;
    dense.n_expert = 0;
    dense.n_expert_used = 0;
    uint64_t dense_params = oc_model_config_n_params(&dense);

    /* MoE FFN portion is 4x the dense FFN portion (n_expert=4). So total
     * MoE params must exceed dense params. */
    cr_assert_gt(moe_params, dense_params);
}
