#include <criterion/criterion.h>
#include <string.h>

#include "oxidize/gguf.h"
#include "oxidize/glm_arch.h"
#include "oxidize/model.h"


Test(glm_arch, version_from_str_glm)
{
    cr_assert_eq(oc_glm_version_from_str("glm"), OC_GLM_VERSION_1,
        "glm → version 1 (ChatGLM-6B)");
}

Test(glm_arch, version_from_str_chatglm)
{
    cr_assert_eq(oc_glm_version_from_str("chatglm"), OC_GLM_VERSION_1,
        "chatglm → version 1 (strip 'chat' prefix)");
}

Test(glm_arch, version_from_str_glm4)
{
    cr_assert_eq(oc_glm_version_from_str("glm4"), OC_GLM_VERSION_4,
        "glm4 → version 4 (GLM-4)");
}

Test(glm_arch, version_from_str_glm4_hyphen)
{
    cr_assert_eq(oc_glm_version_from_str("glm-4"), OC_GLM_VERSION_4,
        "glm-4 → version 4 (hyphen normalized to underscore)");
}

Test(glm_arch, version_from_str_glm2)
{
    cr_assert_eq(oc_glm_version_from_str("glm2"), OC_GLM_VERSION_2,
        "glm2 → version 2 (ChatGLM2)");
}

Test(glm_arch, version_from_str_glm3)
{
    cr_assert_eq(oc_glm_version_from_str("glm3"), OC_GLM_VERSION_3,
        "glm3 → version 3 (ChatGLM3)");
}

Test(glm_arch, version_from_str_glm_moe_variants)
{
    /* MoE/DSA variants are built on GLM-4. */
    cr_assert_eq(oc_glm_version_from_str("glm_moe"), OC_GLM_VERSION_4,
        "glm_moe → version 4");
    cr_assert_eq(oc_glm_version_from_str("glm_moe_dsa"), OC_GLM_VERSION_4,
        "glm_moe_dsa → version 4");
    cr_assert_eq(oc_glm_version_from_str("glm_dsa"), OC_GLM_VERSION_4,
        "glm_dsa → version 4");
    cr_assert_eq(oc_glm_version_from_str("glmmoe"), OC_GLM_VERSION_4,
        "glmmoe → version 4");
}

Test(glm_arch, version_from_str_case_insensitive)
{
    cr_assert_eq(oc_glm_version_from_str("GLM"), OC_GLM_VERSION_1,
        "GLM (uppercase) → version 1");
    cr_assert_eq(oc_glm_version_from_str("GLM4"), OC_GLM_VERSION_4,
        "GLM4 (uppercase) → version 4");
    cr_assert_eq(oc_glm_version_from_str("ChatGLM"), OC_GLM_VERSION_1,
        "ChatGLM (mixed case) → version 1");
}

Test(glm_arch, version_from_str_unknown)
{
    cr_assert_eq(oc_glm_version_from_str("llama"), OC_GLM_VERSION_UNKNOWN,
        "llama → unknown (not a GLM model)");
    cr_assert_eq(oc_glm_version_from_str("qwen"), OC_GLM_VERSION_UNKNOWN,
        "qwen → unknown");
    cr_assert_eq(oc_glm_version_from_str("hunyuan"), OC_GLM_VERSION_UNKNOWN,
        "hunyuan → unknown");
}


Test(glm_arch, version_from_str_null)
{
    cr_assert_eq(oc_glm_version_from_str(NULL), OC_GLM_VERSION_UNKNOWN,
        "NULL → unknown");
}

Test(glm_arch, version_from_str_empty)
{
    cr_assert_eq(oc_glm_version_from_str(""), OC_GLM_VERSION_UNKNOWN,
        "empty string → unknown");
}

Test(glm_arch, config_parse_null_file)
{
    OcGlmConfig cfg;
    cr_assert_eq(oc_glm_config_parse(NULL, "glm", &cfg), OC_ERR_INVALID_ARG,
        "NULL file → OC_ERR_INVALID_ARG");
}

Test(glm_arch, config_parse_null_cfg)
{
    /* Build a minimal valid GGUF so the file arg is non-NULL. */
    OcGgufFile f;
    memset(&f, 0, sizeof(f));
    cr_assert_eq(oc_glm_config_parse(&f, "glm", NULL), OC_ERR_INVALID_ARG,
        "NULL cfg → OC_ERR_INVALID_ARG");
}

Test(glm_arch, hunyuan_config_parse_null_file)
{
    OcHunyuanConfig cfg;
    cr_assert_eq(oc_hunyuan_config_parse(NULL, "hunyuan", &cfg),
                 OC_ERR_INVALID_ARG,
        "NULL file → OC_ERR_INVALID_ARG");
}

Test(glm_arch, hunyuan_config_parse_null_cfg)
{
    OcGgufFile f;
    memset(&f, 0, sizeof(f));
    cr_assert_eq(oc_hunyuan_config_parse(&f, "hunyuan", NULL),
                 OC_ERR_INVALID_ARG,
        "NULL cfg → OC_ERR_INVALID_ARG");
}

Test(glm_arch, config_defaults_null)
{
    /* Should not crash. */
    oc_glm_config_defaults(NULL);
    oc_hunyuan_config_defaults(NULL);
}


Test(glm_arch, glm_config_defaults)
{
    OcGlmConfig cfg;
    oc_glm_config_defaults(&cfg);

    cr_assert_eq(cfg.vocab_size, 32000, "default vocab_size");
    cr_assert_eq(cfg.hidden_size, 4096, "default hidden_size");
    cr_assert_eq(cfg.n_layer, 28, "default n_layer");
    cr_assert_eq(cfg.num_attention_heads, 32, "default num_attention_heads");
    cr_assert_eq(cfg.num_kv_heads, 32, "default num_kv_heads");
    cr_assert_eq(cfg.intermediate_size, 11008, "default intermediate_size");
    cr_assert_eq(cfg.max_position_embeddings, 32768, "default max_position");
    cr_assert_eq(cfg.head_dim, 128, "default head_dim");
    cr_assert_eq(cfg.kv_head_dim, 128, "default kv_head_dim");
    cr_assert_eq(cfg.rope_dim, 128, "default rope_dim");
    cr_assert_float_eq(cfg.rope_theta, 10000.0f, 1e-3f, "default rope_theta");
    cr_assert_float_eq(cfg.rms_norm_eps, 1e-5f, 1e-7f, "default rms_norm_eps");
    cr_assert_eq(cfg.uses_mla, false, "default uses_mla");
    cr_assert_eq(cfg.apply_qk_norm, false, "default apply_qk_norm");
    cr_assert_eq(cfg.uses_interleaved_rope, false, "default interleaved_rope");
    cr_assert_eq(cfg.tied_embeddings, false, "default tied_embeddings");
    cr_assert_eq(cfg.glm_version, OC_GLM_VERSION_UNKNOWN,
        "default glm_version");
}

Test(glm_arch, hunyuan_config_defaults)
{
    OcHunyuanConfig cfg;
    oc_hunyuan_config_defaults(&cfg);

    cr_assert_eq(cfg.vocab_size, 32000, "default vocab_size");
    cr_assert_eq(cfg.hidden_size, 4096, "default hidden_size");
    cr_assert_eq(cfg.n_layer, 32, "default n_layer");
    cr_assert_eq(cfg.num_attention_heads, 32, "default num_attention_heads");
    cr_assert_eq(cfg.num_kv_heads, 8, "default num_kv_heads");
    cr_assert_eq(cfg.intermediate_size, 11008, "default intermediate_size");
    cr_assert_eq(cfg.max_position_embeddings, 32768, "default max_position");
    cr_assert_eq(cfg.head_dim, 128, "default head_dim");
    cr_assert_eq(cfg.rope_dim, 128, "default rope_dim");
    cr_assert_float_eq(cfg.rope_theta, 10000.0f, 1e-3f, "default rope_theta");
    cr_assert_float_eq(cfg.rms_norm_eps, 1e-5f, 1e-7f, "default rms_norm_eps");
    cr_assert_eq(cfg.n_routed_experts, 0, "default n_routed_experts");
    cr_assert_eq(cfg.n_active_experts, 0, "default n_active_experts");
    cr_assert_eq(cfg.expert_intermediate_size, 0,
        "default expert_intermediate_size");
    cr_assert_eq(cfg.moe_layer_start, 0, "default moe_layer_start");
    cr_assert_eq(cfg.has_shared_expert, false, "default has_shared_expert");
    cr_assert_eq(cfg.uses_mla, false, "default uses_mla");
    cr_assert_eq(cfg.tied_embeddings, false, "default tied_embeddings");
}


/* Helper: append a u32 value to a buffer at offset, return new offset. */
static size_t put_u32(uint8_t *buf, size_t off, uint32_t val)
{
    memcpy(buf + off, &val, 4);
    return off + 4;
}

static size_t put_u64(uint8_t *buf, size_t off, uint64_t val)
{
    memcpy(buf + off, &val, 8);
    return off + 8;
}

static size_t put_f32(uint8_t *buf, size_t off, float val)
{
    memcpy(buf + off, &val, 4);
    return off + 4;
}

static size_t put_str_key(uint8_t *buf, size_t off, const char *key)
{
    size_t len = strlen(key);
    off = put_u64(buf, off, (uint64_t)len);
    memcpy(buf + off, key, len);
    off += len;
    buf[off++] = '\0';   /* GGUF strings are NUL-terminated after the length */
    return off;
}

/* Build a minimal GGUF buffer with the given KV pairs. For strings, we only store the u32/f32 path here (tests don't need strings). */
static OcGgufFile build_minimal_gguf(const char **keys, const uint32_t *types,
                                      const uint32_t *u32_vals,
                                      const float *f32_vals, size_t n_kv)
{
    /* Allocate a generous buffer (4 KB is ample for ~20 KV pairs). */
    static uint8_t buf[4096];
    size_t off = 0;

    /* Magic: "GGUF" = 0x46554747 LE. */
    off = put_u32(buf, off, 0x46554747u);
    /* Version 3. */
    off = put_u32(buf, off, 3u);
    /* n_tensors = 0. */
    off = put_u64(buf, off, 0ull);
    /* n_kv. */
    off = put_u64(buf, off, (uint64_t)n_kv);

    for (size_t i = 0; i < n_kv; i++) {
        off = put_str_key(buf, off, keys[i]);
        off = put_u32(buf, off, types[i]);   /* value type */
        if (types[i] == 4) {                  /* U32 */
            off = put_u32(buf, off, u32_vals[i]);
        } else if (types[i] == 6) {           /* F32 */
            off = put_f32(buf, off, f32_vals[i]);
        } else if (types[i] == 7) {           /* BOOL (stored as u8) */
            buf[off++] = (uint8_t)u32_vals[i];
        }
    }

    /* Pad to 32-byte alignment. */
    while (off % 32 != 0) buf[off++] = 0;

    OcGgufFile f;
    memset(&f, 0, sizeof(f));
    /* We pass a stack buffer; oc_gguf_parse will dup into its arena. */
    OcError e = oc_gguf_parse(buf, off, &f);
    cr_assert_eq(e, OC_OK, "oc_gguf_parse failed for test GGUF");

    /* Return by value — the caller must oc_gguf_free it. */
    return f;
}

Test(glm_arch, config_parse_glm4_metadata, .disabled=true)
{
    /* Build a GGUF with GLM-4 metadata keys. */
    const char *keys[] = {
        "general.architecture",
        "glm.vocab_size",
        "glm.hidden_size",
        "glm.num_layers",
        "glm.num_attention_heads",
        "glm.num_key_value_heads",
        "glm.intermediate_size",
        "glm.max_position_embeddings",
        "glm.rope_theta",
        "glm.rms_norm_eps",
        "glm.apply_qk_norm",
        "glm.tied_word_embeddings",
    };
    const uint32_t types[] = {
        8,    /* STRING (we store a dummy u32 since we don't test the string) */
        4, 4, 4, 4, 4, 4, 4,   /* u32 */
        6,    /* f32 rope_theta */
        6,    /* f32 rms_norm_eps */
        7,    /* bool apply_qk_norm */
        7,    /* bool tied_word_embeddings */
    };
    const uint32_t u32_vals[] = {
        0,        /* general.architecture (placeholder; we pass arch_str) */
        151552,   /* vocab_size */
        4096,     /* hidden_size */
        40,       /* num_layers */
        32,       /* num_attention_heads */
        2,        /* num_key_value_heads */
        14436,    /* intermediate_size */
        32768,    /* max_position_embeddings */
        0,        /* (rope_theta is f32) */
        0,        /* (rms_norm_eps is f32) */
        1,        /* apply_qk_norm = true */
        0,        /* tied_word_embeddings = false */
    };
    const float f32_vals[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        500000.0f,  /* rope_theta */
        1e-6f,      /* rms_norm_eps */
        0.0f, 0.0f,
    };

    OcGgufFile f = build_minimal_gguf(keys, types, u32_vals, f32_vals, 12);

    OcGlmConfig cfg;
    OcError e = oc_glm_config_parse(&f, "glm4", &cfg);
    cr_assert_eq(e, OC_OK, "oc_glm_config_parse should succeed");

    cr_assert_eq(cfg.vocab_size, 151552, "parsed vocab_size");
    cr_assert_eq(cfg.hidden_size, 4096, "parsed hidden_size");
    cr_assert_eq(cfg.n_layer, 40, "parsed n_layer");
    cr_assert_eq(cfg.num_attention_heads, 32, "parsed num_attention_heads");
    cr_assert_eq(cfg.num_kv_heads, 2, "parsed num_kv_heads");
    cr_assert_eq(cfg.intermediate_size, 14436, "parsed intermediate_size");
    cr_assert_eq(cfg.max_position_embeddings, 32768, "parsed max_position");
    cr_assert_float_eq(cfg.rope_theta, 500000.0f, 1e-1f, "parsed rope_theta");
    cr_assert_float_eq(cfg.rms_norm_eps, 1e-6f, 1e-8f, "parsed rms_norm_eps");
    cr_assert_eq(cfg.apply_qk_norm, true, "parsed apply_qk_norm");
    cr_assert_eq(cfg.tied_embeddings, false, "parsed tied_embeddings");
    cr_assert_eq(cfg.glm_version, OC_GLM_VERSION_4, "detected GLM-4 version");
    cr_assert_eq(cfg.uses_interleaved_rope, false,
        "GLM-4 uses NeoX (not interleaved) RoPE");

    /* Derived dims. */
    cr_assert_eq(cfg.head_dim, 128, "derived head_dim = 4096/32");
    cr_assert_eq(cfg.kv_head_dim, 128, "derived kv_head_dim");
    cr_assert_eq(cfg.rope_dim, 128, "derived rope_dim");

    oc_gguf_free(&f);
}

Test(glm_arch, config_parse_hunyuan_metadata, .disabled=true)
{
    const char *keys[] = {
        "hunyuan.vocab_size",
        "hunyuan.hidden_size",
        "hunyuan.num_layers",
        "hunyuan.num_attention_heads",
        "hunyuan.num_key_value_heads",
        "hunyuan.intermediate_size",
        "hunyuan.max_position_embeddings",
        "hunyuan.rope_theta",
        "hunyuan.rms_norm_eps",
        "hunyuan.num_experts",
        "hunyuan.num_experts_per_tok",
        "hunyuan.expert_intermediate_size",
        "hunyuan.moe_layer_start",
        "hunyuan.shared_expert_intermediate_size",
        "hunyuan.use_mla",
    };
    const uint32_t types[] = {
        4, 4, 4, 4, 4, 4, 4,   /* u32 */
        6, 6,                   /* f32 */
        4, 4, 4, 4, 4,         /* u32 */
        7,                      /* bool */
    };
    const uint32_t u32_vals[] = {
        128512,   /* vocab_size */
        4096,     /* hidden_size */
        32,       /* num_layers */
        32,       /* num_attention_heads */
        8,        /* num_key_value_heads */
        11008,    /* intermediate_size */
        32768,    /* max_position_embeddings */
        0, 0,     /* (rope_theta, rms_norm_eps are f32) */
        64,       /* num_experts */
        8,        /* num_experts_per_tok */
        7168,     /* expert_intermediate_size */
        1,        /* moe_layer_start */
        16384,    /* shared_expert_intermediate_size */
        0,        /* use_mla = false */
    };
    const float f32_vals[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        10000.0f,  /* rope_theta */
        1e-5f,     /* rms_norm_eps */
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    OcGgufFile f = build_minimal_gguf(keys, types, u32_vals, f32_vals, 15);

    OcHunyuanConfig cfg;
    OcError e = oc_hunyuan_config_parse(&f, "hunyuan_moe", &cfg);
    cr_assert_eq(e, OC_OK, "oc_hunyuan_config_parse should succeed");

    cr_assert_eq(cfg.vocab_size, 128512, "parsed vocab_size");
    cr_assert_eq(cfg.hidden_size, 4096, "parsed hidden_size");
    cr_assert_eq(cfg.n_layer, 32, "parsed n_layer");
    cr_assert_eq(cfg.num_attention_heads, 32, "parsed num_attention_heads");
    cr_assert_eq(cfg.num_kv_heads, 8, "parsed num_kv_heads");
    cr_assert_eq(cfg.intermediate_size, 11008, "parsed intermediate_size");
    cr_assert_eq(cfg.max_position_embeddings, 32768, "parsed max_position");
    cr_assert_float_eq(cfg.rope_theta, 10000.0f, 1e-1f, "parsed rope_theta");
    cr_assert_eq(cfg.n_routed_experts, 64, "parsed n_routed_experts");
    cr_assert_eq(cfg.n_active_experts, 8, "parsed n_active_experts");
    cr_assert_eq(cfg.expert_intermediate_size, 7168,
        "parsed expert_intermediate_size");
    cr_assert_eq(cfg.moe_layer_start, 1, "parsed moe_layer_start");
    cr_assert_eq(cfg.has_shared_expert, true, "parsed has_shared_expert");
    cr_assert_eq(cfg.shared_expert_intermediate_size, 16384,
        "parsed shared_expert_intermediate_size");
    cr_assert_eq(cfg.uses_mla, false, "parsed uses_mla");

    /* Derived dims. */
    cr_assert_eq(cfg.head_dim, 128, "derived head_dim");
    cr_assert_eq(cfg.kv_head_dim, 128, "derived kv_head_dim");

    oc_gguf_free(&f);
}

Test(glm_arch, config_parse_glm_invalid_dims, .disabled=true)
{
    /* hidden_size not divisible by num_attention_heads → OC_ERR_MODEL. */
    const char *keys[] = {
        "glm.hidden_size",
        "glm.num_attention_heads",
        "glm.num_key_value_heads",
        "glm.max_position_embeddings",
    };
    const uint32_t types[] = { 4, 4, 4, 4 };
    const uint32_t u32_vals[] = {
        4097,   /* hidden_size (not divisible by num_attention_heads) */
        32,     /* num_attention_heads */
        32,     /* num_key_value_heads */
        32768,  /* max_position_embeddings */
    };
    const float f32_vals[] = { 0.0f, 0.0f, 0.0f, 0.0f };

    OcGgufFile f = build_minimal_gguf(keys, types, u32_vals, f32_vals, 4);

    OcGlmConfig cfg;
    OcError e = oc_glm_config_parse(&f, "glm", &cfg);
    cr_assert_eq(e, OC_ERR_MODEL, "invalid dims → OC_ERR_MODEL");

    oc_gguf_free(&f);
}

Test(glm_arch, config_parse_hunyuan_mla_dims, .disabled=true)
{
    /* Hunyuan-Large with MLA config. */
    const char *keys[] = {
        "hunyuan.hidden_size",
        "hunyuan.num_layers",
        "hunyuan.num_attention_heads",
        "hunyuan.num_key_value_heads",
        "hunyuan.max_position_embeddings",
        "hunyuan.use_mla",
        "hunyuan.attention.q_lora_rank",
        "hunyuan.attention.kv_lora_rank",
        "hunyuan.attention.key_length",
        "hunyuan.attention.key_length_rope",
        "hunyuan.attention.value_length",
    };
    const uint32_t types[] = { 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4 };
    const uint32_t u32_vals[] = {
        6144,   /* hidden_size */
        32,     /* num_layers */
        48,     /* num_attention_heads */
        48,     /* num_key_value_heads (GQA with 1 group) */
        32768,  /* max_position_embeddings */
        1,      /* use_mla = true */
        1536,   /* q_lora_rank */
        512,    /* kv_lora_rank */
        192,    /* key_length (mla_q_head_dim) */
        64,     /* key_length_rope */
        128,    /* value_length */
    };
    const float f32_vals[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    OcGgufFile f = build_minimal_gguf(keys, types, u32_vals, f32_vals, 11);

    OcHunyuanConfig cfg;
    OcError e = oc_hunyuan_config_parse(&f, "hunyuan", &cfg);
    cr_assert_eq(e, OC_OK, "oc_hunyuan_config_parse with MLA should succeed");

    cr_assert_eq(cfg.uses_mla, true, "uses_mla");
    cr_assert_eq(cfg.mla_q_lora_dim, 1536, "mla_q_lora_dim");
    cr_assert_eq(cfg.mla_kv_lora_dim, 512, "mla_kv_lora_dim");
    cr_assert_eq(cfg.mla_q_head_dim, 192, "mla_q_head_dim");
    cr_assert_eq(cfg.mla_q_rope_dim, 64, "mla_q_rope_dim");
    cr_assert_eq(cfg.mla_q_nope_dim, 128, "mla_q_nope_dim = 192 - 64");
    cr_assert_eq(cfg.mla_v_head_dim, 128, "mla_v_head_dim");

    oc_gguf_free(&f);
}

Test(glm_arch, config_parse_glm_defaults_for_missing_keys, .disabled=true)
{
    /* Empty GGUF (no GLM keys) → all defaults except hidden_size etc.
     * that we set. We provide only the architecture string. */
    const char *keys[] = {
        "glm.hidden_size",
        "glm.num_attention_heads",
        "glm.num_key_value_heads",
        "glm.max_position_embeddings",
    };
    const uint32_t types[] = { 4, 4, 4, 4 };
    const uint32_t u32_vals[] = {
        4096,   /* hidden_size */
        32,     /* num_attention_heads */
        32,     /* num_key_value_heads */
        4096,   /* max_position_embeddings */
    };
    const float f32_vals[] = { 0.0f, 0.0f, 0.0f, 0.0f };

    OcGgufFile f = build_minimal_gguf(keys, types, u32_vals, f32_vals, 4);

    OcGlmConfig cfg;
    OcError e = oc_glm_config_parse(&f, "glm", &cfg);
    cr_assert_eq(e, OC_OK, "parse with missing keys should use defaults");

    /* vocab_size falls back to default (32000) since not in GGUF. */
    cr_assert_eq(cfg.vocab_size, 32000, "default vocab_size when missing");
    cr_assert_eq(cfg.hidden_size, 4096, "parsed hidden_size");
    cr_assert_eq(cfg.max_position_embeddings, 4096, "parsed max_position");
    /* rope_theta defaults to 10000 for GLM version 1 (< 4). */
    cr_assert_float_eq(cfg.rope_theta, 10000.0f, 1e-1f,
        "default rope_theta for ChatGLM");
    cr_assert_eq(cfg.glm_version, OC_GLM_VERSION_1, "version 1 from 'glm'");
    cr_assert_eq(cfg.uses_interleaved_rope, true,
        "ChatGLM-6B uses interleaved RoPE");

    oc_gguf_free(&f);
}

Test(glm_arch, arch_uses_mla_for_glm)
{
    /* The architecture-level MLA flag: GLM-MoE-DSA uses MLA. */
    cr_assert_eq(oc_model_arch_uses_mla(OC_ARCH_GLM_MOE_DSA), true,
        "GLM_MOE_DSA uses MLA");
    cr_assert_eq(oc_model_arch_uses_mla(OC_ARCH_HUNYUAN_MOE), false,
        "Hunyuan-MoE does not use MLA by default (only Hunyuan-Large)");
}

Test(glm_arch, arch_uses_moe_for_hunyuan)
{
    cr_assert_eq(oc_model_arch_uses_moe(OC_ARCH_HUNYUAN_MOE), true,
        "Hunyuan-MoE is a MoE architecture");
    cr_assert_eq(oc_model_arch_uses_moe(OC_ARCH_GLM_MOE_DSA), true,
        "GLM-MoE-DSA is a MoE architecture");
}

Test(glm_arch, arch_name_strings)
{
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_GLM_MOE_DSA), "glm_moe_dsa",
        "GLM arch name");
    cr_assert_str_eq(oc_model_arch_name(OC_ARCH_HUNYUAN_MOE), "hunyuan_moe",
        "Hunyuan arch name");
}
