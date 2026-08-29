/*
 * glm_arch.c — GLM (ChatGLM/Zhipu) and Hunyuan configuration parsing.
 *
 * Implements:
 *   - GLM config parsing from GGUF metadata (oc_glm_config_parse)
 *   - Hunyuan config parsing from GGUF metadata (oc_hunyuan_config_parse)
 *   - Version-string mapping (oc_glm_version_from_str)
 *
 * Dedicated oc_arch_forward_glm / oc_arch_forward_hunyuan passes were
 * removed: no loader, dispatcher, or test ever reached them. GLM/Hunyuan
 * inference runs through the llama.c session paths. Config parsing and
 * arch-registry entries they fed are kept.
 */
#include "oxidize/glm_arch.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "oxidize/gguf.h"

static uint32_t glm_cfg_u32(const OcGgufFile *f, const char *key, uint32_t def)
{
    uint32_t v;
    return oc_gguf_metadata_get_u32(f, key, &v) ? v : def;
}

static float glm_cfg_f32(const OcGgufFile *f, const char *key, float def)
{
    float v;
    return oc_gguf_metadata_get_f32(f, key, &v) ? v : def;
}

static bool glm_cfg_bool(const OcGgufFile *f, const char *key, bool def)
{
    bool v;
    return oc_gguf_metadata_get_bool(f, key, &v) ? v : def;
}

static uint32_t glm_key_u32(const OcGgufFile *f, const char *prefix,
                           const char *suffix, uint32_t def)
{
    char key[192];
    snprintf(key, sizeof(key), "%s%s", prefix, suffix);
    return glm_cfg_u32(f, key, def);
}

static float glm_key_f32(const OcGgufFile *f, const char *prefix,
                        const char *suffix, float def)
{
    char key[192];
    snprintf(key, sizeof(key), "%s%s", prefix, suffix);
    return glm_cfg_f32(f, key, def);
}

static bool glm_key_bool(const OcGgufFile *f, const char *prefix,
                        const char *suffix, bool def)
{
    char key[192];
    snprintf(key, sizeof(key), "%s%s", prefix, suffix);
    return glm_cfg_bool(f, key, def);
}

static void glm_norm_prefix(char *prefix, size_t n, const char *arch,
                            const char *fallback)
{
    const char *p = (arch != NULL) ? arch : fallback;
    size_t pi = 0;
    for (; pi + 2 < n && p[pi]; pi++) {
        char c = p[pi];
        if (c == '-')
            c = '_';
        else
            c = (char)tolower((unsigned char)c);
        prefix[pi] = c;
    }
    prefix[pi] = '.';
    prefix[pi + 1] = '\0';
}

static OcError glm_derive_head_dims(uint32_t n_head, uint32_t n_kv,
                                    uint32_t hidden, uint32_t key_len,
                                    uint32_t rope_dim, uint32_t *head_dim,
                                    uint32_t *kv_head_dim, uint32_t *rope_out)
{
    if (n_head == 0 || n_kv == 0 || hidden == 0 || n_kv > n_head
        || n_head % n_kv != 0 || hidden % n_head != 0)
        return OC_ERR_MODEL;
    *head_dim = (key_len > 0) ? key_len : (hidden / n_head);
    *kv_head_dim = *head_dim;
    *rope_out = (rope_dim > 0) ? rope_dim : *kv_head_dim;
    if (*rope_out > *kv_head_dim)
        *rope_out = *kv_head_dim;
    return OC_OK;
}

/* ─── GLM version detection ───────────────────────────────────────────── */

OcGlmVersion oc_glm_version_from_str(const char *s)
{
    if (!s || !*s) return OC_GLM_VERSION_UNKNOWN;

    /* Normalize: lowercase + '-' → '_' in a stack buffer. */
    char norm[32];
    size_t n = 0;
    for (; n < sizeof(norm) - 1 && s[n]; n++) {
        char c = s[n];
        if (c == '-') c = '_';
        else c = (char)tolower((unsigned char)c);
        norm[n] = c;
    }
    norm[n] = '\0';
    if (n == sizeof(norm) - 1 && s[n]) return OC_GLM_VERSION_UNKNOWN;

    /* Strip optional "chat" prefix ("chatglm" → "glm"). */
    const char *p = norm;
    if (strncmp(p, "chat", 4) == 0) p += 4;

    if (strcmp(p, "glm") == 0)       return OC_GLM_VERSION_1;
    if (strcmp(p, "glm2") == 0)      return OC_GLM_VERSION_2;
    if (strcmp(p, "glm3") == 0)      return OC_GLM_VERSION_3;
    if (strcmp(p, "glm4") == 0
        || strcmp(p, "glm_4") == 0)  return OC_GLM_VERSION_4;

    /* "glm_moe" / "glm_moe_dsa" / "glm_dsa" → GLM-4 (the MoE variants are
     * built on GLM-4). The actual MoE forward is handled by Hunyuan or
     * DeepSeek paths; here we just report the version. */
    if (strncmp(p, "glm", 3) == 0)  return OC_GLM_VERSION_4;

    return OC_GLM_VERSION_UNKNOWN;
}

/* ─── Config defaults ─────────────────────────────────────────────────── */

void oc_glm_config_defaults(OcGlmConfig *cfg)
{
    if (cfg == NULL) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->vocab_size           = 32000;
    cfg->hidden_size          = 4096;
    cfg->n_layer              = 28;
    cfg->num_attention_heads  = 32;
    cfg->num_kv_heads         = 32;
    cfg->intermediate_size    = 11008;
    cfg->max_position_embeddings = 32768;
    cfg->head_dim             = 128;
    cfg->kv_head_dim          = 128;
    cfg->rope_dim             = 128;
    cfg->rope_theta           = 10000.0f;
    cfg->rms_norm_eps         = 1e-5f;
    cfg->uses_mla             = false;
    cfg->apply_qk_norm        = false;
    cfg->uses_interleaved_rope = false;
    cfg->tied_embeddings      = false;
    cfg->glm_version          = OC_GLM_VERSION_UNKNOWN;
}

void oc_hunyuan_config_defaults(OcHunyuanConfig *cfg)
{
    if (cfg == NULL) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->vocab_size           = 32000;
    cfg->hidden_size          = 4096;
    cfg->n_layer              = 32;
    cfg->num_attention_heads  = 32;
    cfg->num_kv_heads         = 8;
    cfg->intermediate_size    = 11008;
    cfg->max_position_embeddings = 32768;
    cfg->head_dim             = 128;
    cfg->kv_head_dim          = 128;
    cfg->rope_dim             = 128;
    cfg->rope_theta           = 10000.0f;
    cfg->rms_norm_eps         = 1e-5f;
    cfg->n_routed_experts     = 0;
    cfg->n_active_experts     = 0;
    cfg->expert_intermediate_size = 0;
    cfg->moe_layer_start      = 0;
    cfg->has_shared_expert    = false;
    cfg->shared_expert_intermediate_size = 0;
    cfg->uses_mla             = false;
    cfg->mla_q_lora_dim      = 0;
    cfg->mla_kv_lora_dim     = 0;
    cfg->mla_q_head_dim      = 0;
    cfg->mla_q_rope_dim     = 0;
    cfg->mla_q_nope_dim     = 0;
    cfg->mla_v_head_dim     = 0;
    cfg->tied_embeddings     = false;
}

/* ─── GLM config parsing ───────────────────────────────────────────────── */

OcError oc_glm_config_parse(const OcGgufFile *f, const char *arch_str,
                             OcGlmConfig *cfg)
{
    if (f == NULL || cfg == NULL) return OC_ERR_INVALID_ARG;

    oc_glm_config_defaults(cfg);

    /* GLM GGUFs use either "glm." or "chatglm." — prefer the architecture
     * string when it is a known GLM variant; fall back to "glm." */
    char prefix[64];
    glm_norm_prefix(prefix, sizeof(prefix), arch_str, "glm");
    if (strncmp(prefix, "chatglm.", 8) == 0)
        memmove(prefix, prefix + 4, strlen(prefix + 4) + 1);

    cfg->glm_version = oc_glm_version_from_str(arch_str);

    cfg->vocab_size = glm_key_u32(f, prefix, "vocab_size", cfg->vocab_size);
    if (cfg->vocab_size == 32000) {
        uint32_t gv;
        if (oc_gguf_metadata_get_u32(f, "general.vocab_size", &gv))
            cfg->vocab_size = gv;
    }

    cfg->hidden_size = glm_key_u32(f, prefix, "hidden_size", cfg->hidden_size);
    cfg->n_layer = glm_key_u32(f, prefix, "num_layers", cfg->n_layer);
    cfg->num_attention_heads = glm_key_u32(f, prefix, "num_attention_heads",
                                           cfg->num_attention_heads);
    cfg->num_kv_heads = glm_key_u32(f, prefix, "num_key_value_heads",
                                    cfg->num_attention_heads);
    cfg->intermediate_size = glm_key_u32(f, prefix, "intermediate_size",
                                         cfg->intermediate_size);
    cfg->max_position_embeddings = glm_key_u32(f, prefix,
                                               "max_position_embeddings",
                                               cfg->max_position_embeddings);
    cfg->rope_theta = glm_key_f32(f, prefix, "rope_theta",
                                  (cfg->glm_version >= OC_GLM_VERSION_4)
                                      ? 500000.0f : 10000.0f);
    cfg->rms_norm_eps = glm_key_f32(f, prefix, "rms_norm_eps", 1e-5f);
    uint32_t key_len = glm_key_u32(f, prefix, "attention.key_length", 0);
    uint32_t rope_dim = glm_key_u32(f, prefix, "rope.dimension_count", 0);
    cfg->apply_qk_norm = glm_key_bool(f, prefix, "apply_qk_norm",
                                      cfg->glm_version >= OC_GLM_VERSION_4);
    cfg->tied_embeddings = glm_key_bool(f, prefix, "tied_word_embeddings",
                                        false);
    /* MLA is not used by GLM-4 itself; GLM-MoE-DSA is routed through the
     * DeepSeek path. Parse the flag for completeness. */
    cfg->uses_mla = glm_key_bool(f, prefix, "use_mla", false);

    OcError dim_err = glm_derive_head_dims(cfg->num_attention_heads,
                                           cfg->num_kv_heads, cfg->hidden_size,
                                           key_len, rope_dim, &cfg->head_dim,
                                           &cfg->kv_head_dim, &cfg->rope_dim);
    if (dim_err != OC_OK)
        return dim_err;

    cfg->uses_interleaved_rope = (cfg->glm_version == OC_GLM_VERSION_1);
    if (cfg->max_position_embeddings == 0)
        return OC_ERR_MODEL;
    return OC_OK;
}

/* ─── Hunyuan config parsing ──────────────────────────────────────────── */

OcError oc_hunyuan_config_parse(const OcGgufFile *f, const char *arch_str,
                                 OcHunyuanConfig *cfg)
{
    if (f == NULL || cfg == NULL) return OC_ERR_INVALID_ARG;

    oc_hunyuan_config_defaults(cfg);

    /* "hunyuan-moe" → "hunyuan_moe."; converter keys use the bare
     * "hunyuan." namespace, so strip variant suffixes. */
    char prefix[64];
    glm_norm_prefix(prefix, sizeof(prefix), arch_str, "hunyuan");
    if (strncmp(prefix, "hunyuan_", 8) == 0) {
        prefix[7] = '.';
        prefix[8] = '\0';
    }

    cfg->vocab_size = glm_key_u32(f, prefix, "vocab_size", cfg->vocab_size);
    if (cfg->vocab_size == 32000) {
        uint32_t gv;
        if (oc_gguf_metadata_get_u32(f, "general.vocab_size", &gv))
            cfg->vocab_size = gv;
    }

    cfg->hidden_size = glm_key_u32(f, prefix, "hidden_size", cfg->hidden_size);
    cfg->n_layer = glm_key_u32(f, prefix, "num_layers", cfg->n_layer);
    cfg->num_attention_heads = glm_key_u32(f, prefix, "num_attention_heads",
                                           cfg->num_attention_heads);
    cfg->num_kv_heads = glm_key_u32(f, prefix, "num_key_value_heads",
                                    cfg->num_attention_heads);
    cfg->intermediate_size = glm_key_u32(f, prefix, "intermediate_size",
                                         cfg->intermediate_size);
    cfg->max_position_embeddings = glm_key_u32(f, prefix,
                                               "max_position_embeddings",
                                               cfg->max_position_embeddings);
    cfg->rope_theta = glm_key_f32(f, prefix, "rope_theta", 10000.0f);
    cfg->rms_norm_eps = glm_key_f32(f, prefix, "rms_norm_eps", 1e-5f);
    uint32_t key_len = glm_key_u32(f, prefix, "attention.key_length", 0);
    uint32_t rope_dim = glm_key_u32(f, prefix, "rope.dimension_count", 0);
    cfg->n_routed_experts = glm_key_u32(f, prefix, "num_experts", 0);
    cfg->n_active_experts = glm_key_u32(f, prefix, "num_experts_per_tok", 0);
    cfg->expert_intermediate_size = glm_key_u32(f, prefix,
                                                "expert_intermediate_size", 0);
    cfg->moe_layer_start = glm_key_u32(f, prefix, "moe_layer_start", 0);
    uint32_t shexp_size = glm_key_u32(f, prefix,
                                      "shared_expert_intermediate_size", 0);
    cfg->has_shared_expert = (shexp_size > 0);
    cfg->shared_expert_intermediate_size = shexp_size;
    cfg->uses_mla = glm_key_bool(f, prefix, "use_mla", false);
    cfg->mla_q_lora_dim = glm_key_u32(f, prefix, "attention.q_lora_rank", 0);
    cfg->mla_kv_lora_dim = glm_key_u32(f, prefix, "attention.kv_lora_rank", 0);
    uint32_t mla_key_len = glm_key_u32(f, prefix, "attention.key_length", 0);
    cfg->mla_q_rope_dim = glm_key_u32(f, prefix, "attention.key_length_rope", 0);
    cfg->mla_v_head_dim = glm_key_u32(f, prefix, "attention.value_length", 0);
    cfg->tied_embeddings = glm_key_bool(f, prefix, "tied_word_embeddings",
                                        false);

    OcError dim_err = glm_derive_head_dims(cfg->num_attention_heads,
                                           cfg->num_kv_heads, cfg->hidden_size,
                                           key_len, rope_dim, &cfg->head_dim,
                                           &cfg->kv_head_dim, &cfg->rope_dim);
    if (dim_err != OC_OK)
        return dim_err;

    if (cfg->uses_mla) {
        if (mla_key_len > 0)
            cfg->mla_q_head_dim = mla_key_len;
        if (cfg->mla_q_rope_dim > cfg->mla_q_head_dim)
            cfg->mla_q_rope_dim = cfg->mla_q_head_dim;
        cfg->mla_q_nope_dim = cfg->mla_q_head_dim - cfg->mla_q_rope_dim;
        if (cfg->mla_v_head_dim == 0)
            cfg->mla_v_head_dim = cfg->mla_q_nope_dim;
    }

    if (cfg->n_routed_experts > 0) {
        if (cfg->n_active_experts == 0)
            cfg->n_active_experts = 1;
        if (cfg->n_active_experts > cfg->n_routed_experts)
            cfg->n_active_experts = cfg->n_routed_experts;
        if (cfg->expert_intermediate_size == 0)
            cfg->expert_intermediate_size = cfg->intermediate_size;
    }
    if (cfg->moe_layer_start > cfg->n_layer)
        cfg->moe_layer_start = cfg->n_layer;
    if (cfg->max_position_embeddings == 0)
        return OC_ERR_MODEL;
    return OC_OK;
}
