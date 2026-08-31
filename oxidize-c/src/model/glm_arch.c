/*
 * glm_arch.c — GLM (ChatGLM/Zhipu) and Hunyuan configuration parsing.
 *
 * Implements:
 *   - GLM config parsing from GGUF metadata (oc_glm_config_parse)
 *   - Hunyuan config parsing from GGUF metadata (oc_hunyuan_config_parse)
 *   - Version-string mapping (oc_glm_version_from_str)
 *
 * The historical oc_arch_forward_glm / oc_arch_forward_hunyuan forward
 * passes were removed: no loader, dispatcher, or test ever reached them
 * (GLM/Hunyuan inference runs through the llama.c session paths). The
 * config parsing and arch-registry entries they fed are kept.
 */
#include "oxidize/glm_arch.h"

#include <ctype.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/gguf.h"


/* ─── Helpers ────────────────────────────────────────────────────────────
 *
 * These mirror the static helpers in arch_forward.c / llama.c. We keep
 * local copies to remain self-contained (the llama.c originals are static).
 */

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

    /* Determine the metadata key prefix. GLM GGUFs use either "glm." or
     * "chatglm." — prefer the architecture string when it is one of the
     * known GLM variants; fall back to "glm." */
    char prefix[64];
    const char *p = (arch_str != NULL) ? arch_str : "glm";
    /* Normalize the arch string for the prefix (lowercase, '-' → '_'). */
    size_t pi = 0;
    for (; pi < sizeof(prefix) - 2 && p[pi]; pi++) {
        char c = p[pi];
        if (c == '-') c = '_';
        else c = (char)tolower((unsigned char)c);
        prefix[pi] = c;
    }
    prefix[pi] = '.';
    prefix[pi + 1] = '\0';

    /* If the normalized arch starts with "chatglm", strip "chat" → "glm". */
    if (strncmp(prefix, "chatglm.", 8) == 0) {
        memmove(prefix, prefix + 4, strlen(prefix + 4) + 1);
    }

    /* Detect version from the arch string (overrides the default). */
    cfg->glm_version = oc_glm_version_from_str(arch_str);

    char key[192];

    snprintf(key, sizeof(key), "%svocab_size", prefix);
    cfg->vocab_size = glm_cfg_u32(f, key, cfg->vocab_size);
    /* Fall back to general.vocab_size. */
    if (cfg->vocab_size == 32000) {
        uint32_t gv;
        if (oc_gguf_metadata_get_u32(f, "general.vocab_size", &gv)) {
            cfg->vocab_size = gv;
        }
    }

    snprintf(key, sizeof(key), "%shidden_size", prefix);
    cfg->hidden_size = glm_cfg_u32(f, key, cfg->hidden_size);

    snprintf(key, sizeof(key), "%snum_layers", prefix);
    cfg->n_layer = glm_cfg_u32(f, key, cfg->n_layer);

    snprintf(key, sizeof(key), "%snum_attention_heads", prefix);
    cfg->num_attention_heads = glm_cfg_u32(f, key, cfg->num_attention_heads);

    snprintf(key, sizeof(key), "%snum_key_value_heads", prefix);
    cfg->num_kv_heads = glm_cfg_u32(f, key, cfg->num_attention_heads);

    snprintf(key, sizeof(key), "%sintermediate_size", prefix);
    cfg->intermediate_size = glm_cfg_u32(f, key, cfg->intermediate_size);

    snprintf(key, sizeof(key), "%smax_position_embeddings", prefix);
    cfg->max_position_embeddings = glm_cfg_u32(f, key,
                                                 cfg->max_position_embeddings);

    snprintf(key, sizeof(key), "%srope_theta", prefix);
    cfg->rope_theta = glm_cfg_f32(f, key,
                                   (cfg->glm_version >= OC_GLM_VERSION_4)
                                       ? 500000.0f : 10000.0f);

    snprintf(key, sizeof(key), "%srms_norm_eps", prefix);
    cfg->rms_norm_eps = glm_cfg_f32(f, key, 1e-5f);

    snprintf(key, sizeof(key), "%sattention.key_length", prefix);
    uint32_t key_len = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%srope.dimension_count", prefix);
    uint32_t rope_dim = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%sapply_qk_norm", prefix);
    cfg->apply_qk_norm = glm_cfg_bool(f, key,
        (cfg->glm_version >= OC_GLM_VERSION_4));

    snprintf(key, sizeof(key), "%stied_word_embeddings", prefix);
    bool tied = glm_cfg_bool(f, key, false);
    cfg->tied_embeddings = tied;

    /* MLA is not used by GLM-4 itself; the GLM-MoE-DSA variant uses MLA but
     * is routed through the DeepSeek forward path. We still parse the flag
     * for completeness. */
    snprintf(key, sizeof(key), "%suse_mla", prefix);
    cfg->uses_mla = glm_cfg_bool(f, key, false);

    /* Derived dims. */
    if (cfg->num_attention_heads == 0 || cfg->num_kv_heads == 0
        || cfg->hidden_size == 0
        || cfg->num_kv_heads > cfg->num_attention_heads
        || cfg->num_attention_heads % cfg->num_kv_heads != 0
        || cfg->hidden_size % cfg->num_attention_heads != 0) {
        return OC_ERR_MODEL;
    }
    cfg->head_dim = (key_len > 0) ? key_len
                                  : (cfg->hidden_size / cfg->num_attention_heads);
    cfg->kv_head_dim = cfg->head_dim;
    cfg->rope_dim = (rope_dim > 0) ? rope_dim : cfg->kv_head_dim;
    if (cfg->rope_dim > cfg->kv_head_dim) cfg->rope_dim = cfg->kv_head_dim;

    /* ChatGLM-6B (version 1) uses interleaved (GPT-J style) RoPE. */
    cfg->uses_interleaved_rope = (cfg->glm_version == OC_GLM_VERSION_1);

    /* Validate max_position_embeddings. */
    if (cfg->max_position_embeddings == 0) {
        return OC_ERR_MODEL;
    }

    return OC_OK;
}

/* ─── Hunyuan config parsing ──────────────────────────────────────────── */

OcError oc_hunyuan_config_parse(const OcGgufFile *f, const char *arch_str,
                                 OcHunyuanConfig *cfg)
{
    if (f == NULL || cfg == NULL) return OC_ERR_INVALID_ARG;

    oc_hunyuan_config_defaults(cfg);

    /* Build the prefix. Normalize the arch string: "hunyuan-moe" →
     * "hunyuan_moe", "hunyuan_v3" → "hunyuan_v3". Use the bare "hunyuan."
     * prefix for GGUF keys (the converter strips variant suffixes). */
    char prefix[64];
    const char *p = (arch_str != NULL) ? arch_str : "hunyuan";
    size_t pi = 0;
    for (; pi < sizeof(prefix) - 2 && p[pi]; pi++) {
        char c = p[pi];
        if (c == '-') c = '_';
        else c = (char)tolower((unsigned char)c);
        prefix[pi] = c;
    }
    prefix[pi] = '.';
    prefix[pi + 1] = '\0';

    /* If the prefix has a variant suffix (e.g. "hunyuan_moe.", "hunyuan_v3."),
     * strip it back to "hunyuan." for the GGUF key namespace. */
    if (strncmp(prefix, "hunyuan_", 8) == 0) {
        prefix[7] = '.';
        prefix[8] = '\0';
    }

    char key[192];

    snprintf(key, sizeof(key), "%svocab_size", prefix);
    cfg->vocab_size = glm_cfg_u32(f, key, cfg->vocab_size);
    if (cfg->vocab_size == 32000) {
        uint32_t gv;
        if (oc_gguf_metadata_get_u32(f, "general.vocab_size", &gv)) {
            cfg->vocab_size = gv;
        }
    }

    snprintf(key, sizeof(key), "%shidden_size", prefix);
    cfg->hidden_size = glm_cfg_u32(f, key, cfg->hidden_size);

    snprintf(key, sizeof(key), "%snum_layers", prefix);
    cfg->n_layer = glm_cfg_u32(f, key, cfg->n_layer);

    snprintf(key, sizeof(key), "%snum_attention_heads", prefix);
    cfg->num_attention_heads = glm_cfg_u32(f, key, cfg->num_attention_heads);

    snprintf(key, sizeof(key), "%snum_key_value_heads", prefix);
    cfg->num_kv_heads = glm_cfg_u32(f, key, cfg->num_attention_heads);

    snprintf(key, sizeof(key), "%sintermediate_size", prefix);
    cfg->intermediate_size = glm_cfg_u32(f, key, cfg->intermediate_size);

    snprintf(key, sizeof(key), "%smax_position_embeddings", prefix);
    cfg->max_position_embeddings = glm_cfg_u32(f, key,
                                                 cfg->max_position_embeddings);

    snprintf(key, sizeof(key), "%srope_theta", prefix);
    cfg->rope_theta = glm_cfg_f32(f, key, 10000.0f);

    snprintf(key, sizeof(key), "%srms_norm_eps", prefix);
    cfg->rms_norm_eps = glm_cfg_f32(f, key, 1e-5f);

    snprintf(key, sizeof(key), "%sattention.key_length", prefix);
    uint32_t key_len = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%srope.dimension_count", prefix);
    uint32_t rope_dim = glm_cfg_u32(f, key, 0);

    /* MoE fields. */
    snprintf(key, sizeof(key), "%snum_experts", prefix);
    cfg->n_routed_experts = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%snum_experts_per_tok", prefix);
    cfg->n_active_experts = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%sexpert_intermediate_size", prefix);
    cfg->expert_intermediate_size = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%smoe_layer_start", prefix);
    cfg->moe_layer_start = glm_cfg_u32(f, key, 0);

    /* Shared expert. */
    snprintf(key, sizeof(key), "%sshared_expert_intermediate_size", prefix);
    uint32_t shexp_size = glm_cfg_u32(f, key, 0);
    cfg->has_shared_expert = (shexp_size > 0);
    cfg->shared_expert_intermediate_size = shexp_size;

    /* MLA fields (Hunyuan-Large). */
    snprintf(key, sizeof(key), "%suse_mla", prefix);
    cfg->uses_mla = glm_cfg_bool(f, key, false);

    snprintf(key, sizeof(key), "%sattention.q_lora_rank", prefix);
    cfg->mla_q_lora_dim = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%sattention.kv_lora_rank", prefix);
    cfg->mla_kv_lora_dim = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%sattention.key_length", prefix);
    uint32_t mla_key_len = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%sattention.key_length_rope", prefix);
    cfg->mla_q_rope_dim = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%sattention.value_length", prefix);
    cfg->mla_v_head_dim = glm_cfg_u32(f, key, 0);

    snprintf(key, sizeof(key), "%stied_word_embeddings", prefix);
    cfg->tied_embeddings = glm_cfg_bool(f, key, false);

    /* Derived dims. */
    if (cfg->num_attention_heads == 0 || cfg->num_kv_heads == 0
        || cfg->hidden_size == 0
        || cfg->num_kv_heads > cfg->num_attention_heads
        || cfg->num_attention_heads % cfg->num_kv_heads != 0
        || cfg->hidden_size % cfg->num_attention_heads != 0) {
        return OC_ERR_MODEL;
    }
    cfg->head_dim = (key_len > 0) ? key_len
                                  : (cfg->hidden_size / cfg->num_attention_heads);
    cfg->kv_head_dim = cfg->head_dim;
    cfg->rope_dim = (rope_dim > 0) ? rope_dim : cfg->kv_head_dim;
    if (cfg->rope_dim > cfg->kv_head_dim) cfg->rope_dim = cfg->kv_head_dim;

    /* MLA derived dims. */
    if (cfg->uses_mla) {
        if (mla_key_len > 0) {
            cfg->mla_q_head_dim = mla_key_len;
        }
        if (cfg->mla_q_rope_dim > cfg->mla_q_head_dim) {
            cfg->mla_q_rope_dim = cfg->mla_q_head_dim;
        }
        cfg->mla_q_nope_dim = cfg->mla_q_head_dim - cfg->mla_q_rope_dim;
        if (cfg->mla_v_head_dim == 0) {
            cfg->mla_v_head_dim = cfg->mla_q_nope_dim;
        }
    }

    /* MoE sanity. */
    if (cfg->n_routed_experts > 0) {
        if (cfg->n_active_experts == 0) cfg->n_active_experts = 1;
        if (cfg->n_active_experts > cfg->n_routed_experts) {
            cfg->n_active_experts = cfg->n_routed_experts;
        }
        if (cfg->expert_intermediate_size == 0) {
            cfg->expert_intermediate_size = cfg->intermediate_size;
        }
    }
    if (cfg->moe_layer_start > cfg->n_layer) {
        cfg->moe_layer_start = cfg->n_layer;
    }
    if (cfg->max_position_embeddings == 0) {
        return OC_ERR_MODEL;
    }
    return OC_OK;
}
