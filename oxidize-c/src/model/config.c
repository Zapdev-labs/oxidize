#include "oxidize/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Typed convenience getters mirroring llama.c's cfg_u32/cfg_f32/cfg_str. */
static uint32_t cfg_u32(const OcGgufFile *f, const char *key, uint32_t def)
{
    uint32_t v;
    return oc_gguf_metadata_get_u32(f, key, &v) ? v : def;
}

static float cfg_f32(const OcGgufFile *f, const char *key, float def)
{
    float v;
    return oc_gguf_metadata_get_f32(f, key, &v) ? v : def;
}

static double cfg_f64(const OcGgufFile *f, const char *key, double def)
{
    double v;
    return oc_gguf_metadata_get_f64(f, key, &v) ? v : def;
}

static const char *cfg_str(const OcGgufFile *f, const char *key, const char *def)
{
    const char *v = NULL;
    return oc_gguf_metadata_get_str(f, key, &v, NULL) ? v : def;
}

/* Copy at most dst_cap-1 chars from `src` into `dst`, NUL-terminating. */
static void set_str(char *dst, size_t dst_cap, const char *src)
{
    if (dst_cap == 0) return;
    if (src == NULL) { dst[0] = '\0'; return; }
    strncpy(dst, src, dst_cap - 1);
    dst[dst_cap - 1] = '\0';
}


OcError oc_model_config_init(OcModelConfig *cfg)
{
    if (cfg == NULL) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers = 0;
    cfg->n_heads = 0;
    cfg->n_kv_heads = 0;
    cfg->head_dim = 0;
    cfg->hidden_dim = 0;
    cfg->intermediate_dim = 0;
    cfg->vocab_size = 0;
    cfg->n_ctx = 0;
    cfg->rope_theta = 10000.0;
    cfg->rope_scaling_factor = 1.0;
    cfg->norm_eps = 1e-5;
    cfg->n_expert = 0;
    cfg->n_expert_used = 0;
    cfg->sliding_window = 0;
    set_str(cfg->arch, OC_CONFIG_ARCH_LEN, "");
    set_str(cfg->rope_scaling_type, OC_CONFIG_ROPE_SCALING_LEN, "");
    set_str(cfg->ffn_type, OC_CONFIG_FFN_TYPE_LEN, "swiglu");
    return OC_OK;
}

OcError oc_model_config_from_gguf(const OcGgufFile *gguf, OcModelConfig *cfg)
{
    if (gguf == NULL || cfg == NULL) return OC_ERR_INVALID_ARG;
    OcError e = oc_model_config_init(cfg);
    if (e != OC_OK) return e;

    /* Read architecture; fall back to "llama" if absent. */
    const char *arch = cfg_str(gguf, "general.architecture", NULL);
    if (arch == NULL) {
        /* No architecture key at all — treat as malformed. */
        return OC_ERR_FORMAT;
    }
    set_str(cfg->arch, OC_CONFIG_ARCH_LEN, arch);

    /* Build the arch-prefixed key namespace. */
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%s.", arch);

    char key[160];

    /* Vocab size: prefer general.vocab_size, fall back to arch.vocab_size. */
    snprintf(key, sizeof(key), "%svocab_size", prefix);
    cfg->vocab_size = cfg_u32(gguf, key, 0);
    if (cfg->vocab_size == 0) {
        cfg->vocab_size = cfg_u32(gguf, "general.vocab_size", 0);
    }

    snprintf(key, sizeof(key), "%scontext_length", prefix);
    cfg->n_ctx = cfg_u32(gguf, key, 0);

    snprintf(key, sizeof(key), "%sblock_count", prefix);
    cfg->n_layers = cfg_u32(gguf, key, 0);

    snprintf(key, sizeof(key), "%sembedding_length", prefix);
    cfg->hidden_dim = cfg_u32(gguf, key, 0);

    snprintf(key, sizeof(key), "%sfeed_forward_length", prefix);
    cfg->intermediate_dim = cfg_u32(gguf, key, 0);

    snprintf(key, sizeof(key), "%sattention.head_count", prefix);
    cfg->n_heads = cfg_u32(gguf, key, 0);

    snprintf(key, sizeof(key), "%sattention.head_count_kv", prefix);
    {
        uint32_t kv = cfg_u32(gguf, key, 0);
        /* If KV heads is absent, default to n_heads (MHA). */
        cfg->n_kv_heads = (kv == 0) ? cfg->n_heads : kv;
    }

    snprintf(key, sizeof(key), "%sattention.key_length", prefix);
    {
        uint32_t kl = cfg_u32(gguf, key, 0);
        if (kl == 0 && cfg->n_heads > 0) {
            kl = cfg->hidden_dim / cfg->n_heads;
        }
        cfg->head_dim = kl;
    }

    snprintf(key, sizeof(key), "%srope.freq_base", prefix);
    cfg->rope_theta = (double)cfg_f32(gguf, key, (float)cfg->rope_theta);

    snprintf(key, sizeof(key), "%srope.scaling.type", prefix);
    {
        const char *rst = cfg_str(gguf, key, NULL);
        if (rst != NULL) {
            set_str(cfg->rope_scaling_type, OC_CONFIG_ROPE_SCALING_LEN, rst);
        }
    }

    snprintf(key, sizeof(key), "%srope.scaling.factor", prefix);
    {
        float f = cfg_f32(gguf, key, 0.0f);
        if (f != 0.0f) cfg->rope_scaling_factor = (double)f;
    }

    snprintf(key, sizeof(key), "%sattention.layer_norm_rms_epsilon", prefix);
    {
        float eps = cfg_f32(gguf, key, 0.0f);
        if (eps != 0.0f) {
            cfg->norm_eps = (double)eps;
        } else {
            /* Some arches use "attention.layer_norm_epsilon". */
            snprintf(key, sizeof(key), "%sattention.layer_norm_epsilon", prefix);
            eps = cfg_f32(gguf, key, 0.0f);
            if (eps != 0.0f) cfg->norm_eps = (double)eps;
        }
    }

    snprintf(key, sizeof(key), "%sexpert_count", prefix);
    cfg->n_expert = cfg_u32(gguf, key, 0);

    snprintf(key, sizeof(key), "%sexpert_used_count", prefix);
    cfg->n_expert_used = cfg_u32(gguf, key, 0);

    snprintf(key, sizeof(key), "%sattention.sliding_window", prefix);
    {
        uint32_t sw = cfg_u32(gguf, key, 0);
        /* sliding_window: 0 means full attention; store as int32. */
        cfg->sliding_window = (int32_t)sw;
    }

    /* Determine FFN type from arch + expert presence. */
    if (cfg->n_expert > 0) {
        set_str(cfg->ffn_type, OC_CONFIG_FFN_TYPE_LEN, "swiglu");
    } else {
        /* Most Llama-family arches use SwiGLU; Gemma uses GeGLU. */
        if (strncmp(arch, "gemma", 5) == 0) {
            set_str(cfg->ffn_type, OC_CONFIG_FFN_TYPE_LEN, "geglu");
        } else {
            set_str(cfg->ffn_type, OC_CONFIG_FFN_TYPE_LEN, "swiglu");
        }
    }

    (void)cfg_f64; /* reserved for future double-precision keys */
    return OC_OK;
}

OcError oc_model_config_validate(const OcModelConfig *cfg)
{
    if (cfg == NULL) return OC_ERR_INVALID_ARG;
    if (cfg->n_layers == 0) return OC_ERR_INVALID_ARG;
    if (cfg->n_heads == 0) return OC_ERR_INVALID_ARG;
    if (cfg->hidden_dim == 0) return OC_ERR_INVALID_ARG;
    if (cfg->vocab_size == 0) return OC_ERR_INVALID_ARG;
    if (cfg->n_kv_heads == 0) return OC_ERR_INVALID_ARG;
    if (cfg->n_kv_heads > cfg->n_heads) return OC_ERR_INVALID_ARG;
    if (cfg->head_dim == 0) {
        /* Derive head_dim if unset. */
        if (cfg->hidden_dim % cfg->n_heads != 0) return OC_ERR_INVALID_ARG;
    }
    if (cfg->n_expert > 0) {
        if (cfg->n_expert_used == 0) return OC_ERR_INVALID_ARG;
        if (cfg->n_expert_used > cfg->n_expert) return OC_ERR_INVALID_ARG;
    }
    return OC_OK;
}

size_t oc_model_config_print(const OcModelConfig *cfg, char *out, size_t out_size)
{
    if (cfg == NULL) {
        if (out && out_size) out[0] = '\0';
        return 0;
    }
    /* Compute the length that would be written. */
    int n = snprintf(out, out_size,
        "arch=%s n_layers=%u n_heads=%u n_kv_heads=%u head_dim=%u "
        "hidden_dim=%u intermediate_dim=%u vocab_size=%u n_ctx=%u "
        "rope_theta=%.1f rope_scaling=%s@%.2f norm_eps=%.1e "
        "ffn_type=%s n_expert=%u n_expert_used=%u sliding_window=%d "
        "moe=%s gqa=%s",
        cfg->arch, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
        cfg->head_dim, cfg->hidden_dim, cfg->intermediate_dim,
        cfg->vocab_size, cfg->n_ctx, cfg->rope_theta,
        cfg->rope_scaling_type[0] ? cfg->rope_scaling_type : "none",
        cfg->rope_scaling_factor, cfg->norm_eps, cfg->ffn_type,
        cfg->n_expert, cfg->n_expert_used, cfg->sliding_window,
        oc_model_config_is_moe(cfg) ? "yes" : "no",
        oc_model_config_has_gqa(cfg) ? "yes" : "no");
    if (n < 0) {
        if (out && out_size) out[0] = '\0';
        return 0;
    }
    return (size_t)n;
}

const char *oc_model_config_arch_name(const OcModelConfig *cfg)
{
    return cfg ? cfg->arch : "";
}

bool oc_model_config_is_moe(const OcModelConfig *cfg)
{
    return cfg != NULL && cfg->n_expert > 0;
}

bool oc_model_config_has_gqa(const OcModelConfig *cfg)
{
    return cfg != NULL && cfg->n_kv_heads > 0 &&
           cfg->n_kv_heads < cfg->n_heads;
}

uint64_t oc_model_config_n_params(const OcModelConfig *cfg)
{
    if (cfg == NULL) return 0;
    if (oc_model_config_validate(cfg) != OC_OK) return 0;

    uint32_t head_dim = cfg->head_dim;
    if (head_dim == 0 && cfg->n_heads > 0) {
        head_dim = cfg->hidden_dim / cfg->n_heads;
    }
    uint64_t hidden = cfg->hidden_dim;
    uint64_t inter = cfg->intermediate_dim;
    uint64_t n_layers = cfg->n_layers;
    uint64_t vocab = cfg->vocab_size;

    /* Embedding (token) table. */
    uint64_t total = vocab * hidden;

    /* Per-layer parameters. */
    /* Attention: q_proj (hidden -> n_heads*head_dim), */
    uint64_t q_out = (uint64_t)cfg->n_heads * head_dim;
    uint64_t kv_out = (uint64_t)cfg->n_kv_heads * head_dim;
    uint64_t attn = hidden * q_out + hidden * kv_out + hidden * kv_out +
                    q_out * hidden;

    /* FFN: for SwiGLU/GEG it is 3 matrices (gate, up, down) of hidden * inter + hidden * inter + inter * hidden. */
    uint64_t ffn_per_layer = hidden * inter + hidden * inter + inter * hidden;
    uint64_t ffn = ffn_per_layer;
    if (cfg->n_expert > 0) {
        ffn *= cfg->n_expert;
    }

    /* Per-layer attention + FFN norms (2 * hidden each layer). */
    uint64_t norms = 2 * hidden;

    total += n_layers * (attn + ffn + norms);

    /* Final norm + output projection (lm_head). */
    total += hidden;            /* final norm */
    total += vocab * hidden;    /* lm_head */

    return total;
}
