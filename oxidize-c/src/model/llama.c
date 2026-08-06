/*
 * llama.c — Llama-family dense forward pass (CPU).
 *
 * Port of oxidize-core/src/model/inference.rs (Llama/Mistral path of
 * `InferenceModel::forward_single`) + inference/{forward,layers,load}.rs.
 *
 * Weight matrices are read zero-copy from the mmap'd GGUF. The matvec
 * dequantizes one weight row at a time via the SIMD-accelerated
 * `oc_quant_dequant_row`, then does an f32 dot with the activation.
 *
 * Bit-exactness target (VAL-FWD-001..004): RMSNorm, RoPE, SwiGLU, attention
 * match the Rust scalar reference. F32-accumulated RMSNorm (not f64) to
 * mirror Rust's `x.iter().map(|v| v*v).sum::<f32>()`.
 */
#include "oxidize/llama.h"
#include "oxidize/rope_scaling.h"

#include "oxidize/activation.h"
#include "oxidize/arch_forward.h"
#include "oxidize/arena.h"
#include "oxidize/gguf.h"
#include "oxidize/log.h"
#include "oxidize/matvec.h"
#include "oxidize/model.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n, sz);
    return p;
}

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

static const char *cfg_str(const OcGgufFile *f, const char *key, const char *def)
{
    const char *v;
    return oc_gguf_metadata_get_str(f, key, &v, NULL) ? v : def;
}

/* Build a weight view from a resolved tensor info. dims[0]=cols, dims[1]=rows
 * for 2-D; for 1-D (norms) dims[0]=n. */
static OcWeightView view_from_info(const OcGgufMmappedFile *m,
                                   const OcGgufTensorInfo *info)
{
    OcWeightView v = {0};
    if (info == NULL) return v;
    v.data = oc_gguf_map_tensor_data(m, info);
    v.qtype = oc_quant_type_from_ggml_id(info->ggml_type);
    if (info->n_dims == 1) {
        v.rows = 1;
        v.cols = (size_t)info->dims[0];
    } else if (info->n_dims >= 2) {
        v.rows = (size_t)info->dims[1];
        v.cols = (size_t)info->dims[0];
    }
    v.row_bytes = oc_quantized_size(v.qtype, v.cols);
    if (v.row_bytes == 0) v.row_bytes = v.cols * sizeof(float);  /* F32 etc. */
    return v;
}

/* Dequantize a 1-D norm tensor (length n) into a freshly malloc'd f32 array.
 * Norms are almost always stored as F32 in practice; this also handles the
 * rare quantized case. Returns NULL on OOM. */
static float *load_norm(const OcGgufMmappedFile *m, const OcGgufTensorInfo *info,
                        size_t n)
{
    if (info == NULL) return NULL;
    float *out = xcalloc(n, sizeof(float));
    if (out == NULL) return NULL;
    OcWeightView v = view_from_info(m, info);
    if (v.qtype == OC_QUANT_F32) {
        memcpy(out, v.data, n * sizeof(float));
    } else {
        oc_quant_dequant_row(v.qtype, v.data, v.row_bytes, out, n);
    }
    return out;
}

/* ─── Config parsing ──────────────────────────────────────────────────── */

/* Read element `idx` of a u32-ish metadata array, falling back to `def` when
 * the key is missing, is not an array, or is too short.
 *
 * Gemma 4 stores attention.head_count_kv and attention.sliding_window_pattern
 * as per-layer arrays rather than scalars, which the cfg_u32 getters cannot
 * see at all — they ask for a scalar and get nothing. GGUF writers are not
 * consistent about the integer width used for such arrays, so accept any of
 * the unsigned/signed 32-and-under types rather than requiring UINT32. */
static uint32_t cfg_u32_at(const OcGgufFile *f, const char *key, size_t idx,
                           uint32_t def)
{
    const OcGgufMetadataValue *val = oc_gguf_metadata_get(f, key);
    if (val == NULL || val->type != OC_GGUF_MT_ARRAY) return def;
    const OcGgufMetadataArray *arr = &val->v.arr;
    if (idx >= arr->len || arr->values == NULL) return def;
    const OcGgufMetadataValue *e = &arr->values[idx];
    switch (e->type) {
    case OC_GGUF_MT_UINT8:  return (uint32_t)e->v.u8;
    case OC_GGUF_MT_INT8:   return (uint32_t)e->v.i8;
    case OC_GGUF_MT_UINT16: return (uint32_t)e->v.u16;
    case OC_GGUF_MT_INT16:  return (uint32_t)e->v.i16;
    case OC_GGUF_MT_UINT32: return e->v.u32;
    case OC_GGUF_MT_INT32:  return (uint32_t)e->v.i32;
    case OC_GGUF_MT_BOOL:   return e->v.b ? 1u : 0u;
    default:                return def;
    }
}

/* Length of a metadata array, or 0 if the key is absent or not an array. */
static size_t cfg_array_len(const OcGgufFile *f, const char *key)
{
    const OcGgufMetadataValue *val = oc_gguf_metadata_get(f, key);
    if (val == NULL || val->type != OC_GGUF_MT_ARRAY) return 0;
    return val->v.arr.len;
}

static OcError parse_config(const OcGgufFile *f, const char *arch_str,
                            OcLlamaConfig *cfg)
{
    /* The metadata keys are namespaced by architecture string ("llama.",
     * "qwen2.", "mistral.", ...). Build the key prefix. */
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%s.", arch_str ? arch_str : "llama");

    char key[160];
    /* Build keys with the arch prefix. */
    snprintf(key, sizeof(key), "%svocab_size", prefix);
    cfg->vocab_size  = cfg_u32(f, key, 32000);
    snprintf(key, sizeof(key), "%scontext_length", prefix);
    cfg->n_ctx       = cfg_u32(f, key, 4096);
    snprintf(key, sizeof(key), "%sblock_count", prefix);
    cfg->n_layer     = cfg_u32(f, key, 32);
    snprintf(key, sizeof(key), "%sembedding_length", prefix);
    cfg->n_embd      = cfg_u32(f, key, 4096);
    snprintf(key, sizeof(key), "%sfeed_forward_length", prefix);
    cfg->n_ff        = cfg_u32(f, key, 11008);
    snprintf(key, sizeof(key), "%sattention.head_count", prefix);
    cfg->n_head      = cfg_u32(f, key, 32);
    snprintf(key, sizeof(key), "%sattention.head_count_kv", prefix);
    cfg->n_head_kv   = cfg_u32(f, key, cfg->n_head);
    if (cfg->n_head == 0 || cfg->n_head_kv == 0 || cfg->n_embd == 0 ||
        cfg->n_head_kv > cfg->n_head || cfg->n_head % cfg->n_head_kv != 0 ||
        cfg->n_embd % cfg->n_head != 0)
        return OC_ERR_MODEL;
    snprintf(key, sizeof(key), "%sattention.key_length", prefix);
    uint32_t key_len = cfg_u32(f, key, 0);
    snprintf(key, sizeof(key), "%srope.dimension_count", prefix);
    uint32_t rope_dim = cfg_u32(f, key, 0);
    snprintf(key, sizeof(key), "%srope.freq_base", prefix);
    cfg->rope_theta   = cfg_f32(f, key, 10000.0f);
    snprintf(key, sizeof(key), "%sattention.layer_norm_rms_epsilon", prefix);
    cfg->rms_norm_eps = cfg_f32(f, key, 1e-5f);

    /* MoE config (Qwen3-MoE / Mixtral). Defaults: no MoE, softmax gating,
     * no routed scaling. The bare-name fallback matches Mixtral which uses
     * unprefixed `expert_count` etc. */
    snprintf(key, sizeof(key), "%sexpert_count", prefix);
    cfg->num_experts = cfg_u32(f, key, 0);
    if (cfg->num_experts == 0) {
        cfg->num_experts = cfg_u32(f, "expert_count", 0);
    }
    snprintf(key, sizeof(key), "%sexpert_used_count", prefix);
    cfg->num_experts_per_tok = cfg_u32(f, key, 0);
    if (cfg->num_experts_per_tok == 0) {
        cfg->num_experts_per_tok = cfg_u32(f, "expert_used_count", 0);
    }
    snprintf(key, sizeof(key), "%sexpert_feed_forward_length", prefix);
    cfg->expert_intermediate_size = cfg_u32(f, key, 0);
    snprintf(key, sizeof(key), "%sexpert_gating_func", prefix);
    uint32_t gating = cfg_u32(f, key, 1);
    cfg->expert_gating_sigmoid = (gating == 2);
    snprintf(key, sizeof(key), "%sexpert_weights_scale", prefix);
    float scale = cfg_f32(f, key, 1.0f);
    cfg->expert_weights_scale = (scale > 0.0f) ? scale : 1.0f;

    /* Gemma-specific: GeGLU FFN + norm scaling. */
    bool no_rope = false;   /* GPT-2/J: learned positional embeddings */
    cfg->uses_geglu = false;
    cfg->norm_scale = 1.0f;
    cfg->sliding_window = 0;
    cfg->sliding_window_pattern = 1;
    /* YaRN long-context scaling. */
    cfg->yarn_factor = 0.0f;
    cfg->yarn_orig_ctx = 0;
    cfg->uses_gemma4 = false;
    cfg->head_dim_swa = 0;
    cfg->n_head_kv_swa = 0;
    cfg->rope_dim_swa = 0;
    cfg->rope_theta_swa = 0.0f;
    cfg->layer_is_swa = NULL;
    cfg->logit_softcap = 0.0f;
    cfg->k_eq_v = false;
    cfg->attn_scale = 0.0f;
    cfg->v_rms_norm = false;
    if (arch_str) {
        /* Checked before the generic "gemma" prefix below, which would
         * otherwise match "gemma4" and apply the Gemma-2 single-geometry
         * interpretation to a dual-geometry model. */
        if (strcmp(arch_str, "gemma4") == 0) {
            cfg->uses_geglu = true;
            cfg->uses_gemma4 = true;
            /* Gemma scales the token embedding by sqrt(n_embd); it is applied
             * once after the embedding lookup, not inside each RMSNorm. */
            cfg->norm_scale = sqrtf((float)cfg->n_embd);

            snprintf(key, sizeof(key), "%sfinal_logit_softcapping", prefix);
            cfg->logit_softcap = cfg_f32(f, key, 0.0f);
            snprintf(key, sizeof(key), "%sattention.sliding_window", prefix);
            cfg->sliding_window = cfg_u32(f, key, 0);

            /* Sliding geometry. The *_swa metadata keys are the sliding-layer
             * values; the unsuffixed ones already parsed above are global. */
            snprintf(key, sizeof(key), "%sattention.key_length_swa", prefix);
            cfg->head_dim_swa = cfg_u32(f, key, 0);
            snprintf(key, sizeof(key), "%srope.dimension_count_swa", prefix);
            cfg->rope_dim_swa = cfg_u32(f, key, 0);
            snprintf(key, sizeof(key), "%srope.freq_base_swa", prefix);
            cfg->rope_theta_swa = cfg_f32(f, key, cfg->rope_theta);

            /* Per-layer attention kind. The pattern is an explicit array
             * (1 = sliding, 0 = global) — read it rather than assuming the
             * documented 5:1 repeat, so an unusual layout still loads. */
            snprintf(key, sizeof(key), "%sattention.sliding_window_pattern",
                     prefix);
            const size_t pat_len = cfg_array_len(f, key);
            cfg->layer_is_swa = xcalloc(cfg->n_layer, sizeof(uint8_t));
            if (cfg->layer_is_swa == NULL) return OC_ERR_OOM;
            for (uint32_t l = 0; l < cfg->n_layer; l++) {
                /* Absent pattern → treat every layer as sliding, which is the
                 * majority kind and keeps the window bound in play. */
                cfg->layer_is_swa[l] =
                    (uint8_t)(pat_len > 0 ? (cfg_u32_at(f, key, l, 1u) != 0u)
                                          : 1u);
            }

            /* KV head counts are a per-layer array. Take the global and
             * sliding values from the first layer of each kind actually
             * present rather than assuming which index is which. */
            char kvkey[128];
            snprintf(kvkey, sizeof(kvkey), "%sattention.head_count_kv", prefix);
            uint32_t kv_swa = 0, kv_glb = 0;
            for (uint32_t l = 0; l < cfg->n_layer; l++) {
                const uint32_t kv = cfg_u32_at(f, kvkey, l, 0);
                if (kv == 0) continue;
                if (cfg->layer_is_swa[l]) { if (!kv_swa) kv_swa = kv; }
                else                      { if (!kv_glb) kv_glb = kv; }
            }
            /* Scalar fallback for a writer that emitted one value. */
            if (kv_swa == 0 && kv_glb == 0)
                kv_swa = kv_glb = cfg_u32(f, kvkey, cfg->n_head);
            if (kv_glb == 0) kv_glb = kv_swa;
            if (kv_swa == 0) kv_swa = kv_glb;
            cfg->n_head_kv     = kv_glb;
            cfg->n_head_kv_swa = kv_swa;

            /* Global layers carry no attn_v tensor: V is the K projection
             * (config.json attention_k_eq_v). resolve_weights() aliases them
             * and asserts the tensor really is absent. */
            cfg->k_eq_v = true;
            /* Gemma 4 applies no 1/sqrt(head_dim) factor, and RMS-normalizes
             * V (weightless) after projection. Both come from llama.cpp's
             * gemma4 builder; neither is expressible in GGUF metadata. */
            cfg->attn_scale = 1.0f;
            cfg->v_rms_norm = true;
        } else if (strncmp(arch_str, "gemma", 5) == 0) {
            cfg->uses_geglu = true;
            cfg->norm_scale = sqrtf((float)cfg->n_embd);
            /* Gemma2 sliding window: alternating global/sliding layers. */
            snprintf(key, sizeof(key), "%sattention.sliding_window", prefix);
            uint32_t sw = cfg_u32(f, key, 0);
            if (sw > 0) {
                cfg->sliding_window = sw;
                cfg->sliding_window_pattern = 2; /* alternating */
            }
        } else if (strncmp(arch_str, "phi", 3) == 0) {
            /* Phi-3 uses GELU activation (same as GeGLU) but no norm scaling. */
            cfg->uses_geglu = true;
            cfg->norm_scale = 1.0f;
        } else if (strncmp(arch_str, "gpt2", 4) == 0 ||
                   strncmp(arch_str, "gptj", 4) == 0) {
            /* GPT-2/J: GELU FFN (handled by the arch_forward path), no
             * RoPE — learned positional embeddings. uses_geglu is NOT set:
             * these run the dedicated forward pass in arch_forward.c. */
            cfg->uses_gpt2 = true;
            cfg->uses_par_attn = true;
            cfg->norm_scale = 1.0f;
            no_rope = true;
        } else if (strncmp(arch_str, "gptneox", 7) == 0 ||
                   strncmp(arch_str, "falcon", 6) == 0) {
            /* NeoX/Falcon: LayerNorm + GeLU MLP with (partial) RoPE.
             * Dispatched to arch_forward.c; rope_dim comes from
             * rope.dimension_count metadata (falls back to head_dim). */
            cfg->uses_par_attn = true;
        }
    }
    /* YaRN: read from GGUF metadata if present. */
    snprintf(key, sizeof(key), "%srope.scaling.type", prefix);
    const char *scale_type = cfg_str(f, key, NULL);
    if (scale_type) {
        if (strcmp(scale_type, "yarn") == 0) {
            snprintf(key, sizeof(key), "%srope.scaling.factor", prefix);
            cfg->yarn_factor = cfg_f32(f, key, 1.0f);
            snprintf(key, sizeof(key),
                     "%srope.scaling.original_context_length", prefix);
            cfg->yarn_orig_ctx = cfg_u32(f, key, cfg->n_ctx);
        }
    }

    /* Also accept general.vocab_size as a fallback. */
    if (cfg->vocab_size == 32000) {
        uint32_t gv;
        if (oc_gguf_metadata_get_u32(f, "general.vocab_size", &gv)) {
            cfg->vocab_size = gv;
        }
    }

    /* DeepSeek MLA config. */
    cfg->uses_mla = false;
    cfg->mla_q_lora_dim = 0;
    cfg->mla_kv_lora_dim = 0;
    cfg->mla_q_rope_dim = 0;
    cfg->mla_kv_nope_head_dim = 0;
    cfg->mla_v_head_dim = 0;
    cfg->is_longcat = false;
    cfg->zero_expert_count = 0;
    cfg->yarn_beta_fast = 0.0f;
    cfg->yarn_beta_slow = 0.0f;
    cfg->yarn_mscale = 0.0f;
    cfg->yarn_mscale_all_dim = 0.0f;
    cfg->ngram_n_grams = 0;
    cfg->ngram_split_num = 0;
    if (arch_str && strcmp(arch_str, "longcat") == 0) {
        /* LongCat states its MLA geometry directly rather than through
         * DeepSeek's key_length_mla, and splits each block into two
         * sub-blocks. */
        cfg->uses_mla   = true;
        cfg->is_longcat = true;

        snprintf(key, sizeof(key), "%sattention.q_lora_rank", prefix);
        cfg->mla_q_lora_dim = cfg_u32(f, key, 1536);
        snprintf(key, sizeof(key), "%sattention.kv_lora_rank", prefix);
        cfg->mla_kv_lora_dim = cfg_u32(f, key, 512);
        cfg->mla_q_rope_dim = (rope_dim > 0) ? rope_dim : 64;
        /* key_length (192) = nope (128) + rope (64). */
        cfg->mla_kv_nope_head_dim =
            (key_len > cfg->mla_q_rope_dim) ? key_len - cfg->mla_q_rope_dim : 128;
        snprintf(key, sizeof(key), "%sattention.value_length", prefix);
        cfg->mla_v_head_dim = cfg_u32(f, key, cfg->mla_kv_nope_head_dim);
        cfg->n_head_kv = 1;             /* MLA caches one shared latent */

        /* Each GGUF block holds two sub-blocks. */
        cfg->n_layer *= 2;

        snprintf(key, sizeof(key), "%szero_expert_count", prefix);
        cfg->zero_expert_count = cfg_u32(f, key, 0);

        /* deepseek_yarn constants are not written to GGUF metadata; these are
         * LongCat's config.json values. */
        cfg->yarn_beta_fast      = 32.0f;
        cfg->yarn_beta_slow      = 1.0f;
        cfg->yarn_mscale         = 1.0f;
        cfg->yarn_mscale_all_dim = 1.0f;

        snprintf(key, sizeof(key), "%sngram.neighbor_num", prefix);
        uint32_t nn = cfg_u32(f, key, 0);
        snprintf(key, sizeof(key), "%sngram.split_num", prefix);
        uint32_t sn = cfg_u32(f, key, 0);
        cfg->ngram_split_num = sn;
        cfg->ngram_n_grams   = (nn > 1 && sn > 0) ? (nn - 1) * sn : 0;
    } else if (arch_str && (strncmp(arch_str, "deepseek", 8) == 0)) {
        snprintf(key, sizeof(key), "%sattention.key_length_mla", prefix);
        uint32_t mla_key_len = cfg_u32(f, key, 0);
        if (mla_key_len > 0) {
            cfg->uses_mla = true;
            snprintf(key, sizeof(key), "%sattention.lora_q", prefix);
            cfg->mla_q_lora_dim = cfg_u32(f, key, 512);
            snprintf(key, sizeof(key), "%sattention.lora_kv", prefix);
            cfg->mla_kv_lora_dim = cfg_u32(f, key, 512);
            snprintf(key, sizeof(key), "%sattention.key_length_rope", prefix);
            cfg->mla_q_rope_dim = cfg_u32(f, key, 64);
            cfg->mla_kv_nope_head_dim = mla_key_len - cfg->mla_q_rope_dim;
            cfg->mla_v_head_dim = cfg->mla_kv_nope_head_dim;
            /* For MLA, head_dim is the full per-head dim. */
            cfg->head_dim = mla_key_len;
            cfg->kv_head_dim = mla_key_len;
            cfg->n_head_kv = 1; /* MLA uses MQA (single KV head) */
        }
    }

    cfg->head_dim   = (key_len > 0) ? key_len : (cfg->n_embd / cfg->n_head);
    cfg->kv_head_dim = cfg->head_dim;   /* Llama: same; attention.key_length
                                         * applies to both. */
    cfg->rope_dim   = (rope_dim > 0) ? rope_dim : cfg->kv_head_dim;
    if (cfg->rope_dim > cfg->kv_head_dim) cfg->rope_dim = cfg->kv_head_dim;

    /* Gemma 4: fill in any sliding-geometry field the metadata left unset, and
     * validate both geometries. Done here because head_dim/rope_dim above are
     * the global values these fall back to. */
    if (cfg->uses_gemma4) {
        if (cfg->head_dim_swa == 0)   cfg->head_dim_swa = cfg->head_dim;
        if (cfg->n_head_kv_swa == 0)  cfg->n_head_kv_swa = cfg->n_head_kv;
        if (cfg->rope_theta_swa == 0.0f) cfg->rope_theta_swa = cfg->rope_theta;
        if (cfg->rope_dim_swa == 0)   cfg->rope_dim_swa = cfg->head_dim_swa;
        if (cfg->rope_dim_swa > cfg->head_dim_swa)
            cfg->rope_dim_swa = cfg->head_dim_swa;
        /* Both KV head counts must divide the (shared) query head count, or
         * the GQA grouping in the attention kernels is undefined. */
        if (cfg->n_head_kv == 0 || cfg->n_head_kv_swa == 0 ||
            cfg->n_head_kv > cfg->n_head || cfg->n_head_kv_swa > cfg->n_head ||
            cfg->n_head % cfg->n_head_kv != 0 ||
            cfg->n_head % cfg->n_head_kv_swa != 0) {
            free(cfg->layer_is_swa);
            cfg->layer_is_swa = NULL;
            return OC_ERR_MODEL;
        }
    }
    if (no_rope) cfg->rope_dim = 0;   /* preserve arch-specific no-RoPE */
    cfg->tied_embeddings = false;
    cfg->moe_layer_start = 0;
    /* MoE per-token top-k: default to 1 when experts exist but no top-k set. */
    if (cfg->num_experts > 0) {
        if (cfg->num_experts_per_tok == 0) cfg->num_experts_per_tok = 1;
        if (cfg->num_experts_per_tok > cfg->num_experts) {
            cfg->num_experts_per_tok = cfg->num_experts;
        }
        if (cfg->expert_intermediate_size == 0) cfg->expert_intermediate_size = cfg->n_ff;
        /* Parse moe_layer_start (default 0 = all layers are MoE). */
        snprintf(key, sizeof(key), "%smoe_layer_start", prefix);
        cfg->moe_layer_start = cfg_u32(f, key, 0);
        if (cfg->moe_layer_start == 0) {
            cfg->moe_layer_start = cfg_u32(f, "moe_layer_start", 0);
        }
        /* Also check hunyuan-specific key. */
        snprintf(key, sizeof(key), "%smoe_layer_start", prefix);
        uint32_t mlstart = cfg_u32(f, key, 0);
        if (mlstart > 0) cfg->moe_layer_start = mlstart;
    }
    return OC_OK;
}

/* ─── Weight resolution ───────────────────────────────────────────────── */

/* Match a canonical tensor name against the per-layer / global pattern set
 * and store into the model. Returns true if matched. */
static bool assign_tensor(OcLlamaModel *m, const char *cname,
                          const OcGgufMmappedFile *mm,
                          const OcGgufTensorInfo *info)
{
    /* Global tensors. */
    if (strcmp(cname, "tok_embeddings.weight") == 0 ||
        strcmp(cname, "token_embd.weight") == 0) {
        m->tok_embeddings = view_from_info(mm, info);
        return true;
    }
    if (strcmp(cname, "output.weight") == 0) {
        m->output = view_from_info(mm, info);
        return true;
    }
    if (strcmp(cname, "norm.weight") == 0 ||
        strcmp(cname, "output_norm.weight") == 0 ||
        strcmp(cname, "token_embd_norm.weight") == 0) {
        m->final_norm = load_norm(mm, info, m->cfg.n_embd);
        return true;
    }
    /* Final-LayerNorm bias (GPT-2/NeoX/Falcon). */
    if (strcmp(cname, "norm.bias") == 0 ||
        strcmp(cname, "output_norm.bias") == 0) {
        m->final_norm_bias = load_norm(mm, info, m->cfg.n_embd);
        return true;
    }
    /* LongCat n-gram over-embedding: ngram_embd_<i>.weight / ngram_proj_<i>. */
    if (m->cfg.is_longcat && strncmp(cname, "ngram_", 6) == 0) {
        const char *rest = cname + 6;
        bool is_embd = strncmp(rest, "embd_", 5) == 0;
        bool is_proj = strncmp(rest, "proj_", 5) == 0;
        if (is_embd || is_proj) {
            char *nend = NULL;
            unsigned long ti = strtoul(rest + 5, &nend, 10);
            if (nend != rest + 5 && strcmp(nend, ".weight") == 0 &&
                ti < OC_LONGCAT_MAX_NGRAM) {
                if (is_embd) m->ngram_embd[ti] = view_from_info(mm, info);
                else         m->ngram_proj[ti] = view_from_info(mm, info);
                return true;
            }
        }
        return false;
    }
    /* Per-layer: blk.<N>.<suffix> */
    if (strncmp(cname, "blk.", 4) == 0) {
        char *end = NULL;
        unsigned long layer_idx = strtoul(cname + 4, &end, 10);
        if (end == cname + 4 || *end != '.') return false;
        const char *suf = end + 1;
        /* LongCat ScMoE: one GGUF block holds two attention+FFN sub-blocks.
         * Sub-block tensors carry a `_0`/`_1` marker on the stem (before
         * `.weight`/`.bias`); everything else in the block — router, router
         * bias, expert pool, indexer — exists once and belongs to the even
         * sub-layer. */
        char stem[96];
        if (m->cfg.is_longcat) {
            const char *dot = strrchr(suf, '.');
            size_t base = dot ? (size_t)(dot - suf) : strlen(suf);
            if (base >= 2 && suf[base - 2] == '_' &&
                (suf[base - 1] == '0' || suf[base - 1] == '1')) {
                size_t keep = base - 2;
                size_t tail = dot ? strlen(dot) : 0;
                if (keep + tail + 1 > sizeof stem) return false;
                memcpy(stem, suf, keep);
                if (dot) memcpy(stem + keep, dot, tail);
                stem[keep + tail] = '\0';
                layer_idx = layer_idx * 2 + (unsigned long)(suf[base - 1] - '0');
                suf = stem;
            } else {
                layer_idx = layer_idx * 2;
            }
        }
        if (layer_idx >= m->cfg.n_layer) return false;
        OcLlamaLayer *L = &m->layers[layer_idx];
        if (strcmp(suf, "attn_norm.weight") == 0) {
            L->attn_norm = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "ffn_norm.weight") == 0) {
            L->ffn_norm = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "attn_norm.bias") == 0) {
            L->attn_norm_bias = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "ffn_norm.bias") == 0) {
            L->ffn_norm_bias = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "attn_q.weight") == 0) {
            L->attn_q = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_k.weight") == 0) {
            L->attn_k = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_v.weight") == 0) {
            L->attn_v = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_q_norm.weight") == 0) {
            /* Gemma per-head Q/K RMSNorm. Length is the LAYER's head_dim, not
             * the model-wide one — 256 on sliding layers and 512 on global
             * ones — so take it from the tensor's own shape. */
            L->attn_q_norm = load_norm(mm, info, (size_t)info->dims[0]);
        } else if (strcmp(suf, "attn_k_norm.weight") == 0) {
            L->attn_k_norm = load_norm(mm, info, (size_t)info->dims[0]);
        } else if (strcmp(suf, "post_attention_norm.weight") == 0) {
            L->post_attention_norm = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "post_ffw_norm.weight") == 0) {
            L->post_ffw_norm = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "layer_output_scale.weight") == 0) {
            /* Single f32 scalar per layer. */
            float *s = load_norm(mm, info, 1);
            if (s != NULL) { L->layer_output_scale = s[0]; free(s); }
        } else if (strcmp(suf, "attn_q.bias") == 0) {
            L->attn_q_bias = load_norm(mm, info, (size_t)info->dims[0]);
        } else if (strcmp(suf, "attn_k.bias") == 0) {
            L->attn_k_bias = load_norm(mm, info, (size_t)info->dims[0]);
        } else if (strcmp(suf, "attn_v.bias") == 0) {
            L->attn_v_bias = load_norm(mm, info, (size_t)info->dims[0]);
        } else if (strcmp(suf, "attn_output.weight") == 0) {
            L->attn_output = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_gate.weight") == 0) {
            L->ffn_gate = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_up.weight") == 0) {
            L->ffn_up = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_down.weight") == 0) {
            L->ffn_down = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_gate_inp.weight") == 0) {
            L->ffn_gate_inp = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_gate_exps.weight") == 0) {
            L->ffn_gate_exps = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_up_exps.weight") == 0) {
            L->ffn_up_exps = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_down_exps.weight") == 0) {
            L->ffn_down_exps = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_gate_shexp.weight") == 0) {
            L->ffn_gate_shexp = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_up_shexp.weight") == 0) {
            L->ffn_up_shexp = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_down_shexp.weight") == 0) {
            L->ffn_down_shexp = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_gate_inp_shexp.weight") == 0) {
            L->ffn_gate_inp_shexp = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_q_a.weight") == 0) {
            L->mla_q_a = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_q_a_norm.weight") == 0) {
            L->mla_q_a_norm = load_norm(mm, info, m->cfg.mla_q_lora_dim);
        } else if (strcmp(suf, "attn_q_b.weight") == 0) {
            L->mla_q_b = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_kv_a_mqa.weight") == 0) {
            L->mla_kv_a_mqa = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_kv_a_norm.weight") == 0) {
            L->mla_kv_a_norm = load_norm(mm, info, m->cfg.mla_kv_lora_dim);
        } else if (strcmp(suf, "attn_k_b.weight") == 0) {
            L->mla_k_b = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_v_b.weight") == 0) {
            L->mla_v_b = view_from_info(mm, info);
        } else if (strcmp(suf, "exp_probs_b.bias") == 0) {
            /* Length is num_experts + zero_expert_count; take it from the
             * tensor rather than recomputing, so a model with a different
             * zero-expert split still binds. */
            L->exp_probs_b = load_norm(mm, info, (size_t)info->dims[0]);
        } else {
            return false;   /* unrecognized suffix; not an error */
        }
        return true;
    }
    return false;
}

static OcError resolve_weights(OcLlamaModel *m)
{
    OcArena *arena = oc_arena_new(1u << 20);   /* 1 MiB for mapped names */
    if (arena == NULL) return OC_ERR_OOM;
    OcGgufTensorInfo *infos = NULL;
    size_t n = 0;
    OcError e = oc_gguf_map_mapped_tensor_infos(&m->gguf, arena, &infos, &n);
    if (e != OC_OK) { oc_arena_free(arena); return e; }

    /* Per-layer defaults, applied before binding so a tensor that is present
     * simply overwrites them. layer_output_scale must start at 1.0 (identity),
     * not the 0.0 that calloc gives — a zero scale silently zeroes the layer. */
    for (uint32_t l = 0; l < m->cfg.n_layer; l++) {
        m->layers[l].layer_output_scale = 1.0f;
    }

    for (size_t i = 0; i < n; i++) {
        const char *cname = infos[i].name;   /* already canonical-mapped */
        assign_tensor(m, cname, &m->gguf, &infos[i]);
    }
    oc_arena_free(arena);

    /* Validate critical tensors. */
    if (m->tok_embeddings.data == NULL) {
        oc_log(OC_LOG_ERROR, "llama: tok_embeddings.weight not found");
        return OC_ERR_MODEL;
    }
    /* If output.weight absent → tied embeddings. */
    if (m->output.data == NULL) {
        m->output = m->tok_embeddings;
        m->cfg.tied_embeddings = true;
    }
    if (m->final_norm == NULL) {
        oc_log(OC_LOG_ERROR, "llama: norm.weight not found");
        return OC_ERR_MODEL;
    }
    /* Infer n_ff from ffn_gate if metadata was absent. */
    if (m->cfg.n_ff == 0 && m->layers[0].ffn_gate.rows > 0) {
        m->cfg.n_ff = (uint32_t)m->layers[0].ffn_gate.rows;
    }

    /* Resolve each layer's attention geometry once, here, so neither forward
     * pass has to re-derive it from the layer index. For everything except
     * Gemma 4 this is just the model-wide values copied down. */
    for (uint32_t l = 0; l < m->cfg.n_layer; l++) {
        OcLlamaLayer *L = &m->layers[l];
        const bool swa = m->cfg.uses_gemma4 && m->cfg.layer_is_swa != NULL &&
                         m->cfg.layer_is_swa[l];
        L->head_dim   = swa ? m->cfg.head_dim_swa   : m->cfg.head_dim;
        L->n_head_kv  = swa ? m->cfg.n_head_kv_swa  : m->cfg.n_head_kv;
        L->rope_dim   = swa ? m->cfg.rope_dim_swa   : m->cfg.rope_dim;
        L->rope_theta = swa ? m->cfg.rope_theta_swa : m->cfg.rope_theta;
        L->sliding_window = swa ? m->cfg.sliding_window : 0u;

        /* Gemma 4 global layers reuse the K projection as V, and so ship no
         * attn_v tensor. Alias rather than special-casing every consumer.
         * Only do this when attn_v is genuinely absent: a model that has both
         * must keep them distinct even if the flag is set. */
        if (m->cfg.k_eq_v && L->attn_v.data == NULL && L->attn_k.data != NULL) {
            L->attn_v = L->attn_k;
        }

        /* Every layer must now have a coherent shape, or the forward passes
         * would read past a projection. Catch it at load with a clear
         * message rather than as a segfault mid-generation. */
        if (L->attn_q.data != NULL) {
            if (L->head_dim == 0 || L->n_head_kv == 0 ||
                L->attn_q.rows != (size_t)m->cfg.n_head * L->head_dim ||
                L->attn_k.rows != (size_t)L->n_head_kv * L->head_dim) {
                oc_log(OC_LOG_ERROR,
                       "llama: layer %u geometry mismatch: attn_q rows=%zu "
                       "(expected %u*%u), attn_k rows=%zu (expected %u*%u)",
                       l, L->attn_q.rows, m->cfg.n_head, L->head_dim,
                       L->attn_k.rows, L->n_head_kv, L->head_dim);
                return OC_ERR_MODEL;
            }
        }
    }
    if (m->tok_embeddings.rows == 0 || m->tok_embeddings.rows > UINT32_MAX ||
        m->tok_embeddings.cols != m->cfg.n_embd ||
        m->output.rows != m->tok_embeddings.rows ||
        m->output.cols != m->cfg.n_embd)
        return OC_ERR_MODEL;
    m->cfg.vocab_size = (uint32_t)m->tok_embeddings.rows;
    return OC_OK;
}

/* ─── Load ─────────────────────────────────────────────────────────────── */

OcError oc_llama_load(const char *path, OcLlamaModel *out)
{
    if (path == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    OcError e = oc_gguf_map_open(path, &out->gguf);
    if (e != OC_OK) return e;

    out->arch = oc_gguf_arch_from_file(&out->gguf.unified);
    /* Read the raw `general.architecture` string for the config-key prefix
     * (must match the GGUF on-disk value exactly: "qwen2", not "qwen";
     * "deepseek2", not "deepseek"). Fall back to the enum name only if the
     * metadata key is absent. */
    const char *arch_str = NULL;
    size_t arch_len = 0;
    if (!oc_gguf_metadata_get_str(&out->gguf.unified, "general.architecture",
                                  &arch_str, &arch_len)) {
        arch_str = oc_model_arch_name(out->arch);
    }
    if (arch_str == NULL) arch_str = "llama";
    if (strcmp(arch_str, "llama") != 0 && strcmp(arch_str, "mistral") != 0 &&
        strcmp(arch_str, "qwen2") != 0 && strcmp(arch_str, "gpt2") != 0 &&
        strcmp(arch_str, "gptneox") != 0 && strcmp(arch_str, "falcon") != 0 &&
        strcmp(arch_str, "gemma4") != 0 && strcmp(arch_str, "longcat") != 0) {
        oc_gguf_map_free(&out->gguf);
        return OC_ERR_MODEL;
    }

    OcError e2 = parse_config(&out->gguf.unified, arch_str, &out->cfg);
    if (e2 != OC_OK) { oc_gguf_map_free(&out->gguf); return e2; }

    out->layers = xcalloc(out->cfg.n_layer, sizeof(OcLlamaLayer));
    if (out->layers == NULL) {
        oc_gguf_map_free(&out->gguf);
        return OC_ERR_OOM;
    }

    e = resolve_weights(out);
    if (e != OC_OK) {
        oc_llama_free(out);
        return e;
    }
    oc_log(OC_LOG_INFO, "llama: loaded %s n_layer=%u n_embd=%u n_head=%u "
           "n_head_kv=%u head_dim=%u n_ff=%u vocab=%u ctx=%u%s",
           arch_str, out->cfg.n_layer, out->cfg.n_embd, out->cfg.n_head,
           out->cfg.n_head_kv, out->cfg.head_dim, out->cfg.n_ff,
           out->cfg.vocab_size, out->cfg.n_ctx,
           out->cfg.tied_embeddings ? " (tied)" : "");
    return OC_OK;
}

/* ─── Session ─────────────────────────────────────────────────────────── */

/* ─── Q8 KV cache helpers ─────────────────────────────────────────────────
 *
 * Symmetric per-group int8, the same scheme as Q8_0 weights: one f32 scale
 * per (layer, position, kv head), codes in [-127, 127]. 127 rather than 128
 * keeps it symmetric so negation is exact and there is no asymmetric overflow.
 */
#define OC_KV_Q8_QMAX 127.0f

static void kv_q8_encode(const float *src, int8_t *codes, float *scale_out,
                         size_t n)
{
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        /* All-zero group: any nonzero scale reconstructs zeros exactly. */
        *scale_out = 1.0f;
        memset(codes, 0, n);
        return;
    }
    const float scale = amax / OC_KV_Q8_QMAX;
    const float inv = 1.0f / scale;
    for (size_t i = 0; i < n; i++) {
        float q = roundf(src[i] * inv);
        if (q >  OC_KV_Q8_QMAX) q =  OC_KV_Q8_QMAX;
        if (q < -OC_KV_Q8_QMAX) q = -OC_KV_Q8_QMAX;
        codes[i] = (int8_t)q;
    }
    *scale_out = scale;
}

/* Environment default for the KV type; F32 unless OX_KV_TYPE=q8. */
static OcKvCacheType kv_type_from_env(void)
{
    const char *v = getenv("OX_KV_TYPE");
    if (v == NULL) return OC_KV_F32;
    if (strcmp(v, "q8") == 0 || strcmp(v, "Q8") == 0) return OC_KV_Q8;
    return OC_KV_F32;
}

/* Floats per cached position per layer.
 *
 * The cache is indexed at a single uniform stride across layers, so on a model
 * whose layers disagree it must be the MAXIMUM, not any one layer's value.
 * Gemma 4 is exactly that case: sliding layers hold 16 KV heads x 256 = 4096
 * floats while global layers hold 4 x 512 = 2048. Sizing from cfg.n_head_kv
 * (the global count) would under-allocate every sliding layer by 2x and
 * corrupt the cache. Global layers leave the upper half of their row unused,
 * which is the price of a uniform stride. */
static size_t kv_row_floats_for(const OcLlamaModel *model)
{
    if (model->cfg.uses_mla)
        return (size_t)model->cfg.n_head * model->cfg.head_dim;

    size_t row = (size_t)model->cfg.n_head_kv * model->cfg.kv_head_dim;
    if (model->cfg.uses_gemma4) {
        const size_t swa_row =
            (size_t)model->cfg.n_head_kv_swa * model->cfg.head_dim_swa;
        const size_t glb_row =
            (size_t)model->cfg.n_head_kv * model->cfg.head_dim;
        row = (swa_row > glb_row) ? swa_row : glb_row;
    }
    return row;
}

size_t oc_llama_kv_cache_bytes(const OcLlamaModel *model, OcKvCacheType kv_type)
{
    if (model == NULL) return 0;
    size_t row = kv_row_floats_for(model);
    size_t elems = (size_t)model->cfg.n_layer * model->cfg.n_ctx * row;
    if (kv_type == OC_KV_Q8) {
        size_t groups = (size_t)model->cfg.n_layer * model->cfg.n_ctx
                      * model->cfg.n_head_kv;
        return 2 * (elems * sizeof(int8_t) + groups * sizeof(float));
    }
    return 2 * elems * sizeof(float);
}

OcError oc_llama_session_init(OcLlamaModel *model, OcLlamaSession *out)
{
    return oc_llama_session_init_kv(model, out, kv_type_from_env());
}

OcError oc_llama_session_init_kv(OcLlamaModel *model, OcLlamaSession *out,
                                 OcKvCacheType kv_type)
{
    if (model == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    /* uses_geglu is fully handled by forward_dense_ffn (GeGLU vs SwiGLU),
     * so it is NOT rejected here. MLA is handled by forward_mla_attention.
     *
     * The MLA KV cache is still the EXPANDED per-head K/V, n_head * head_dim
     * floats per row rather than the kv_lora + rope latent it decompresses
     * from. That is 24x larger than it needs to be (7.5 MB/token on
     * LongCat-2.0 against 88 KB), which caps usable context well below the
     * model's 256k. Correct, but the obvious thing to fix next. */
    if (model->cfg.num_experts > 0 && model->cfg.expert_intermediate_size == 0) {
        if (model->cfg.n_ff == 0) return OC_ERR_MODEL;
        model->cfg.expert_intermediate_size = model->cfg.n_ff;
    }
    memset(out, 0, sizeof(*out));
    out->model = model;
    /* For MLA, each head has its own K/V (no GQA sharing); for Gemma 4 this is
     * the max over the two layer geometries. See kv_row_floats_for(). */
    out->kv_row_floats = kv_row_floats_for(model);
    size_t per_layer = (size_t)model->cfg.n_ctx * out->kv_row_floats;
    size_t total = (size_t)model->cfg.n_layer * per_layer;

    /* The Q8 quantization group is one kv head's row; the attention read
     * strides by head_dim. If those disagree the offsets would not line up,
     * so fall back to f32 rather than compute the wrong thing. */
    out->kv_group = model->cfg.kv_head_dim;
    /* Q8 KV indexes its scale array at a single (n_head_kv, kv_head_dim)
     * stride. Gemma 4's two layer geometries give two different strides, so
     * the scales for sliding and global layers would alias. Rather than carry
     * a per-layer scale layout for a path that is not the bottleneck, use f32
     * KV there. */
    if (kv_type == OC_KV_Q8 && model->cfg.uses_gemma4) {
        oc_log(OC_LOG_WARN, "llama: Q8 KV does not support Gemma 4's "
               "per-layer KV geometry; using f32 KV");
        kv_type = OC_KV_F32;
    }
    if (kv_type == OC_KV_Q8 && model->cfg.head_dim != model->cfg.kv_head_dim) {
        oc_log(OC_LOG_WARN, "llama: Q8 KV needs head_dim == kv_head_dim "
               "(%u vs %u); using f32 KV",
               model->cfg.head_dim, model->cfg.kv_head_dim);
        kv_type = OC_KV_F32;
    }
    out->kv_type = kv_type;

    if (kv_type == OC_KV_Q8) {
        size_t groups = (size_t)model->cfg.n_layer * model->cfg.n_ctx
                      * model->cfg.n_head_kv;
        out->kv_k_q = xcalloc(total, sizeof(int8_t));
        out->kv_v_q = xcalloc(total, sizeof(int8_t));
        out->kv_k_scale = xcalloc(groups, sizeof(float));
        out->kv_v_scale = xcalloc(groups, sizeof(float));
        if (!out->kv_k_q || !out->kv_v_q || !out->kv_k_scale ||
            !out->kv_v_scale) {
            oc_llama_session_free(out);
            return OC_ERR_OOM;
        }
    } else {
        out->kv_k = xcalloc(total, sizeof(float));
        out->kv_v = xcalloc(total, sizeof(float));
    }
    size_t maxw = model->cfg.n_embd > model->cfg.n_ff ? model->cfg.n_embd
                                                     : model->cfg.n_ff;
    if (model->cfg.expert_intermediate_size > maxw) {
        maxw = model->cfg.expert_intermediate_size;
    }
    out->x = xcalloc(model->cfg.n_embd, sizeof(float));
    out->normed = xcalloc(model->cfg.n_embd, sizeof(float));
    out->q = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
    out->k = xcalloc(out->kv_row_floats, sizeof(float));
    out->v = xcalloc(out->kv_row_floats, sizeof(float));
    out->attn_out = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
    out->ffn_gate = xcalloc(model->cfg.n_ff, sizeof(float));
    out->ffn_up = xcalloc(model->cfg.n_ff, sizeof(float));
    out->dequant_temp = xcalloc(maxw, sizeof(float));
    out->logits = xcalloc(model->cfg.vocab_size, sizeof(float));
    /* MoE temporaries (only allocated when num_experts > 0). */
    if (model->cfg.num_experts > 0) {
        out->router_logits = xcalloc(model->cfg.num_experts, sizeof(float));
        out->expert_gate = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_up   = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_out  = xcalloc(model->cfg.n_embd, sizeof(float));
        out->shexp_gate  = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->shexp_up    = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->shexp_out   = xcalloc(model->cfg.n_embd, sizeof(float));
    }
    oc_log(OC_LOG_INFO, "llama: KV cache %s, %.1f MB (%.1f MB as f32)",
           out->kv_type == OC_KV_Q8 ? "int8" : "f32",
           (double)oc_llama_kv_cache_bytes(model, out->kv_type) / 1e6,
           (double)oc_llama_kv_cache_bytes(model, OC_KV_F32) / 1e6);

    /* Q8 leaves kv_k/kv_v NULL; its buffers were checked at allocation. */
    if ((out->kv_type == OC_KV_F32 && (!out->kv_k || !out->kv_v)) ||
        !out->x || !out->normed || !out->q ||
        !out->k || !out->v || !out->attn_out || !out->ffn_gate ||
        !out->ffn_up || !out->dequant_temp || !out->logits) {
        oc_llama_session_free(out);
        return OC_ERR_OOM;
    }
    if (model->cfg.num_experts > 0 &&
        (!out->router_logits || !out->expert_gate || !out->expert_up ||
         !out->expert_out || !out->shexp_gate || !out->shexp_up || !out->shexp_out)) {
        oc_llama_session_free(out);
        return OC_ERR_OOM;
    }
    /* MLA temporaries. */
    if (model->cfg.uses_mla) {
        out->mla_c_q = xcalloc(model->cfg.mla_q_lora_dim, sizeof(float));
        out->mla_c_kv = xcalloc(model->cfg.mla_kv_lora_dim, sizeof(float));
        out->mla_q_full = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
        out->mla_kv_compressed = xcalloc(model->cfg.mla_kv_lora_dim + model->cfg.mla_q_rope_dim, sizeof(float));
        if (!out->mla_c_q || !out->mla_c_kv || !out->mla_q_full || !out->mla_kv_compressed) {
            oc_llama_session_free(out);
            return OC_ERR_OOM;
        }
    }
    /* KV cache quantization (Q8_0-style: 32-element blocks with f16 scale). */
    out->pos = 0;
    return OC_OK;
}

void oc_llama_session_reset(OcLlamaSession *sess)
{
    if (sess) sess->pos = 0;
}

void oc_llama_session_rewind(OcLlamaSession *sess, uint32_t pos)
{
    if (sess) sess->pos = pos;
}

void oc_llama_session_free(OcLlamaSession *sess)
{
    if (sess == NULL) return;
    free(sess->kv_k); free(sess->kv_v);
    free(sess->kv_k_q); free(sess->kv_v_q);
    free(sess->kv_k_scale); free(sess->kv_v_scale);
    free(sess->x); free(sess->normed);
    free(sess->q); free(sess->k); free(sess->v);
    free(sess->attn_out);
    free(sess->ffn_gate); free(sess->ffn_up);
    free(sess->dequant_temp);
    free(sess->logits);
    free(sess->router_logits);
    free(sess->expert_gate); free(sess->expert_up); free(sess->expert_out);
    free(sess->shexp_gate); free(sess->shexp_up); free(sess->shexp_out);
    free(sess->mla_c_q); free(sess->mla_c_kv);
    free(sess->mla_q_full); free(sess->mla_kv_compressed);
    memset(sess, 0, sizeof(*sess));
}

void oc_llama_free(OcLlamaModel *model)
{
    if (model == NULL) return;
    if (model->layers) {
        for (uint32_t i = 0; i < model->cfg.n_layer; i++) {
            free(model->layers[i].attn_norm);
            free(model->layers[i].ffn_norm);
            free(model->layers[i].attn_norm_bias);
            free(model->layers[i].ffn_norm_bias);
            free(model->layers[i].attn_q_bias);
            free(model->layers[i].attn_k_bias);
            free(model->layers[i].attn_v_bias);
            free(model->layers[i].mla_q_a_norm);
            free(model->layers[i].mla_kv_a_norm);
            free(model->layers[i].exp_probs_b);
            free(model->layers[i].attn_q_norm);
            free(model->layers[i].attn_k_norm);
            free(model->layers[i].post_attention_norm);
            free(model->layers[i].post_ffw_norm);
        }
        free(model->layers);
    }
    free(model->final_norm);
    free(model->final_norm_bias);
    free(model->cfg.layer_is_swa);
    oc_gguf_map_free(&model->gguf);
    memset(model, 0, sizeof(*model));
}

/* ─── Forward pass ─────────────────────────────────────────────────────── */

/* Embed one token: dequantize row `token` of tok_embeddings into `dst`. */
static void embed_token(OcLlamaSession *s, uint32_t token)
{
    OcWeightView *w = &s->model->tok_embeddings;
    if (token >= s->model->cfg.vocab_size) token = s->model->cfg.vocab_size - 1;
    if (w->qtype == OC_QUANT_F32) {
        memcpy(s->x, w->data + (size_t)token * w->row_bytes,
               s->model->cfg.n_embd * sizeof(float));
    } else {
        oc_quant_dequant_row(w->qtype,
            w->data + (size_t)token * w->row_bytes, w->row_bytes,
            s->x, s->model->cfg.n_embd);
    }
    /* Gemma scales the token embedding by sqrt(n_embd) once, here — it is a
     * property of the embedding, not of each RMSNorm. The pre-existing
     * `norm_scale` handling in forward_layer multiplies every normed
     * activation instead, which is a different (and for Gemma 4, wrong)
     * computation; uses_gemma4 takes this path and leaves that one alone. */
    if (s->model->cfg.uses_gemma4 && s->model->cfg.norm_scale != 1.0f) {
        const float sc = s->model->cfg.norm_scale;
        for (size_t i = 0; i < s->model->cfg.n_embd; i++) s->x[i] *= sc;
    }
}

/* matvec wrapper: pick f32 or quantized path based on qtype. */
static void matvec(const OcWeightView *w, const float *in, float *out,
                   float *temp)
{
    if (w->qtype == OC_QUANT_F32) {
        oc_matvec_f32((const float *)w->data, w->rows, w->cols, in, out);
    } else {
        oc_matvec_quantized(w->qtype, w->data, w->rows, w->cols, w->row_bytes,
                            in, out, temp);
    }
}

/* Online-softmax attention for one Q head against all cached K/V up to `pos`.
 * GQA: Q head h attends to KV head (h / group_size). */
static void forward_layer(OcLlamaSession *s, uint32_t layer);
static void forward_dense_ffn(OcLlamaSession *s, const OcLlamaLayer *L);
static void attention_head(const OcLlamaSession *s, uint32_t head,
                           uint32_t layer, const float *q_vec, float *out_vec)
{
    const OcLlamaConfig *c = &s->model->cfg;
    /* Geometry is per-layer: on Gemma 4 a sliding layer has head_dim 256 with
     * 16 KV heads while a global layer has 512 with 4. The loader resolved
     * both into the layer, so read them rather than the model-wide config. */
    const OcLlamaLayer *GL = &s->model->layers[layer];
    size_t hd = GL->head_dim ? (size_t)GL->head_dim : (size_t)c->head_dim;
    uint32_t n_kv = GL->n_head_kv ? GL->n_head_kv : c->n_head_kv;
    uint32_t group = c->n_head / n_kv;
    uint32_t kv_head = head / group;
    size_t kv_off = ((size_t)layer * c->n_ctx + 0) * s->kv_row_floats
                  + (size_t)kv_head * hd;
    const float *k_layer = s->kv_k;
    const float *v_layer = s->kv_v;
    /* Gemma 4 uses no pre-attention scaling; everything else uses the usual
     * 1/sqrt(head_dim). */
    float scale = (c->attn_scale > 0.0f) ? c->attn_scale
                                         : (1.0f / sqrtf((float)hd));

    /* Online softmax (Milakov & Gimelshein 2018). seq_len = pos+1. */
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    for (size_t i = 0; i < hd; i++) out_vec[i] = 0.0f;

    /* Sliding window: only attend to the last `sliding_window` tokens
     * for layers matching the alternating pattern. Layer L uses sliding
     * window when (L % pattern) == 1 (i.e., every other layer for pattern=2). */
    int64_t seq_len = s->pos + 1;
    int64_t start = 0;
    if (GL->sliding_window > 0) {
        /* Gemma 4 (and any model whose loader filled in a per-layer window):
         * the pattern came from metadata, so trust the resolved value rather
         * than re-deriving it from a modulus. */
        int64_t sw = (int64_t)GL->sliding_window;
        start = (seq_len > sw) ? (seq_len - sw) : 0;
    } else if (!c->uses_gemma4 && c->sliding_window > 0 &&
               c->sliding_window_pattern > 1) {
        if (layer % c->sliding_window_pattern == 1) {
            /* Sliding window layer: only attend to recent tokens. */
            int64_t sw = (int64_t)c->sliding_window;
            start = (seq_len > sw) ? (seq_len - sw) : 0;
        }
    }

    /* Q8: the dot product runs against the int8 codes and is scaled once at
     * the end, and the V accumulation folds the scale into exp_score — so
     * neither K nor V is ever materialized back to f32. */
    const bool q8 = (s->kv_type == OC_KV_Q8);
    const size_t sc_stride = c->n_head_kv;
    const size_t sc_base = (size_t)layer * c->n_ctx * sc_stride + kv_head;

    for (int64_t t = start; t < seq_len; t++) {
        float dot = 0.0f;
        const int8_t *kq_t = NULL;
        const int8_t *vq_t = NULL;
        float v_scale = 0.0f;
        if (q8) {
            kq_t = s->kv_k_q + kv_off + (size_t)t * s->kv_row_floats;
            vq_t = s->kv_v_q + kv_off + (size_t)t * s->kv_row_floats;
            for (size_t i = 0; i < hd; i++) dot += q_vec[i] * (float)kq_t[i];
            dot *= s->kv_k_scale[sc_base + (size_t)t * sc_stride];
            v_scale = s->kv_v_scale[sc_base + (size_t)t * sc_stride];
        } else {
            const float *k_t = k_layer + kv_off + (size_t)t * s->kv_row_floats;
            for (size_t i = 0; i < hd; i++) dot += q_vec[i] * k_t[i];
        }
        float score = dot * scale;
        float new_max = (score > run_max) ? score : run_max;
        float exp_factor = expf(run_max - new_max);
        float exp_score = expf(score - new_max);
        for (size_t i = 0; i < hd; i++) out_vec[i] *= exp_factor;
        if (q8) {
            const float w = exp_score * v_scale;
            for (size_t i = 0; i < hd; i++) out_vec[i] += w * (float)vq_t[i];
        } else {
            const float *v_t = v_layer + kv_off + (size_t)t * s->kv_row_floats;
            for (size_t i = 0; i < hd; i++) out_vec[i] += exp_score * v_t[i];
        }
        run_sum = run_sum * exp_factor + exp_score;
        run_max = new_max;
    }
    float inv = 1.0f / run_sum;
    for (size_t i = 0; i < hd; i++) out_vec[i] *= inv;
}

/* ─── Dense FFN (Llama/Mistral/Qwen2-dense path) ──────────────────────── */

static void forward_dense_ffn(OcLlamaSession *s, const OcLlamaLayer *L)
{
    const OcLlamaConfig *c = &s->model->cfg;
    /* FFN: gate = act(W_gate·x) * (W_up·x); out = W_down·gate.
     * Llama/Mistral/Qwen use SwiGLU (silu). Gemma uses GeGLU (gelu). */
    matvec(&L->ffn_gate, s->normed, s->ffn_gate, s->dequant_temp);
    matvec(&L->ffn_up,   s->normed, s->ffn_up,   s->dequant_temp);
    if (c->uses_geglu) {
        oc_geglu_inplace_f32(s->ffn_gate, s->ffn_up, c->n_ff);
    } else {
        oc_swiglu_inplace_f32(s->ffn_gate, s->ffn_up, c->n_ff);
    }
    matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    /* Gemma sandwich norm on the FFN branch output, mirroring the attention
     * branch in forward_layer. */
    if (L->post_ffw_norm != NULL) {
        oc_rms_norm_f32(s->normed, L->post_ffw_norm, s->attn_out,
                        c->n_embd, c->rms_norm_eps);
        memcpy(s->normed, s->attn_out, c->n_embd * sizeof(float));
    }
    /* Residual add. */
    for (size_t i = 0; i < c->n_embd; i++) s->x[i] += s->normed[i];
}

/* ─── MoE FFN (Qwen3-MoE / Mixtral path) ────────────────────────────────
 *
 * Port of oxidize-core inference.rs::moe_ffn_forward_weights +
 * layers.rs MoE branch. For Qwen3-MoE: softmax gating over all experts,
 * top-k selection, renormalize top-k weights, per-expert SwiGLU FFN,
 * weighted sum, then add the shared expert (optionally sigmoid-gated).
 *
 * Expert weights are STACKED in ffn_gate_exps/ffn_up_exps/ffn_down_exps.
 * Expert i occupies rows [i*exp_i_size, (i+1)*exp_i_size) in gate/up and
 * [i*n_embd, (i+1)*n_embd) in down. We synthesize a per-expert OcWeightView
 * by slicing the stacked tensor's data pointer + per-expert row count.
 *
 * Each expert's down-projection output goes into s->shexp_out as a temp
 * (n_embd-sized), then is scaled by w and accumulated into s->expert_out.
 */
static void forward_moe_ffn(OcLlamaSession *s, const OcLlamaLayer *L)
{
    const OcLlamaConfig *c = &s->model->cfg;
    uint32_t n_exp = c->num_experts;
    uint32_t k = c->num_experts_per_tok;
    uint32_t i_size = c->expert_intermediate_size;
    if (i_size == 0) i_size = c->n_ff;

    /* 1. Router logits: ffn_gate_inp @ normed → [num_experts]. */
    matvec(&L->ffn_gate_inp, s->normed, s->router_logits, s->dequant_temp);

    /* 2. Gating: softmax (Qwen3-MoE) or sigmoid (DeepSeek). */
    if (!c->expert_gating_sigmoid) {
        float mx = s->router_logits[0];
        for (uint32_t i = 1; i < n_exp; i++) {
            if (s->router_logits[i] > mx) mx = s->router_logits[i];
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < n_exp; i++) {
            s->router_logits[i] = expf(s->router_logits[i] - mx);
            sum += (double)s->router_logits[i];
        }
        if (sum > 0.0) {
            float inv = (float)(1.0 / sum);
            for (uint32_t i = 0; i < n_exp; i++) s->router_logits[i] *= inv;
        }
    } else {
        for (uint32_t i = 0; i < n_exp; i++) {
            s->router_logits[i] = 1.0f / (1.0f + expf(-s->router_logits[i]));
        }
    }

    /* 3. Top-k selection by descending weight (partial selection sort). */
    uint32_t sel_buf[256];
    uint32_t *sel = (n_exp <= 256) ? sel_buf
                                   : (uint32_t *)malloc(n_exp * sizeof(uint32_t));
    if (sel == NULL) { forward_dense_ffn(s, L); return; }
    for (uint32_t i = 0; i < n_exp; i++) sel[i] = i;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < n_exp; j++) {
            if (s->router_logits[sel[j]] > s->router_logits[sel[best]]) best = j;
        }
        uint32_t tmp = sel[i]; sel[i] = sel[best]; sel[best] = tmp;
    }

    /* 4. Renormalize top-k weights (norm_topk_prob). */
    double weight_norm = 0.0;
    for (uint32_t i = 0; i < k; i++) weight_norm += (double)s->router_logits[sel[i]];
    if (weight_norm <= 0.0) weight_norm = 1.0;
    float routed_scale = c->expert_weights_scale;

    /* 5. Per-expert SwiGLU FFN, accumulated into s->expert_out (zeroed).
     *    Each expert's down output goes into s->shexp_out (temp, n_embd),
     *    then is scaled by w and added to s->expert_out. */
    memset(s->expert_out, 0, c->n_embd * sizeof(float));
    size_t gate_row_bytes = L->ffn_gate_exps.row_bytes;
    size_t up_row_bytes   = L->ffn_up_exps.row_bytes;
    size_t down_row_bytes = L->ffn_down_exps.row_bytes;

    for (uint32_t ei = 0; ei < k; ei++) {
        uint32_t idx = sel[ei];
        float w = routed_scale * (float)(s->router_logits[idx] / weight_norm);

        OcWeightView gate_v = L->ffn_gate_exps;
        gate_v.data = L->ffn_gate_exps.data + (size_t)idx * i_size * gate_row_bytes;
        gate_v.rows = i_size;
        OcWeightView up_v = L->ffn_up_exps;
        up_v.data = L->ffn_up_exps.data + (size_t)idx * i_size * up_row_bytes;
        up_v.rows = i_size;
        OcWeightView down_v = L->ffn_down_exps;
        down_v.data = L->ffn_down_exps.data + (size_t)idx * c->n_embd * down_row_bytes;
        down_v.rows = c->n_embd;

        matvec(&gate_v, s->normed, s->expert_gate, s->dequant_temp);
        matvec(&up_v,   s->normed, s->expert_up,   s->dequant_temp);
        oc_swiglu_inplace_f32(s->expert_gate, s->expert_up, i_size);
        /* down → temp (shexp_out), then accumulate w*temp into expert_out. */
        matvec(&down_v, s->expert_gate, s->shexp_out, s->dequant_temp);
        for (size_t i = 0; i < c->n_embd; i++) {
            s->expert_out[i] += w * s->shexp_out[i];
        }
    }
    if (sel != sel_buf) free(sel);

    /* 6. Shared expert (always active, added with weight 1.0). Optional
     *    sigmoid gate via ffn_gate_inp_shexp (Qwen2-MoE shared_expert_gate). */
    if (L->ffn_gate_shexp.data != NULL && L->ffn_up_shexp.data != NULL &&
        L->ffn_down_shexp.data != NULL) {
        matvec(&L->ffn_gate_shexp, s->normed, s->shexp_gate, s->dequant_temp);
        matvec(&L->ffn_up_shexp,   s->normed, s->shexp_up,   s->dequant_temp);
        oc_swiglu_inplace_f32(s->shexp_gate, s->shexp_up, i_size);
        matvec(&L->ffn_down_shexp, s->shexp_gate, s->shexp_out, s->dequant_temp);
        /* Optional sigmoid gate. */
        if (L->ffn_gate_inp_shexp.data != NULL) {
            float gate_logit = 0.0f;
            matvec(&L->ffn_gate_inp_shexp, s->normed, &gate_logit, s->dequant_temp);
            float scale = 1.0f / (1.0f + expf(-gate_logit));
            for (size_t i = 0; i < c->n_embd; i++) s->shexp_out[i] *= scale;
        }
        for (size_t i = 0; i < c->n_embd; i++) s->expert_out[i] += s->shexp_out[i];
    }

    /* 7. Residual add: x += ffn_out. */
    for (size_t i = 0; i < c->n_embd; i++) s->x[i] += s->expert_out[i];
}

/* ─── DeepSeek MLA (Multi-head Latent Attention) forward ────────────────
 *
 * Port of oxidize-core inference/layers.rs::deepseek_mla_layer.
 *
 * Q path:  x → q_a_proj → q_a_norm → q_b_proj → split (q_nope | q_pe)
 * KV path: x → kv_a_proj_with_mqa → [kv_a_norm'd latent | k_pe]
 *          per-head: k_b_proj (k_nope), v_b_proj (v)
 * RoPE on q_pe and k_pe (decoupled), then standard attention.
 */
static void forward_mla_attention(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    uint32_t n_embd = c->n_embd;
    uint32_t n_head = c->n_head;
    uint32_t q_lora = c->mla_q_lora_dim;
    uint32_t kv_lora = c->mla_kv_lora_dim;
    uint32_t q_rope = c->mla_q_rope_dim;
    uint32_t k_nope_hd = c->mla_kv_nope_head_dim;
    uint32_t v_hd = c->mla_v_head_dim;
    uint32_t q_hd = c->head_dim;

    /* 1. Q down-projection: q_a_proj @ normed → c_q [q_lora] */
    matvec(&L->mla_q_a, s->normed, s->mla_c_q, s->dequant_temp);

    /* 2. RMSNorm c_q with q_a_norm. */
    oc_rms_norm_f32(s->mla_c_q, L->mla_q_a_norm, s->mla_c_q,
                    q_lora, c->rms_norm_eps);

    /* 3. Q up-projection: q_b_proj @ c_q → q_full [n_head * q_hd] */
    matvec(&L->mla_q_b, s->mla_c_q, s->mla_q_full, s->dequant_temp);

    /* 4. KV down-projection: kv_a_proj_with_mqa @ normed → kv_compressed */
    matvec(&L->mla_kv_a_mqa, s->normed, s->mla_kv_compressed, s->dequant_temp);

    /* 5. RMSNorm the kv_lora portion of kv_compressed. */
    oc_rms_norm_f32(s->mla_kv_compressed, L->mla_kv_a_norm,
                    s->mla_c_kv, kv_lora, c->rms_norm_eps);
    memcpy(s->mla_kv_compressed, s->mla_c_kv, kv_lora * sizeof(float));

    /* 6. RoPE on k_pe (the trailing q_rope slots of kv_compressed).
     *
     * YaRN changes the ROTATION FREQUENCIES, not just an amplitude, so it is
     * still required here even for LongCat where the mscale ratio cancels to
     * 1.0 -- a 120x context extension without the frequency ramp would place
     * every position wrong. */
    float *k_pe = s->mla_kv_compressed + kv_lora;
    if (c->yarn_factor > 1.0f) {
        oc_apply_rope_yarn_f32(k_pe, k_pe, q_rope, q_rope, s->pos,
                               c->rope_theta, c->yarn_factor, c->yarn_orig_ctx);
    } else {
        oc_apply_rope_f32(k_pe, k_pe, q_rope, q_rope, s->pos, c->rope_theta);
    }

    /* 7. Per-head K and V up-projection. */
    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos) * s->kv_row_floats;
    float *k_cache = s->kv_k + kv_off;
    float *v_cache = s->kv_v + kv_off;

    for (uint32_t h = 0; h < n_head; h++) {
        OcWeightView k_b_h = L->mla_k_b;
        k_b_h.data = L->mla_k_b.data + (size_t)h * k_nope_hd * L->mla_k_b.row_bytes;
        k_b_h.rows = k_nope_hd;
        float *k_nope = k_cache + (size_t)h * q_hd;
        matvec(&k_b_h, s->mla_c_kv, k_nope, s->dequant_temp);
        memcpy(k_nope + k_nope_hd, k_pe, q_rope * sizeof(float));

        OcWeightView v_b_h = L->mla_v_b;
        v_b_h.data = L->mla_v_b.data + (size_t)h * v_hd * L->mla_v_b.row_bytes;
        v_b_h.rows = v_hd;
        float *v_out = v_cache + (size_t)h * q_hd;
        matvec(&v_b_h, s->mla_c_kv, v_out, s->dequant_temp);
    }

    /* 8. RoPE on q_pe (the trailing q_rope slots of each q head). Must match
     * the k_pe treatment above exactly, or Q and K rotate apart. */
    for (uint32_t h = 0; h < n_head; h++) {
        float *q_pe = s->mla_q_full + h * q_hd + k_nope_hd;
        if (c->yarn_factor > 1.0f) {
            oc_apply_rope_yarn_f32(q_pe, q_pe, q_rope, q_rope, s->pos,
                                   c->rope_theta, c->yarn_factor,
                                   c->yarn_orig_ctx);
        } else {
            oc_apply_rope_f32(q_pe, q_pe, q_rope, q_rope, s->pos, c->rope_theta);
        }
    }

    /* 9. Attention per head (MLA: each head has own K/V, group=1).
     *
     * Q and K are q_hd wide (nope + rope, 192 for LongCat) but V is only
     * v_hd (128). The cache strides both by q_hd for a uniform layout, so V
     * rows carry q_hd - v_hd trailing slots that were never written; the
     * accumulation must stop at v_hd rather than read them.
     *
     * The output is written PACKED at v_hd, not at the Q stride: o_proj
     * takes n_head * v_hd inputs, so a q_hd-strided buffer would feed it
     * every head interleaved with dead floats and run out of input a third
     * of the way through the heads. */
    float scale = 1.0f / sqrtf((float)q_hd);
    if (c->yarn_factor > 1.0f && c->yarn_mscale_all_dim > 0.0f) {
        /* deepseek_yarn moves the mscale correction onto the logits; see
         * oc_rope_deepseek_yarn_scales(). */
        oc_rope_deepseek_yarn_scales(c->yarn_factor, c->yarn_mscale,
                                      c->yarn_mscale_all_dim, q_hd,
                                      NULL, &scale);
    }
    for (uint32_t h = 0; h < n_head; h++) {
        float *q_vec = s->mla_q_full + h * q_hd;
        float *out_vec = s->attn_out + (size_t)h * v_hd;
        const float *k_head = k_cache + (size_t)h * q_hd;
        const float *v_head = v_cache + (size_t)h * q_hd;

        float run_max = -INFINITY;
        float run_sum = 0.0f;
        for (size_t i = 0; i < v_hd; i++) out_vec[i] = 0.0f;

        int64_t seq_len = s->pos + 1;
        for (int64_t t = 0; t < seq_len; t++) {
            const float *k_t = k_head + (size_t)t * s->kv_row_floats;
            float dot = 0.0f;
            for (size_t i = 0; i < q_hd; i++) dot += q_vec[i] * k_t[i];
            float score = dot * scale;
            float new_max = (score > run_max) ? score : run_max;
            float exp_factor = expf(run_max - new_max);
            float exp_score = expf(score - new_max);
            for (size_t i = 0; i < v_hd; i++) out_vec[i] *= exp_factor;
            const float *v_t = v_head + (size_t)t * s->kv_row_floats;
            for (size_t i = 0; i < v_hd; i++) out_vec[i] += exp_score * v_t[i];
            run_sum = run_sum * exp_factor + exp_score;
            run_max = new_max;
        }
        float inv = (run_sum > 0.0f) ? 1.0f / run_sum : 0.0f;
        for (size_t i = 0; i < v_hd; i++) out_vec[i] *= inv;
    }

    /* 10. Output projection. */
    matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

static void forward_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    /* Per-layer geometry (see attention_head). */
    size_t hd = L->head_dim ? (size_t)L->head_dim : (size_t)c->head_dim;
    const uint32_t n_kv = L->n_head_kv ? L->n_head_kv : c->n_head_kv;
    const uint32_t rope_dim = L->head_dim ? L->rope_dim : c->rope_dim;
    const float rope_theta = L->head_dim ? L->rope_theta : c->rope_theta;

    /* Pre-attention RMSNorm (+ Gemma scaling).
     * Gemma 4 folds its sqrt(n_embd) factor into the embedding instead (see
     * embed_token), so norm_scale must not be applied again per layer. */
    oc_rms_norm_f32(s->x, L->attn_norm, s->normed, c->n_embd, c->rms_norm_eps);
    if (!c->uses_gemma4 && c->norm_scale != 1.0f) {
        for (size_t i = 0; i < c->n_embd; i++) s->normed[i] *= c->norm_scale;
    }

    /* Attention: MLA or standard GQA. */
    if (c->uses_mla && L->mla_kv_a_mqa.data != NULL) {
        forward_mla_attention(s, layer);
    } else {
        /* Q/K/V projections, plus the optional projection biases that
         * Qwen2-family models carry. The bias is added before RoPE, as in
         * llama.cpp's build_qkv / HF's q_proj(x) + b_q. */
        matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
        matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
        matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);
        if (L->attn_q_bias != NULL) {
            size_t nq = (size_t)c->n_head * hd;
            for (size_t i = 0; i < nq; i++) s->q[i] += L->attn_q_bias[i];
        }
        if (L->attn_k_bias != NULL) {
            for (size_t i = 0; i < s->kv_row_floats; i++)
                s->k[i] += L->attn_k_bias[i];
        }
        if (L->attn_v_bias != NULL) {
            for (size_t i = 0; i < s->kv_row_floats; i++)
                s->v[i] += L->attn_v_bias[i];
        }

        /* Gemma Q/K RMSNorm: applied per head, after projection and BEFORE
         * RoPE. The norm weight is the layer's own head_dim long. */
        if (L->attn_q_norm != NULL) {
            for (uint32_t h = 0; h < c->n_head; h++) {
                oc_rms_norm_f32(s->q + h * hd, L->attn_q_norm,
                                s->dequant_temp, hd, c->rms_norm_eps);
                memcpy(s->q + h * hd, s->dequant_temp, hd * sizeof(float));
            }
        }
        if (L->attn_k_norm != NULL) {
            for (uint32_t h = 0; h < n_kv; h++) {
                oc_rms_norm_f32(s->k + h * hd, L->attn_k_norm,
                                s->dequant_temp, hd, c->rms_norm_eps);
                memcpy(s->k + h * hd, s->dequant_temp, hd * sizeof(float));
            }
        }

        /* Gemma 4 RMS-normalizes V too, with no weight — a plain
         * normalization applied per KV head, before the cache write. V never
         * gets RoPE, so this is the last thing that touches it. */
        if (c->v_rms_norm) {
            for (uint32_t h = 0; h < n_kv; h++) {
                float *vh = s->v + (size_t)h * hd;
                double ss = 0.0;
                for (size_t i = 0; i < hd; i++) ss += (double)vh[i] * vh[i];
                const float inv =
                    1.0f / sqrtf((float)(ss / (double)hd) + c->rms_norm_eps);
                for (size_t i = 0; i < hd; i++) vh[i] *= inv;
            }
        }

        /* RoPE on Q (per head) and K (per kv head). YaRN when configured.
         * rope_dim/rope_theta are the layer's: Gemma 4 rotates 256 dims at
         * base 1e4 on sliding layers and 512 at 1e6 on global ones. */
        for (uint32_t h = 0; h < c->n_head; h++) {
            if (c->yarn_factor > 0.0f) {
                oc_apply_rope_yarn_f32(s->q + h * hd, s->q + h * hd, hd,
                                       rope_dim, s->pos, rope_theta,
                                       c->yarn_factor, c->yarn_orig_ctx);
            } else {
                oc_apply_rope_f32(s->q + h * hd, s->q + h * hd, hd, rope_dim,
                                  s->pos, rope_theta);
            }
        }
        for (uint32_t h = 0; h < n_kv; h++) {
            if (c->yarn_factor > 0.0f) {
                oc_apply_rope_yarn_f32(s->k + h * hd, s->k + h * hd, hd,
                                       rope_dim, s->pos, rope_theta,
                                       c->yarn_factor, c->yarn_orig_ctx);
            } else {
                oc_apply_rope_f32(s->k + h * hd, s->k + h * hd, hd, rope_dim,
                                  s->pos, rope_theta);
            }
        }

        /* KV cache write at position `pos`. */
        size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos) * s->kv_row_floats;
        if (s->kv_type == OC_KV_Q8) {
            /* One scale per kv head, so each head's row quantizes against its
             * own magnitude rather than the whole row's. */
            size_t g = s->kv_group;
            size_t sc_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                          * c->n_head_kv;
            for (uint32_t h = 0; h < c->n_head_kv; h++) {
                kv_q8_encode(s->k + h * g, s->kv_k_q + kv_off + h * g,
                             &s->kv_k_scale[sc_off + h], g);
                kv_q8_encode(s->v + h * g, s->kv_v_q + kv_off + h * g,
                             &s->kv_v_scale[sc_off + h], g);
            }
        } else {
            memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
            memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));
        }

        /* Attention per head → s->attn_out. */
        for (uint32_t h = 0; h < c->n_head; h++) {
            attention_head(s, h, layer, s->q + h * hd, s->attn_out + h * hd);
        }

        /* Output projection. */
        matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
        /* Gemma "sandwich" norm: the attention branch output is normed again
         * before it rejoins the residual stream (HF post_attention_layernorm
         * on the branch, not on the input). Skipped when the tensor is absent,
         * which is every non-Gemma model. */
        if (L->post_attention_norm != NULL) {
            oc_rms_norm_f32(s->normed, L->post_attention_norm,
                            s->attn_out, c->n_embd, c->rms_norm_eps);
            memcpy(s->normed, s->attn_out, c->n_embd * sizeof(float));
        }
        /* Residual add (reusing normed as the projection output buffer). */
        for (size_t i = 0; i < c->n_embd; i++) s->x[i] += s->normed[i];
    }

    /* Pre-FFN RMSNorm (+ Gemma scaling; see the pre-attention norm above). */
    oc_rms_norm_f32(s->x, L->ffn_norm, s->normed, c->n_embd, c->rms_norm_eps);
    if (!c->uses_gemma4 && c->norm_scale != 1.0f) {
        for (size_t i = 0; i < c->n_embd; i++) s->normed[i] *= c->norm_scale;
    }

    /* FFN branch: MoE when the layer has a router, dense otherwise. */
    if (c->num_experts > 0 && L->ffn_gate_inp.data != NULL &&
        L->ffn_gate_exps.data != NULL) {
        forward_moe_ffn(s, L);
    } else {
        forward_dense_ffn(s, L);
    }

    /* Gemma 4 per-layer output scale. It multiplies the layer's whole output
     * — the running residual stream after the FFN residual add — not just one
     * branch. (llama.cpp gemma4.cpp: `cur = ggml_mul(cur, out_scale)` right
     * before `inpL = cur`.) 1.0 when the tensor is absent. */
    /* 0 means "unset", not "scale by zero". resolve_weights() seeds this to
     * 1.0, but a hand-built OcLlamaModel (tests, embedders) is typically
     * memset to zero, and treating that as a real scale silently zeroes the
     * whole residual stream — all-zero logits with no error anywhere. */
    if (L->layer_output_scale != 0.0f && L->layer_output_scale != 1.0f) {
        const float os = L->layer_output_scale;
        for (size_t i = 0; i < c->n_embd; i++) s->x[i] *= os;
    }
}

OcError oc_llama_forward(OcLlamaSession *sess, uint32_t token, float *logits_out)
{
    if (sess == NULL || sess->model == NULL) return OC_ERR_INVALID_ARG;
    if ((uint64_t)sess->pos >= sess->model->cfg.n_ctx) return OC_ERR_INVALID_ARG;

    /* Architecture dispatch: LayerNorm-family models use the dedicated
     * forward passes in arch_forward.c. */
    switch (sess->model->arch) {
    case OC_ARCH_GPT2:    return oc_arch_forward_gpt2(sess, token, logits_out);
    case OC_ARCH_GPTNEOX: return oc_arch_forward_gpt_neox(sess, token, logits_out);
    case OC_ARCH_FALCON:  return oc_arch_forward_falcon(sess, token, logits_out);
    default: break;
    }

    embed_token(sess, token);
    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        forward_layer(sess, l);
    }
    /* Final RMSNorm + lm_head. */
    OcLlamaModel *m = sess->model;
    oc_rms_norm_f32(sess->x, m->final_norm, sess->normed, m->cfg.n_embd,
                    m->cfg.rms_norm_eps);
    if (!m->cfg.uses_gemma4 && m->cfg.norm_scale != 1.0f) {
        for (size_t i = 0; i < m->cfg.n_embd; i++) sess->normed[i] *= m->cfg.norm_scale;
    }
    if (logits_out != NULL) {
        if (m->output.qtype == OC_QUANT_F32) {
            oc_matvec_f32((const float *)m->output.data, m->output.rows,
                          m->output.cols, sess->normed, logits_out);
        } else {
            oc_matvec_quantized(m->output.qtype, m->output.data, m->output.rows,
                                 m->output.cols, m->output.row_bytes,
                                 sess->normed, logits_out, sess->dequant_temp);
        }
        /* Gemma 4 softcaps the final logits: l = tanh(l/c) * c. This bounds
         * them to (-c, c) and materially reshapes the sampling distribution,
         * so it is part of the model, not an optional nicety. */
        if (m->cfg.logit_softcap > 0.0f) {
            const float cap = m->cfg.logit_softcap;
            const float inv = 1.0f / cap;
            for (size_t i = 0; i < m->cfg.vocab_size; i++)
                logits_out[i] = tanhf(logits_out[i] * inv) * cap;
        }
    }
    sess->pos++;
    return OC_OK;
}

/* ─── Batched decode ─────────────────────────────────────────────────── */

OcError oc_batch_session_init(OcLlamaModel *model, size_t max_seqs,
                               OcBatchSession *out)
{
    if (model == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    /* Batch decode only implements the RMSNorm Llama-family layer; reject
     * LayerNorm architectures (they use oc_llama_forward's dispatch). */
    if (model->arch == OC_ARCH_GPT2 || model->arch == OC_ARCH_GPTNEOX ||
        model->arch == OC_ARCH_FALCON)
        return OC_ERR_MODEL;
    if (model->cfg.num_experts > 0 && model->cfg.expert_intermediate_size == 0) {
        if (model->cfg.n_ff == 0) return OC_ERR_MODEL;
        model->cfg.expert_intermediate_size = model->cfg.n_ff;
    }
    if (max_seqs == 0) max_seqs = 1;
    if (max_seqs > OC_MAX_BATCH_SEQ) max_seqs = OC_MAX_BATCH_SEQ;
    memset(out, 0, sizeof(*out));
    out->model = model;
    out->max_seqs = max_seqs;
    if (model->cfg.uses_mla) {
        out->kv_row_floats = (size_t)model->cfg.n_head * model->cfg.head_dim;
    } else {
        out->kv_row_floats = (size_t)model->cfg.n_head_kv * model->cfg.kv_head_dim;
    }
    size_t per_layer = (size_t)model->cfg.n_ctx * out->kv_row_floats * max_seqs;
    size_t total = (size_t)model->cfg.n_layer * per_layer;
    out->kv_k = xcalloc(total, sizeof(float));
    out->kv_v = xcalloc(total, sizeof(float));
    size_t maxw = model->cfg.n_embd > model->cfg.n_ff ? model->cfg.n_embd : model->cfg.n_ff;
    if (model->cfg.expert_intermediate_size > maxw)
        maxw = model->cfg.expert_intermediate_size;
    out->x = xcalloc(model->cfg.n_embd, sizeof(float));
    out->normed = xcalloc(model->cfg.n_embd, sizeof(float));
    out->q = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
    out->k = xcalloc(out->kv_row_floats, sizeof(float));
    out->v = xcalloc(out->kv_row_floats, sizeof(float));
    out->attn_out = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
    out->ffn_gate = xcalloc(model->cfg.n_ff, sizeof(float));
    out->ffn_up = xcalloc(model->cfg.n_ff, sizeof(float));
    out->dequant_temp = xcalloc(maxw, sizeof(float));
    if (model->cfg.num_experts > 0) {
        out->router_logits = xcalloc(model->cfg.num_experts, sizeof(float));
        out->expert_gate = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_up = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_out = xcalloc(model->cfg.n_embd, sizeof(float));
        out->shexp_gate = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->shexp_up = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->shexp_out = xcalloc(model->cfg.n_embd, sizeof(float));
    }
    /* MLA workspace: same shapes as oc_llama_session_init. Shared across
     * sequences, so it must be sized for one sequence only. */
    if (model->cfg.uses_mla) {
        out->mla_c_q = xcalloc(model->cfg.mla_q_lora_dim, sizeof(float));
        out->mla_c_kv = xcalloc(model->cfg.mla_kv_lora_dim, sizeof(float));
        out->mla_q_full = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim,
                                  sizeof(float));
        out->mla_kv_compressed = xcalloc(model->cfg.mla_kv_lora_dim +
                                         model->cfg.mla_q_rope_dim, sizeof(float));
        if (!out->mla_c_q || !out->mla_c_kv || !out->mla_q_full ||
            !out->mla_kv_compressed) {
            oc_batch_session_free(out);
            return OC_ERR_OOM;
        }
    }
    if (!out->kv_k || !out->kv_v || !out->x || !out->normed || !out->q ||
        !out->k || !out->v || !out->attn_out || !out->ffn_gate ||
        !out->ffn_up || !out->dequant_temp ||
        (model->cfg.num_experts > 0 &&
         (!out->router_logits || !out->expert_gate || !out->expert_up ||
          !out->expert_out || !out->shexp_gate || !out->shexp_up ||
          !out->shexp_out))) {
        oc_batch_session_free(out);
        return OC_ERR_OOM;
    }
    return OC_OK;
}

OcError oc_batch_forward(OcBatchSession *bs, OcBatchSeq *seqs)
{
    if (bs == NULL || seqs == NULL) return OC_ERR_INVALID_ARG;
    OcLlamaModel *m = bs->model;
    /* Process each active sequence sequentially through the same weights.
     * This is a simple batch implementation - true parallel batch (multiple
     * sequences through the same matvec) requires a batched matvec. */
    for (size_t s = 0; s < bs->max_seqs; s++) {
        if (!seqs[s].active) continue;
        if (seqs[s].pos < 0 || (uint64_t)seqs[s].pos >= m->cfg.n_ctx)
            return OC_ERR_INVALID_ARG;
        /* Set up a temporary session view sharing the workspace. */
        OcLlamaSession tmp;
        tmp.model = m;
        size_t sequence_stride = (size_t)m->cfg.n_layer * m->cfg.n_ctx *
                                 bs->kv_row_floats;
        tmp.kv_k = bs->kv_k + s * sequence_stride;
        tmp.kv_v = bs->kv_v + s * sequence_stride;
        tmp.kv_row_floats = bs->kv_row_floats;
        tmp.pos = seqs[s].pos;
        tmp.x = bs->x;
        tmp.normed = bs->normed;
        tmp.q = bs->q;
        tmp.k = bs->k;
        tmp.v = bs->v;
        tmp.attn_out = bs->attn_out;
        tmp.ffn_gate = bs->ffn_gate;
        tmp.ffn_up = bs->ffn_up;
        tmp.dequant_temp = bs->dequant_temp;
        tmp.logits = NULL;
        tmp.router_logits = bs->router_logits;
        tmp.expert_gate = bs->expert_gate;
        tmp.expert_up = bs->expert_up;
        tmp.expert_out = bs->expert_out;
        tmp.shexp_gate = bs->shexp_gate;
        tmp.shexp_up = bs->shexp_up;
        tmp.shexp_out = bs->shexp_out;
        tmp.mla_c_q = bs->mla_c_q;
        tmp.mla_c_kv = bs->mla_c_kv;
        tmp.mla_q_full = bs->mla_q_full;
        tmp.mla_kv_compressed = bs->mla_kv_compressed;

        /* Embed and forward. */
        embed_token(&tmp, seqs[s].token);
        for (uint32_t l = 0; l < m->cfg.n_layer; l++) {
            forward_layer(&tmp, l);
        }
        /* Final RMSNorm + lm_head. */
        oc_rms_norm_f32(tmp.x, m->final_norm, tmp.normed, m->cfg.n_embd,
                        m->cfg.rms_norm_eps);
        if (m->cfg.norm_scale != 1.0f) {
            for (size_t i = 0; i < m->cfg.n_embd; i++) tmp.normed[i] *= m->cfg.norm_scale;
        }
        if (seqs[s].logits != NULL) {
            if (m->output.qtype == OC_QUANT_F32) {
                oc_matvec_f32((const float *)m->output.data, m->output.rows,
                              m->output.cols, tmp.normed, seqs[s].logits);
            } else {
                oc_matvec_quantized(m->output.qtype, m->output.data, m->output.rows,
                                     m->output.cols, m->output.row_bytes,
                                     tmp.normed, seqs[s].logits, tmp.dequant_temp);
            }
        }
        seqs[s].pos++;
        bs->pos[s] = seqs[s].pos;
    }
    return OC_OK;
}

void oc_batch_session_free(OcBatchSession *bs)
{
    if (bs == NULL) return;
    free(bs->kv_k); free(bs->kv_v);
    free(bs->x); free(bs->normed);
    free(bs->q); free(bs->k); free(bs->v);
    free(bs->attn_out);
    free(bs->ffn_gate); free(bs->ffn_up);
    free(bs->dequant_temp);
    free(bs->router_logits);
    free(bs->expert_gate); free(bs->expert_up); free(bs->expert_out);
    free(bs->shexp_gate); free(bs->shexp_up); free(bs->shexp_out);
    free(bs->mla_c_q); free(bs->mla_c_kv);
    free(bs->mla_q_full); free(bs->mla_kv_compressed);
    memset(bs, 0, sizeof(*bs));
}
