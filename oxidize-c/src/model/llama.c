#define _POSIX_C_SOURCE 200809L
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
#include "oxidize/attn_kernels.h"
#include "oxidize/gguf.h"
#include "oxidize/log.h"
#include "oxidize/matvec.h"
#include "oxidize/model.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"
#include "oxidize/sampling.h"

#include <math.h>
#include <stdatomic.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static inline size_t oc_max_sz(size_t a, size_t b) { return a > b ? a : b; }

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n, sz);
    return p;
}

static bool size_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
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

static bool is_qwen35_arch(const char *arch)
{
    return arch != NULL &&
        (strcmp(arch, "qwen35moe") == 0 || strcmp(arch, "qwen35") == 0 ||
         strcmp(arch, "qwen3_5") == 0 || strcmp(arch, "qwen3_5_text") == 0 ||
         strcmp(arch, "qwen3_5_moe") == 0 || strcmp(arch, "qwen35_text") == 0 ||
         strcmp(arch, "qwen3_5_moe_text") == 0);
}

static bool is_longcat_arch(const char *arch)
{
    return arch != NULL &&
        (strcmp(arch, "longcat") == 0 || strcmp(arch, "longcat2") == 0 ||
         strcmp(arch, "longcat_2") == 0 || strcmp(arch, "longcat_flash") == 0 ||
         strcmp(arch, "longcatflash") == 0);
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
        (!is_qwen35_arch(arch_str) && cfg->n_embd % cfg->n_head != 0))
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
    snprintf(key, sizeof(key), "%sexpert_shared_feed_forward_length", prefix);
    cfg->shared_expert_intermediate_size = cfg_u32(f, key, 0);
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
    cfg->embd_rms_norm = false;
    cfg->attn_out_gate = false;
    cfg->rope_swa_only = false;
    cfg->post_norm_eps = 0.0f;
    cfg->logit_scale = 0.0f;
    cfg->rope_norm_pairs = false;
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
        } else if (strcmp(arch_str, "muse-glimmer") == 0 ||
                   strcmp(arch_str, "muse_glimmer") == 0) {
            /* Muse Glimmer: dense SwiGLU, sandwich norms, per-head QK-norm,
             * gated attention output, [local,local,local,global] attention
             * with RoPE on the local layers only, and a scaled + softcapped
             * lm_head. See llama.cpp src/models/muse-glimmer.cpp. */
            cfg->embd_rms_norm = true;
            cfg->attn_out_gate = true;
            cfg->rope_swa_only = true;
            /* llama.cpp assigns this arch LLAMA_ROPE_TYPE_NORM. */
            cfg->rope_norm_pairs = true;
            /* llama.cpp hardcodes 1e-8 for the two sandwich norms; it is not
             * in the metadata. */
            cfg->post_norm_eps = 1e-8f;

            snprintf(key, sizeof(key), "%sfinal_logit_softcapping", prefix);
            cfg->logit_softcap = cfg_f32(f, key, 0.0f);
            snprintf(key, sizeof(key), "%slogit_scale", prefix);
            cfg->logit_scale = cfg_f32(f, key, 1.0f);
            snprintf(key, sizeof(key), "%sattention.sliding_window", prefix);
            cfg->sliding_window = cfg_u32(f, key, 0);

            /* Per-layer sliding/global pattern. The metadata carries the
             * period (4), and llama.cpp's set_swa_pattern makes layer il
             * sliding when il % period < period - 1 — i.e. three local layers
             * then one global, repeating. Materialize it per layer so the
             * forward passes never re-derive it. */
            snprintf(key, sizeof(key), "%sattention.sliding_window_pattern",
                     prefix);
            uint32_t period = cfg_u32(f, key, 4);
            if (period == 0) period = 4;
            cfg->sliding_window_pattern = period;
            cfg->layer_is_swa = xcalloc(cfg->n_layer, sizeof(uint8_t));
            if (cfg->layer_is_swa == NULL) return OC_ERR_OOM;
            for (uint32_t l = 0; l < cfg->n_layer; l++)
                cfg->layer_is_swa[l] = (uint8_t)((l % period) < (period - 1));

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
        } else if (strncmp(arch_str, "gpt2", 4) == 0) {
            /* GPT-2: GELU FFN, learned positional embeddings, no RoPE. */
            cfg->uses_gpt2 = true;
            cfg->uses_par_attn = true;
            cfg->norm_scale = 1.0f;
            no_rope = true;
        } else if (strncmp(arch_str, "gptj", 4) == 0) {
            /* GPT-J: GELU FFN, interleaved RoPE, no learned WPE. */
            cfg->uses_par_attn = true;
            cfg->norm_scale = 1.0f;
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
    cfg->yarn_mscale = 0.0f;
    cfg->yarn_mscale_all_dim = 0.0f;
    cfg->ngram_n_grams = 0;
    cfg->ngram_split_num = 0;
    cfg->is_qwen35 = is_qwen35_arch(arch_str);
    cfg->nextn_predict_layers = 0;
    cfg->full_attention_interval = 0;
    cfg->n_full_attention_layers = 0;
    cfg->n_recurrent_layers = 0;
    cfg->ssm_conv_kernel = 0;
    cfg->ssm_state_size = 0;
    cfg->ssm_group_count = 0;
    cfg->ssm_value_heads = 0;
    cfg->ssm_inner_size = 0;
    if (cfg->is_qwen35) {
        snprintf(key, sizeof(key), "%snextn_predict_layers", prefix);
        cfg->nextn_predict_layers = cfg_u32(f, key, 0);
        if (cfg->nextn_predict_layers >= cfg->n_layer) return OC_ERR_MODEL;
        cfg->n_layer -= cfg->nextn_predict_layers;
        snprintf(key, sizeof(key), "%sfull_attention_interval", prefix);
        cfg->full_attention_interval = cfg_u32(f, key, 0);
        snprintf(key, sizeof(key), "%sssm.conv_kernel", prefix);
        cfg->ssm_conv_kernel = cfg_u32(f, key, 0);
        snprintf(key, sizeof(key), "%sssm.state_size", prefix);
        cfg->ssm_state_size = cfg_u32(f, key, 0);
        snprintf(key, sizeof(key), "%sssm.group_count", prefix);
        cfg->ssm_group_count = cfg_u32(f, key, 0);
        snprintf(key, sizeof(key), "%sssm.time_step_rank", prefix);
        cfg->ssm_value_heads = cfg_u32(f, key, 0);
        snprintf(key, sizeof(key), "%sssm.inner_size", prefix);
        cfg->ssm_inner_size = cfg_u32(f, key, 0);
        const bool has_recurrent_layers = cfg->full_attention_interval > 1 ||
            cfg->ssm_conv_kernel > 0 || cfg->ssm_state_size > 0 ||
            cfg->ssm_group_count > 0 || cfg->ssm_value_heads > 0 ||
            cfg->ssm_inner_size > 0;
        if (cfg->full_attention_interval == 0)
            cfg->full_attention_interval = 1;
        if (has_recurrent_layers &&
            (cfg->ssm_conv_kernel < 2 || cfg->ssm_state_size == 0 ||
             cfg->ssm_group_count == 0 || cfg->ssm_value_heads == 0 ||
             cfg->ssm_inner_size == 0 ||
             cfg->ssm_inner_size % cfg->ssm_value_heads != 0 ||
             cfg->ssm_value_heads % cfg->ssm_group_count != 0)) {
            return OC_ERR_MODEL;
        }
    }
    if (is_longcat_arch(arch_str)) {
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

        /* deepseek_yarn mscale constants are not written to GGUF metadata;
         * these are LongCat's config.json values. */
        cfg->yarn_mscale         = 1.0f;
        cfg->yarn_mscale_all_dim = 1.0f;

        snprintf(key, sizeof(key), "%sngram.neighbor_num", prefix);
        uint32_t nn = cfg_u32(f, key, 0);
        snprintf(key, sizeof(key), "%sngram.split_num", prefix);
        uint32_t sn = cfg_u32(f, key, 0);
        cfg->ngram_split_num = sn;
        cfg->ngram_n_grams   = (nn > 1 && sn > 0) ? (nn - 1) * sn : 0;
        if (cfg->ngram_n_grams > OC_LONGCAT_MAX_NGRAM) return OC_ERR_MODEL;
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
    /* Muse Glimmer has a single attention geometry — sliding and global
     * layers differ only in window and RoPE/NoPE — so the _swa fields mirror
     * the base ones. Done here, after head_dim/rope_dim are resolved. */
    if (cfg->layer_is_swa != NULL && !cfg->uses_gemma4) {
        cfg->head_dim_swa   = cfg->head_dim;
        cfg->n_head_kv_swa  = cfg->n_head_kv;
        cfg->rope_dim_swa   = cfg->rope_dim;
        cfg->rope_theta_swa = cfg->rope_theta;
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
        const bool mtp_block = (m->cfg.nextn_predict_layers > 0 &&
                                layer_idx == m->cfg.n_layer);
        if (layer_idx > m->cfg.n_layer ||
            (layer_idx == m->cfg.n_layer && !mtp_block))
            return false;
        OcLlamaLayer *L = mtp_block ? &m->mtp.layer : &m->layers[layer_idx];
        if (mtp_block && strncmp(suf, "nextn.", 6) == 0) {
            suf += 6;
            if (strcmp(suf, "eh_proj.weight") == 0) {
                m->mtp.eh_proj = view_from_info(mm, info);
            } else if (strcmp(suf, "enorm.weight") == 0) {
                m->mtp.enorm = load_norm(mm, info, m->cfg.n_embd);
            } else if (strcmp(suf, "hnorm.weight") == 0) {
                m->mtp.hnorm = load_norm(mm, info, m->cfg.n_embd);
            } else if (strcmp(suf, "shared_head_norm.weight") == 0) {
                m->mtp.shared_head_norm = load_norm(mm, info, m->cfg.n_embd);
            } else if (strcmp(suf, "embed_tokens.weight") == 0) {
                m->mtp.embed_tokens = view_from_info(mm, info);
            } else if (strcmp(suf, "shared_head.weight") == 0) {
                m->mtp.shared_head_head = view_from_info(mm, info);
            }
            return true;
        }
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
        } else if (strcmp(suf, "attn_gate.weight") == 0) {
            L->attn_gate = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_qkv.weight") == 0) {
            L->attn_qkv = view_from_info(mm, info);
        } else if (strcmp(suf, "ssm_a") == 0) {
            L->ssm_a = view_from_info(mm, info);
        } else if (strcmp(suf, "ssm_alpha.weight") == 0) {
            L->ssm_alpha = view_from_info(mm, info);
        } else if (strcmp(suf, "ssm_beta.weight") == 0) {
            L->ssm_beta = view_from_info(mm, info);
        } else if (strcmp(suf, "ssm_conv1d.weight") == 0) {
            L->ssm_conv1d = view_from_info(mm, info);
            /* oxidize-convert flattens HF [channels, 1, kernel] conv weights
             * to one GGUF dimension. Restore the logical row geometry here. */
            if (info->n_dims == 1 && m->cfg.ssm_conv_kernel != 0 &&
                info->dims[0] % m->cfg.ssm_conv_kernel == 0) {
                L->ssm_conv1d.cols = m->cfg.ssm_conv_kernel;
                L->ssm_conv1d.rows =
                    (size_t)info->dims[0] / m->cfg.ssm_conv_kernel;
                L->ssm_conv1d.row_bytes = oc_quantized_size(
                    L->ssm_conv1d.qtype, L->ssm_conv1d.cols);
            }
        } else if (strcmp(suf, "ssm_dt.bias") == 0) {
            L->ssm_dt_bias = view_from_info(mm, info);
        } else if (strcmp(suf, "ssm_norm.weight") == 0) {
            L->ssm_norm = view_from_info(mm, info);
        } else if (strcmp(suf, "ssm_out.weight") == 0) {
            L->ssm_out = view_from_info(mm, info);
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
    m->mtp.layer.layer_output_scale = 1.0f;

    for (size_t i = 0; i < n; i++) {
        const char *cname = infos[i].name;   /* already canonical-mapped */
        if (m->cfg.is_qwen35 && strstr(cname, "_exps.weight") != NULL &&
            (infos[i].n_dims != 3 ||
             infos[i].dims[2] != m->cfg.num_experts)) {
            oc_log(OC_LOG_ERROR, "llama: qwen35 expert tensor shape mismatch: %s",
                   cname);
            oc_arena_free(arena);
            return OC_ERR_TENSOR;
        }
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
        const bool swa = m->cfg.layer_is_swa != NULL && m->cfg.layer_is_swa[l];
        L->head_dim   = swa ? m->cfg.head_dim_swa   : m->cfg.head_dim;
        L->n_head_kv  = swa ? m->cfg.n_head_kv_swa  : m->cfg.n_head_kv;
        L->rope_dim   = swa ? m->cfg.rope_dim_swa   : m->cfg.rope_dim;
        L->rope_theta = swa ? m->cfg.rope_theta_swa : m->cfg.rope_theta;
        L->sliding_window = swa ? m->cfg.sliding_window : 0u;
        /* Muse Glimmer runs RoPE on sliding layers and NoPE on global ones;
         * every other architecture ropes every layer. */
        L->use_rope = !m->cfg.rope_swa_only || swa;

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
        if (L->attn_q.data != NULL && !m->cfg.is_qwen35) {
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
    if (m->cfg.is_qwen35) {
        const bool has_recurrent_layers = m->cfg.full_attention_interval > 1;
        if (m->cfg.full_attention_interval == 0 ||
            (has_recurrent_layers &&
             (m->cfg.ssm_group_count == 0 || m->cfg.ssm_state_size == 0 ||
              m->cfg.ssm_value_heads == 0 || m->cfg.ssm_inner_size == 0 ||
              m->cfg.ssm_conv_kernel == 0 ||
              m->cfg.ssm_inner_size % m->cfg.ssm_value_heads != 0 ||
              m->cfg.ssm_group_count > SIZE_MAX / m->cfg.ssm_state_size))) {
            oc_log(OC_LOG_ERROR, "llama: invalid qwen35 recurrent geometry");
            return OC_ERR_MODEL;
        }
        uint32_t state_index = 0;
        uint32_t kv_index = 0;
        const size_t key_dim = (size_t)m->cfg.ssm_group_count *
                               m->cfg.ssm_state_size;
        if (key_dim > (SIZE_MAX - m->cfg.ssm_inner_size) / 2u) {
            oc_log(OC_LOG_ERROR, "llama: qwen35 recurrent geometry overflow");
            return OC_ERR_MODEL;
        }
        const size_t conv_dim = 2 * key_dim + m->cfg.ssm_inner_size;
        const size_t value_dim = has_recurrent_layers
            ? m->cfg.ssm_inner_size / m->cfg.ssm_value_heads
            : 0;
        for (uint32_t l = 0; l < m->cfg.n_layer; l++) {
            OcLlamaLayer *L = &m->layers[l];
            const bool has_full = L->attn_q.data || L->attn_k.data ||
                                  L->attn_v.data || L->attn_output.data;
            const bool has_recurrent = L->attn_qkv.data || L->attn_gate.data ||
                                       L->ssm_alpha.data || L->ssm_conv1d.data;
            if (has_full == has_recurrent) {
                const bool scheduled_full =
                    (l + 1u) % m->cfg.full_attention_interval == 0;
                oc_log(OC_LOG_ERROR, "llama: missing qwen35 tensor blk.%u.%s",
                       l, scheduled_full ? "attn_q.weight" :
                                           "ssm_alpha.weight");
                return OC_ERR_TENSOR;
            }
            if (has_recurrent) {
                L->kind = OC_LLAMA_LAYER_QWEN35_RECURRENT;
                L->state_index = state_index++;
                if (!L->ssm_alpha.data) {
                    oc_log(OC_LOG_ERROR,
                           "llama: missing qwen35 tensor blk.%u.ssm_alpha.weight",
                           l);
                    return OC_ERR_TENSOR;
                }
                if (!L->attn_gate.data || L->attn_gate.cols != m->cfg.n_embd ||
                    L->attn_gate.rows != m->cfg.ssm_inner_size ||
                    !L->attn_qkv.data || L->attn_qkv.cols != m->cfg.n_embd ||
                    L->attn_qkv.rows != conv_dim || !L->ssm_a.data ||
                    L->ssm_a.cols != m->cfg.ssm_value_heads ||
                    L->ssm_alpha.cols != m->cfg.n_embd ||
                    L->ssm_alpha.rows != m->cfg.ssm_value_heads ||
                    !L->ssm_beta.data || L->ssm_beta.cols != m->cfg.n_embd ||
                    L->ssm_beta.rows != m->cfg.ssm_value_heads ||
                    !L->ssm_conv1d.data || L->ssm_conv1d.qtype != OC_QUANT_F32 ||
                    L->ssm_conv1d.cols != m->cfg.ssm_conv_kernel ||
                    L->ssm_conv1d.rows != conv_dim || !L->ssm_dt_bias.data ||
                    L->ssm_dt_bias.qtype != OC_QUANT_F32 ||
                    L->ssm_dt_bias.cols != m->cfg.ssm_value_heads ||
                    !L->ssm_norm.data || L->ssm_norm.qtype != OC_QUANT_F32 ||
                    L->ssm_norm.cols != value_dim ||
                    L->ssm_a.qtype != OC_QUANT_F32 ||
                    !L->ssm_out.data ||
                    L->ssm_out.cols != m->cfg.ssm_inner_size ||
                    L->ssm_out.rows != m->cfg.n_embd) {
                    oc_log(OC_LOG_ERROR,
                           "llama: qwen35 recurrent tensor shape mismatch in blk.%u",
                           l);
                    return OC_ERR_TENSOR;
                }
            } else {
                L->kind = OC_LLAMA_LAYER_FULL_ATTENTION;
                L->kv_cache_index = kv_index++;
                const size_t q_rows = 2u * (size_t)m->cfg.n_head *
                                      m->cfg.head_dim;
                const size_t kv_rows = (size_t)m->cfg.n_head_kv *
                                       m->cfg.head_dim;
                if (!L->attn_q.data || L->attn_q.cols != m->cfg.n_embd ||
                    L->attn_q.rows != q_rows ||
                    !L->attn_k.data || L->attn_k.cols != m->cfg.n_embd ||
                    L->attn_k.rows != kv_rows ||
                    !L->attn_v.data || L->attn_v.cols != m->cfg.n_embd ||
                    L->attn_v.rows != kv_rows ||
                    !L->attn_output.data ||
                    L->attn_output.cols !=
                        (size_t)m->cfg.n_head * m->cfg.head_dim ||
                    L->attn_output.rows != m->cfg.n_embd ||
                    !L->attn_q_norm || !L->attn_k_norm) {
                    oc_log(OC_LOG_ERROR,
                           "llama: qwen35 full-attention tensor shape mismatch in blk.%u",
                           l);
                    return OC_ERR_TENSOR;
                }
            }
            const bool invalid_moe = m->cfg.num_experts > 0 &&
                (!L->ffn_gate_inp.data || !L->ffn_gate_exps.data ||
                 !L->ffn_up_exps.data || !L->ffn_down_exps.data ||
                 !L->ffn_gate_shexp.data || !L->ffn_up_shexp.data ||
                 !L->ffn_down_shexp.data || !L->ffn_gate_inp_shexp.data ||
                 L->ffn_gate_inp.cols != m->cfg.n_embd ||
                 L->ffn_gate_inp.rows != m->cfg.num_experts ||
                 L->ffn_gate_exps.cols != m->cfg.n_embd ||
                 L->ffn_gate_exps.rows != m->cfg.expert_intermediate_size ||
                 L->ffn_up_exps.cols != m->cfg.n_embd ||
                 L->ffn_up_exps.rows != m->cfg.expert_intermediate_size ||
                 L->ffn_down_exps.cols != m->cfg.expert_intermediate_size ||
                 L->ffn_down_exps.rows != m->cfg.n_embd ||
                 L->ffn_gate_shexp.cols != m->cfg.n_embd ||
                 L->ffn_gate_shexp.rows !=
                     m->cfg.shared_expert_intermediate_size ||
                 L->ffn_up_shexp.cols != m->cfg.n_embd ||
                 L->ffn_up_shexp.rows !=
                     m->cfg.shared_expert_intermediate_size ||
                 L->ffn_down_shexp.cols !=
                     m->cfg.shared_expert_intermediate_size ||
                 L->ffn_down_shexp.rows != m->cfg.n_embd ||
                 L->ffn_gate_inp_shexp.cols != m->cfg.n_embd);
            const bool invalid_dense = m->cfg.num_experts == 0 &&
                (!L->ffn_gate.data || !L->ffn_up.data || !L->ffn_down.data ||
                 L->ffn_gate.cols != m->cfg.n_embd ||
                 L->ffn_gate.rows != m->cfg.n_ff ||
                 L->ffn_up.cols != m->cfg.n_embd ||
                 L->ffn_up.rows != m->cfg.n_ff ||
                 L->ffn_down.cols != m->cfg.n_ff ||
                 L->ffn_down.rows != m->cfg.n_embd);
            if (!L->attn_norm || !L->post_attention_norm || invalid_moe ||
                invalid_dense) {
                oc_log(OC_LOG_ERROR, "llama: incomplete qwen35 block %u", l);
                return OC_ERR_TENSOR;
            }
        }
        m->cfg.n_recurrent_layers = state_index;
        m->cfg.n_full_attention_layers = kv_index;
        OcLlamaLayer *M = &m->mtp.layer;
        const bool mtp_full = M->attn_q.data && M->attn_k.data &&
                              M->attn_v.data && M->attn_output.data &&
                              M->attn_norm && m->mtp.eh_proj.data &&
                              m->mtp.enorm && m->mtp.hnorm;
        if (mtp_full) {
            const size_t q_rows = 2u * (size_t)m->cfg.n_head * m->cfg.head_dim;
            const size_t kv_rows = (size_t)m->cfg.n_head_kv * m->cfg.head_dim;
            const bool dense_ok = m->cfg.num_experts > 0 ||
                (M->ffn_gate.data && M->ffn_up.data && M->ffn_down.data);
            if (M->attn_q.rows == q_rows && M->attn_k.rows == kv_rows &&
                M->attn_v.rows == kv_rows && dense_ok &&
                m->mtp.eh_proj.cols == 2u * m->cfg.n_embd &&
                m->mtp.eh_proj.rows == m->cfg.n_embd) {
                M->kind = OC_LLAMA_LAYER_FULL_ATTENTION;
                M->head_dim = m->cfg.head_dim;
                M->n_head_kv = m->cfg.n_head_kv;
                M->rope_dim = m->cfg.rope_dim;
                M->rope_theta = m->cfg.rope_theta;
                M->use_rope = true;
                M->kv_cache_index = kv_index;
                m->mtp.present = true;
                oc_log(OC_LOG_INFO, "llama: MTP/nextn block loaded (draft k=%u)",
                       OC_MTP_DEFAULT_DRAFT);
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
        strcmp(arch_str, "gptj") != 0 &&
        strcmp(arch_str, "gptneox") != 0 && strcmp(arch_str, "falcon") != 0 &&
        strcmp(arch_str, "gemma4") != 0 && !is_longcat_arch(arch_str) &&
        strcmp(arch_str, "qwen3moe") != 0 &&
        strcmp(arch_str, "muse-glimmer") != 0 &&
        strcmp(arch_str, "muse_glimmer") != 0 &&
        !is_qwen35_arch(arch_str)) {
        oc_gguf_map_free(&out->gguf);
        oc_log(OC_LOG_ERROR, "llama: unsupported architecture \"%s\"", arch_str);
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
           "n_head_kv=%u head_dim=%u n_ff=%u vocab=%u ctx=%u%s%s",
           arch_str, out->cfg.n_layer, out->cfg.n_embd, out->cfg.n_head,
           out->cfg.n_head_kv, out->cfg.head_dim, out->cfg.n_ff,
           out->cfg.vocab_size, out->cfg.n_ctx,
           out->cfg.tied_embeddings ? " (tied)" : "",
           out->mtp.present ? " (MTP)" : "");
    if (out->cfg.is_longcat && out->cfg.ngram_n_grams > 0)
        oc_log(OC_LOG_WARN,
               "llama: LongCat n-gram tables are bound but not applied in forward");
    if (out->cfg.uses_mla || out->cfg.is_longcat)
        oc_log(OC_LOG_INFO,
               "llama: batched prefill unavailable for this architecture; using per-token prefill");
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

OcKvCacheType oc_llama_select_kv_type(uint32_t n_ctx, const char *explicit)
{
    if (explicit != NULL) {
        if (strcmp(explicit, "q8") == 0 || strcmp(explicit, "Q8") == 0)
            return OC_KV_Q8;
        if (strcmp(explicit, "f32") == 0 || strcmp(explicit, "F32") == 0)
            return OC_KV_F32;
    }
    if (getenv("OX_KV_TYPE") != NULL) return kv_type_from_env();
    return n_ctx >= 8192u ? OC_KV_Q8 : OC_KV_F32;
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
    /* MLA caches the COMPRESSED latent [c_kv | k_pe], not the expanded
     * per-head K/V. Every head's K and V are linear in these same numbers,
     * so forward_mla_attention folds k_b into the query and applies v_b to
     * the attention-weighted latent. On LongCat-2.0 that is 576 floats per
     * row instead of 64*192 = 12288 -- 21x less cache, and less arithmetic
     * per step besides. */
    if (model->cfg.uses_mla)
        return (size_t)model->cfg.mla_kv_lora_dim + model->cfg.mla_q_rope_dim;

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

static size_t kv_cache_layer_count(const OcLlamaModel *model)
{
    size_t cache_layers = model->cfg.is_qwen35
                            ? model->cfg.n_full_attention_layers
                            : model->cfg.n_layer;
    if (model->mtp.present) cache_layers += 1;
    return cache_layers;
}

size_t oc_llama_kv_cache_bytes(const OcLlamaModel *model, OcKvCacheType kv_type)
{
    if (model == NULL) return 0;
    size_t row = kv_row_floats_for(model);
    size_t cache_layers = kv_cache_layer_count(model);
    size_t elems;
    if (!size_mul(cache_layers, model->cfg.n_ctx, &elems) ||
        !size_mul(elems, row, &elems))
        return SIZE_MAX;
    if (kv_type == OC_KV_Q8) {
        size_t groups, group_bytes, bytes;
        if (!size_mul(cache_layers, model->cfg.n_ctx, &groups) ||
            !size_mul(groups, model->cfg.n_head_kv, &groups) ||
            !size_mul(groups, sizeof(float), &group_bytes) ||
            elems > SIZE_MAX - group_bytes ||
            !size_mul(elems + group_bytes, 2u, &bytes))
            return SIZE_MAX;
        return bytes;
    }
    size_t buffers = model->cfg.uses_mla ? 1u : 2u;
    size_t bytes;
    return size_mul(elems, buffers * sizeof(float), &bytes) ? bytes : SIZE_MAX;
}

OcError oc_llama_session_init(OcLlamaModel *model, OcLlamaSession *out)
{
    return oc_llama_session_init_kv(model, out, kv_type_from_env());
}

static OcError session_init_kv_impl(OcLlamaModel *model, OcLlamaSession *out,
                                    OcKvCacheType kv_type, int skip_dense_kv)
{
    if (model == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    /* uses_geglu is fully handled by forward_dense_ffn (GeGLU vs SwiGLU),
     * so it is NOT rejected here. MLA is handled by forward_mla_attention
     * with its compressed [c_kv | k_pe] cache. */
    if (model->cfg.num_experts > 0 && model->cfg.expert_intermediate_size == 0) {
        if (model->cfg.n_ff == 0) return OC_ERR_MODEL;
        model->cfg.expert_intermediate_size = model->cfg.n_ff;
    }
    memset(out, 0, sizeof(*out));
    out->model = model;
    /* For MLA, each head has its own K/V (no GQA sharing); for Gemma 4 this is
     * the max over the two layer geometries. See kv_row_floats_for(). */
    out->kv_row_floats = kv_row_floats_for(model);
    size_t per_layer;
    size_t cache_layers = kv_cache_layer_count(model);
    size_t total;
    if (!size_mul(model->cfg.n_ctx, out->kv_row_floats, &per_layer) ||
        !size_mul(cache_layers, per_layer, &total))
        return OC_ERR_MODEL;

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
    /* MLA does not store per-head K/V at all -- it caches the [c_kv | k_pe]
     * latent, which has no kv-head structure for Q8's per-head scales to key
     * off, and forward_mla_attention writes s->kv_k directly rather than
     * going through kv_q8_encode. Q8 here would dereference the NULL kv_k
     * that the Q8 path leaves behind. The latent is already 21x smaller than
     * the expanded cache, so there is little left for Q8 to win. */
    if (kv_type == OC_KV_Q8 && model->cfg.uses_mla) {
        oc_log(OC_LOG_WARN, "llama: Q8 KV does not apply to MLA's compressed "
               "latent cache; using f32 KV");
        kv_type = OC_KV_F32;
    }
    if (kv_type == OC_KV_Q8 && model->cfg.head_dim != model->cfg.kv_head_dim) {
        oc_log(OC_LOG_WARN, "llama: Q8 KV needs head_dim == kv_head_dim "
               "(%u vs %u); using f32 KV",
               model->cfg.head_dim, model->cfg.kv_head_dim);
        kv_type = OC_KV_F32;
    }
    out->kv_type = kv_type;

    if (!skip_dense_kv && kv_type == OC_KV_Q8) {
        size_t groups;
        if (!size_mul(cache_layers, model->cfg.n_ctx, &groups) ||
            !size_mul(groups, model->cfg.n_head_kv, &groups))
            return OC_ERR_MODEL;
        out->kv_k_q = xcalloc(total, sizeof(int8_t));
        out->kv_v_q = xcalloc(total, sizeof(int8_t));
        out->kv_k_scale = xcalloc(groups, sizeof(float));
        out->kv_v_scale = xcalloc(groups, sizeof(float));
        if (!out->kv_k_q || !out->kv_v_q || !out->kv_k_scale ||
            !out->kv_v_scale) {
            oc_llama_session_free(out);
            return OC_ERR_OOM;
        }
    } else if (!skip_dense_kv) {
        out->kv_k = xcalloc(total, sizeof(float));
        /* MLA has no separate V cache: forward_mla_attention stores the
         * [c_kv | k_pe] latent in kv_k and reconstructs V from it via v_b at
         * the end of each head. Allocating kv_v would silently double the
         * cache for a buffer nothing reads -- 171 KB per token on
         * LongCat-2.0, i.e. 22 GB at a 128k context. */
        if (!model->cfg.uses_mla)
            out->kv_v = xcalloc(total, sizeof(float));
    }
    size_t maxw = model->cfg.n_embd > model->cfg.n_ff ? model->cfg.n_embd
                                                     : model->cfg.n_ff;
    if (model->cfg.expert_intermediate_size > maxw) {
        maxw = model->cfg.expert_intermediate_size;
    }
    if (model->cfg.shared_expert_intermediate_size > maxw)
        maxw = model->cfg.shared_expert_intermediate_size;
    if (model->cfg.n_embd * 2u > maxw)
        maxw = model->cfg.n_embd * 2u;
    out->x = xcalloc(model->cfg.n_embd, sizeof(float));
    out->normed = xcalloc(model->cfg.n_embd, sizeof(float));
    out->q = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
    out->k = xcalloc(out->kv_row_floats, sizeof(float));
    out->v = xcalloc(out->kv_row_floats, sizeof(float));
    /* n_head*head_dim is what attention writes, but the Gemma-style sandwich
     * norms reuse this buffer as an n_embd-wide scratch for the branch
     * output. On a heavily-GQA model (Muse Glimmer: 32*128 = 4096 vs
     * n_embd 6656) that is the larger of the two, so size it for both. */
    out->attn_out = xcalloc(oc_max_sz((size_t)model->cfg.n_head *
                                      model->cfg.head_dim,
                                      (size_t)model->cfg.n_embd),
                            sizeof(float));
    out->ffn_gate = xcalloc(model->cfg.n_ff, sizeof(float));
    out->ffn_up = xcalloc(model->cfg.n_ff, sizeof(float));
    out->dequant_temp = xcalloc(maxw, sizeof(float));
    out->logits = xcalloc(model->cfg.vocab_size, sizeof(float));
    out->last_hidden = xcalloc(model->cfg.n_embd, sizeof(float));
    out->mtp_hidden = xcalloc(model->cfg.n_embd, sizeof(float));
    out->mtp_concat = xcalloc((size_t)model->cfg.n_embd * 2u, sizeof(float));
    /* MoE temporaries (only allocated when num_experts > 0). */
    if (model->cfg.num_experts > 0) {
        /* The router spans routed AND zero-expert slots. Sizing this to
         * num_experts alone overruns it by zero_expert_count floats on every
         * MoE layer of every token -- 512 bytes per layer on LongCat-2.0. */
        out->router_logits = xcalloc((size_t)model->cfg.num_experts
                                     + model->cfg.zero_expert_count,
                                     sizeof(float));
        out->expert_gate = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_up   = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_out  = xcalloc(model->cfg.n_embd, sizeof(float));
        out->expert_gate_all = xcalloc(
            (size_t)model->cfg.num_experts_per_tok *
            model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_up_all = xcalloc(
            (size_t)model->cfg.num_experts_per_tok *
            model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_down_all = xcalloc(
            (size_t)model->cfg.num_experts_per_tok * model->cfg.n_embd,
            sizeof(float));
        out->selected_experts = xcalloc(
            (size_t)model->cfg.num_experts + model->cfg.zero_expert_count,
            sizeof(uint32_t));
        const size_t shexp_size = model->cfg.shared_expert_intermediate_size
                                    ? model->cfg.shared_expert_intermediate_size
                                    : model->cfg.expert_intermediate_size;
        out->shexp_gate  = xcalloc(shexp_size, sizeof(float));
        out->shexp_up    = xcalloc(shexp_size, sizeof(float));
        out->shexp_out   = xcalloc(model->cfg.n_embd, sizeof(float));
    }
    if (skip_dense_kv) {
        oc_log(OC_LOG_INFO, "llama: dense KV skipped (compressed cache)");
    } else {
        oc_log(OC_LOG_INFO, "llama: KV cache %s, %.1f MB (%.1f MB as f32)",
               out->kv_type == OC_KV_Q8 ? "int8" : "f32",
               (double)oc_llama_kv_cache_bytes(model, out->kv_type) / 1e6,
               (double)oc_llama_kv_cache_bytes(model, OC_KV_F32) / 1e6);
    }

    /* Q8 leaves kv_k/kv_v NULL; its buffers were checked at allocation. */
    if ((!skip_dense_kv && out->kv_type == OC_KV_F32 &&
         (!out->kv_k || (!out->kv_v && !model->cfg.uses_mla))) ||
        !out->x || !out->normed || !out->q ||
        !out->k || !out->v || !out->attn_out || !out->ffn_gate ||
        !out->ffn_up || !out->dequant_temp || !out->logits ||
        !out->last_hidden || !out->mtp_hidden || !out->mtp_concat) {
        oc_llama_session_free(out);
        return OC_ERR_OOM;
    }
    if (model->cfg.num_experts > 0 &&
        (!out->router_logits || !out->expert_gate || !out->expert_up ||
         !out->expert_out || !out->expert_gate_all || !out->expert_up_all ||
         !out->expert_down_all || !out->selected_experts ||
         !out->shexp_gate || !out->shexp_up || !out->shexp_out)) {
        oc_llama_session_free(out);
        return OC_ERR_OOM;
    }
    if (model->cfg.attn_out_gate) {
        out->muse_gate = xcalloc((size_t)model->cfg.n_head *
                                 model->cfg.head_dim, sizeof(float));
        if (out->muse_gate == NULL) {
            oc_llama_session_free(out);
            return OC_ERR_OOM;
        }
    }

    /* MLA temporaries. */
    if (model->cfg.uses_mla) {
        out->mla_c_q = xcalloc(model->cfg.mla_q_lora_dim, sizeof(float));
        out->mla_c_kv = xcalloc(model->cfg.mla_kv_lora_dim, sizeof(float));
        out->mla_q_full = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
        out->mla_kv_compressed = xcalloc(model->cfg.mla_kv_lora_dim + model->cfg.mla_q_rope_dim, sizeof(float));
        size_t mla_per_head = (size_t)model->cfg.n_head * model->cfg.mla_kv_lora_dim;
        out->mla_q_absorbed = xcalloc(mla_per_head, sizeof(float));
        out->mla_ctx_latent = xcalloc(mla_per_head, sizeof(float));
        out->mla_run_max = xcalloc(model->cfg.n_head, sizeof(float));
        out->mla_run_sum = xcalloc(model->cfg.n_head, sizeof(float));
        if (!out->mla_c_q || !out->mla_c_kv || !out->mla_q_full ||
            !out->mla_kv_compressed || !out->mla_q_absorbed ||
            !out->mla_ctx_latent || !out->mla_run_max || !out->mla_run_sum) {
            oc_llama_session_free(out);
            return OC_ERR_OOM;
        }
    }
    if (model->cfg.is_qwen35) {
        OcQwen35DeltaGeometry geometry = {
            .n_key_heads = model->cfg.ssm_group_count,
            .n_value_heads = model->cfg.ssm_value_heads,
            .key_head_dim = model->cfg.ssm_state_size,
            .value_head_dim = model->cfg.ssm_inner_size /
                              model->cfg.ssm_value_heads,
            .conv_kernel = model->cfg.ssm_conv_kernel,
        };
        const size_t conv_dim = 2 * geometry.n_key_heads *
                                geometry.key_head_dim + model->cfg.ssm_inner_size;
        size_t conv_per_layer, recurrent_per_layer, conv_total, recurrent_total;
        size_t full_qgate;
        if (!size_mul(conv_dim, geometry.conv_kernel - 1u, &conv_per_layer) ||
            !size_mul(model->cfg.ssm_inner_size, model->cfg.ssm_state_size,
                      &recurrent_per_layer) ||
            !size_mul(model->cfg.n_recurrent_layers, conv_per_layer,
                      &conv_total) ||
            !size_mul(model->cfg.n_recurrent_layers, recurrent_per_layer,
                      &recurrent_total) ||
            !size_mul(model->cfg.n_head, model->cfg.head_dim, &full_qgate) ||
            !size_mul(full_qgate, 2u, &full_qgate)) {
            oc_llama_session_free(out);
            return OC_ERR_MODEL;
        }
        const size_t qkv_scratch = conv_dim > full_qgate
                                     ? conv_dim : full_qgate;
        const size_t full_gate = (size_t)model->cfg.n_head *
                                 model->cfg.head_dim;
        const size_t gate_scratch = model->cfg.ssm_inner_size > full_gate
                                      ? model->cfg.ssm_inner_size : full_gate;
        out->qwen35_delta = xcalloc(model->cfg.n_layer,
                                    sizeof(*out->qwen35_delta));
        out->qwen35_conv_state = xcalloc(
            conv_total,
            sizeof(*out->qwen35_conv_state));
        out->qwen35_recurrent_state = xcalloc(
            recurrent_total,
            sizeof(*out->qwen35_recurrent_state));
        out->qwen35_qkv = xcalloc(qkv_scratch, sizeof(*out->qwen35_qkv));
        out->qwen35_gate = xcalloc(gate_scratch, sizeof(*out->qwen35_gate));
        out->qwen35_beta = xcalloc(geometry.n_value_heads,
                                   sizeof(*out->qwen35_beta));
        out->qwen35_alpha = xcalloc(geometry.n_value_heads,
                                    sizeof(*out->qwen35_alpha));
        out->qwen35_conv_output = xcalloc(conv_dim,
                                          sizeof(*out->qwen35_conv_output));
        out->qwen35_delta_output = xcalloc(model->cfg.ssm_inner_size,
                                           sizeof(*out->qwen35_delta_output));
        if (!out->qwen35_delta || !out->qwen35_conv_state ||
            !out->qwen35_recurrent_state || !out->qwen35_qkv ||
            !out->qwen35_gate || !out->qwen35_beta || !out->qwen35_alpha ||
            !out->qwen35_conv_output || !out->qwen35_delta_output) {
            oc_llama_session_free(out);
            return OC_ERR_OOM;
        }
        for (uint32_t l = 0; l < model->cfg.n_layer; l++) {
            if (model->layers[l].kind != OC_LLAMA_LAYER_QWEN35_RECURRENT)
                continue;
            const size_t slot = model->layers[l].state_index;
            OcError state_error = oc_qwen35_delta_state_init(
                &out->qwen35_delta[l], &geometry,
                out->qwen35_conv_state + slot * conv_per_layer,
                conv_per_layer,
                out->qwen35_recurrent_state + slot * recurrent_per_layer,
                recurrent_per_layer);
            if (state_error != OC_OK) {
                oc_llama_session_free(out);
                return state_error;
            }
        }
    }
    /* KV cache quantization (Q8_0-style: 32-element blocks with f16 scale). */
    out->pos = 0;
    out->kv_compress = NULL;
    return OC_OK;
}

OcError oc_llama_session_init_kv(OcLlamaModel *model, OcLlamaSession *out,
                                 OcKvCacheType kv_type)
{
    return session_init_kv_impl(model, out, kv_type, 0);
}

static int layer_uses_legacy_sliding_window(const OcLlamaModel *m, uint32_t layer)
{
    const OcLlamaConfig *c;
    if (!m) return 0;
    c = &m->cfg;
    if (c->uses_gemma4 || c->layer_is_swa != NULL) return 0;
    if (c->sliding_window == 0 || c->sliding_window_pattern <= 1) return 0;
    return (layer % c->sliding_window_pattern) == 1;
}

static int arch_refuses_compressed_kv(const OcLlamaModel *m)
{
    if (!m) return 1;
    if (m->cfg.uses_mla || m->cfg.is_qwen35 || m->cfg.uses_gemma4) return 1;
    if (m->arch == OC_ARCH_GPT2 || m->arch == OC_ARCH_GPTJ ||
        m->arch == OC_ARCH_GPTNEOX || m->arch == OC_ARCH_FALCON)
        return 1;
    return 0;
}

static int layer_qualifies_compressed(const OcLlamaModel *m, uint32_t layer)
{
    const OcLlamaLayer *L;
    if (!m || !m->layers || layer >= m->cfg.n_layer) return 0;
    if (arch_refuses_compressed_kv(m)) return 0;
    L = &m->layers[layer];
    if (!L->use_rope || m->cfg.yarn_factor > 0.0f)
        return 0;
    if (L->sliding_window > 0) return 0;
    if (layer_uses_legacy_sliding_window(m, layer)) return 0;
    return 1;
}

static int model_all_layers_compressed(const OcLlamaModel *m)
{
    uint32_t l;
    if (!m || !m->layers || m->cfg.n_layer == 0) return 0;
    if (arch_refuses_compressed_kv(m)) return 0;
    if (m->cfg.yarn_factor > 0.0f) return 0;
    for (l = 0; l < m->cfg.n_layer; l++) {
        if (!layer_qualifies_compressed(m, l)) return 0;
    }
    return 1;
}

static void release_dense_kv(OcLlamaSession *sess)
{
    if (!sess) return;
    free(sess->kv_k); sess->kv_k = NULL;
    free(sess->kv_v); sess->kv_v = NULL;
    free(sess->kv_k_q); sess->kv_k_q = NULL;
    free(sess->kv_v_q); sess->kv_v_q = NULL;
    free(sess->kv_k_scale); sess->kv_k_scale = NULL;
    free(sess->kv_v_scale); sess->kv_v_scale = NULL;
}

OcError oc_llama_session_enable_kv_compress(OcLlamaSession *sess,
                                            OcKvScheme scheme)
{
    OcCompressedKvCache *cache;
    OcError e;
    size_t page_size, head_dim, rope_dim;
    float theta;
    if (!sess || !sess->model) return OC_ERR_INVALID_ARG;
    if (sess->pos > 0) return OC_ERR_INVALID_ARG;
    if (scheme != OC_KV_SCHEME_ROTOR && scheme != OC_KV_SCHEME_HELIX)
        return OC_ERR_INVALID_ARG;
    head_dim = sess->model->cfg.head_dim;
    if (head_dim == 0) return OC_ERR_INVALID_ARG;
    if (scheme == OC_KV_SCHEME_HELIX && (head_dim % 8) != 0)
        return OC_ERR_INVALID_ARG;
    if (arch_refuses_compressed_kv(sess->model))
        return OC_ERR_INVALID_ARG;
    page_size = OC_COMPRESSED_KV_PAGE_SIZE;
    theta = sess->model->cfg.rope_theta > 0.0f ? sess->model->cfg.rope_theta
                                               : 10000.0f;
    cache = (OcCompressedKvCache *)calloc(1, sizeof(*cache));
    if (!cache) return OC_ERR_OOM;
    e = oc_compressed_kv_init(cache, head_dim, scheme, page_size, theta);
    if (e != OC_OK) {
        free(cache);
        return e;
    }
    oc_compressed_kv_set_rope_layout(
        cache, sess->model->cfg.rope_norm_pairs ? OC_KV_ROPE_INTERLEAVED
                                                : OC_KV_ROPE_SPLIT_HALVES);
    rope_dim = sess->model->cfg.rope_dim;
    if (rope_dim == 0) rope_dim = head_dim;
    e = oc_compressed_kv_set_rope_dim(cache, rope_dim);
    if (e != OC_OK) {
        oc_compressed_kv_free(cache);
        free(cache);
        return e;
    }
    if (sess->kv_compress) {
        oc_compressed_kv_free(sess->kv_compress);
        free(sess->kv_compress);
        sess->kv_compress = NULL;
    }
    sess->kv_compress = cache;
    if (model_all_layers_compressed(sess->model))
        release_dense_kv(sess);
    oc_log(OC_LOG_INFO, "llama: compressed KV %s (head_dim=%zu)",
           scheme == OC_KV_SCHEME_HELIX ? "helix" : "rotor", head_dim);
    return OC_OK;
}

OcError oc_llama_session_enable_kv_compress_name(OcLlamaSession *sess,
                                                 const char *name)
{
    if (!name || name[0] == '\0' || strcmp(name, "none") == 0) return OC_OK;
    if (strcmp(name, "rotor") == 0)
        return oc_llama_session_enable_kv_compress(sess, OC_KV_SCHEME_ROTOR);
    if (strcmp(name, "helix") == 0)
        return oc_llama_session_enable_kv_compress(sess, OC_KV_SCHEME_HELIX);
    return OC_ERR_INVALID_ARG;
}

OcError oc_llama_session_init_compressed(OcLlamaModel *model,
                                         OcLlamaSession *out,
                                         OcKvScheme scheme)
{
    OcError e;
    int skip = model && model_all_layers_compressed(model);
    e = session_init_kv_impl(model, out, OC_KV_F32, skip);
    if (e != OC_OK) return e;
    e = oc_llama_session_enable_kv_compress(out, scheme);
    if (e != OC_OK) {
        oc_llama_session_free(out);
        return e;
    }
    return OC_OK;
}

OcError oc_llama_session_init_with_compress(OcLlamaModel *model,
                                            OcLlamaSession *out,
                                            OcKvCacheType kv_type,
                                            const char *name)
{
    if (!name || name[0] == '\0' || strcmp(name, "none") == 0)
        return oc_llama_session_init_kv(model, out, kv_type);
    if (strcmp(name, "rotor") == 0)
        return oc_llama_session_init_compressed(model, out, OC_KV_SCHEME_ROTOR);
    if (strcmp(name, "helix") == 0)
        return oc_llama_session_init_compressed(model, out, OC_KV_SCHEME_HELIX);
    return OC_ERR_INVALID_ARG;
}

void oc_llama_session_reset(OcLlamaSession *sess)
{
    if (sess == NULL) return;
    sess->pos = 0;
    sess->mtp_pos = 0;
    if (sess->kv_compress) oc_compressed_kv_clear(sess->kv_compress);
    if (sess->qwen35_delta && sess->model) {
        for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++)
            oc_qwen35_delta_state_reset(&sess->qwen35_delta[l]);
    }
}

void oc_llama_session_rewind(OcLlamaSession *sess, uint32_t pos)
{
    if (!sess) return;
    sess->pos = pos;
    if (sess->kv_compress)
        oc_compressed_kv_rewind(sess->kv_compress, (size_t)pos);
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
    free(sess->last_hidden);
    free(sess->mtp_hidden);
    free(sess->mtp_concat);
    free(sess->router_logits);
    free(sess->expert_gate); free(sess->expert_up); free(sess->expert_out);
    free(sess->expert_gate_all); free(sess->expert_up_all);
    free(sess->expert_down_all); free(sess->selected_experts);
    free(sess->shexp_gate); free(sess->shexp_up); free(sess->shexp_out);
    free(sess->mla_c_q); free(sess->mla_c_kv);
    free(sess->mla_q_full); free(sess->mla_kv_compressed);
    free(sess->mla_q_absorbed); free(sess->mla_ctx_latent);
    free(sess->mla_run_max); free(sess->mla_run_sum);
    if (sess->qwen35_delta && sess->model) {
        for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++)
            oc_qwen35_delta_state_free(&sess->qwen35_delta[l]);
    }
    free(sess->qwen35_delta);
    free(sess->qwen35_conv_state);
    free(sess->qwen35_recurrent_state);
    free(sess->qwen35_qkv);
    free(sess->qwen35_gate);
    free(sess->qwen35_beta);
    free(sess->qwen35_alpha);
    free(sess->qwen35_conv_output);
    free(sess->qwen35_delta_output);
    free(sess->muse_gate);
    if (sess->kv_compress) {
        oc_compressed_kv_free(sess->kv_compress);
        free(sess->kv_compress);
    }
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
    free(model->mtp.layer.attn_norm);
    free(model->mtp.layer.ffn_norm);
    free(model->mtp.layer.attn_q_norm);
    free(model->mtp.layer.attn_k_norm);
    free(model->mtp.layer.post_attention_norm);
    free(model->mtp.layer.post_ffw_norm);
    free(model->mtp.enorm);
    free(model->mtp.hnorm);
    free(model->mtp.shared_head_norm);
    free(model->final_norm);
    free(model->final_norm_bias);
    free(model->cfg.layer_is_swa);
    oc_gguf_map_free(&model->gguf);
    memset(model, 0, sizeof(*model));
}

/* ─── Forward pass ─────────────────────────────────────────────────────── */

/* Embed one token: dequantize row `token` of tok_embeddings into `dst`. */
static void embed_token_into(OcLlamaSession *s, uint32_t token, float *dst)
{
    OcWeightView *w = &s->model->tok_embeddings;
    if (token >= s->model->cfg.vocab_size) token = s->model->cfg.vocab_size - 1;
    if (w->qtype == OC_QUANT_F32) {
        memcpy(dst, w->data + (size_t)token * w->row_bytes,
               s->model->cfg.n_embd * sizeof(float));
    } else {
        oc_quant_dequant_row(w->qtype,
            w->data + (size_t)token * w->row_bytes, w->row_bytes,
            dst, s->model->cfg.n_embd);
    }
    /* Gemma scales the token embedding by sqrt(n_embd) once, here — it is a
     * property of the embedding, not of each RMSNorm. The pre-existing
     * `norm_scale` handling in forward_layer multiplies every normed
     * activation instead, which is a different (and for Gemma 4, wrong)
     * computation; uses_gemma4 takes this path and leaves that one alone. */
    if (s->model->cfg.uses_gemma4 && s->model->cfg.norm_scale != 1.0f) {
        const float sc = s->model->cfg.norm_scale;
        for (size_t i = 0; i < s->model->cfg.n_embd; i++) dst[i] *= sc;
    }
    /* Muse Glimmer RMS-normalizes the embedding itself — no weight, no
     * scale — before layer 0 sees it. */
    if (s->model->cfg.embd_rms_norm) {
        const size_t n = s->model->cfg.n_embd;
        double ss = 0.0;
        for (size_t i = 0; i < n; i++) ss += (double)dst[i] * dst[i];
        const float inv = 1.0f / sqrtf((float)(ss / (double)n) +
                                       s->model->cfg.rms_norm_eps);
        for (size_t i = 0; i < n; i++) dst[i] *= inv;
    }
}

static void embed_token(OcLlamaSession *s, uint32_t token)
{
    embed_token_into(s, token, s->x);
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

/* out[0..cols) = W^T @ in, i.e. sum over rows of in[r] * W[r][:].
 *
 * This is the "absorption" step of MLA: folding k_b into the query so that
 * scoring can run against the cached latent directly instead of
 * reconstructing per-head K for every past position. Row-major access, so a
 * quantized W is walked exactly as matvec() walks it. `out` must be `cols`
 * long, `in` must be `rows` long. */
static void matvec_transpose_acc(const OcWeightView *w, const float *in,
                                 float *out, float *temp)
{
    for (size_t c = 0; c < w->cols; c++) out[c] = 0.0f;
    for (size_t r = 0; r < w->rows; r++) {
        const float x = in[r];
        if (x == 0.0f) continue;
        const float *row;
        if (w->qtype == OC_QUANT_F32) {
            row = (const float *)w->data + r * w->cols;
        } else {
            oc_quant_dequant_row_scalar(w->qtype, w->data + r * w->row_bytes,
                                        w->row_bytes, temp, w->cols);
            row = temp;
        }
        for (size_t c = 0; c < w->cols; c++) out[c] += x * row[c];
    }
}

/* Online-softmax attention for one Q head against all cached K/V up to `pos`.
 * GQA: Q head h attends to KV head (h / group_size). */
static OcError forward_layer(OcLlamaSession *s, uint32_t layer);
static void forward_dense_ffn(OcLlamaSession *s, const OcLlamaLayer *L);
/* `query_pos` is the position of the query token: attention runs over cached
 * positions [start, query_pos]. Split out from attention_head() so batched
 * prefill can drive many query positions concurrently — the single-token
 * path always passes s->pos. */
static const OcLlamaLayer *layer_for_attn(const OcLlamaSession *s, uint32_t layer)
{
    if (s->model->mtp.present && layer >= s->model->cfg.n_layer)
        return &s->model->mtp.layer;
    return &s->model->layers[layer];
}

static void attention_head_at(const OcLlamaSession *s, uint32_t head,
                              uint32_t layer, int64_t query_pos,
                              const float *q_vec, float *out_vec)
{
    const OcLlamaConfig *c = &s->model->cfg;
    /* Geometry is per-layer: on Gemma 4 a sliding layer has head_dim 256 with
     * 16 KV heads while a global layer has 512 with 4. The loader resolved
     * both into the layer, so read them rather than the model-wide config. */
    const OcLlamaLayer *GL = layer_for_attn(s, layer);
    const uint32_t cache_layer = c->is_qwen35 ? GL->kv_cache_index : layer;
    size_t hd = GL->head_dim ? (size_t)GL->head_dim : (size_t)c->head_dim;
    uint32_t n_kv = GL->n_head_kv ? GL->n_head_kv : c->n_head_kv;
    uint32_t group = c->n_head / n_kv;
    uint32_t kv_head = head / group;
    size_t kv_off = ((size_t)cache_layer * c->n_ctx + 0) * s->kv_row_floats
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
    int64_t seq_len = query_pos + 1;
    int64_t start = 0;
    if (GL->sliding_window > 0) {
        /* Gemma 4 (and any model whose loader filled in a per-layer window):
         * the pattern came from metadata, so trust the resolved value rather
         * than re-deriving it from a modulus. */
        int64_t sw = (int64_t)GL->sliding_window;
        start = (seq_len > sw) ? (seq_len - sw) : 0;
    } else if (!c->uses_gemma4 && c->layer_is_swa == NULL &&
               c->sliding_window > 0 && c->sliding_window_pattern > 1) {
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
    const size_t sc_base = (size_t)cache_layer * c->n_ctx * sc_stride + kv_head;

    for (int64_t t = start; t < seq_len; t++) {
        float dot = 0.0f;
        const int8_t *kq_t = NULL;
        const int8_t *vq_t = NULL;
        float v_scale = 0.0f;
        if (q8) {
            kq_t = s->kv_k_q + kv_off + (size_t)t * s->kv_row_floats;
            vq_t = s->kv_v_q + kv_off + (size_t)t * s->kv_row_floats;
            dot = oc_attn_dot_q8(q_vec, kq_t, hd);
            dot *= s->kv_k_scale[sc_base + (size_t)t * sc_stride];
            v_scale = s->kv_v_scale[sc_base + (size_t)t * sc_stride];
        } else {
            const float *k_t = k_layer + kv_off + (size_t)t * s->kv_row_floats;
            dot = oc_attn_dot_f32(q_vec, k_t, hd);
        }
        float score = dot * scale;
        float new_max = (score > run_max) ? score : run_max;
        float exp_factor = expf(run_max - new_max);
        float exp_score = expf(score - new_max);
        oc_attn_scale_f32(out_vec, exp_factor, hd);
        if (q8) {
            oc_attn_axpy_q8(out_vec, vq_t, exp_score * v_scale, hd);
        } else {
            const float *v_t = v_layer + kv_off + (size_t)t * s->kv_row_floats;
            oc_attn_axpy_f32(out_vec, v_t, exp_score, hd);
        }
        run_sum = run_sum * exp_factor + exp_score;
        run_max = new_max;
    }
    if (run_sum > 0.0f)
        oc_attn_scale_f32(out_vec, 1.0f / run_sum, hd);
}

typedef struct {
    const OcLlamaSession *s;
    uint32_t layer;
    int64_t start;
    int64_t seq_len;
    size_t hd;
    uint32_t group;
    uint32_t n_kv;
    float scale;
    bool q8;
} AttnDecodeJob;

/* Position-outer GQA attention: each KV head's rows are read once and
 * scored against every query head in the group. At long context this
 * cuts KV traffic by the GQA ratio (typically 4x on Qwen3-27B). */
static void attn_decode_kv_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    const AttnDecodeJob *j = (const AttnDecodeJob *)ud;
    const OcLlamaSession *s = j->s;
    const OcLlamaConfig *c = &s->model->cfg;
    const OcLlamaLayer *GL = layer_for_attn(s, j->layer);
    const uint32_t cache_layer = c->is_qwen35 ? GL->kv_cache_index : j->layer;
    const size_t hd = j->hd;
    const size_t sc_stride = c->n_head_kv;
    const size_t kv_row = s->kv_row_floats;

    for (size_t kv_head = begin; kv_head < end; kv_head++) {
        if (kv_head >= j->n_kv) break;
        const uint32_t h0 = (uint32_t)kv_head * j->group;
        const uint32_t h1 = h0 + j->group;
        float *outs[64];
        const float *qs[64];
        float run_max[64];
        float run_sum[64];
        const uint32_t nq = h1 - h0;
        if (nq > 64) {
            for (uint32_t h = h0; h < h1; h++)
                attention_head_at(s, h, j->layer, s->pos,
                                  s->q + h * hd, s->attn_out + h * hd);
            continue;
        }
        for (uint32_t g = 0; g < nq; g++) {
            qs[g] = s->q + (h0 + g) * hd;
            outs[g] = s->attn_out + (h0 + g) * hd;
            run_max[g] = -INFINITY;
            run_sum[g] = 0.0f;
            for (size_t i = 0; i < hd; i++) outs[g][i] = 0.0f;
        }
        const size_t kv_off = ((size_t)cache_layer * c->n_ctx + 0) * kv_row
                            + kv_head * hd;
        const size_t sc_base = (size_t)cache_layer * c->n_ctx * sc_stride
                             + kv_head;
        for (int64_t t = j->start; t < j->seq_len; t++) {
            if (j->q8) {
                const int8_t *kq = s->kv_k_q + kv_off + (size_t)t * kv_row;
                const int8_t *vq = s->kv_v_q + kv_off + (size_t)t * kv_row;
                const float k_scale = s->kv_k_scale[sc_base + (size_t)t * sc_stride];
                const float v_scale = s->kv_v_scale[sc_base + (size_t)t * sc_stride];
                for (uint32_t g = 0; g < nq; g++) {
                    float score = oc_attn_dot_q8(qs[g], kq, hd) * k_scale * j->scale;
                    float new_max = (score > run_max[g]) ? score : run_max[g];
                    float exp_factor = expf(run_max[g] - new_max);
                    float exp_score = expf(score - new_max);
                    oc_attn_scale_f32(outs[g], exp_factor, hd);
                    oc_attn_axpy_q8(outs[g], vq, exp_score * v_scale, hd);
                    run_sum[g] = run_sum[g] * exp_factor + exp_score;
                    run_max[g] = new_max;
                }
            } else {
                const float *k_t = s->kv_k + kv_off + (size_t)t * kv_row;
                const float *v_t = s->kv_v + kv_off + (size_t)t * kv_row;
                for (uint32_t g = 0; g < nq; g++) {
                    float score = oc_attn_dot_f32(qs[g], k_t, hd) * j->scale;
                    float new_max = (score > run_max[g]) ? score : run_max[g];
                    float exp_factor = expf(run_max[g] - new_max);
                    float exp_score = expf(score - new_max);
                    oc_attn_scale_f32(outs[g], exp_factor, hd);
                    oc_attn_axpy_f32(outs[g], v_t, exp_score, hd);
                    run_sum[g] = run_sum[g] * exp_factor + exp_score;
                    run_max[g] = new_max;
                }
            }
        }
        for (uint32_t g = 0; g < nq; g++) {
            if (run_sum[g] > 0.0f)
                oc_attn_scale_f32(outs[g], 1.0f / run_sum[g], hd);
        }
    }
}

static int use_compressed_attn(const OcLlamaSession *s, uint32_t layer)
{
    if (!s || !s->kv_compress || !s->model) return 0;
    return layer_qualifies_compressed(s->model, layer);
}

static void rewind_compressed_to(OcLlamaSession *s, size_t n_keep)
{
    if (!s || !s->kv_compress) return;
    (void)oc_compressed_kv_rewind(s->kv_compress, n_keep);
}

static OcError store_compressed_token(OcLlamaSession *s, uint32_t layer,
                                      uint32_t n_kv, size_t hd, const float *k,
                                      const float *v, size_t pos)
{
    uint32_t h;
    for (h = 0; h < n_kv; h++) {
        OcError e = oc_compressed_kv_append(s->kv_compress, layer, h,
                                            k + h * hd, v + h * hd, &pos,
                                            1);
        if (e != OC_OK) return e;
    }
    return OC_OK;
}

static OcError attention_decode_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    const OcLlamaLayer *GL = layer_for_attn(s, layer);
    size_t hd = GL->head_dim ? (size_t)GL->head_dim : (size_t)c->head_dim;
    uint32_t n_kv = GL->n_head_kv ? GL->n_head_kv : c->n_head_kv;
    uint32_t group = c->n_head / n_kv;
    if (use_compressed_attn(s, layer)) {
        uint32_t kh, g, h;
        for (kh = 0; kh < n_kv; kh++) {
            for (g = 0; g < group; g++) {
                OcError e;
                h = kh * group + g;
                if (h >= c->n_head) break;
                e = oc_compressed_kv_attention(s->kv_compress, layer, kh,
                                               s->q + h * hd, hd, (size_t)s->pos,
                                               s->attn_out + h * hd);
                if (e != OC_OK) {
                    rewind_compressed_to(s, (size_t)s->pos);
                    return e;
                }
            }
        }
        return OC_OK;
    }
    float scale = (c->attn_scale > 0.0f) ? c->attn_scale
                                         : (1.0f / sqrtf((float)hd));
    int64_t seq_len = s->pos + 1;
    int64_t start = 0;
    if (GL->sliding_window > 0) {
        int64_t sw = (int64_t)GL->sliding_window;
        start = (seq_len > sw) ? (seq_len - sw) : 0;
    } else if (!c->uses_gemma4 && c->layer_is_swa == NULL &&
               c->sliding_window > 0 && c->sliding_window_pattern > 1) {
        if (layer % c->sliding_window_pattern == 1) {
            int64_t sw = (int64_t)c->sliding_window;
            start = (seq_len > sw) ? (seq_len - sw) : 0;
        }
    }
    AttnDecodeJob job = {
        .s = s,
        .layer = layer,
        .start = start,
        .seq_len = seq_len,
        .hd = hd,
        .group = group,
        .n_kv = n_kv,
        .scale = scale,
        .q8 = (s->kv_type == OC_KV_Q8),
    };
    /* The pool refuses to split below 8 items (matvec dispatch overhead).
     * Qwen3.5/3.8 often have 4 KV heads, which would otherwise stay serial
     * at 262k context. Pad the iteration space so groups still fan out. */
    oc_parallel_for(n_kv < 8u ? 8u : n_kv, attn_decode_kv_slice, &job);
    return OC_OK;
}

/* Epsilon for the sandwich (post-attention / post-FFN) norms. Muse Glimmer
 * uses 1e-8 there rather than the model's attention epsilon; every other
 * architecture leaves post_norm_eps at 0 and gets rms_norm_eps. */
static inline float post_norm_eps(const OcLlamaConfig *c)
{
    return c->post_norm_eps > 0.0f ? c->post_norm_eps : c->rms_norm_eps;
}

/* ─── Dense FFN (Llama/Mistral/Qwen2-dense path) ──────────────────────── */

static void forward_dense_ffn(OcLlamaSession *s, const OcLlamaLayer *L)
{
    const OcLlamaConfig *c = &s->model->cfg;
    /* FFN: gate = act(W_gate·x) * (W_up·x); out = W_down·gate.
     * Llama/Mistral/Qwen use SwiGLU (silu). Gemma uses GeGLU (gelu). */
    if (L->ffn_gate.qtype != OC_QUANT_F32 &&
        L->ffn_up.qtype != OC_QUANT_F32) {
        OcGgufQuantizationType qtypes[2] = {
            L->ffn_gate.qtype, L->ffn_up.qtype,
        };
        const uint8_t *data[2] = { L->ffn_gate.data, L->ffn_up.data };
        size_t rows[2] = { L->ffn_gate.rows, L->ffn_up.rows };
        size_t row_bytes[2] = { L->ffn_gate.row_bytes, L->ffn_up.row_bytes };
        float *outputs[2] = { s->ffn_gate, s->ffn_up };
        oc_matvec_quantized_fused(qtypes, data, rows, c->n_embd,
                                  row_bytes, 2, s->normed, outputs,
                                  s->dequant_temp);
    } else {
        matvec(&L->ffn_gate, s->normed, s->ffn_gate, s->dequant_temp);
        matvec(&L->ffn_up,   s->normed, s->ffn_up,   s->dequant_temp);
    }
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
                        c->n_embd, post_norm_eps(c));
        memcpy(s->normed, s->attn_out, c->n_embd * sizeof(float));
    }
    oc_attn_add_f32(s->x, s->normed, c->n_embd);
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
    uint32_t n_zero = c->zero_expert_count;
    uint32_t n_slots = n_exp + n_zero;
    uint32_t k = c->num_experts_per_tok;
    if (k > n_slots) k = n_slots;
    uint32_t i_size = c->expert_intermediate_size;
    if (i_size == 0) i_size = c->n_ff;

    /* 1. Router logits: ffn_gate_inp @ normed → [n_slots]. */
    matvec(&L->ffn_gate_inp, s->normed, s->router_logits, s->dequant_temp);

    /* 2. Gating: softmax (Qwen3-MoE) or sigmoid (DeepSeek). */
    if (!c->expert_gating_sigmoid) {
        float mx = s->router_logits[0];
        for (uint32_t i = 1; i < n_slots; i++) {
            if (s->router_logits[i] > mx) mx = s->router_logits[i];
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < n_slots; i++) {
            s->router_logits[i] = expf(s->router_logits[i] - mx);
            sum += (double)s->router_logits[i];
        }
        if (sum > 0.0) {
            float inv = (float)(1.0 / sum);
            for (uint32_t i = 0; i < n_slots; i++) s->router_logits[i] *= inv;
        }
    } else {
        for (uint32_t i = 0; i < n_slots; i++) {
            s->router_logits[i] = 1.0f / (1.0f + expf(-s->router_logits[i]));
        }
    }

    /* 3. Top-k selection by descending weight (partial selection sort). */
    uint32_t *sel = s->selected_experts;
    if (sel == NULL) { forward_dense_ffn(s, L); return; }
    for (uint32_t i = 0; i < n_slots; i++) sel[i] = i;
    /* exp_probs_b steers WHICH slots win top-k without changing how much
     * their output counts, so ranking uses logit + bias while the applied
     * gate below stays the unbiased probability. */
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < n_slots; j++) {
            float sj = s->router_logits[sel[j]];
            float sb = s->router_logits[sel[best]];
            if (L->exp_probs_b) {
                sj += L->exp_probs_b[sel[j]];
                sb += L->exp_probs_b[sel[best]];
            }
            if (sj > sb) best = j;
        }
        uint32_t tmp = sel[i]; sel[i] = sel[best]; sel[best] = tmp;
    }

    /* 4. Renormalize top-k weights (norm_topk_prob).
     *
     * NOT for zero-expert models: their softmax already spans every slot and
     * routed mass summing to less than 1 is exactly how a token skips work.
     * Rescaling to 1 would force every token through a full-strength FFN. */
    double weight_norm = 1.0;
    if (n_zero == 0) {
        weight_norm = 0.0;
        for (uint32_t i = 0; i < k; i++)
            weight_norm += (double)s->router_logits[sel[i]];
        if (weight_norm <= 0.0) weight_norm = 1.0;
    }
    float routed_scale = c->expert_weights_scale;

    /* 5. Per-expert SwiGLU FFN, accumulated into s->expert_out (zeroed).
     *    Each expert's down output goes into s->shexp_out (temp, n_embd),
     *    then is scaled by w and added to s->expert_out. */
    memset(s->expert_out, 0, c->n_embd * sizeof(float));
    size_t gate_row_bytes = L->ffn_gate_exps.row_bytes;
    size_t up_row_bytes   = L->ffn_up_exps.row_bytes;
    size_t down_row_bytes = L->ffn_down_exps.row_bytes;

#if defined(OC_MOE_SERIAL)
    bool grouped = false;
#else
    bool grouped = s->expert_gate_all && s->expert_up_all &&
                   s->expert_down_all &&
                   L->ffn_gate_exps.qtype != OC_QUANT_F32 &&
                   L->ffn_up_exps.qtype != OC_QUANT_F32 &&
                   L->ffn_down_exps.qtype != OC_QUANT_F32;
#endif
    uint32_t n_routed = 0;
    for (uint32_t ei = 0; ei < k; ei++)
        if (sel[ei] < n_exp) n_routed++;

    if (grouped && n_routed > 0) {
        const size_t n_gate_up = (size_t)n_routed * 2;
        OcGgufQuantizationType gate_up_qtypes[n_gate_up];
        const uint8_t *gate_up_data[n_gate_up];
        size_t gate_up_rows[n_gate_up];
        size_t gate_up_strides[n_gate_up];
        float *gate_up_outputs[n_gate_up];
        OcGgufQuantizationType down_qtypes[n_routed];
        const uint8_t *down_data[n_routed];
        const float *down_inputs[n_routed];
        float *down_outputs[n_routed];
        size_t down_rows[n_routed];
        size_t down_strides[n_routed];

        uint32_t routed = 0;
        for (uint32_t ei = 0; ei < k; ei++) {
            const uint32_t idx = sel[ei];
            if (idx >= n_exp) continue;
            const size_t gate_slot = (size_t)routed * 2;
            gate_up_qtypes[gate_slot] = L->ffn_gate_exps.qtype;
            gate_up_qtypes[gate_slot + 1] = L->ffn_up_exps.qtype;
            gate_up_data[gate_slot] = L->ffn_gate_exps.data +
                (size_t)idx * i_size * gate_row_bytes;
            gate_up_data[gate_slot + 1] = L->ffn_up_exps.data +
                (size_t)idx * i_size * up_row_bytes;
            gate_up_rows[gate_slot] = i_size;
            gate_up_rows[gate_slot + 1] = i_size;
            gate_up_strides[gate_slot] = gate_row_bytes;
            gate_up_strides[gate_slot + 1] = up_row_bytes;
            gate_up_outputs[gate_slot] = s->expert_gate_all +
                (size_t)routed * i_size;
            gate_up_outputs[gate_slot + 1] = s->expert_up_all +
                (size_t)routed * i_size;
            routed++;
        }
        oc_matvec_quantized_fused(
            gate_up_qtypes, gate_up_data, gate_up_rows, c->n_embd,
            gate_up_strides, n_gate_up, s->normed, gate_up_outputs,
            s->dequant_temp);

        for (uint32_t r = 0; r < n_routed; r++) {
            float *gate = s->expert_gate_all + (size_t)r * i_size;
            float *up = s->expert_up_all + (size_t)r * i_size;
            oc_swiglu_inplace_f32(gate, up, i_size);
        }

        routed = 0;
        for (uint32_t ei = 0; ei < k; ei++) {
            const uint32_t idx = sel[ei];
            if (idx >= n_exp) continue;
            down_qtypes[routed] = L->ffn_down_exps.qtype;
            down_data[routed] = L->ffn_down_exps.data +
                (size_t)idx * c->n_embd * down_row_bytes;
            down_inputs[routed] = s->expert_gate_all +
                (size_t)routed * i_size;
            down_outputs[routed] = s->expert_down_all +
                (size_t)routed * c->n_embd;
            down_rows[routed] = c->n_embd;
            down_strides[routed] = down_row_bytes;
            routed++;
        }
        oc_matvec_quantized_multi_input(
            down_qtypes, down_data, down_rows, i_size, down_strides,
            n_routed, down_inputs, down_outputs, s->dequant_temp);

        routed = 0;
        for (uint32_t ei = 0; ei < k; ei++) {
            const uint32_t idx = sel[ei];
            const float w = routed_scale *
                (float)(s->router_logits[idx] / weight_norm);
            if (idx < n_exp) {
                const float *down = s->expert_down_all +
                    (size_t)routed * c->n_embd;
                for (size_t i = 0; i < c->n_embd; i++)
                    s->expert_out[i] += w * down[i];
                routed++;
            } else {
                /* Identity ("zero") expert: holds no weights and returns its
                 * input, gated like any other expert. Same contract as the
                 * batched path in prefill_moe_ffn(). */
                for (size_t i = 0; i < c->n_embd; i++)
                    s->expert_out[i] += w * s->normed[i];
            }
        }
    } else {
        for (uint32_t ei = 0; ei < k; ei++) {
            uint32_t idx = sel[ei];
            float w = routed_scale *
                (float)(s->router_logits[idx] / weight_norm);
            if (idx >= n_exp) {
                /* Identity ("zero") expert: passthrough, gated. */
                for (size_t i = 0; i < c->n_embd; i++)
                    s->expert_out[i] += w * s->normed[i];
                continue;
            }
            OcWeightView gate_v = L->ffn_gate_exps;
            gate_v.data += (size_t)idx * i_size * gate_row_bytes;
            gate_v.rows = i_size;
            OcWeightView up_v = L->ffn_up_exps;
            up_v.data += (size_t)idx * i_size * up_row_bytes;
            up_v.rows = i_size;
            OcWeightView down_v = L->ffn_down_exps;
            down_v.data += (size_t)idx * c->n_embd * down_row_bytes;
            down_v.rows = c->n_embd;
            matvec(&gate_v, s->normed, s->expert_gate, s->dequant_temp);
            matvec(&up_v, s->normed, s->expert_up, s->dequant_temp);
            oc_swiglu_inplace_f32(s->expert_gate, s->expert_up, i_size);
            matvec(&down_v, s->expert_gate, s->shexp_out, s->dequant_temp);
            for (size_t i = 0; i < c->n_embd; i++)
                s->expert_out[i] += w * s->shexp_out[i];
        }
    }

    /* 6. Shared expert (always active, added with weight 1.0). Optional
     *    sigmoid gate via ffn_gate_inp_shexp (Qwen2-MoE shared_expert_gate). */
    if (L->ffn_gate_shexp.data != NULL && L->ffn_up_shexp.data != NULL &&
        L->ffn_down_shexp.data != NULL) {
        matvec(&L->ffn_gate_shexp, s->normed, s->shexp_gate, s->dequant_temp);
        matvec(&L->ffn_up_shexp,   s->normed, s->shexp_up,   s->dequant_temp);
        const size_t shared_size = c->shared_expert_intermediate_size;
        oc_swiglu_inplace_f32(s->shexp_gate, s->shexp_up, shared_size);
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

    /* Resolve the deepseek_yarn split ONCE: how much of the correction rides
     * on cos/sin (rope_amp) versus the attention logits (attn_scale).
     * DeepSeek-V3 (mscale_all_dim 0) puts it on cos/sin; LongCat (both terms
     * 1) cancels the RoPE share to exactly 1.0 and puts it all on the
     * logits. Applying the standard YaRN mscale here as well would
     * double-count -- inflating the 64 rope dims against the 128 nope dims
     * AND scaling the logits. */
    float rope_amp = -1.0f;   /* < 0 = standard YaRN mscale */
    float attn_scale = 1.0f / sqrtf((float)q_hd);
    if (c->yarn_factor > 1.0f && c->yarn_mscale_all_dim > 0.0f) {
        oc_rope_deepseek_yarn_scales(c->yarn_factor, c->yarn_mscale,
                                      c->yarn_mscale_all_dim, q_hd,
                                      &rope_amp, &attn_scale);
    }

    /* 6. RoPE on k_pe (the trailing q_rope slots of kv_compressed).
     *
     * YaRN changes the ROTATION FREQUENCIES, not just an amplitude, so it is
     * still required here even for LongCat where the amplitude cancels to
     * 1.0 -- a 120x context extension without the frequency ramp would place
     * every position wrong. */
    float *k_pe = s->mla_kv_compressed + kv_lora;
    if (c->yarn_factor > 1.0f) {
        oc_apply_rope_yarn_scaled_f32(k_pe, k_pe, q_rope, q_rope, s->pos,
                                      c->rope_theta, c->yarn_factor,
                                      c->yarn_orig_ctx, rope_amp);
    } else {
        oc_apply_rope_f32(k_pe, k_pe, q_rope, q_rope, s->pos, c->rope_theta);
    }

    /* 7. Cache the LATENT, not the expanded per-head K/V.
     *
     * The obvious implementation decompresses c_kv into 64 heads of K and V
     * and caches those: 64 * 192 floats for K plus the same for V, i.e.
     * 7.5 MB per token on LongCat-2.0, which puts even an 8k context out of
     * reach. But every head's K and V are linear functions of the SAME 576
     * cached numbers, so storing [c_kv | k_pe] and folding the projections
     * into the query instead costs 576 floats per token per layer -- 21x
     * smaller -- and is also less arithmetic per step, because k_b and v_b
     * are then applied once per token rather than once per cached position.
     *
     *   score_t = q_nope . (k_b[h] @ c_kv_t) + q_pe . k_pe_t
     *           = (k_b[h]^T @ q_nope) . c_kv_t + q_pe . k_pe_t
     *   out     = v_b[h] @ (sum_t p_t * c_kv_t)
     *
     * so k_b is absorbed into the query up front and v_b is applied once to
     * the attention-weighted latent at the end. Both are exact -- this is a
     * re-association, not an approximation. */
    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos) * s->kv_row_floats;
    float *latent_row = s->kv_k + kv_off;
    memcpy(latent_row, s->mla_c_kv, kv_lora * sizeof(float));
    memcpy(latent_row + kv_lora, k_pe, q_rope * sizeof(float));

    /* 8. RoPE on q_pe (the trailing q_rope slots of each q head). Must match
     * the k_pe treatment above exactly, or Q and K rotate apart. */
    for (uint32_t h = 0; h < n_head; h++) {
        float *q_pe = s->mla_q_full + h * q_hd + k_nope_hd;
        if (c->yarn_factor > 1.0f) {
            oc_apply_rope_yarn_scaled_f32(q_pe, q_pe, q_rope, q_rope, s->pos,
                                          c->rope_theta, c->yarn_factor,
                                          c->yarn_orig_ctx, rope_amp);
        } else {
            oc_apply_rope_f32(q_pe, q_pe, q_rope, q_rope, s->pos, c->rope_theta);
        }
    }

    /* 9. Attention, scoring against the cached latent.
     *
     * The position loop is OUTER and the head loop INNER. That ordering is
     * the whole point: all 64 heads score against the SAME cached row, so a
     * head-outer loop re-reads the entire layer cache once per head. At an
     * 8k context that is 76 * 64 * 8192 * 2304 B = 92 GB of KV traffic per
     * token, five times the ~18.5 GB of weight traffic the model itself
     * moves. Read each row once and fan it out across heads instead: 1.4 GB.
     *
     * The cost is carrying per-head online-softmax state (running max, sum,
     * and a kv_lora accumulator) concurrently rather than one head at a
     * time -- 64 * 512 floats each for the absorbed queries and the
     * accumulators, 256 KB total. */
    const float scale = attn_scale;
    const size_t layer_base = (size_t)layer * c->n_ctx * s->kv_row_floats;
    const int64_t seq_len = s->pos + 1;

    /* Absorb k_b into every head's query up front: q_abs[h] = k_b[h]^T @ q_nope. */
    for (uint32_t h = 0; h < n_head; h++) {
        const float *q_nope = s->mla_q_full + (size_t)h * q_hd;
        OcWeightView k_b_h = L->mla_k_b;
        k_b_h.data = L->mla_k_b.data + (size_t)h * k_nope_hd * L->mla_k_b.row_bytes;
        k_b_h.rows = k_nope_hd;
        matvec_transpose_acc(&k_b_h, q_nope,
                             s->mla_q_absorbed + (size_t)h * kv_lora,
                             s->dequant_temp);
        s->mla_run_max[h] = -INFINITY;
        s->mla_run_sum[h] = 0.0f;
        float *ctx = s->mla_ctx_latent + (size_t)h * kv_lora;
        for (size_t i = 0; i < kv_lora; i++) ctx[i] = 0.0f;
    }

    for (int64_t t = 0; t < seq_len; t++) {
        const float *row = s->kv_k + layer_base + (size_t)t * s->kv_row_floats;
        const float *row_pe = row + kv_lora;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *q_abs = s->mla_q_absorbed + (size_t)h * kv_lora;
            const float *q_pe  = s->mla_q_full + (size_t)h * q_hd + k_nope_hd;
            float dot = 0.0f;
            for (size_t i = 0; i < kv_lora; i++) dot += q_abs[i] * row[i];
            for (size_t i = 0; i < q_rope; i++) dot += q_pe[i] * row_pe[i];

            const float score = dot * scale;
            float *ctx = s->mla_ctx_latent + (size_t)h * kv_lora;
            const float run_max = s->mla_run_max[h];
            const float new_max = (score > run_max) ? score : run_max;
            const float exp_factor = expf(run_max - new_max);
            const float exp_score = expf(score - new_max);
            for (size_t i = 0; i < kv_lora; i++)
                ctx[i] = ctx[i] * exp_factor + exp_score * row[i];
            s->mla_run_sum[h] = s->mla_run_sum[h] * exp_factor + exp_score;
            s->mla_run_max[h] = new_max;
        }
    }

    /* out[h] = v_b[h] @ context_latent. Packed at v_hd, because o_proj takes
     * n_head * v_hd inputs -- writing at the Q stride would feed it every
     * head interleaved with dead floats. */
    for (uint32_t h = 0; h < n_head; h++) {
        float *ctx = s->mla_ctx_latent + (size_t)h * kv_lora;
        const float inv = (s->mla_run_sum[h] > 0.0f) ? 1.0f / s->mla_run_sum[h] : 0.0f;
        for (size_t i = 0; i < kv_lora; i++) ctx[i] *= inv;

        OcWeightView v_b_h = L->mla_v_b;
        v_b_h.data = L->mla_v_b.data + (size_t)h * v_hd * L->mla_v_b.row_bytes;
        v_b_h.rows = v_hd;
        matvec(&v_b_h, ctx, s->attn_out + (size_t)h * v_hd, s->dequant_temp);
    }

    /* 10. Output projection. */
    matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    oc_attn_add_f32(s->x, s->normed, n_embd);
}

static float qwen35_sigmoid(float value)
{
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    const float exponential = expf(value);
    return exponential / (1.0f + exponential);
}

static OcError forward_qwen35_recurrent(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    const OcLlamaLayer *weights = &s->model->layers[layer];
    const size_t key_dim = (size_t)c->ssm_group_count * c->ssm_state_size;
    const size_t conv_dim = 2u * key_dim + c->ssm_inner_size;

    const OcWeightView *projections[] = {
        &weights->attn_qkv, &weights->attn_gate,
        &weights->ssm_beta, &weights->ssm_alpha,
    };
    float *outputs[] = {
        s->qwen35_qkv, s->qwen35_gate,
        s->qwen35_beta, s->qwen35_alpha,
    };
    if (weights->attn_qkv.qtype != OC_QUANT_F32 &&
        weights->attn_gate.qtype != OC_QUANT_F32 &&
        weights->ssm_beta.qtype != OC_QUANT_F32 &&
        weights->ssm_alpha.qtype != OC_QUANT_F32) {
        OcGgufQuantizationType qtypes[4];
        const uint8_t *data[4];
        size_t rows[4];
        size_t row_bytes[4];
        for (size_t i = 0; i < 4; i++) {
            qtypes[i] = projections[i]->qtype;
            data[i] = projections[i]->data;
            rows[i] = projections[i]->rows;
            row_bytes[i] = projections[i]->row_bytes;
        }
        oc_matvec_quantized_fused(qtypes, data, rows, c->n_embd,
                                  row_bytes, 4, s->normed, outputs,
                                  s->dequant_temp);
    } else {
        for (size_t i = 0; i < 4; i++)
            matvec(projections[i], s->normed, outputs[i], s->dequant_temp);
    }

    OcQwen35DeltaParams params = {
        .conv_weight = (const float *)weights->ssm_conv1d.data,
        .conv_weight_len = weights->ssm_conv1d.rows *
                           weights->ssm_conv1d.cols,
        .ssm_a = (const float *)weights->ssm_a.data,
        .ssm_a_len = weights->ssm_a.rows * weights->ssm_a.cols,
        .dt_bias = (const float *)weights->ssm_dt_bias.data,
        .dt_bias_len = weights->ssm_dt_bias.rows * weights->ssm_dt_bias.cols,
        .norm_weight = (const float *)weights->ssm_norm.data,
        .norm_weight_len = weights->ssm_norm.rows * weights->ssm_norm.cols,
        .norm_eps = c->rms_norm_eps,
    };
    OcQwen35DeltaInput input = {
        .qkv = s->qwen35_qkv,
        .qkv_len = conv_dim,
        .gate = s->qwen35_gate,
        .gate_len = c->ssm_inner_size,
        .beta = s->qwen35_beta,
        .beta_len = c->ssm_value_heads,
        .alpha = s->qwen35_alpha,
        .alpha_len = c->ssm_value_heads,
    };
    OcError error = oc_qwen35_delta_step(
        &s->qwen35_delta[layer], &params, &input,
        s->qwen35_conv_output, conv_dim,
        s->qwen35_delta_output, c->ssm_inner_size);
    if (error != OC_OK) return error;
    matvec(&weights->ssm_out, s->qwen35_delta_output, s->normed,
           s->dequant_temp);
    oc_attn_add_f32(s->x, s->normed, c->n_embd);
    return OC_OK;
}

static void forward_qwen35_full_attention_w(OcLlamaSession *s,
                                            OcLlamaLayer *weights,
                                            uint32_t attn_layer,
                                            int64_t rope_pos)
{
    const OcLlamaConfig *c = &s->model->cfg;
    const size_t head_dim = weights->head_dim;
    const size_t kv_dim = (size_t)weights->n_head_kv * head_dim;

    if (weights->attn_q.qtype != OC_QUANT_F32 &&
        weights->attn_k.qtype != OC_QUANT_F32 &&
        weights->attn_v.qtype != OC_QUANT_F32) {
        OcGgufQuantizationType qtypes[3] = {
            weights->attn_q.qtype, weights->attn_k.qtype, weights->attn_v.qtype,
        };
        const uint8_t *data[3] = {
            weights->attn_q.data, weights->attn_k.data, weights->attn_v.data,
        };
        size_t rows[3] = {
            weights->attn_q.rows, weights->attn_k.rows, weights->attn_v.rows,
        };
        size_t row_bytes[3] = {
            weights->attn_q.row_bytes, weights->attn_k.row_bytes,
            weights->attn_v.row_bytes,
        };
        float *outputs[3] = { s->qwen35_qkv, s->k, s->v };
        oc_matvec_quantized_fused(qtypes, data, rows, c->n_embd,
                                  row_bytes, 3, s->normed, outputs,
                                  s->dequant_temp);
    } else {
        matvec(&weights->attn_q, s->normed, s->qwen35_qkv, s->dequant_temp);
        matvec(&weights->attn_k, s->normed, s->k, s->dequant_temp);
        matvec(&weights->attn_v, s->normed, s->v, s->dequant_temp);
    }
    for (uint32_t head = 0; head < c->n_head; head++) {
        const float *packed = s->qwen35_qkv + 2u * head * head_dim;
        memcpy(s->q + (size_t)head * head_dim, packed,
               head_dim * sizeof(float));
        memcpy(s->qwen35_gate + (size_t)head * head_dim, packed + head_dim,
               head_dim * sizeof(float));
        oc_rms_norm_f32(s->q + (size_t)head * head_dim,
                        weights->attn_q_norm, s->qwen35_conv_output,
                        head_dim, c->rms_norm_eps);
        memcpy(s->q + (size_t)head * head_dim, s->qwen35_conv_output,
               head_dim * sizeof(float));
        oc_apply_rope_f32(s->q + (size_t)head * head_dim,
                          s->q + (size_t)head * head_dim, head_dim,
                          weights->rope_dim, rope_pos, weights->rope_theta);
    }
    for (uint32_t head = 0; head < weights->n_head_kv; head++) {
        oc_rms_norm_f32(s->k + (size_t)head * head_dim,
                        weights->attn_k_norm, s->qwen35_conv_output,
                        head_dim, c->rms_norm_eps);
        memcpy(s->k + (size_t)head * head_dim, s->qwen35_conv_output,
               head_dim * sizeof(float));
        oc_apply_rope_f32(s->k + (size_t)head * head_dim,
                          s->k + (size_t)head * head_dim, head_dim,
                          weights->rope_dim, rope_pos, weights->rope_theta);
    }

    const size_t cache_row = s->kv_row_floats;
    const size_t cache_offset =
        ((size_t)weights->kv_cache_index * c->n_ctx + (size_t)s->pos) *
        cache_row;
    if (s->kv_type == OC_KV_Q8) {
        const size_t scale_offset =
            ((size_t)weights->kv_cache_index * c->n_ctx + (size_t)s->pos) *
            c->n_head_kv;
        for (uint32_t head = 0; head < weights->n_head_kv; head++) {
            kv_q8_encode(s->k + (size_t)head * head_dim,
                         s->kv_k_q + cache_offset + (size_t)head * head_dim,
                         &s->kv_k_scale[scale_offset + head], head_dim);
            kv_q8_encode(s->v + (size_t)head * head_dim,
                         s->kv_v_q + cache_offset + (size_t)head * head_dim,
                         &s->kv_v_scale[scale_offset + head], head_dim);
        }
    } else {
        memcpy(s->kv_k + cache_offset, s->k, kv_dim * sizeof(float));
        memcpy(s->kv_v + cache_offset, s->v, kv_dim * sizeof(float));
    }

    (void)attention_decode_layer(s, attn_layer);
    for (uint32_t head = 0; head < c->n_head; head++) {
        float *head_output = s->attn_out + (size_t)head * head_dim;
        const float *gate = s->qwen35_gate + (size_t)head * head_dim;
        for (size_t i = 0; i < head_dim; i++)
            head_output[i] *= qwen35_sigmoid(gate[i]);
    }
    matvec(&weights->attn_output, s->attn_out, s->normed,
           s->dequant_temp);
    oc_attn_add_f32(s->x, s->normed, c->n_embd);
}

static void forward_qwen35_full_attention(OcLlamaSession *s, uint32_t layer)
{
    forward_qwen35_full_attention_w(s, &s->model->layers[layer], layer,
                                    s->pos);
}

static OcError forward_layer(OcLlamaSession *s, uint32_t layer)
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

    if (c->is_qwen35 && L->kind == OC_LLAMA_LAYER_QWEN35_RECURRENT) {
        OcError error = forward_qwen35_recurrent(s, layer);
        if (error != OC_OK) return error;
    } else if (c->is_qwen35) {
        forward_qwen35_full_attention(s, layer);
    } else if (c->uses_mla && L->mla_kv_a_mqa.data != NULL) {
        forward_mla_attention(s, layer);
    } else {
        /* Q/K/V projections, plus the optional projection biases that
         * Qwen2-family models carry. The bias is added before RoPE, as in
         * llama.cpp's build_qkv / HF's q_proj(x) + b_q. */
        matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
        matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
        matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);
        /* Muse Glimmer's attention-output gate is projected from the same
         * pre-attention normed hidden state as Q/K/V, so it has to be taken
         * before `normed` is reused as the output-projection buffer below. */
        if (c->attn_out_gate && L->attn_gate.data != NULL) {
            matvec(&L->attn_gate, s->normed, s->muse_gate, s->dequant_temp);
        }
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
        /* Compressed path stores pre-RoPE K/V; the facade owns RoPE. */
        if (use_compressed_attn(s, layer)) {
            OcError se = store_compressed_token(s, layer, n_kv, hd, s->k, s->v,
                                                (size_t)s->pos);
            if (se != OC_OK) {
                rewind_compressed_to(s, (size_t)s->pos);
                return se;
            }
        } else if (L->use_rope) {
        for (uint32_t h = 0; h < c->n_head; h++) {
            if (c->yarn_factor > 0.0f) {
                oc_apply_rope_yarn_f32(s->q + h * hd, s->q + h * hd, hd,
                                       rope_dim, s->pos, rope_theta,
                                       c->yarn_factor, c->yarn_orig_ctx);
            } else if (c->rope_norm_pairs) {
                oc_apply_rope_norm_f32(s->q + h * hd, s->q + h * hd, hd,
                                       rope_dim, s->pos, rope_theta);
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
            } else if (c->rope_norm_pairs) {
                oc_apply_rope_norm_f32(s->k + h * hd, s->k + h * hd, hd,
                                       rope_dim, s->pos, rope_theta);
            } else {
                oc_apply_rope_f32(s->k + h * hd, s->k + h * hd, hd, rope_dim,
                                  s->pos, rope_theta);
            }
        }
        }

        /* KV cache write at position `pos`. */
        if (!use_compressed_attn(s, layer)) {
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
        }

        /* Attention: KV streamed once per GQA group. */
        {
            OcError ae = attention_decode_layer(s, layer);
            if (ae != OC_OK) return ae;
        }

        /* Attention output gate: out *= sigmoid(gate), elementwise over the
         * whole n_head*head_dim vector, before the output projection. */
        if (c->attn_out_gate && L->attn_gate.data != NULL) {
            const size_t nq = (size_t)c->n_head * hd;
            for (size_t i = 0; i < nq; i++)
                s->attn_out[i] *= qwen35_sigmoid(s->muse_gate[i]);
        }

        /* Output projection. */
        matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
        /* Gemma "sandwich" norm: the attention branch output is normed again
         * before it rejoins the residual stream (HF post_attention_layernorm
         * on the branch, not on the input). Skipped when the tensor is absent,
         * which is every non-Gemma model. */
        if (L->post_attention_norm != NULL) {
            oc_rms_norm_f32(s->normed, L->post_attention_norm,
                            s->attn_out, c->n_embd, post_norm_eps(c));
            memcpy(s->normed, s->attn_out, c->n_embd * sizeof(float));
        }
        oc_attn_add_f32(s->x, s->normed, c->n_embd);
    }

    /* Pre-FFN RMSNorm (+ Gemma scaling; see the pre-attention norm above). */
    const float *ffn_norm = c->is_qwen35 ? L->post_attention_norm : L->ffn_norm;
    oc_rms_norm_f32(s->x, ffn_norm, s->normed, c->n_embd, c->rms_norm_eps);
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
    return OC_OK;
}

OcError oc_llama_forward(OcLlamaSession *sess, uint32_t token, float *logits_out)
{
    if (sess == NULL || sess->model == NULL) return OC_ERR_INVALID_ARG;
    if ((uint64_t)sess->pos >= sess->model->cfg.n_ctx) return OC_ERR_INVALID_ARG;

    /* Architecture dispatch: LayerNorm-family models use the dedicated
     * forward passes in arch_forward.c. */
    switch (sess->model->arch) {
    case OC_ARCH_GPT2:
        return oc_arch_forward_gpt2(sess, token, logits_out);
    case OC_ARCH_GPTJ:
        return oc_arch_forward_gptj(sess, token, logits_out);
    case OC_ARCH_GPTNEOX: return oc_arch_forward_gpt_neox(sess, token, logits_out);
    case OC_ARCH_FALCON:  return oc_arch_forward_falcon(sess, token, logits_out);
    default: break;
    }

    embed_token(sess, token);
    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        OcError error = forward_layer(sess, l);
        if (error != OC_OK) return error;
    }
    /* Final RMSNorm + lm_head. */
    OcLlamaModel *m = sess->model;
    if (sess->last_hidden != NULL)
        memcpy(sess->last_hidden, sess->x, m->cfg.n_embd * sizeof(float));
    oc_rms_norm_f32(sess->x, m->final_norm, sess->normed, m->cfg.n_embd,
                    m->cfg.rms_norm_eps);
    if (!m->cfg.uses_gemma4 && m->cfg.norm_scale != 1.0f) {
        for (size_t i = 0; i < m->cfg.n_embd; i++) sess->normed[i] *= m->cfg.norm_scale;
    }
    sess->last_token = token;
    if (logits_out != NULL) {
        if (m->output.qtype == OC_QUANT_F32) {
            oc_matvec_f32((const float *)m->output.data, m->output.rows,
                          m->output.cols, sess->normed, logits_out);
        } else {
            oc_matvec_quantized(m->output.qtype, m->output.data, m->output.rows,
                                 m->output.cols, m->output.row_bytes,
                                 sess->normed, logits_out, sess->dequant_temp);
        }
        /* Muse Glimmer scales the lm_head output before the softcap. 0 means
         * "unset" — treat it as 1.0, not as zeroing the logits. */
        if (m->cfg.logit_scale > 0.0f && m->cfg.logit_scale != 1.0f) {
            const float ls = m->cfg.logit_scale;
            for (size_t i = 0; i < m->cfg.vocab_size; i++)
                logits_out[i] *= ls;
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

bool oc_llama_mtp_present(const OcLlamaModel *model)
{
    return model != NULL && model->mtp.present;
}

static OcError mtp_forward_one(OcLlamaSession *sess, uint32_t token,
                               const float *prev_hidden, float *logits_out)
{
    OcLlamaModel *m = sess->model;
    OcLlamaLayer *L = &m->mtp.layer;
    const size_t h = m->cfg.n_embd;
    OcWeightView *emb = m->mtp.embed_tokens.data ? &m->mtp.embed_tokens
                                                 : &m->tok_embeddings;
    if (token >= m->cfg.vocab_size) token = m->cfg.vocab_size - 1;
    if (emb->qtype == OC_QUANT_F32) {
        memcpy(sess->dequant_temp, emb->data + (size_t)token * emb->row_bytes,
               h * sizeof(float));
    } else {
        oc_quant_dequant_row(emb->qtype,
            emb->data + (size_t)token * emb->row_bytes, emb->row_bytes,
            sess->dequant_temp, h);
    }
    oc_rms_norm_f32(sess->dequant_temp, m->mtp.enorm, sess->mtp_concat, h,
                    m->cfg.rms_norm_eps);
    oc_rms_norm_f32(prev_hidden, m->mtp.hnorm, sess->mtp_concat + h, h,
                    m->cfg.rms_norm_eps);
    matvec(&m->mtp.eh_proj, sess->mtp_concat, sess->x, sess->dequant_temp);

    const int64_t saved_pos = sess->pos;
    sess->pos = (int64_t)sess->mtp_pos;
    oc_rms_norm_f32(sess->x, L->attn_norm, sess->normed, h, m->cfg.rms_norm_eps);
    forward_qwen35_full_attention_w(sess, L, m->cfg.n_layer,
                                    saved_pos + (int64_t)sess->mtp_pos);
    const float *ffn_norm = L->post_attention_norm ? L->post_attention_norm
                                                   : L->ffn_norm;
    oc_rms_norm_f32(sess->x, ffn_norm, sess->normed, h, m->cfg.rms_norm_eps);
    if (m->cfg.num_experts > 0 && L->ffn_gate_inp.data != NULL)
        forward_moe_ffn(sess, L);
    else
        forward_dense_ffn(sess, L);
    sess->pos = saved_pos;
    sess->mtp_pos++;

    memcpy(sess->mtp_hidden, sess->x, h * sizeof(float));
    const float *hn = m->mtp.shared_head_norm ? m->mtp.shared_head_norm
                                              : m->final_norm;
    oc_rms_norm_f32(sess->x, hn, sess->normed, h, m->cfg.rms_norm_eps);
    OcWeightView *head = m->mtp.shared_head_head.data ? &m->mtp.shared_head_head
                                                      : &m->output;
    if (head->qtype == OC_QUANT_F32) {
        oc_matvec_f32((const float *)head->data, head->rows, head->cols,
                      sess->normed, logits_out);
    } else {
        oc_matvec_quantized(head->qtype, head->data, head->rows, head->cols,
                            head->row_bytes, sess->normed, logits_out,
                            sess->dequant_temp);
    }
    return OC_OK;
}

static float mtp_pairwise_conf(const float *logits, size_t vocab)
{
    if (logits == NULL || vocab == 0) return 0.0f;
    float b1 = logits[0], b2 = -INFINITY;
    for (size_t i = 1; i < vocab; i++) {
        float v = logits[i];
        if (v > b1) { b2 = b1; b1 = v; }
        else if (v > b2) b2 = v;
    }
    return 1.0f / (1.0f + expf(b2 - b1));
}

OcError oc_llama_mtp_draft_tokens(OcLlamaSession *sess, uint32_t k,
                                  uint32_t *out_tokens, float *out_conf,
                                  uint32_t *n_out)
{
    if (sess == NULL || out_tokens == NULL || n_out == NULL)
        return OC_ERR_INVALID_ARG;
    *n_out = 0;
    if (!sess->model || !sess->model->mtp.present || sess->last_hidden == NULL)
        return OC_ERR_MODEL;
    if (k == 0) return OC_OK;
    if (k > 8) k = 8;
    OcLlamaModel *m = sess->model;
    uint64_t remaining = sess->pos < (int64_t)m->cfg.n_ctx
        ? (uint64_t)m->cfg.n_ctx - (uint64_t)sess->pos : 0;
    if ((uint64_t)k > remaining) k = (uint32_t)remaining;
    if (k == 0) return OC_OK;
    memcpy(sess->mtp_hidden, sess->last_hidden, m->cfg.n_embd * sizeof(float));
    uint32_t cur = sess->last_token;
    sess->mtp_pos = 0;
    float *logits = sess->logits;
    for (uint32_t i = 0; i < k; i++) {
        OcError e = mtp_forward_one(sess, cur, sess->mtp_hidden, logits);
        if (e != OC_OK) return e;
        out_tokens[i] = oc_argmax(logits, m->cfg.vocab_size);
        if (out_conf) out_conf[i] = mtp_pairwise_conf(logits, m->cfg.vocab_size);
        cur = out_tokens[i];
        (*n_out)++;
    }
    return OC_OK;
}

OcError oc_llama_mtp_greedy_advance(OcLlamaSession *sess, float *logits,
                                    uint32_t *out_tokens, size_t max_out,
                                    size_t *n_out)
{
    if (sess == NULL || logits == NULL || out_tokens == NULL || n_out == NULL)
        return OC_ERR_INVALID_ARG;
    *n_out = 0;
    if (sess->model == NULL)
        return OC_ERR_INVALID_ARG;
    if (max_out == 0) return OC_OK;
    OcLlamaModel *m = sess->model;
    const uint32_t vocab = m->cfg.vocab_size;
    if (!m->mtp.present || sess->last_hidden == NULL || max_out == 1) {
        uint32_t t = oc_argmax(logits, vocab);
        out_tokens[(*n_out)++] = t;
        return oc_llama_forward(sess, t, logits);
    }

    const uint32_t target0 = oc_argmax(logits, vocab);
    uint32_t drafts[8];
    size_t k = OC_MTP_DEFAULT_DRAFT;
    if (k > 8) k = 8;
    if (k > max_out) k = max_out;
    uint64_t remaining = sess->pos < (int64_t)m->cfg.n_ctx
        ? (uint64_t)m->cfg.n_ctx - (uint64_t)sess->pos : 0;
    if ((uint64_t)k > remaining) k = (size_t)remaining;
    if (k == 0) return OC_ERR_INVALID_ARG;

    memcpy(sess->mtp_hidden, sess->last_hidden, m->cfg.n_embd * sizeof(float));
    uint32_t cur = sess->last_token;
    sess->mtp_pos = 0;
    for (size_t i = 0; i < k; i++) {
        OcError e = mtp_forward_one(sess, cur, sess->mtp_hidden, logits);
        if (e != OC_OK) return e;
        drafts[i] = oc_argmax(logits, vocab);
        cur = drafts[i];
    }

    out_tokens[(*n_out)++] = target0;
    OcError e = oc_llama_forward(sess, target0, logits);
    if (e != OC_OK) return e;
    if (drafts[0] != target0) return OC_OK;
    size_t accepted = 1;
    while (accepted < k && *n_out < max_out) {
        uint32_t t = oc_argmax(logits, vocab);
        if (t != drafts[accepted]) break;
        out_tokens[(*n_out)++] = t;
        e = oc_llama_forward(sess, t, logits);
        if (e != OC_OK) return e;
        accepted++;
    }
    return OC_OK;
}

/* ─── Batched prefill ────────────────────────────────────────────────────
 *
 * Prompt processing is throughput work, not latency work: every prompt token
 * is known up front, so they can share one pass over the weights instead of
 * each dragging the whole model through DRAM. The per-token path costs one
 * full weight sweep per token — on Qwen3-30B-A3B that is ~1.5 GB per token,
 * which measured 2.9 tok/s against llama.cpp's 129 tok/s for the same file.
 *
 * The arithmetic here is deliberately IDENTICAL to forward_layer(): same
 * norms, same RoPE, same online-softmax attention (reused verbatim, one
 * token at a time), same routing. Only the *matmuls* change shape — they go
 * through oc_matvec_*_batch(), which dots each weight row against a tile of
 * activations while the row is in cache. Attention itself is left per-token
 * because it touches the KV cache, not the weights, so it was never the
 * bottleneck.
 *
 * For the MoE FFN the win needs one extra step: tokens are grouped BY EXPERT
 * (a counting sort over the top-k choices) so each expert's weights are read
 * once for all the tokens that routed to it. Without that grouping a batch
 * of 512 tokens would still touch 512 * 8 expert weight sets independently
 * and nothing would be amortized.
 */

/* Tokens per prefill chunk. Larger amortizes the weight sweep further —
 * especially for MoE, where the win scales with tokens-per-expert — at a
 * linear cost in scratch (~127 KB/token on a 30B MoE, so ~65 MB here). */
#define OC_PREFILL_CHUNK 512u

/* Per-phase prefill timers. Prefill is a mix of weight-bound matmuls,
 * cache-bound attention and plain memory shuffling, and which one dominates
 * is not guessable — it moves with chunk size, expert count and thread
 * count. Accumulated per oc_llama_prefill() call and reported at DEBUG. */
typedef struct {
    double norm, qkv, rope_kv, attn, proj, router, gather, expert_mm, scatter;
} PrefillTimers;

static PrefillTimers g_pf_t;

static double pf_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}


typedef struct {
    size_t cap;             /* tokens per chunk                          */
    size_t n_embd, n_qo, kv_row, ffw, n_slots, k;
    float *x, *normed, *q, *k_buf, *v_buf, *attn_out, *proj;
    float *ffn_a, *ffn_b, *gath, *expert_out, *router;
    float *dequant_temp;
    float *q35_qgate, *q35_rqkv, *q35_rgate, *q35_rbeta, *q35_ralpha;
    float *q35_delta;
    float *mgate;           /* [cap][n_qo] Muse attention-output gate    */
    uint8_t *act;           /* oc_matvec_quantized_batch scratch         */
    size_t   act_bytes;
    uint32_t *sel;          /* [cap][k] chosen expert slots              */
    float    *sel_w;        /* [cap][k] applied gate weights             */
    uint32_t *ex_tok;       /* [cap*k] token ids, grouped by expert      */
    float    *ex_w;         /* [cap*k] matching gate weights             */
    uint32_t *ex_off;       /* [n_slots+1] group offsets                 */
    uint32_t *ex_fill;      /* [n_slots] cursor while filling            */
} PrefillBuf;

static void prefill_buf_free(PrefillBuf *b)
{
    free(b->x); free(b->normed); free(b->q); free(b->k_buf); free(b->v_buf);
    free(b->attn_out); free(b->proj); free(b->ffn_a); free(b->ffn_b);
    free(b->gath); free(b->expert_out); free(b->router);
    free(b->dequant_temp); free(b->act);
    free(b->q35_qgate); free(b->q35_rqkv); free(b->q35_rgate);
    free(b->q35_rbeta); free(b->q35_ralpha); free(b->q35_delta);
    free(b->mgate);
    free(b->sel); free(b->sel_w); free(b->ex_tok); free(b->ex_w);
    free(b->ex_off); free(b->ex_fill);
    memset(b, 0, sizeof(*b));
}

static OcError prefill_buf_init(const OcLlamaModel *m, size_t cap,
                                PrefillBuf *b)
{
    const OcLlamaConfig *c = &m->cfg;
    memset(b, 0, sizeof(*b));
    b->cap     = cap;
    b->n_embd  = c->n_embd;
    b->n_qo    = (size_t)c->n_head * c->head_dim;
    b->kv_row  = (size_t)c->n_head_kv * c->kv_head_dim;
    size_t i_size = c->expert_intermediate_size ? c->expert_intermediate_size
                                                : c->n_ff;
    b->ffw     = c->n_ff > i_size ? c->n_ff : i_size;
    b->n_slots = (size_t)c->num_experts + c->zero_expert_count;
    b->k       = c->num_experts_per_tok;
    if (b->k > b->n_slots) b->k = b->n_slots;

    /* Widest matmul input across the pass — the dequant buffer and the fused
     * activation scratch must both cover it. attn_output's input is n_qo,
     * which on a GQA model exceeds n_embd. */
    size_t max_cols = b->n_embd;
    if (b->n_qo > max_cols) max_cols = b->n_qo;
    if (b->ffw  > max_cols) max_cols = b->ffw;
    if (c->ssm_inner_size > max_cols) max_cols = c->ssm_inner_size;

    b->x          = xcalloc(cap * b->n_embd, sizeof(float));
    b->normed     = xcalloc(cap * b->n_embd, sizeof(float));
    b->proj       = xcalloc(cap * b->n_embd, sizeof(float));
    b->gath       = xcalloc(cap * b->n_embd, sizeof(float));
    b->q          = xcalloc(cap * b->n_qo, sizeof(float));
    b->attn_out   = xcalloc(cap * b->n_qo, sizeof(float));
    b->k_buf      = xcalloc(cap * b->kv_row, sizeof(float));
    b->v_buf      = xcalloc(cap * b->kv_row, sizeof(float));
    b->ffn_a      = xcalloc(cap * b->ffw, sizeof(float));
    b->ffn_b      = xcalloc(cap * b->ffw, sizeof(float));
    b->dequant_temp = xcalloc(max_cols, sizeof(float));
    if (!b->x || !b->normed || !b->proj || !b->gath || !b->q || !b->attn_out ||
        !b->k_buf || !b->v_buf || !b->ffn_a || !b->ffn_b || !b->dequant_temp) {
        prefill_buf_free(b);
        return OC_ERR_OOM;
    }

    if (c->attn_out_gate) {
        b->mgate = xcalloc(cap * b->n_qo, sizeof(float));
        if (b->mgate == NULL) { prefill_buf_free(b); return OC_ERR_OOM; }
    }

    if (c->is_qwen35) {
        const size_t key_dim = (size_t)c->ssm_group_count * c->ssm_state_size;
        const size_t conv_dim = 2u * key_dim + c->ssm_inner_size;
        b->q35_qgate  = xcalloc(cap * 2u * b->n_qo, sizeof(float));
        b->q35_rqkv   = xcalloc(cap * conv_dim, sizeof(float));
        b->q35_rgate  = xcalloc(cap * c->ssm_inner_size, sizeof(float));
        b->q35_rbeta  = xcalloc(cap * c->ssm_value_heads, sizeof(float));
        b->q35_ralpha = xcalloc(cap * c->ssm_value_heads, sizeof(float));
        b->q35_delta  = xcalloc(cap * c->ssm_inner_size, sizeof(float));
        if (!b->q35_qgate || !b->q35_rqkv || !b->q35_rgate ||
            !b->q35_rbeta || !b->q35_ralpha || !b->q35_delta) {
            prefill_buf_free(b);
            return OC_ERR_OOM;
        }
    }

    /* One fused-activation buffer covering every matmul in the pass. Its size
     * is a function of the widest input only — the kernel clamps its tile to
     * fit — so it does not grow with `cap`. */
    b->act_bytes = oc_matvec_batch_scratch_bytes(max_cols);
    if (b->act_bytes > 0) {
        b->act = (uint8_t *)malloc(b->act_bytes);
        if (b->act == NULL) { prefill_buf_free(b); return OC_ERR_OOM; }
    }

    if (b->n_slots > 0) {
        b->expert_out = xcalloc(cap * b->n_embd, sizeof(float));
        b->router     = xcalloc(cap * b->n_slots, sizeof(float));
        b->sel        = xcalloc(cap * (b->k ? b->k : 1), sizeof(uint32_t));
        b->sel_w      = xcalloc(cap * (b->k ? b->k : 1), sizeof(float));
        b->ex_tok     = xcalloc(cap * (b->k ? b->k : 1), sizeof(uint32_t));
        b->ex_w       = xcalloc(cap * (b->k ? b->k : 1), sizeof(float));
        b->ex_off     = xcalloc(b->n_slots + 1, sizeof(uint32_t));
        b->ex_fill    = xcalloc(b->n_slots, sizeof(uint32_t));
        if (!b->expert_out || !b->router || !b->sel || !b->sel_w ||
            !b->ex_tok || !b->ex_w || !b->ex_off || !b->ex_fill) {
            prefill_buf_free(b);
            return OC_ERR_OOM;
        }
    }
    return OC_OK;
}

/* Batched weight × activations. Mirrors matvec()'s f32/quantized dispatch. */
static void mm_batch(const OcWeightView *w, const float *in, size_t in_stride,
                     float *out, size_t out_stride, size_t n, PrefillBuf *b)
{
    if (w->qtype == OC_QUANT_F32) {
        oc_matvec_f32_batch((const float *)w->data, w->rows, w->cols,
                            in, in_stride, out, out_stride, n);
    } else {
        oc_matvec_quantized_batch(w->qtype, w->data, w->rows, w->cols,
                                  w->row_bytes, in, in_stride, out, out_stride,
                                  n, b->dequant_temp, b->act, b->act_bytes);
    }
}

/* Batched MoE FFN. Same routing decision per token as forward_moe_ffn(); the
 * difference is that the expert matmuls run once per expert over all of its
 * tokens instead of once per (token, expert) pair. */
static void prefill_moe_ffn(OcLlamaSession *s, const OcLlamaLayer *L,
                            PrefillBuf *b, size_t n)
{
    const OcLlamaConfig *c = &s->model->cfg;
    const uint32_t n_exp = c->num_experts;
    const uint32_t n_zero = c->zero_expert_count;
    const uint32_t n_slots = n_exp + n_zero;
    const uint32_t k = (uint32_t)b->k;
    uint32_t i_size = c->expert_intermediate_size;
    if (i_size == 0) i_size = c->n_ff;
    const float routed_scale = c->expert_weights_scale;

    /* 1. Router logits for every token in one matmul. */
    double t_r0 = pf_now();
    mm_batch(&L->ffn_gate_inp, b->normed, b->n_embd, b->router, b->n_slots,
             n, b);

    g_pf_t.router += pf_now() - t_r0;
    memset(b->expert_out, 0, n * b->n_embd * sizeof(float));
    memset(b->ex_fill, 0, n_slots * sizeof(uint32_t));
    for (uint32_t e = 0; e <= n_slots; e++) b->ex_off[e] = 0;

    /* 2-4. Gating, top-k and weight normalization, per token. Identical to
     *      the single-token path — this is cheap and touches no weights. */
    for (size_t j = 0; j < n; j++) {
        float *rl = b->router + j * b->n_slots;
        if (!c->expert_gating_sigmoid) {
            float mx = rl[0];
            for (uint32_t i = 1; i < n_slots; i++) if (rl[i] > mx) mx = rl[i];
            double sum = 0.0;
            for (uint32_t i = 0; i < n_slots; i++) {
                rl[i] = expf(rl[i] - mx);
                sum += (double)rl[i];
            }
            if (sum > 0.0) {
                float inv = (float)(1.0 / sum);
                for (uint32_t i = 0; i < n_slots; i++) rl[i] *= inv;
            }
        } else {
            for (uint32_t i = 0; i < n_slots; i++)
                rl[i] = 1.0f / (1.0f + expf(-rl[i]));
        }

        uint32_t *sel = b->sel + j * k;
        for (uint32_t i = 0; i < n_slots && i < k; i++) sel[i] = i;
        /* Partial selection sort over a scratch index list, matching the
         * single-token path's ordering exactly (including exp_probs_b
         * steering selection but not the applied gate). */
        uint32_t idx_buf[256];
        uint32_t *idx = (n_slots <= 256)
                      ? idx_buf
                      : (uint32_t *)malloc(n_slots * sizeof(uint32_t));
        if (idx == NULL) { /* fall back to no routing for this token */
            for (uint32_t i = 0; i < k; i++) { b->sel[j * k + i] = 0; }
            continue;
        }
        for (uint32_t i = 0; i < n_slots; i++) idx[i] = i;
        for (uint32_t i = 0; i < k; i++) {
            uint32_t best = i;
            for (uint32_t jj = i + 1; jj < n_slots; jj++) {
                float sj = rl[idx[jj]];
                float sb = rl[idx[best]];
                if (L->exp_probs_b) {
                    sj += L->exp_probs_b[idx[jj]];
                    sb += L->exp_probs_b[idx[best]];
                }
                if (sj > sb) best = jj;
            }
            uint32_t tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp;
            sel[i] = idx[i];
        }
        if (idx != idx_buf) free(idx);

        double weight_norm = 1.0;
        if (n_zero == 0) {
            weight_norm = 0.0;
            for (uint32_t i = 0; i < k; i++) weight_norm += (double)rl[sel[i]];
            if (weight_norm <= 0.0) weight_norm = 1.0;
        }
        for (uint32_t i = 0; i < k; i++) {
            b->sel_w[j * k + i] =
                routed_scale * (float)(rl[sel[i]] / weight_norm);
            b->ex_off[sel[i] + 1]++;      /* histogram for the counting sort */
        }
    }

    /* 5. Counting sort of (token, weight) pairs into per-expert groups. */
    for (uint32_t e = 0; e < n_slots; e++) b->ex_off[e + 1] += b->ex_off[e];
    for (size_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < k; i++) {
            uint32_t e = b->sel[j * k + i];
            uint32_t at = b->ex_off[e] + b->ex_fill[e]++;
            b->ex_tok[at] = (uint32_t)j;
            b->ex_w[at]   = b->sel_w[j * k + i];
        }
    }

    /* 6. One pass per expert over all of its tokens. */
    const size_t gate_rb = L->ffn_gate_exps.row_bytes;
    const size_t up_rb   = L->ffn_up_exps.row_bytes;
    const size_t down_rb = L->ffn_down_exps.row_bytes;

    for (uint32_t e = 0; e < n_slots; e++) {
        const uint32_t m = b->ex_off[e + 1] - b->ex_off[e];
        if (m == 0) continue;
        const uint32_t *toks = b->ex_tok + b->ex_off[e];
        const float    *ws   = b->ex_w   + b->ex_off[e];

        /* Zero ("identity") experts hold no weights: they return their input,
         * gated like any other expert. */
        if (e >= n_exp) {
            for (uint32_t t = 0; t < m; t++) {
                const float *src = b->normed + (size_t)toks[t] * b->n_embd;
                float *dst = b->expert_out + (size_t)toks[t] * b->n_embd;
                const float w = ws[t];
                for (size_t i = 0; i < b->n_embd; i++) dst[i] += w * src[i];
            }
            continue;
        }

        /* Gather this expert's tokens so the matmul sees them contiguously. */
        double t_g0 = pf_now();
        for (uint32_t t = 0; t < m; t++) {
            memcpy(b->gath + (size_t)t * b->n_embd,
                   b->normed + (size_t)toks[t] * b->n_embd,
                   b->n_embd * sizeof(float));
        }
        g_pf_t.gather += pf_now() - t_g0;

        OcWeightView gate_v = L->ffn_gate_exps;
        gate_v.data = L->ffn_gate_exps.data + (size_t)e * i_size * gate_rb;
        gate_v.rows = i_size;
        OcWeightView up_v = L->ffn_up_exps;
        up_v.data = L->ffn_up_exps.data + (size_t)e * i_size * up_rb;
        up_v.rows = i_size;
        OcWeightView down_v = L->ffn_down_exps;
        down_v.data = L->ffn_down_exps.data + (size_t)e * b->n_embd * down_rb;
        down_v.rows = b->n_embd;

        double t_m0 = pf_now();
        mm_batch(&gate_v, b->gath, b->n_embd, b->ffn_a, b->ffw, m, b);
        mm_batch(&up_v,   b->gath, b->n_embd, b->ffn_b, b->ffw, m, b);
        for (uint32_t t = 0; t < m; t++) {
            oc_swiglu_inplace_f32(b->ffn_a + (size_t)t * b->ffw,
                                  b->ffn_b + (size_t)t * b->ffw, i_size);
        }
        /* down → gath (free again now that gate/up have consumed it). */
        mm_batch(&down_v, b->ffn_a, b->ffw, b->gath, b->n_embd, m, b);
        g_pf_t.expert_mm += pf_now() - t_m0;
        double t_s0 = pf_now();
        for (uint32_t t = 0; t < m; t++) {
            const float *src = b->gath + (size_t)t * b->n_embd;
            float *dst = b->expert_out + (size_t)toks[t] * b->n_embd;
            const float w = ws[t];
            for (size_t i = 0; i < b->n_embd; i++) dst[i] += w * src[i];
        }
        g_pf_t.scatter += pf_now() - t_s0;
    }

    /* 7. Shared expert (always active), batched over every token. */
    if (L->ffn_gate_shexp.data != NULL && L->ffn_up_shexp.data != NULL &&
        L->ffn_down_shexp.data != NULL) {
        mm_batch(&L->ffn_gate_shexp, b->normed, b->n_embd, b->ffn_a, b->ffw,
                 n, b);
        mm_batch(&L->ffn_up_shexp,   b->normed, b->n_embd, b->ffn_b, b->ffw,
                 n, b);
        for (size_t j = 0; j < n; j++) {
            oc_swiglu_inplace_f32(b->ffn_a + j * b->ffw,
                                  b->ffn_b + j * b->ffw, i_size);
        }
        mm_batch(&L->ffn_down_shexp, b->ffn_a, b->ffw, b->gath, b->n_embd,
                 n, b);
        if (L->ffn_gate_inp_shexp.data != NULL) {
            /* One logit per token; rows == 1, so this is a cheap batched dot. */
            mm_batch(&L->ffn_gate_inp_shexp, b->normed, b->n_embd,
                     b->ffn_a, 1, n, b);
            for (size_t j = 0; j < n; j++) {
                const float sc = 1.0f / (1.0f + expf(-b->ffn_a[j]));
                float *g = b->gath + j * b->n_embd;
                for (size_t i = 0; i < b->n_embd; i++) g[i] *= sc;
            }
        }
        for (size_t j = 0; j < n; j++) {
            const float *src = b->gath + j * b->n_embd;
            float *dst = b->expert_out + j * b->n_embd;
            for (size_t i = 0; i < b->n_embd; i++) dst[i] += src[i];
        }
    }

    /* 8. Residual add. */
    for (size_t j = 0; j < n; j++) {
        const float *src = b->expert_out + j * b->n_embd;
        float *dst = b->x + j * b->n_embd;
        for (size_t i = 0; i < b->n_embd; i++) dst[i] += src[i];
    }
}

static void prefill_dense_ffn(OcLlamaSession *s, const OcLlamaLayer *L,
                              PrefillBuf *b, size_t n)
{
    const OcLlamaConfig *c = &s->model->cfg;
    mm_batch(&L->ffn_gate, b->normed, b->n_embd, b->ffn_a, b->ffw, n, b);
    mm_batch(&L->ffn_up,   b->normed, b->n_embd, b->ffn_b, b->ffw, n, b);
    for (size_t j = 0; j < n; j++) {
        if (c->uses_geglu) {
            oc_geglu_inplace_f32(b->ffn_a + j * b->ffw,
                                 b->ffn_b + j * b->ffw, c->n_ff);
        } else {
            oc_swiglu_inplace_f32(b->ffn_a + j * b->ffw,
                                  b->ffn_b + j * b->ffw, c->n_ff);
        }
    }
    mm_batch(&L->ffn_down, b->ffn_a, b->ffw, b->proj, b->n_embd, n, b);
    for (size_t j = 0; j < n; j++) {
        float *p = b->proj + j * b->n_embd;
        if (L->post_ffw_norm != NULL) {
            oc_rms_norm_f32(p, L->post_ffw_norm, b->gath, c->n_embd,
                            post_norm_eps(c));
            memcpy(p, b->gath, c->n_embd * sizeof(float));
        }
        float *x = b->x + j * b->n_embd;
        for (size_t i = 0; i < c->n_embd; i++) x[i] += p[i];
    }
}

/* Parallel region body for batched attention: index i encodes the (token,
 * head) pair as token = i / n_head, head = i % n_head. */
typedef struct {
    OcLlamaSession *s;
    PrefillBuf     *b;
    uint32_t        layer;
    int64_t         pos0;
    size_t          hd;
    uint32_t        n_head;
    /* qwen35 only: the fused attn_q output, whose odd half per head is the
     * sigmoid output gate. NULL on every other architecture. */
    const float    *qgate;
    /* Muse Glimmer: a separate n_qo-wide gate per token, laid out exactly
     * like `q`. NULL on every other architecture. */
    const float    *mgate;
    _Atomic int     error;
} AttnJob;

static void attention_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    AttnJob *j = (AttnJob *)ud;
    const OcLlamaLayer *GL = layer_for_attn(j->s, j->layer);
    const int compressed = use_compressed_attn(j->s, j->layer);
    uint32_t n_kv = 0, group = 1;
    if (compressed) {
        n_kv = GL->n_head_kv ? GL->n_head_kv : j->s->model->cfg.n_head_kv;
        if (n_kv == 0) n_kv = 1;
        group = j->n_head / n_kv;
        if (group == 0) group = 1;
    }
    for (size_t i = begin; i < end; i++) {
        const size_t tok = i / j->n_head;
        const uint32_t h = (uint32_t)(i % j->n_head);
        float *out = j->b->attn_out + tok * j->b->n_qo + h * j->hd;
        if (compressed) {
            uint32_t kv_head = h / group;
            OcError e = oc_compressed_kv_attention(
                j->s->kv_compress, j->layer, kv_head,
                j->b->q + tok * j->b->n_qo + h * j->hd, j->hd,
                (size_t)(j->pos0 + (int64_t)tok), out);
            if (e != OC_OK) {
                int expected = OC_OK;
                atomic_compare_exchange_strong(&j->error, &expected, (int)e);
                continue;
            }
        } else {
            attention_head_at(j->s, h, j->layer, j->pos0 + (int64_t)tok,
                              j->b->q + tok * j->b->n_qo + h * j->hd, out);
        }
        if (j->qgate != NULL) {
            const float *gate = j->qgate + tok * 2u * (size_t)j->n_head * j->hd
                              + (2u * (size_t)h + 1u) * j->hd;
            for (size_t d = 0; d < j->hd; d++)
                out[d] *= qwen35_sigmoid(gate[d]);
        } else if (j->mgate != NULL) {
            const float *gate = j->mgate + tok * j->b->n_qo + h * j->hd;
            for (size_t d = 0; d < j->hd; d++)
                out[d] *= qwen35_sigmoid(gate[d]);
        }
    }
}

/* Per-token half of a qwen35 full-attention layer: split Q out of the fused
 * attn_q output, per-head QK RMSNorm, partial RoPE, then the KV cache store. */
typedef struct {
    OcLlamaSession     *s;
    PrefillBuf         *b;
    const OcLlamaLayer *L;
    int64_t             pos0;
    size_t              hd, kvdim;
    uint32_t            n_head, n_head_kv;
    uint32_t            kv_scale_stride;   /* cfg.n_head_kv: kv scale row */
    size_t              qdim;
    float               eps;
    uint32_t            n_ctx;
} Qwen35QkJob;

static void qwen35_qk_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    const Qwen35QkJob *j = (const Qwen35QkJob *)ud;
    OcLlamaSession *s = j->s;
    PrefillBuf *b = j->b;
    const OcLlamaLayer *L = j->L;
    const size_t hd = j->hd;
    float *tmp = (float *)oc_parallel_scratch(tid, hd * sizeof(float));
    if (tmp == NULL) return;

    for (size_t t = begin; t < end; t++) {
        const float *packed = b->q35_qgate + t * 2u * j->qdim;
        float *q = b->q + t * b->n_qo;
        float *k = b->k_buf + t * b->kv_row;
        const int64_t pos = j->pos0 + (int64_t)t;
        for (uint32_t h = 0; h < j->n_head; h++) {
            float *qh = q + (size_t)h * hd;
            memcpy(qh, packed + 2u * (size_t)h * hd, hd * sizeof(float));
            oc_rms_norm_f32(qh, L->attn_q_norm, tmp, hd, j->eps);
            memcpy(qh, tmp, hd * sizeof(float));
            oc_apply_rope_f32(qh, qh, hd, L->rope_dim, pos, L->rope_theta);
        }
        for (uint32_t h = 0; h < j->n_head_kv; h++) {
            float *kh = k + (size_t)h * hd;
            oc_rms_norm_f32(kh, L->attn_k_norm, tmp, hd, j->eps);
            memcpy(kh, tmp, hd * sizeof(float));
            oc_apply_rope_f32(kh, kh, hd, L->rope_dim, pos, L->rope_theta);
        }
        const size_t kv_off = ((size_t)L->kv_cache_index * j->n_ctx +
                               (size_t)pos) * s->kv_row_floats;
        const float *v = b->v_buf + t * b->kv_row;
        if (s->kv_type == OC_KV_Q8) {
            const size_t sc_off = ((size_t)L->kv_cache_index * j->n_ctx +
                                   (size_t)pos) * j->kv_scale_stride;
            for (uint32_t h = 0; h < j->n_head_kv; h++) {
                kv_q8_encode(k + (size_t)h * hd, s->kv_k_q + kv_off +
                             (size_t)h * hd, &s->kv_k_scale[sc_off + h], hd);
                kv_q8_encode(v + (size_t)h * hd, s->kv_v_q + kv_off +
                             (size_t)h * hd, &s->kv_v_scale[sc_off + h], hd);
            }
        } else {
            memcpy(s->kv_k + kv_off, k, j->kvdim * sizeof(float));
            memcpy(s->kv_v + kv_off, v, j->kvdim * sizeof(float));
        }
    }
}

static void prefill_qwen35_ffn(OcLlamaSession *s, const OcLlamaLayer *L,
                               PrefillBuf *b, size_t n)
{
    const OcLlamaConfig *c = &s->model->cfg;
    for (size_t j = 0; j < n; j++)
        oc_rms_norm_f32(b->x + j * b->n_embd, L->post_attention_norm,
                        b->normed + j * b->n_embd, c->n_embd,
                        c->rms_norm_eps);
    /* Dense when the layer has no router — Qwen3.6-27B (arch qwen35) ships a
     * plain SwiGLU FFN on every block, MoE variants (qwen35moe) a routed one. */
    if (c->num_experts > 0 && L->ffn_gate_inp.data != NULL &&
        L->ffn_gate_exps.data != NULL) {
        prefill_moe_ffn(s, L, b, n);
    } else {
        prefill_dense_ffn(s, L, b, n);
    }
}

static OcError prefill_qwen35_recurrent(OcLlamaSession *s, uint32_t layer,
                                        PrefillBuf *b, size_t n)
{
    const OcLlamaConfig *c = &s->model->cfg;
    const OcLlamaLayer *L = &s->model->layers[layer];
    const size_t key_dim = (size_t)c->ssm_group_count * c->ssm_state_size;
    const size_t conv_dim = 2u * key_dim + c->ssm_inner_size;

    double t_n0 = pf_now();
    for (size_t j = 0; j < n; j++)
        oc_rms_norm_f32(b->x + j * b->n_embd, L->attn_norm,
                        b->normed + j * b->n_embd, c->n_embd,
                        c->rms_norm_eps);
    g_pf_t.norm += pf_now() - t_n0;

    double t_q0 = pf_now();
    mm_batch(&L->attn_qkv, b->normed, b->n_embd, b->q35_rqkv, conv_dim, n, b);
    mm_batch(&L->attn_gate, b->normed, b->n_embd, b->q35_rgate,
             c->ssm_inner_size, n, b);
    mm_batch(&L->ssm_beta, b->normed, b->n_embd, b->q35_rbeta,
             c->ssm_value_heads, n, b);
    mm_batch(&L->ssm_alpha, b->normed, b->n_embd, b->q35_ralpha,
             c->ssm_value_heads, n, b);
    g_pf_t.qkv += pf_now() - t_q0;

    const OcQwen35DeltaParams params = {
        .conv_weight = (const float *)L->ssm_conv1d.data,
        .conv_weight_len = L->ssm_conv1d.rows * L->ssm_conv1d.cols,
        .ssm_a = (const float *)L->ssm_a.data,
        .ssm_a_len = L->ssm_a.rows * L->ssm_a.cols,
        .dt_bias = (const float *)L->ssm_dt_bias.data,
        .dt_bias_len = L->ssm_dt_bias.rows * L->ssm_dt_bias.cols,
        .norm_weight = (const float *)L->ssm_norm.data,
        .norm_weight_len = L->ssm_norm.rows * L->ssm_norm.cols,
        .norm_eps = c->rms_norm_eps,
    };
    double t_d0 = pf_now();
    for (size_t j = 0; j < n; j++) {
        const OcQwen35DeltaInput input = {
            .qkv = b->q35_rqkv + j * conv_dim,
            .qkv_len = conv_dim,
            .gate = b->q35_rgate + j * c->ssm_inner_size,
            .gate_len = c->ssm_inner_size,
            .beta = b->q35_rbeta + j * c->ssm_value_heads,
            .beta_len = c->ssm_value_heads,
            .alpha = b->q35_ralpha + j * c->ssm_value_heads,
            .alpha_len = c->ssm_value_heads,
        };
        OcError e = oc_qwen35_delta_step(
            &s->qwen35_delta[layer], &params, &input,
            s->qwen35_conv_output, conv_dim,
            b->q35_delta + j * c->ssm_inner_size, c->ssm_inner_size);
        if (e != OC_OK) return e;
    }
    g_pf_t.attn += pf_now() - t_d0;
    double t_p0 = pf_now();
    mm_batch(&L->ssm_out, b->q35_delta, c->ssm_inner_size,
             b->proj, b->n_embd, n, b);
    for (size_t j = 0; j < n; j++) {
        float *x = b->x + j * b->n_embd;
        const float *p = b->proj + j * b->n_embd;
        for (size_t i = 0; i < b->n_embd; i++) x[i] += p[i];
    }
    g_pf_t.proj += pf_now() - t_p0;
    double t_f0 = pf_now();
    prefill_qwen35_ffn(s, L, b, n);
    g_pf_t.expert_mm += pf_now() - t_f0;
    return OC_OK;
}

static void prefill_qwen35_attention(OcLlamaSession *s, uint32_t layer,
                                     PrefillBuf *b, size_t n, int64_t pos0)
{
    const OcLlamaConfig *c = &s->model->cfg;
    const OcLlamaLayer *L = &s->model->layers[layer];
    const size_t hd = L->head_dim;
    const size_t qdim = (size_t)c->n_head * hd;
    const size_t kvdim = (size_t)L->n_head_kv * hd;

    for (size_t j = 0; j < n; j++)
        oc_rms_norm_f32(b->x + j * b->n_embd, L->attn_norm,
                        b->normed + j * b->n_embd, c->n_embd,
                        c->rms_norm_eps);
    mm_batch(&L->attn_q, b->normed, b->n_embd, b->q35_qgate,
             2u * qdim, n, b);
    mm_batch(&L->attn_k, b->normed, b->n_embd, b->k_buf, b->kv_row, n, b);
    mm_batch(&L->attn_v, b->normed, b->n_embd, b->v_buf, b->kv_row, n, b);

    /* Per-token QK-norm + RoPE + KV store. Every token is independent, and at
     * a 512-token chunk this is 512*24 head-sized RMSNorms — enough that
     * running it on one thread showed up as half the cost of the whole
     * full-attention layer. Each worker needs its own head-sized scratch, so
     * reserve it up front (a failure inside the region could not be reported)
     * and fall back to serial if the pool cannot provide it. */
    Qwen35QkJob qjob = { s, b, L, pos0, hd, kvdim, c->n_head, L->n_head_kv,
                         c->n_head_kv, (size_t)qdim, c->rms_norm_eps,
                         c->n_ctx };
    const size_t qk_scratch = hd * sizeof(float);
    bool qk_serial = false;
    const size_t nt = oc_parallel_n_threads();
    for (size_t t = 0; t < nt; t++) {
        if (oc_parallel_scratch(t, qk_scratch) == NULL) { qk_serial = true; break; }
    }
    if (qk_serial) qwen35_qk_slice(0, n, 0, &qjob);
    else           oc_parallel_for(n, qwen35_qk_slice, &qjob);

    /* attention_slice applies the sigmoid output gate per head, so the gating
     * rides along on the same parallel region instead of a serial sweep. */
    AttnJob ajob = { s, b, layer, pos0, hd, c->n_head, b->q35_qgate, NULL, OC_OK };
    oc_parallel_for(n * c->n_head, attention_slice, &ajob);
    mm_batch(&L->attn_output, b->attn_out, b->n_qo,
             b->proj, b->n_embd, n, b);
    for (size_t j = 0; j < n; j++) {
        float *x = b->x + j * b->n_embd;
        const float *p = b->proj + j * b->n_embd;
        for (size_t i = 0; i < b->n_embd; i++) x[i] += p[i];
    }
    prefill_qwen35_ffn(s, L, b, n);
}

static OcError prefill_layer(OcLlamaSession *s, uint32_t layer, PrefillBuf *b,
                             size_t n, int64_t pos0)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    const size_t hd = L->head_dim ? (size_t)L->head_dim : (size_t)c->head_dim;
    const uint32_t n_kv = L->n_head_kv ? L->n_head_kv : c->n_head_kv;
    const uint32_t rope_dim = L->head_dim ? L->rope_dim : c->rope_dim;
    const float rope_theta = L->head_dim ? L->rope_theta : c->rope_theta;

    if (c->is_qwen35) {
        if (L->kind == OC_LLAMA_LAYER_QWEN35_RECURRENT)
            return prefill_qwen35_recurrent(s, layer, b, n);
        double t_a0 = pf_now();
        prefill_qwen35_attention(s, layer, b, n, pos0);
        g_pf_t.rope_kv += pf_now() - t_a0;
        return OC_OK;
    }

    /* Pre-attention RMSNorm. */
    for (size_t j = 0; j < n; j++) {
        oc_rms_norm_f32(b->x + j * b->n_embd, L->attn_norm,
                        b->normed + j * b->n_embd, c->n_embd, c->rms_norm_eps);
        if (!c->uses_gemma4 && c->norm_scale != 1.0f) {
            float *nm = b->normed + j * b->n_embd;
            for (size_t i = 0; i < c->n_embd; i++) nm[i] *= c->norm_scale;
        }
    }

    /* Q/K/V for the whole chunk. */
    double t_qkv0 = pf_now();
    mm_batch(&L->attn_q, b->normed, b->n_embd, b->q,     b->n_qo,   n, b);
    mm_batch(&L->attn_k, b->normed, b->n_embd, b->k_buf, b->kv_row, n, b);
    mm_batch(&L->attn_v, b->normed, b->n_embd, b->v_buf, b->kv_row, n, b);
    /* Muse Glimmer's attention-output gate is a fourth projection of the same
     * normed input; attention_slice applies the sigmoid per head. */
    if (c->attn_out_gate && L->attn_gate.data != NULL) {
        mm_batch(&L->attn_gate, b->normed, b->n_embd, b->mgate, b->n_qo, n, b);
    }
    g_pf_t.qkv += pf_now() - t_qkv0;
    double t_rk0 = pf_now();

    /* Per-token: biases, Q/K norms, RoPE, KV cache write. */
    for (size_t j = 0; j < n; j++) {
        float *qj = b->q + j * b->n_qo;
        float *kj = b->k_buf + j * b->kv_row;
        float *vj = b->v_buf + j * b->kv_row;
        const int64_t pos = pos0 + (int64_t)j;

        if (L->attn_q_bias != NULL) {
            const size_t nq = (size_t)c->n_head * hd;
            for (size_t i = 0; i < nq; i++) qj[i] += L->attn_q_bias[i];
        }
        if (L->attn_k_bias != NULL)
            for (size_t i = 0; i < b->kv_row; i++) kj[i] += L->attn_k_bias[i];
        if (L->attn_v_bias != NULL)
            for (size_t i = 0; i < b->kv_row; i++) vj[i] += L->attn_v_bias[i];

        if (L->attn_q_norm != NULL) {
            for (uint32_t h = 0; h < c->n_head; h++) {
                oc_rms_norm_f32(qj + h * hd, L->attn_q_norm, b->dequant_temp,
                                hd, c->rms_norm_eps);
                memcpy(qj + h * hd, b->dequant_temp, hd * sizeof(float));
            }
        }
        if (L->attn_k_norm != NULL) {
            for (uint32_t h = 0; h < n_kv; h++) {
                oc_rms_norm_f32(kj + h * hd, L->attn_k_norm, b->dequant_temp,
                                hd, c->rms_norm_eps);
                memcpy(kj + h * hd, b->dequant_temp, hd * sizeof(float));
            }
        }
        if (c->v_rms_norm) {
            for (uint32_t h = 0; h < n_kv; h++) {
                float *vh = vj + (size_t)h * hd;
                double ss = 0.0;
                for (size_t i = 0; i < hd; i++) ss += (double)vh[i] * vh[i];
                const float inv =
                    1.0f / sqrtf((float)(ss / (double)hd) + c->rms_norm_eps);
                for (size_t i = 0; i < hd; i++) vh[i] *= inv;
            }
        }

        if (use_compressed_attn(s, layer)) {
            OcError se = store_compressed_token(s, layer, n_kv, hd, kj, vj,
                                                (size_t)pos);
            if (se != OC_OK) {
                rewind_compressed_to(s, (size_t)pos0);
                return se;
            }
            continue;
        }

        if (L->use_rope) {
        for (uint32_t h = 0; h < c->n_head; h++) {
            if (c->yarn_factor > 0.0f)
                oc_apply_rope_yarn_f32(qj + h * hd, qj + h * hd, hd, rope_dim,
                                       pos, rope_theta, c->yarn_factor,
                                       c->yarn_orig_ctx);
            else if (c->rope_norm_pairs)
                oc_apply_rope_norm_f32(qj + h * hd, qj + h * hd, hd, rope_dim,
                                       pos, rope_theta);
            else
                oc_apply_rope_f32(qj + h * hd, qj + h * hd, hd, rope_dim, pos,
                                  rope_theta);
        }
        for (uint32_t h = 0; h < n_kv; h++) {
            if (c->yarn_factor > 0.0f)
                oc_apply_rope_yarn_f32(kj + h * hd, kj + h * hd, hd, rope_dim,
                                       pos, rope_theta, c->yarn_factor,
                                       c->yarn_orig_ctx);
            else if (c->rope_norm_pairs)
                oc_apply_rope_norm_f32(kj + h * hd, kj + h * hd, hd, rope_dim,
                                       pos, rope_theta);
            else
                oc_apply_rope_f32(kj + h * hd, kj + h * hd, hd, rope_dim, pos,
                                  rope_theta);
        }
        }

        /* Every token's K/V must land in the cache before ANY token attends:
         * token j attends over 0..j, which includes tokens later in this
         * same chunk's prefix. */
        const size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)pos)
                            * s->kv_row_floats;
        if (s->kv_type == OC_KV_Q8) {
            const size_t g = s->kv_group;
            const size_t sc_off = ((size_t)layer * c->n_ctx + (size_t)pos)
                                * c->n_head_kv;
            for (uint32_t h = 0; h < c->n_head_kv; h++) {
                kv_q8_encode(kj + h * g, s->kv_k_q + kv_off + h * g,
                             &s->kv_k_scale[sc_off + h], g);
                kv_q8_encode(vj + h * g, s->kv_v_q + kv_off + h * g,
                             &s->kv_v_scale[sc_off + h], g);
            }
        } else {
            memcpy(s->kv_k + kv_off, kj, s->kv_row_floats * sizeof(float));
            memcpy(s->kv_v + kv_off, vj, s->kv_row_floats * sizeof(float));
        }
    }

    g_pf_t.rope_kv += pf_now() - t_rk0;

    /* Attention over the whole chunk, parallel across (token, head).
     *
     * This has to be threaded, not just looped: prefill attention is
     * quadratic in the chunk length, and every (token, head) pair sweeps its
     * own slice of the KV cache. Left serial it dominates everything the
     * batched matmuls just saved — for a 500-token chunk it is ~200 GB of
     * cache reads on one core. Each pair writes only its own head slice of
     * attn_out, so there is nothing to synchronize. */
    double t0 = pf_now();
    AttnJob ajob = { s, b, layer, pos0, hd, c->n_head, NULL,
                     c->attn_out_gate ? b->mgate : NULL, OC_OK };
    oc_parallel_for(n * c->n_head, attention_slice, &ajob);
    g_pf_t.attn += pf_now() - t0;
    if (atomic_load(&ajob.error) != OC_OK) {
        OcError ae = (OcError)atomic_load(&ajob.error);
        if (use_compressed_attn(s, layer))
            rewind_compressed_to(s, (size_t)pos0);
        return ae;
    }

    /* Output projection + residual. */
    double t_pr0 = pf_now();
    mm_batch(&L->attn_output, b->attn_out, b->n_qo, b->proj, b->n_embd, n, b);
    g_pf_t.proj += pf_now() - t_pr0;
    for (size_t j = 0; j < n; j++) {
        float *p = b->proj + j * b->n_embd;
        if (L->post_attention_norm != NULL) {
            oc_rms_norm_f32(p, L->post_attention_norm, b->gath, c->n_embd,
                            post_norm_eps(c));
            memcpy(p, b->gath, c->n_embd * sizeof(float));
        }
        float *x = b->x + j * b->n_embd;
        for (size_t i = 0; i < c->n_embd; i++) x[i] += p[i];
    }

    /* Pre-FFN RMSNorm. */
    for (size_t j = 0; j < n; j++) {
        oc_rms_norm_f32(b->x + j * b->n_embd, L->ffn_norm,
                        b->normed + j * b->n_embd, c->n_embd, c->rms_norm_eps);
        if (!c->uses_gemma4 && c->norm_scale != 1.0f) {
            float *nm = b->normed + j * b->n_embd;
            for (size_t i = 0; i < c->n_embd; i++) nm[i] *= c->norm_scale;
        }
    }

    if (c->num_experts > 0 && L->ffn_gate_inp.data != NULL &&
        L->ffn_gate_exps.data != NULL) {
        prefill_moe_ffn(s, L, b, n);
    } else {
        prefill_dense_ffn(s, L, b, n);
    }

    if (L->layer_output_scale != 0.0f && L->layer_output_scale != 1.0f) {
        const float os = L->layer_output_scale;
        for (size_t j = 0; j < n; j++) {
            float *x = b->x + j * b->n_embd;
            for (size_t i = 0; i < c->n_embd; i++) x[i] *= os;
        }
    }
    return OC_OK;
}

/* Whether the batched path covers this model. Everything it does not cover
 * falls back to the per-token loop, which is unchanged. MLA, Gemma 4's dual
 * geometry (per-layer head_dim / k_eq_v aliasing) and the LayerNorm archs
 * with their own forward passes each need their own batched form; none of
 * them is the MoE prefill case this targets. */
static bool prefill_batch_supported(const OcLlamaModel *m)
{
    if (m->arch == OC_ARCH_GPT2 || m->arch == OC_ARCH_GPTJ ||
        m->arch == OC_ARCH_GPTNEOX ||
        m->arch == OC_ARCH_FALCON) return false;
    if (m->cfg.uses_mla || m->cfg.is_longcat) return false;
    if (m->cfg.uses_gemma4) return false;
    return true;
}

OcError oc_llama_prefill(OcLlamaSession *sess, const uint32_t *tokens,
                         size_t n_tokens, size_t chunk, float *logits_out)
{
    if (sess == NULL || sess->model == NULL || tokens == NULL)
        return OC_ERR_INVALID_ARG;
    if (n_tokens == 0) return OC_OK;

    OcLlamaModel *m = sess->model;
    if ((uint64_t)sess->pos + n_tokens > m->cfg.n_ctx) return OC_ERR_INVALID_ARG;

    /* Fall back to the per-token path when batching cannot help or is not
     * supported: identical results, just the old speed. */
    if (!prefill_batch_supported(m) || n_tokens < 2) {
        for (size_t i = 0; i < n_tokens; i++) {
            float *lg = (i + 1 == n_tokens) ? logits_out : NULL;
            OcError e = oc_llama_forward(sess, tokens[i], lg);
            if (e != OC_OK) return e;
        }
        return OC_OK;
    }

    if (chunk == 0) chunk = OC_PREFILL_CHUNK;
    if (chunk > n_tokens) chunk = n_tokens;
    memset(&g_pf_t, 0, sizeof(g_pf_t));

    PrefillBuf buf;
    OcError e = prefill_buf_init(m, chunk, &buf);
    if (e != OC_OK) {
        /* Out of scratch — the per-token path needs none, so use it. */
        for (size_t i = 0; i < n_tokens; i++) {
            float *lg = (i + 1 == n_tokens) ? logits_out : NULL;
            OcError e2 = oc_llama_forward(sess, tokens[i], lg);
            if (e2 != OC_OK) return e2;
        }
        return OC_OK;
    }

    for (size_t base = 0; base < n_tokens; base += chunk) {
        const size_t n = (n_tokens - base < chunk) ? (n_tokens - base) : chunk;
        const int64_t pos0 = sess->pos;

        for (size_t j = 0; j < n; j++)
            embed_token_into(sess, tokens[base + j], buf.x + j * buf.n_embd);

        for (uint32_t l = 0; l < m->cfg.n_layer; l++) {
            e = prefill_layer(sess, l, &buf, n, pos0);
            if (e != OC_OK) {
                prefill_buf_free(&buf);
                return e;
            }
        }

        sess->pos = pos0 + (int64_t)n;

        /* Logits only for the very last token of the prompt: the lm_head is a
         * vocab_size × n_embd matmul and nothing reads the others. */
        const bool last_chunk = (base + n == n_tokens);
        if (last_chunk) {
            const float *xl = buf.x + (n - 1) * buf.n_embd;
            if (sess->last_hidden != NULL)
                memcpy(sess->last_hidden, xl, m->cfg.n_embd * sizeof(float));
            sess->last_token = tokens[n_tokens - 1];
            if (logits_out != NULL) {
            oc_rms_norm_f32(xl, m->final_norm, sess->normed, m->cfg.n_embd,
                            m->cfg.rms_norm_eps);
            if (!m->cfg.uses_gemma4 && m->cfg.norm_scale != 1.0f) {
                for (size_t i = 0; i < m->cfg.n_embd; i++)
                    sess->normed[i] *= m->cfg.norm_scale;
            }
            sess->last_token = tokens[n_tokens - 1];
            if (m->output.qtype == OC_QUANT_F32) {
                oc_matvec_f32((const float *)m->output.data, m->output.rows,
                              m->output.cols, sess->normed, logits_out);
            } else {
                oc_matvec_quantized(m->output.qtype, m->output.data,
                                    m->output.rows, m->output.cols,
                                    m->output.row_bytes, sess->normed,
                                    logits_out, sess->dequant_temp);
            }
            /* Mirrors the per-token path: scale, then softcap. */
            if (m->cfg.logit_scale > 0.0f && m->cfg.logit_scale != 1.0f) {
                const float ls = m->cfg.logit_scale;
                for (size_t i = 0; i < m->cfg.vocab_size; i++)
                    logits_out[i] *= ls;
            }
            if (m->cfg.logit_softcap > 0.0f) {
                const float cap = m->cfg.logit_softcap;
                const float inv = 1.0f / cap;
                for (size_t i = 0; i < m->cfg.vocab_size; i++)
                    logits_out[i] = tanhf(logits_out[i] * inv) * cap;
            }
            }
        }
    }

    prefill_buf_free(&buf);
    oc_log(OC_LOG_DEBUG,
           "prefill phases (s): norm=%.2f qkv=%.2f rope_kv=%.2f attn=%.2f "
           "proj=%.2f router=%.2f gather=%.2f expert_mm=%.2f scatter=%.2f",
           g_pf_t.norm, g_pf_t.qkv, g_pf_t.rope_kv, g_pf_t.attn, g_pf_t.proj,
           g_pf_t.router, g_pf_t.gather, g_pf_t.expert_mm, g_pf_t.scatter);
    return OC_OK;
}

OcError oc_llama_session_copy_prefix(OcLlamaSession *dst,
                                     const OcLlamaSession *src)
{
    if (dst == NULL || src == NULL || dst->model == NULL ||
        dst->model != src->model || dst->kv_type != src->kv_type ||
        src->pos < 0 || (uint64_t)src->pos > src->model->cfg.n_ctx ||
        dst->kv_row_floats != src->kv_row_floats)
        return OC_ERR_INVALID_ARG;
    if (dst->kv_compress || src->kv_compress)
        return OC_ERR_INVALID_ARG;

    const OcLlamaConfig *c = &src->model->cfg;
    const size_t cache_layers = c->is_qwen35
                              ? c->n_full_attention_layers : c->n_layer;
    size_t rows, elems;
    if (!size_mul(cache_layers, (size_t)src->pos, &rows) ||
        !size_mul(rows, src->kv_row_floats, &elems))
        return OC_ERR_MODEL;
    if (src->kv_type == OC_KV_Q8) {
        memcpy(dst->kv_k_q, src->kv_k_q, elems * sizeof(*src->kv_k_q));
        memcpy(dst->kv_v_q, src->kv_v_q, elems * sizeof(*src->kv_v_q));
        size_t groups;
        if (!size_mul(rows, c->n_head_kv, &groups)) return OC_ERR_MODEL;
        memcpy(dst->kv_k_scale, src->kv_k_scale,
               groups * sizeof(*src->kv_k_scale));
        memcpy(dst->kv_v_scale, src->kv_v_scale,
               groups * sizeof(*src->kv_v_scale));
    } else {
        memcpy(dst->kv_k, src->kv_k, elems * sizeof(*src->kv_k));
        if (!c->uses_mla)
            memcpy(dst->kv_v, src->kv_v, elems * sizeof(*src->kv_v));
    }

    if (c->is_qwen35) {
        const size_t key_dim = (size_t)c->ssm_group_count * c->ssm_state_size;
        const size_t conv_dim = 2u * key_dim + c->ssm_inner_size;
        size_t conv_per_layer, recurrent_per_layer, conv_total, recurrent_total;
        if (!size_mul(conv_dim, c->ssm_conv_kernel - 1u, &conv_per_layer) ||
            !size_mul(c->ssm_inner_size, c->ssm_state_size,
                      &recurrent_per_layer) ||
            !size_mul(c->n_recurrent_layers, conv_per_layer, &conv_total) ||
            !size_mul(c->n_recurrent_layers, recurrent_per_layer,
                      &recurrent_total))
            return OC_ERR_MODEL;
        memcpy(dst->qwen35_conv_state, src->qwen35_conv_state,
               conv_total * sizeof(*src->qwen35_conv_state));
        memcpy(dst->qwen35_recurrent_state, src->qwen35_recurrent_state,
               recurrent_total * sizeof(*src->qwen35_recurrent_state));
    }
    memcpy(dst->logits, src->logits, c->vocab_size * sizeof(*src->logits));
    dst->pos = src->pos;
    dst->last_token = src->last_token;
    dst->mtp_pos = src->mtp_pos;
    if (dst->last_hidden != NULL && src->last_hidden != NULL)
        memcpy(dst->last_hidden, src->last_hidden,
               c->n_embd * sizeof(*src->last_hidden));
    if (dst->mtp_hidden != NULL && src->mtp_hidden != NULL)
        memcpy(dst->mtp_hidden, src->mtp_hidden,
               c->n_embd * sizeof(*src->mtp_hidden));
    return OC_OK;
}

/* ─── Batched decode ─────────────────────────────────────────────────── */

OcError oc_batch_session_init(OcLlamaModel *model, size_t max_seqs,
                               OcBatchSession *out)
{
    if (model == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    /* Batch decode only implements the RMSNorm Llama-family layer; reject
     * LayerNorm architectures (they use oc_llama_forward's dispatch). */
    if (model->arch == OC_ARCH_GPT2 || model->arch == OC_ARCH_GPTJ ||
        model->arch == OC_ARCH_GPTNEOX ||
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
        /* Must agree with kv_row_floats_for(): the latent, not the expanded
         * per-head K/V. Sizing this the old way allocated 12288 floats per
         * row per sequence -- about 76 GB for a 2-sequence 8k-context
         * LongCat batch, against the 3.6 GB the latent needs. */
        out->kv_row_floats = kv_row_floats_for(model);
    } else {
        out->kv_row_floats = (size_t)model->cfg.n_head_kv * model->cfg.kv_head_dim;
    }
    size_t per_layer, total;
    if (!size_mul(model->cfg.n_ctx, out->kv_row_floats, &per_layer) ||
        !size_mul(per_layer, max_seqs, &per_layer) ||
        !size_mul(model->cfg.n_layer, per_layer, &total))
        return OC_ERR_MODEL;
    out->kv_k = xcalloc(total, sizeof(float));
    /* MLA reconstructs V from the cached latent; see the session path. */
    if (!model->cfg.uses_mla)
        out->kv_v = xcalloc(total, sizeof(float));
    size_t maxw = model->cfg.n_embd > model->cfg.n_ff ? model->cfg.n_embd : model->cfg.n_ff;
    if (model->cfg.expert_intermediate_size > maxw)
        maxw = model->cfg.expert_intermediate_size;
    out->x = xcalloc(model->cfg.n_embd, sizeof(float));
    out->normed = xcalloc(model->cfg.n_embd, sizeof(float));
    out->q = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
    out->k = xcalloc(out->kv_row_floats, sizeof(float));
    out->v = xcalloc(out->kv_row_floats, sizeof(float));
    /* n_head*head_dim is what attention writes, but the Gemma-style sandwich
     * norms reuse this buffer as an n_embd-wide scratch for the branch
     * output. On a heavily-GQA model (Muse Glimmer: 32*128 = 4096 vs
     * n_embd 6656) that is the larger of the two, so size it for both. */
    out->attn_out = xcalloc(oc_max_sz((size_t)model->cfg.n_head *
                                      model->cfg.head_dim,
                                      (size_t)model->cfg.n_embd),
                            sizeof(float));
    out->ffn_gate = xcalloc(model->cfg.n_ff, sizeof(float));
    out->ffn_up = xcalloc(model->cfg.n_ff, sizeof(float));
    out->dequant_temp = xcalloc(maxw, sizeof(float));
    if (model->cfg.num_experts > 0) {
        /* The router spans routed AND zero-expert slots. Sizing this to
         * num_experts alone overruns it by zero_expert_count floats on every
         * MoE layer of every token -- 512 bytes per layer on LongCat-2.0. */
        out->router_logits = xcalloc((size_t)model->cfg.num_experts
                                     + model->cfg.zero_expert_count,
                                     sizeof(float));
        out->expert_gate = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_up = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_out = xcalloc(model->cfg.n_embd, sizeof(float));
        const size_t shexp_size = model->cfg.shared_expert_intermediate_size
                                    ? model->cfg.shared_expert_intermediate_size
                                    : model->cfg.expert_intermediate_size;
        out->shexp_gate = xcalloc(shexp_size, sizeof(float));
        out->shexp_up = xcalloc(shexp_size, sizeof(float));
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
        size_t mla_per_head = (size_t)model->cfg.n_head * model->cfg.mla_kv_lora_dim;
        out->mla_q_absorbed = xcalloc(mla_per_head, sizeof(float));
        out->mla_ctx_latent = xcalloc(mla_per_head, sizeof(float));
        out->mla_run_max = xcalloc(model->cfg.n_head, sizeof(float));
        out->mla_run_sum = xcalloc(model->cfg.n_head, sizeof(float));
        if (!out->mla_c_q || !out->mla_c_kv || !out->mla_q_full ||
            !out->mla_kv_compressed || !out->mla_q_absorbed ||
            !out->mla_ctx_latent || !out->mla_run_max || !out->mla_run_sum) {
            oc_batch_session_free(out);
            return OC_ERR_OOM;
        }
    }
    if (!out->kv_k || (!out->kv_v && !model->cfg.uses_mla) ||
        !out->x || !out->normed || !out->q ||
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
        /* Zeroed, not just field-by-field assigned: this struct has grown
         * several times (MLA scratch, Q8 KV code/scale pointers) and each
         * time the un-assigned tail became stack garbage that the forward
         * pass then wrote through. kv_type 0 is OC_KV_F32, which is what the
         * batch path uses. */
        OcLlamaSession tmp;
        memset(&tmp, 0, sizeof tmp);
        tmp.model = m;
        size_t sequence_stride;
        if (!size_mul(m->cfg.n_layer, m->cfg.n_ctx, &sequence_stride) ||
            !size_mul(sequence_stride, bs->kv_row_floats, &sequence_stride))
            return OC_ERR_MODEL;
        tmp.kv_k = bs->kv_k + s * sequence_stride;
        tmp.kv_v = bs->kv_v ? bs->kv_v + s * sequence_stride : NULL;
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
        tmp.mla_q_absorbed = bs->mla_q_absorbed;
        tmp.mla_ctx_latent = bs->mla_ctx_latent;
        tmp.mla_run_max = bs->mla_run_max;
        tmp.mla_run_sum = bs->mla_run_sum;

        /* Embed and forward. */
        embed_token(&tmp, seqs[s].token);
        for (uint32_t l = 0; l < m->cfg.n_layer; l++) {
            OcError error = forward_layer(&tmp, l);
            if (error != OC_OK) return error;
        }
        /* Final RMSNorm + lm_head. */
        oc_rms_norm_f32(tmp.x, m->final_norm, tmp.normed, m->cfg.n_embd,
                        m->cfg.rms_norm_eps);
        if (!m->cfg.uses_gemma4 && m->cfg.norm_scale != 1.0f) {
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
    free(bs->mla_q_absorbed); free(bs->mla_ctx_latent);
    free(bs->mla_run_max); free(bs->mla_run_sum);
    memset(bs, 0, sizeof(*bs));
}
