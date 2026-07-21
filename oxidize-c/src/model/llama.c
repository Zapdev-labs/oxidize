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

#include "oxidize/activation.h"
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
    cfg->uses_geglu = false;
    cfg->norm_scale = 1.0f;
    cfg->sliding_window = 0;
    cfg->sliding_window_pattern = 1;
    /* YaRN long-context scaling. */
    cfg->yarn_factor = 0.0f;
    cfg->yarn_orig_ctx = 0;
    if (arch_str) {
        if (strncmp(arch_str, "gemma", 5) == 0) {
            cfg->uses_geglu = true;
            cfg->norm_scale = sqrtf((float)cfg->n_embd);
            /* Gemma2 sliding window: alternating global/sliding layers. */
            snprintf(key, sizeof(key), "%sattention.sliding_window", prefix);
            uint32_t sw = cfg_u32(f, key, 0);
            if (sw > 0) {
                cfg->sliding_window = sw;
                cfg->sliding_window_pattern = 2; /* alternating */
            }
        }
    }
    /* YaRN: read from GGUF metadata if present. */
    snprintf(key, sizeof(key), "%srope.scaling.type", prefix);
    const char *scale_type = cfg_str(f, key, NULL);
    if (scale_type) {
        if (strcmp(scale_type, "yarn") == 0) {
            snprintf(key, sizeof(key), "%srope.scaling.factor", prefix);
            cfg->yarn_factor = cfg_f32(f, key, 1.0f);
            cfg->yarn_orig_ctx = cfg->n_ctx;
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
    if (arch_str && (strncmp(arch_str, "deepseek", 8) == 0)) {
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
    cfg->tied_embeddings = false;
    if (cfg->n_head == 0) cfg->n_head = 32;
    if (cfg->n_head_kv == 0) cfg->n_head_kv = cfg->n_head;
    /* MoE per-token top-k: default to 1 when experts exist but no top-k set. */
    if (cfg->num_experts > 0) {
        if (cfg->num_experts_per_tok == 0) cfg->num_experts_per_tok = 1;
        if (cfg->num_experts_per_tok > cfg->num_experts) {
            cfg->num_experts_per_tok = cfg->num_experts;
        }
        if (cfg->expert_intermediate_size == 0) cfg->expert_intermediate_size = cfg->n_ff;
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
    /* Per-layer: blk.<N>.<suffix> */
    if (strncmp(cname, "blk.", 4) == 0) {
        char *end = NULL;
        unsigned long layer_idx = strtoul(cname + 4, &end, 10);
        if (end == cname + 4 || *end != '.') return false;
        if (layer_idx >= m->cfg.n_layer) return false;
        OcLlamaLayer *L = &m->layers[layer_idx];
        const char *suf = end + 1;
        if (strcmp(suf, "attn_norm.weight") == 0) {
            L->attn_norm = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "ffn_norm.weight") == 0) {
            L->ffn_norm = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "attn_q.weight") == 0) {
            L->attn_q = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_k.weight") == 0) {
            L->attn_k = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_v.weight") == 0) {
            L->attn_v = view_from_info(mm, info);
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
    /* Infer vocab_size from tok_embeddings rows if still default. */
    if (m->tok_embeddings.rows > 0 && (m->cfg.vocab_size == 32000 ||
        m->cfg.vocab_size != m->tok_embeddings.rows)) {
        /* Trust the actual tensor shape over metadata default. */
    }
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

OcError oc_llama_session_init(OcLlamaModel *model, OcLlamaSession *out)
{
    if (model == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->model = model;
    /* For MLA, each head has its own K/V (no GQA sharing). */
    if (model->cfg.uses_mla) {
        out->kv_row_floats = (size_t)model->cfg.n_head * model->cfg.head_dim;
    } else {
        out->kv_row_floats = (size_t)model->cfg.n_head_kv * model->cfg.kv_head_dim;
    }
    size_t per_layer = (size_t)model->cfg.n_ctx * out->kv_row_floats;
    size_t total = (size_t)model->cfg.n_layer * per_layer;
    out->kv_k = xcalloc(total, sizeof(float));
    out->kv_v = xcalloc(total, sizeof(float));
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
    if (!out->kv_k || !out->kv_v || !out->x || !out->normed || !out->q ||
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
    out->kv_quantized = false;
    out->kv_k_q8 = NULL; out->kv_v_q8 = NULL;
    out->kv_k_scale = NULL; out->kv_v_scale = NULL;
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
    free(sess->kv_k_q8); free(sess->kv_v_q8);
    free(sess->kv_k_scale); free(sess->kv_v_scale);
    memset(sess, 0, sizeof(*sess));
}

void oc_llama_free(OcLlamaModel *model)
{
    if (model == NULL) return;
    if (model->layers) {
        for (uint32_t i = 0; i < model->cfg.n_layer; i++) {
            free(model->layers[i].attn_norm);
            free(model->layers[i].ffn_norm);
            free(model->layers[i].mla_q_a_norm);
            free(model->layers[i].mla_kv_a_norm);
        }
        free(model->layers);
    }
    free(model->final_norm);
    oc_gguf_map_free(&model->gguf);
    memset(model, 0, sizeof(*model));
}

/* ─── Forward pass ─────────────────────────────────────────────────────── */

/* ─── KV cache quantization (Q8 asymmetric, 32-element blocks) ─────────── */

#define KV_Q8_BLOCK_SIZE 32

/* Quantize a row of floats to Q8 (asymmetric, per-block f16 scale).
 * `src` has `n` floats. `dst` receives n int8 values. `scale` receives
 * one f16-scale per block (as float). */
static void kv_quantize_q8(const float *src, int8_t *dst, float *scale, size_t n)
{
    size_t n_blocks = (n + KV_Q8_BLOCK_SIZE - 1) / KV_Q8_BLOCK_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        size_t off = b * KV_Q8_BLOCK_SIZE;
        size_t len = (off + KV_Q8_BLOCK_SIZE <= n) ? KV_Q8_BLOCK_SIZE : (n - off);
        float amax = 0.0f;
        for (size_t i = 0; i < len; i++) {
            float a = fabsf(src[off + i]);
            if (a > amax) amax = a;
        }
        float s = amax / 127.0f;
        if (s == 0.0f) s = 1.0f;
        scale[b] = s;
        float inv = 1.0f / s;
        for (size_t i = 0; i < len; i++) {
            float v = src[off + i] * inv;
            int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            dst[off + i] = (int8_t)q;
        }
    }
}

/* Dequantize a Q8 row back to floats. */
static void kv_dequantize_q8(const int8_t *src, const float *scale,
                             float *dst, size_t n)
{
    size_t n_blocks = (n + KV_Q8_BLOCK_SIZE - 1) / KV_Q8_BLOCK_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        size_t off = b * KV_Q8_BLOCK_SIZE;
        size_t len = (off + KV_Q8_BLOCK_SIZE <= n) ? KV_Q8_BLOCK_SIZE : (n - off);
        float s = scale[b];
        for (size_t i = 0; i < len; i++) {
            dst[off + i] = (float)src[off + i] * s;
        }
    }
}

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
    size_t hd = c->head_dim;
    uint32_t group = c->n_head / c->n_head_kv;
    uint32_t kv_head = head / group;
    size_t kv_off = ((size_t)layer * c->n_ctx + 0) * s->kv_row_floats
                  + (size_t)kv_head * hd;
    const float *k_layer = s->kv_k;
    const float *v_layer = s->kv_v;
    float scale = 1.0f / sqrtf((float)hd);

    /* Online softmax (Milakov & Gimelshein 2018). seq_len = pos+1. */
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    for (size_t i = 0; i < hd; i++) out_vec[i] = 0.0f;

    /* Sliding window: only attend to the last `sliding_window` tokens
     * for layers matching the alternating pattern. Layer L uses sliding
     * window when (L % pattern) == 1 (i.e., every other layer for pattern=2). */
    int64_t seq_len = s->pos + 1;
    int64_t start = 0;
    if (c->sliding_window > 0 && c->sliding_window_pattern > 1) {
        if (layer % c->sliding_window_pattern == 1) {
            /* Sliding window layer: only attend to recent tokens. */
            int64_t sw = (int64_t)c->sliding_window;
            start = (seq_len > sw) ? (seq_len - sw) : 0;
        }
    }

    for (int64_t t = start; t < seq_len; t++) {
        const float *k_t = k_layer + kv_off + (size_t)t * s->kv_row_floats;
        float dot = 0.0f;
        for (size_t i = 0; i < hd; i++) dot += q_vec[i] * k_t[i];
        float score = dot * scale;
        float new_max = (score > run_max) ? score : run_max;
        float exp_factor = expf(run_max - new_max);
        float exp_score = expf(score - new_max);
        for (size_t i = 0; i < hd; i++) out_vec[i] *= exp_factor;
        const float *v_t = v_layer + kv_off + (size_t)t * s->kv_row_floats;
        for (size_t i = 0; i < hd; i++) out_vec[i] += exp_score * v_t[i];
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

    /* 6. RoPE on k_pe (the trailing q_rope slots of kv_compressed). */
    float *k_pe = s->mla_kv_compressed + kv_lora;
    oc_apply_rope_f32(k_pe, k_pe, q_rope, q_rope, s->pos, c->rope_theta);

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

    /* 8. RoPE on q_pe (the trailing q_rope slots of each q head). */
    for (uint32_t h = 0; h < n_head; h++) {
        float *q_pe = s->mla_q_full + h * q_hd + k_nope_hd;
        oc_apply_rope_f32(q_pe, q_pe, q_rope, q_rope, s->pos, c->rope_theta);
    }

    /* 9. Attention per head (MLA: each head has own K/V, group=1). */
    float scale = 1.0f / sqrtf((float)q_hd);
    for (uint32_t h = 0; h < n_head; h++) {
        float *q_vec = s->mla_q_full + h * q_hd;
        float *out_vec = s->attn_out + h * q_hd;
        const float *k_head = k_cache + (size_t)h * q_hd;
        const float *v_head = v_cache + (size_t)h * q_hd;

        float run_max = -INFINITY;
        float run_sum = 0.0f;
        for (size_t i = 0; i < q_hd; i++) out_vec[i] = 0.0f;

        int64_t seq_len = s->pos + 1;
        for (int64_t t = 0; t < seq_len; t++) {
            const float *k_t = k_head + (size_t)t * s->kv_row_floats;
            float dot = 0.0f;
            for (size_t i = 0; i < q_hd; i++) dot += q_vec[i] * k_t[i];
            float score = dot * scale;
            float new_max = (score > run_max) ? score : run_max;
            float exp_factor = expf(run_max - new_max);
            float exp_score = expf(score - new_max);
            for (size_t i = 0; i < q_hd; i++) out_vec[i] *= exp_factor;
            const float *v_t = v_head + (size_t)t * s->kv_row_floats;
            for (size_t i = 0; i < q_hd; i++) out_vec[i] += exp_score * v_t[i];
            run_sum = run_sum * exp_factor + exp_score;
            run_max = new_max;
        }
        float inv = 1.0f / run_sum;
        for (size_t i = 0; i < q_hd; i++) out_vec[i] *= inv;
    }

    /* 10. Output projection. */
    matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

static void forward_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;

    /* Pre-attention RMSNorm (+ Gemma scaling). */
    oc_rms_norm_f32(s->x, L->attn_norm, s->normed, c->n_embd, c->rms_norm_eps);
    if (c->norm_scale != 1.0f) {
        for (size_t i = 0; i < c->n_embd; i++) s->normed[i] *= c->norm_scale;
    }

    /* Attention: MLA or standard GQA. */
    if (c->uses_mla && L->mla_kv_a_mqa.data != NULL) {
        forward_mla_attention(s, layer);
    } else {
        /* Q/K/V projections. */
        matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
        matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
        matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

        /* RoPE on Q (per head) and K (per kv head). YaRN when configured. */
        for (uint32_t h = 0; h < c->n_head; h++) {
            if (c->yarn_factor > 0.0f) {
                oc_apply_rope_yarn_f32(s->q + h * hd, s->q + h * hd, hd,
                                       c->rope_dim, s->pos, c->rope_theta,
                                       c->yarn_factor, c->yarn_orig_ctx);
            } else {
                oc_apply_rope_f32(s->q + h * hd, s->q + h * hd, hd, c->rope_dim,
                                  s->pos, c->rope_theta);
            }
        }
        for (uint32_t h = 0; h < c->n_head_kv; h++) {
            if (c->yarn_factor > 0.0f) {
                oc_apply_rope_yarn_f32(s->k + h * hd, s->k + h * hd, hd,
                                       c->rope_dim, s->pos, c->rope_theta,
                                       c->yarn_factor, c->yarn_orig_ctx);
            } else {
                oc_apply_rope_f32(s->k + h * hd, s->k + h * hd, hd, c->rope_dim,
                                  s->pos, c->rope_theta);
            }
        }

        /* KV cache write at position `pos`. */
        size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos) * s->kv_row_floats;
        memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
        memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

        /* Attention per head → s->attn_out. */
        for (uint32_t h = 0; h < c->n_head; h++) {
            attention_head(s, h, layer, s->q + h * hd, s->attn_out + h * hd);
        }

        /* Output projection. */
        matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
        /* Residual add (reusing normed as the projection output buffer). */
        for (size_t i = 0; i < c->n_embd; i++) s->x[i] += s->normed[i];
    }

    /* Pre-FFN RMSNorm (+ Gemma scaling). */
    oc_rms_norm_f32(s->x, L->ffn_norm, s->normed, c->n_embd, c->rms_norm_eps);
    if (c->norm_scale != 1.0f) {
        for (size_t i = 0; i < c->n_embd; i++) s->normed[i] *= c->norm_scale;
    }

    /* FFN branch: MoE when the layer has a router, dense otherwise. */
    if (c->num_experts > 0 && L->ffn_gate_inp.data != NULL &&
        L->ffn_gate_exps.data != NULL) {
        forward_moe_ffn(s, L);
    } else {
        forward_dense_ffn(s, L);
    }
}

OcError oc_llama_forward(OcLlamaSession *sess, uint32_t token, float *logits_out)
{
    if (sess == NULL || sess->model == NULL) return OC_ERR_INVALID_ARG;
    if ((uint64_t)sess->pos >= sess->model->cfg.n_ctx) return OC_ERR_INVALID_ARG;

    embed_token(sess, token);
    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        forward_layer(sess, l);
    }
    /* Final RMSNorm + lm_head. */
    OcLlamaModel *m = sess->model;
    oc_rms_norm_f32(sess->x, m->final_norm, sess->normed, m->cfg.n_embd,
                    m->cfg.rms_norm_eps);
    if (m->cfg.norm_scale != 1.0f) {
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
    }
    sess->pos++;
    return OC_OK;
}

/* ─── Batched decode ─────────────────────────────────────────────────── */

OcError oc_batch_session_init(OcLlamaModel *model, size_t max_seqs,
                               OcBatchSession *out)
{
    if (model == NULL || out == NULL) return OC_ERR_INVALID_ARG;
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
    out->logits = xcalloc(max_seqs * model->cfg.vocab_size, sizeof(float));
    if (model->cfg.num_experts > 0) {
        out->router_logits = xcalloc(model->cfg.num_experts, sizeof(float));
        out->expert_gate = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_up = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->expert_out = xcalloc(model->cfg.n_embd, sizeof(float));
        out->shexp_gate = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->shexp_up = xcalloc(model->cfg.expert_intermediate_size, sizeof(float));
        out->shexp_out = xcalloc(model->cfg.n_embd, sizeof(float));
    }
    if (!out->kv_k || !out->kv_v || !out->x || !out->normed || !out->q ||
        !out->k || !out->v || !out->attn_out || !out->ffn_gate ||
        !out->ffn_up || !out->dequant_temp || !out->logits) {
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
        /* Set up a temporary session view sharing the workspace. */
        OcLlamaSession tmp;
        tmp.model = m;
        tmp.kv_k = bs->kv_k;
        tmp.kv_v = bs->kv_v;
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
        tmp.logits = bs->logits + s * m->cfg.vocab_size;
        tmp.router_logits = bs->router_logits;
        tmp.expert_gate = bs->expert_gate;
        tmp.expert_up = bs->expert_up;
        tmp.expert_out = bs->expert_out;
        tmp.shexp_gate = bs->shexp_gate;
        tmp.shexp_up = bs->shexp_up;
        tmp.shexp_out = bs->shexp_out;
        tmp.mla_c_q = NULL; /* MLA not supported in batch mode yet */
        tmp.mla_c_kv = NULL;
        tmp.mla_q_full = NULL;
        tmp.mla_kv_compressed = NULL;

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
    free(bs->logits);
    free(bs->router_logits);
    free(bs->expert_gate); free(bs->expert_up); free(bs->expert_out);
    free(bs->shexp_gate); free(bs->shexp_up); free(bs->shexp_out);
    memset(bs, 0, sizeof(*bs));
}
