/* Model load + forward. Two layer kinds:
 *  - dense Llama/Mistral/Qwen attention+FFN (TinyLlama, Qwen2.5, ...)
 *  - qwen3.5 hybrid: gated-DeltaNet SSM layers interleaved with gated
 *    full-attention layers, plus an optional MTP/nextn draft block.
 * GDN + MTP math ported from oxidize-core (layer_wise/ssm.rs, inference/mtp.rs),
 * following llama.cpp GGUF conventions (auto-detected at load):
 *  - ssm_a stores baked A = -exp(A_log)  <=> all values negative
 *  - ssm_conv1d dims (4, ch) = kernel-contiguous
 *  - norm weights have +1 baked         <=> mean(final_norm) > 0.5
 * YaRN rope scaling is ignored (matches the Rust reference; fine below the
 * original 256k context). */
#include "oc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONV_K 4

static void tensor_count_dims(const oc_tensor_info *ti, size_t *count) {
  size_t c = 1;
  for (uint32_t d = 0; d < ti->n_dims; ++d) c *= (size_t)ti->dims[d];
  *count = c;
}

static float *load_vec(const oc_gguf *g, const char *name) {
  const oc_tensor_info *ti = oc_find_tensor(g, name);
  if (!ti) return NULL;
  size_t count;
  tensor_count_dims(ti, &count);
  float *out = malloc(count * sizeof(float));
  oc_dequant_row(ti->quant, g->base + ti->offset, out, count);
  return out;
}

static float *load_vec_n(const oc_gguf *g, const char *name, size_t *n) {
  const oc_tensor_info *ti = oc_find_tensor(g, name);
  if (!ti) { *n = 0; return NULL; }
  tensor_count_dims(ti, n);
  return load_vec(g, name);
}

static bool keepq_ok(oc_quant q) { return q <= OC_IQ4_NL; }

static oc_weight load_weight(const oc_gguf *g, const char *name,
                             const oc_config *c) {
  oc_weight w = {0};
  const oc_tensor_info *ti = oc_find_tensor(g, name);
  if (!ti) return w;
  size_t count;
  tensor_count_dims(ti, &count);
  if (ti->n_dims >= 3) {
    w.cols = (size_t)ti->dims[ti->n_dims - 1];
    w.rows = (size_t)ti->dims[ti->n_dims - 2];
  } else if (ti->n_dims == 2 && c) {
    /* PyTorch / oxidize-convert [out,in] vs ggml [in,out] */
    if (ti->dims[1] == c->hidden_size) {
      w.rows = (size_t)ti->dims[0];
      w.cols = (size_t)ti->dims[1];
    } else if (ti->dims[0] == c->hidden_size) {
      w.rows = (size_t)ti->dims[1];
      w.cols = (size_t)ti->dims[0];
    } else {
      w.cols = (size_t)ti->dims[0];
      w.rows = (size_t)ti->dims[1];
    }
  } else {
    w.cols = (size_t)ti->dims[0];
    w.rows = ti->n_dims >= 2 ? (size_t)ti->dims[1] : 1;
  }
  if (keepq_ok(ti->quant)) {
    w.quantized = true;
    w.quant = ti->quant;
    w.data = g->base + ti->offset;
  } else {
    w.quantized = false;
    w.f32 = malloc(count * sizeof(float));
    oc_dequant_row(ti->quant, g->base + ti->offset, w.f32, count);
  }
  return w;
}

static bool weight_empty(const oc_weight *w) { return !w->quantized && !w->f32; }

static size_t weight_plane_bytes(const oc_weight *w) {
  return w->quantized ? w->rows * oc_row_bytes(w->quant, w->cols)
                      : w->rows * w->cols * sizeof(float);
}

/* llama.cpp / mixtral-style split experts: ffn_gate.0.weight … ffn_gate.N.weight */
static oc_weight stack_split_experts(const oc_gguf *g, const char *prefix,
                                     const char *base, size_t n_expert,
                                     const oc_config *c) {
  oc_weight w = {0};
  char name[160];
  snprintf(name, sizeof name, "%s%s.0.weight", prefix, base);
  oc_weight e0 = load_weight(g, name, c);
  if (weight_empty(&e0) || n_expert == 0) return w;

  size_t plane = weight_plane_bytes(&e0);
  uint8_t *buf = malloc(n_expert * plane);
  if (!buf) oc_die("model: MoE expert stack alloc failed");

  for (size_t e = 0; e < n_expert; ++e) {
    snprintf(name, sizeof name, "%s%s.%zu.weight", prefix, base, e);
    oc_weight ew = load_weight(g, name, c);
    if (weight_empty(&ew) || ew.cols != e0.cols || ew.rows != e0.rows ||
        ew.quant != e0.quant)
      oc_die("model: MoE expert %s missing or mismatched", name);
    if (ew.quantized)
      memcpy(buf + e * plane, ew.data, plane);
    else
      memcpy(buf + e * plane, ew.f32, plane);
    if (e > 0) free(ew.f32);
  }
  if (!e0.quantized) free(e0.f32);

  w = e0;
  if (w.quantized)
    w.data = buf;
  else
    w.f32 = (float *)buf;
  return w;
}

static void build_config(const oc_gguf *g, oc_config *c) {
  const char *arch = oc_meta_str(g, "general.architecture");
  if (!arch) arch = "llama";
  char key[128];
  uint32_t u;
  float f;
#define K(suffix) (snprintf(key, sizeof key, "%s." suffix, arch), key)

  const oc_tensor_info *emb = oc_find_tensor(g, "token_embd.weight");
  if (!emb) emb = oc_find_tensor(g, "tok_embeddings.weight");

  /* token_embd.weight is [n_embd, n_vocab] in GGUF. No arch defines a
     vocab_size metadata key, so derive from the embedding: vocab >> hidden for
     every real LLM, so max=vocab, min=hidden is robust to either storage order.
     Getting this backwards truncates the lm_head to the first `hidden` tokens
     and clamps input ids, yielding control-token word-salad instead of text. */
  c->hidden_size = oc_meta_u32(g, K("embedding_length"), &u) ? u : 0;
  if (!c->hidden_size && emb && emb->n_dims >= 2) {
    size_t d0 = (size_t)emb->dims[0], d1 = (size_t)emb->dims[1];
    c->hidden_size = d1 > d0 ? d0 : d1;
  }
  c->vocab_size = oc_meta_u32(g, K("vocab_size"), &u) ? u : 0;
  if (!c->vocab_size && emb && emb->n_dims >= 2) {
    size_t d0 = (size_t)emb->dims[0], d1 = (size_t)emb->dims[1];
    c->vocab_size = d1 > d0 ? d1 : d0;
  }
  if (!c->hidden_size) c->hidden_size = 4096;
  if (!c->vocab_size) c->vocab_size = 32000;
  c->context_size = oc_meta_u32(g, K("context_length"), &u) ? u : 4096;
  size_t nextn = oc_meta_u32(g, K("nextn_predict_layers"), &u) ? u : 0;
  size_t blocks = oc_meta_u32(g, K("block_count"), &u) ? u : 32;
  c->layer_count = blocks > nextn ? blocks - nextn : blocks;
  c->intermediate_size = oc_meta_u32(g, K("feed_forward_length"), &u) ? u : 11008;
  c->n_heads = oc_meta_u32(g, K("attention.head_count"), &u) ? u : 32;
  c->kv_heads = oc_meta_u32(g, K("attention.head_count_kv"), &u) && u > 0
                    ? u : c->n_heads;
  c->head_dim = oc_meta_u32(g, K("attention.key_length"), &u) && u > 0
                    ? u : c->hidden_size / c->n_heads;
  c->rms_eps = oc_meta_f32(g, K("attention.layer_norm_rms_epsilon"), &f) ? f : 1e-5f;
  c->rope_theta = oc_meta_f32(g, K("rope.freq_base"), &f) ? f : 10000.0f;
  c->rope_dim = oc_meta_u32(g, K("rope.dimension_count"), &u) ? u : 0;
  c->sliding_window = oc_meta_u32(g, K("attention.sliding_window"), &u) ? u : 0;
  if (strncmp(arch, "qwen35", 6) == 0 || strncmp(arch, "qwen3_5", 7) == 0) {
    if (c->rope_theta <= 10000.0f) c->rope_theta = 10000000.0f;
    if (c->rope_dim == 0) c->rope_dim = c->head_dim / 4;
  }
  const char *sctype = oc_meta_str(g, K("rope.scaling.type"));
  if (sctype && strcmp(sctype, "yarn") == 0) {
    c->yarn_factor = oc_meta_f32(g, K("rope.scaling.factor"), &f) ? f : 0.0f;
    c->yarn_orig_ctx =
        oc_meta_u32(g, K("rope.scaling.original_context_length"), &u) ? (float)u
                                                                      : 0.0f;
  }

  if (oc_meta_u32(g, K("expert_count"), &u) && u > 0) {
    c->n_expert = u;
    c->n_expert_used = oc_meta_u32(g, K("expert_used_count"), &u) ? u : 2;
    c->expert_ff = oc_meta_u32(g, K("expert_feed_forward_length"), &u) && u > 0
                       ? u : c->intermediate_size;
    /* norm_topk_prob: default on for softmax routers (mixtral/qwen3moe). */
    c->expert_weights_norm = 1;
    /* expert_gating_func: 1 = softmax (default), 2 = sigmoid (hunyuan/deepseek). */
    c->expert_gating_sigmoid =
        (oc_meta_u32(g, K("expert_gating_func"), &u) && u == 2) ? 1 : 0;
    /* routed-expert output scale (hunyuan router_scaling_factor / deepseek
     * routed_scaling_factor); absent -> 1.0 (no scaling). */
    c->expert_weights_scale =
        (oc_meta_f32(g, K("expert_weights_scale"), &f) && f > 0.0f) ? f : 1.0f;
  }
  if (oc_meta_u32(g, K("attention.kv_lora_rank"), &u) && u > 0)
    oc_die("model: MLA attention is out of scope for oxidize-c");
#undef K
}

/* Load one transformer block's weights into L. Prefix e.g. "blk.7.". */
static void load_block(const oc_gguf *g, const char *prefix, oc_layer *L,
                       const oc_config *c) {
  char name[160];
#define T(suffix) (snprintf(name, sizeof name, "%s" suffix, prefix), name)
  L->attn_norm = load_vec(g, T("attn_norm.weight"));
  L->ffn_norm = load_vec(g, T("ffn_norm.weight"));
  if (!L->ffn_norm) L->ffn_norm = load_vec(g, T("post_attention_norm.weight"));
  L->gate = load_weight(g, T("ffn_gate.weight"), c);
  L->up = load_weight(g, T("ffn_up.weight"), c);
  L->down = load_weight(g, T("ffn_down.weight"), c);

  /* MoE: router present -> expert-stacked FFN (dense gate/up/down are empty). */
  L->router = load_weight(g, T("ffn_gate_inp.weight"), c);
  if (!weight_empty(&L->router)) {
    L->is_moe = true;
    L->e_gate = load_weight(g, T("ffn_gate_exps.weight"), c);
    L->e_up = load_weight(g, T("ffn_up_exps.weight"), c);
    L->e_down = load_weight(g, T("ffn_down_exps.weight"), c);
    if (weight_empty(&L->e_gate) || weight_empty(&L->e_up) ||
        weight_empty(&L->e_down)) {
      L->e_gate = stack_split_experts(g, prefix, "ffn_gate", c->n_expert, c);
      L->e_up = stack_split_experts(g, prefix, "ffn_up", c->n_expert, c);
      L->e_down = stack_split_experts(g, prefix, "ffn_down", c->n_expert, c);
      L->split_moe = !weight_empty(&L->e_gate);
    }
    if (weight_empty(&L->e_gate) || weight_empty(&L->e_up) ||
        weight_empty(&L->e_down))
      oc_die("model: MoE layer %s missing expert tensors", prefix);
    /* Per-expert selection bias (hunyuan sigmoid router), NULL when absent. */
    L->exp_probs_b = load_vec_n(g, T("exp_probs_b.bias"), &L->exp_probs_b_n);
    /* Always-on shared expert (hunyuan shared_mlp / qwen-moe shared_expert). */
    L->sh_gate = load_weight(g, T("ffn_gate_shexp.weight"), c);
    L->sh_up = load_weight(g, T("ffn_up_shexp.weight"), c);
    L->sh_down = load_weight(g, T("ffn_down_shexp.weight"), c);
  }

  size_t na;
  L->ssm_a = load_vec_n(g, T("ssm_a"), &na);
  if (!L->ssm_a) L->ssm_a = load_vec_n(g, T("ssm_a.weight"), &na);
  if (L->ssm_a) {
    /* ---- gated-DeltaNet layer ---- */
    L->is_gdn = true;
    L->kv_slot = -1;
    L->n_v_heads = na;
    L->qkv = load_weight(g, T("attn_qkv.weight"), c);
    L->gdn_gate = load_weight(g, T("attn_gate.weight"), c);
    L->ssm_alpha = load_weight(g, T("ssm_alpha.weight"), c);
    L->ssm_beta = load_weight(g, T("ssm_beta.weight"), c);
    L->ssm_out = load_weight(g, T("ssm_out.weight"), c);
    size_t nn, nd;
    L->ssm_norm = load_vec_n(g, T("ssm_norm.weight"), &nn);
    L->ssm_dt_bias = load_vec_n(g, T("ssm_dt.bias"), &nd);
    if (weight_empty(&L->qkv) || weight_empty(&L->gdn_gate) ||
        weight_empty(&L->ssm_alpha) || !L->ssm_norm)
      oc_die("model: GDN layer %s missing tensors", prefix);
    L->qkv_out = L->qkv.rows;
    L->value_dim = L->gdn_gate.rows;
    L->key_dim = (L->qkv_out - L->value_dim) / 2;
    L->head_v = nn;
    L->n_k_heads = L->head_v > 0 ? L->key_dim / L->head_v : 1;
    L->head_k = L->n_k_heads > 0 ? L->key_dim / L->n_k_heads : L->head_v;

    /* conv weights, normalized to tap-major [tap][channel] at load */
    const oc_tensor_info *cv = oc_find_tensor(g, T("ssm_conv1d.weight"));
    if (cv) {
      size_t cn;
      float *raw = load_vec_n(g, T("ssm_conv1d.weight"), &cn);
      if (cn == CONV_K * L->qkv_out) {
        L->ssm_conv1d = malloc(cn * sizeof(float));
        bool kernel_contig = cv->dims[0] == CONV_K;
        for (size_t ch = 0; ch < L->qkv_out; ++ch)
          for (size_t tap = 0; tap < CONV_K; ++tap)
            L->ssm_conv1d[tap * L->qkv_out + ch] =
                kernel_contig ? raw[ch * CONV_K + tap] : raw[tap * L->qkv_out + ch];
      }
      free(raw);
    }
    L->state = calloc(L->n_v_heads * L->head_k * L->head_v, sizeof(float));
    L->conv_ring = calloc(CONV_K * L->qkv_out, sizeof(float));
  } else {
    /* ---- (gated) full-attention layer ---- */
    L->wq = load_weight(g, T("attn_q.weight"), c);
    L->wk = load_weight(g, T("attn_k.weight"), c);
    L->wv = load_weight(g, T("attn_v.weight"), c);
    L->wo = load_weight(g, T("attn_output.weight"), c);
    L->q_bias = load_vec_n(g, T("attn_q.bias"), &L->q_bias_n);
    L->k_bias = load_vec_n(g, T("attn_k.bias"), &L->k_bias_n);
    L->v_bias = load_vec_n(g, T("attn_v.bias"), &L->v_bias_n);
    L->q_norm = load_vec(g, T("attn_q_norm.weight"));
    L->k_norm = load_vec(g, T("attn_k_norm.weight"));
    if (!L->attn_norm || weight_empty(&L->wq) ||
        (weight_empty(&L->gate) && !L->is_moe))
      oc_die("model: layer %s missing dense weights (unsupported arch?)", prefix);
  }
  (void)c;
#undef T
}

static void bake_plus_one(float *v, size_t n) {
  if (v) for (size_t i = 0; i < n; ++i) v[i] += 1.0f;
}

oc_model *oc_model_load(const char *path, size_t max_ctx, int kv_int8) {
  oc_gguf *g = oc_gguf_load(path);
  oc_model *m = calloc(1, sizeof(*m));
  m->g = g;
  build_config(g, &m->cfg);
  m->cfg.max_ctx = max_ctx;
  m->cfg.kv_int8 = kv_int8;
  oc_config *c = &m->cfg;

  const char *emb_name = oc_find_tensor(g, "token_embd.weight")
                             ? "token_embd.weight" : "tok_embeddings.weight";
  m->tok_emb = load_weight(g, emb_name, c);
  if (weight_empty(&m->tok_emb)) oc_die("model: missing token embedding");

  m->final_norm = load_vec(g, "output_norm.weight");
  if (!m->final_norm) m->final_norm = load_vec(g, "norm.weight");
  if (!m->final_norm) oc_die("model: missing final norm");

  m->lm_head = load_weight(g, "output.weight", c);
  if (weight_empty(&m->lm_head)) {
    m->lm_head = m->tok_emb;
    m->tied = true;
  }

  m->layers = calloc(c->layer_count, sizeof(oc_layer));
  char prefix[32];
  size_t kv_slots = 0;
  for (size_t l = 0; l < c->layer_count; ++l) {
    snprintf(prefix, sizeof prefix, "blk.%zu.", l);
    load_block(g, prefix, &m->layers[l], c);
    if (!m->layers[l].is_gdn) m->layers[l].kv_slot = (int)kv_slots++;
  }
  m->n_kv_layers = kv_slots ? kv_slots : 1;

  /* Per-layer attention geometry defaults (global config values). */
  m->emb_scale = 1.0f;
  for (size_t l = 0; l < c->layer_count; ++l) {
    oc_layer *L = &m->layers[l];
    L->out_scale_v = 1.0f;
    if (L->is_gdn) continue;
    L->hd = c->head_dim;
    L->n_kv = c->kv_heads;
    L->n_rot = c->rope_dim;
    L->theta = c->rope_theta;
  }

  /* ---- gemma4: per-layer head dims (256 SWA / 512 global), GELU ---- */
  const char *arch_s = oc_meta_str(g, "general.architecture");
  if (arch_s && strcmp(arch_s, "gemma4") == 0) {
    m->gemma = true;
    m->emb_scale = sqrtf((float)c->hidden_size);
    oc_meta_f32(g, "gemma4.final_logit_softcapping", &m->logit_softcap);
    m->rope_freqs = load_vec(g, "rope_freqs.weight");
    float f;
    float theta_swa =
        oc_meta_f32(g, "gemma4.rope.freq_base_swa", &f) ? f : 10000.0f;
    size_t window = c->sliding_window ? c->sliding_window : 1024;
    const oc_meta *swa = oc_meta_get(g, "gemma4.attention.sliding_window_pattern");
    char nm[160];
    for (size_t l = 0; l < c->layer_count; ++l) {
      oc_layer *L = &m->layers[l];
      bool is_swa = (swa && swa->kind == 2 && l < swa->count)
                        ? swa->nums[l] != 0
                        : ((l + 1) % 6 != 0);
      snprintf(nm, sizeof nm, "blk.%zu.attn_q_norm.weight", l);
      const oc_tensor_info *qni = oc_find_tensor(g, nm);
      snprintf(nm, sizeof nm, "blk.%zu.attn_k_norm.weight", l);
      const oc_tensor_info *kni = oc_find_tensor(g, nm);
      size_t hd_q = qni ? (size_t)qni->dims[0]
                        : (c->n_heads ? L->wq.rows / c->n_heads : c->head_dim);
      size_t hd_k = kni ? (size_t)kni->dims[0] : hd_q;
      L->hd = hd_q;
      if (!weight_empty(&L->wk) && hd_k > 0)
        L->n_kv = L->wk.rows / hd_k;
      L->attn_scale = 1.0f;
      L->n_rot = L->hd;
      L->theta = is_swa ? theta_swa : c->rope_theta;
      L->rope_ff = is_swa ? NULL : m->rope_freqs;
      L->v_from_k = weight_empty(&L->wv);
      L->v_rms = true;
      snprintf(nm, sizeof nm, "blk.%zu.post_attention_norm.weight", l);
      L->attn_post_norm = load_vec(g, nm);
      snprintf(nm, sizeof nm, "blk.%zu.post_ffw_norm.weight", l);
      L->ffn_post_norm = load_vec(g, nm);
      snprintf(nm, sizeof nm, "blk.%zu.layer_output_scale.weight", l);
      float *osv = load_vec(g, nm);
      if (osv) { L->out_scale_v = osv[0]; free(osv); }
      L->kv_cap = is_swa ? window : 0;   /* 0 = full context, set below */
      if (!L->attn_post_norm || !L->ffn_post_norm)
        oc_die("model: gemma4 layer %zu missing post-norms", l);
    }
  }

  /* MTP/nextn block: first block index past the main stack. */
  snprintf(prefix, sizeof prefix, "blk.%zu.", c->layer_count);
  char probe[160];
  snprintf(probe, sizeof probe, "%snextn.eh_proj.weight", prefix);
  if (oc_find_tensor(g, probe)) {
    m->mtp = calloc(1, sizeof(oc_mtp));
    load_block(g, prefix, &m->mtp->layer, c);
    char name[160];
    snprintf(name, sizeof name, "%snextn.eh_proj.weight", prefix);
    m->mtp->eh_proj = load_weight(g, name, c);
    snprintf(name, sizeof name, "%snextn.enorm.weight", prefix);
    m->mtp->enorm = load_vec(g, name);
    snprintf(name, sizeof name, "%snextn.hnorm.weight", prefix);
    m->mtp->hnorm = load_vec(g, name);
    snprintf(name, sizeof name, "%snextn.shared_head_norm.weight", prefix);
    m->mtp->head_norm = load_vec(g, name);
    m->mtp->layer.kv_slot = 0;
    m->mtp->layer.hd = c->head_dim;
    m->mtp->layer.n_kv = c->kv_heads;
    m->mtp->layer.n_rot = c->rope_dim;
    m->mtp->layer.theta = c->rope_theta;
    m->mtp->layer.out_scale_v = 1.0f;
    m->mtp->draft_max = 16;
    size_t kvn = c->kv_heads * c->head_dim;
    m->mtp->kv_k = calloc(m->mtp->draft_max * kvn, sizeof(float));
    m->mtp->kv_v = calloc(m->mtp->draft_max * kvn, sizeof(float));
    if (weight_empty(&m->mtp->eh_proj) || !m->mtp->enorm || !m->mtp->hnorm) {
      fprintf(stderr, "warning: incomplete MTP block; drafting disabled\n");
      m->mtp = NULL; /* ponytail: leak tiny buffers once at load */
    }
  }

  /* Zero-centered norm detection: raw qwen3.5 norms sit near 0, baked near 1. */
  {
    double s = 0;
    for (size_t i = 0; i < c->hidden_size; ++i) s += m->final_norm[i];
    if (s / (double)c->hidden_size < 0.5) {
      fprintf(stderr, "model: baking (1+w) into norm weights\n");
      size_t h = c->hidden_size;
      bake_plus_one(m->final_norm, h);
      for (size_t l = 0; l < c->layer_count; ++l) {
        oc_layer *L = &m->layers[l];
        bake_plus_one(L->attn_norm, h);
        bake_plus_one(L->ffn_norm, h);
        size_t hd = L->hd ? L->hd : c->head_dim;
        if (L->q_norm) bake_plus_one(L->q_norm, hd);
        if (L->k_norm) bake_plus_one(L->k_norm, hd);
        if (L->ssm_norm && L->head_v) bake_plus_one(L->ssm_norm, L->head_v);
        bake_plus_one(L->attn_post_norm, h);
        bake_plus_one(L->ffn_post_norm, h);
      }
      if (m->mtp) {
        oc_layer *L = &m->mtp->layer;
        bake_plus_one(L->attn_norm, h);
        bake_plus_one(L->ffn_norm, h);
        size_t mhd = L->hd ? L->hd : c->head_dim;
        if (L->q_norm) bake_plus_one(L->q_norm, mhd);
        if (L->k_norm) bake_plus_one(L->k_norm, mhd);
        bake_plus_one(m->mtp->enorm, h);
        bake_plus_one(m->mtp->hnorm, h);
        bake_plus_one(m->mtp->head_norm, h);
      }
    }
  }

  m->kv_stride = c->kv_heads * c->head_dim;
  m->kv_ctx = c->max_ctx > 0 && c->max_ctx < c->context_size ? c->max_ctx
                                                             : c->context_size;
  int kvq = c->kv_int8;
  if (kvq && c->head_dim > 512)
    oc_die("model: kv-int8 needs head_dim<=512 (got %zu)", c->head_dim);
  size_t kesz = kvq ? sizeof(int8_t) : sizeof(float);
  if (m->gemma) {
    /* per-layer caches; sliding-window layers ring at `window` positions */
    for (size_t l = 0; l < c->layer_count; ++l) {
      oc_layer *L = &m->layers[l];
      if (L->is_gdn) continue;
      if (L->kv_cap == 0 || L->kv_cap > m->kv_ctx) L->kv_cap = m->kv_ctx;
      size_t elems = L->kv_cap * L->n_kv * L->hd;
      if (kvq) {
        L->kv_ck8 = calloc(elems, 1);
        L->kv_cv8 = calloc(elems, 1);
        L->kv_cks = calloc(L->kv_cap * L->n_kv, sizeof(float));
        L->kv_cvs = calloc(L->kv_cap * L->n_kv, sizeof(float));
        if (!L->kv_ck8 || !L->kv_cv8 || !L->kv_cks || !L->kv_cvs)
          oc_die("model: KV cache allocation failed");
      } else {
        L->kv_ck = calloc(elems, sizeof(float));
        L->kv_cv = calloc(elems, sizeof(float));
        if (!L->kv_ck || !L->kv_cv) oc_die("model: KV cache allocation failed");
      }
    }
  } else {
    /* cap the up-front KV cache at 4 GB */
    size_t per_pos = m->n_kv_layers * m->kv_stride * kesz * 2;
    size_t budget = (size_t)4 << 30;
    if (per_pos > 0 && m->kv_ctx * per_pos > budget) {
      m->kv_ctx = budget / per_pos;
      if (m->kv_ctx < 4096) m->kv_ctx = 4096;
      fprintf(stderr, "note: KV cache capped to %zu positions\n", m->kv_ctx);
    }
    size_t kv_elems = m->n_kv_layers * m->kv_ctx * m->kv_stride;
    if (kvq) {
      size_t scales = m->n_kv_layers * m->kv_ctx * c->kv_heads;
      m->kv_k8 = calloc(kv_elems, 1);
      m->kv_v8 = calloc(kv_elems, 1);
      m->kv_ks = calloc(scales, sizeof(float));
      m->kv_vs = calloc(scales, sizeof(float));
      if (!m->kv_k8 || !m->kv_v8 || !m->kv_ks || !m->kv_vs)
        oc_die("model: KV cache allocation failed");
    } else {
      m->kv_k = calloc(kv_elems, sizeof(float));
      m->kv_v = calloc(kv_elems, sizeof(float));
      if (!m->kv_k || !m->kv_v) oc_die("model: KV cache allocation failed");
    }
  }
  m->x = calloc(c->hidden_size, sizeof(float));

#ifdef OC_CUDA
  /* Build the device-resident forward (weights + KV/SSM state on GPU). Falls
   * back to CPU if no device / disabled. MTP spec decode is CPU-only, so it is
   * auto-disabled by the CLI when gpu_active. */
  oc_cuda_build(m);
#endif
  return m;
}

static void free_block(oc_layer *L) {
  free(L->attn_norm); free(L->ffn_norm);
  free(L->wq.f32); free(L->wk.f32); free(L->wv.f32); free(L->wo.f32);
  free(L->gate.f32); free(L->up.f32); free(L->down.f32);
  free(L->router.f32);
  if (L->split_moe) {
    if (L->e_gate.quantized) free((void *)L->e_gate.data);
    else free(L->e_gate.f32);
    if (L->e_up.quantized) free((void *)L->e_up.data);
    else free(L->e_up.f32);
    if (L->e_down.quantized) free((void *)L->e_down.data);
    else free(L->e_down.f32);
  } else {
    free(L->e_gate.f32);
    free(L->e_up.f32);
    free(L->e_down.f32);
  }
  free(L->exp_probs_b);
  free(L->sh_gate.f32); free(L->sh_up.f32); free(L->sh_down.f32);
  free(L->q_bias); free(L->k_bias); free(L->v_bias);
  free(L->q_norm); free(L->k_norm);
  free(L->attn_post_norm); free(L->ffn_post_norm);
  free(L->kv_ck); free(L->kv_cv);
  free(L->kv_ck8); free(L->kv_cv8); free(L->kv_cks); free(L->kv_cvs);
  free(L->qkv.f32); free(L->gdn_gate.f32); free(L->ssm_alpha.f32);
  free(L->ssm_beta.f32); free(L->ssm_out.f32);
  free(L->ssm_a); free(L->ssm_dt_bias); free(L->ssm_conv1d); free(L->ssm_norm);
  free(L->state); free(L->conv_ring);
}

void oc_model_free(oc_model *m) {
  if (!m) return;
  for (size_t l = 0; l < m->cfg.layer_count; ++l) free_block(&m->layers[l]);
  free(m->layers);
  if (m->mtp) {
    free_block(&m->mtp->layer);
    free(m->mtp->eh_proj.f32);
    free(m->mtp->enorm); free(m->mtp->hnorm); free(m->mtp->head_norm);
    free(m->mtp->kv_k); free(m->mtp->kv_v);
    free(m->mtp);
  }
  free(m->tok_emb.f32);
  if (!m->tied) free(m->lm_head.f32);
  free(m->final_norm);
  free(m->kv_k); free(m->kv_v);
  free(m->kv_k8); free(m->kv_v8); free(m->kv_ks); free(m->kv_vs);
  free(m->x);
  free(m->rope_freqs);
  oc_gguf_free(m->g);
  free(m);
}

void oc_reset_state(oc_model *m) {
#ifdef OC_CUDA
  if (m->gpu_active) { oc_cuda_reset(m); return; }
#endif
  for (size_t l = 0; l < m->cfg.layer_count; ++l) {
    oc_layer *L = &m->layers[l];
    if (!L->is_gdn) continue;
    memset(L->state, 0, L->n_v_heads * L->head_k * L->head_v * sizeof(float));
    memset(L->conv_ring, 0, CONV_K * L->qkv_out * sizeof(float));
    L->ring_head = L->ring_len = 0;
  }
}

size_t oc_state_bytes(const oc_model *m) {
  size_t n = 0;
  for (size_t l = 0; l < m->cfg.layer_count; ++l) {
    const oc_layer *L = &m->layers[l];
    if (!L->is_gdn) continue;
    n += (L->n_v_heads * L->head_k * L->head_v + CONV_K * L->qkv_out) *
             sizeof(float) +
         2 * sizeof(int);
  }
  return n;
}

void oc_state_save(const oc_model *m, uint8_t *buf) {
  for (size_t l = 0; l < m->cfg.layer_count; ++l) {
    const oc_layer *L = &m->layers[l];
    if (!L->is_gdn) continue;
    size_t sb = L->n_v_heads * L->head_k * L->head_v * sizeof(float);
    size_t cb = CONV_K * L->qkv_out * sizeof(float);
    memcpy(buf, L->state, sb); buf += sb;
    memcpy(buf, L->conv_ring, cb); buf += cb;
    memcpy(buf, &L->ring_head, sizeof(int)); buf += sizeof(int);
    memcpy(buf, &L->ring_len, sizeof(int)); buf += sizeof(int);
  }
}

void oc_state_load(oc_model *m, const uint8_t *buf) {
  for (size_t l = 0; l < m->cfg.layer_count; ++l) {
    oc_layer *L = &m->layers[l];
    if (!L->is_gdn) continue;
    size_t sb = L->n_v_heads * L->head_k * L->head_v * sizeof(float);
    size_t cb = CONV_K * L->qkv_out * sizeof(float);
    memcpy(L->state, buf, sb); buf += sb;
    memcpy(L->conv_ring, buf, cb); buf += cb;
    memcpy(&L->ring_head, buf, sizeof(int)); buf += sizeof(int);
    memcpy(&L->ring_len, buf, sizeof(int)); buf += sizeof(int);
  }
}

static void embed_token(const oc_model *m, uint32_t tok, float *x) {
  size_t h = m->cfg.hidden_size;
  size_t idx = tok < m->cfg.vocab_size ? tok : m->cfg.vocab_size - 1;
  if (m->tok_emb.quantized) {
    size_t rb = oc_row_bytes(m->tok_emb.quant, h);
    oc_dequant_row(m->tok_emb.quant, m->tok_emb.data + idx * rb, x, h);
  } else {
    memcpy(x, m->tok_emb.f32 + idx * h, h * sizeof(float));
  }
  if (m->emb_scale != 1.0f)
    for (size_t i = 0; i < h; ++i) x[i] *= m->emb_scale;
}

static void add_bias(float *y, const float *bias, size_t bn, size_t n) {
  if (!bias) return;
  for (size_t i = 0; i < n; ++i) y[i] += bias[i % bn];
}

static void head_norm(float *v, const float *w, size_t heads, size_t hd, float eps) {
  for (size_t h = 0; h < heads; ++h) {
    float *p = v + h * hd;
    float ss = 0.0f;
    for (size_t i = 0; i < hd; ++i) ss += p[i] * p[i];
    float inv = 1.0f / sqrtf(ss / (float)hd + eps);
    if (w)
      for (size_t i = 0; i < hd; ++i) p[i] *= inv * w[i];
    else
      for (size_t i = 0; i < hd; ++i) p[i] *= inv;
  }
}

static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
static float softplusf_(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

/* ggml_l2_norm: x / max(sqrt(sum), eps) */
static void l2norm(float *v, size_t n) {
  float s = 0;
  for (size_t i = 0; i < n; ++i) s += v[i] * v[i];
  float inv = 1.0f / fmaxf(sqrtf(s), 1e-6f);
  for (size_t i = 0; i < n; ++i) v[i] *= inv;
}

/* gated RMS norm, HF Qwen3NextRMSNormGated order: gate FIRST (variance over
 * the gated values), then normalize and scale. Set OC_GDN_GATE_FIRST=0 to use
 * the alternative order (norm of raw x, then gate). */
static void gated_rms_norm(float *x, const float *w, const float *gate, size_t n,
                           float eps) {
  static int gate_after = -1;
  if (gate_after < 0) gate_after = getenv("OC_GDN_GATE_FIRST") == NULL;
  if (!gate_after)
    for (size_t i = 0; i < n; ++i) {
      float g = gate[i];
      x[i] *= g * sigmoidf_(g);
    }
  float var = 0;
  for (size_t i = 0; i < n; ++i) var += x[i] * x[i];
  var /= (float)n;
  float inv = 1.0f / sqrtf(var + eps);
  for (size_t i = 0; i < n; ++i) {
    float scale = inv * w[i];
    if (gate_after) {
      float g = gate[i];
      scale *= g * sigmoidf_(g);
    }
    x[i] *= scale;
  }
}

typedef struct {
  float *normed, *q, *k, *v, *attn, *attn_out, *gate, *up, *ffn_out;
  float *mixed, *conv, *ab, *z, *core;   /* GDN scratch (batch-sized) */
  float *e_logits, *e_g, *e_u, *e_out;   /* MoE scratch (single token) */
  oc_q8blk *xq;
  size_t batch;
} scratch_t;

/* max projection widths across all layers (q incl. gate half, qkv_out) */
static void model_maxdims(const oc_model *m, size_t *max_q, size_t *max_qkv,
                          size_t *max_kv) {
  size_t q = m->cfg.n_heads * m->cfg.head_dim, s = 0;
  size_t kv = m->cfg.kv_heads * m->cfg.head_dim;
  for (size_t l = 0; l < m->cfg.layer_count; ++l) {
    const oc_layer *L = &m->layers[l];
    size_t lkv = L->n_kv * L->hd;
    if (lkv > kv) kv = lkv;
    if (L->is_gdn) { if (L->qkv_out > s) s = L->qkv_out; }
    else if (L->wq.rows > q) q = L->wq.rows;
  }
  if (m->mtp && m->mtp->layer.wq.rows > q) q = m->mtp->layer.wq.rows;
  *max_q = q;
  *max_qkv = s;
  *max_kv = kv;
}

static scratch_t scratch_alloc(const oc_model *m, size_t batch) {
  const oc_config *c = &m->cfg;
  size_t h = c->hidden_size, in = c->intermediate_size;
  size_t max_q, max_qkv, max_kv;
  model_maxdims(m, &max_q, &max_qkv, &max_kv);
  scratch_t s = {0};
  s.batch = batch;
  s.normed = malloc(batch * h * sizeof(float));
  s.q = malloc(batch * max_q * sizeof(float));
  s.k = malloc(batch * (max_kv ? max_kv : 1) * sizeof(float));
  s.v = malloc(batch * (max_kv ? max_kv : 1) * sizeof(float));
  s.attn = malloc(batch * max_q * sizeof(float));
  s.attn_out = malloc(batch * h * sizeof(float));
  s.gate = malloc(batch * in * sizeof(float));
  s.up = malloc(batch * in * sizeof(float));
  s.ffn_out = malloc(batch * h * sizeof(float));
  if (max_qkv) {
    s.mixed = malloc(batch * max_qkv * sizeof(float));
    s.conv = malloc(batch * max_qkv * sizeof(float));
    s.ab = malloc(batch * 2 * 4096 * sizeof(float));
    s.z = malloc(batch * max_qkv * sizeof(float));
    s.core = malloc(batch * max_qkv * sizeof(float));
  }
  size_t ff = c->expert_ff;
  if (c->n_expert) {
    s.e_logits = malloc(c->n_expert * sizeof(float));
    s.e_g = malloc(ff * sizeof(float));
    s.e_u = malloc(ff * sizeof(float));
    s.e_out = malloc(h * sizeof(float));
  }
  size_t maxc = h > in ? h : in;
  if (max_q > maxc) maxc = max_q;
  if (max_qkv > maxc) maxc = max_qkv;
  if (ff > maxc) maxc = ff;
  s.xq = malloc(batch * (maxc / QK + 1) * sizeof(oc_q8blk));
  return s;
}

static void scratch_free(scratch_t *s) {
  free(s->normed); free(s->q); free(s->k); free(s->v); free(s->attn);
  free(s->attn_out); free(s->gate); free(s->up); free(s->ffn_out);
  free(s->mixed); free(s->conv); free(s->ab); free(s->z); free(s->core);
  free(s->e_logits); free(s->e_g); free(s->e_u); free(s->e_out);
  free(s->xq);
}

static const oc_q8blk *quant_acts(scratch_t *s, const float *src, size_t cols,
                                  size_t batch, bool needed) {
  if (!needed || cols % QK != 0) return NULL;
  size_t stride = cols / QK;
  for (size_t b = 0; b < batch; ++b)
    oc_quantize_act(src + b * cols, s->xq + b * stride, cols);
  return s->xq;
}

static bool w_needs_q8(const oc_weight *w) {
  return w->quantized && (w->quant == OC_Q4_0 || w->quant == OC_AL5 ||
                          w->quant == OC_Q8_0 || w->quant == OC_Q4_K ||
                          w->quant == OC_Q6_K);
}

/* ---- gated-DeltaNet layer (batch: projections batched, recurrence
 * sequential over tokens). Residual output written to out_all [batch*h]. */
static void gdn_layer(oc_model *m, oc_layer *L, const float *normed_all,
                      size_t batch, float *out_all, scratch_t *s) {
  const oc_config *c = &m->cfg;
  size_t h = c->hidden_size;
  size_t qo = L->qkv_out, vd = L->value_dim, kd = L->key_dim;
  size_t nvh = L->n_v_heads, hk = L->head_k, hv = L->head_v;
  size_t nkh = L->n_k_heads ? L->n_k_heads : 1;
  float out_scale = 1.0f / sqrtf((float)hv); /* ggml gated_delta_net kernel */
  bool a_baked = true;
  for (size_t i = 0; i < nvh; ++i)
    if (L->ssm_a[i] > 0) { a_baked = false; break; }

  const oc_q8blk *nq = quant_acts(s, normed_all, h, batch,
      w_needs_q8(&L->qkv) || w_needs_q8(&L->gdn_gate) ||
      w_needs_q8(&L->ssm_alpha) || w_needs_q8(&L->ssm_beta));
  float *mixed_all = s->mixed, *conv_all = s->conv, *z_all = s->z;
  float *a_all = s->ab, *b_all = s->ab + batch * nvh;
  oc_gemm(&L->qkv, qo, h, normed_all, nq, mixed_all, batch);
  oc_gemm(&L->ssm_alpha, nvh, h, normed_all, nq, a_all, batch);
  if (!weight_empty(&L->ssm_beta))
    oc_gemm(&L->ssm_beta, nvh, h, normed_all, nq, b_all, batch);
  else
    memset(b_all, 0, batch * nvh * sizeof(float));
  oc_gemm(&L->gdn_gate, vd, h, normed_all, nq, z_all, batch);

  /* causal conv (ring) + SiLU, sequential over tokens */
  for (size_t t = 0; t < batch; ++t) {
    const float *mixed = mixed_all + t * qo;
    float *conv = conv_all + t * qo;
    if (L->ssm_conv1d) {
      for (size_t ch = 0; ch < qo; ++ch) {
        float sum = L->ssm_conv1d[(CONV_K - 1) * qo + ch] * mixed[ch];
        for (int back = 1; back < CONV_K; ++back) {
          if (back > L->ring_len) break;
          int idx = (L->ring_head + CONV_K - back) % CONV_K;
          sum += L->ssm_conv1d[(CONV_K - 1 - back) * qo + ch] *
                 L->conv_ring[(size_t)idx * qo + ch];
        }
        conv[ch] = sum;
      }
      memcpy(L->conv_ring + (size_t)L->ring_head * qo, mixed, qo * sizeof(float));
      L->ring_head = (L->ring_head + 1) % CONV_K;
      if (L->ring_len < CONV_K) L->ring_len++;
    } else {
      memcpy(conv, mixed, qo * sizeof(float));
    }
    for (size_t i = 0; i < qo; ++i) conv[i] *= sigmoidf_(conv[i]);
  }

  /* delta-rule recurrence: parallel over heads, sequential over tokens */
  float *core_all = s->core; /* token-major [batch][vd] */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long long vh_ = 0; vh_ < (long long)nvh; ++vh_) {
    size_t vh = (size_t)vh_;
    float *st = L->state + vh * hk * hv;
    /* GQA grouping: each key head is shared by nvh/nkh value heads. */
    size_t k_head = (nkh > 0 && nvh >= nkh) ? (vh / (nvh / nkh)) : (vh % nkh);
    size_t q_off = k_head * hk, k_off = kd + k_head * hk;
    size_t v_off = kd * 2 + vh * hv;
    float qbuf[512], kbuf[512], kv_mem[512], delta[512];
    for (size_t t = 0; t < batch; ++t) {
      const float *conv = conv_all + t * qo;
      memcpy(qbuf, conv + q_off, hk * sizeof(float));
      memcpy(kbuf, conv + k_off, hk * sizeof(float));
      l2norm(qbuf, hk);
      l2norm(kbuf, hk);
      const float *v = conv + v_off;
      float beta = sigmoidf_(b_all[t * nvh + vh]);
      float dt = softplusf_(a_all[t * nvh + vh] +
                            (L->ssm_dt_bias ? L->ssm_dt_bias[vh] : 0.0f));
      float g = a_baked ? L->ssm_a[vh] * dt : -expf(L->ssm_a[vh]) * dt;
      float decay = expf(g);
      for (size_t i = 0; i < hk * hv; ++i) st[i] *= decay;
      for (size_t j = 0; j < hv; ++j) {
        float sum = 0;
        for (size_t i = 0; i < hk; ++i) sum += st[i * hv + j] * kbuf[i];
        kv_mem[j] = sum;
      }
      for (size_t j = 0; j < hv; ++j) delta[j] = (v[j] - kv_mem[j]) * beta;
      for (size_t i = 0; i < hk; ++i) {
        float ki = kbuf[i];
        float *row = st + i * hv;
        for (size_t j = 0; j < hv; ++j) row[j] += ki * delta[j];
      }
      float *out = core_all + t * vd + vh * hv;
      for (size_t j = 0; j < hv; ++j) {
        float sum = 0;
        for (size_t i = 0; i < hk; ++i) sum += st[i * hv + j] * qbuf[i];
        out[j] = sum * out_scale;
      }
    }
  }

  /* per-head gated RMS norm, then output projection */
  for (size_t t = 0; t < batch; ++t)
    for (size_t vh = 0; vh < nvh; ++vh)
      gated_rms_norm(core_all + t * vd + vh * hv, L->ssm_norm,
                     z_all + t * vd + vh * hv, hv, c->rms_eps);
  const oc_q8blk *cq = quant_acts(s, core_all, vd, batch, w_needs_q8(&L->ssm_out));
  oc_gemm(&L->ssm_out, h, vd, core_all, cq, out_all, batch);
}

/* ---- (gated) attention layer for one token. kv_k/kv_v are the cache arrays,
 * kv_ctx the per-layer stride in positions. Writes attention output
 * (pre-residual, post-wo) into attn_out [h]. */
/* rope_pos = absolute position for RoPE; slot = KV cache slot index. */
static void attn_layer_token(const oc_model *m, const oc_layer *L,
                             const float *normed, size_t rope_pos, size_t slot,
                             float *kv_k, float *kv_v, size_t kv_ctx,
                             int8_t *kv_k8, int8_t *kv_v8, float *kv_ks,
                             float *kv_vs, float *attn_out, scratch_t *s) {
  const oc_config *c = &m->cfg;
  size_t h = c->hidden_size;
  size_t hd = L->hd ? L->hd : c->head_dim;
  size_t n_kv = L->n_kv ? L->n_kv : c->kv_heads;
  size_t qg_len = L->wq.rows;
  size_t q_len = L->wo.rows;
  size_t kvn = n_kv * hd;
  float *qg = s->q, *k = s->k, *v = s->v;

  const oc_q8blk *nq = quant_acts((scratch_t *)s, normed, h, 1,
      w_needs_q8(&L->wq) || w_needs_q8(&L->wk) || w_needs_q8(&L->wv));
  oc_gemv(&L->wq, qg_len, h, normed, nq, qg);
  oc_gemv(&L->wk, kvn, h, normed, nq, k);
  if (L->v_from_k)
    memcpy(v, k, kvn * sizeof(float));       /* gemma4: no v_proj, V = K-proj */
  else
    oc_gemv(&L->wv, kvn, h, normed, nq, v);
  add_bias(qg, L->q_bias, L->q_bias_n, qg_len);
  add_bias(k, L->k_bias, L->k_bias_n, kvn);
  add_bias(v, L->v_bias, L->v_bias_n, kvn);

  /* Gated attention (qwen3.5): wq emits per-head interleaved [q(hd)|gate(hd)]
   * pairs (llama.cpp qwen3next layout). De-interleave into q + gate. */
  float *q = qg;
  float *gate = NULL;
  size_t q_heads = q_len / hd;
  if (qg_len >= 2 * q_len) {
    gate = s->gate; /* borrowed: ffn_block reuses it later, after we're done */
    for (size_t hh = 0; hh < q_heads; ++hh) {
      memmove(q + hh * hd, qg + hh * 2 * hd, hd * sizeof(float));
      memcpy(gate + hh * hd, qg + hh * 2 * hd + hd, hd * sizeof(float));
    }
  }
  if (L->q_norm) head_norm(q, L->q_norm, q_heads, hd, c->rms_eps);
  if (L->k_norm) head_norm(k, L->k_norm, n_kv, hd, c->rms_eps);
  if (L->v_rms) head_norm(v, NULL, n_kv, hd, c->rms_eps);
  float theta = L->theta != 0.0f ? L->theta : c->rope_theta;
  size_t n_rot = L->n_rot ? L->n_rot : c->rope_dim;
  oc_rope(q, hd, q_heads, rope_pos, theta, n_rot, c->yarn_factor,
          c->yarn_orig_ctx, L->rope_ff);
  oc_rope(k, hd, n_kv, rope_pos, theta, n_rot, c->yarn_factor,
          c->yarn_orig_ctx, L->rope_ff);

  size_t pos = slot % kv_ctx;
  size_t base = pos * kvn;
  size_t seq_len = slot + 1;
  if (seq_len > kv_ctx) seq_len = kv_ctx;

  if (kv_k8) {
    oc_quantize_kv(k, kv_k8 + base, kv_ks + pos * n_kv, n_kv, hd);
    oc_quantize_kv(v, kv_v8 + base, kv_vs + pos * n_kv, n_kv, hd);
    oc_attention_q8(s->attn, q, kv_k8, kv_ks, kv_v8, kv_vs, seq_len, q_heads,
                    n_kv, hd, L->attn_scale);
  } else {
    memcpy(kv_k + base, k, kvn * sizeof(float));
    memcpy(kv_v + base, v, kvn * sizeof(float));
    oc_attention(s->attn, q, kv_k, kv_v, seq_len, q_heads, n_kv, hd,
                 L->attn_scale);
  }
  if (gate)
    for (size_t i = 0; i < q_len; ++i) s->attn[i] *= sigmoidf_(gate[i]);

  const oc_q8blk *aq = quant_acts((scratch_t *)s, s->attn, q_len, 1,
                                  w_needs_q8(&L->wo));
  oc_gemv(&L->wo, h, q_len, s->attn, aq, attn_out);
}

/* View expert e of a stacked [n_expert][rows x cols] weight as a 2D oc_weight. */
static oc_weight expert_view(const oc_weight *w, size_t e) {
  oc_weight v = *w;
  size_t plane = w->rows * w->cols;
  if (w->quantized)
    v.data = w->data + (size_t)e * w->rows * oc_row_bytes(w->quant, w->cols);
  else
    v.f32 = w->f32 + (size_t)e * plane;
  return v;
}

/* MoE FFN for one token: router (softmax or sigmoid) -> top-k experts ->
 * weighted sum, plus an optional always-on shared expert (hunyuan/qwen-moe).
 * ponytail: normed is re-quantized per expert (quant_acts shares s->xq, and the
 * down-proj quant clobbers it); k is small so this is cheap. */
static void ffn_moe(const oc_model *m, const oc_layer *L, float *x,
                    scratch_t *s) {
  const oc_config *c = &m->cfg;
  size_t h = c->hidden_size, ff = c->expert_ff;
  size_t ne = c->n_expert, k = c->n_expert_used;
  if (k > 64) k = 64;
  oc_rms_norm(s->normed, x, L->ffn_norm, h, c->rms_eps);
  const oc_q8blk *rq = quant_acts(s, s->normed, h, 1, w_needs_q8(&L->router));
  oc_gemv(&L->router, ne, h, s->normed, rq, s->e_logits);
  /* Gating: sigmoid (hunyuan/deepseek) keeps per-expert probabilities as the
   * routing weight; softmax (mixtral/qwen-moe) normalizes across all experts.
   * For sigmoid, selection uses weight + per-expert bias but the weight itself
   * stays the raw sigmoid score (renormalized over the top-k below). */
  if (c->expert_gating_sigmoid)
    for (size_t e = 0; e < ne; ++e)
      s->e_logits[e] = 1.0f / (1.0f + expf(-s->e_logits[e]));
  else
    oc_softmax(s->e_logits, ne);

  size_t idx[64];
  float wts[64];
  for (size_t j = 0; j < k; ++j) {
    float best = -INFINITY;
    size_t bi = 0;
    for (size_t e = 0; e < ne; ++e) {
      bool taken = false;
      for (size_t t = 0; t < j; ++t) if (idx[t] == e) taken = true;
      if (taken) continue;
      float score = s->e_logits[e];
      if (L->exp_probs_b && e < L->exp_probs_b_n) score += L->exp_probs_b[e];
      if (score > best) { best = score; bi = e; }
    }
    idx[j] = bi;
    wts[j] = s->e_logits[bi]; /* routing weight, not the biased selection score */
  }
  if (c->expert_weights_norm) {
    float wsum = 0.0f;
    for (size_t j = 0; j < k; ++j) wsum += wts[j];
    if (wsum > 0.0f) for (size_t j = 0; j < k; ++j) wts[j] /= wsum;
  }
  if (c->expert_weights_scale != 1.0f)
    for (size_t j = 0; j < k; ++j) wts[j] *= c->expert_weights_scale;

  memset(s->ffn_out, 0, h * sizeof(float));
  for (size_t j = 0; j < k; ++j) {
    oc_weight wg = expert_view(&L->e_gate, idx[j]);
    oc_weight wu = expert_view(&L->e_up, idx[j]);
    oc_weight wd = expert_view(&L->e_down, idx[j]);
    const oc_q8blk *nq = quant_acts(s, s->normed, h, 1,
                                    w_needs_q8(&wg) || w_needs_q8(&wu));
    oc_gemv(&wg, ff, h, s->normed, nq, s->e_g);
    oc_gemv(&wu, ff, h, s->normed, nq, s->e_u);
    oc_swiglu(s->e_g, s->e_u, ff);
    const oc_q8blk *gq = quant_acts(s, s->e_g, ff, 1, w_needs_q8(&wd));
    oc_gemv(&wd, h, ff, s->e_g, gq, s->e_out);
    float wj = wts[j];
    for (size_t i = 0; i < h; ++i) s->ffn_out[i] += wj * s->e_out[i];
  }

  /* Always-on shared expert (unscaled). Uses s->gate/s->up as scratch (sized to
   * the dense intermediate width >= shared FF) so it never overruns e_g/e_u. */
  if (!weight_empty(&L->sh_gate)) {
    size_t sff = L->sh_gate.rows;
    const oc_q8blk *snq = quant_acts(s, s->normed, h, 1,
                                     w_needs_q8(&L->sh_gate) || w_needs_q8(&L->sh_up));
    oc_gemv(&L->sh_gate, sff, h, s->normed, snq, s->gate);
    oc_gemv(&L->sh_up, sff, h, s->normed, snq, s->up);
    oc_swiglu(s->gate, s->up, sff);
    const oc_q8blk *sgq = quant_acts(s, s->gate, sff, 1, w_needs_q8(&L->sh_down));
    oc_gemv(&L->sh_down, h, sff, s->gate, sgq, s->e_out);
    for (size_t i = 0; i < h; ++i) s->ffn_out[i] += s->e_out[i];
  }

  for (size_t i = 0; i < h; ++i) x[i] += s->ffn_out[i];
}

/* shared FFN block: x += down(swiglu(gate(norm(x)), up(norm(x)))) */
static void ffn_block(const oc_model *m, const oc_layer *L, float *x_batch,
                      size_t batch, scratch_t *s) {
  const oc_config *c = &m->cfg;
  size_t h = c->hidden_size, in = c->intermediate_size;
  for (size_t b = 0; b < batch; ++b)
    oc_rms_norm(s->normed + b * h, x_batch + b * h, L->ffn_norm, h, c->rms_eps);
  const oc_q8blk *fq = quant_acts(s, s->normed, h, batch,
      w_needs_q8(&L->gate) || w_needs_q8(&L->up));
  oc_gemm(&L->gate, in, h, s->normed, fq, s->gate, batch);
  oc_gemm(&L->up, in, h, s->normed, fq, s->up, batch);
  for (size_t b = 0; b < batch; ++b) {
    if (m->gemma) oc_geglu(s->gate + b * in, s->up + b * in, in);
    else oc_swiglu(s->gate + b * in, s->up + b * in, in);
  }
  const oc_q8blk *gq = quant_acts(s, s->gate, in, batch, w_needs_q8(&L->down));
  oc_gemm(&L->down, h, in, s->gate, gq, s->ffn_out, batch);
  if (L->ffn_post_norm)
    for (size_t b = 0; b < batch; ++b)
      oc_rms_norm(s->ffn_out + b * h, s->ffn_out + b * h, L->ffn_post_norm, h,
                  c->rms_eps);
  for (size_t b = 0; b < batch; ++b)
    for (size_t i = 0; i < h; ++i) x_batch[b * h + i] += s->ffn_out[b * h + i];
}

static void run_layers(oc_model *m, float *x_batch, size_t batch,
                       size_t start_pos, scratch_t *s) {
  const oc_config *c = &m->cfg;
  size_t h = c->hidden_size;

  for (size_t l = 0; l < c->layer_count; ++l) {
    oc_layer *L = &m->layers[l];
    for (size_t b = 0; b < batch; ++b)
      oc_rms_norm(s->normed + b * h, x_batch + b * h, L->attn_norm, h, c->rms_eps);

    if (L->is_gdn) {
      gdn_layer(m, L, s->normed, batch, s->attn_out, s);
      for (size_t b = 0; b < batch; ++b)
        for (size_t i = 0; i < h; ++i)
          x_batch[b * h + i] += s->attn_out[b * h + i];
    } else {
      float *kv_k = NULL, *kv_v = NULL;
      int8_t *kv_k8 = NULL, *kv_v8 = NULL;
      float *kv_ks = NULL, *kv_vs = NULL;
      size_t cap;
      if (L->kv_ck || L->kv_ck8) {
        cap = L->kv_cap;
        kv_k = L->kv_ck; kv_v = L->kv_cv;
        kv_k8 = L->kv_ck8; kv_v8 = L->kv_cv8;
        kv_ks = L->kv_cks; kv_vs = L->kv_cvs;
      } else {
        cap = m->kv_ctx;
        size_t off = (size_t)L->kv_slot * m->kv_ctx * m->kv_stride;
        size_t soff = (size_t)L->kv_slot * m->kv_ctx * c->kv_heads;
        if (m->kv_k8) {
          kv_k8 = m->kv_k8 + off; kv_v8 = m->kv_v8 + off;
          kv_ks = m->kv_ks + soff; kv_vs = m->kv_vs + soff;
        } else {
          kv_k = m->kv_k + off; kv_v = m->kv_v + off;
        }
      }
      for (size_t b = 0; b < batch; ++b) {
        attn_layer_token(m, L, s->normed + b * h, start_pos + b, start_pos + b,
                         kv_k, kv_v, cap, kv_k8, kv_v8, kv_ks, kv_vs,
                         s->attn_out, s);
        if (L->attn_post_norm)
          oc_rms_norm(s->attn_out, s->attn_out, L->attn_post_norm, h, c->rms_eps);
        for (size_t i = 0; i < h; ++i) x_batch[b * h + i] += s->attn_out[i];
      }
    }
    if (L->is_moe)
      for (size_t b = 0; b < batch; ++b) ffn_moe(m, L, x_batch + b * h, s);
    else
      ffn_block(m, L, x_batch, batch, s);
    if (L->out_scale_v != 1.0f && L->out_scale_v != 0.0f)
      for (size_t b = 0; b < batch; ++b)
        for (size_t i = 0; i < h; ++i) x_batch[b * h + i] *= L->out_scale_v;
    if (start_pos == 0 && getenv("OC_TRACE")) {
      double sum = 0, asum = 0;
      for (size_t i = 0; i < h; ++i) {
        sum += x_batch[i];
        asum += fabsf(x_batch[i]);
      }
      fprintf(stderr, "TRACE L%zu %s t0 sum=%.6e |sum|=%.6e x[0..4]=%.4f %.4f %.4f %.4f\n",
              l, L->is_gdn ? "gdn " : "attn", sum, asum, x_batch[0], x_batch[1],
              x_batch[2], x_batch[3]);
    }
  }
}

static void final_logits(oc_model *m, const float *x_batch, size_t batch,
                         float *logits_all, scratch_t *s) {
  const oc_config *c = &m->cfg;
  size_t h = c->hidden_size;
  for (size_t b = 0; b < batch; ++b)
    oc_rms_norm(s->normed + b * h, x_batch + b * h, m->final_norm, h, c->rms_eps);
  const oc_q8blk *hq = quant_acts(s, s->normed, h, batch, w_needs_q8(&m->lm_head));
  oc_gemm(&m->lm_head, c->vocab_size, h, s->normed, hq, logits_all, batch);
  if (m->logit_softcap > 0.0f) {
    float cap = m->logit_softcap;
    for (size_t i = 0; i < batch * c->vocab_size; ++i)
      logits_all[i] = cap * tanhf(logits_all[i] / cap);
  }
}

static void forward_impl(oc_model *m, const uint32_t *tokens, size_t n,
                         size_t start_pos, float *logits, bool all) {
  const oc_config *c = &m->cfg;
  if (n == 0) oc_die("forward: empty input");
  if (start_pos + n > m->kv_ctx)
    oc_die("forward: context exceeded (%zu > %zu)", start_pos + n, m->kv_ctx);
  size_t h = c->hidden_size;

#ifdef OC_CUDA
  if (m->gpu_active) {
    /* Resident GPU path: one token at a time (updates device KV + SSM state).
     * Only the last token needs logits (or every token in `all` verify mode). */
    float *emb = malloc(h * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
      bool want = all ? (logits != NULL) : (logits != NULL && i + 1 == n);
      embed_token(m, tokens[i], emb);
      float *lp = all ? (logits ? logits + i * c->vocab_size : NULL) : logits;
      oc_cuda_forward(m, emb, start_pos + i, want, want ? lp : NULL, NULL);
    }
    free(emb);
    return;
  }
#endif

  scratch_t s = scratch_alloc(m, n);
  float *xb = malloc(n * h * sizeof(float));
  for (size_t i = 0; i < n; ++i) embed_token(m, tokens[i], xb + i * h);
  run_layers(m, xb, n, start_pos, &s);
  memcpy(m->x, xb + (n - 1) * h, h * sizeof(float));
  if (logits) {
    if (all) final_logits(m, xb, n, logits, &s);
    else final_logits(m, m->x, 1, logits, &s);
  }
  free(xb);
  scratch_free(&s);
}

void oc_forward(oc_model *m, const uint32_t *tokens, size_t n, size_t start_pos,
                float *logits) {
  forward_impl(m, tokens, n, start_pos, logits, false);
}

void oc_forward_all(oc_model *m, const uint32_t *tokens, size_t n,
                    size_t start_pos, float *logits_all) {
  forward_impl(m, tokens, n, start_pos, logits_all, true);
}

/* Training path (finetune.c): GPU resident forward when available, else CPU.
 * Exposes post-final-norm hidden rows for the LoRA head. */
void oc_forward_train(oc_model *m, const uint32_t *tokens, size_t n,
                      size_t start_pos, float *normed_all, float *logits_all) {
  const oc_config *c = &m->cfg;
  if (n == 0) oc_die("forward: empty input");
  if (start_pos + n > m->kv_ctx)
    oc_die("forward: context exceeded (%zu > %zu)", start_pos + n, m->kv_ctx);
  size_t h = c->hidden_size;
  size_t vocab = c->vocab_size;

#ifdef OC_CUDA
  if (m->gpu_active) {
    float *emb = malloc(h * sizeof(float));
    scratch_t s = scratch_alloc(m, 1);
    for (size_t i = 0; i < n; ++i) {
      embed_token(m, tokens[i], emb);
      oc_cuda_forward(m, emb, start_pos + i, 0, NULL, normed_all + i * h);
      const oc_q8blk *hq =
          quant_acts(&s, normed_all + i * h, h, 1, w_needs_q8(&m->lm_head));
      oc_gemm(&m->lm_head, vocab, h, normed_all + i * h, hq,
              logits_all + i * vocab, 1);
    }
    scratch_free(&s);
    free(emb);
    if (m->logit_softcap > 0.0f) {
      float cap = m->logit_softcap;
      for (size_t i = 0; i < n * vocab; ++i)
        logits_all[i] = cap * tanhf(logits_all[i] / cap);
    }
    return;
  }
#endif

  scratch_t s = scratch_alloc(m, n);
  float *xb = malloc(n * h * sizeof(float));
  for (size_t i = 0; i < n; ++i) embed_token(m, tokens[i], xb + i * h);
  run_layers(m, xb, n, start_pos, &s);
  memcpy(m->x, xb + (n - 1) * h, h * sizeof(float));
  for (size_t b = 0; b < n; ++b)
    oc_rms_norm(normed_all + b * h, xb + b * h, m->final_norm, h, c->rms_eps);
  const oc_q8blk *hq = quant_acts(&s, normed_all, h, n, w_needs_q8(&m->lm_head));
  oc_gemm(&m->lm_head, c->vocab_size, h, normed_all, hq, logits_all, n);
  if (m->logit_softcap > 0.0f) {
    float cap = m->logit_softcap;
    for (size_t i = 0; i < n * c->vocab_size; ++i)
      logits_all[i] = cap * tanhf(logits_all[i] / cap);
  }
  free(xb);
  scratch_free(&s);
}

/* ---- MTP drafting (greedy). Anchored on m->x (last committed hidden). ---- */
size_t oc_mtp_draft(oc_model *m, uint32_t start_token, size_t start_pos, size_t k,
                    uint32_t *out) {
  if (!m->mtp || k == 0) return 0;
  oc_mtp *t = m->mtp;
  const oc_config *c = &m->cfg;
  size_t h = c->hidden_size;
  if (k > t->draft_max) k = t->draft_max;

  scratch_t s = scratch_alloc(m, 1);
  float *hidden = malloc(h * sizeof(float));
  float *x = malloc(h * sizeof(float));
  float *concat = malloc(2 * h * sizeof(float));
  float *logits = malloc(c->vocab_size * sizeof(float));
  /* anchor = main model's POST-output_norm hidden (llama.cpp t_h_nextn) */
  oc_rms_norm(hidden, m->x, m->final_norm, h, c->rms_eps);
  uint32_t cur = start_token;
  size_t produced = 0;

  for (size_t step = 0; step < k; ++step) {
    /* fuse embed(cur) with previous hidden */
    embed_token(m, cur, x);
    oc_rms_norm(concat, x, t->enorm, h, c->rms_eps);
    oc_rms_norm(concat + h, hidden, t->hnorm, h, c->rms_eps);
    const oc_q8blk *eq = quant_acts(&s, concat, 2 * h, 1,
                                    w_needs_q8(&t->eh_proj));
    oc_gemv(&t->eh_proj, h, 2 * h, concat, eq, x);

    /* one attention layer against the tiny draft KV (positions 0..step) */
    oc_rms_norm(s.normed, x, t->layer.attn_norm, h, c->rms_eps);
    attn_layer_token(m, &t->layer, s.normed, start_pos + step, step, t->kv_k,
                     t->kv_v, t->draft_max, NULL, NULL, NULL, NULL, s.attn_out,
                     &s);
    for (size_t i = 0; i < h; ++i) x[i] += s.attn_out[i];
    ffn_block(m, &t->layer, x, 1, &s);

    /* shared head */
    const float *hn = t->head_norm ? t->head_norm : m->final_norm;
    oc_rms_norm(hidden, x, hn, h, c->rms_eps);
    const oc_q8blk *hq = quant_acts(&s, hidden, h, 1, w_needs_q8(&m->lm_head));
    oc_gemv(&m->lm_head, c->vocab_size, h, hidden, hq, logits);
    uint32_t best = 0;
    for (size_t i = 1; i < c->vocab_size; ++i)
      if (logits[i] > logits[best]) best = (uint32_t)i;
    out[produced++] = best;
    cur = best;
  }
  free(hidden); free(x); free(concat); free(logits);
  scratch_free(&s);
  return produced;
}
