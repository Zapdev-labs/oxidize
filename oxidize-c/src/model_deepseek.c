#include "model_deepseek.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"
#include "tensor.h"

/* ---- small helpers (mirrors of model_llama.c; kept local, no cross-TU statics) */

static void seterr(char* err, size_t n, const char* fmt, const char* a) {
  if (err && n) snprintf(err, n, fmt, a);
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* Dequantize a whole (1-D) tensor into a fresh f32 buffer; NULL if absent. */
static float* load_vec(const GgufFile* g, const char* name, size_t* n_out) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  if (!t) return NULL;
  size_t count = 1;
  for (uint32_t d = 0; d < t->n_dims; ++d) count *= (size_t)t->dims[d];
  float* v = malloc(count * sizeof(float));
  if (!v) return NULL;
  if (oc_dequant_row(t->ggml_type, t->data, v, count) != 0) {
    free(v);
    return NULL;
  }
  if (n_out) *n_out = count;
  return v;
}

/* A REQUIRED 1-D vector. Every caller gets a specific "missing tensor X" error,
 * so no load path has to inspect err[0] to find out whether one was already
 * written (that read is both a NULL-deref and an uninitialized read waiting to
 * happen — the caller's err buffer is not guaranteed zeroed). */
static float* load_vec_req(const GgufFile* g, const char* name, char* err, size_t errlen) {
  float* v = load_vec(g, name, NULL);
  if (!v) seterr(err, errlen, "deepseek: missing or unreadable tensor %s", name);
  return v;
}

/* Weight matrix by name; validates that we have a dequant kernel for its type. */
static const GgufTensorInfo* load_mat(const GgufFile* g, const char* name, char* err,
                                      size_t errlen) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  if (!t) {
    seterr(err, errlen, "deepseek: missing tensor %s", name);
    return NULL;
  }
  if (oc_row_bytes(t->ggml_type, (size_t)t->dims[0]) == 0) {
    if (err && errlen)
      snprintf(err, errlen, "deepseek: tensor %s has unsupported quant type %u", name,
               t->ggml_type);
    return NULL;
  }
  return t;
}

static bool arch_u32(const GgufFile* g, const char* arch, const char* suffix, uint32_t* out) {
  char key[192];
  snprintf(key, sizeof key, "%s.%s", arch, suffix);
  return gguf_get_u32(g, key, out);
}
static bool arch_f32(const GgufFile* g, const char* arch, const char* suffix, float* out) {
  char key[192];
  snprintf(key, sizeof key, "%s.%s", arch, suffix);
  return gguf_get_f32(g, key, out);
}

/* The V3 routed-bias vector is a ".bias" tensor; some converts store ".weight". */
static float* load_bias_probs(const GgufFile* g, size_t l) {
  char name[128];
  snprintf(name, sizeof name, "blk.%zu.exp_probs_b.bias", l);
  float* v = load_vec(g, name, NULL);
  if (v) return v;
  snprintf(name, sizeof name, "blk.%zu.exp_probs_b.weight", l);
  return load_vec(g, name, NULL);
}

/* ---- load ----------------------------------------------------------------- */

/* Load one layer's FFN: the MoE block (router + 3 expert stacks, optional shared
 * expert) when present, else the dense SwiGLU triple (a leading-dense layer).
 * Detection is per layer (tensor presence), so DeepSeek's leading dense blocks
 * then MoE both load without trusting leading_dense_block_count. Confirms
 * n_experts / expert_inter against the expert SHAPES. `name` is >=128 scratch. */
static int deepseek_load_ffn(DeepseekModel* m, DeepseekLayer* L, GgufFile* g, size_t l,
                             char* name, char* err, size_t errlen) {
  snprintf(name, 128, "blk.%zu.ffn_gate_inp.weight", l);
  const GgufTensorInfo* ginp = gguf_tensor(g, name);
  snprintf(name, 128, "blk.%zu.ffn_gate_exps.weight", l);
  const GgufTensorInfo* gexps = gguf_tensor(g, name);
  snprintf(name, 128, "blk.%zu.ffn_up_exps.weight", l);
  const GgufTensorInfo* uexps = gguf_tensor(g, name);
  snprintf(name, 128, "blk.%zu.ffn_down_exps.weight", l);
  const GgufTensorInfo* dexps = gguf_tensor(g, name);

  if (!(ginp && gexps && uexps && dexps)) { /* dense (leading) layer */
    snprintf(name, 128, "blk.%zu.ffn_gate.weight", l);
    L->ffn_gate = load_mat(g, name, err, errlen);
    snprintf(name, 128, "blk.%zu.ffn_up.weight", l);
    L->ffn_up = load_mat(g, name, err, errlen);
    snprintf(name, 128, "blk.%zu.ffn_down.weight", l);
    L->ffn_down = load_mat(g, name, err, errlen);
    return (L->ffn_gate && L->ffn_up && L->ffn_down) ? 0 : -1;
  }

  L->is_moe = true;
  m->has_moe = true;
  snprintf(name, 128, "blk.%zu.ffn_gate_inp.weight", l);
  L->ffn_gate_inp = load_mat(g, name, err, errlen);
  snprintf(name, 128, "blk.%zu.ffn_gate_exps.weight", l);
  L->ffn_gate_exps = load_mat(g, name, err, errlen);
  snprintf(name, 128, "blk.%zu.ffn_up_exps.weight", l);
  L->ffn_up_exps = load_mat(g, name, err, errlen);
  snprintf(name, 128, "blk.%zu.ffn_down_exps.weight", l);
  L->ffn_down_exps = load_mat(g, name, err, errlen);
  if (!L->ffn_gate_inp || !L->ffn_gate_exps || !L->ffn_up_exps || !L->ffn_down_exps)
    return -1;

  /* GGUF (ggml) dim order [cols, rows, n_expert]: gate/up [hidden, expert_inter,
   * n_expert]; down [expert_inter, hidden, n_expert]; router [hidden, n_expert]. */
  size_t t_ei = (size_t)L->ffn_gate_exps->dims[1];
  size_t t_ne = (size_t)L->ffn_gate_exps->dims[2];
  if (m->expert_inter == 0) m->expert_inter = t_ei;
  if (m->n_experts == 0) m->n_experts = t_ne;
  if (L->ffn_gate_exps->n_dims != 3 || L->ffn_up_exps->n_dims != 3 ||
      L->ffn_down_exps->n_dims != 3 ||
      (size_t)L->ffn_gate_exps->dims[0] != m->hidden || t_ei != m->expert_inter ||
      t_ne != m->n_experts || (size_t)L->ffn_up_exps->dims[0] != m->hidden ||
      (size_t)L->ffn_up_exps->dims[1] != m->expert_inter ||
      (size_t)L->ffn_up_exps->dims[2] != m->n_experts ||
      (size_t)L->ffn_down_exps->dims[0] != m->expert_inter ||
      (size_t)L->ffn_down_exps->dims[1] != m->hidden ||
      (size_t)L->ffn_down_exps->dims[2] != m->n_experts ||
      (size_t)L->ffn_gate_inp->dims[0] != m->hidden ||
      (size_t)L->ffn_gate_inp->dims[1] != m->n_experts) {
    seterr(err, errlen, "deepseek: MoE expert tensor shape mismatch (layer %s)", name);
    return -1;
  }

  L->ffn_exp_probs_b = load_bias_probs(g, l); /* optional V3 bias */

  /* Shared expert (a single merged SwiGLU triple; always-on, no sigmoid gate). */
  snprintf(name, 128, "blk.%zu.ffn_gate_shexp.weight", l);
  if (gguf_tensor(g, name)) {
    L->ffn_gate_shexp = load_mat(g, name, err, errlen);
    snprintf(name, 128, "blk.%zu.ffn_up_shexp.weight", l);
    L->ffn_up_shexp = load_mat(g, name, err, errlen);
    snprintf(name, 128, "blk.%zu.ffn_down_shexp.weight", l);
    L->ffn_down_shexp = load_mat(g, name, err, errlen);
    if (!L->ffn_gate_shexp || !L->ffn_up_shexp || !L->ffn_down_shexp) return -1;
    size_t si = (size_t)L->ffn_gate_shexp->dims[1];
    if (m->shexp_inter == 0) m->shexp_inter = si;
    if (si != m->shexp_inter || (size_t)L->ffn_gate_shexp->dims[0] != m->hidden ||
        (size_t)L->ffn_down_shexp->dims[0] != m->shexp_inter ||
        (size_t)L->ffn_down_shexp->dims[1] != m->hidden) {
      seterr(err, errlen, "deepseek: MoE shared-expert shape mismatch%s", "");
      return -1;
    }
  }
  return 0;
}

int deepseek_load(DeepseekModel* m, GgufFile* g, size_t max_ctx, char* err, size_t errlen) {
  memset(m, 0, sizeof(*m));
  char* arch = gguf_architecture(g);
  if (!arch) {
    seterr(err, errlen, "deepseek: model has no general.architecture%s", "");
    return -1;
  }

  uint32_t u;
  float f;
  m->hidden = arch_u32(g, arch, "embedding_length", &u) ? u : 0;
  m->n_layers = arch_u32(g, arch, "block_count", &u) ? u : 0;
  m->n_head = arch_u32(g, arch, "attention.head_count", &u) ? u : 0;
  m->inter = arch_u32(g, arch, "feed_forward_length", &u) ? u : 0;
  m->ctx = arch_u32(g, arch, "context_length", &u) ? u : 4096;
  m->eps = arch_f32(g, arch, "attention.layer_norm_rms_epsilon", &f) ? f : 1e-6f;
  m->rope_theta = arch_f32(g, arch, "rope.freq_base", &f) ? f : 1e4f;
  /* MLA geometry (KV first; validated against tensor shapes below). */
  m->kv_lora = arch_u32(g, arch, "attention.kv_lora_rank", &u) ? u : 0;
  m->q_lora = arch_u32(g, arch, "attention.q_lora_rank", &u) ? u : 0;
  m->qk_head_dim = arch_u32(g, arch, "attention.key_length", &u) ? u : 0;
  m->qk_rope = arch_u32(g, arch, "rope.dimension_count", &u) ? u : 0;
  m->v_head_dim = arch_u32(g, arch, "attention.value_length", &u) ? u : 0;
  /* MoE config (0 / false when no MoE layer). */
  m->n_experts = arch_u32(g, arch, "expert_count", &u) ? u : 0;
  m->n_experts_used = arch_u32(g, arch, "expert_used_count", &u) ? u : 0;
  m->n_experts_shared = arch_u32(g, arch, "expert_shared_count", &u) ? u : 0;
  m->expert_inter = arch_u32(g, arch, "expert_feed_forward_length", &u) ? u : 0;
  m->routed_scale = arch_f32(g, arch, "expert_weights_scale", &f) ? f : 1.0f;
  m->norm_topk_prob = arch_u32(g, arch, "expert_weights_norm", &u) ? (u != 0) : false;
  m->gating_sigmoid = arch_u32(g, arch, "expert_gating_func", &u) ? (u == 2) : false;
  m->n_group = arch_u32(g, arch, "expert_group_count", &u) ? u : 0;
  m->topk_group = arch_u32(g, arch, "expert_group_used_count", &u) ? u : 0;
  m->leading_dense = arch_u32(g, arch, "leading_dense_block_count", &u) ? u : 0;

  /* Softmax scale: mscale^2 / sqrt(qk_head_dim). mscale is the YaRN log-multiplier
   * correction (llama.cpp deepseek2) when rope.scaling.yarn_log_multiplier is set;
   * 1.0 otherwise. NOTE: only the attention-scale side of YaRN is applied here —
   * YaRN rope-frequency interpolation is not implemented, so a long-context model
   * relying on it needs that follow-up. Untested against a real YaRN model. */
  float yln, fac, mscale = 1.0f;
  if (arch_f32(g, arch, "rope.scaling.yarn_log_multiplier", &yln) &&
      arch_f32(g, arch, "rope.scaling.factor", &fac) && fac > 1.0f)
    mscale = 1.0f + yln * logf(fac);
  free(arch);

  if (max_ctx > 0 && max_ctx < m->ctx) m->ctx = max_ctx;
  if (!m->hidden || !m->n_layers || !m->n_head) {
    seterr(err, errlen, "deepseek: missing embedding_length/block_count/head_count%s", "");
    return -1;
  }

  m->tok_embd = load_mat(g, "token_embd.weight", err, errlen);
  if (!m->tok_embd) return -1;
  m->vocab = (size_t)m->tok_embd->dims[1];

  const GgufTensorInfo* ow = gguf_tensor(g, "output.weight");
  if (ow) {
    m->out_w = load_mat(g, "output.weight", err, errlen);
    if (!m->out_w) return -1;
    m->vocab = (size_t)m->out_w->dims[1];
  } else {
    m->out_w = m->tok_embd; /* tied */
  }

  m->out_norm = load_vec(g, "output_norm.weight", NULL);
  if (!m->out_norm) {
    seterr(err, errlen, "deepseek: missing tensor %s", "output_norm.weight");
    return -1;
  }

  m->layers = calloc(m->n_layers, sizeof(DeepseekLayer));
  if (!m->layers) {
    seterr(err, errlen, "deepseek: layer allocation failed%s", "");
    return -1;
  }

  for (size_t l = 0; l < m->n_layers; ++l) {
    DeepseekLayer* L = &m->layers[l];
    char name[128];

    snprintf(name, 128, "blk.%zu.attn_norm.weight", l);
    L->attn_norm = load_vec_req(g, name, err, errlen);
    if (!L->attn_norm) return -1;
    snprintf(name, 128, "blk.%zu.ffn_norm.weight", l);
    L->ffn_norm = load_vec_req(g, name, err, errlen);
    if (!L->ffn_norm) return -1;

    snprintf(name, 128, "blk.%zu.attn_kv_a_mqa.weight", l);
    L->mla_kv_a_mqa = load_mat(g, name, err, errlen);
    if (!L->mla_kv_a_mqa) return -1;
    snprintf(name, 128, "blk.%zu.attn_kv_a_norm.weight", l);
    L->mla_kv_a_norm = load_vec_req(g, name, err, errlen);
    if (!L->mla_kv_a_norm) return -1;
    snprintf(name, 128, "blk.%zu.attn_kv_b.weight", l);
    L->mla_kv_b = load_mat(g, name, err, errlen);
    if (!L->mla_kv_b) return -1;
    snprintf(name, 128, "blk.%zu.attn_output.weight", l);
    L->attn_out = load_mat(g, name, err, errlen);
    if (!L->attn_out) return -1;

    /* Q path: compressed (attn_q_a + attn_q_b, DeepSeek-V2/V3) or direct (attn_q,
     * DeepSeek-V2-Lite). Detected by tensor presence on layer 0; EVERY layer must
     * then agree, because the geometry below is cross-validated against layer 0
     * alone — a model that mixed the two would read q_lora from one layer and a
     * NULL q_a from another. */
    snprintf(name, 128, "blk.%zu.attn_q_a.weight", l);
    bool q_a_here = gguf_tensor(g, name) != NULL;
    if (l == 0) m->q_compressed = q_a_here;
    if (q_a_here != m->q_compressed) {
      seterr(err, errlen, "deepseek: layer %s disagrees with layer 0 on the q path "
                          "(some layers compressed attn_q_a, some plain attn_q)", name);
      return -1;
    }
    if (q_a_here) {
      L->mla_q_a = load_mat(g, name, err, errlen);
      if (!L->mla_q_a) return -1;
      snprintf(name, 128, "blk.%zu.attn_q_a_norm.weight", l);
      L->mla_q_a_norm = load_vec_req(g, name, err, errlen);
      if (!L->mla_q_a_norm) return -1;
      snprintf(name, 128, "blk.%zu.attn_q_b.weight", l);
      L->mla_q_b = load_mat(g, name, err, errlen);
      if (!L->mla_q_b) return -1;
    } else {
      snprintf(name, 128, "blk.%zu.attn_q.weight", l);
      L->mla_q_b = load_mat(g, name, err, errlen);
      if (!L->mla_q_b) return -1;
    }

    if (deepseek_load_ffn(m, L, g, l, name, err, errlen) != 0) return -1;
  }

  /* Fill geometry from shapes where the KV was absent, then cross-validate ALL of
   * it (a wrong stride here is silent garbage, so every dim is checked). */
  const DeepseekLayer* L0 = &m->layers[0];
  size_t kva_rows = (size_t)L0->mla_kv_a_mqa->dims[1];
  size_t kvb_rows = (size_t)L0->mla_kv_b->dims[1];
  size_t qb_rows = (size_t)L0->mla_q_b->dims[1];
  if (m->kv_lora == 0) m->kv_lora = (size_t)L0->mla_kv_b->dims[0];
  if (m->qk_rope == 0 && kva_rows > m->kv_lora) m->qk_rope = kva_rows - m->kv_lora;
  if (m->qk_head_dim == 0 && m->n_head) m->qk_head_dim = qb_rows / m->n_head;
  if (m->qk_head_dim > m->qk_rope) m->qk_nope = m->qk_head_dim - m->qk_rope;
  /* Guard the subtraction: on a malformed kv_b it would wrap size_t into a huge
   * v_head_dim that sails past the `v_head_dim == 0` reject below. Left at 0, the
   * validator rejects it loudly with the full geometry dump. */
  if (m->v_head_dim == 0 && m->n_head && kvb_rows / m->n_head > m->qk_nope)
    m->v_head_dim = (kvb_rows / m->n_head) - m->qk_nope;
  if (m->q_compressed) m->q_lora = (size_t)L0->mla_q_b->dims[0];

  size_t hd_kv = m->qk_nope + m->v_head_dim; /* per-head kv_b output width */
  if (m->kv_lora == 0 || m->qk_rope == 0 || m->qk_head_dim == 0 || m->qk_nope == 0 ||
      m->v_head_dim == 0 || (m->qk_rope % 2) != 0 ||
      kva_rows != m->kv_lora + m->qk_rope || (size_t)L0->mla_kv_b->dims[0] != m->kv_lora ||
      kvb_rows != m->n_head * hd_kv || qb_rows != m->n_head * m->qk_head_dim ||
      (size_t)L0->attn_out->dims[0] != m->n_head * m->v_head_dim ||
      (size_t)L0->attn_out->dims[1] != m->hidden ||
      (m->q_compressed && ((size_t)L0->mla_q_a->dims[0] != m->hidden ||
                           (size_t)L0->mla_q_b->dims[0] != m->q_lora)) ||
      (!m->q_compressed && (size_t)L0->mla_q_b->dims[0] != m->hidden)) {
    fprintf(stderr,
            "deepseek: MLA geometry mismatch: hidden=%zu n_head=%zu kv_lora=%zu "
            "q_lora=%zu qk_nope=%zu qk_rope=%zu qk_head=%zu v_head=%zu; "
            "kv_a_mqa_rows=%zu kv_b=[%llu,%llu] q_b_rows=%zu attn_out=[%llu,%llu]\n",
            m->hidden, m->n_head, m->kv_lora, m->q_lora, m->qk_nope, m->qk_rope,
            m->qk_head_dim, m->v_head_dim, kva_rows,
            (unsigned long long)L0->mla_kv_b->dims[0],
            (unsigned long long)L0->mla_kv_b->dims[1], qb_rows,
            (unsigned long long)L0->attn_out->dims[0],
            (unsigned long long)L0->attn_out->dims[1]);
    seterr(err, errlen, "deepseek: inconsistent MLA geometry%s", "");
    return -1;
  }

  m->softmax_scale = mscale * mscale / sqrtf((float)m->qk_head_dim);

  /* MoE counts: clamp used to [1, n_experts]; group counts to sane ranges. */
  if (m->has_moe) {
    if (m->n_experts_used == 0) m->n_experts_used = m->n_experts < 8 ? m->n_experts : 8;
    if (m->n_experts_used > m->n_experts) m->n_experts_used = m->n_experts;
    if (m->n_experts_used < 1) m->n_experts_used = 1;
    if (m->n_group > 1) {
      if (m->n_experts % m->n_group != 0) {
        seterr(err, errlen, "deepseek: expert_count not divisible by expert_group_count%s", "");
        return -1;
      }
      if (m->topk_group == 0 || m->topk_group > m->n_group) m->topk_group = m->n_group;
    }
  }

  fprintf(stderr,
          "deepseek: %zu layers hidden=%zu heads=%zu MLA[kv_lora=%zu q_lora=%zu "
          "qk_nope=%zu qk_rope=%zu v_head=%zu] scale=%.4g vocab=%zu ctx=%zu%s\n",
          m->n_layers, m->hidden, m->n_head, m->kv_lora, m->q_lora, m->qk_nope,
          m->qk_rope, m->v_head_dim, (double)m->softmax_scale, m->vocab, m->ctx,
          m->out_w == m->tok_embd ? " tied" : " untied");
  if (m->has_moe) {
    /* Report the bias off the first MoE layer, not layers[0] — in a real DeepSeek
     * layers[0] is a leading-DENSE block with no router at all. */
    const DeepseekLayer* moe0 = NULL;
    for (size_t l = 0; l < m->n_layers && !moe0; ++l)
      if (m->layers[l].is_moe) moe0 = &m->layers[l];
    fprintf(stderr,
            "deepseek: MoE %zu experts top-%zu%s expert_inter=%zu shared=%zu(inter=%zu) "
            "gating=%s norm_topk=%d scale=%.3g groups=%zu/topk_group=%zu leading_dense=%zu\n",
            m->n_experts, m->n_experts_used, (moe0 && moe0->ffn_exp_probs_b) ? " +bias" : "",
            m->expert_inter, m->n_experts_shared, m->shexp_inter,
            m->gating_sigmoid ? "sigmoid" : "softmax", (int)m->norm_topk_prob,
            (double)m->routed_scale, m->n_group, m->topk_group, m->leading_dense);
  }

  /* Latent KV cache (f32; already the compressed form, so --kv-type does not
   * apply). Reports the saving over a dense per-head K/V cache. */
  size_t hidden = m->hidden, nH = m->n_head;
  size_t lat_row = m->kv_lora + m->qk_rope;                 /* per pos, per layer */
  size_t dense_row = nH * (m->qk_head_dim + m->v_head_dim); /* what MHA would store */
  for (size_t l = 0; l < m->n_layers; ++l) {
    m->layers[l].kv_lat_cache = calloc(m->ctx * m->kv_lora, sizeof(float));
    m->layers[l].k_pe_cache = calloc(m->ctx * m->qk_rope, sizeof(float));
    if (!m->layers[l].kv_lat_cache || !m->layers[l].k_pe_cache) {
      seterr(err, errlen, "deepseek: latent cache allocation failed%s", "");
      return -1;
    }
  }
  fprintf(stderr, "deepseek: MLA latent cache %zu vals/pos/layer vs %zu dense (%.1fx smaller)\n",
          lat_row, dense_row, dense_row / (double)lat_row);
  m->kv_len = 0;

  size_t q_full = nH * m->qk_head_dim, v_full = nH * m->v_head_dim;
  size_t kva = m->kv_lora + m->qk_rope, kvb = nH * hd_kv, di = m->inter ? m->inter : 1;
  size_t qlo = m->q_lora ? m->q_lora : 1;
  m->octx = oc_ctx_new();
  m->x = calloc(hidden, sizeof(float));
  m->logits = calloc(m->vocab, sizeof(float));
  m->normed = calloc(hidden, sizeof(float));
  m->c_q = calloc(qlo, sizeof(float));
  m->c_q_normed = calloc(qlo, sizeof(float));
  m->q = calloc(q_full, sizeof(float));
  m->kv_a = calloc(kva, sizeof(float));
  m->c_kv = calloc(m->kv_lora, sizeof(float));
  m->k_pe = calloc(m->qk_rope, sizeof(float));
  m->attn_res = calloc(v_full, sizeof(float));
  m->attn_proj = calloc(hidden, sizeof(float));
  m->gate = calloc(di, sizeof(float));
  m->up = calloc(di, sizeof(float));
  m->ffn_out = calloc(hidden, sizeof(float));
  m->kv_b_recon = calloc(m->ctx * kvb, sizeof(float));
  if (!m->octx || !m->x || !m->logits || !m->normed || !m->c_q || !m->c_q_normed ||
      !m->q || !m->kv_a || !m->c_kv || !m->k_pe || !m->attn_res || !m->attn_proj ||
      !m->gate || !m->up || !m->ffn_out || !m->kv_b_recon) {
    seterr(err, errlen, "deepseek: scratch allocation failed%s", "");
    return -1;
  }

  if (m->has_moe) {
    size_t ng = m->n_group ? m->n_group : 1;
    size_t ff = m->expert_inter > m->shexp_inter ? m->expert_inter : m->shexp_inter;
    m->me_prob = calloc(m->n_experts, sizeof(float));
    m->me_selscore = calloc(m->n_experts, sizeof(float));
    m->me_grp = calloc(ng, sizeof(float));
    m->me_grpsel = calloc(ng, sizeof(int));
    m->me_w = calloc(m->n_experts_used, sizeof(float));
    m->me_sel = calloc(m->n_experts_used, sizeof(int));
    m->me_gate = calloc(ff, sizeof(float));
    m->me_up = calloc(ff, sizeof(float));
    m->me_eout = calloc(hidden, sizeof(float));
    if (!m->me_prob || !m->me_selscore || !m->me_grp || !m->me_grpsel || !m->me_w ||
        !m->me_sel || !m->me_gate || !m->me_up || !m->me_eout) {
      seterr(err, errlen, "deepseek: MoE scratch allocation failed%s", "");
      return -1;
    }
  }

  /* Batched-prefill scratch: the same vectors with a row per token. */
  {
    const char* e = getenv("OC_BATCH");
    long b = e ? atol(e) : 32;
    m->batch = (size_t)(b < 1 ? 1 : b > 512 ? 512 : b);
  }
  size_t B = m->batch;
  m->batch_cap = B;
  m->bx = calloc(B * hidden, sizeof(float));
  m->bnormed = calloc(B * hidden, sizeof(float));
  m->bcq = calloc(B * qlo, sizeof(float));
  m->bq = calloc(B * q_full, sizeof(float));
  m->battn = calloc(B * v_full, sizeof(float));
  m->bproj = calloc(B * hidden, sizeof(float));
  m->bgate = calloc(B * di, sizeof(float));
  m->bup = calloc(B * di, sizeof(float));
  m->bffn = calloc(B * hidden, sizeof(float));
  if (!m->bx || !m->bnormed || !m->bcq || !m->bq || !m->battn || !m->bproj ||
      !m->bgate || !m->bup || !m->bffn) {
    seterr(err, errlen, "deepseek: batch scratch allocation failed%s", "");
    return -1;
  }

  m->g = *g; /* take ownership */
  memset(g, 0, sizeof(*g));
  return 0;
}

void deepseek_free(DeepseekModel* m) {
  for (size_t l = 0; m->layers && l < m->n_layers; ++l) {
    DeepseekLayer* L = &m->layers[l];
    free(L->attn_norm);
    free(L->ffn_norm);
    free(L->mla_q_a_norm);
    free(L->mla_kv_a_norm);
    free(L->ffn_exp_probs_b);
    free(L->kv_lat_cache);
    free(L->k_pe_cache);
  }
  free(m->layers);
  free(m->out_norm);
  free(m->x);
  free(m->logits);
  free(m->normed);
  free(m->c_q);
  free(m->c_q_normed);
  free(m->q);
  free(m->kv_a);
  free(m->c_kv);
  free(m->k_pe);
  free(m->attn_res);
  free(m->attn_proj);
  free(m->gate);
  free(m->up);
  free(m->ffn_out);
  free(m->kv_b_recon);
  free(m->me_prob);
  free(m->me_selscore);
  free(m->me_grp);
  free(m->me_grpsel);
  free(m->me_w);
  free(m->me_sel);
  free(m->me_gate);
  free(m->me_up);
  free(m->me_eout);
  free(m->bx);
  free(m->bnormed);
  free(m->bcq);
  free(m->bq);
  free(m->battn);
  free(m->bproj);
  free(m->bgate);
  free(m->bup);
  free(m->bffn);
  oc_ctx_free(m->octx);
  gguf_close(&m->g);
  memset(m, 0, sizeof(*m));
}

/* ---- MLA attention -------------------------------------------------------- */

/* One kernel serves decode (n_q == 1, pos0 == current pos) and batched prefill
 * (n_q == chunk, pos0 == chunk base). Token i (global pos pos0+i) attends causally
 * over t in [0, pos0+i]; `seq = pos0 + i + 1` IS the causal mask — widen it and a
 * token attends to the future, silently. Per-head K/V is read from the shared
 * reconstruction buffer kv_b_recon (kv_b_proj of every cached latent) spliced with
 * the decoupled RoPE key cache; the full score is q_nope.k_nope + q_pe.k_pe. */
typedef struct {
  const float* kv_b_recon; /* [total_seq][n_head*(qk_nope+v_head)] */
  const float* k_pe_cache; /* [ctx][qk_rope] */
  const float* q;          /* [n_q][n_head*qk_head_dim] (q_pe already RoPE'd) */
  float* out;              /* [n_q][n_head*v_head_dim] */
  size_t n_head, qk_nope, qk_rope, qk_head_dim, v_head_dim, kvb_row, pos0;
  float scale;
} MlaAttnJob;

static void mla_attn(void* ctx, size_t i0, size_t i1) {
  MlaAttnJob* j = ctx;
  const size_t nH = j->n_head, qn = j->qk_nope, qr = j->qk_rope, qh = j->qk_head_dim;
  const size_t vd = j->v_head_dim, KVB = j->kvb_row, hd_kv = qn + vd;
  const size_t q_row = nH * qh, v_row = nH * vd;
  for (size_t idx = i0; idx < i1; ++idx) {
    size_t i = idx / nH, h = idx % nH;
    size_t seq = j->pos0 + i + 1;
    const float* q_nope = j->q + i * q_row + h * qh;
    const float* q_pe = q_nope + qn;
    float* oh = j->out + i * v_row + h * vd;
    float rmax = -INFINITY, rsum = 0.0f;
    for (size_t d = 0; d < vd; ++d) oh[d] = 0.0f;
    for (size_t t = 0; t < seq; ++t) {
      const float* base = j->kv_b_recon + t * KVB + h * hd_kv;
      const float* k_nope = base;
      const float* v = base + qn;
      const float* k_pe = j->k_pe_cache + t * qr;
      float score = (oc_dot_f32(q_nope, k_nope, qn) + oc_dot_f32(q_pe, k_pe, qr)) * j->scale;
      float nm = rmax > score ? rmax : score;
      float ef = expf(rmax - nm), es = expf(score - nm);
      if (ef != 1.0f)
        for (size_t d = 0; d < vd; ++d) oh[d] *= ef;
      for (size_t d = 0; d < vd; ++d) oh[d] += es * v[d];
      rsum = rsum * ef + es;
      rmax = nm;
    }
    if (rsum > 0.0f) {
      float inv = 1.0f / rsum;
      for (size_t d = 0; d < vd; ++d) oh[d] *= inv;
    }
  }
}

/* ---- MoE ------------------------------------------------------------------ */

/* DeepSeek group-limited router — see model_deepseek.h. sigmoid/softmax over the
 * experts, add the V3 bias for SELECTION only, keep the top-2-sum groups, take
 * top-k experts, weight by the ORIGINAL (unbiased) prob, then norm_topk + scale.
 * k is tiny so the O(k^2 n) skip-scan is cheaper than a heap; ties break to the
 * lower index (first-max-wins, `>`), so the hand reference and the model agree. */
void deepseek_moe_route(const DeepseekModel* m, const float* logits, const float* bias,
                        float* prob, float* selscore, float* grp, int* grpsel, int* sel,
                        float* w) {
  const size_t n = m->n_experts, used = m->n_experts_used;
  if (m->gating_sigmoid) {
    for (size_t e = 0; e < n; ++e) prob[e] = 1.0f / (1.0f + expf(-logits[e]));
  } else {
    float mx = -INFINITY;
    for (size_t e = 0; e < n; ++e)
      if (logits[e] > mx) mx = logits[e];
    float s = 0.0f;
    for (size_t e = 0; e < n; ++e) {
      prob[e] = expf(logits[e] - mx);
      s += prob[e];
    }
    if (s > 0.0f)
      for (size_t e = 0; e < n; ++e) prob[e] /= s;
  }
  for (size_t e = 0; e < n; ++e) selscore[e] = prob[e] + (bias ? bias[e] : 0.0f);

  /* Group-limited routing: keep only experts in the top topk_group groups, where
   * a group's score is the SUM of its top-2 selection scores (llama.cpp/V3). */
  if (m->n_group > 1 && m->topk_group < m->n_group) {
    const size_t ng = m->n_group, epg = n / ng, tkg = m->topk_group;
    for (size_t gI = 0; gI < ng; ++gI) {
      float b1 = -INFINITY, b2 = -INFINITY;
      for (size_t e = gI * epg; e < (gI + 1) * epg; ++e) {
        float v = selscore[e];
        if (v > b1) { b2 = b1; b1 = v; }
        else if (v > b2) { b2 = v; }
      }
      grp[gI] = (epg >= 2) ? b1 + b2 : b1;
    }
    for (size_t s = 0; s < tkg; ++s) {
      size_t best = 0;
      float bv = -INFINITY;
      for (size_t gI = 0; gI < ng; ++gI) {
        bool taken = false;
        for (size_t t = 0; t < s; ++t)
          if (grpsel[t] == (int)gI) { taken = true; break; }
        if (!taken && grp[gI] > bv) { bv = grp[gI]; best = gI; }
      }
      grpsel[s] = (int)best;
    }
    for (size_t e = 0; e < n; ++e) {
      size_t gI = e / epg;
      bool keep = false;
      for (size_t t = 0; t < tkg; ++t)
        if (grpsel[t] == (int)gI) { keep = true; break; }
      if (!keep) selscore[e] = -INFINITY;
    }
  }

  float wsum = 0.0f;
  for (size_t s = 0; s < used; ++s) {
    size_t best = 0;
    float bv = -INFINITY;
    for (size_t e = 0; e < n; ++e) {
      bool taken = false;
      for (size_t t = 0; t < s; ++t)
        if (sel[t] == (int)e) { taken = true; break; }
      if (!taken && selscore[e] > bv) { bv = selscore[e]; best = e; }
    }
    sel[s] = (int)best;
    w[s] = prob[best]; /* weight from the ORIGINAL prob, never the biased score */
    wsum += prob[best];
  }
  /* norm_topk divides by the selected-weight sum UNCONDITIONALLY when set (no
   * used>1 special case) — that is what llama.cpp's build_moe_ffn norm_w and this
   * repo's llama_moe_route both do, and with used==1 it is the difference between
   * a weight of 1.0 and a weight of prob[best]. */
  float scale = m->routed_scale > 0.0f ? m->routed_scale : 1.0f;
  float denom = (m->norm_topk_prob && wsum > 0.0f) ? wsum : 1.0f;
  for (size_t s = 0; s < used; ++s) w[s] = scale * w[s] / denom;
}

/* Accumulate w * SwiGLU_e(x) into dst[hidden] for expert e of a stacked
 * [n_expert][rows][cols] triple (expert e begins at e*rows*row_bytes(type,cols)).
 * A shared/dense expert is e=0 over a 2-D tensor, so this serves both. */
static void moe_expert_swiglu(DeepseekModel* m, const GgufTensorInfo* gate,
                              const GgufTensorInfo* up, const GgufTensorInfo* down,
                              size_t e, size_t inter, const float* x, float w, float* dst) {
  const size_t h = m->hidden;
  const uint8_t* gp = gate->data + e * inter * oc_row_bytes(gate->ggml_type, h);
  const uint8_t* upp = up->data + e * inter * oc_row_bytes(up->ggml_type, h);
  const uint8_t* dp = down->data + e * h * oc_row_bytes(down->ggml_type, inter);
  oc_matvec(m->octx, m->me_gate, gate->ggml_type, gp, inter, h, x);
  oc_matvec(m->octx, m->me_up, up->ggml_type, upp, inter, h, x);
  for (size_t i = 0; i < inter; ++i) m->me_gate[i] = silu(m->me_gate[i]) * m->me_up[i];
  oc_matvec(m->octx, m->me_eout, down->ggml_type, dp, h, inter, m->me_gate);
  for (size_t i = 0; i < h; ++i) dst[i] += w * m->me_eout[i];
}

/* DeepSeek MoE FFN for one token: group-routed top-k experts (weighted SwiGLU
 * sum) plus the always-on shared expert. x is the post-ffn_norm hidden vector;
 * writes the full FFN output into out[hidden].
 *
 * ponytail: correctness-first token-serial loop over experts via oc_matvec — the
 * batched-prefill path calls this once per token. The throughput upgrade is to
 * gather tokens per expert and issue ONE oc_matmul per selected expert. */
static void deepseek_moe_ffn(DeepseekModel* m, const DeepseekLayer* L, const float* x,
                             float* out) {
  const size_t h = m->hidden;
  oc_matvec(m->octx, m->me_prob, L->ffn_gate_inp->ggml_type, L->ffn_gate_inp->data,
            m->n_experts, h, x); /* raw logits into me_prob (route overwrites it) */
  deepseek_moe_route(m, m->me_prob, L->ffn_exp_probs_b, m->me_prob, m->me_selscore,
                     m->me_grp, m->me_grpsel, m->me_sel, m->me_w);

  for (size_t i = 0; i < h; ++i) out[i] = 0.0f;
  for (size_t s = 0; s < m->n_experts_used; ++s)
    moe_expert_swiglu(m, L->ffn_gate_exps, L->ffn_up_exps, L->ffn_down_exps,
                      (size_t)m->me_sel[s], m->expert_inter, x, m->me_w[s], out);
  if (L->ffn_gate_shexp) /* shared expert: always-on, weight 1 (no sigmoid gate) */
    moe_expert_swiglu(m, L->ffn_gate_shexp, L->ffn_up_shexp, L->ffn_down_shexp, 0,
                      m->shexp_inter, x, 1.0f, out);
}

/* Dense SwiGLU FFN (leading-dense layer) for one token into out[hidden]. */
static void deepseek_dense_ffn(DeepseekModel* m, const DeepseekLayer* L, const float* x,
                               float* out) {
  const size_t h = m->hidden, inter = m->inter;
  oc_matvec(m->octx, m->gate, L->ffn_gate->ggml_type, L->ffn_gate->data, inter, h, x);
  oc_matvec(m->octx, m->up, L->ffn_up->ggml_type, L->ffn_up->data, inter, h, x);
  for (size_t i = 0; i < inter; ++i) m->gate[i] = silu(m->gate[i]) * m->up[i];
  oc_matvec(m->octx, out, L->ffn_down->ggml_type, L->ffn_down->data, h, inter, m->gate);
}

/* ---- MLA per-position projections (shared by decode and each batch token) ---
 * From the input-normed hidden `normed`, write the full RoPE'd query for this
 * position into `qout` [n_head*qk_head_dim], the normed latent into the cache at
 * `pos`, and the RoPE'd decoupled key into its cache at `pos`. `cq`/`cqn` are
 * q_lora scratch; `kva` is (kv_lora+qk_rope) scratch. */
static void mla_project_pos(DeepseekModel* m, const DeepseekLayer* L, const float* normed,
                            size_t pos, float* qout, float* cq, float* cqn, float* kva) {
  const float eps = m->eps, theta = m->rope_theta;
  const size_t nH = m->n_head, qh = m->qk_head_dim, qn = m->qk_nope, qr = m->qk_rope;
  /* Query */
  if (m->q_compressed) {
    oc_matvec(m->octx, cq, L->mla_q_a->ggml_type, L->mla_q_a->data, m->q_lora, m->hidden, normed);
    oc_rms_norm(cqn, cq, L->mla_q_a_norm, m->q_lora, eps);
    oc_matvec(m->octx, qout, L->mla_q_b->ggml_type, L->mla_q_b->data, nH * qh, m->q_lora, cqn);
  } else {
    oc_matvec(m->octx, qout, L->mla_q_b->ggml_type, L->mla_q_b->data, nH * qh, m->hidden, normed);
  }
  /* ADJACENT-PAIR rope (oc_rope_normal), NOT split-half (oc_rope) — see the
   * rope-mode note in model_deepseek.h. Only the pe slice of each q head rotates;
   * q_nope passes through untouched. */
  for (size_t hh = 0; hh < nH; ++hh)
    oc_rope_normal(qout + hh * qh + qn, qr, 1, pos, theta, 0);

  /* Key/Value latent + decoupled RoPE key */
  oc_matvec(m->octx, kva, L->mla_kv_a_mqa->ggml_type, L->mla_kv_a_mqa->data,
            m->kv_lora + qr, m->hidden, normed);
  oc_rms_norm(L->kv_lat_cache + pos * m->kv_lora, kva, L->mla_kv_a_norm, m->kv_lora, eps);
  float* kpe = L->k_pe_cache + pos * qr;
  memcpy(kpe, kva + m->kv_lora, qr * sizeof(float));
  oc_rope_normal(kpe, qr, 1, pos, theta, 0);
}

/* ---- forward (decode) ----------------------------------------------------- */

float* deepseek_forward(DeepseekModel* m, int32_t token, size_t pos, bool need_logits) {
  const size_t h = m->hidden, nH = m->n_head;
  const size_t kvb = nH * (m->qk_nope + m->v_head_dim), seq = pos + 1;
  if (pos >= m->ctx) {
    fprintf(stderr, "deepseek: position %zu exceeds context %zu\n", pos, m->ctx);
    return NULL;
  }

  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, h);
  oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * emb_row, m->x, h);

  for (size_t l = 0; l < m->n_layers; ++l) {
    const DeepseekLayer* L = &m->layers[l];

    /* ---- MLA attention ---- */
    oc_rms_norm(m->normed, m->x, L->attn_norm, h, m->eps);
    mla_project_pos(m, L, m->normed, pos, m->q, m->c_q, m->c_q_normed, m->kv_a);

    /* Reconstruct per-head K/V for every cached position from the latent, then
     * attend. This is where the small latent cache becomes full K/V — transiently,
     * one layer at a time, so the persistent cache stays the latent. */
    oc_matmul(m->octx, m->kv_b_recon, L->mla_kv_b->ggml_type, L->mla_kv_b->data, kvb,
              m->kv_lora, L->kv_lat_cache, seq);

    MlaAttnJob job = {m->kv_b_recon, L->k_pe_cache, m->q, m->attn_res, nH, m->qk_nope,
                      m->qk_rope, m->qk_head_dim, m->v_head_dim, kvb, pos, m->softmax_scale};
    oc_parallel_for(nH, mla_attn, &job);

    oc_matvec(m->octx, m->attn_proj, L->attn_out->ggml_type, L->attn_out->data, h,
              nH * m->v_head_dim, m->attn_res);
    for (size_t i = 0; i < h; ++i) m->x[i] += m->attn_proj[i];

    /* ---- FFN: dense (leading) or group-routed MoE ---- */
    oc_rms_norm(m->normed, m->x, L->ffn_norm, h, m->eps);
    if (L->is_moe)
      deepseek_moe_ffn(m, L, m->normed, m->ffn_out);
    else
      deepseek_dense_ffn(m, L, m->normed, m->ffn_out);
    for (size_t i = 0; i < h; ++i) m->x[i] += m->ffn_out[i];
  }
  m->kv_len = pos + 1;

  if (!need_logits) return NULL;
  oc_rms_norm(m->normed, m->x, m->out_norm, h, m->eps);
  oc_matvec(m->octx, m->logits, m->out_w->ggml_type, m->out_w->data, m->vocab, h, m->normed);
  return m->logits;
}

/* ---- forward (batched prefill) -------------------------------------------- */

float* deepseek_forward_batch(DeepseekModel* m, const int32_t* tokens, size_t n,
                              size_t pos0, bool need_logits) {
  const size_t h = m->hidden, nH = m->n_head, qh = m->qk_head_dim;
  const size_t q_row = nH * qh, v_row = nH * m->v_head_dim;
  const size_t kvb = nH * (m->qk_nope + m->v_head_dim);
  if (n == 0) return NULL;
  if (pos0 + n > m->ctx) {
    fprintf(stderr, "deepseek: batch [%zu,%zu) exceeds context %zu\n", pos0, pos0 + n, m->ctx);
    return NULL;
  }
  size_t bs = m->batch < m->batch_cap ? m->batch : m->batch_cap;
  if (bs < 1) bs = 1;
  if (n > bs) { /* one chunk at a time; logits come from the last */
    float* out = NULL;
    for (size_t i = 0; i < n; i += bs) {
      size_t c = n - i < bs ? n - i : bs;
      out = deepseek_forward_batch(m, tokens + i, c, pos0 + i, need_logits && i + c == n);
    }
    return out;
  }

  const size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, h);
  for (size_t i = 0; i < n; ++i) {
    size_t tk = (size_t)tokens[i] < m->vocab ? (size_t)tokens[i] : m->vocab - 1;
    oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * emb_row, m->bx + i * h, h);
  }

  for (size_t l = 0; l < m->n_layers; ++l) {
    const DeepseekLayer* L = &m->layers[l];

    for (size_t i = 0; i < n; ++i)
      oc_rms_norm(m->bnormed + i * h, m->bx + i * h, L->attn_norm, h, m->eps);

    /* Per-token MLA projections (RoPE and latent store are per position; the
     * projections themselves are cheap GEMVs reused from the decode path). Doing
     * them per token keeps batched and sequential bit-identical. */
    for (size_t i = 0; i < n; ++i)
      mla_project_pos(m, L, m->bnormed + i * h, pos0 + i, m->bq + i * q_row, m->c_q,
                      m->c_q_normed, m->kv_a);

    /* Reconstruct K/V for every position [0, pos0+n) once (shared by all query
     * tokens in the chunk), then attend with the per-token causal bound. */
    size_t seq = pos0 + n;
    oc_matmul(m->octx, m->kv_b_recon, L->mla_kv_b->ggml_type, L->mla_kv_b->data, kvb,
              m->kv_lora, L->kv_lat_cache, seq);

    MlaAttnJob job = {m->kv_b_recon, L->k_pe_cache, m->bq, m->battn, nH, m->qk_nope,
                      m->qk_rope, qh, m->v_head_dim, kvb, pos0, m->softmax_scale};
    oc_parallel_for(n * nH, mla_attn, &job);

    oc_matmul(m->octx, m->bproj, L->attn_out->ggml_type, L->attn_out->data, h, v_row,
              m->battn, n);
    for (size_t i = 0; i < n; ++i) {
      float* p = m->bproj + i * h;
      const float* xi = m->bx + i * h;
      for (size_t d = 0; d < h; ++d) p[d] += xi[d]; /* bproj = attn_out + residual */
    }

    if (L->is_moe) {
      for (size_t i = 0; i < n; ++i) {
        const float* pj = m->bproj + i * h;
        oc_rms_norm(m->normed, pj, L->ffn_norm, h, m->eps);
        deepseek_moe_ffn(m, L, m->normed, m->ffn_out);
        float* xi = m->bx + i * h;
        for (size_t d = 0; d < h; ++d) xi[d] = pj[d] + m->ffn_out[d];
      }
    } else {
      for (size_t i = 0; i < n; ++i)
        oc_rms_norm(m->bnormed + i * h, m->bproj + i * h, L->ffn_norm, h, m->eps);
      oc_matmul(m->octx, m->bgate, L->ffn_gate->ggml_type, L->ffn_gate->data, m->inter, h, m->bnormed, n);
      oc_matmul(m->octx, m->bup, L->ffn_up->ggml_type, L->ffn_up->data, m->inter, h, m->bnormed, n);
      for (size_t i = 0; i < n * m->inter; ++i) m->bgate[i] = silu(m->bgate[i]) * m->bup[i];
      oc_matmul(m->octx, m->bffn, L->ffn_down->ggml_type, L->ffn_down->data, h, m->inter, m->bgate, n);
      for (size_t i = 0; i < n; ++i) {
        float* xi = m->bx + i * h;
        const float* fo = m->bffn + i * h, *pj = m->bproj + i * h;
        for (size_t d = 0; d < h; ++d) xi[d] = pj[d] + fo[d];
      }
    }
  }
  m->kv_len = pos0 + n;

  if (!need_logits) return NULL;
  oc_rms_norm(m->normed, m->bx + (n - 1) * h, m->out_norm, h, m->eps);
  oc_matvec(m->octx, m->logits, m->out_w->ggml_type, m->out_w->data, m->vocab, h, m->normed);
  return m->logits;
}

void deepseek_kv_rewind(DeepseekModel* m, size_t pos) {
  if (pos < m->kv_len) m->kv_len = pos;
}
