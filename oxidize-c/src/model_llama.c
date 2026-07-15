#include "model_llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"
#include "tensor.h"

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

/* Weight matrix by name; validates that we have a kernel for its type. */
static const GgufTensorInfo* load_mat(const GgufFile* g, const char* name,
                                      char* err, size_t errlen) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  if (!t) {
    seterr(err, errlen, "llama: missing tensor %s", name);
    return NULL;
  }
  size_t cols = (size_t)t->dims[0];
  if (oc_row_bytes(t->ggml_type, cols) == 0) {
    if (err && errlen)
      snprintf(err, errlen, "llama: tensor %s has unsupported quant type %u",
               name, t->ggml_type);
    return NULL;
  }
  return t;
}

/* KV readers under the model's own arch prefix ("llama.", "qwen2.", ...). */
static bool arch_u32(const GgufFile* g, const char* arch, const char* suffix,
                     uint32_t* out) {
  char key[160];
  snprintf(key, sizeof key, "%s.%s", arch, suffix);
  return gguf_get_u32(g, key, out);
}
static bool arch_f32(const GgufFile* g, const char* arch, const char* suffix,
                     float* out) {
  char key[160];
  snprintf(key, sizeof key, "%s.%s", arch, suffix);
  return gguf_get_f32(g, key, out);
}

/* Load a layer's FFN tensors: the Mixture-of-Experts block (router + the three
 * 3D expert stacks, plus an optional shared expert) when those tensors are
 * present, else the dense SwiGLU triple. Detection is per layer, so DeepSeek's
 * leading dense blocks then MoE both load. Confirms n_experts / expert_inter
 * against the expert tensor SHAPES (a wrong stride here is silent garbage) and
 * records them on the model. `name` is a caller-owned >=128-byte scratch.
 * Returns 0, or -1 with err set. */
static int llama_load_ffn(LlamaModel* m, LlamaLayer* L, GgufFile* g, size_t l,
                          char* name, char* err, size_t errlen) {
  snprintf(name, 128, "blk.%zu.ffn_gate_inp.weight", l);
  const GgufTensorInfo* ginp = gguf_tensor(g, name);
  snprintf(name, 128, "blk.%zu.ffn_gate_exps.weight", l);
  const GgufTensorInfo* gexps = gguf_tensor(g, name);
  snprintf(name, 128, "blk.%zu.ffn_up_exps.weight", l);
  const GgufTensorInfo* uexps = gguf_tensor(g, name);
  snprintf(name, 128, "blk.%zu.ffn_down_exps.weight", l);
  const GgufTensorInfo* dexps = gguf_tensor(g, name);

  if (!(ginp && gexps && uexps && dexps)) {
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

  /* Geometry from the expert tensors (shapes win). GGUF (ggml) dim order is
   * [cols, rows, n_expert]: gate/up are [hidden, expert_inter, n_expert]; down
   * is [expert_inter, hidden, n_expert]; router is [hidden, n_expert]. */
  size_t t_ei = (size_t)L->ffn_gate_exps->dims[1];
  size_t t_ne = (size_t)L->ffn_gate_exps->dims[2];
  if (m->expert_inter == 0) m->expert_inter = t_ei;
  if (m->n_experts == 0) m->n_experts = t_ne;
  if (L->ffn_gate_exps->n_dims != 3 || L->ffn_up_exps->n_dims != 3 ||
      L->ffn_down_exps->n_dims != 3 ||
      (size_t)L->ffn_gate_exps->dims[0] != m->hidden || t_ei != m->expert_inter ||
      t_ne != m->n_experts ||
      (size_t)L->ffn_up_exps->dims[0] != m->hidden ||
      (size_t)L->ffn_up_exps->dims[1] != m->expert_inter ||
      (size_t)L->ffn_up_exps->dims[2] != m->n_experts ||
      (size_t)L->ffn_down_exps->dims[0] != m->expert_inter ||
      (size_t)L->ffn_down_exps->dims[1] != m->hidden ||
      (size_t)L->ffn_down_exps->dims[2] != m->n_experts ||
      (size_t)L->ffn_gate_inp->dims[0] != m->hidden ||
      (size_t)L->ffn_gate_inp->dims[1] != m->n_experts) {
    fprintf(stderr,
            "llama: MoE layer %zu bad expert shapes: gate=[%llu,%llu,%llu] "
            "up=[%llu,%llu,%llu] down=[%llu,%llu,%llu] inp=[%llu,%llu]; "
            "want hidden=%zu expert_inter=%zu n_experts=%zu\n",
            l, (unsigned long long)L->ffn_gate_exps->dims[0],
            (unsigned long long)L->ffn_gate_exps->dims[1],
            (unsigned long long)L->ffn_gate_exps->dims[2],
            (unsigned long long)L->ffn_up_exps->dims[0],
            (unsigned long long)L->ffn_up_exps->dims[1],
            (unsigned long long)L->ffn_up_exps->dims[2],
            (unsigned long long)L->ffn_down_exps->dims[0],
            (unsigned long long)L->ffn_down_exps->dims[1],
            (unsigned long long)L->ffn_down_exps->dims[2],
            (unsigned long long)L->ffn_gate_inp->dims[0],
            (unsigned long long)L->ffn_gate_inp->dims[1], m->hidden,
            m->expert_inter, m->n_experts);
    seterr(err, errlen, "llama: MoE expert tensor shape mismatch%s", "");
    return -1;
  }

  /* Shared expert (optional; DeepSeek/Qwen2-MoE): a dense SwiGLU triple, always
   * on, optionally scaled by sigmoid(ffn_gate_inp_shexp . x) (Qwen2-MoE). */
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
      seterr(err, errlen, "llama: MoE shared-expert shape mismatch%s", "");
      return -1;
    }
    snprintf(name, 128, "blk.%zu.ffn_gate_inp_shexp.weight", l);
    L->ffn_gate_inp_shexp = load_vec(g, name, NULL); /* optional */
  }
  return 0;
}

int llama_load(LlamaModel* m, GgufFile* g, size_t max_ctx, char* err,
               size_t errlen) {
  memset(m, 0, sizeof(*m));
  char* arch = gguf_architecture(g);
  if (!arch) {
    seterr(err, errlen, "llama: model has no general.architecture%s", "");
    return -1;
  }

  uint32_t u;
  float f;
  m->hidden = arch_u32(g, arch, "embedding_length", &u) ? u : 4096;
  m->n_layers = arch_u32(g, arch, "block_count", &u) ? u : 32;
  m->n_head = arch_u32(g, arch, "attention.head_count", &u) ? u : 32;
  m->n_kv_heads = arch_u32(g, arch, "attention.head_count_kv", &u) ? u : m->n_head;
  m->inter = arch_u32(g, arch, "feed_forward_length", &u) ? u : 11008;
  m->ctx = arch_u32(g, arch, "context_length", &u) ? u : 4096;
  m->eps = arch_f32(g, arch, "attention.layer_norm_rms_epsilon", &f) ? f : 1e-5f;
  m->rope_theta = arch_f32(g, arch, "rope.freq_base", &f) ? f : 1e4f;
  uint32_t rope_dim_kv = arch_u32(g, arch, "rope.dimension_count", &u) ? u : 0;
  /* head_dim from KV; overridden by tensor shape below (shapes win). */
  m->head_dim = arch_u32(g, arch, "attention.key_length", &u) ? u
                : m->n_head ? m->hidden / m->n_head : 0;
  /* Rope layout: llama.cpp permutes q/k for ggml NORMAL (adjacent-pair) rope on
   * llama/mistral/yi; qwen2/qwen3/phi3 ship unpermuted q/k with NeoX rope. The
   * arch string is the only signal (there is no per-file rope_type KV). Note:
   * oxidize-convert's own GGUFs are unpermuted — they declare a distinct arch or
   * must be requantized to match this llama.cpp convention. */
  m->rope_norm = strcmp(arch, "llama") == 0 || strcmp(arch, "mistral") == 0 ||
                 strcmp(arch, "yi") == 0 || strcmp(arch, "mixtral") == 0;

  /* MoE config: 0 / false on a dense model. Counts and flags are read here;
   * geometry (n_experts / expert_inter) is confirmed against the expert tensor
   * shapes in the layer loop, where a mismatch is a loud error, not a silent
   * stride. norm_topk_prob (expert_weights_norm) reads through arch_u32 because
   * gguf_get_u32 already decodes GGUF_T_BOOL. */
  m->n_experts = arch_u32(g, arch, "expert_count", &u) ? u : 0;
  m->n_experts_used = arch_u32(g, arch, "expert_used_count", &u) ? u : 0;
  m->n_experts_shared = arch_u32(g, arch, "expert_shared_count", &u) ? u : 0;
  m->expert_weights_scale = arch_f32(g, arch, "expert_weights_scale", &f) ? f : 1.0f;
  m->norm_topk_prob = arch_u32(g, arch, "expert_weights_norm", &u) ? (u != 0) : true;
  /* Routing variants we do NOT model (sigmoid gating func == 2, DeepSeek-V3
   * group-limited routing) — warned about later iff this model is actually MoE. */
  uint32_t moe_gating = arch_u32(g, arch, "expert_gating_func", &u) ? u : 1;
  uint32_t moe_groups = arch_u32(g, arch, "expert_group_count", &u) ? u : 0;
  free(arch);
  if (max_ctx > 0 && max_ctx < m->ctx) m->ctx = max_ctx;
  if (m->n_head == 0 || m->n_kv_heads == 0 || m->n_head % m->n_kv_heads != 0) {
    seterr(err, errlen, "llama: bad head counts%s", "");
    return -1;
  }

  m->tok_embd = load_mat(g, "token_embd.weight", err, errlen);
  if (!m->tok_embd) return -1;
  m->vocab = (size_t)m->tok_embd->dims[1];

  /* Untied output head if output.weight is present; else tied to token_embd. */
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
    seterr(err, errlen, "llama: missing tensor %s", "output_norm.weight");
    return -1;
  }

  m->layers = calloc(m->n_layers, sizeof(LlamaLayer));
  if (!m->layers) {
    seterr(err, errlen, "llama: layer allocation failed%s", "");
    return -1;
  }

  bool has_bias = false, has_qknorm = false;
  for (size_t l = 0; l < m->n_layers; ++l) {
    LlamaLayer* L = &m->layers[l];
    char name[128];

#define MAT(field, suffix)                               \
  do {                                                   \
    snprintf(name, sizeof(name), "blk.%zu." suffix, l);  \
    L->field = load_mat(g, name, err, errlen);           \
    if (!L->field) return -1;                            \
  } while (0)
#define VEC(field, suffix)                               \
  do {                                                   \
    snprintf(name, sizeof(name), "blk.%zu." suffix, l);  \
    L->field = load_vec(g, name, NULL);                  \
  } while (0)

    MAT(attn_q, "attn_q.weight");
    MAT(attn_k, "attn_k.weight");
    MAT(attn_v, "attn_v.weight");
    MAT(attn_out, "attn_output.weight");
    if (llama_load_ffn(m, L, g, l, name, err, errlen) != 0) return -1;
    VEC(attn_norm, "attn_norm.weight");
    VEC(ffn_norm, "ffn_norm.weight");
    if (!L->attn_norm || !L->ffn_norm) {
      seterr(err, errlen, "llama: layer missing attn_norm/ffn_norm: %s", name);
      return -1;
    }
    /* Optional q/k/v/o bias (Qwen2) and per-head q/k RMSNorm (Qwen3). */
    VEC(bias_q, "attn_q.bias");
    VEC(bias_k, "attn_k.bias");
    VEC(bias_v, "attn_v.bias");
    VEC(bias_o, "attn_output.bias");
    VEC(attn_q_norm, "attn_q_norm.weight");
    VEC(attn_k_norm, "attn_k_norm.weight");
    if (L->bias_q || L->bias_k || L->bias_v) has_bias = true;
    if (L->attn_q_norm || L->attn_k_norm) has_qknorm = true;
#undef MAT
#undef VEC

    /* Geometry from tensor shapes; layer 0 sets head_dim, all layers validate. */
    size_t q_rows = (size_t)L->attn_q->dims[1];
    if (l == 0) m->head_dim = q_rows / m->n_head;
    size_t exp_q = m->n_head * m->head_dim;
    size_t exp_kv = m->n_kv_heads * m->head_dim;
    if (m->head_dim == 0 || q_rows != exp_q ||
        (size_t)L->attn_k->dims[1] != exp_kv ||
        (size_t)L->attn_v->dims[1] != exp_kv) {
      fprintf(stderr,
              "llama: layer %zu shape mismatch: q_rows=%zu (want %zu) "
              "k_rows=%llu v_rows=%llu (want %zu) head_dim=%zu heads=%zu/%zu\n",
              l, q_rows, exp_q, (unsigned long long)L->attn_k->dims[1],
              (unsigned long long)L->attn_v->dims[1], exp_kv, m->head_dim,
              m->n_head, m->n_kv_heads);
      seterr(err, errlen, "llama: inconsistent attention geometry%s", "");
      return -1;
    }
  }

  /* Partial rotary: rotate rope_dim dims per head, pass the rest through.
   * rope_dim == head_dim (or unset) => full rotary (oc_rope wants 0 for that). */
  m->rope_dim = rope_dim_kv > 0 && (size_t)rope_dim_kv < m->head_dim ? rope_dim_kv : 0;

  fprintf(stderr,
          "llama: %zu layers hidden=%zu heads=%zu/%zu head_dim=%zu inter=%zu "
          "vocab=%zu ctx=%zu rope_theta=%.0f rope_dim=%zu%s%s%s\n",
          m->n_layers, m->hidden, m->n_head, m->n_kv_heads, m->head_dim,
          m->inter, m->vocab, m->ctx, (double)m->rope_theta,
          m->rope_dim ? m->rope_dim : m->head_dim,
          m->out_w == m->tok_embd ? " tied" : " untied",
          has_bias ? " +qkv-bias" : "", has_qknorm ? " +qk-norm" : "");

  /* KV cache precision (process-wide, set by the CLI before load). The rotated
   * int4 rotoquant is gemma4-only (it needs the FHT + power-of-two head dims),
   * so a llama model asked for q4 falls back to f16 — still half-size, ~lossless.
   * The f16/q8 attention decodes one head at a time into a 512-float stack
   * buffer, so an exotic head_dim > 512 falls back to exact f32. */
  m->kv_type = oc_kv_get_type();
  if (m->kv_type == OC_KV_Q4) {
    fprintf(stderr, "llama: --kv-type q4 (rotoquant) is gemma4-only; using f16\n");
    m->kv_type = OC_KV_F16;
  }
  if (m->kv_type != OC_KV_F32 && m->head_dim > OC_KV_MAX_HEAD) {
    fprintf(stderr, "llama: head_dim %zu > %d; using f32 KV cache\n", m->head_dim,
            OC_KV_MAX_HEAD);
    m->kv_type = OC_KV_F32;
  }
  m->kv_len = 0;

  /* KV cache: ctx positions per layer, row = n_kv_heads * head_dim values,
   * oc_kv_elem_bytes each; q8 adds one scale per (position, kv-head). */
  size_t kv_row = m->n_kv_heads * m->head_dim;
  size_t elem = oc_kv_elem_bytes(m->kv_type);
  for (size_t l = 0; l < m->n_layers; ++l) {
    m->layers[l].k_cache = calloc(m->ctx * kv_row * elem, 1);
    m->layers[l].v_cache = calloc(m->ctx * kv_row * elem, 1);
    if (!m->layers[l].k_cache || !m->layers[l].v_cache) {
      seterr(err, errlen, "llama: KV cache allocation failed%s", "");
      return -1;
    }
    if (m->kv_type == OC_KV_Q8) {
      m->layers[l].k_scale = calloc(m->ctx * m->n_kv_heads, sizeof(float));
      m->layers[l].v_scale = calloc(m->ctx * m->n_kv_heads, sizeof(float));
      if (!m->layers[l].k_scale || !m->layers[l].v_scale) {
        seterr(err, errlen, "llama: KV scale allocation failed%s", "");
        return -1;
      }
    }
  }
  if (m->kv_type != OC_KV_F32)
    fprintf(stderr, "llama: KV cache %s (%zux smaller than f32)\n",
            oc_kv_type_name(m->kv_type), 4 / elem);

  size_t q_len = m->n_head * m->head_dim;
  m->x = calloc(m->hidden, sizeof(float));
  m->logits = calloc(m->vocab, sizeof(float));
  m->normed = calloc(m->hidden, sizeof(float));
  m->q = calloc(q_len, sizeof(float));
  m->k = calloc(kv_row, sizeof(float));
  m->v = calloc(kv_row, sizeof(float));
  m->attn_res = calloc(q_len, sizeof(float));
  m->attn_proj = calloc(m->hidden, sizeof(float));
  m->gate = calloc(m->inter, sizeof(float));
  m->up = calloc(m->inter, sizeof(float));
  m->ffn_out = calloc(m->hidden, sizeof(float));
  m->head_tmp = calloc(m->head_dim, sizeof(float));
  if (!m->x || !m->logits || !m->normed || !m->q || !m->k || !m->v ||
      !m->attn_res || !m->attn_proj || !m->gate || !m->up || !m->ffn_out ||
      !m->head_tmp) {
    seterr(err, errlen, "llama: scratch allocation failed%s", "");
    return -1;
  }

  /* Batched-prefill scratch: the same vectors with a row per token. */
  {
    const char* e = getenv("OC_BATCH");
    long b = e ? atol(e) : 32;
    m->batch = (size_t)(b < 1 ? 1 : b > 512 ? 512 : b);
  }
  size_t B = m->batch;
  m->batch_cap = B;
  m->octx = oc_ctx_new();
  m->bx = calloc(B * m->hidden, sizeof(float));
  m->bnormed = calloc(B * m->hidden, sizeof(float));
  m->bq = calloc(B * q_len, sizeof(float));
  m->bk = calloc(B * kv_row, sizeof(float));
  m->bv = calloc(B * kv_row, sizeof(float));
  m->battn = calloc(B * q_len, sizeof(float));
  m->bproj = calloc(B * m->hidden, sizeof(float));
  m->bgate = calloc(B * m->inter, sizeof(float));
  m->bup = calloc(B * m->inter, sizeof(float));
  m->bffn = calloc(B * m->hidden, sizeof(float));
  if (!m->octx || !m->bx || !m->bnormed || !m->bq || !m->bk || !m->bv ||
      !m->battn || !m->bproj || !m->bgate || !m->bup || !m->bffn) {
    seterr(err, errlen, "llama: batch scratch allocation failed%s", "");
    return -1;
  }

  /* MoE scratch (one token at a time; decode and the batched token loop share
   * it). n_experts_used clamps to [1, n_experts]; a MoE file that omits the key
   * is malformed — default to top-2 (Mixtral) with a note rather than refuse. */
  if (m->has_moe) {
    if (m->n_experts_used == 0) {
      m->n_experts_used = m->n_experts < 2 ? m->n_experts : 2;
      fprintf(stderr, "llama: MoE expert_used_count missing; defaulting to %zu\n",
              m->n_experts_used);
    }
    if (m->n_experts_used > m->n_experts) m->n_experts_used = m->n_experts;
    if (m->n_experts_used < 1) m->n_experts_used = 1;
    size_t ff = m->expert_inter > m->shexp_inter ? m->expert_inter : m->shexp_inter;
    m->me_logits = calloc(m->n_experts, sizeof(float));
    m->me_w = calloc(m->n_experts_used, sizeof(float));
    m->me_sel = calloc(m->n_experts_used, sizeof(int));
    m->me_gate = calloc(ff, sizeof(float));
    m->me_up = calloc(ff, sizeof(float));
    m->me_eout = calloc(m->hidden, sizeof(float));
    if (!m->me_logits || !m->me_w || !m->me_sel || !m->me_gate || !m->me_up ||
        !m->me_eout) {
      seterr(err, errlen, "llama: MoE scratch allocation failed%s", "");
      return -1;
    }
    fprintf(stderr,
            "llama: MoE %zu experts, top-%zu, expert_inter=%zu%s, "
            "norm_topk=%d scale=%.3g\n",
            m->n_experts, m->n_experts_used, m->expert_inter,
            m->shexp_inter ? " +shared" : "", (int)m->norm_topk_prob,
            (double)m->expert_weights_scale);
    if (moe_gating == 2 || moe_groups > 1)
      fprintf(stderr,
              "llama: MoE note: expert_gating_func=%u group_count=%u not "
              "modeled; using plain softmax top-%zu (exact for mixtral/qwen/"
              "olmoe; approximate for deepseek-v3/grok)\n",
              moe_gating, moe_groups, m->n_experts_used);
  }

  m->g = *g; /* take ownership */
  memset(g, 0, sizeof(*g));
  return 0;
}

void llama_free(LlamaModel* m) {
  for (size_t l = 0; m->layers && l < m->n_layers; ++l) {
    LlamaLayer* L = &m->layers[l];
    free(L->attn_norm);
    free(L->ffn_norm);
    free(L->attn_q_norm);
    free(L->attn_k_norm);
    free(L->bias_q);
    free(L->bias_k);
    free(L->bias_v);
    free(L->bias_o);
    free(L->ffn_gate_inp_shexp);
    free(L->k_cache);
    free(L->v_cache);
    free(L->k_scale);
    free(L->v_scale);
  }
  free(m->layers);
  free(m->out_norm);
  free(m->x);
  free(m->logits);
  free(m->normed);
  free(m->q);
  free(m->k);
  free(m->v);
  free(m->attn_res);
  free(m->attn_proj);
  free(m->gate);
  free(m->up);
  free(m->ffn_out);
  free(m->head_tmp);
  free(m->me_logits);
  free(m->me_w);
  free(m->me_sel);
  free(m->me_gate);
  free(m->me_up);
  free(m->me_eout);
  free(m->bx);
  free(m->bnormed);
  free(m->bq);
  free(m->bk);
  free(m->bv);
  free(m->battn);
  free(m->bproj);
  free(m->bgate);
  free(m->bup);
  free(m->bffn);
  oc_ctx_free(m->octx);
  gguf_close(&m->g);
  memset(m, 0, sizeof(*m));
}

/* Add an optional bias vector into a projection output (per token). NULL = skip. */
static inline void add_bias(float* y, const float* bias, size_t n) {
  if (!bias) return;
  for (size_t i = 0; i < n; ++i) y[i] += bias[i];
}

/* Per-head optional q/k RMSNorm (over head_dim) then RoPE. rope_norm selects
 * ggml NORMAL (adjacent-pair) rotation for llama/mistral/yi — whose llama.cpp
 * GGUFs ship q/k PERMUTED for exactly that — vs NeoX split-half for
 * qwen2/qwen3/phi3 (matching oxidize-cpp/forward.rs). Both reproduce HF rope;
 * only the q/k weight permutation, and hence the pairing, differs. */
static inline void norm_rope_head(float* p, const float* qk_norm, float* tmp,
                                  size_t hd, size_t pos, float theta,
                                  size_t rope_dim, float eps, bool rope_norm) {
  if (qk_norm) {
    oc_rms_norm(tmp, p, qk_norm, hd, eps);
    memcpy(p, tmp, hd * sizeof(float));
  }
  if (rope_norm)
    oc_rope_normal(p, hd, 1, pos, theta, rope_dim);
  else
    oc_rope(p, hd, 1, pos, theta, rope_dim, NULL);
}

/* Read cached K or V for (position t, kv-head kvh) as f32. f32 points straight
 * into the cache; f16/q8 decode one head into `buf`. `cache`/`scale` are the
 * layer's k_ or v_ pair, `hd` the head width (== v-head width for llama). */
static inline const float* decode_head(const uint8_t* cache, const float* scale,
                                       OcKvType kt, size_t kv_row, size_t hd,
                                       size_t n_kv_heads, size_t t, size_t kvh,
                                       float* buf) {
  if (kt == OC_KV_F32) return (const float*)cache + t * kv_row + kvh * hd;
  size_t elem = oc_kv_elem_bytes(kt);
  float sc = scale ? scale[t * n_kv_heads + kvh] : 0.0f;
  oc_kv_decode(kt, cache + (t * kv_row + kvh * hd) * elem, hd, sc, buf);
  return buf;
}

/* Commit position `pos`'s K and V (each n_kv_heads*head_dim values) into the
 * typed cache. f32 is a straight copy; f16/q8 encode per head (q8 also writes
 * the per-head scale). Store and decode_head are the one round-trip the batch
 * and decode paths share, so both observe the same rounded K/V. */
static void llama_store_kv(const LlamaModel* m, const LlamaLayer* L, size_t pos,
                           const float* k, const float* v) {
  size_t hd = m->head_dim, kv_row = m->n_kv_heads * hd;
  if (m->kv_type == OC_KV_F32) {
    memcpy((float*)L->k_cache + pos * kv_row, k, kv_row * sizeof(float));
    memcpy((float*)L->v_cache + pos * kv_row, v, kv_row * sizeof(float));
    return;
  }
  size_t elem = oc_kv_elem_bytes(m->kv_type);
  for (size_t h = 0; h < m->n_kv_heads; ++h) {
    size_t off = (pos * kv_row + h * hd) * elem, si = pos * m->n_kv_heads + h;
    oc_kv_encode(m->kv_type, k + h * hd, hd, L->k_cache + off,
                 L->k_scale ? &L->k_scale[si] : NULL);
    oc_kv_encode(m->kv_type, v + h * hd, hd, L->v_cache + off,
                 L->v_scale ? &L->v_scale[si] : NULL);
  }
}

/* ---- decode attention, online softmax over the cache, threaded over q heads
 * (full causal: token attends to [0, pos]). Mirrors qwen36 full attention. */
typedef struct {
  const LlamaLayer* L;
  const float* q;
  float* out;
  size_t n_head, n_kv_heads, head_dim, seq;
  OcKvType kv_type;
  float scale;
} AttnJob;

static void attn_heads(void* ctx, size_t h0, size_t h1) {
  AttnJob* j = ctx;
  size_t hd = j->head_dim, group = j->n_head / j->n_kv_heads;
  size_t kv_row = j->n_kv_heads * hd;
  const LlamaLayer* L = j->L;
  float kbuf[OC_KV_MAX_HEAD], vbuf[OC_KV_MAX_HEAD]; /* f16/q8 decode scratch */
  for (size_t h = h0; h < h1; ++h) {
    const float* qh = j->q + h * hd;
    float* oh = j->out + h * hd;
    size_t kvh = h / group;
    float rmax = -INFINITY, rsum = 0.0f;
    for (size_t d = 0; d < hd; ++d) oh[d] = 0.0f;
    for (size_t t = 0; t < j->seq; ++t) {
      const float* krow = decode_head(L->k_cache, L->k_scale, j->kv_type, kv_row,
                                      hd, j->n_kv_heads, t, kvh, kbuf);
      const float* vrow = decode_head(L->v_cache, L->v_scale, j->kv_type, kv_row,
                                      hd, j->n_kv_heads, t, kvh, vbuf);
      float score = oc_dot_f32(qh, krow, hd) * j->scale;
      float new_max = rmax > score ? rmax : score;
      float ef = expf(rmax - new_max), es = expf(score - new_max);
      if (ef != 1.0f)
        for (size_t d = 0; d < hd; ++d) oh[d] *= ef;
      for (size_t d = 0; d < hd; ++d) oh[d] += es * vrow[d];
      rsum = rsum * ef + es;
      rmax = new_max;
    }
    if (rsum > 0.0f) {
      float inv = 1.0f / rsum;
      for (size_t d = 0; d < hd; ++d) oh[d] *= inv;
    }
  }
}

/* ---- batched (prefill) attention. Cache is linear (no ring), so the batch's
 * K/V is committed before this runs and token i reads [0, pos0+i]. `t < seq` IS
 * the causal mask — widen it and a token attends to the future, silently. */
typedef struct {
  const LlamaLayer* L;
  const float* q; /* [n][n_head * head_dim] */
  float* out;     /* [n][n_head * head_dim] */
  size_t n_head, n_kv_heads, head_dim, pos0;
  OcKvType kv_type;
  float scale;
} AttnBatchJob;

static void attn_batch_heads(void* ctx, size_t i0, size_t i1) {
  AttnBatchJob* j = ctx;
  size_t hd = j->head_dim, group = j->n_head / j->n_kv_heads;
  size_t kv_row = j->n_kv_heads * hd, q_row = j->n_head * hd;
  const LlamaLayer* L = j->L;
  float kbuf[OC_KV_MAX_HEAD], vbuf[OC_KV_MAX_HEAD];
  for (size_t idx = i0; idx < i1; ++idx) {
    size_t i = idx / j->n_head, h = idx % j->n_head;
    size_t seq = j->pos0 + i + 1, kvh = h / group;
    const float* qh = j->q + i * q_row + h * hd;
    float* oh = j->out + i * q_row + h * hd;
    float rmax = -INFINITY, rsum = 0.0f;
    for (size_t d = 0; d < hd; ++d) oh[d] = 0.0f;
    for (size_t t = 0; t < seq; ++t) {
      const float* krow = decode_head(L->k_cache, L->k_scale, j->kv_type, kv_row,
                                      hd, j->n_kv_heads, t, kvh, kbuf);
      const float* vrow = decode_head(L->v_cache, L->v_scale, j->kv_type, kv_row,
                                      hd, j->n_kv_heads, t, kvh, vbuf);
      float score = oc_dot_f32(qh, krow, hd) * j->scale;
      float new_max = rmax > score ? rmax : score;
      float ef = expf(rmax - new_max), es = expf(score - new_max);
      if (ef != 1.0f)
        for (size_t d = 0; d < hd; ++d) oh[d] *= ef;
      for (size_t d = 0; d < hd; ++d) oh[d] += es * vrow[d];
      rsum = rsum * ef + es;
      rmax = new_max;
    }
    if (rsum > 0.0f) {
      float inv = 1.0f / rsum;
      for (size_t d = 0; d < hd; ++d) oh[d] *= inv;
    }
  }
}

/* MoE router — see model_llama.h. Softmax over the experts, then k argmax
 * passes for the top-k, then renormalize the selected weights (norm_topk) and
 * scale. k is tiny (2-8) so the O(k^2 n) skip-scan is cheaper than any heap. */
void llama_moe_route(const float* logits, size_t n, size_t k, bool norm_topk,
                     float scale, float* probs, int* sel, float* w) {
  float mx = -INFINITY;
  for (size_t e = 0; e < n; ++e)
    if (logits[e] > mx) mx = logits[e];
  float s = 0.0f;
  for (size_t e = 0; e < n; ++e) {
    probs[e] = expf(logits[e] - mx);
    s += probs[e];
  }
  if (s > 0.0f)
    for (size_t e = 0; e < n; ++e) probs[e] /= s;

  float wsum = 0.0f;
  for (size_t j = 0; j < k; ++j) {
    size_t best = 0;
    float bv = -INFINITY;
    for (size_t e = 0; e < n; ++e) {
      bool used = false;
      for (size_t t = 0; t < j; ++t)
        if (sel[t] == (int)e) { used = true; break; }
      if (!used && probs[e] > bv) { bv = probs[e]; best = e; }
    }
    sel[j] = (int)best;
    w[j] = probs[best];
    wsum += probs[best];
  }
  float norm = (norm_topk && wsum > 0.0f) ? wsum : 1.0f;
  for (size_t j = 0; j < k; ++j) w[j] = scale * w[j] / norm;
}

/* Accumulate w * SwiGLU_e(x) into dst[hidden] for one expert e of a stacked
 * [n_expert][rows][cols] tensor triple (expert e's matrix begins at
 * e*rows*oc_row_bytes(type, cols)). A shared/dense expert is just e=0 over a 2D
 * tensor, so this serves both. Uses m->me_gate/me_up/me_eout scratch. */
static void moe_expert_swiglu(LlamaModel* m, const GgufTensorInfo* gate,
                              const GgufTensorInfo* up, const GgufTensorInfo* down,
                              size_t e, size_t inter, const float* x, float w,
                              float* dst) {
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

/* MoE FFN for one token: router -> top-k -> sum of weighted expert SwiGLUs, plus
 * an always-on (optionally sigmoid-gated) shared expert. Writes the full FFN
 * output into out[hidden]; the caller adds it to the residual exactly like the
 * dense path's m->ffn_out.
 *
 * ponytail: correctness-first token-serial loop over experts via oc_matvec — the
 * batched-prefill path calls this once per token. The throughput upgrade is to
 * gather tokens per expert and issue ONE oc_matmul per selected expert (reads
 * each expert's weights from DRAM once per batch instead of once per token); it
 * is memory-bound the same as decode here, so prefill of an MoE layer costs the
 * same per token as decode until then. */
void llama_moe_ffn(LlamaModel* m, const LlamaLayer* L, const float* x, float* out) {
  const size_t h = m->hidden, k = m->n_experts_used;
  const float scale = m->expert_weights_scale > 0.0f ? m->expert_weights_scale : 1.0f;

  oc_matvec(m->octx, m->me_logits, L->ffn_gate_inp->ggml_type,
            L->ffn_gate_inp->data, m->n_experts, h, x);
  llama_moe_route(m->me_logits, m->n_experts, k, m->norm_topk_prob, scale,
                  m->me_logits, m->me_sel, m->me_w);

  for (size_t i = 0; i < h; ++i) out[i] = 0.0f;
  for (size_t j = 0; j < k; ++j)
    moe_expert_swiglu(m, L->ffn_gate_exps, L->ffn_up_exps, L->ffn_down_exps,
                      (size_t)m->me_sel[j], m->expert_inter, x, m->me_w[j], out);

  if (L->ffn_gate_shexp) {
    float sg = 1.0f; /* Qwen2-MoE gates the shared expert by sigmoid(gate . x). */
    if (L->ffn_gate_inp_shexp)
      sg = 1.0f / (1.0f + expf(-oc_dot_f32(L->ffn_gate_inp_shexp, x, h)));
    moe_expert_swiglu(m, L->ffn_gate_shexp, L->ffn_up_shexp, L->ffn_down_shexp,
                      0, m->shexp_inter, x, sg, out);
  }
}

float* llama_forward(LlamaModel* m, int32_t token, size_t pos, bool need_logits) {
  const size_t h = m->hidden;
  if (pos >= m->ctx) {
    fprintf(stderr, "llama: position %zu exceeds context %zu\n", pos, m->ctx);
    return NULL;
  }
  /* Embedding lookup (dequant one row). Llama does NOT scale embeddings. */
  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, h);
  oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * emb_row, m->x, h);
  return llama_forward_from(m, pos, 0, need_logits);
}

/* The layer loop + head of llama_forward, entered at layer l0 with m->x already
 * holding the hidden state for `pos`. l0 == 0 is what llama_forward runs after
 * the embedding; l0 > 0 is the CPU half of the CUDA -ngl partial offload. */
float* llama_forward_from(LlamaModel* m, size_t pos, size_t l0, bool need_logits) {
  const size_t h = m->hidden, hd = m->head_dim;
  const size_t q_len = m->n_head * hd, kv_row = m->n_kv_heads * hd;
  const float eps = m->eps, scale = 1.0f / sqrtf((float)hd);
  if (pos >= m->ctx) {
    fprintf(stderr, "llama: position %zu exceeds context %zu\n", pos, m->ctx);
    return NULL;
  }

  for (size_t l = l0; l < m->n_layers; ++l) {
    const LlamaLayer* L = &m->layers[l];

    /* ---- attention ---- */
    oc_rms_norm(m->normed, m->x, L->attn_norm, h, eps);
    oc_matvec(m->octx, m->q, L->attn_q->ggml_type, L->attn_q->data, q_len, h, m->normed);
    oc_matvec(m->octx, m->k, L->attn_k->ggml_type, L->attn_k->data, kv_row, h, m->normed);
    oc_matvec(m->octx, m->v, L->attn_v->ggml_type, L->attn_v->data, kv_row, h, m->normed);
    add_bias(m->q, L->bias_q, q_len);
    add_bias(m->k, L->bias_k, kv_row);
    add_bias(m->v, L->bias_v, kv_row);

    for (size_t hh = 0; hh < m->n_head; ++hh)
      norm_rope_head(m->q + hh * hd, L->attn_q_norm, m->head_tmp, hd, pos,
                     m->rope_theta, m->rope_dim, eps, m->rope_norm);
    for (size_t hh = 0; hh < m->n_kv_heads; ++hh)
      norm_rope_head(m->k + hh * hd, L->attn_k_norm, m->head_tmp, hd, pos,
                     m->rope_theta, m->rope_dim, eps, m->rope_norm);

    /* Full causal cache: unique slot per position, commit before attention. */
    llama_store_kv(m, L, pos, m->k, m->v);

    AttnJob job = {L,  m->q, m->attn_res, m->n_head, m->n_kv_heads,
                   hd, pos + 1, m->kv_type, scale};
    oc_parallel_for(m->n_head, attn_heads, &job);

    oc_matvec(m->octx, m->attn_proj, L->attn_out->ggml_type, L->attn_out->data, h,
              q_len, m->attn_res);
    add_bias(m->attn_proj, L->bias_o, h);
    for (size_t i = 0; i < h; ++i) m->x[i] += m->attn_proj[i];

    /* ---- FFN: dense SwiGLU or Mixture-of-Experts ---- */
    oc_rms_norm(m->normed, m->x, L->ffn_norm, h, eps);
    if (L->is_moe) {
      llama_moe_ffn(m, L, m->normed, m->ffn_out);
    } else {
      oc_matvec(m->octx, m->gate, L->ffn_gate->ggml_type, L->ffn_gate->data, m->inter, h, m->normed);
      oc_matvec(m->octx, m->up, L->ffn_up->ggml_type, L->ffn_up->data, m->inter, h, m->normed);
      for (size_t i = 0; i < m->inter; ++i) m->gate[i] = silu(m->gate[i]) * m->up[i];
      oc_matvec(m->octx, m->ffn_out, L->ffn_down->ggml_type, L->ffn_down->data, h, m->inter, m->gate);
    }
    for (size_t i = 0; i < h; ++i) m->x[i] += m->ffn_out[i];
  }
  m->kv_len = pos + 1;

  if (!need_logits) return NULL;
  oc_rms_norm(m->normed, m->x, m->out_norm, h, eps);
  oc_matvec(m->octx, m->logits, m->out_w->ggml_type, m->out_w->data, m->vocab, h, m->normed);
  return m->logits;
}

float* llama_forward_batch(LlamaModel* m, const int32_t* tokens, size_t n,
                           size_t pos0, bool need_logits) {
  const size_t h = m->hidden, hd = m->head_dim;
  const size_t q_len = m->n_head * hd, kv_row = m->n_kv_heads * hd;
  const float eps = m->eps, scale = 1.0f / sqrtf((float)hd);
  if (n == 0) return NULL;
  if (pos0 + n > m->ctx) {
    fprintf(stderr, "llama: batch [%zu,%zu) exceeds context %zu\n", pos0,
            pos0 + n, m->ctx);
    return NULL;
  }
  size_t bs = m->batch < m->batch_cap ? m->batch : m->batch_cap;
  if (bs < 1) bs = 1;
  if (n > bs) { /* one chunk at a time; logits come from the last */
    float* out = NULL;
    for (size_t i = 0; i < n; i += bs) {
      size_t c = n - i < bs ? n - i : bs;
      out = llama_forward_batch(m, tokens + i, c, pos0 + i, need_logits && i + c == n);
    }
    return out;
  }

  const size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, h);
  for (size_t i = 0; i < n; ++i) {
    size_t tk = (size_t)tokens[i] < m->vocab ? (size_t)tokens[i] : m->vocab - 1;
    oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * emb_row,
                   m->bx + i * h, h);
  }

  for (size_t l = 0; l < m->n_layers; ++l) {
    const LlamaLayer* L = &m->layers[l];

    for (size_t i = 0; i < n; ++i)
      oc_rms_norm(m->bnormed + i * h, m->bx + i * h, L->attn_norm, h, eps);
    oc_matmul(m->octx, m->bq, L->attn_q->ggml_type, L->attn_q->data, q_len, h, m->bnormed, n);
    oc_matmul(m->octx, m->bk, L->attn_k->ggml_type, L->attn_k->data, kv_row, h, m->bnormed, n);
    oc_matmul(m->octx, m->bv, L->attn_v->ggml_type, L->attn_v->data, kv_row, h, m->bnormed, n);

    for (size_t i = 0; i < n; ++i) {
      const size_t pos = pos0 + i;
      float* qi = m->bq + i * q_len;
      float* ki = m->bk + i * kv_row;
      float* vi = m->bv + i * kv_row;
      add_bias(qi, L->bias_q, q_len);
      add_bias(ki, L->bias_k, kv_row);
      add_bias(vi, L->bias_v, kv_row);
      for (size_t hh = 0; hh < m->n_head; ++hh)
        norm_rope_head(qi + hh * hd, L->attn_q_norm, m->head_tmp, hd, pos,
                       m->rope_theta, m->rope_dim, eps, m->rope_norm);
      for (size_t hh = 0; hh < m->n_kv_heads; ++hh)
        norm_rope_head(ki + hh * hd, L->attn_k_norm, m->head_tmp, hd, pos,
                       m->rope_theta, m->rope_dim, eps, m->rope_norm);
      /* Unique slots: safe to commit before attention. Attention then decodes
       * back from the typed cache, so in-batch tokens see the same rounded K/V
       * a sequential decode would — the batched==sequential invariant. */
      llama_store_kv(m, L, pos, ki, vi);
    }

    AttnBatchJob job = {L,  m->bq, m->battn, m->n_head, m->n_kv_heads,
                        hd, pos0, m->kv_type, scale};
    oc_parallel_for(n * m->n_head, attn_batch_heads, &job);

    oc_matmul(m->octx, m->bproj, L->attn_out->ggml_type, L->attn_out->data, h, q_len, m->battn, n);
    for (size_t i = 0; i < n; ++i) {
      float* p = m->bproj + i * h;
      add_bias(p, L->bias_o, h);
      const float* xi = m->bx + i * h;
      for (size_t d = 0; d < h; ++d) p[d] += xi[d];
    }

    if (L->is_moe) {
      /* ponytail: token-serial MoE reusing the decode scratch (m->normed /
       * m->ffn_out / m->me_*). Correct and identical to the decode path per
       * token; the gather-per-expert oc_matmul upgrade is noted in moe_ffn. */
      for (size_t i = 0; i < n; ++i) {
        const float* pj = m->bproj + i * h;
        oc_rms_norm(m->normed, pj, L->ffn_norm, h, eps);
        llama_moe_ffn(m, L, m->normed, m->ffn_out);
        float* xi = m->bx + i * h;
        for (size_t d = 0; d < h; ++d) xi[d] = pj[d] + m->ffn_out[d];
      }
    } else {
      for (size_t i = 0; i < n; ++i)
        oc_rms_norm(m->bnormed + i * h, m->bproj + i * h, L->ffn_norm, h, eps);
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
  /* Only the last token of a prompt is ever sampled from. */
  oc_rms_norm(m->normed, m->bx + (n - 1) * h, m->out_norm, h, eps);
  oc_matvec(m->octx, m->logits, m->out_w->ggml_type, m->out_w->data, m->vocab, h, m->normed);
  return m->logits;
}

/* See model_llama.h. Positionally-addressed cache, so dropping >= pos is pure
 * bookkeeping — attention reads only [0, pos] and the next store overwrites the
 * slot. Clamp so a caller cannot fast-forward kv_len past what was written. */
void llama_kv_rewind(LlamaModel* m, size_t pos) {
  if (pos < m->kv_len) m->kv_len = pos;
}
