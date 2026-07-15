#include "model_qwen36.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"
#include "tensor.h"

static void seterr(char* err, size_t n, const char* fmt, const char* a) {
  if (err && n) snprintf(err, n, fmt, a);
}

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

static const GgufTensorInfo* load_mat(const GgufFile* g, const char* name,
                                      char* err, size_t errlen) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  if (!t) {
    seterr(err, errlen, "qwen36: missing tensor %s", name);
    return NULL;
  }
  size_t cols = (size_t)t->dims[0];
  if (oc_row_bytes(t->ggml_type, cols) == 0) {
    if (err && errlen)
      snprintf(err, errlen, "qwen36: tensor %s has unsupported quant type %u",
               name, t->ggml_type);
    return NULL;
  }
  return t;
}

/* Optional matrix: NULL if absent or of an unsupported quant, no error set. */
static const GgufTensorInfo* opt_mat(const GgufFile* g, const char* name) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  if (!t || oc_row_bytes(t->ggml_type, (size_t)t->dims[0]) == 0) return NULL;
  return t;
}

/* OXC_PROF=1: per-section wall-time accumulators, printed at process exit. */
#include <time.h>
enum { P_QKV, P_SMALL, P_CONV, P_DELTA, P_GNORM, P_SSMOUT, P_ATTN, P_FFN, P_LOGITS, P_MISC, P_N };
static const char* prof_names[P_N] = {"lin.qkv+z", "lin.beta/alpha", "lin.conv",
    "lin.delta", "lin.gnorm", "lin.outproj", "full.attn", "ffn", "logits", "misc"};
static double prof_acc[P_N];
static int prof_on = -1;
static size_t prof_tokens;
static inline double prof_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void prof_dump(void) {
  double tot = 0;
  for (int i = 0; i < P_N; ++i) tot += prof_acc[i];
  fprintf(stderr, "prof: %zu tokens, %.1f ms/token\n", prof_tokens,
          prof_tokens ? tot * 1000.0 / (double)prof_tokens : 0.0);
  for (int i = 0; i < P_N; ++i)
    fprintf(stderr, "  %-15s %7.2f ms/tok  %5.1f%%\n", prof_names[i],
            prof_tokens ? prof_acc[i] * 1000.0 / (double)prof_tokens : 0.0,
            tot > 0 ? prof_acc[i] / tot * 100.0 : 0.0);
}
#define PROF_STEP(slot)                          \
  do {                                           \
    if (prof_on) {                               \
      double t_ = prof_now();                    \
      prof_acc[slot] += t_ - prof_t;             \
      prof_t = t_;                               \
    }                                            \
  } while (0)

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf_(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

/* Load the trailing NextN/MTP draft block (blk.N.*, N == m->n_layers). Returns
 * 0 and sets m->has_mtp on success; -1 (block left unusable — caller warns and
 * disables the draft) if a required tensor is absent. The optional embed / head
 * tensors, when absent, fall back to the main model's at forward time. Requires
 * all main-model geometry (n_layers, hidden, vocab, head dims) already set. */
static int qwen36_load_mtp(Qwen36Model* m, const GgufFile* g) {
  size_t N = m->n_layers;
  Qwen36Mtp* x = &m->mtp;
  char nm[128];
#define NX_MAT(field, suffix)                          \
  do {                                                 \
    snprintf(nm, sizeof nm, "blk.%zu." suffix, N);     \
    x->field = opt_mat(g, nm);                         \
    if (!x->field) return -1;                          \
  } while (0)
#define NX_VEC(field, suffix)                          \
  do {                                                 \
    snprintf(nm, sizeof nm, "blk.%zu." suffix, N);     \
    x->field = load_vec(g, nm, NULL);                  \
    if (!x->field) return -1;                          \
  } while (0)
  NX_MAT(eh_proj, "nextn.eh_proj.weight");
  NX_VEC(enorm, "nextn.enorm.weight");
  NX_VEC(hnorm, "nextn.hnorm.weight");
  NX_VEC(attn_norm, "attn_norm.weight");
  NX_MAT(attn_q, "attn_q.weight");
  NX_MAT(attn_k, "attn_k.weight");
  NX_MAT(attn_v, "attn_v.weight");
  NX_MAT(attn_out, "attn_output.weight");
  NX_VEC(attn_q_norm, "attn_q_norm.weight");
  NX_VEC(attn_k_norm, "attn_k_norm.weight");
  NX_MAT(ffn_gate, "ffn_gate.weight");
  NX_MAT(ffn_up, "ffn_up.weight");
  NX_MAT(ffn_down, "ffn_down.weight");
#undef NX_MAT
#undef NX_VEC
  snprintf(nm, sizeof nm, "blk.%zu.post_attention_norm.weight", N);
  x->post_attn_norm = load_vec(g, nm, NULL);
  if (!x->post_attn_norm) {
    snprintf(nm, sizeof nm, "blk.%zu.ffn_norm.weight", N);
    x->post_attn_norm = load_vec(g, nm, NULL);
  }
  if (!x->post_attn_norm) return -1;
  snprintf(nm, sizeof nm, "blk.%zu.nextn.embed_tokens.weight", N);
  x->embed_tokens = opt_mat(g, nm); /* optional */
  snprintf(nm, sizeof nm, "blk.%zu.nextn.shared_head_norm.weight", N);
  x->shared_head_norm = load_vec(g, nm, NULL); /* optional */
  snprintf(nm, sizeof nm, "blk.%zu.nextn.shared_head_head.weight", N);
  x->shared_head_head = opt_mat(g, nm); /* optional */
  m->has_mtp = true;
  return 0;
}

int qwen36_load(Qwen36Model* m, GgufFile* g, size_t max_ctx, char* err,
                size_t errlen) {
  memset(m, 0, sizeof(*m));
  char* arch = gguf_architecture(g);
  if (!arch || strcmp(arch, "qwen35") != 0) {
    if (err && errlen)
      snprintf(err, errlen, "unsupported architecture '%s' (qwen36 path wants qwen35)",
               arch ? arch : "(none)");
    free(arch);
    return -1;
  }
  free(arch);

  uint32_t u;
  float f;
  m->hidden = gguf_get_u32(g, "qwen35.embedding_length", &u) ? u : 5120;
  size_t block_count = gguf_get_u32(g, "qwen35.block_count", &u) ? u : 65;
  size_t nextn = gguf_get_u32(g, "qwen35.nextn_predict_layers", &u) ? u : 0;
  m->n_layers = block_count - nextn; /* MTP block(s) at the end are skipped */
  m->n_head = gguf_get_u32(g, "qwen35.attention.head_count", &u) ? u : 24;
  m->n_kv_heads = gguf_get_u32(g, "qwen35.attention.head_count_kv", &u) ? u : 4;
  m->head_dim = gguf_get_u32(g, "qwen35.attention.key_length", &u) ? u : 256;
  m->inter = gguf_get_u32(g, "qwen35.feed_forward_length", &u) ? u : 17408;
  m->ctx = gguf_get_u32(g, "qwen35.context_length", &u) ? u : 4096;
  m->eps = gguf_get_f32(g, "qwen35.attention.layer_norm_rms_epsilon", &f) ? f : 1e-6f;
  m->rope_theta = gguf_get_f32(g, "qwen35.rope.freq_base", &f) ? f : 1e7f;
  m->rope_dim = gguf_get_u32(g, "qwen35.rope.dimension_count", &u) ? u : 64;
  m->d_conv = gguf_get_u32(g, "qwen35.ssm.conv_kernel", &u) ? u : 4;
  m->d_state = gguf_get_u32(g, "qwen35.ssm.state_size", &u) ? u : 128;
  m->n_k_heads = gguf_get_u32(g, "qwen35.ssm.group_count", &u) ? u : 16;
  m->n_v_heads = gguf_get_u32(g, "qwen35.ssm.time_step_rank", &u) ? u : 48;
  size_t d_inner = gguf_get_u32(g, "qwen35.ssm.inner_size", &u) ? u : 6144;
  m->head_v_dim = d_inner / m->n_v_heads;
  m->key_dim = m->d_state * m->n_k_heads;
  m->value_dim = m->head_v_dim * m->n_v_heads;
  m->conv_dim = m->key_dim * 2 + m->value_dim;
  size_t full_interval = gguf_get_u32(g, "qwen35.full_attention_interval", &u) ? u : 4;
  if (max_ctx > 0 && max_ctx < m->ctx) m->ctx = max_ctx;

  m->tok_embd = load_mat(g, "token_embd.weight", err, errlen);
  if (!m->tok_embd) return -1;
  m->vocab = (size_t)m->tok_embd->dims[1];
  m->out_w = load_mat(g, "output.weight", err, errlen);
  if (!m->out_w) m->out_w = m->tok_embd; /* tied fallback */
  m->out_norm = load_vec(g, "output_norm.weight", NULL);
  if (!m->out_norm) {
    seterr(err, errlen, "qwen36: missing tensor %s", "output_norm.weight");
    return -1;
  }

  m->layers = calloc(m->n_layers, sizeof(Qwen36Layer));
  if (!m->layers) return -1;

  size_t n_lin = 0, n_full = 0;
  for (size_t l = 0; l < m->n_layers; ++l) {
    Qwen36Layer* L = &m->layers[l];
    L->is_linear = ((l + 1) % full_interval != 0);
    char name[128];

#define MAT(field, suffix)                               \
  do {                                                   \
    snprintf(name, sizeof(name), "blk.%zu." suffix, l); \
    L->field = load_mat(g, name, err, errlen);           \
    if (!L->field) return -1;                            \
  } while (0)
#define VEC(field, suffix)                               \
  do {                                                   \
    snprintf(name, sizeof(name), "blk.%zu." suffix, l); \
    L->field = load_vec(g, name, NULL);                  \
    if (!L->field) {                                     \
      seterr(err, errlen, "qwen36: missing %s", name);   \
      return -1;                                         \
    }                                                    \
  } while (0)

    VEC(attn_norm, "attn_norm.weight");
    VEC(post_attn_norm, "post_attention_norm.weight");
    MAT(ffn_gate, "ffn_gate.weight");
    MAT(ffn_up, "ffn_up.weight");
    MAT(ffn_down, "ffn_down.weight");

    if (L->is_linear) {
      n_lin++;
      MAT(wqkv, "attn_qkv.weight");
      MAT(wgate, "attn_gate.weight");
      MAT(ssm_out, "ssm_out.weight");
      MAT(beta_w, "ssm_beta.weight");
      MAT(alpha_w, "ssm_alpha.weight");
      VEC(dt_bias, "ssm_dt.bias");
      VEC(ssm_a, "ssm_a");
      VEC(ssm_norm, "ssm_norm.weight");
      VEC(conv_w, "ssm_conv1d.weight"); /* [conv_dim rows][d_conv] f32 */
      L->conv_state = calloc((m->d_conv - 1) * m->conv_dim, sizeof(float));
      L->S = calloc(m->n_v_heads * m->head_v_dim * m->d_state, sizeof(float));
      if (!L->conv_state || !L->S) return -1;
    } else {
      n_full++;
      MAT(attn_q, "attn_q.weight");
      MAT(attn_k, "attn_k.weight");
      MAT(attn_v, "attn_v.weight");
      MAT(attn_out, "attn_output.weight");
      VEC(attn_q_norm, "attn_q_norm.weight");
      VEC(attn_k_norm, "attn_k_norm.weight");
      L->k_cache = calloc(m->ctx * m->n_kv_heads * m->head_dim, sizeof(float));
      L->v_cache = calloc(m->ctx * m->n_kv_heads * m->head_dim, sizeof(float));
      if (!L->k_cache || !L->v_cache) {
        seterr(err, errlen, "qwen36: KV cache allocation failed%s", "");
        return -1;
      }
    }
#undef MAT
#undef VEC
  }
  fprintf(stderr,
          "qwen36: %zu layers (%zu linear + %zu full), hidden=%zu heads=%zu/%zu "
          "head_dim=%zu rope_dim=%zu v_heads=%zu k_heads=%zu state=%zu ctx=%zu\n",
          m->n_layers, n_lin, n_full, m->hidden, m->n_head, m->n_kv_heads,
          m->head_dim, m->rope_dim, m->n_v_heads, m->n_k_heads, m->d_state, m->ctx);

  size_t qg = m->n_head * m->head_dim * 2; /* q+gate interleaved */
  m->x = calloc(m->hidden, sizeof(float));
  m->logits = calloc(m->vocab, sizeof(float));
  m->normed = calloc(m->hidden, sizeof(float));
  m->q = calloc(qg, sizeof(float));
  m->k = calloc(m->n_kv_heads * m->head_dim, sizeof(float));
  m->v = calloc(m->n_kv_heads * m->head_dim, sizeof(float));
  m->attn_res = calloc(m->n_head * m->head_dim, sizeof(float));
  m->qpack = calloc(m->n_head * m->head_dim, sizeof(float));
  m->attn_proj = calloc(m->hidden, sizeof(float));
  m->gate = calloc(m->inter, sizeof(float));
  m->up = calloc(m->inter, sizeof(float));
  m->ffn_out = calloc(m->hidden, sizeof(float));
  m->qkv = calloc(m->conv_dim, sizeof(float));
  m->z = calloc(m->value_dim, sizeof(float));
  m->conv_out = calloc(m->conv_dim, sizeof(float));
  m->o_lin = calloc(m->value_dim, sizeof(float));
  if (!m->x || !m->logits || !m->normed || !m->q || !m->k || !m->v ||
      !m->attn_res || !m->qpack || !m->attn_proj || !m->gate || !m->up ||
      !m->ffn_out || !m->qkv || !m->z || !m->conv_out || !m->o_lin) {
    seterr(err, errlen, "qwen36: scratch allocation failed%s", "");
    return -1;
  }

  /* Batched-prefill scratch: the same vectors with a row per token. */
  {
    const char* e = getenv("OC_BATCH");
    long b = e ? atol(e) : 32;
    m->batch = (size_t)(b < 1 ? 1 : b > 512 ? 512 : b);
  }
  size_t B = m->batch, kvd = m->n_kv_heads * m->head_dim, qd = m->n_head * m->head_dim;
  m->batch_cap = B; /* scratch below is sized to B; chunks must never exceed it */
  m->octx = oc_ctx_new();
  m->bx = calloc(B * m->hidden, sizeof(float));
  m->bnormed = calloc(B * m->hidden, sizeof(float));
  m->bq = calloc(B * qg, sizeof(float));
  m->bqpack = calloc(B * qd, sizeof(float));
  m->bk = calloc(B * kvd, sizeof(float));
  m->bv = calloc(B * kvd, sizeof(float));
  m->battn = calloc(B * qd, sizeof(float));
  m->bproj = calloc(B * m->hidden, sizeof(float));
  m->bgate = calloc(B * m->inter, sizeof(float));
  m->bup = calloc(B * m->inter, sizeof(float));
  m->bffn = calloc(B * m->hidden, sizeof(float));
  m->bqkv = calloc(B * m->conv_dim, sizeof(float));
  m->bz = calloc(B * m->value_dim, sizeof(float));
  m->bconv = calloc(B * m->conv_dim, sizeof(float));
  m->bolin = calloc(B * m->value_dim, sizeof(float));
  m->bbeta = calloc(B * m->n_v_heads, sizeof(float));
  m->bgdec = calloc(B * m->n_v_heads, sizeof(float));
  if (!m->octx || !m->bx || !m->bnormed || !m->bq || !m->bqpack || !m->bk ||
      !m->bv || !m->battn || !m->bproj || !m->bgate || !m->bup || !m->bffn ||
      !m->bqkv || !m->bz || !m->bconv || !m->bolin || !m->bbeta || !m->bgdec) {
    seterr(err, errlen, "qwen36: batch scratch allocation failed%s", "");
    return -1;
  }

  /* NextN/MTP draft head (P16): load it if the GGUF advertises one, then size
   * its scratch, its small draft KV cache, and the per-linear-layer recurrent-
   * state snapshots the speculative rollback needs. A malformed/partial block
   * disables the draft (loud warning) rather than failing the whole load. */
  if (nextn > 0 && qwen36_load_mtp(m, g) != 0)
    fprintf(stderr, "qwen36: nextn_predict_layers=%zu but blk.%zu MTP tensors "
            "are incomplete; speculative decode disabled\n", nextn, m->n_layers);
  if (m->has_mtp) {
    m->mtp.cap = m->batch_cap;
    m->mtp.k_cache = calloc(m->mtp.cap * kvd, sizeof(float));
    m->mtp.v_cache = calloc(m->mtp.cap * kvd, sizeof(float));
    m->mtp_concat = calloc(2 * m->hidden, sizeof(float));
    m->mtp_prev = calloc(m->hidden, sizeof(float));
    m->mtp_logits = calloc(m->vocab, sizeof(float));
    bool ok = m->mtp.k_cache && m->mtp.v_cache && m->mtp_concat && m->mtp_prev &&
              m->mtp_logits;
    for (size_t l = 0; l < m->n_layers && ok; ++l) {
      Qwen36Layer* L = &m->layers[l];
      if (!L->is_linear) continue;
      L->conv_snap = calloc((m->d_conv - 1) * m->conv_dim, sizeof(float));
      L->S_snap = calloc(m->n_v_heads * m->head_v_dim * m->d_state, sizeof(float));
      ok = L->conv_snap && L->S_snap;
    }
    if (!ok) {
      seterr(err, errlen, "qwen36: MTP scratch allocation failed%s", "");
      return -1;
    }
    fprintf(stderr, "qwen36: MTP/nextn draft head loaded (speculative decode "
            "available; --spec)\n");
  }

  m->g = *g;
  memset(g, 0, sizeof(*g));
  return 0;
}

void qwen36_free(Qwen36Model* m) {
  for (size_t l = 0; m->layers && l < m->n_layers; ++l) {
    Qwen36Layer* L = &m->layers[l];
    free(L->attn_norm);
    free(L->post_attn_norm);
    free(L->attn_q_norm);
    free(L->attn_k_norm);
    free(L->k_cache);
    free(L->v_cache);
    free(L->dt_bias);
    free(L->ssm_a);
    free(L->conv_w);
    free(L->ssm_norm);
    free(L->conv_state);
    free(L->S);
    free(L->conv_snap);
    free(L->S_snap);
  }
  free(m->layers);
  free(m->mtp.enorm);
  free(m->mtp.hnorm);
  free(m->mtp.attn_norm);
  free(m->mtp.post_attn_norm);
  free(m->mtp.attn_q_norm);
  free(m->mtp.attn_k_norm);
  free(m->mtp.shared_head_norm);
  free(m->mtp.k_cache);
  free(m->mtp.v_cache);
  free(m->mtp_concat);
  free(m->mtp_prev);
  free(m->mtp_logits);
  free(m->spec_logits);
  free(m->out_norm);
  free(m->x);
  free(m->logits);
  free(m->normed);
  free(m->q);
  free(m->k);
  free(m->v);
  free(m->attn_res);
  free(m->qpack);
  free(m->attn_proj);
  free(m->gate);
  free(m->up);
  free(m->ffn_out);
  free(m->qkv);
  free(m->z);
  free(m->conv_out);
  free(m->o_lin);
  free(m->bx);
  free(m->bnormed);
  free(m->bq);
  free(m->bqpack);
  free(m->bk);
  free(m->bv);
  free(m->battn);
  free(m->bproj);
  free(m->bgate);
  free(m->bup);
  free(m->bffn);
  free(m->bqkv);
  free(m->bz);
  free(m->bconv);
  free(m->bolin);
  free(m->bbeta);
  free(m->bgdec);
  oc_ctx_free(m->octx);
  gguf_close(&m->g);
  memset(m, 0, sizeof(*m));
}

/* depthwise conv4 + SiLU, threaded over channels */
typedef struct {
  const Qwen36Model* m;
  Qwen36Layer* L;
} ConvJob;

static void conv_channels(void* ctx, size_t c0, size_t c1) {
  ConvJob* j = ctx;
  const Qwen36Model* m = j->m;
  size_t k = m->d_conv;
  for (size_t c = c0; c < c1; ++c) {
    const float* w = j->L->conv_w + c * k;
    float* st = j->L->conv_state + c * (k - 1);
    float acc = w[k - 1] * m->qkv[c];
    for (size_t i = 0; i + 1 < k; ++i) acc += w[i] * st[i];
    memmove(st, st + 1, (k - 2) * sizeof(float));
    st[k - 2] = m->qkv[c];
    m->conv_out[c] = acc / (1.0f + expf(-acc)); /* silu */
  }
}

/* ---- gated delta net decode step, threaded over v-heads ---- */
typedef struct {
  const Qwen36Model* m;
  Qwen36Layer* L;
  const float* qh;   /* [n_k_heads * d_state], L2-normed, scaled */
  const float* kh;   /* [n_k_heads * d_state], L2-normed */
  const float* vh;   /* [n_v_heads * head_v_dim] */
  const float* g_dec; /* [n_v_heads] exp(gate) */
  const float* beta;  /* [n_v_heads] */
  float* out;         /* [n_v_heads * head_v_dim] */
} DeltaJob;

static void delta_heads(void* ctx, size_t h0, size_t h1) {
  DeltaJob* j = ctx;
  const Qwen36Model* m = j->m;
  size_t dk = m->d_state, dv = m->head_v_dim;
  /* k/q head broadcast matches ggml_repeat_4d tiling: v-head h -> k-head
   * h % n_k_heads (NOT grouped h/3). */
  for (size_t h = h0; h < h1; ++h) {
    const float* q = j->qh + (h % m->n_k_heads) * dk;
    const float* k = j->kh + (h % m->n_k_heads) * dk;
    const float* v = j->vh + h * dv;
    float* S = j->L->S + h * dv * dk; /* S[i][j]: v-dim i, k-dim j */
    float* o = j->out + h * dv;
    float dec = j->g_dec[h], b = j->beta[h];
    for (size_t i = 0; i < dv; ++i) {
      float* Si = S + i * dk;
      float sk = 0.0f;
      for (size_t jj = 0; jj < dk; ++jj) {
        Si[jj] *= dec;
        sk += Si[jj] * k[jj];
      }
      float d = (v[i] - sk) * b;
      float oi = 0.0f;
      for (size_t jj = 0; jj < dk; ++jj) {
        Si[jj] += d * k[jj];
        oi += Si[jj] * q[jj];
      }
      o[i] = oi;
    }
  }
}

/* ---- full attention, threaded over q heads (mirrors gemma4) ---- */
typedef struct {
  const Qwen36Model* m;
  const Qwen36Layer* L;
  const float* q; /* [n_head * head_dim] (gate stripped) */
  float* out;
  size_t seq;
  float scale;
} AttnJob;

static void attn_heads(void* ctx, size_t h0, size_t h1) {
  AttnJob* j = ctx;
  const Qwen36Model* m = j->m;
  const Qwen36Layer* L = j->L;
  size_t hd = m->head_dim;
  size_t group = m->n_head / m->n_kv_heads;
  size_t kv_row = m->n_kv_heads * hd;
  for (size_t h = h0; h < h1; ++h) {
    const float* qh = j->q + h * hd;
    float* oh = j->out + h * hd;
    size_t kvh = h / group;
    float running_max = -INFINITY, running_sum = 0.0f;
    for (size_t d = 0; d < hd; ++d) oh[d] = 0.0f;
    for (size_t t = 0; t < j->seq; ++t) {
      const float* krow = L->k_cache + t * kv_row + kvh * hd;
      float score = oc_dot_f32(qh, krow, hd) * j->scale;
      float new_max = running_max > score ? running_max : score;
      float ef = expf(running_max - new_max);
      float es = expf(score - new_max);
      if (ef != 1.0f)
        for (size_t d = 0; d < hd; ++d) oh[d] *= ef;
      const float* vrow = L->v_cache + t * kv_row + kvh * hd;
      for (size_t d = 0; d < hd; ++d) oh[d] += es * vrow[d];
      running_sum = running_sum * ef + es;
      running_max = new_max;
    }
    if (running_sum > 0.0f) {
      float inv = 1.0f / running_sum;
      for (size_t d = 0; d < hd; ++d) oh[d] *= inv;
    }
  }
}

static void l2_norm_head(float* p, size_t n, float eps) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) sum += p[i] * p[i];
  float inv = 1.0f / sqrtf(sum > eps ? sum : eps);
  for (size_t i = 0; i < n; ++i) p[i] *= inv;
}

/* ---- batched (prefill) jobs -------------------------------------------------
 * The two recurrent kernels do NOT parallelize over tokens — the conv window
 * and the delta state S are carried forward token by token. What IS independent
 * is the channel (conv) and the v-head (delta), so both thread over that axis
 * and SCAN the batch inside. Same arithmetic, same order, same state after. */

typedef struct {
  const Qwen36Model* m;
  Qwen36Layer* L;
  const float* qkv; /* [n][conv_dim] */
  float* out;       /* [n][conv_dim] */
  size_t n;
} ConvBatchJob;

static void conv_batch_channels(void* ctx, size_t c0, size_t c1) {
  ConvBatchJob* j = ctx;
  const Qwen36Model* m = j->m;
  size_t k = m->d_conv, cd = m->conv_dim;
  for (size_t c = c0; c < c1; ++c) {
    const float* w = j->L->conv_w + c * k;
    float* st = j->L->conv_state + c * (k - 1);
    for (size_t i = 0; i < j->n; ++i) {
      float xv = j->qkv[i * cd + c];
      float acc = w[k - 1] * xv;
      for (size_t z = 0; z + 1 < k; ++z) acc += w[z] * st[z];
      memmove(st, st + 1, (k - 2) * sizeof(float));
      st[k - 2] = xv;
      j->out[i * cd + c] = acc / (1.0f + expf(-acc)); /* silu */
    }
  }
}

typedef struct {
  const Qwen36Model* m;
  Qwen36Layer* L;
  const float* cout;  /* [n][conv_dim]: q | k | v, L2-normed and scaled */
  const float* g_dec; /* [n][n_v_heads] */
  const float* beta;  /* [n][n_v_heads] */
  float* out;         /* [n][value_dim] */
  size_t n;
} DeltaBatchJob;

static void delta_batch_heads(void* ctx, size_t h0, size_t h1) {
  DeltaBatchJob* j = ctx;
  const Qwen36Model* m = j->m;
  size_t dk = m->d_state, dv = m->head_v_dim;
  size_t cd = m->conv_dim, kd = m->key_dim, vdim = m->value_dim, nv = m->n_v_heads;
  for (size_t hh = h0; hh < h1; ++hh) {
    float* S = j->L->S + hh * dv * dk;
    size_t kh = (hh % m->n_k_heads) * dk; /* ggml_repeat_4d tiling, not h/group */
    for (size_t i = 0; i < j->n; ++i) {
      const float* q = j->cout + i * cd + kh;
      const float* k = j->cout + i * cd + kd + kh;
      const float* v = j->cout + i * cd + 2 * kd + hh * dv;
      float* o = j->out + i * vdim + hh * dv;
      float dec = j->g_dec[i * nv + hh], b = j->beta[i * nv + hh];
      for (size_t d = 0; d < dv; ++d) {
        float* Si = S + d * dk;
        float sk = 0.0f;
        for (size_t z = 0; z < dk; ++z) {
          Si[z] *= dec;
          sk += Si[z] * k[z];
        }
        float delta = (v[d] - sk) * b;
        float oi = 0.0f;
        for (size_t z = 0; z < dk; ++z) {
          Si[z] += delta * k[z];
          oi += Si[z] * q[z];
        }
        o[d] = oi;
      }
    }
  }
}

/* Full attention over the batch, one (token, head) pair per index. The cache is
 * linear here (no ring), so the batch's K/V is already written when this runs
 * and token i simply reads [0, pos0+i]. `t < seq` IS the causal mask. */
typedef struct {
  const Qwen36Model* m;
  const Qwen36Layer* L;
  const float* q; /* [n][n_head * head_dim], gates stripped */
  float* out;     /* [n][n_head * head_dim] */
  size_t n_head, pos0;
  float scale;
} AttnBatchJob;

static void attn_batch_heads(void* ctx, size_t i0, size_t i1) {
  AttnBatchJob* j = ctx;
  const Qwen36Model* m = j->m;
  const Qwen36Layer* L = j->L;
  size_t hd = m->head_dim;
  size_t group = m->n_head / m->n_kv_heads;
  size_t kv_row = m->n_kv_heads * hd, q_row = j->n_head * hd;
  for (size_t idx = i0; idx < i1; ++idx) {
    size_t i = idx / j->n_head, h = idx % j->n_head;
    size_t seq = j->pos0 + i + 1, kvh = h / group;
    const float* qh = j->q + i * q_row + h * hd;
    float* oh = j->out + i * q_row + h * hd;
    float rmax = -INFINITY, rsum = 0.0f;
    for (size_t d = 0; d < hd; ++d) oh[d] = 0.0f;
    for (size_t t = 0; t < seq; ++t) {
      const float* krow = L->k_cache + t * kv_row + kvh * hd;
      const float* vrow = L->v_cache + t * kv_row + kvh * hd;
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

float* qwen36_forward(Qwen36Model* m, int32_t token, size_t pos, bool need_logits) {
  const size_t h = m->hidden;
  const float eps = m->eps;
  float head_tmp[512];
  if (prof_on < 0) {
    prof_on = getenv("OXC_PROF") != NULL;
    if (prof_on) atexit(prof_dump);
  }
  double prof_t = prof_on ? prof_now() : 0.0;
  prof_tokens++;

  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, h);
  oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * emb_row, m->x, h);

  for (size_t l = 0; l < m->n_layers; ++l) {
    Qwen36Layer* L = &m->layers[l];
    oc_rms_norm(m->normed, m->x, L->attn_norm, h, eps);

    if (L->is_linear) {
      /* projections */
      oc_matvec(m->octx, m->qkv, L->wqkv->ggml_type, L->wqkv->data, m->conv_dim, h, m->normed);
      oc_matvec(m->octx, m->z, L->wgate->ggml_type, L->wgate->data, m->value_dim, h, m->normed);
      PROF_STEP(P_QKV);
      float beta[64], gdec[64]; /* n_v_heads <= 64 */
      /* 48-row f32 matvecs: serial beats a 96-thread pool rendezvous */
      for (size_t r = 0; r < m->n_v_heads; ++r) {
        size_t rb = oc_row_bytes(L->beta_w->ggml_type, h);
        beta[r] = oc_dot_row(L->beta_w->ggml_type, L->beta_w->data + r * rb, m->normed, h);
        gdec[r] = oc_dot_row(L->alpha_w->ggml_type, L->alpha_w->data + r * rb, m->normed, h);
      }
      PROF_STEP(P_SMALL);
      for (size_t i = 0; i < m->n_v_heads; ++i) {
        beta[i] = sigmoidf_(beta[i]);
        /* gate = -exp(A_log) * softplus(alpha + dt_bias); ssm_a already negative */
        gdec[i] = expf(L->ssm_a[i] * softplusf_(gdec[i] + L->dt_bias[i]));
      }

      /* depthwise conv4 + SiLU over [conv_state | qkv], then advance state */
      ConvJob cj = {m, L};
      oc_parallel_for(m->conv_dim, conv_channels, &cj);
      PROF_STEP(P_CONV);

      float* qc = m->conv_out;                 /* [n_k_heads * d_state] */
      float* kc = m->conv_out + m->key_dim;    /* [n_k_heads * d_state] */
      float* vc = m->conv_out + 2 * m->key_dim;/* [n_v_heads * head_v_dim] */
      float qscale = 1.0f / sqrtf((float)m->d_state);
      for (size_t hh = 0; hh < m->n_k_heads; ++hh) {
        l2_norm_head(qc + hh * m->d_state, m->d_state, eps);
        l2_norm_head(kc + hh * m->d_state, m->d_state, eps);
      }
      for (size_t i = 0; i < m->key_dim; ++i) qc[i] *= qscale;

      DeltaJob dj = {m, L, qc, kc, vc, gdec, beta, m->o_lin};
      oc_parallel_for(m->n_v_heads, delta_heads, &dj);
      PROF_STEP(P_DELTA);

      /* gated RMSNorm per head: rms(o) * silu(z), then output projection */
      for (size_t hh = 0; hh < m->n_v_heads; ++hh) {
        float* o = m->o_lin + hh * m->head_v_dim;
        oc_rms_norm(head_tmp, o, L->ssm_norm, m->head_v_dim, eps);
        const float* zh = m->z + hh * m->head_v_dim;
        for (size_t i = 0; i < m->head_v_dim; ++i) o[i] = head_tmp[i] * silu(zh[i]);
      }
      PROF_STEP(P_GNORM);
      oc_matvec(m->octx, m->attn_proj, L->ssm_out->ggml_type, L->ssm_out->data, h,
                m->value_dim, m->o_lin);
      PROF_STEP(P_SSMOUT);
    } else {
      /* full attention: q proj emits [q(hd) | gate(hd)] per head */
      if (pos >= m->ctx) {
        fprintf(stderr, "qwen36: position %zu exceeds context %zu\n", pos, m->ctx);
        return NULL;
      }
      size_t hd = m->head_dim;
      oc_matvec(m->octx, m->q, L->attn_q->ggml_type, L->attn_q->data, m->n_head * hd * 2,
                h, m->normed);
      oc_matvec(m->octx, m->k, L->attn_k->ggml_type, L->attn_k->data, m->n_kv_heads * hd,
                h, m->normed);
      oc_matvec(m->octx, m->v, L->attn_v->ggml_type, L->attn_v->data, m->n_kv_heads * hd,
                h, m->normed);
      for (size_t hh = 0; hh < m->n_head; ++hh) {
        float* p = m->q + hh * hd * 2;
        oc_rms_norm(head_tmp, p, L->attn_q_norm, hd, eps);
        memcpy(p, head_tmp, hd * sizeof(float));
        oc_rope(p, hd, 1, pos, m->rope_theta, m->rope_dim, NULL);
      }
      for (size_t hh = 0; hh < m->n_kv_heads; ++hh) {
        float* p = m->k + hh * hd;
        oc_rms_norm(head_tmp, p, L->attn_k_norm, hd, eps);
        memcpy(p, head_tmp, hd * sizeof(float));
        oc_rope(p, hd, 1, pos, m->rope_theta, m->rope_dim, NULL);
      }
      size_t kv_row = m->n_kv_heads * hd;
      memcpy(L->k_cache + pos * kv_row, m->k, kv_row * sizeof(float));
      memcpy(L->v_cache + pos * kv_row, m->v, kv_row * sizeof(float));

      /* Pack q, stripping the interleaved gates. This used to borrow m->o_lin,
       * which is value_dim floats — fine only because Qwen3.5 happens to have
       * value_dim == n_head*head_dim. Any geometry where it does not (and the
       * batched==sequential test uses one) overflowed the heap by the
       * difference. Use a buffer that is the right size by construction. */
      for (size_t hh = 0; hh < m->n_head; ++hh)
        memcpy(m->qpack + hh * hd, m->q + hh * hd * 2, hd * sizeof(float));
      AttnJob job = {m, L, m->qpack, m->attn_res, pos + 1,
                     1.0f / sqrtf((float)hd)};
      oc_parallel_for(m->n_head, attn_heads, &job);

      /* per-head output gate: sigmoid(gate half of q proj) */
      for (size_t hh = 0; hh < m->n_head; ++hh) {
        const float* gt = m->q + hh * hd * 2 + hd;
        float* oh = m->attn_res + hh * hd;
        for (size_t d = 0; d < hd; ++d) oh[d] *= sigmoidf_(gt[d]);
      }
      oc_matvec(m->octx, m->attn_proj, L->attn_out->ggml_type, L->attn_out->data, h,
                m->n_head * hd, m->attn_res);
      PROF_STEP(P_ATTN);
    }

    for (size_t i = 0; i < h; ++i) m->x[i] += m->attn_proj[i];

    /* FFN (SiLU swiglu) with pre-norm; residual from pre-norm x */
    oc_rms_norm(m->normed, m->x, L->post_attn_norm, h, eps);
    oc_matvec(m->octx, m->gate, L->ffn_gate->ggml_type, L->ffn_gate->data, m->inter, h, m->normed);
    oc_matvec(m->octx, m->up, L->ffn_up->ggml_type, L->ffn_up->data, m->inter, h, m->normed);
    for (size_t i = 0; i < m->inter; ++i) m->gate[i] = silu(m->gate[i]) * m->up[i];
    oc_matvec(m->octx, m->ffn_out, L->ffn_down->ggml_type, L->ffn_down->data, h, m->inter, m->gate);
    for (size_t i = 0; i < h; ++i) m->x[i] += m->ffn_out[i];
    PROF_STEP(P_FFN);
  }

  if (!need_logits) return NULL;
  oc_rms_norm(m->normed, m->x, m->out_norm, h, eps);
  oc_matvec(m->octx, m->logits, m->out_w->ggml_type, m->out_w->data, m->vocab, h, m->normed);
  PROF_STEP(P_LOGITS);
  return m->logits;
}

/* Run the whole layer stack over n tokens (single chunk, n <= batch_cap),
 * leaving the final residual for every row in m->bx. Shared by
 * qwen36_forward_batch (last-row head) and the speculative verify (all-row
 * heads). Same graph as qwen36_forward with a token axis: projections and the
 * FFN become oc_matmul over the batch; the DeltaNet conv/state and the full-
 * attention cache are scanned per token. */
static void qwen36_run_stack(Qwen36Model* m, const int32_t* tokens, size_t n,
                             size_t pos0) {
  const size_t h = m->hidden;
  const float eps = m->eps;
  const size_t hd = m->head_dim, nv = m->n_v_heads, dv = m->head_v_dim;
  const size_t cd = m->conv_dim, kd = m->key_dim, vdim = m->value_dim;
  const size_t qg = m->n_head * hd * 2, qd = m->n_head * hd, kvd = m->n_kv_heads * hd;
  const size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, h);
  float head_tmp[512];

  for (size_t i = 0; i < n; ++i) {
    size_t tk = (size_t)tokens[i] < m->vocab ? (size_t)tokens[i] : m->vocab - 1;
    oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * emb_row,
                   m->bx + i * h, h);
  }

  for (size_t l = 0; l < m->n_layers; ++l) {
    Qwen36Layer* L = &m->layers[l];
    for (size_t i = 0; i < n; ++i)
      oc_rms_norm(m->bnormed + i * h, m->bx + i * h, L->attn_norm, h, eps);

    if (L->is_linear) {
      oc_matmul(m->octx, m->bqkv, L->wqkv->ggml_type, L->wqkv->data, cd, h,
                m->bnormed, n);
      oc_matmul(m->octx, m->bz, L->wgate->ggml_type, L->wgate->data, vdim, h,
                m->bnormed, n);
      oc_matmul(m->octx, m->bbeta, L->beta_w->ggml_type, L->beta_w->data, nv, h,
                m->bnormed, n);
      oc_matmul(m->octx, m->bgdec, L->alpha_w->ggml_type, L->alpha_w->data, nv, h,
                m->bnormed, n);
      for (size_t i = 0; i < n * nv; ++i) {
        size_t r = i % nv;
        m->bbeta[i] = sigmoidf_(m->bbeta[i]);
        /* gate = -exp(A_log) * softplus(alpha + dt_bias); ssm_a already negative */
        m->bgdec[i] = expf(L->ssm_a[r] * softplusf_(m->bgdec[i] + L->dt_bias[r]));
      }

      ConvBatchJob cj = {m, L, m->bqkv, m->bconv, n};
      oc_parallel_for(cd, conv_batch_channels, &cj);

      float qscale = 1.0f / sqrtf((float)m->d_state);
      for (size_t i = 0; i < n; ++i) {
        float* qc = m->bconv + i * cd;
        float* kc = qc + kd;
        for (size_t hh = 0; hh < m->n_k_heads; ++hh) {
          l2_norm_head(qc + hh * m->d_state, m->d_state, eps);
          l2_norm_head(kc + hh * m->d_state, m->d_state, eps);
        }
        for (size_t d = 0; d < kd; ++d) qc[d] *= qscale;
      }

      DeltaBatchJob dj = {m, L, m->bconv, m->bgdec, m->bbeta, m->bolin, n};
      oc_parallel_for(nv, delta_batch_heads, &dj);

      for (size_t i = 0; i < n; ++i)
        for (size_t hh = 0; hh < nv; ++hh) {
          float* o = m->bolin + i * vdim + hh * dv;
          oc_rms_norm(head_tmp, o, L->ssm_norm, dv, eps);
          const float* zh = m->bz + i * vdim + hh * dv;
          for (size_t d = 0; d < dv; ++d) o[d] = head_tmp[d] * silu(zh[d]);
        }
      oc_matmul(m->octx, m->bproj, L->ssm_out->ggml_type, L->ssm_out->data, h,
                vdim, m->bolin, n);
    } else {
      oc_matmul(m->octx, m->bq, L->attn_q->ggml_type, L->attn_q->data, qg, h,
                m->bnormed, n);
      oc_matmul(m->octx, m->bk, L->attn_k->ggml_type, L->attn_k->data, kvd, h,
                m->bnormed, n);
      oc_matmul(m->octx, m->bv, L->attn_v->ggml_type, L->attn_v->data, kvd, h,
                m->bnormed, n);

      for (size_t i = 0; i < n; ++i) {
        size_t pos = pos0 + i;
        for (size_t hh = 0; hh < m->n_head; ++hh) {
          float* p = m->bq + i * qg + hh * hd * 2;
          oc_rms_norm(head_tmp, p, L->attn_q_norm, hd, eps);
          memcpy(p, head_tmp, hd * sizeof(float));
          oc_rope(p, hd, 1, pos, m->rope_theta, m->rope_dim, NULL);
          memcpy(m->bqpack + i * qd + hh * hd, p, hd * sizeof(float));
        }
        for (size_t hh = 0; hh < m->n_kv_heads; ++hh) {
          float* p = m->bk + i * kvd + hh * hd;
          oc_rms_norm(head_tmp, p, L->attn_k_norm, hd, eps);
          memcpy(p, head_tmp, hd * sizeof(float));
          oc_rope(p, hd, 1, pos, m->rope_theta, m->rope_dim, NULL);
        }
        /* Linear cache, unique slots: safe to write before attention. */
        memcpy(L->k_cache + pos * kvd, m->bk + i * kvd, kvd * sizeof(float));
        memcpy(L->v_cache + pos * kvd, m->bv + i * kvd, kvd * sizeof(float));
      }

      AttnBatchJob job = {m, L, m->bqpack, m->battn, m->n_head, pos0,
                          1.0f / sqrtf((float)hd)};
      oc_parallel_for(n * m->n_head, attn_batch_heads, &job);

      for (size_t i = 0; i < n; ++i) /* per-head output gate */
        for (size_t hh = 0; hh < m->n_head; ++hh) {
          const float* gt = m->bq + i * qg + hh * hd * 2 + hd;
          float* oh = m->battn + i * qd + hh * hd;
          for (size_t d = 0; d < hd; ++d) oh[d] *= sigmoidf_(gt[d]);
        }
      oc_matmul(m->octx, m->bproj, L->attn_out->ggml_type, L->attn_out->data, h,
                qd, m->battn, n);
    }

    for (size_t i = 0; i < n * h; ++i) m->bx[i] += m->bproj[i];

    for (size_t i = 0; i < n; ++i)
      oc_rms_norm(m->bnormed + i * h, m->bx + i * h, L->post_attn_norm, h, eps);
    oc_matmul(m->octx, m->bgate, L->ffn_gate->ggml_type, L->ffn_gate->data,
              m->inter, h, m->bnormed, n);
    oc_matmul(m->octx, m->bup, L->ffn_up->ggml_type, L->ffn_up->data, m->inter, h,
              m->bnormed, n);
    for (size_t i = 0; i < n * m->inter; ++i)
      m->bgate[i] = silu(m->bgate[i]) * m->bup[i];
    oc_matmul(m->octx, m->bffn, L->ffn_down->ggml_type, L->ffn_down->data, h,
              m->inter, m->bgate, n);
    for (size_t i = 0; i < n * h; ++i) m->bx[i] += m->bffn[i];
  }

}

/* Prefill / batched decode: n tokens at pos0.., returning the LAST row's logits.
 * Numerically equal to n sequential qwen36_forward calls (the test asserts it).
 * A long batch is chunked to batch_cap; the recurrent state carries across
 * chunks. */
float* qwen36_forward_batch(Qwen36Model* m, const int32_t* tokens, size_t n,
                            size_t pos0, bool need_logits) {
  const size_t h = m->hidden;
  const float eps = m->eps;
  if (n == 0) return NULL;
  if (pos0 + n > m->ctx) {
    fprintf(stderr, "qwen36: batch [%zu,%zu) exceeds context %zu\n", pos0,
            pos0 + n, m->ctx);
    return NULL;
  }
  size_t bs = m->batch < m->batch_cap ? m->batch : m->batch_cap; /* never over cap */
  if (bs < 1) bs = 1;
  if (n > bs) {
    float* out = NULL;
    for (size_t i = 0; i < n; i += bs) {
      size_t c = n - i < bs ? n - i : bs;
      out = qwen36_forward_batch(m, tokens + i, c, pos0 + i,
                                 need_logits && i + c == n);
    }
    return out;
  }
  qwen36_run_stack(m, tokens, n, pos0);
  if (!need_logits) return NULL;
  oc_rms_norm(m->normed, m->bx + (n - 1) * h, m->out_norm, h, eps);
  oc_matvec(m->octx, m->logits, m->out_w->ggml_type, m->out_w->data, m->vocab, h,
            m->normed);
  return m->logits;
}

/* ==== Speculative decoding: native NextN/MTP draft head (P16) ================
 *
 * Structure follows oxidize-core/src/model/inference/mtp.rs (draft head) and
 * .../model/generation.rs run_mtp_step (verify+commit). The greedy acceptance
 * is bit-exact to plain autoregressive decode because every emitted token is
 * the TARGET's own argmax at a position it forwarded with the correct prefix;
 * the draft only ever changes how MANY tokens one target batch commits. */

static int32_t argmax_f32(const float* v, size_t n) {
  size_t best = 0;
  for (size_t i = 1; i < n; ++i)
    if (v[i] > v[best]) best = i; /* first-max wins, matching greedy sampling */
  return (int32_t)best;
}

/* Dequantize the embedding row of `token` (MTP's own table if present, else the
 * shared token_embd) into out[hidden]. */
static void mtp_embed(Qwen36Model* m, int32_t token, float* out) {
  const GgufTensorInfo* e = m->mtp.embed_tokens ? m->mtp.embed_tokens : m->tok_embd;
  size_t h = m->hidden;
  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t rb = oc_row_bytes(e->ggml_type, h);
  oc_dequant_row(e->ggml_type, e->data + tk * rb, out, h);
}

/* One MTP draft step: fuse embed(token) with the previous hidden (m->mtp_prev),
 * run the gated full-attention block over its own mini-KV at position mtp_pos,
 * apply the shared head, and return the argmax draft token. Updates m->mtp_prev
 * to this step's post-head-norm hidden so the next step chains from it. Uses the
 * shared single-token scratch (m->x/q/k/v/qpack/attn_res/gate/up/ffn_out/normed)
 * — safe because no target forward is in flight during a draft burst. */
static int32_t mtp_forward_one(Qwen36Model* m, int32_t token, size_t mtp_pos) {
  const size_t h = m->hidden, hd = m->head_dim;
  const size_t kvd = m->n_kv_heads * hd;
  const float eps = m->eps;
  Qwen36Mtp* x = &m->mtp;
  float head_tmp[512];

  /* fusion: concat = [ enorm(embed(token)) | hnorm(prev_hidden) ] -> eh_proj */
  mtp_embed(m, token, m->mtp_concat);
  oc_rms_norm(m->mtp_concat, m->mtp_concat, x->enorm, h, eps);
  oc_rms_norm(m->mtp_concat + h, m->mtp_prev, x->hnorm, h, eps);
  oc_matvec(m->octx, m->x, x->eh_proj->ggml_type, x->eh_proj->data, h, 2 * h,
            m->mtp_concat);

  /* gated full attention (mirrors the qwen35 full-layer path) */
  oc_rms_norm(m->normed, m->x, x->attn_norm, h, eps);
  oc_matvec(m->octx, m->q, x->attn_q->ggml_type, x->attn_q->data, m->n_head * hd * 2,
            h, m->normed);
  oc_matvec(m->octx, m->k, x->attn_k->ggml_type, x->attn_k->data, kvd, h, m->normed);
  oc_matvec(m->octx, m->v, x->attn_v->ggml_type, x->attn_v->data, kvd, h, m->normed);
  for (size_t hh = 0; hh < m->n_head; ++hh) {
    float* p = m->q + hh * hd * 2;
    oc_rms_norm(head_tmp, p, x->attn_q_norm, hd, eps);
    memcpy(p, head_tmp, hd * sizeof(float));
    oc_rope(p, hd, 1, mtp_pos, m->rope_theta, m->rope_dim, NULL);
    memcpy(m->qpack + hh * hd, p, hd * sizeof(float));
  }
  for (size_t hh = 0; hh < m->n_kv_heads; ++hh) {
    float* p = m->k + hh * hd;
    oc_rms_norm(head_tmp, p, x->attn_k_norm, hd, eps);
    memcpy(p, head_tmp, hd * sizeof(float));
    oc_rope(p, hd, 1, mtp_pos, m->rope_theta, m->rope_dim, NULL);
  }
  memcpy(x->k_cache + mtp_pos * kvd, m->k, kvd * sizeof(float));
  memcpy(x->v_cache + mtp_pos * kvd, m->v, kvd * sizeof(float));
  Qwen36Layer kv = {0}; /* borrow attn_heads with the MTP mini-KV */
  kv.k_cache = x->k_cache;
  kv.v_cache = x->v_cache;
  AttnJob job = {m, &kv, m->qpack, m->attn_res, mtp_pos + 1, 1.0f / sqrtf((float)hd)};
  oc_parallel_for(m->n_head, attn_heads, &job);
  for (size_t hh = 0; hh < m->n_head; ++hh) {
    const float* gt = m->q + hh * hd * 2 + hd;
    float* oh = m->attn_res + hh * hd;
    for (size_t d = 0; d < hd; ++d) oh[d] *= sigmoidf_(gt[d]);
  }
  oc_matvec(m->octx, m->attn_proj, x->attn_out->ggml_type, x->attn_out->data, h,
            m->n_head * hd, m->attn_res);
  for (size_t i = 0; i < h; ++i) m->x[i] += m->attn_proj[i];

  /* SwiGLU FFN */
  oc_rms_norm(m->normed, m->x, x->post_attn_norm, h, eps);
  oc_matvec(m->octx, m->gate, x->ffn_gate->ggml_type, x->ffn_gate->data, m->inter, h,
            m->normed);
  oc_matvec(m->octx, m->up, x->ffn_up->ggml_type, x->ffn_up->data, m->inter, h,
            m->normed);
  for (size_t i = 0; i < m->inter; ++i) m->gate[i] = silu(m->gate[i]) * m->up[i];
  oc_matvec(m->octx, m->ffn_out, x->ffn_down->ggml_type, x->ffn_down->data, h,
            m->inter, m->gate);
  for (size_t i = 0; i < h; ++i) m->x[i] += m->ffn_out[i];

  /* shared head: mtp_hidden = norm(x); logits = head @ mtp_hidden. mtp_hidden is
   * written into m->mtp_prev so it seeds the next draft step. */
  const float* norm_w = x->shared_head_norm ? x->shared_head_norm : m->out_norm;
  oc_rms_norm(m->mtp_prev, m->x, norm_w, h, eps);
  const GgufTensorInfo* head = x->shared_head_head ? x->shared_head_head : m->out_w;
  oc_matvec(m->octx, m->mtp_logits, head->ggml_type, head->data, m->vocab, h,
            m->mtp_prev);
  return argmax_f32(m->mtp_logits, m->vocab);
}

void qwen36_mtp_draft(Qwen36Model* m, int32_t seed_token, int32_t* draft, size_t k) {
  if (!m->has_mtp) {
    for (size_t i = 0; i < k; ++i) draft[i] = -1;
    return;
  }
  /* The draft mini-KV holds mtp.cap (== batch_cap) rows and mtp_forward_one writes
   * row i, so never draft past it. Internal callers already clamp to batch_cap-1,
   * but this public API must not overflow its own cache if a caller ignores the
   * documented precondition; the surplus slots become -1 (a guess that can only be
   * rejected) — the same sentinel the !has_mtp path uses. */
  size_t kd = k > m->mtp.cap ? m->mtp.cap : k;
  for (size_t i = kd; i < k; ++i) draft[i] = -1;
  /* Each burst is a fresh mini-sequence at MTP positions 0..kd-1; the seed hidden
   * is the target's out-norm hidden at the last committed position (m->normed). */
  memcpy(m->mtp_prev, m->normed, m->hidden * sizeof(float));
  int32_t tok = seed_token;
  for (size_t i = 0; i < kd; ++i) {
    draft[i] = mtp_forward_one(m, tok, i);
    tok = draft[i];
  }
}

void qwen36_state_snapshot(Qwen36Model* m) {
  size_t cn = (m->d_conv - 1) * m->conv_dim;
  size_t sn = m->n_v_heads * m->head_v_dim * m->d_state;
  for (size_t l = 0; l < m->n_layers; ++l) {
    Qwen36Layer* L = &m->layers[l];
    if (!L->is_linear || !L->conv_snap || !L->S_snap) continue;
    memcpy(L->conv_snap, L->conv_state, cn * sizeof(float));
    memcpy(L->S_snap, L->S, sn * sizeof(float));
  }
}

void qwen36_state_restore(Qwen36Model* m) {
  size_t cn = (m->d_conv - 1) * m->conv_dim;
  size_t sn = m->n_v_heads * m->head_v_dim * m->d_state;
  for (size_t l = 0; l < m->n_layers; ++l) {
    Qwen36Layer* L = &m->layers[l];
    if (!L->is_linear || !L->conv_snap || !L->S_snap) continue;
    memcpy(L->conv_state, L->conv_snap, cn * sizeof(float));
    memcpy(L->S, L->S_snap, sn * sizeof(float));
  }
}

/* Run the layer stack over the drafts and emit per-row logits (dist for the
 * token AFTER each fed position) into out_logits[row * vocab]. Clobbers
 * m->bx/m->normed — the caller recomputes committed state afterwards. */
static void qwen36_verify(Qwen36Model* m, const int32_t* tokens, size_t n,
                          size_t pos0, float* out_logits) {
  const size_t h = m->hidden;
  const float eps = m->eps;
  qwen36_run_stack(m, tokens, n, pos0);
  for (size_t r = 0; r < n; ++r) {
    oc_rms_norm(m->normed, m->bx + r * h, m->out_norm, h, eps);
    oc_matvec(m->octx, out_logits + r * m->vocab, m->out_w->ggml_type,
              m->out_w->data, m->vocab, h, m->normed);
  }
}

size_t qwen36_spec_step(Qwen36Model* m, const int32_t* draft, size_t k, size_t pos,
                        int32_t* out, size_t* accepted) {
  if (accepted) *accepted = 0;
  if (!m->has_mtp || k == 0) return 0;
  if (k + 1 > m->batch_cap)                          /* verify k + commit k+1 rows */
    k = m->batch_cap > 1 ? m->batch_cap - 1 : 0;
  if (k == 0) return 0;               /* batch_cap too small for even one draft */
  if (pos + k + 1 > m->ctx) return 0; /* caller falls back near ctx end */
  if (m->spec_cap < k) {
    free(m->spec_logits);
    m->spec_logits = malloc(k * m->vocab * sizeof(float));
    m->spec_cap = m->spec_logits ? k : 0;
    if (!m->spec_logits) return 0;
  }
  const size_t V = m->vocab;

  /* target's committed next token @ pos, from the pending logits (saved before
   * the verification batch overwrites m->logits). */
  int32_t g0 = argmax_f32(m->logits, V);

  qwen36_state_snapshot(m);                     /* DeltaNet state at pos */
  qwen36_verify(m, draft, k, pos, m->spec_logits); /* row i = dist @ (pos+1+i) */

  /* Greedy acceptance: draft[i] is kept iff it equals the target argmax at the
   * position it occupies (target_logits[0]=pending, [i>=1]=verify row i-1). */
  size_t a = 0;
  bool rejected = false;
  for (size_t step = 0; step < k && !rejected; ++step) {
    int32_t t = step == 0 ? g0 : argmax_f32(m->spec_logits + (step - 1) * V, V);
    if (draft[step] == t) {
      out[step] = draft[step];
      a = step + 1;
    } else {
      out[step] = t; /* correction */
      rejected = true;
    }
  }
  size_t len;
  if (rejected) {
    len = a + 1; /* out[a] holds the correction */
  } else {
    out[k] = argmax_f32(m->spec_logits + (k - 1) * V, V); /* bonus continuation */
    len = k + 1;
  }

  /* Roll the recurrent state back to `pos` and re-forward ONLY the committed
   * tokens; this rewrites the full-attention KV at pos..pos+len-1 (positional,
   * so no explicit KV rewind) and leaves m->logits/m->normed describing the
   * position after the last emitted token, ready for the next step.
   * ponytail: re-forwards the accepted prefix too (== the reference); a
   * per-layer recurrent-state checkpoint could skip that if decode is state-
   * bound rather than weight-bound. */
  qwen36_state_restore(m);
  qwen36_forward_batch(m, out, len, pos, true);
  if (accepted) *accepted = a;
  return len;
}
