#include "model_gemma4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"
#include "tensor.h"

static void seterr(char* err, size_t n, const char* fmt, const char* a) {
  if (err && n) snprintf(err, n, fmt, a);
}

/* Dequantize a whole (1-D norm) tensor into a fresh f32 buffer. */
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

/* Weight matrix tensor by name; validates that we have a kernel for its type. */
static const GgufTensorInfo* load_mat(const GgufFile* g, const char* name,
                                      char* err, size_t errlen) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  if (!t) {
    seterr(err, errlen, "gemma4: missing tensor %s", name);
    return NULL;
  }
  size_t cols = (size_t)t->dims[0];
  if (oc_row_bytes(t->ggml_type, cols) == 0) {
    if (err && errlen)
      snprintf(err, errlen, "gemma4: tensor %s has unsupported quant type %u",
               name, t->ggml_type);
    return NULL;
  }
  return t;
}

static uint32_t arr_u32(const GgufValue* a, size_t i) {
  const GgufValue* e = &a->v.arr.items[i];
  switch (e->kind) {
    case GGUF_T_I8:
    case GGUF_T_I16:
    case GGUF_T_I32:
    case GGUF_T_I64:
      return e->v.i >= 0 ? (uint32_t)e->v.i : 0;
    case GGUF_T_F32:
    case GGUF_T_F64:
      return (uint32_t)e->v.f;
    default:
      return (uint32_t)e->v.u;
  }
}

int gemma4_load(Gemma4Model* m, GgufFile* g, size_t max_ctx, bool kv_quant,
                char* err, size_t errlen) {
  memset(m, 0, sizeof(*m));
  /* KV precision: the bool param forces the rotated int4 rotoquant (kept for the
   * CUDA backend and the direct-call test), otherwise take the process-wide type
   * the CLI set. Eligibility (power-of-two head dims for q4; head_dim <= 512 for
   * f16/q8) is checked once geometry is known and may downgrade to f32 below. */
  m->kv_type = kv_quant ? OC_KV_Q4 : oc_kv_get_type();
  m->kv_len = 0;
  char* arch = gguf_architecture(g);
  if (!arch || strcmp(arch, "gemma4") != 0) {
    if (err && errlen)
      snprintf(err, errlen, "unsupported architecture '%s' (only gemma4)",
               arch ? arch : "(none)");
    free(arch);
    return -1;
  }
  free(arch);

  uint32_t u;
  float f;
  m->hidden = gguf_get_u32(g, "gemma4.embedding_length", &u) ? u : 5376;
  m->n_layers = gguf_get_u32(g, "gemma4.block_count", &u) ? u : 60;
  m->n_head = gguf_get_u32(g, "gemma4.attention.head_count", &u) ? u : 32;
  m->inter = gguf_get_u32(g, "gemma4.feed_forward_length", &u) ? u : 21504;
  m->ctx = gguf_get_u32(g, "gemma4.context_length", &u) ? u : 4096;
  m->window = gguf_get_u32(g, "gemma4.attention.sliding_window", &u) ? u : 1024;
  m->eps = gguf_get_f32(g, "gemma4.attention.layer_norm_rms_epsilon", &f) ? f : 1e-6f;
  m->final_softcap =
      gguf_get_f32(g, "gemma4.final_logit_softcapping", &f) ? f : 30.0f;
  /* Reference (diffusion_gemma.rs): attention scale is 1.0 unless overridden. */
  m->attn_scale = gguf_get_f32(g, "gemma4.attention.scale", &f) ? f : 1.0f;
  if (max_ctx > 0 && max_ctx < m->ctx) m->ctx = max_ctx;
  m->emb_scale = sqrtf((float)m->hidden);

  m->tok_embd = load_mat(g, "token_embd.weight", err, errlen);
  if (!m->tok_embd) return -1;
  m->vocab = (size_t)m->tok_embd->dims[1];
  if (gguf_get_u32(g, "gemma4.vocab_size", &u)) m->vocab = u;

  m->out_norm = load_vec(g, "output_norm.weight", NULL);
  if (!m->out_norm) {
    seterr(err, errlen, "gemma4: missing tensor %s", "output_norm.weight");
    return -1;
  }

  /* Per-layer geometry KVs. */
  float theta_full = gguf_get_f32(g, "gemma4.rope.freq_base", &f) ? f : 1e6f;
  float theta_swa = gguf_get_f32(g, "gemma4.rope.freq_base_swa", &f) ? f : 1e4f;
  uint32_t rope_dim_full =
      gguf_get_u32(g, "gemma4.rope.dimension_count", &u) ? u : 0;
  uint32_t rope_dim_swa =
      gguf_get_u32(g, "gemma4.rope.dimension_count_swa", &u) ? u : 0;
  float prf = 0.0f; /* partial rotary factor (0.25 on full layers per HF cfg) */
  bool have_prf = gguf_get_f32(g, "gemma4.rope.partial_rotary_factor", &f) && (prf = f, f > 0.0f);
  const GgufValue* swa_pat = gguf_get_arr(g, "gemma4.attention.sliding_window_pattern");
  uint32_t swa_pat_scalar = 0;
  if (!swa_pat) {
    if (!gguf_get_u32(g, "gemma4.attention.sliding_window_pattern", &swa_pat_scalar))
      swa_pat_scalar = 6; /* 5 SWA : 1 full */
  }
  const GgufValue* kv_heads_arr = gguf_get_arr(g, "gemma4.attention.head_count_kv");
  uint32_t kv_heads_scalar = 0;
  if (!kv_heads_arr)
    gguf_get_u32(g, "gemma4.attention.head_count_kv", &kv_heads_scalar);

  m->layers = calloc(m->n_layers, sizeof(Gemma4Layer));
  if (!m->layers) return -1;

  size_t max_q = 0, max_k = 0, max_hd = 0;
  bool logged[2] = {false, false};
  for (size_t l = 0; l < m->n_layers; ++l) {
    Gemma4Layer* L = &m->layers[l];
    char name[128];

    if (swa_pat && l < swa_pat->v.arr.n)
      L->is_swa = arr_u32(swa_pat, l) != 0;
    else
      L->is_swa = swa_pat_scalar == 0 ? false : ((l + 1) % swa_pat_scalar != 0);

    L->n_kv_heads = kv_heads_arr && l < kv_heads_arr->v.arr.n
                        ? arr_u32(kv_heads_arr, l)
                        : (kv_heads_scalar ? kv_heads_scalar : m->n_head);
    if (L->n_kv_heads == 0 || m->n_head % L->n_kv_heads != 0) {
      seterr(err, errlen, "gemma4: bad kv head count on layer %s", name);
      return -1;
    }

#define MAT(field, suffix)                                       \
  do {                                                           \
    snprintf(name, sizeof(name), "blk.%zu." suffix, l);         \
    L->field = load_mat(g, name, err, errlen);                   \
    if (!L->field) return -1;                                    \
  } while (0)
#define VEC(field, suffix)                                       \
  do {                                                           \
    snprintf(name, sizeof(name), "blk.%zu." suffix, l);         \
    L->field = load_vec(g, name, NULL);                          \
  } while (0)

    MAT(attn_q, "attn_q.weight");
    MAT(attn_k, "attn_k.weight");
    /* attn_v is ABSENT on global layers (attention_k_eq_v: K and V share one
     * projection). Optional; error only if present with an unsupported quant. */
    snprintf(name, sizeof(name), "blk.%zu.attn_v.weight", l);
    if (gguf_tensor(g, name)) {
      L->attn_v = load_mat(g, name, err, errlen);
      if (!L->attn_v) return -1;
    }
    MAT(attn_out, "attn_output.weight");
    MAT(ffn_gate, "ffn_gate.weight");
    MAT(ffn_up, "ffn_up.weight");
    MAT(ffn_down, "ffn_down.weight");
    VEC(attn_norm, "attn_norm.weight");
    VEC(attn_q_norm, "attn_q_norm.weight");
    VEC(attn_k_norm, "attn_k_norm.weight");
    VEC(post_attn_norm, "post_attention_norm.weight");
    VEC(ffn_norm, "ffn_norm.weight");
    VEC(post_ffn_norm, "post_ffw_norm.weight");
    if (!L->attn_norm || !L->ffn_norm) {
      seterr(err, errlen, "gemma4: layer missing attn_norm/ffn_norm: %s", name);
      return -1;
    }
#undef MAT
#undef VEC

    /* Scalar residual scale (blk.N.layer_output_scale.weight), default 1. */
    L->output_scale = 1.0f;
    {
      char sn[128];
      snprintf(sn, sizeof(sn), "blk.%zu.layer_output_scale.weight", l);
      size_t n1 = 0;
      float* s = load_vec(g, sn, &n1);
      if (s) {
        if (n1 >= 1) L->output_scale = s[0];
        free(s);
        if (getenv("OXC_DEBUG"))
          fprintf(stderr, "dbg l=%zu layer_output_scale=%.6g\n", l,
                  (double)L->output_scale);
      }
    }

    /* Geometry from tensor shapes (they win over KV assumptions). */
    size_t q_rows = (size_t)L->attn_q->dims[1];
    size_t k_rows = (size_t)L->attn_k->dims[1];
    size_t v_rows = L->attn_v ? (size_t)L->attn_v->dims[1] : k_rows; /* K=V */
    L->head_dim = q_rows / m->n_head;
    size_t k_head_dim = k_rows / L->n_kv_heads;
    L->v_head_dim = v_rows / L->n_kv_heads;
    if (L->head_dim == 0 || L->head_dim != k_head_dim) {
      fprintf(stderr,
              "gemma4: layer %zu: q head_dim %zu != k head_dim %zu "
              "(q_rows=%zu k_rows=%zu kv_heads=%zu)\n",
              l, L->head_dim, k_head_dim, q_rows, k_rows, L->n_kv_heads);
      seterr(err, errlen, "gemma4: inconsistent head dims on layer %s", name);
      return -1;
    }

    /* Per-kind RoPE config: dimension_count = dims rotated per head (clamped
     * to head_dim); partial_rotary_factor overrides when present. */
    L->rope.theta = L->is_swa ? theta_swa : theta_full;
    uint32_t rd = L->is_swa ? rope_dim_swa : rope_dim_full;
    L->rope.rope_dim = rd > 0 && (size_t)rd < L->head_dim ? rd : 0; /* 0 = full */
    if (!L->is_swa && have_prf) {
      size_t d = (size_t)((float)L->head_dim * prf);
      d &= ~(size_t)1;
      if (d > 0 && d < L->head_dim) L->rope.rope_dim = d;
    }

    /* KV cache: ring of `window` positions on SWA layers, full ctx on global.
     * Allocated after the loop, once kv_quant eligibility is known. */
    L->cache_cap = L->is_swa && m->window > 0 && m->window < m->ctx ? m->window : m->ctx;

    if (!logged[L->is_swa ? 1 : 0]) {
      logged[L->is_swa ? 1 : 0] = true;
      fprintf(stderr,
              "gemma4: %s layer geometry: kv_heads=%zu head_dim=%zu "
              "v_head_dim=%zu rope_theta=%.0f rope_dim=%zu cache=%zu\n",
              L->is_swa ? "SWA" : "global", L->n_kv_heads, L->head_dim,
              L->v_head_dim, (double)L->rope.theta,
              L->rope.rope_dim ? L->rope.rope_dim : L->head_dim, L->cache_cap);
    }
    if (q_rows > max_q) max_q = q_rows;
    if (k_rows > max_k) max_k = k_rows;
    if (v_rows > max_k) max_k = v_rows;
    if (m->n_head * L->v_head_dim > max_q) max_q = m->n_head * L->v_head_dim;
    if (L->head_dim > max_hd) max_hd = L->head_dim;
    if (L->v_head_dim > max_hd) max_hd = L->v_head_dim;
  }

  /* KV cache eligibility. The rotoquant needs power-of-two head dims (the FHT)
   * and a bounded stack scratch; f16/q8 only need head_dim <= OC_KV_MAX_HEAD for
   * their per-head decode buffer. Either failing downgrades to exact f32. */
  for (size_t l = 0; l < m->n_layers && m->kv_type != OC_KV_F32; ++l) {
    size_t hd = m->layers[l].head_dim, vd = m->layers[l].v_head_dim;
    if (m->kv_type == OC_KV_Q4 &&
        ((hd & (hd - 1)) || (vd & (vd - 1)) || hd > 512 || vd > 512 || hd < 2 ||
         vd < 2)) {
      fprintf(stderr,
              "gemma4: --kv-quant needs power-of-2 head dims <= 512 "
              "(layer %zu: %zu/%zu); using f32 KV cache\n", l, hd, vd);
      m->kv_type = OC_KV_F32;
    } else if (hd > OC_KV_MAX_HEAD || vd > OC_KV_MAX_HEAD) {
      fprintf(stderr, "gemma4: head_dim %zu/%zu > %d; using f32 KV cache\n", hd,
              vd, OC_KV_MAX_HEAD);
      m->kv_type = OC_KV_F32;
    }
  }
  m->kv_quant = m->kv_type == OC_KV_Q4;

  /* Allocation. f32 -> k_cache/v_cache floats; everything else packs into the
   * k_qcache/v_qcache byte buffers at oc_kv_elem_bytes each (q4 at 0.5 B/value),
   * with a scale (q8) or scale+min (q4) pair per (slot, kv-head) in *_qmeta. */
  size_t meta_per = m->kv_type == OC_KV_Q8 || m->kv_type == OC_KV_Q4 ? 2 : 0;
  for (size_t l = 0; l < m->n_layers; ++l) {
    Gemma4Layer* L = &m->layers[l];
    size_t kn = L->cache_cap * L->n_kv_heads * L->head_dim;
    size_t vn = L->cache_cap * L->n_kv_heads * L->v_head_dim;
    if (m->kv_type == OC_KV_F32) {
      L->k_cache = calloc(kn, sizeof(float));
      L->v_cache = calloc(vn, sizeof(float));
      if (!L->k_cache || !L->v_cache) {
        seterr(err, errlen, "gemma4: KV cache allocation failed%s", "");
        return -1;
      }
      continue;
    }
    size_t kb = m->kv_type == OC_KV_Q4 ? kn / 2 : kn * oc_kv_elem_bytes(m->kv_type);
    size_t vb = m->kv_type == OC_KV_Q4 ? vn / 2 : vn * oc_kv_elem_bytes(m->kv_type);
    L->k_qcache = calloc(kb, 1);
    L->v_qcache = calloc(vb, 1);
    if (meta_per) {
      L->k_qmeta = calloc(L->cache_cap * L->n_kv_heads * meta_per, sizeof(float));
      L->v_qmeta = calloc(L->cache_cap * L->n_kv_heads * meta_per, sizeof(float));
    }
    if (!L->k_qcache || !L->v_qcache || (meta_per && (!L->k_qmeta || !L->v_qmeta))) {
      seterr(err, errlen, "gemma4: KV cache allocation failed%s", "");
      return -1;
    }
  }
  if (m->kv_type != OC_KV_F32)
    fprintf(stderr, "gemma4: KV cache %s%s\n", oc_kv_type_name(m->kv_type),
            m->kv_type == OC_KV_Q4 ? " (rotated int4, ~8x smaller)"
            : m->kv_type == OC_KV_Q8 ? " (int8/head, ~4x smaller)"
                                     : " (~2x smaller)");

  /* rope_freqs.weight: proportional divisors for global-layer rope; all-ones
   * fallback. ones: scale-less V-norm weight. */
  {
    size_t nf = 0;
    m->rope_freqs = load_vec(g, "rope_freqs.weight", &nf);
    if (!m->rope_freqs || nf < max_hd / 2) {
      free(m->rope_freqs);
      m->rope_freqs = malloc((max_hd / 2) * sizeof(float));
      if (m->rope_freqs)
        for (size_t i = 0; i < max_hd / 2; ++i) m->rope_freqs[i] = 1.0f;
    }
    m->ones = malloc(max_hd * sizeof(float));
    if (m->ones)
      for (size_t i = 0; i < max_hd; ++i) m->ones[i] = 1.0f;
  }

  m->x = calloc(m->hidden, sizeof(float));
  m->logits = calloc(m->vocab, sizeof(float));
  m->normed = calloc(m->hidden, sizeof(float));
  m->q = calloc(max_q, sizeof(float));
  m->k = calloc(max_k, sizeof(float));
  m->v = calloc(max_k, sizeof(float));
  m->attn_res = calloc(max_q, sizeof(float));
  m->attn_proj = calloc(m->hidden, sizeof(float));
  m->gate = calloc(m->inter, sizeof(float));
  m->up = calloc(m->inter, sizeof(float));
  m->ffn_out = calloc(m->hidden, sizeof(float));
  m->head_tmp = calloc(max_hd, sizeof(float));
  if (!m->x || !m->logits || !m->normed || !m->q || !m->k || !m->v ||
      !m->attn_res || !m->attn_proj || !m->gate || !m->up || !m->ffn_out ||
      !m->head_tmp) {
    seterr(err, errlen, "gemma4: scratch allocation failed%s", "");
    return -1;
  }

  /* Batched-prefill scratch: the same vectors with a row per token. */
  {
    const char* e = getenv("OC_BATCH");
    long b = e ? atol(e) : 32;
    m->batch = (size_t)(b < 1 ? 1 : b > 512 ? 512 : b);
  }
  size_t B = m->batch;
  m->batch_cap = B; /* scratch below is sized to B; chunks must never exceed it */
  m->octx = oc_ctx_new();
  m->bx = calloc(B * m->hidden, sizeof(float));
  m->bnormed = calloc(B * m->hidden, sizeof(float));
  m->bq = calloc(B * max_q, sizeof(float));
  m->bk = calloc(B * max_k, sizeof(float));
  m->bv = calloc(B * max_k, sizeof(float));
  m->battn = calloc(B * max_q, sizeof(float));
  m->bproj = calloc(B * m->hidden, sizeof(float));
  m->bgate = calloc(B * m->inter, sizeof(float));
  m->bup = calloc(B * m->inter, sizeof(float));
  m->bffn = calloc(B * m->hidden, sizeof(float));
  if (!m->octx || !m->bx || !m->bnormed || !m->bq || !m->bk || !m->bv ||
      !m->battn || !m->bproj || !m->bgate || !m->bup || !m->bffn) {
    seterr(err, errlen, "gemma4: batch scratch allocation failed%s", "");
    return -1;
  }

  m->g = *g; /* take ownership */
  memset(g, 0, sizeof(*g));
  return 0;
}

void gemma4_free(Gemma4Model* m) {
  for (size_t l = 0; m->layers && l < m->n_layers; ++l) {
    Gemma4Layer* L = &m->layers[l];
    free(L->attn_norm);
    free(L->attn_q_norm);
    free(L->attn_k_norm);
    free(L->post_attn_norm);
    free(L->ffn_norm);
    free(L->post_ffn_norm);
    free(L->k_cache);
    free(L->v_cache);
    free(L->k_qcache);
    free(L->v_qcache);
    free(L->k_qmeta);
    free(L->v_qmeta);
  }
  free(m->layers);
  free(m->out_norm);
  free(m->rope_freqs);
  free(m->ones);
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

/* One online-softmax step: fold key/value row t into the running head output.
 * Shared by the decode and prefill attention kernels so the two cannot drift. */
static inline void attn_accum(float* oh, const float* qh, const float* krow,
                              const float* vrow, size_t hd, size_t vd,
                              float scale, float* rmax, float* rsum) {
  float score = oc_dot_f32(qh, krow, hd) * scale;
  float new_max = *rmax > score ? *rmax : score;
  float ef = expf(*rmax - new_max);
  float es = expf(score - new_max);
  if (ef != 1.0f)
    for (size_t d = 0; d < vd; ++d) oh[d] *= ef;
  for (size_t d = 0; d < vd; ++d) oh[d] += es * vrow[d];
  *rsum = *rsum * ef + es;
  *rmax = new_max;
}

/* Decode one cached (slot, kv-head) head vector. f32 points straight into the
 * ring; f16/q8/q4 decode `dim` values into buf. `row` is the per-slot stride
 * (n_kv_heads*dim). One codebook for K and V; cache_k/cache_v just bind it. */
static inline const float* decode_ring(const float* fcache, const uint8_t* qcache,
                                       const float* qmeta, OcKvType kt, size_t row,
                                       size_t dim, size_t n_kv_heads, size_t slot,
                                       size_t kvh, float* buf) {
  size_t base = slot * row + kvh * dim, r = slot * n_kv_heads + kvh;
  switch (kt) {
    case OC_KV_F32: return fcache + base;
    case OC_KV_F16: oc_kv_decode(OC_KV_F16, qcache + base * 2, dim, 0.0f, buf); break;
    case OC_KV_Q8: oc_kv_decode(OC_KV_Q8, qcache + base, dim, qmeta[r * 2], buf); break;
    default: /* Q4 */ oc_kvq_decode(qcache + base / 2, dim, qmeta + r * 2, buf); break;
  }
  return buf;
}

/* Encode one head vector into the ring at (slot, kvh) — inverse of decode_ring,
 * same layout, so store-then-decode round-trips exactly. */
static inline void encode_ring(float* fcache, uint8_t* qcache, float* qmeta,
                               OcKvType kt, size_t row, size_t dim, size_t n_kv_heads,
                               size_t slot, size_t kvh, const float* x) {
  size_t base = slot * row + kvh * dim, r = slot * n_kv_heads + kvh;
  switch (kt) {
    case OC_KV_F32: memcpy(fcache + base, x, dim * sizeof(float)); break;
    case OC_KV_F16: oc_kv_encode(OC_KV_F16, x, dim, qcache + base * 2, NULL); break;
    case OC_KV_Q8: oc_kv_encode(OC_KV_Q8, x, dim, qcache + base, &qmeta[r * 2]); break;
    default: /* Q4 */ oc_kvq_encode(x, dim, qcache + base / 2, qmeta + r * 2); break;
  }
}

static inline const float* cache_k(const Gemma4Layer* L, OcKvType kt, size_t slot,
                                   size_t kvh, float* buf) {
  return decode_ring(L->k_cache, L->k_qcache, L->k_qmeta, kt,
                     L->n_kv_heads * L->head_dim, L->head_dim, L->n_kv_heads, slot,
                     kvh, buf);
}
static inline const float* cache_v(const Gemma4Layer* L, OcKvType kt, size_t slot,
                                   size_t kvh, float* buf) {
  return decode_ring(L->v_cache, L->v_qcache, L->v_qmeta, kt,
                     L->n_kv_heads * L->v_head_dim, L->v_head_dim, L->n_kv_heads,
                     slot, kvh, buf);
}

/* Commit position `slot`'s K and V (each n_kv_heads heads) into the ring. */
static void gemma4_store_kv(const Gemma4Model* m, const Gemma4Layer* L, size_t slot,
                            const float* k, const float* v) {
  size_t hd = L->head_dim, vd = L->v_head_dim;
  size_t k_row = L->n_kv_heads * hd, v_row = L->n_kv_heads * vd;
  for (size_t h = 0; h < L->n_kv_heads; ++h) {
    encode_ring(L->k_cache, L->k_qcache, L->k_qmeta, m->kv_type, k_row, hd,
                L->n_kv_heads, slot, h, k + h * hd);
    encode_ring(L->v_cache, L->v_qcache, L->v_qmeta, m->kv_type, v_row, vd,
                L->n_kv_heads, slot, h, v + h * vd);
  }
}

/* Round-trip one head vector through the cache codec in place. The batch keeps
 * its in-window K/V in a panel (not yet in the ring), so it must see the same
 * rounding the ring would apply — else batched prefill is silently MORE precise
 * than sequential decode, which fails batched==sequential the same way being
 * less precise would. No-op for f32. */
static void kv_roundtrip(OcKvType kt, float* x, size_t dim) {
  if (kt == OC_KV_F32) return;
  uint8_t tmp[OC_KV_MAX_HEAD * 2]; /* f16 = 2 B/value, dim <= OC_KV_MAX_HEAD */
  if (kt == OC_KV_Q4) {
    float meta[2];
    oc_kvq_encode(x, dim, tmp, meta);
    oc_kvq_decode(tmp, dim, meta, x);
    return;
  }
  float sc;
  oc_kv_encode(kt, x, dim, tmp, &sc);
  oc_kv_decode(kt, tmp, dim, sc, x);
}

/* Online-softmax causal attention over the ring cache, threaded over q heads. */
typedef struct {
  const Gemma4Layer* L;
  const float* q;
  float* out;       /* [n_head * v_head_dim] */
  size_t n_head;
  size_t t0, t1;    /* attended positions [t0, t1) (absolute) */
  OcKvType kv_type;
  float scale;
} AttnJob;

static void attn_heads(void* ctx, size_t h0, size_t h1) {
  AttnJob* j = ctx;
  const Gemma4Layer* L = j->L;
  size_t hd = L->head_dim, vd = L->v_head_dim;
  size_t group = j->n_head / L->n_kv_heads;
  for (size_t h = h0; h < h1; ++h) {
    const float* qh = j->q + h * hd;
    float* oh = j->out + h * vd;
    size_t kvh = h / group;
    float rmax = -INFINITY, rsum = 0.0f;
    for (size_t d = 0; d < vd; ++d) oh[d] = 0.0f;
    float kbuf[OC_KV_MAX_HEAD], vbuf[OC_KV_MAX_HEAD]; /* f16/q8/q4 decode scratch */
    for (size_t t = j->t0; t < j->t1; ++t) {
      size_t slot = t % L->cache_cap;
      attn_accum(oh, qh, cache_k(L, j->kv_type, slot, kvh, kbuf),
                 cache_v(L, j->kv_type, slot, kvh, vbuf), hd, vd, j->scale, &rmax,
                 &rsum);
    }
    if (rsum > 0.0f) {
      float inv = 1.0f / rsum;
      for (size_t d = 0; d < vd; ++d) oh[d] *= inv;
    }
  }
}

/* ---- batched (prefill) attention -------------------------------------------
 * One (token, head) pair per index. Token i attends to [t0_i, pos0+i]: the ring
 * cache supplies the positions before the batch, the K/V panel supplies the
 * ones inside it.
 *
 * TWO THINGS ARE LOAD-BEARING HERE.
 *
 * 1. `t < seq` (seq = pos0+i+1) is the causal mask, and t0_i re-derives the
 *    sliding window per token. Widen either and a prompt token attends to the
 *    future or past its window, which no crash will tell you about.
 *
 * 2. The batch's K/V is NOT in the ring yet — it is committed after this runs.
 *    It cannot be committed before: on an SWA layer, slot p % cap holds
 *    position p - cap until p overwrites it, and p - cap is exactly the oldest
 *    position token p - 1 still needs. Writing the batch first therefore eats
 *    the window of every earlier token in it. A ring of cap slots holding a
 *    cap-wide window has no slack for a second token, so this is not tunable —
 *    it is why the panel exists. */
typedef struct {
  const Gemma4Layer* L;
  const float* q;  /* [n_tok][q_len] */
  const float* kp; /* [n_tok][n_kv_heads * head_dim]   in-batch K, post-rope */
  const float* vp; /* [n_tok][n_kv_heads * v_head_dim] in-batch V */
  float* out;      /* [n_tok][n_head * v_head_dim] */
  size_t n_head, pos0, q_len;
  OcKvType kv_type;
  float scale;
} AttnBatchJob;

static void attn_batch_heads(void* ctx, size_t i0, size_t i1) {
  AttnBatchJob* j = ctx;
  const Gemma4Layer* L = j->L;
  size_t hd = L->head_dim, vd = L->v_head_dim;
  size_t group = j->n_head / L->n_kv_heads;
  size_t k_row = L->n_kv_heads * hd, v_row = L->n_kv_heads * vd;
  size_t out_row = j->n_head * vd;
  for (size_t idx = i0; idx < i1; ++idx) {
    size_t i = idx / j->n_head, h = idx % j->n_head;
    size_t seq = j->pos0 + i + 1;
    size_t t0 = seq > L->cache_cap ? seq - L->cache_cap : 0;
    size_t kvh = h / group;
    const float* qh = j->q + i * j->q_len + h * hd;
    float* oh = j->out + i * out_row + h * vd;
    float rmax = -INFINITY, rsum = 0.0f;
    for (size_t d = 0; d < vd; ++d) oh[d] = 0.0f;
    float kbuf[OC_KV_MAX_HEAD], vbuf[OC_KV_MAX_HEAD];
    for (size_t t = t0; t < seq; ++t) {
      const float *krow, *vrow;
      if (t < j->pos0) {
        size_t slot = t % L->cache_cap;
        krow = cache_k(L, j->kv_type, slot, kvh, kbuf);
        vrow = cache_v(L, j->kv_type, slot, kvh, vbuf);
      } else {
        size_t b = t - j->pos0;
        krow = j->kp + b * k_row + kvh * hd;
        vrow = j->vp + b * v_row + kvh * vd;
      }
      attn_accum(oh, qh, krow, vrow, hd, vd, j->scale, &rmax, &rsum);
    }
    if (rsum > 0.0f) {
      float inv = 1.0f / rsum;
      for (size_t d = 0; d < vd; ++d) oh[d] *= inv;
    }
  }
}

/* OXC_DEBUG: report vector stats at pos 0 to localize NaN/blow-up. */
static void dbg_vec(size_t pos, const char* tag, size_t l, const float* v, size_t n) {
  static int on = -1;
  if (on < 0) on = getenv("OXC_DEBUG") != NULL;
  if (!on || pos != 0) return;
  size_t bad = 0;
  float mx = 0;
  for (size_t i = 0; i < n; ++i) {
    if (isnan(v[i]) || isinf(v[i])) bad++;
    else if (fabsf(v[i]) > mx) mx = fabsf(v[i]);
  }
  if (bad || l < 2 || strcmp(tag, "post-ffn") == 0 || strcmp(tag, "logits") == 0)
    fprintf(stderr, "dbg l=%zu %-9s nan/inf=%zu absmax=%.4g first=[%.4g %.4g %.4g %.4g]\n",
            l, tag, bad, (double)mx, (double)v[0], (double)v[1], (double)v[2], (double)v[3]);
}

float* gemma4_forward(Gemma4Model* m, int32_t token, size_t pos, bool need_logits) {
  const size_t h = m->hidden;
  if (pos >= m->ctx) {
    fprintf(stderr, "gemma4: position %zu exceeds context %zu\n", pos, m->ctx);
    return NULL;
  }

  /* Embedding lookup (dequant one row) scaled by sqrt(hidden). */
  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, h);
  oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * emb_row, m->x, h);
  for (size_t i = 0; i < h; ++i) m->x[i] *= m->emb_scale;
  dbg_vec(pos, "embed", 0, m->x, h);

  return gemma4_forward_from(m, pos, 0, need_logits);
}

/* The layer loop + head of gemma4_forward, entered at layer l0 with m->x
 * already holding the hidden state. l0 == 0 is exactly what gemma4_forward
 * runs; l0 > 0 is the CPU half of the CUDA backend's -ngl partial offload
 * (the GPU ran [0, l0) and handed the residual stream back). */
float* gemma4_forward_from(Gemma4Model* m, size_t pos, size_t l0,
                           bool need_logits) {
  const size_t h = m->hidden;
  if (pos >= m->ctx) {
    fprintf(stderr, "gemma4: position %zu exceeds context %zu\n", pos, m->ctx);
    return NULL;
  }

  for (size_t l = l0; l < m->n_layers; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    size_t hd = L->head_dim, vd = L->v_head_dim;
    size_t q_len = m->n_head * hd;
    size_t k_len = L->n_kv_heads * hd;
    size_t v_len = L->n_kv_heads * vd;

    /* ---- attention (reference graph: oxidize-core diffusion_gemma.rs) ---- */
    oc_rms_norm(m->normed, m->x, L->attn_norm, h, m->eps);
    oc_matvec(m->octx, m->q, L->attn_q->ggml_type, L->attn_q->data, q_len, h, m->normed);
    oc_matvec(m->octx, m->k, L->attn_k->ggml_type, L->attn_k->data, k_len, h, m->normed);
    if (L->attn_v)
      oc_matvec(m->octx, m->v, L->attn_v->ggml_type, L->attn_v->data, v_len, h, m->normed);
    else
      memcpy(m->v, m->k, k_len * sizeof(float)); /* full layers: V = raw K proj */

    /* Global layers use proportional rope_freqs divisors; SWA layers none. */
    const float* freqs = L->is_swa ? NULL : m->rope_freqs;

    /* Per-head: Q/K RMSNorm then rope; V gets a SCALE-LESS RMSNorm, no rope. */
    for (size_t hh = 0; hh < m->n_head; ++hh) {
      float* p = m->q + hh * hd;
      if (L->attn_q_norm) {
        oc_rms_norm(m->head_tmp, p, L->attn_q_norm, hd, m->eps);
        memcpy(p, m->head_tmp, hd * sizeof(float));
      }
      oc_rope(p, hd, 1, pos, L->rope.theta, L->rope.rope_dim, freqs);
      if (m->kv_type == OC_KV_Q4) oc_fht(p, hd); /* rotate q to match rotated cached k */
    }
    for (size_t hh = 0; hh < L->n_kv_heads; ++hh) {
      float* p = m->k + hh * hd;
      if (L->attn_k_norm) {
        oc_rms_norm(m->head_tmp, p, L->attn_k_norm, hd, m->eps);
        memcpy(p, m->head_tmp, hd * sizeof(float));
      }
      oc_rope(p, hd, 1, pos, L->rope.theta, L->rope.rope_dim, freqs);
      if (m->kv_type == OC_KV_Q4) oc_fht(p, hd);
      float* pv = m->v + hh * vd;
      oc_rms_norm(m->head_tmp, pv, m->ones, vd, m->eps); /* scale-less V norm */
      memcpy(pv, m->head_tmp, vd * sizeof(float));
      if (m->kv_type == OC_KV_Q4) oc_fht(pv, vd);
    }

    /* Store K/V into the ring cache in the chosen precision (q4 rotates first). */
    size_t slot = pos % L->cache_cap;
    gemma4_store_kv(m, L, slot, m->k, m->v);

    size_t seq = pos + 1;
    size_t t0 = seq > L->cache_cap ? seq - L->cache_cap : 0;
    /* Gemma4 attention scale is 1.0 (llama.cpp: f_attention_scale = 1.0f). */
    AttnJob job = {L, m->q, m->attn_res, m->n_head, t0, seq, m->kv_type,
                   m->attn_scale > 0.0f ? m->attn_scale : 1.0f};
    oc_parallel_for(m->n_head, attn_heads, &job);
    /* Cached V is rotated; softmax mixing commutes with the rotation, so one
     * self-inverse FHT per output head undoes it. */
    if (m->kv_type == OC_KV_Q4)
      for (size_t hh = 0; hh < m->n_head; ++hh)
        oc_fht(m->attn_res + hh * vd, vd);

    oc_matvec(m->octx, m->attn_proj, L->attn_out->ggml_type, L->attn_out->data, h,
              m->n_head * vd, m->attn_res);
    dbg_vec(pos, "attn_res", l, m->attn_res, m->n_head * vd);
    /* attn_out = post_attention_norm(attn_proj) + x  (stored in attn_proj) */
    if (L->post_attn_norm)
      oc_rms_norm(m->attn_proj, m->attn_proj, L->post_attn_norm, h, m->eps);
    for (size_t i = 0; i < h; ++i) m->attn_proj[i] += m->x[i];
    dbg_vec(pos, "post-attn", l, m->attn_proj, h);

    /* ---- FFN (GeGLU) over attn_out ---- */
    oc_rms_norm(m->normed, m->attn_proj, L->ffn_norm, h, m->eps);
    oc_matvec(m->octx, m->gate, L->ffn_gate->ggml_type, L->ffn_gate->data, m->inter, h, m->normed);
    oc_matvec(m->octx, m->up, L->ffn_up->ggml_type, L->ffn_up->data, m->inter, h, m->normed);
    oc_geglu(m->gate, m->up, m->gate, m->inter);
    oc_matvec(m->octx, m->ffn_out, L->ffn_down->ggml_type, L->ffn_down->data, h, m->inter, m->gate);
    /* x = (post_ffw_norm(ffn_out) + attn_out) * layer_output_scale */
    if (L->post_ffn_norm)
      oc_rms_norm(m->ffn_out, m->ffn_out, L->post_ffn_norm, h, m->eps);
    for (size_t i = 0; i < h; ++i)
      m->x[i] = (m->ffn_out[i] + m->attn_proj[i]) * L->output_scale;
    dbg_vec(pos, "post-ffn", l, m->x, h);
  }
  m->kv_len = pos + 1;

  if (!need_logits) return NULL;

  oc_rms_norm(m->normed, m->x, m->out_norm, h, m->eps);
  /* Tied embeddings: logits = token_embd @ hidden. */
  oc_matvec(m->octx, m->logits, m->tok_embd->ggml_type, m->tok_embd->data, m->vocab, h, m->normed);
  dbg_vec(pos, "logits", 99, m->logits, m->vocab);
  if (m->final_softcap > 0.0f) {
    float c = m->final_softcap;
    for (size_t i = 0; i < m->vocab; ++i) m->logits[i] = c * tanhf(m->logits[i] / c);
  }
  return m->logits;
}

/* Same graph as gemma4_forward with a token axis: every projection becomes one
 * oc_matmul over the whole batch (weights read once, not once per token), the
 * per-head norms/rope stay per token, and attention is one dispatch over
 * (token, head). The only structural difference is that the K/V cache is
 * committed after attention rather than before — see attn_batch_heads. */
float* gemma4_forward_batch(Gemma4Model* m, const int32_t* tokens, size_t n,
                            size_t pos0, bool need_logits) {
  const size_t h = m->hidden;
  const float eps = m->eps;
  if (n == 0) return NULL;
  if (pos0 + n > m->ctx) {
    fprintf(stderr, "gemma4: batch [%zu,%zu) exceeds context %zu\n", pos0,
            pos0 + n, m->ctx);
    return NULL;
  }
  size_t bs = m->batch < m->batch_cap ? m->batch : m->batch_cap; /* never over cap */
  if (bs < 1) bs = 1;
  if (n > bs) { /* one chunk at a time; logits come from the last */
    float* out = NULL;
    for (size_t i = 0; i < n; i += bs) {
      size_t c = n - i < bs ? n - i : bs;
      out = gemma4_forward_batch(m, tokens + i, c, pos0 + i,
                                 need_logits && i + c == n);
    }
    return out;
  }

  const size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, h);
  for (size_t i = 0; i < n; ++i) {
    size_t tk = (size_t)tokens[i] < m->vocab ? (size_t)tokens[i] : m->vocab - 1;
    float* xi = m->bx + i * h;
    oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * emb_row, xi, h);
    for (size_t d = 0; d < h; ++d) xi[d] *= m->emb_scale;
  }

  for (size_t l = 0; l < m->n_layers; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    const size_t hd = L->head_dim, vd = L->v_head_dim;
    const size_t q_len = m->n_head * hd, o_len = m->n_head * vd;
    const size_t k_len = L->n_kv_heads * hd, v_len = L->n_kv_heads * vd;

    for (size_t i = 0; i < n; ++i)
      oc_rms_norm(m->bnormed + i * h, m->bx + i * h, L->attn_norm, h, eps);
    oc_matmul(m->octx, m->bq, L->attn_q->ggml_type, L->attn_q->data, q_len, h,
              m->bnormed, n);
    oc_matmul(m->octx, m->bk, L->attn_k->ggml_type, L->attn_k->data, k_len, h,
              m->bnormed, n);
    if (L->attn_v)
      oc_matmul(m->octx, m->bv, L->attn_v->ggml_type, L->attn_v->data, v_len, h,
                m->bnormed, n);
    else
      memcpy(m->bv, m->bk, n * k_len * sizeof(float)); /* global layers: V = K proj */

    const float* freqs = L->is_swa ? NULL : m->rope_freqs;
    for (size_t i = 0; i < n; ++i) {
      const size_t pos = pos0 + i;
      for (size_t hh = 0; hh < m->n_head; ++hh) {
        float* p = m->bq + i * q_len + hh * hd;
        if (L->attn_q_norm) {
          oc_rms_norm(m->head_tmp, p, L->attn_q_norm, hd, eps);
          memcpy(p, m->head_tmp, hd * sizeof(float));
        }
        oc_rope(p, hd, 1, pos, L->rope.theta, L->rope.rope_dim, freqs);
        if (m->kv_type == OC_KV_Q4) oc_fht(p, hd);
      }
      for (size_t hh = 0; hh < L->n_kv_heads; ++hh) {
        float* p = m->bk + i * k_len + hh * hd;
        if (L->attn_k_norm) {
          oc_rms_norm(m->head_tmp, p, L->attn_k_norm, hd, eps);
          memcpy(p, m->head_tmp, hd * sizeof(float));
        }
        oc_rope(p, hd, 1, pos, L->rope.theta, L->rope.rope_dim, freqs);
        if (m->kv_type == OC_KV_Q4) oc_fht(p, hd);
        float* pv = m->bv + i * v_len + hh * vd;
        oc_rms_norm(m->head_tmp, pv, m->ones, vd, eps); /* scale-less V norm */
        memcpy(pv, m->head_tmp, vd * sizeof(float));
        if (m->kv_type == OC_KV_Q4) oc_fht(pv, vd);
      }
    }

    /* Attention reads the panel for in-batch positions, so the panel must hold
     * exactly what the cache would hand back — round-trip it through the same
     * codec (f16/q8/q4). Skipping this makes batched prefill silently MORE
     * precise than decode, which is the same class of bug as being less precise. */
    if (m->kv_type != OC_KV_F32)
      for (size_t i = 0; i < n; ++i)
        for (size_t hh = 0; hh < L->n_kv_heads; ++hh) {
          kv_roundtrip(m->kv_type, m->bk + i * k_len + hh * hd, hd);
          kv_roundtrip(m->kv_type, m->bv + i * v_len + hh * vd, vd);
        }

    AttnBatchJob job = {L,    m->bq, m->bk,  m->bv,   m->battn, m->n_head, pos0,
                        q_len, m->kv_type, m->attn_scale > 0.0f ? m->attn_scale : 1.0f};
    oc_parallel_for(n * m->n_head, attn_batch_heads, &job);
    if (m->kv_type == OC_KV_Q4)
      for (size_t i = 0; i < n; ++i)
        for (size_t hh = 0; hh < m->n_head; ++hh)
          oc_fht(m->battn + i * o_len + hh * vd, vd);

    /* Commit now, in position order: the ring ends up exactly as sequential
     * decode would have left it. */
    for (size_t i = 0; i < n; ++i) {
      size_t slot = (pos0 + i) % L->cache_cap;
      gemma4_store_kv(m, L, slot, m->bk + i * k_len, m->bv + i * v_len);
    }

    oc_matmul(m->octx, m->bproj, L->attn_out->ggml_type, L->attn_out->data, h,
              o_len, m->battn, n);
    for (size_t i = 0; i < n; ++i) {
      float* p = m->bproj + i * h;
      if (L->post_attn_norm) oc_rms_norm(p, p, L->post_attn_norm, h, eps);
      const float* xi = m->bx + i * h;
      for (size_t d = 0; d < h; ++d) p[d] += xi[d];
    }

    for (size_t i = 0; i < n; ++i)
      oc_rms_norm(m->bnormed + i * h, m->bproj + i * h, L->ffn_norm, h, eps);
    oc_matmul(m->octx, m->bgate, L->ffn_gate->ggml_type, L->ffn_gate->data,
              m->inter, h, m->bnormed, n);
    oc_matmul(m->octx, m->bup, L->ffn_up->ggml_type, L->ffn_up->data, m->inter, h,
              m->bnormed, n);
    for (size_t i = 0; i < n; ++i)
      oc_geglu(m->bgate + i * m->inter, m->bup + i * m->inter,
               m->bgate + i * m->inter, m->inter);
    oc_matmul(m->octx, m->bffn, L->ffn_down->ggml_type, L->ffn_down->data, h,
              m->inter, m->bgate, n);
    for (size_t i = 0; i < n; ++i) {
      float* fo = m->bffn + i * h;
      if (L->post_ffn_norm) oc_rms_norm(fo, fo, L->post_ffn_norm, h, eps);
      const float* p = m->bproj + i * h;
      float* xi = m->bx + i * h;
      for (size_t d = 0; d < h; ++d) xi[d] = (fo[d] + p[d]) * L->output_scale;
    }
  }

  m->kv_len = pos0 + n;

  if (!need_logits) return NULL;
  /* Only the last token of a prompt is ever sampled from. */
  oc_rms_norm(m->normed, m->bx + (n - 1) * h, m->out_norm, h, eps);
  oc_matvec(m->octx, m->logits, m->tok_embd->ggml_type, m->tok_embd->data,
            m->vocab, h, m->normed);
  if (m->final_softcap > 0.0f) {
    float c = m->final_softcap;
    for (size_t i = 0; i < m->vocab; ++i) m->logits[i] = c * tanhf(m->logits[i] / c);
  }
  return m->logits;
}

/* See model_gemma4.h. The ring is positionally addressed (slot = pos %
 * cache_cap), so dropping >= pos is normally just bookkeeping. But on an SWA
 * ring that has already wrapped, the window tail [pos+1-cap, pos-1] that
 * regenerating `pos` re-reads may have been overwritten by a later position
 * sharing its slot — evicted K/V that is gone for good. Rather than silently
 * serve those stale slots (wrong logits, no crash), refuse: leave kv_len and
 * return false. Reproducible iff nothing has evicted yet (kv_len <= cap) or only
 * the last token is dropped (pos >= kv_len-1). `cap` is the smallest ring across
 * layers = the SWA window when it evicts (global layers use the full ctx). */
bool gemma4_kv_rewind(Gemma4Model* m, size_t pos) {
  if (pos >= m->kv_len) return true; /* forward rewind: nothing to drop */
  size_t cap = m->ctx;
  for (size_t l = 0; l < m->n_layers; ++l)
    if (m->layers[l].cache_cap < cap) cap = m->layers[l].cache_cap;
  if (m->kv_len > cap && pos + 1 < m->kv_len) return false; /* would re-read an evicted slot */
  m->kv_len = pos;
  return true;
}
