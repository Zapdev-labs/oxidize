#include "vision.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"
#include "tensor.h"

/* ---- geometry / small helpers ---------------------------------------------- */

static void verr(char* err, size_t n, const char* msg) {
  if (err && n) snprintf(err, n, "%s", msg);
}

static uint32_t vkv_u32(const GgufFile* g, const char* key, uint32_t dflt) {
  uint32_t o;
  return gguf_get_u32(g, key, &o) ? o : dflt;
}
static float vkv_f32(const GgufFile* g, const char* key, float dflt) {
  float o;
  return gguf_get_f32(g, key, &o) ? o : dflt;
}
static int vkv_bool(const GgufFile* g, const char* key, int dflt) {
  const GgufValue* v = gguf_find(g, key);
  if (!v) return dflt;
  if (v->kind == GGUF_T_BOOL) return v->v.u != 0;
  return dflt;
}

/* CLIP uses quick-GELU (x*sigmoid(1.702x)); SigLIP the tanh GELU. llama.cpp
 * picks between them from clip.use_gelu, and the projector MLP always uses the
 * tanh GELU. Both are elementwise — NOT oc_geglu, which is the GATED gelu(gate)*
 * up used by SwiGLU FFNs; a CLIP MLP is the ungated fc2(gelu(fc1(x))). */
static inline float gelu_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
static inline float gelu_quick(float x) { return x / (1.0f + expf(-1.702f * x)); }

/* LayerNorm (mean+var, with weight and optional bias). Vision towers use this,
 * not RMSNorm, so oc_rms_norm cannot stand in: it skips the mean subtraction
 * and the bias. out may alias x. */
static void layer_norm(float* out, const float* x, const float* w, const float* b,
                       size_t n, float eps) {
  float mean = 0.0f;
  for (size_t i = 0; i < n; ++i) mean += x[i];
  mean /= (float)n;
  float var = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = x[i] - mean;
    var += d * d;
  }
  var /= (float)n;
  float inv = 1.0f / sqrtf(var + eps);
  for (size_t i = 0; i < n; ++i)
    out[i] = (x[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
}

/* ---- weights ---------------------------------------------------------------- */

/* A weight matrix kept as its mmap view + type, consumed by oc_matmul as a
 * row-major [rows][cols] matrix. rows is the tensor's LAST (slowest) GGUF dim,
 * cols the product of the rest: that is the natural [out_features, in_features]
 * layout of a linear weight AND the [in_ch*kh*kw, out_ch] layout ggml gives a
 * 4-D conv patch-embed weight, so one rule covers both. */
typedef struct {
  uint32_t type;
  const uint8_t* data;
  size_t rows, cols;
} VMat;

typedef struct {
  VMat q, k, v, o, up, down;
  float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
  float *bq, *bk, *bv, *bo, *bup, *bdown; /* optional biases; NULL if absent */
} VLayer;

struct VisionEncoder {
  VisionConfig cfg;
  const GgufFile* g; /* borrowed; weight matrices point into its mmap */
  OcCtx* ctx;
  VMat patch;  /* [hidden][3*patch*patch] */
  float* patch_b;
  float* class_embd; /* [hidden]; NULL for SigLIP */
  float* pos_embd;   /* [n_positions*hidden] */
  float *pre_ln_w, *pre_ln_b, *post_ln_w, *post_ln_b;
  VLayer* layers;
  VMat mm0;    /* projector fc1 [proj0][hidden] */
  float* mm0_b;
  size_t proj0;
  int has_mm2;
  VMat mm2; /* projector fc2 [proj_dim][proj0]; absent => single-layer projector */
  float* mm2_b;
};

/* Load a weight matrix by name; validates a kernel exists for its type. */
static int mat_load(const GgufFile* g, const char* name, VMat* out, char* err,
                    size_t errlen) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  if (!t || t->n_dims == 0) {
    snprintf(err, errlen, "vision: missing tensor %s", name);
    return -1;
  }
  size_t rows = (size_t)t->dims[t->n_dims - 1], cols = 1;
  for (uint32_t d = 0; d + 1 < t->n_dims; ++d) cols *= (size_t)t->dims[d];
  if (oc_row_bytes(t->ggml_type, cols) == 0) {
    snprintf(err, errlen, "vision: tensor %s has unsupported quant type %u", name,
             t->ggml_type);
    return -1;
  }
  out->type = t->ggml_type;
  out->data = t->data;
  out->rows = rows;
  out->cols = cols;
  return 0;
}

/* Dequantize a (1-D or flattened) tensor into a fresh f32 buffer. required =>
 * absence is an error; otherwise absence returns 0 with *out == NULL. `expect`
 * (when non-zero) is asserted against the value count. */
static int vec_try(const GgufFile* g, const char* name, size_t expect, int required,
                   float** out, char* err, size_t errlen) {
  *out = NULL;
  const GgufTensorInfo* t = gguf_tensor(g, name);
  if (!t) {
    if (required) {
      snprintf(err, errlen, "vision: missing tensor %s", name);
      return -1;
    }
    return 0;
  }
  size_t count = 1;
  for (uint32_t d = 0; d < t->n_dims; ++d) count *= (size_t)t->dims[d];
  if (expect && count != expect) {
    snprintf(err, errlen, "vision: %s has %zu values, expected %zu", name, count,
             expect);
    return -1;
  }
  float* buf = malloc((count ? count : 1) * sizeof(float));
  if (!buf) {
    verr(err, errlen, "vision: out of memory");
    return -1;
  }
  if (oc_dequant_row(t->ggml_type, t->data, buf, count) != 0) {
    free(buf);
    snprintf(err, errlen, "vision: cannot dequant %s (type %u)", name, t->ggml_type);
    return -1;
  }
  *out = buf;
  return 0;
}

/* Require a square [hidden][hidden] linear (q/k/v/out proj). */
static int mat_square(const GgufFile* g, const char* name, size_t h, VMat* out,
                      char* err, size_t errlen) {
  if (mat_load(g, name, out, err, errlen)) return -1;
  if (out->rows != h || out->cols != h) {
    snprintf(err, errlen, "vision: %s is %zux%zu, expected %zux%zu", name, out->rows,
             out->cols, h, h);
    return -1;
  }
  return 0;
}

int vision_load(VisionEncoder** out, const GgufFile* g, char* err, size_t errlen) {
  if (err && errlen) err[0] = 0;
  VisionConfig C = {0};
  C.image_size = vkv_u32(g, "clip.vision.image_size", 0);
  C.patch_size = vkv_u32(g, "clip.vision.patch_size", 0);
  C.hidden = vkv_u32(g, "clip.vision.embedding_length", 0);
  C.n_head = vkv_u32(g, "clip.vision.attention.head_count", 0);
  C.n_layer = vkv_u32(g, "clip.vision.block_count", 0);
  C.inter = vkv_u32(g, "clip.vision.feed_forward_length", 0);
  C.proj_dim = vkv_u32(g, "clip.vision.projection_dim", 0);
  C.eps = vkv_f32(g, "clip.vision.attention.layer_norm_epsilon", 1e-5f);
  C.use_gelu = vkv_bool(g, "clip.use_gelu", 0);
  C.has_class_token = gguf_tensor(g, "v.class_embd") != NULL;

  if (!C.image_size || !C.patch_size || !C.hidden || !C.n_head || !C.n_layer ||
      !C.inter) {
    verr(err, errlen, "vision: missing clip.vision.* geometry keys");
    return -1;
  }
  if (C.image_size % C.patch_size) {
    verr(err, errlen, "vision: image_size not divisible by patch_size");
    return -1;
  }
  if (C.hidden % C.n_head) {
    verr(err, errlen, "vision: embedding_length not divisible by head_count");
    return -1;
  }
  size_t grid = C.image_size / C.patch_size;
  C.n_patches = grid * grid;
  C.n_positions = C.n_patches + (C.has_class_token ? 1 : 0);

  VisionEncoder* v = calloc(1, sizeof *v);
  if (!v) {
    verr(err, errlen, "vision: out of memory");
    return -1;
  }
  v->cfg = C; /* n_layer set so vision_free can walk layers on the failure path */
  v->g = g;
  v->ctx = oc_ctx_new();
  if (!v->ctx) {
    verr(err, errlen, "vision: out of memory");
    goto fail;
  }

  /* patch/conv embedding: [hidden][3*patch*patch] */
  size_t pdim = 3 * C.patch_size * C.patch_size;
  if (mat_load(g, "v.patch_embd.weight", &v->patch, err, errlen)) goto fail;
  if (v->patch.rows != C.hidden || v->patch.cols != pdim) {
    snprintf(err, errlen, "vision: v.patch_embd.weight is %zux%zu, expected %zux%zu",
             v->patch.rows, v->patch.cols, C.hidden, pdim);
    goto fail;
  }
  if (vec_try(g, "v.patch_embd.bias", C.hidden, 0, &v->patch_b, err, errlen)) goto fail;
  if (C.has_class_token &&
      vec_try(g, "v.class_embd", C.hidden, 1, &v->class_embd, err, errlen))
    goto fail;
  if (vec_try(g, "v.position_embd.weight", C.n_positions * C.hidden, 1, &v->pos_embd,
              err, errlen))
    goto fail;
  if (vec_try(g, "v.pre_ln.weight", C.hidden, 0, &v->pre_ln_w, err, errlen)) goto fail;
  if (v->pre_ln_w &&
      vec_try(g, "v.pre_ln.bias", C.hidden, 0, &v->pre_ln_b, err, errlen))
    goto fail;
  if (vec_try(g, "v.post_ln.weight", C.hidden, 0, &v->post_ln_w, err, errlen))
    goto fail;
  if (v->post_ln_w &&
      vec_try(g, "v.post_ln.bias", C.hidden, 0, &v->post_ln_b, err, errlen))
    goto fail;

  v->layers = calloc(C.n_layer, sizeof(VLayer));
  if (!v->layers) {
    verr(err, errlen, "vision: out of memory");
    goto fail;
  }
  for (size_t l = 0; l < C.n_layer; ++l) {
    VLayer* L = &v->layers[l];
    char nm[128];
#define VN(suf) (snprintf(nm, sizeof nm, "v.blk.%zu." suf, l), nm)
    if (vec_try(g, VN("ln1.weight"), C.hidden, 1, &L->ln1_w, err, errlen)) goto fail;
    if (vec_try(g, VN("ln1.bias"), C.hidden, 0, &L->ln1_b, err, errlen)) goto fail;
    if (vec_try(g, VN("ln2.weight"), C.hidden, 1, &L->ln2_w, err, errlen)) goto fail;
    if (vec_try(g, VN("ln2.bias"), C.hidden, 0, &L->ln2_b, err, errlen)) goto fail;
    if (mat_square(g, VN("attn_q.weight"), C.hidden, &L->q, err, errlen)) goto fail;
    if (mat_square(g, VN("attn_k.weight"), C.hidden, &L->k, err, errlen)) goto fail;
    if (mat_square(g, VN("attn_v.weight"), C.hidden, &L->v, err, errlen)) goto fail;
    if (mat_square(g, VN("attn_out.weight"), C.hidden, &L->o, err, errlen)) goto fail;
    if (vec_try(g, VN("attn_q.bias"), C.hidden, 0, &L->bq, err, errlen)) goto fail;
    if (vec_try(g, VN("attn_k.bias"), C.hidden, 0, &L->bk, err, errlen)) goto fail;
    if (vec_try(g, VN("attn_v.bias"), C.hidden, 0, &L->bv, err, errlen)) goto fail;
    if (vec_try(g, VN("attn_out.bias"), C.hidden, 0, &L->bo, err, errlen)) goto fail;
    if (mat_load(g, VN("ffn_up.weight"), &L->up, err, errlen)) goto fail;
    if (L->up.rows != C.inter || L->up.cols != C.hidden) {
      snprintf(err, errlen, "vision: ffn_up[%zu] is %zux%zu, expected %zux%zu", l,
               L->up.rows, L->up.cols, C.inter, C.hidden);
      goto fail;
    }
    if (mat_load(g, VN("ffn_down.weight"), &L->down, err, errlen)) goto fail;
    if (L->down.rows != C.hidden || L->down.cols != C.inter) {
      snprintf(err, errlen, "vision: ffn_down[%zu] is %zux%zu, expected %zux%zu", l,
               L->down.rows, L->down.cols, C.hidden, C.inter);
      goto fail;
    }
    if (vec_try(g, VN("ffn_up.bias"), C.inter, 0, &L->bup, err, errlen)) goto fail;
    if (vec_try(g, VN("ffn_down.bias"), C.hidden, 0, &L->bdown, err, errlen)) goto fail;
#undef VN
  }

  /* projector: mm.0 -> GELU -> mm.2 (LLaVA "mlp"); mm.2 absent => single layer. */
  if (mat_load(g, "mm.0.weight", &v->mm0, err, errlen)) goto fail;
  if (v->mm0.cols != C.hidden) {
    snprintf(err, errlen, "vision: mm.0.weight cols %zu != hidden %zu", v->mm0.cols,
             C.hidden);
    goto fail;
  }
  v->proj0 = v->mm0.rows;
  if (vec_try(g, "mm.0.bias", v->proj0, 0, &v->mm0_b, err, errlen)) goto fail;

  size_t proj_dim;
  if (gguf_tensor(g, "mm.2.weight")) {
    v->has_mm2 = 1;
    if (mat_load(g, "mm.2.weight", &v->mm2, err, errlen)) goto fail;
    if (v->mm2.cols != v->proj0) {
      snprintf(err, errlen, "vision: mm.2.weight cols %zu != mm.0 rows %zu",
               v->mm2.cols, v->proj0);
      goto fail;
    }
    if (vec_try(g, "mm.2.bias", v->mm2.rows, 0, &v->mm2_b, err, errlen)) goto fail;
    proj_dim = v->mm2.rows;
  } else {
    proj_dim = v->proj0;
  }
  if (C.proj_dim && C.proj_dim != proj_dim) {
    snprintf(err, errlen, "vision: projector output %zu != projection_dim %zu",
             proj_dim, C.proj_dim);
    goto fail;
  }
  v->cfg.proj_dim = proj_dim;

  *out = v;
  return 0;

fail:
  vision_free(v);
  return -1;
}

void vision_free(VisionEncoder* v) {
  if (!v) return;
  free(v->patch_b);
  free(v->class_embd);
  free(v->pos_embd);
  free(v->pre_ln_w);
  free(v->pre_ln_b);
  free(v->post_ln_w);
  free(v->post_ln_b);
  free(v->mm0_b);
  free(v->mm2_b);
  if (v->layers) {
    for (size_t l = 0; l < v->cfg.n_layer; ++l) {
      VLayer* L = &v->layers[l];
      free(L->ln1_w);
      free(L->ln1_b);
      free(L->ln2_w);
      free(L->ln2_b);
      free(L->bq);
      free(L->bk);
      free(L->bv);
      free(L->bo);
      free(L->bup);
      free(L->bdown);
    }
    free(v->layers);
  }
  oc_ctx_free(v->ctx);
  free(v);
}

const VisionConfig* vision_config(const VisionEncoder* v) { return &v->cfg; }

/* ---- forward ---------------------------------------------------------------- */

/* Y[t][r] = dot(W_row_r, X[t]) + bias[r] for t in [0, n_tokens): a linear layer
 * over a token panel, on top of the shared threaded GEMM. */
static void linear(OcCtx* c, const VMat* W, const float* bias, const float* X,
                   float* Y, size_t n_tokens) {
  oc_matmul(c, Y, W->type, W->data, W->rows, W->cols, X, n_tokens);
  if (bias)
    for (size_t t = 0; t < n_tokens; ++t)
      for (size_t r = 0; r < W->rows; ++r) Y[t * W->rows + r] += bias[r];
}

typedef struct {
  const float *Q, *K, *V;
  float* att;
  size_t nseq, hidden, head_dim;
  float scale;
} AttnJob;

/* Bidirectional scaled-dot-product attention for a contiguous set of heads (no
 * causal mask: every query attends to every key). One head's Q/K/V slice is the
 * [head_dim] window at h*head_dim in each token's [hidden] row. */
static void attn_heads(void* ctx, size_t h0, size_t h1) {
  AttnJob* j = ctx;
  float* scores = malloc(j->nseq * sizeof(float));
  if (!scores) {
    fprintf(stderr, "vision: out of memory (attention scores)\n");
    abort();
  }
  for (size_t hh = h0; hh < h1; ++hh) {
    size_t off = hh * j->head_dim;
    for (size_t i = 0; i < j->nseq; ++i) {
      const float* qi = j->Q + i * j->hidden + off;
      for (size_t s = 0; s < j->nseq; ++s)
        scores[s] = j->scale * oc_dot_f32(qi, j->K + s * j->hidden + off, j->head_dim);
      oc_softmax(scores, j->nseq);
      float* ai = j->att + i * j->hidden + off;
      for (size_t d = 0; d < j->head_dim; ++d) ai[d] = 0.0f;
      for (size_t s = 0; s < j->nseq; ++s) {
        float w = scores[s];
        const float* vs = j->V + s * j->hidden + off;
        for (size_t d = 0; d < j->head_dim; ++d) ai[d] += w * vs[d];
      }
    }
  }
  free(scores);
}

float* vision_encode(VisionEncoder* v, const float* image_chw, size_t* n_tokens,
                     size_t* dim, char* err, size_t errlen) {
  if (err && errlen) err[0] = 0;
  const VisionConfig* C = &v->cfg;
  const size_t S = C->image_size, P = C->patch_size, D = C->hidden;
  const size_t np = C->n_patches, nseq = C->n_positions, inter = C->inter;
  const size_t grid = S / P, pdim = 3 * P * P, cls = C->has_class_token ? 1 : 0;
  const size_t head_dim = D / C->n_head;
  OcCtx* ctx = v->ctx;

  float* patches = malloc(np * pdim * sizeof(float));
  float* tok = malloc(nseq * D * sizeof(float));
  float* nrm = malloc(nseq * D * sizeof(float));
  float* q = malloc(nseq * D * sizeof(float));
  float* k = malloc(nseq * D * sizeof(float));
  float* vbuf = malloc(nseq * D * sizeof(float));
  float* att = malloc(nseq * D * sizeof(float));
  float* ob = malloc(nseq * D * sizeof(float));
  float* f1 = malloc(nseq * inter * sizeof(float));
  float* mm0 = malloc(np * v->proj0 * sizeof(float));
  float* out = malloc(np * C->proj_dim * sizeof(float));
  if (!patches || !tok || !nrm || !q || !k || !vbuf || !att || !ob || !f1 || !mm0 ||
      !out) {
    verr(err, errlen, "vision: out of memory");
    goto oom;
  }

  /* 1. patchify. Each patch is flattened [in_ch][kh][kw] (channel-major), which
   * is the contiguous per-output-channel order of ggml's conv weight, so the
   * patch-embed reduces to one matmul against v.patch_embd.weight. */
  for (size_t py = 0; py < grid; ++py)
    for (size_t px = 0; px < grid; ++px) {
      float* pd = patches + (py * grid + px) * pdim;
      for (size_t ic = 0; ic < 3; ++ic)
        for (size_t kh = 0; kh < P; ++kh)
          for (size_t kw = 0; kw < P; ++kw)
            pd[(ic * P + kh) * P + kw] =
                image_chw[(ic * S + py * P + kh) * S + px * P + kw];
    }

  /* 2. patch embed -> patch token rows (leaving room for a leading class token) */
  linear(ctx, &v->patch, v->patch_b, patches, tok + cls * D, np);

  /* 3. class token, 4. learned position embeddings */
  if (cls) memcpy(tok, v->class_embd, D * sizeof(float));
  for (size_t i = 0; i < nseq * D; ++i) tok[i] += v->pos_embd[i];

  /* 5. optional pre-LayerNorm */
  if (v->pre_ln_w)
    for (size_t t = 0; t < nseq; ++t)
      layer_norm(tok + t * D, tok + t * D, v->pre_ln_w, v->pre_ln_b, D, C->eps);

  /* 6. encoder blocks (pre-norm residual; bidirectional attention) */
  float scale = 1.0f / sqrtf((float)head_dim);
  for (size_t l = 0; l < C->n_layer; ++l) {
    VLayer* L = &v->layers[l];
    for (size_t t = 0; t < nseq; ++t)
      layer_norm(nrm + t * D, tok + t * D, L->ln1_w, L->ln1_b, D, C->eps);
    linear(ctx, &L->q, L->bq, nrm, q, nseq);
    linear(ctx, &L->k, L->bk, nrm, k, nseq);
    linear(ctx, &L->v, L->bv, nrm, vbuf, nseq);
    AttnJob job = {q, k, vbuf, att, nseq, D, head_dim, scale};
    oc_parallel_for(C->n_head, attn_heads, &job);
    linear(ctx, &L->o, L->bo, att, ob, nseq);
    for (size_t i = 0; i < nseq * D; ++i) tok[i] += ob[i];

    for (size_t t = 0; t < nseq; ++t)
      layer_norm(nrm + t * D, tok + t * D, L->ln2_w, L->ln2_b, D, C->eps);
    linear(ctx, &L->up, L->bup, nrm, f1, nseq);
    if (C->use_gelu)
      for (size_t i = 0; i < nseq * inter; ++i) f1[i] = gelu_tanh(f1[i]);
    else
      for (size_t i = 0; i < nseq * inter; ++i) f1[i] = gelu_quick(f1[i]);
    linear(ctx, &L->down, L->bdown, f1, ob, nseq);
    for (size_t i = 0; i < nseq * D; ++i) tok[i] += ob[i];
  }

  /* 7. optional post-LayerNorm. ponytail: this runs all layers then post_ln;
   * LLaVA's "use layer -2, skip post_ln" feature-layer selection is NOT applied
   * (documented gap — needs a real mmproj to verify against). */
  if (v->post_ln_w)
    for (size_t t = 0; t < nseq; ++t)
      layer_norm(tok + t * D, tok + t * D, v->post_ln_w, v->post_ln_b, D, C->eps);

  /* 8. projector over the patch tokens only (class token dropped, as LLaVA does) */
  linear(ctx, &v->mm0, v->mm0_b, tok + cls * D, mm0, np);
  for (size_t i = 0; i < np * v->proj0; ++i) mm0[i] = gelu_tanh(mm0[i]);
  if (v->has_mm2)
    linear(ctx, &v->mm2, v->mm2_b, mm0, out, np);
  else
    memcpy(out, mm0, np * C->proj_dim * sizeof(float));

  free(patches);
  free(tok);
  free(nrm);
  free(q);
  free(k);
  free(vbuf);
  free(att);
  free(ob);
  free(f1);
  free(mm0);
  *n_tokens = np;
  *dim = C->proj_dim;
  return out;

oom:
  free(patches);
  free(tok);
  free(nrm);
  free(q);
  free(k);
  free(vbuf);
  free(att);
  free(ob);
  free(f1);
  free(mm0);
  free(out);
  return NULL;
}
