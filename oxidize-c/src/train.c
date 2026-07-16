/* Gradient-checked CPU trainers: a linear softmax classifier and a LoRA adapter
 * over a frozen linear weight. See train.h for the contract.
 *
 * Bottom-up: PRNG + alloc helpers -> AdamW -> classifier -> LoRA -> GGUF export.
 * All forward dot products call oc_dot_f32 (tensor.h); the softmax/CE and LoRA
 * backward passes are derived here and verified by finite differences in
 * tests/test_train.c. */
#include "train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"  /* OC_F32 for the GGUF export */
#include "tensor.h" /* oc_dot_f32 */
#include "../tools/gguf_write.h"

/* ---- seeded PRNG (splitmix64) --------------------------------------------
 * Deterministic given the seed, so a run reproduces bit-for-bit. */
static uint64_t sm_next(uint64_t* s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
/* uniform in [-1, 1). */
static float sm_unif(uint64_t* s) {
  uint32_t u = (uint32_t)(sm_next(s) >> 32);
  return (float)u / 2147483648.0f - 1.0f;
}

static float* zalloc(size_t n) { return (float*)calloc(n ? n : 1, sizeof(float)); }

/* ---- AdamW ----------------------------------------------------------------- */

TrainAdam train_adam_new(float lr, float wd) {
  TrainAdam o = {.lr = lr,
                 .wd = wd,
                 .beta1 = 0.9f,
                 .beta2 = 0.999f,
                 .eps = 1e-8f,
                 .step = 0};
  return o;
}

void train_adamw(float* p, const float* g, float* m, float* v, size_t n,
                 const TrainAdam* o, int apply_wd) {
  /* Bias corrections in double: beta^step underflows slowly and this keeps the
   * effective step size accurate for large step counts. */
  double bc1 = 1.0 - pow((double)o->beta1, (double)o->step);
  double bc2 = 1.0 - pow((double)o->beta2, (double)o->step);
  for (size_t i = 0; i < n; ++i) {
    if (apply_wd) p[i] *= (1.0f - o->lr * o->wd);
    m[i] = o->beta1 * m[i] + (1.0f - o->beta1) * g[i];
    v[i] = o->beta2 * v[i] + (1.0f - o->beta2) * g[i] * g[i];
    float mhat = (float)(m[i] / bc1);
    float vhat = (float)(v[i] / bc2);
    p[i] -= o->lr * mhat / (sqrtf(vhat) + o->eps);
  }
}

/* ---- linear softmax classifier -------------------------------------------- */

int train_clf_init(TrainClf* c, size_t features, size_t classes, uint64_t seed,
                   char* err, size_t errlen) {
  memset(c, 0, sizeof *c);
  if (features == 0 || classes == 0) {
    if (err) snprintf(err, errlen, "classifier needs features>0 and classes>0");
    return -1;
  }
  c->features = features;
  c->classes = classes;
  size_t nw = classes * features;
  c->w = zalloc(nw);
  c->b = zalloc(classes);
  c->gw = zalloc(nw);
  c->gb = zalloc(classes);
  c->mw = zalloc(nw);
  c->vw = zalloc(nw);
  c->mb = zalloc(classes);
  c->vb = zalloc(classes);
  if (!c->w || !c->b || !c->gw || !c->gb || !c->mw || !c->vw || !c->mb || !c->vb) {
    train_clf_free(c);
    if (err) snprintf(err, errlen, "classifier alloc failed");
    return -1;
  }
  /* Xavier-ish: uniform(-1,1) * sqrt(2/(features+classes)) — matches oxidize
   * -train's Linear::new. */
  float scale = sqrtf(2.0f / (float)(features + classes));
  uint64_t st = seed ? seed : 0x1234567ULL;
  for (size_t i = 0; i < nw; ++i) c->w[i] = sm_unif(&st) * scale;
  return 0;
}

void train_clf_free(TrainClf* c) {
  free(c->w);
  free(c->b);
  free(c->gw);
  free(c->gb);
  free(c->mw);
  free(c->vw);
  free(c->mb);
  free(c->vb);
  memset(c, 0, sizeof *c);
}

void train_clf_logits(const TrainClf* c, const float* x, float* logits) {
  for (size_t k = 0; k < c->classes; ++k)
    logits[k] = oc_dot_f32(c->w + k * c->features, x, c->features) + c->b[k];
}

size_t train_clf_predict(const TrainClf* c, const float* x) {
  float* lg = zalloc(c->classes);
  if (!lg) return 0;
  train_clf_logits(c, x, lg);
  size_t best = 0;
  for (size_t k = 1; k < c->classes; ++k)
    if (lg[k] > lg[best]) best = k;
  free(lg);
  return best;
}

/* Stable softmax cross-entropy in double for one row. Returns the loss
 * (logsumexp - logit[label]); if grad != NULL, grad[k] = softmax_k - [k==label]
 * (the exact dL/dlogit, before the 1/batch factor). Caller guarantees
 * label < classes. */
static double ce_softmax(const float* logits, size_t classes, int32_t label,
                         double* grad) {
  double mx = logits[0];
  for (size_t k = 1; k < classes; ++k)
    if ((double)logits[k] > mx) mx = logits[k];
  double se = 0.0;
  for (size_t k = 0; k < classes; ++k) se += exp((double)logits[k] - mx);
  double lse = mx + log(se);
  if (grad) {
    for (size_t k = 0; k < classes; ++k) {
      double p = exp((double)logits[k] - lse);
      grad[k] = p - (k == (size_t)label ? 1.0 : 0.0);
    }
  }
  return lse - (double)logits[label];
}

double train_clf_loss(const TrainClf* c, const float* X, const int32_t* labels,
                      size_t n) {
  float* lg = zalloc(c->classes);
  if (!lg) return NAN;
  double sum = 0.0;
  for (size_t k = 0; k < n; ++k) {
    train_clf_logits(c, X + k * c->features, lg);
    sum += ce_softmax(lg, c->classes, labels[k], NULL);
  }
  free(lg);
  return sum / (double)(n ? n : 1);
}

double train_clf_backward(TrainClf* c, const float* X, const int32_t* labels,
                          size_t n) {
  size_t F = c->features, C = c->classes;
  memset(c->gw, 0, C * F * sizeof(float));
  memset(c->gb, 0, C * sizeof(float));
  float* lg = zalloc(C);
  double* g = (double*)calloc(C ? C : 1, sizeof(double));
  if (!lg || !g) {
    free(lg);
    free(g);
    return NAN;
  }
  double sum = 0.0;
  for (size_t k = 0; k < n; ++k) {
    const float* x = X + k * F;
    train_clf_logits(c, x, lg);
    sum += ce_softmax(lg, C, labels[k], g);
    for (size_t cls = 0; cls < C; ++cls) {
      float gc = (float)g[cls];
      c->gb[cls] += gc;
      float* wrow = c->gw + cls * F;
      for (size_t f = 0; f < F; ++f) wrow[f] += gc * x[f];
    }
  }
  /* Mean over the batch: dL/dparam = (1/n) * accumulated. */
  float inv = 1.0f / (float)(n ? n : 1);
  for (size_t i = 0; i < C * F; ++i) c->gw[i] *= inv;
  for (size_t i = 0; i < C; ++i) c->gb[i] *= inv;
  free(lg);
  free(g);
  return sum / (double)(n ? n : 1);
}

double train_clf_step(TrainClf* c, const float* X, const int32_t* labels,
                      size_t n, TrainAdam* o) {
  double loss = train_clf_backward(c, X, labels, n);
  o->step += 1;
  train_adamw(c->w, c->gw, c->mw, c->vw, c->classes * c->features, o, 1);
  train_adamw(c->b, c->gb, c->mb, c->vb, c->classes, o, 0); /* no decay on bias */
  return loss;
}

/* ---- LoRA adapter over a frozen weight ------------------------------------ */

int train_lora_init(TrainLora* l, size_t in_dim, size_t out_dim, size_t rank,
                    float alpha, const float* w, uint64_t seed, char* err,
                    size_t errlen) {
  memset(l, 0, sizeof *l);
  if (in_dim == 0 || out_dim == 0 || rank == 0) {
    if (err) snprintf(err, errlen, "lora needs in_dim>0, out_dim>0, rank>0");
    return -1;
  }
  l->in_dim = in_dim;
  l->out_dim = out_dim;
  l->rank = rank;
  l->scale = alpha / (float)rank;
  l->w = w;
  size_t na = rank * in_dim, nb = out_dim * rank;
  l->a = zalloc(na);
  l->b = zalloc(nb);
  l->ga = zalloc(na);
  l->gb = zalloc(nb);
  l->ma = zalloc(na);
  l->va = zalloc(na);
  l->mb = zalloc(nb);
  l->vb = zalloc(nb);
  if (!l->a || !l->b || !l->ga || !l->gb || !l->ma || !l->va || !l->mb || !l->vb) {
    train_lora_free(l);
    if (err) snprintf(err, errlen, "lora alloc failed");
    return -1;
  }
  /* A ~ U(-1,1)/sqrt(rank); B stays 0 so the adapter is identity at init. */
  float sc = 1.0f / sqrtf((float)rank);
  uint64_t st = seed ? seed : 0x89abcdefULL;
  for (size_t i = 0; i < na; ++i) l->a[i] = sm_unif(&st) * sc;
  return 0;
}

void train_lora_free(TrainLora* l) {
  free(l->a);
  free(l->b);
  free(l->ga);
  free(l->gb);
  free(l->ma);
  free(l->va);
  free(l->mb);
  free(l->vb);
  memset(l, 0, sizeof *l);
}

void train_lora_forward(const TrainLora* l, const float* x, float* y) {
  float* h = zalloc(l->rank);
  if (!h) return;
  for (size_t r = 0; r < l->rank; ++r)
    h[r] = oc_dot_f32(l->a + r * l->in_dim, x, l->in_dim);
  for (size_t o = 0; o < l->out_dim; ++o) {
    float base = l->w ? oc_dot_f32(l->w + o * l->in_dim, x, l->in_dim) : 0.0f;
    y[o] = base + l->scale * oc_dot_f32(l->b + o * l->rank, h, l->rank);
  }
  free(h);
}

void train_lora_zero_grad(TrainLora* l) {
  memset(l->ga, 0, l->rank * l->in_dim * sizeof(float));
  memset(l->gb, 0, l->out_dim * l->rank * sizeof(float));
}

void train_lora_backward(TrainLora* l, const float* x, const float* grad_y) {
  size_t in = l->in_dim, out = l->out_dim, R = l->rank;
  float scale = l->scale;
  float* h = zalloc(R);
  float* gh = zalloc(R);
  if (!h || !gh) {
    free(h);
    free(gh);
    return;
  }
  for (size_t r = 0; r < R; ++r) h[r] = oc_dot_f32(l->a + r * in, x, in);
  /* grad_b[o][r] += scale * grad_y[o] * hidden[r] */
  for (size_t o = 0; o < out; ++o) {
    float gs = scale * grad_y[o];
    float* gbrow = l->gb + o * R;
    for (size_t r = 0; r < R; ++r) gbrow[r] += gs * h[r];
  }
  /* grad_hidden[r] = scale * sum_o grad_y[o] * B[o][r] */
  for (size_t o = 0; o < out; ++o) {
    float gs = scale * grad_y[o];
    const float* brow = l->b + o * R;
    for (size_t r = 0; r < R; ++r) gh[r] += gs * brow[r];
  }
  /* grad_a[r][i] += grad_hidden[r] * x[i] */
  for (size_t r = 0; r < R; ++r) {
    float ghr = gh[r];
    float* garow = l->ga + r * in;
    for (size_t i = 0; i < in; ++i) garow[i] += ghr * x[i];
  }
  free(h);
  free(gh);
}

double train_lora_mse(const TrainLora* l, const float* x, const float* t,
                      float* grad_y) {
  float* y = zalloc(l->out_dim);
  if (!y) return NAN;
  train_lora_forward(l, x, y);
  double loss = 0.0;
  for (size_t o = 0; o < l->out_dim; ++o) {
    float d = y[o] - t[o];
    grad_y[o] = d;
    loss += 0.5 * (double)d * (double)d;
  }
  free(y);
  return loss;
}

double train_lora_mse_step(TrainLora* l, const float* X, const float* T,
                           size_t n, TrainAdam* o) {
  train_lora_zero_grad(l);
  float* gy = zalloc(l->out_dim);
  if (!gy) return NAN;
  float invn = 1.0f / (float)(n ? n : 1);
  double sum = 0.0;
  for (size_t k = 0; k < n; ++k) {
    const float* x = X + k * l->in_dim;
    const float* t = T + k * l->out_dim;
    sum += train_lora_mse(l, x, t, gy);
    for (size_t j = 0; j < l->out_dim; ++j) gy[j] *= invn; /* mean gradient */
    train_lora_backward(l, x, gy);
  }
  free(gy);
  o->step += 1;
  train_adamw(l->a, l->ga, l->ma, l->va, l->rank * l->in_dim, o, 1);
  train_adamw(l->b, l->gb, l->mb, l->vb, l->out_dim * l->rank, o, 1);
  return sum * (double)invn;
}

/* ---- GGUF export ---------------------------------------------------------- */

int train_lora_export_gguf(const TrainLora* l, const char* path, char* err,
                           size_t errlen) {
  GwTensor ts[2];
  memset(ts, 0, sizeof ts);
  /* ggml dims are [cols, rows]: A is [rank][in_dim] row-major -> [in_dim, rank];
   * B is [out_dim][rank] -> [rank, out_dim]. */
  ts[0].name = "lora.a";
  ts[0].n_dims = 2;
  ts[0].dims[0] = l->in_dim;
  ts[0].dims[1] = l->rank;
  ts[0].type = OC_F32;
  ts[0].size = (uint64_t)l->rank * l->in_dim * sizeof(float);
  ts[1].name = "lora.b";
  ts[1].n_dims = 2;
  ts[1].dims[0] = l->rank;
  ts[1].dims[1] = l->out_dim;
  ts[1].type = OC_F32;
  ts[1].size = (uint64_t)l->out_dim * l->rank * sizeof(float);

  GgufKv kvs[3];
  memset(kvs, 0, sizeof kvs);
  kvs[0].key = (char*)"general.architecture";
  kvs[0].val.kind = GGUF_T_STRING;
  kvs[0].val.v.str.ptr = "lora";
  kvs[0].val.v.str.len = 4;
  kvs[1].key = (char*)"lora.rank";
  kvs[1].val.kind = GGUF_T_U32;
  kvs[1].val.v.u = (uint64_t)l->rank;
  kvs[2].key = (char*)"lora.scale";
  kvs[2].val.kind = GGUF_T_F32;
  kvs[2].val.v.f = (double)l->scale;

  GwWriter w;
  memset(&w, 0, sizeof w);
  if (gw_open(&w, path, kvs, 3, 32, ts, 2, -1) != 0) {
    if (err) snprintf(err, errlen, "gw_open failed for %s", path);
    return -1;
  }
  if (gw_tensor(&w, l->a, ts[0].size) != 0 ||
      gw_tensor(&w, l->b, ts[1].size) != 0) {
    gw_close(&w);
    if (err) snprintf(err, errlen, "gw_tensor write failed");
    return -1;
  }
  if (gw_close(&w) != 0) {
    if (err) snprintf(err, errlen, "gw_close failed");
    return -1;
  }
  return 0;
}
