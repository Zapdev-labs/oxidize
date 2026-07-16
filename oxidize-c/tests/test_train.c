/* tests/test_train.c — the acceptance suite for src/train.c.
 *
 * The centerpiece is gradient checking: for BOTH the linear softmax classifier
 * and the LoRA adapter, every analytic gradient is compared against a central
 * finite difference of an INDEPENDENT double-precision reference forward
 * (reimplemented here, not calling train.c's forward). A wrong backward — a bad
 * sign, a missing term, a stray factor — mis-trains silently; this catches it.
 * Two convergence runs then show the losses actually fall, and the LoRA GGUF
 * export is re-opened with gguf_open to prove it wrote a valid file. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/gguf.h"
#include "../src/train.h"
#include "tests.h"

/* deterministic uniform [-1, 1). */
static float frand(uint32_t* s) {
  *s = *s * 1664525u + 1013904223u;
  return (float)(*s >> 8) / 8388608.0f - 1.0f;
}

static float* alloc_f(size_t n) {
  float* p = (float*)malloc((n ? n : 1) * sizeof(float));
  CHECK(p != NULL);
  return p;
}

/* ---- independent double references (NOT train.c's forward) ----------------- */

static double ref_clf_loss(const float* w, const float* b, size_t F, size_t C,
                           const float* X, const int32_t* labels, size_t n) {
  double* lg = (double*)malloc(C * sizeof(double));
  CHECK(lg != NULL);
  double sum = 0.0;
  for (size_t k = 0; k < n; ++k) {
    const float* x = X + k * F;
    for (size_t c = 0; c < C; ++c) {
      double z = b[c];
      for (size_t f = 0; f < F; ++f) z += (double)w[c * F + f] * (double)x[f];
      lg[c] = z;
    }
    double mx = lg[0];
    for (size_t c = 1; c < C; ++c)
      if (lg[c] > mx) mx = lg[c];
    double se = 0.0;
    for (size_t c = 0; c < C; ++c) se += exp(lg[c] - mx);
    sum += (mx + log(se)) - lg[labels[k]];
  }
  free(lg);
  return sum / (double)n;
}

static double ref_lora_loss(const float* a, const float* b, const float* w,
                            float scale, size_t in, size_t out, size_t R,
                            const float* x, const float* t) {
  double* h = (double*)malloc(R * sizeof(double));
  CHECK(h != NULL);
  for (size_t r = 0; r < R; ++r) {
    double z = 0.0;
    for (size_t i = 0; i < in; ++i) z += (double)a[r * in + i] * (double)x[i];
    h[r] = z;
  }
  double loss = 0.0;
  for (size_t o = 0; o < out; ++o) {
    double base = 0.0;
    if (w)
      for (size_t i = 0; i < in; ++i) base += (double)w[o * in + i] * (double)x[i];
    double bh = 0.0;
    for (size_t r = 0; r < R; ++r) bh += (double)b[o * R + r] * h[r];
    double y = base + (double)scale * bh;
    double d = y - (double)t[o];
    loss += 0.5 * d * d;
  }
  free(h);
  return loss;
}

/* allclose(atol,rtol) hard check + rel-error tracking for the diagnostic. */
static void cmp_grad(double num, double ana, double atol, double rtol,
                     double* max_abs, double* max_rel) {
  double d = fabs(num - ana);
  if (d > *max_abs) *max_abs = d;
  if (fabs(num) > 1e-3) {
    double rel = d / fabs(num);
    if (rel > *max_rel) *max_rel = rel;
  }
  if (!(d <= atol + rtol * fabs(num))) {
    fprintf(stderr, "GRAD MISMATCH: num=%.9g analytic=%.9g |Δ|=%.3g\n", num, ana, d);
    CHECK(0);
  }
}

/* ---- 1. classifier gradient check ----------------------------------------- */

static void test_clf_gradcheck(void) {
  const size_t F = 4, C = 3, n = 6;
  TrainClf c;
  char err[128];
  CHECK(train_clf_init(&c, F, C, 20260714ULL, err, sizeof err) == 0);
  for (size_t k = 0; k < C; ++k) c.b[k] = 0.15f * (float)k - 0.2f; /* nonzero bias */

  float* X = alloc_f(n * F);
  int32_t* labels = (int32_t*)malloc(n * sizeof(int32_t));
  CHECK(labels != NULL);
  uint32_t s = 0x51ed2701u;
  for (size_t i = 0; i < n * F; ++i) X[i] = frand(&s);
  for (size_t k = 0; k < n; ++k) labels[k] = (int32_t)(k % C);

  train_clf_backward(&c, X, labels, n); /* fills c.gw, c.gb */

  const double eps = 1e-3, atol = 1e-4, rtol = 1e-3;
  double max_abs = 0, max_rel = 0;
  for (size_t idx = 0; idx < C * F; ++idx) {
    float save = c.w[idx];
    c.w[idx] = save + (float)eps;
    double lp = ref_clf_loss(c.w, c.b, F, C, X, labels, n);
    c.w[idx] = save - (float)eps;
    double lm = ref_clf_loss(c.w, c.b, F, C, X, labels, n);
    c.w[idx] = save;
    cmp_grad((lp - lm) / (2 * eps), c.gw[idx], atol, rtol, &max_abs, &max_rel);
  }
  for (size_t idx = 0; idx < C; ++idx) {
    float save = c.b[idx];
    c.b[idx] = save + (float)eps;
    double lp = ref_clf_loss(c.w, c.b, F, C, X, labels, n);
    c.b[idx] = save - (float)eps;
    double lm = ref_clf_loss(c.w, c.b, F, C, X, labels, n);
    c.b[idx] = save;
    cmp_grad((lp - lm) / (2 * eps), c.gb[idx], atol, rtol, &max_abs, &max_rel);
  }
  printf("ok train clf gradcheck (%zu params, max|Δ|=%.2e, max rel=%.2e)\n",
         C * F + C, max_abs, max_rel);
  free(X);
  free(labels);
  train_clf_free(&c);
}

/* ---- 2. LoRA gradient check (MSE objective) ------------------------------- */

static void test_lora_gradcheck(void) {
  const size_t in = 4, out = 3, R = 2;
  const float alpha = 4.0f;
  float* W = alloc_f(out * in);
  float* x = alloc_f(in);
  float* t = alloc_f(out);
  uint32_t s = 0x0badf00du;
  for (size_t i = 0; i < out * in; ++i) W[i] = frand(&s);
  for (size_t i = 0; i < in; ++i) x[i] = frand(&s);
  for (size_t i = 0; i < out; ++i) t[i] = frand(&s);

  TrainLora l;
  char err[128];
  CHECK(train_lora_init(&l, in, out, R, alpha, W, 424242ULL, err, sizeof err) == 0);
  /* B is 0 at init, which would make dL/dA identically 0 — set it nonzero so
   * the A gradients are a real test (mirrors the Rust gradient-check). */
  for (size_t i = 0; i < out * R; ++i) l.b[i] = 0.05f * ((float)(i % 7) - 3.0f);

  float* gy = alloc_f(out);
  train_lora_mse(&l, x, t, gy); /* grad_y = y - t */
  train_lora_zero_grad(&l);
  train_lora_backward(&l, x, gy);

  const double eps = 1e-3, atol = 1e-4, rtol = 1e-3;
  double max_abs = 0, max_rel = 0;
  for (size_t idx = 0; idx < R * in; ++idx) {
    float save = l.a[idx];
    l.a[idx] = save + (float)eps;
    double lp = ref_lora_loss(l.a, l.b, W, l.scale, in, out, R, x, t);
    l.a[idx] = save - (float)eps;
    double lm = ref_lora_loss(l.a, l.b, W, l.scale, in, out, R, x, t);
    l.a[idx] = save;
    cmp_grad((lp - lm) / (2 * eps), l.ga[idx], atol, rtol, &max_abs, &max_rel);
  }
  for (size_t idx = 0; idx < out * R; ++idx) {
    float save = l.b[idx];
    l.b[idx] = save + (float)eps;
    double lp = ref_lora_loss(l.a, l.b, W, l.scale, in, out, R, x, t);
    l.b[idx] = save - (float)eps;
    double lm = ref_lora_loss(l.a, l.b, W, l.scale, in, out, R, x, t);
    l.b[idx] = save;
    cmp_grad((lp - lm) / (2 * eps), l.gb[idx], atol, rtol, &max_abs, &max_rel);
  }
  printf("ok train lora gradcheck (%zu params, max|Δ|=%.2e, max rel=%.2e)\n",
         R * in + out * R, max_abs, max_rel);
  free(W);
  free(x);
  free(t);
  free(gy);
  train_lora_free(&l);
}

/* ---- 3. classifier converges on a linearly separable set ------------------ */

static void test_clf_converges(void) {
  /* 11x11 grid in [-1,1]^2, label = (x + y > 0): linearly separable. */
  const size_t F = 2, C = 2;
  size_t N = 0;
  float* X = alloc_f(11 * 11 * F);
  int32_t* labels = (int32_t*)malloc(11 * 11 * sizeof(int32_t));
  CHECK(labels != NULL);
  for (int ix = -5; ix <= 5; ++ix)
    for (int iy = -5; iy <= 5; ++iy) {
      float xf = (float)ix / 5.0f, yf = (float)iy / 5.0f;
      X[N * F] = xf;
      X[N * F + 1] = yf;
      labels[N] = (xf + yf > 0.0f) ? 1 : 0;
      ++N;
    }

  TrainClf c;
  char err[128];
  CHECK(train_clf_init(&c, F, C, 7ULL, err, sizeof err) == 0);
  TrainAdam opt = train_adam_new(0.1f, 0.0f);
  double first = train_clf_loss(&c, X, labels, N);
  for (int e = 0; e < 300; ++e) train_clf_step(&c, X, labels, N, &opt);
  double last = train_clf_loss(&c, X, labels, N);

  size_t correct = 0;
  for (size_t k = 0; k < N; ++k)
    if (train_clf_predict(&c, X + k * F) == (size_t)labels[k]) ++correct;
  double acc = (double)correct / (double)N;

  CHECK(last < first);
  CHECK(acc >= 0.95);
  printf("ok train clf converges (loss %.4f -> %.4f, acc=%.3f)\n", first, last, acc);
  free(X);
  free(labels);
  train_clf_free(&c);
}

/* ---- 4. LoRA fits a rank-r target ----------------------------------------- */

static void test_lora_converges(void) {
  const size_t in = 6, out = 4, R = 2, N = 64;
  const float alpha = 2.0f;
  float* W = alloc_f(out * in);
  uint32_t s = 0x1a2b3c4du;
  for (size_t i = 0; i < out * in; ++i) W[i] = frand(&s);

  /* Ground-truth adapter (same frozen W, same rank/scale) generates targets, so
   * a fresh adapter can in principle fit the data exactly. */
  TrainLora gt;
  char err[128];
  CHECK(train_lora_init(&gt, in, out, R, alpha, W, 111ULL, err, sizeof err) == 0);
  for (size_t i = 0; i < out * R; ++i) gt.b[i] = 0.3f * frand(&s);

  float* X = alloc_f(N * in);
  float* T = alloc_f(N * out);
  for (size_t i = 0; i < N * in; ++i) X[i] = frand(&s);
  for (size_t k = 0; k < N; ++k) train_lora_forward(&gt, X + k * in, T + k * out);

  TrainLora l;
  CHECK(train_lora_init(&l, in, out, R, alpha, W, 999ULL, err, sizeof err) == 0);
  float* gy = alloc_f(out);
  double first = 0.0;
  for (size_t k = 0; k < N; ++k)
    first += train_lora_mse(&l, X + k * in, T + k * out, gy);
  first /= (double)N;

  TrainAdam opt = train_adam_new(0.02f, 0.0f);
  double last = 0.0;
  for (int e = 0; e < 3000; ++e) last = train_lora_mse_step(&l, X, T, N, &opt);

  CHECK(last < first * 0.01);
  CHECK(last < 1e-3);
  printf("ok train lora converges (mse %.4e -> %.4e)\n", first, last);
  free(W);
  free(X);
  free(T);
  free(gy);
  train_lora_free(&gt);
  train_lora_free(&l);
}

/* ---- 5. LoRA GGUF export re-opens as a valid GGUF ------------------------- */

static void test_lora_gguf_export(void) {
  const size_t in = 8, out = 5, R = 3;
  TrainLora l;
  char err[128];
  CHECK(train_lora_init(&l, in, out, R, 6.0f, NULL, 55ULL, err, sizeof err) == 0);
  for (size_t i = 0; i < out * R; ++i) l.b[i] = 0.1f * (float)i;

  char path[] = "/tmp/oc-lora-XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  close(fd);
  CHECK(train_lora_export_gguf(&l, path, err, sizeof err) == 0);

  GgufFile g;
  char gerr[256];
  CHECK(gguf_open(&g, path, gerr, sizeof gerr) == 0);
  CHECK(g.version == 3);
  const GgufTensorInfo* ta = gguf_tensor(&g, "lora.a");
  const GgufTensorInfo* tb = gguf_tensor(&g, "lora.b");
  CHECK(ta && tb);
  CHECK(ta->ggml_type == 0 /* F32 */ && tb->ggml_type == 0);
  CHECK(ta->dims[0] == in && ta->dims[1] == R);   /* ggml [cols, rows] */
  CHECK(tb->dims[0] == R && tb->dims[1] == out);
  uint32_t rank = 0;
  CHECK(gguf_get_u32(&g, "lora.rank", &rank) && rank == R);
  /* payload of A matches what we exported */
  const float* fa = (const float*)ta->data;
  for (size_t i = 0; i < R * in; ++i) CHECK(fa[i] == l.a[i]);

  gguf_close(&g);
  unlink(path);
  train_lora_free(&l);
  printf("ok train lora gguf export (reopened, tensors verified)\n");
}

void test_train(void) {
  test_clf_gradcheck();
  test_lora_gradcheck();
  test_clf_converges();
  test_lora_converges();
  test_lora_gguf_export();
}
