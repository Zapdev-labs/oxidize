/* Minimal, gradient-checked CPU trainers.
 *
 *   TrainClf  — a linear softmax classifier (logits = W x + b, softmax,
 *               cross-entropy). The pure-C equivalent of oxidize-train's
 *               CSV -> classifier path.
 *   TrainLora — a LoRA adapter over a FROZEN linear weight
 *               (y = W x + scale * B (A x); W fixed, A/B trained). The core of
 *               oxidize-finetuning's QLoRA adapter, isolated into one trainable
 *               linear so it is testable without a full transformer backward.
 *
 * Forward dot products go through tensor.h's oc_dot_f32; the softmax /
 * cross-entropy backward and the LoRA backward are hand-derived here. Every
 * analytic gradient is checked against central finite differences in
 * tests/test_train.c — a wrong backward silently mis-trains, so that check is
 * the acceptance criterion, not a nicety.
 *
 * Errors follow the project convention: functions that can fail return 0 on
 * success or -1 with a message written into (err, errlen). Training steps
 * cannot fail and return the (double) mean loss so callers can watch it fall.
 */
#ifndef OC_TRAIN_H
#define OC_TRAIN_H

#include <stddef.h>
#include <stdint.h>

/* ---- AdamW ----------------------------------------------------------------
 * Decoupled weight decay, bias-corrected first/second moments. beta1=0.9,
 * beta2=0.999, eps=1e-8 — identical to oxidize-finetuning/src/fused.rs and
 * oxidize-train's AdamW. Determinism comes from the seeded parameter init plus
 * a fixed data order; the optimizer itself has no randomness. */
typedef struct {
  float lr;
  float wd;
  float beta1;
  float beta2;
  float eps;
  long step; /* advanced once per optimizer step, BEFORE the update */
} TrainAdam;

TrainAdam train_adam_new(float lr, float wd);

/* One AdamW update of p[0..n) from g[0..n); m,v are this tensor's moment
 * buffers. o->step must already be advanced for this update (the bias
 * correction reads it). With apply_wd, p is scaled by (1 - lr*wd) first. */
void train_adamw(float* p, const float* g, float* m, float* v, size_t n,
                 const TrainAdam* o, int apply_wd);

/* ---- linear softmax classifier -------------------------------------------- */
typedef struct {
  size_t features;
  size_t classes;
  float* w; /* [classes][features], row-major */
  float* b; /* [classes] */
  float* gw;
  float* gb;
  float* mw;
  float* vw;
  float* mb;
  float* vb;
} TrainClf;

/* Xavier-ish seeded weight init, bias = 0. Allocates; free with train_clf_free. */
int train_clf_init(TrainClf* c, size_t features, size_t classes, uint64_t seed,
                   char* err, size_t errlen);
void train_clf_free(TrainClf* c);

/* logits[0..classes) = W x + b (via oc_dot_f32). */
void train_clf_logits(const TrainClf* c, const float* x, float* logits);
/* argmax over the logits of one feature row. */
size_t train_clf_predict(const TrainClf* c, const float* x);

/* Mean cross-entropy over n rows of X ([n][features]) with int labels, in
 * double. No gradients. */
double train_clf_loss(const TrainClf* c, const float* X, const int32_t* labels,
                      size_t n);
/* Zeroes gw/gb, accumulates the exact softmax-CE gradient (mean over the
 * batch), returns the mean loss. No optimizer step — this is what the finite-
 * difference check compares against. */
double train_clf_backward(TrainClf* c, const float* X, const int32_t* labels,
                          size_t n);
/* backward + one AdamW step (advances o->step). Returns the mean loss. */
double train_clf_step(TrainClf* c, const float* X, const int32_t* labels,
                      size_t n, TrainAdam* o);

/* ---- LoRA adapter over a frozen weight ------------------------------------ */
typedef struct {
  size_t in_dim;
  size_t out_dim;
  size_t rank;
  float scale;    /* alpha / rank */
  const float* w; /* FROZEN [out_dim][in_dim], BORROWED (NULL => zero base) */
  float* a;       /* [rank][in_dim] */
  float* b;       /* [out_dim][rank] */
  float* ga;
  float* gb;
  float* ma;
  float* va;
  float* mb;
  float* vb;
} TrainLora;

/* A ~ U(-1,1)/sqrt(rank) (seeded), B = 0 so the adapter is the identity at init
 * (standard LoRA). `w` is borrowed, not copied. */
int train_lora_init(TrainLora* l, size_t in_dim, size_t out_dim, size_t rank,
                    float alpha, const float* w, uint64_t seed, char* err,
                    size_t errlen);
void train_lora_free(TrainLora* l);

/* y[0..out_dim) = W x + scale * B (A x) (via oc_dot_f32). */
void train_lora_forward(const TrainLora* l, const float* x, float* y);
void train_lora_zero_grad(TrainLora* l);
/* Accumulate dL/dA, dL/dB for one row given dL/dy (grad_y[0..out_dim)). The
 * objective is the caller's — pass grad_y = (y - t) for MSE, or a softmax-CE
 * gradient, etc. */
void train_lora_backward(TrainLora* l, const float* x, const float* grad_y);
/* MSE objective 0.5*||y - t||^2 for one row: returns the loss and writes
 * grad_y = (y - t). */
double train_lora_mse(const TrainLora* l, const float* x, const float* t,
                      float* grad_y);
/* Fit helper: over n rows (X:[n][in_dim], T:[n][out_dim]) accumulate the mean
 * MSE gradient, take one AdamW step, return the mean loss. */
double train_lora_mse_step(TrainLora* l, const float* X, const float* T,
                           size_t n, TrainAdam* o);

/* Export A and B as F32 tensors in a minimal, valid GGUF v3 (reuses
 * tools/gguf_write.c). ggml dims are [cols, rows], so A -> "lora.a"
 * [in_dim, rank] and B -> "lora.b" [rank, out_dim]. Returns 0, or -1 with a
 * message in (err, errlen). */
int train_lora_export_gguf(const TrainLora* l, const char* path, char* err,
                           size_t errlen);

#endif
