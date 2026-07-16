/* oxidize-c-train — CPU trainer CLI.
 *
 *   oxidize-c-train --csv data.csv [--epochs N] [--lr F] [--wd F] [--batch N]
 *                   [--seed N] [--label-col I]
 * Loads a CSV of numeric features + an integer class label (last column by
 * default, or --label-col), trains the linear softmax classifier from
 * src/train.c, and prints the mean cross-entropy each epoch so you can watch it
 * fall, then the final training accuracy.
 *
 *   oxidize-c-train --lora [--lora-out adapter.gguf] [--rank R] [--seed N]
 * Runs a self-contained LoRA fitting demo: a frozen random base weight and a
 * rank-r target, fit by a trained adapter, MSE reported as it falls. With
 * --lora-out, the trained adapter is written as a GGUF (A, B as F32).
 *
 * The heavy math lives in src/train.c and is gradient-checked in
 * tests/test_train.c; this file is just I/O and the training loops. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/train.h"

/* ---- CSV loading ---------------------------------------------------------- */

static char* trim(char* s) {
  while (*s == ' ' || *s == '\t' || *s == '\r') ++s;
  char* e = s + strlen(s);
  while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) *--e = 0;
  return s;
}

/* Parse one already-split row: features (all columns but label_col) into
 * feat[0..*nf), label into *lab. Returns 0, or -1 if any cell fails to parse. */
static int parse_row(char** cols, size_t ncol, size_t label_col, float* feat,
                     size_t* nf, int32_t* lab) {
  size_t w = 0;
  for (size_t i = 0; i < ncol; ++i) {
    char* tok = trim(cols[i]);
    char* end = NULL;
    if (i == label_col) {
      long v = strtol(tok, &end, 10);
      if (end == tok || *end != 0 || v < 0) return -1;
      *lab = (int32_t)v;
    } else {
      float v = strtof(tok, &end);
      if (end == tok || *end != 0) return -1;
      feat[w++] = v;
    }
  }
  *nf = w;
  return 0;
}

static int load_csv(const char* path, long label_col_opt, float** out_X,
                    int32_t** out_Y, size_t* out_n, size_t* out_F, size_t* out_C,
                    char* err, size_t errlen) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    snprintf(err, errlen, "cannot open %s", path);
    return -1;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz < 0) {
    fclose(f);
    snprintf(err, errlen, "ftell failed on %s", path);
    return -1;
  }
  char* buf = malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    snprintf(err, errlen, "oom reading %s", path);
    return -1;
  }
  size_t got = fread(buf, 1, (size_t)sz, f);
  buf[got] = 0;
  fclose(f);

  float* X = NULL;
  int32_t* Y = NULL;
  size_t n = 0, F = 0;
  int32_t max_label = 0;
  char** cols = NULL;
  size_t cols_cap = 0;
  float* feat = NULL;
  size_t feat_cap = 0;
  int rc = -1;
  size_t line_no = 0;

  char* save = NULL;
  for (char* line = strtok_r(buf, "\n", &save); line;
       line = strtok_r(NULL, "\n", &save), ++line_no) {
    char* t = trim(line);
    if (*t == 0) continue;

    /* split into columns (reuse cols/feat buffers across rows) */
    size_t ncol = 0;
    char* sv2 = NULL;
    for (char* c = strtok_r(t, ",", &sv2); c; c = strtok_r(NULL, ",", &sv2)) {
      if (ncol == cols_cap) {
        cols_cap = cols_cap ? cols_cap * 2 : 16;
        char** nc = realloc(cols, cols_cap * sizeof(char*));
        if (!nc) {
          snprintf(err, errlen, "oom");
          goto done;
        }
        cols = nc;
      }
      cols[ncol++] = c;
    }
    if (ncol < 2) {
      if (line_no == 0) continue; /* header-ish junk on line 0 */
      snprintf(err, errlen, "line %zu: need >=2 columns, got %zu", line_no + 1, ncol);
      goto done;
    }
    size_t label_col = label_col_opt >= 0 ? (size_t)label_col_opt : ncol - 1;
    if (label_col >= ncol) {
      snprintf(err, errlen, "line %zu: label-col %ld out of range", line_no + 1,
               label_col_opt);
      goto done;
    }
    if (ncol - 1 > feat_cap) {
      feat_cap = ncol - 1;
      float* nf = realloc(feat, feat_cap * sizeof(float));
      if (!nf) {
        snprintf(err, errlen, "oom");
        goto done;
      }
      feat = nf;
    }

    size_t nf = 0;
    int32_t lab = 0;
    if (parse_row(cols, ncol, label_col, feat, &nf, &lab) != 0) {
      if (line_no == 0) continue; /* first line: treat as a header, skip */
      snprintf(err, errlen, "line %zu: could not parse a numeric cell", line_no + 1);
      goto done;
    }

    if (F == 0)
      F = nf;
    else if (nf != F) {
      snprintf(err, errlen, "line %zu: %zu features, expected %zu", line_no + 1, nf, F);
      goto done;
    }
    float* nX = realloc(X, (n + 1) * F * sizeof(float));
    int32_t* nY = realloc(Y, (n + 1) * sizeof(int32_t));
    if (!nX || !nY) {
      X = nX ? nX : X;
      Y = nY ? nY : Y;
      snprintf(err, errlen, "oom");
      goto done;
    }
    X = nX;
    Y = nY;
    memcpy(X + n * F, feat, F * sizeof(float));
    Y[n] = lab;
    if (lab > max_label) max_label = lab;
    ++n;
  }

  if (n == 0) {
    snprintf(err, errlen, "%s has no data rows", path);
    goto done;
  }
  *out_X = X;
  *out_Y = Y;
  *out_n = n;
  *out_F = F;
  *out_C = (size_t)max_label + 1;
  X = NULL; /* transferred to caller */
  Y = NULL;
  rc = 0;

done:
  free(buf);
  free(cols);
  free(feat);
  free(X);
  free(Y);
  return rc;
}

/* ---- classifier training -------------------------------------------------- */

/* deterministic Fisher-Yates shuffle (splitmix64). */
static uint64_t sm(uint64_t* s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static void shuffle(size_t* idx, size_t n, uint64_t* st) {
  for (size_t i = n; i > 1; --i) {
    size_t j = (size_t)(sm(st) % i);
    size_t tmp = idx[i - 1];
    idx[i - 1] = idx[j];
    idx[j] = tmp;
  }
}

static int train_from_csv(const char* csv, long label_col, int epochs, float lr,
                          float wd, size_t batch, uint64_t seed) {
  float* X = NULL;
  int32_t* Y = NULL;
  size_t n = 0, F = 0, C = 0;
  char err[256];
  if (load_csv(csv, label_col, &X, &Y, &n, &F, &C, err, sizeof err) != 0) {
    fprintf(stderr, "csv: %s\n", err);
    return 1;
  }
  printf("loaded %s: %zu samples, %zu features, %zu classes\n", csv, n, F, C);

  TrainClf c;
  if (train_clf_init(&c, F, C, seed, err, sizeof err) != 0) {
    fprintf(stderr, "init: %s\n", err);
    free(X);
    free(Y);
    return 1;
  }
  TrainAdam opt = train_adam_new(lr, wd);
  if (batch == 0) batch = 32;

  size_t* idx = malloc(n * sizeof(size_t));
  float* Xb = malloc(batch * F * sizeof(float));
  int32_t* Yb = malloc(batch * sizeof(int32_t));
  if (!idx || !Xb || !Yb) {
    fprintf(stderr, "oom\n");
    free(idx);
    free(Xb);
    free(Yb);
    train_clf_free(&c);
    free(X);
    free(Y);
    return 1;
  }
  for (size_t i = 0; i < n; ++i) idx[i] = i;
  uint64_t st = seed ? seed : 1;

  for (int e = 0; e < epochs; ++e) {
    shuffle(idx, n, &st);
    double wl = 0.0;
    size_t seen = 0;
    for (size_t start = 0; start < n; start += batch) {
      size_t bs = start + batch <= n ? batch : n - start;
      for (size_t k = 0; k < bs; ++k) {
        memcpy(Xb + k * F, X + idx[start + k] * F, F * sizeof(float));
        Yb[k] = Y[idx[start + k]];
      }
      double loss = train_clf_step(&c, Xb, Yb, bs, &opt);
      wl += loss * (double)bs;
      seen += bs;
    }
    printf("epoch %3d/%d  loss=%.5f\n", e + 1, epochs, wl / (double)(seen ? seen : 1));
  }

  size_t correct = 0;
  for (size_t k = 0; k < n; ++k)
    if (train_clf_predict(&c, X + k * F) == (size_t)Y[k]) ++correct;
  printf("final train accuracy: %.4f (%zu/%zu)\n", (double)correct / (double)n, correct, n);

  free(idx);
  free(Xb);
  free(Yb);
  train_clf_free(&c);
  free(X);
  free(Y);
  return 0;
}

/* ---- LoRA demo ------------------------------------------------------------ */

static float dr(uint64_t* st) { return (float)((int64_t)(sm(st) >> 40) - 8388608) / 8388608.0f; }

static int lora_demo(size_t rank, uint64_t seed, const char* out_gguf) {
  const size_t in = 6, out = 4, N = 64;
  const float alpha = (float)rank; /* scale = alpha/rank = 1 */
  uint64_t st = seed ? seed : 1;
  float* W = malloc(out * in * sizeof(float));
  float* X = malloc(N * in * sizeof(float));
  float* T = malloc(N * out * sizeof(float));
  if (!W || !X || !T) {
    fprintf(stderr, "oom\n");
    free(W);
    free(X);
    free(T);
    return 1;
  }
  for (size_t i = 0; i < out * in; ++i) W[i] = dr(&st);
  for (size_t i = 0; i < N * in; ++i) X[i] = dr(&st);

  char err[256];
  TrainLora gt, l;
  if (train_lora_init(&gt, in, out, rank, alpha, W, seed + 1, err, sizeof err) != 0 ||
      train_lora_init(&l, in, out, rank, alpha, W, seed + 2, err, sizeof err) != 0) {
    fprintf(stderr, "lora init: %s\n", err);
    free(W);
    free(X);
    free(T);
    return 1;
  }
  for (size_t i = 0; i < out * rank; ++i) gt.b[i] = 0.3f * dr(&st);
  for (size_t k = 0; k < N; ++k) train_lora_forward(&gt, X + k * in, T + k * out);

  TrainAdam opt = train_adam_new(0.02f, 0.0f);
  printf("lora demo: in=%zu out=%zu rank=%zu, fitting a rank-%zu target\n", in, out,
         rank, rank);
  double last = 0.0;
  for (int e = 0; e < 2000; ++e) {
    last = train_lora_mse_step(&l, X, T, N, &opt);
    if (e == 0 || (e + 1) % 200 == 0) printf("  step %4d  mse=%.4e\n", e + 1, last);
  }
  printf("final mse: %.4e\n", last);

  int rc = 0;
  if (out_gguf) {
    if (train_lora_export_gguf(&l, out_gguf, err, sizeof err) != 0) {
      fprintf(stderr, "export: %s\n", err);
      rc = 1;
    } else {
      printf("wrote adapter GGUF: %s\n", out_gguf);
    }
  }
  train_lora_free(&gt);
  train_lora_free(&l);
  free(W);
  free(X);
  free(T);
  return rc;
}

int main(int argc, char** argv) {
  const char* csv = NULL;
  const char* lora_out = NULL;
  int epochs = 30, do_lora = 0;
  float lr = 0.05f, wd = 0.0f;
  long label_col = -1;
  size_t batch = 32, rank = 4;
  uint64_t seed = 42;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
    else if (!strcmp(argv[i], "--epochs") && i + 1 < argc) epochs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr = strtof(argv[++i], NULL);
    else if (!strcmp(argv[i], "--wd") && i + 1 < argc) wd = strtof(argv[++i], NULL);
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = (size_t)atol(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--label-col") && i + 1 < argc) label_col = atol(argv[++i]);
    else if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = (size_t)atol(argv[++i]);
    else if (!strcmp(argv[i], "--lora")) do_lora = 1;
    else if (!strcmp(argv[i], "--lora-out") && i + 1 < argc) { lora_out = argv[++i]; do_lora = 1; }
    else {
      fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }

  if (do_lora) return lora_demo(rank ? rank : 1, seed, lora_out);

  if (!csv) {
    fprintf(stderr,
            "usage: oxidize-c-train --csv data.csv [--epochs N] [--lr F] [--wd F]\n"
            "                       [--batch N] [--seed N] [--label-col I]\n"
            "       oxidize-c-train --lora [--lora-out adapter.gguf] [--rank R] [--seed N]\n");
    return 1;
  }
  return train_from_csv(csv, label_col, epochs, lr, wd, batch, seed);
}
