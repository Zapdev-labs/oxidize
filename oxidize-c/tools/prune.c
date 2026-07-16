/* oxidize-c-prune: GGUF -> GGUF.
 *   --input in.gguf --output out.gguf [--keep SUB]... [--drop SUB]...
 *     [--sparsity F] [--norms PATH]
 * Without --sparsity: name-filter keep/drop only; payloads copied verbatim.
 * With --sparsity in (0,1): after the same name filters, sparsify 2-D
 * attn_/ffn_ *.weight tensors per row (magnitude, or Wanda if --norms).
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/quant.h"
#include "../src/tensor.h"
#include "gguf_write.h"

static int matches(const char* name, const char* const* subs, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (strstr(name, subs[i])) return 1;
  return 0;
}

static int ends_with(const char* s, const char* suf) {
  size_t ls = strlen(s), lf = strlen(suf);
  return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

/* 2-D linear weights under attn_/ffn_ (not 1-D norms). */
static int is_sparse_target(const GgufTensorInfo* t) {
  if (t->n_dims != 2) return 0;
  if (!ends_with(t->name, ".weight")) return 0;
  return strstr(t->name, "attn_") != NULL || strstr(t->name, "ffn_") != NULL;
}

static uint64_t n_rows(const GgufTensorInfo* t) {
  uint64_t r = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) r *= t->dims[d];
  return r;
}

int tool_prune(const char* in, const char* out, const char* const* keep,
               size_t nkeep, const char* const* drop, size_t ndrop, int verbose) {
  char err[256];
  GgufFile f;
  if (gguf_open(&f, in, err, sizeof err) != 0) {
    fprintf(stderr, "gguf_open: %s\n", err);
    return 1;
  }

  GwWriter w = {0};
  GwTensor* ts = calloc(f.n_tensors ? f.n_tensors : 1, sizeof(GwTensor));
  size_t* idx = calloc(f.n_tensors ? f.n_tensors : 1, sizeof(size_t));
  if (!ts || !idx) {
    fprintf(stderr, "oom\n");
    return 1;
  }
  int rc = 1;
  size_t nt = 0;
  for (size_t i = 0; i < f.n_tensors; ++i) {
    const GgufTensorInfo* t = &f.tensors[i];
    if (nkeep && !matches(t->name, keep, nkeep)) continue;
    if (ndrop && matches(t->name, drop, ndrop)) continue;
    uint64_t rows = n_rows(t);
    size_t rb = oc_row_bytes(t->ggml_type, t->dims[0]);
    if (rb == 0) {
      fprintf(stderr, "tensor %s: unsupported type %u\n", t->name, t->ggml_type);
      goto done;
    }
    if (!gw_data_ok(&f, t, rows * rb)) {
      fprintf(stderr, "tensor %s: data runs past end of file (truncated GGUF?)\n",
              t->name);
      goto done;
    }
    ts[nt].name = t->name;
    ts[nt].n_dims = t->n_dims;
    memcpy(ts[nt].dims, t->dims, sizeof ts[nt].dims);
    ts[nt].type = t->ggml_type;
    ts[nt].size = rows * rb;
    idx[nt++] = i;
  }

  if (gw_open(&w, out, f.kvs, f.n_kv, f.alignment, ts, nt, -1) != 0) goto done;
  for (size_t i = 0; i < nt; ++i) {
    if (gw_tensor(&w, f.tensors[idx[i]].data, ts[i].size) != 0) goto done;
    if (verbose)
      printf("kept %-40s (%" PRIu64 " B)\n", ts[i].name, ts[i].size);
  }
  rc = gw_close(&w) != 0;
  if (!rc && verbose)
    printf("done: %s (%zu of %zu tensors)\n", out, nt, f.n_tensors);

done:
  if (w.f) gw_close(&w);
  free(ts);
  free(idx);
  gguf_close(&f);
  return rc;
}

/* ---- Wanda L2-norms cache: `# <name> <ncols>` then ncols floats ------------ */

typedef struct {
  char* name;
  size_t ncols;
  float* vals;
} NormBlock;

typedef struct {
  NormBlock* blocks;
  size_t n, cap;
} NormCache;

static void norms_free(NormCache* c) {
  for (size_t i = 0; i < c->n; ++i) {
    free(c->blocks[i].name);
    free(c->blocks[i].vals);
  }
  free(c->blocks);
  c->blocks = NULL;
  c->n = c->cap = 0;
}

static const float* norms_lookup(const NormCache* c, const char* name,
                                 size_t ncols) {
  for (size_t i = 0; i < c->n; ++i) {
    if (strcmp(c->blocks[i].name, name) == 0) {
      if (c->blocks[i].ncols != ncols) return NULL; /* length mismatch */
      return c->blocks[i].vals;
    }
  }
  return NULL;
}

/* Returns 1 if `name` is present (any ncols). Used to distinguish miss vs
 * wrong length. */
static int norms_has(const NormCache* c, const char* name) {
  for (size_t i = 0; i < c->n; ++i)
    if (strcmp(c->blocks[i].name, name) == 0) return 1;
  return 0;
}

static int norms_load(NormCache* c, const char* path) {
  memset(c, 0, sizeof *c);
  FILE* fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "prune: cannot open norms %s\n", path);
    return 1;
  }
  char line[4096];
  while (fgets(line, sizeof line, fp)) {
    char* p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0' || *p == '\n' || *p == '\r') continue;
    if (*p != '#') {
      fprintf(stderr, "prune: norms: expected '# name ncols' block header\n");
      fclose(fp);
      norms_free(c);
      return 1;
    }
    ++p;
    while (*p == ' ' || *p == '\t') ++p;
    /* Skip pure comment lines that are not `# name ncols`. */
    char name[512];
    unsigned long ncols_ul = 0;
    if (sscanf(p, "%511s %lu", name, &ncols_ul) != 2 || ncols_ul == 0) continue;
    size_t ncols = (size_t)ncols_ul;
    if (c->n == c->cap) {
      size_t ncap = c->cap ? c->cap * 2 : 8;
      NormBlock* nb = realloc(c->blocks, ncap * sizeof(NormBlock));
      if (!nb) {
        fprintf(stderr, "oom\n");
        fclose(fp);
        norms_free(c);
        return 1;
      }
      c->blocks = nb;
      c->cap = ncap;
    }
    NormBlock* b = &c->blocks[c->n];
    b->name = strdup(name);
    b->ncols = ncols;
    b->vals = calloc(ncols, sizeof(float));
    if (!b->name || !b->vals) {
      fprintf(stderr, "oom\n");
      free(b->name);
      free(b->vals);
      fclose(fp);
      norms_free(c);
      return 1;
    }
    for (size_t i = 0; i < ncols; ++i) {
      if (fscanf(fp, "%f", &b->vals[i]) != 1) {
        fprintf(stderr, "prune: norms: short block for %s (need %zu floats)\n",
                name, ncols);
        free(b->name);
        free(b->vals);
        fclose(fp);
        norms_free(c);
        return 1;
      }
    }
    c->n++;
  }
  fclose(fp);
  return 0;
}

typedef struct {
  float score;
  size_t idx;
} ScoreIdx;

static int cmp_score_asc(const void* a, const void* b) {
  float sa = ((const ScoreIdx*)a)->score, sb = ((const ScoreIdx*)b)->score;
  if (sa < sb) return -1;
  if (sa > sb) return 1;
  return 0;
}

/* Zero the lowest-scoring `drop` columns in a row (keep the rest). */
static int prune_row(float* row, size_t cols, const float* norms, size_t keep) {
  if (keep >= cols) return 0;
  size_t drop = cols - keep;
  ScoreIdx* si = malloc(cols * sizeof(ScoreIdx));
  if (!si) {
    fprintf(stderr, "oom\n");
    return 1;
  }
  for (size_t j = 0; j < cols; ++j) {
    float s = fabsf(row[j]);
    if (norms) s *= norms[j];
    si[j].score = s;
    si[j].idx = j;
  }
  qsort(si, cols, sizeof(ScoreIdx), cmp_score_asc);
  for (size_t i = 0; i < drop; ++i) row[si[i].idx] = 0.0f;
  free(si);
  return 0;
}

static int sparsify_tensor(const GgufTensorInfo* t, uint8_t* dst,
                           const float* norms, float sparsity) {
  size_t cols = (size_t)t->dims[0];
  uint64_t rows = n_rows(t);
  size_t rb = oc_row_bytes(t->ggml_type, cols);
  size_t keep = (size_t)lround((1.0 - (double)sparsity) * (double)cols);
  if (keep > cols) keep = cols;

  float* row = malloc(cols * sizeof(float));
  if (!row) {
    fprintf(stderr, "oom\n");
    return 1;
  }
  for (uint64_t r = 0; r < rows; ++r) {
    if (oc_dequant_row(t->ggml_type, t->data + r * rb, row, cols) != 0) {
      fprintf(stderr, "tensor %s: dequant failed\n", t->name);
      free(row);
      return 1;
    }
    if (prune_row(row, cols, norms, keep) != 0) {
      free(row);
      return 1;
    }
    if (gw_encode_row(t->ggml_type, row, dst + r * rb, cols) != 0) {
      fprintf(stderr, "tensor %s: re-encode failed (type %u)\n", t->name,
              t->ggml_type);
      free(row);
      return 1;
    }
  }
  free(row);
  return 0;
}

int tool_prune_sparse(const char* in, const char* out, float sparsity,
                      const char* norms_path, const char* const* keep,
                      size_t nkeep, const char* const* drop, size_t ndrop,
                      int verbose) {
  if (!(sparsity > 0.0f && sparsity < 1.0f)) {
    fprintf(stderr, "prune: --sparsity must be in (0,1), got %g\n",
            (double)sparsity);
    return 1;
  }

  NormCache norms = {0};
  int have_norms = 0;
  if (norms_path) {
    if (norms_load(&norms, norms_path) != 0) return 1;
    have_norms = 1;
  }

  char err[256];
  GgufFile f;
  if (gguf_open(&f, in, err, sizeof err) != 0) {
    fprintf(stderr, "gguf_open: %s\n", err);
    norms_free(&norms);
    return 1;
  }

  GwWriter w = {0};
  GwTensor* ts = calloc(f.n_tensors ? f.n_tensors : 1, sizeof(GwTensor));
  size_t* idx = calloc(f.n_tensors ? f.n_tensors : 1, sizeof(size_t));
  int* do_sparse = calloc(f.n_tensors ? f.n_tensors : 1, sizeof(int));
  if (!ts || !idx || !do_sparse) {
    fprintf(stderr, "oom\n");
    free(ts);
    free(idx);
    free(do_sparse);
    norms_free(&norms);
    gguf_close(&f);
    return 1;
  }

  int rc = 1;
  size_t nt = 0;
  for (size_t i = 0; i < f.n_tensors; ++i) {
    const GgufTensorInfo* t = &f.tensors[i];
    if (nkeep && !matches(t->name, keep, nkeep)) continue;
    if (ndrop && matches(t->name, drop, ndrop)) continue;
    uint64_t rows = n_rows(t);
    size_t rb = oc_row_bytes(t->ggml_type, t->dims[0]);
    if (rb == 0) {
      fprintf(stderr, "tensor %s: unsupported type %u\n", t->name, t->ggml_type);
      goto done;
    }
    if (!gw_data_ok(&f, t, rows * rb)) {
      fprintf(stderr, "tensor %s: data runs past end of file (truncated GGUF?)\n",
              t->name);
      goto done;
    }
    int sparse = is_sparse_target(t);
    if (sparse && !gw_encodable(t->ggml_type)) {
      fprintf(stderr,
              "tensor %s: type %u not re-encodable for sparse prune "
              "(need F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/AL5_XS)\n",
              t->name, t->ggml_type);
      goto done;
    }
    do_sparse[nt] = sparse;
    ts[nt].name = t->name;
    ts[nt].n_dims = t->n_dims;
    memcpy(ts[nt].dims, t->dims, sizeof ts[nt].dims);
    ts[nt].type = t->ggml_type;
    ts[nt].size = rows * rb;
    idx[nt++] = i;
  }

  if (gw_open(&w, out, f.kvs, f.n_kv, f.alignment, ts, nt, -1) != 0) goto done;

  for (size_t i = 0; i < nt; ++i) {
    const GgufTensorInfo* t = &f.tensors[idx[i]];
    if (!do_sparse[i]) {
      if (gw_tensor(&w, t->data, ts[i].size) != 0) goto done;
      if (verbose)
        printf("kept %-40s copied (%" PRIu64 " B)\n", t->name, ts[i].size);
      continue;
    }

    const float* ncol = NULL;
    if (have_norms) {
      size_t cols = (size_t)t->dims[0];
      ncol = norms_lookup(&norms, t->name, cols);
      if (!ncol) {
        if (norms_has(&norms, t->name)) {
          fprintf(stderr, "prune: norms for %s have wrong ncols (want %zu)\n",
                  t->name, cols);
          goto done;
        }
        fprintf(stderr,
                "prune: warning: no norms for %s; falling back to magnitude\n",
                t->name);
      }
    }

    uint8_t* dst = malloc(ts[i].size);
    if (!dst) {
      fprintf(stderr, "oom (%" PRIu64 " B)\n", ts[i].size);
      goto done;
    }
    int e = sparsify_tensor(t, dst, ncol, sparsity);
    if (e == 0) e = gw_tensor(&w, dst, ts[i].size);
    free(dst);
    if (e != 0) goto done;
    if (verbose)
      printf("sparse %-40s %" PRIu64 "x%" PRIu64 " sparsity=%.3f %s\n", t->name,
             t->dims[0], n_rows(t), (double)sparsity,
             ncol ? "wanda" : "magnitude");
  }

  rc = gw_close(&w) != 0;
  if (!rc && verbose)
    printf("done: %s (%zu of %zu tensors, sparsity=%.3f)\n", out, nt,
           f.n_tensors, (double)sparsity);

done:
  if (w.f) gw_close(&w);
  free(ts);
  free(idx);
  free(do_sparse);
  norms_free(&norms);
  gguf_close(&f);
  return rc;
}

#ifndef OC_TOOLS_LIB
int main(int argc, char** argv) {
  const char *in = NULL, *out = NULL, *norms = NULL;
  const char* keep[64];
  const char* drop[64];
  size_t nkeep = 0, ndrop = 0;
  float sparsity = -1.0f;
  int have_sparsity = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--input") && i + 1 < argc) in = argv[++i];
    else if (!strcmp(argv[i], "--output") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--keep") && i + 1 < argc && nkeep < 64)
      keep[nkeep++] = argv[++i];
    else if (!strcmp(argv[i], "--drop") && i + 1 < argc && ndrop < 64)
      drop[ndrop++] = argv[++i];
    else if (!strcmp(argv[i], "--sparsity") && i + 1 < argc) {
      sparsity = strtof(argv[++i], NULL);
      have_sparsity = 1;
    } else if (!strcmp(argv[i], "--norms") && i + 1 < argc)
      norms = argv[++i];
    else {
      fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }
  if (!in || !out) {
    fprintf(stderr,
            "usage: oxidize-c-prune --input in.gguf --output out.gguf "
            "[--keep SUB]... [--drop SUB]... "
            "[--sparsity F] [--norms PATH]\n");
    return 1;
  }
  if (have_sparsity)
    return tool_prune_sparse(in, out, sparsity, norms, keep, nkeep, drop, ndrop,
                             1);
  if (norms) {
    fprintf(stderr, "prune: --norms requires --sparsity\n");
    return 1;
  }
  return tool_prune(in, out, keep, nkeep, drop, ndrop, 1);
}
#endif
