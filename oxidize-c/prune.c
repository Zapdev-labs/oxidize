/* oxidize-c prune: port of oxidize-prune (name-filter / magnitude / Wanda)
 * over the mmap GGUF reader. Per-output-row masks (Wanda Table 7 comparison
 * group), optional N:M structure, survivors re-quantized and written to a new
 * GGUF (KV metadata copied verbatim from the source).
 *
 * ponytail: re-quantization targets are F32/F16/Q8_0/Q4_0 only — tensors in
 * other quant types (Q4_K etc.) are re-emitted as Q8_0 with a warning. Add
 * K-quant quantizers when preserving those types matters. */
#include "oc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- per-row top-k mask selection ---- */

static int cmp_f32_desc(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return x < y ? 1 : x > y ? -1 : 0;
}

/* keep the (1-sparsity)*cols highest-scoring entries of each row */
static void rowwise_mask(const float *scores, size_t rows, size_t cols,
                         float sparsity, bool *mask) {
  size_t keep = (size_t)((double)cols * (1.0 - (double)sparsity) + 0.5);
  if (keep >= cols) { memset(mask, 1, rows * cols); return; }
  if (keep == 0) { memset(mask, 0, rows * cols); return; }
#pragma omp parallel
  {
    float *sorted = malloc(cols * sizeof(float));
#pragma omp for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
      const float *s = scores + r * cols;
      bool *mk = mask + r * cols;
      memcpy(sorted, s, cols * sizeof(float));
      qsort(sorted, cols, sizeof(float), cmp_f32_desc);
      float thr = sorted[keep - 1];
      size_t kept = 0;
      for (size_t c = 0; c < cols; ++c) {
        mk[c] = s[c] > thr;
        kept += mk[c];
      }
      /* fill ties at the threshold up to the exact keep count */
      for (size_t c = 0; c < cols && kept < keep; ++c)
        if (!mk[c] && s[c] == thr) { mk[c] = true; ++kept; }
    }
    free(sorted);
  }
}

void oc_magnitude_mask(const float *w, size_t rows, size_t cols, float sparsity,
                       bool *mask) {
  float *scores = malloc(rows * cols * sizeof(float));
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < rows * cols; ++i) scores[i] = fabsf(w[i]);
  rowwise_mask(scores, rows, cols, sparsity, mask);
  free(scores);
}

void oc_wanda_mask(const float *w, const float *col_norms, size_t rows,
                   size_t cols, float sparsity, bool *mask) {
  float *scores = malloc(rows * cols * sizeof(float));
#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c)
      scores[r * cols + c] = fabsf(w[r * cols + c]) * col_norms[c];
  rowwise_mask(scores, rows, cols, sparsity, mask);
  free(scores);
}

/* structured N:M: within every m-wide block of each row keep only the n
 * highest-scoring entries (applied ON TOP of the base mask) */
void oc_nm_mask(const float *scores, size_t rows, size_t cols, size_t n,
                size_t m, bool *mask) {
  if (cols % m != 0) oc_die("prune: N:%zu pattern needs cols %% %zu == 0", n, m);
#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < rows; ++r) {
    for (size_t blk = 0; blk < cols / m; ++blk) {
      const float *s = scores + r * cols + blk * m;
      bool *mk = mask + r * cols + blk * m;
      for (size_t kept = n; kept < m; ++kept) {
        /* drop the lowest-scoring still-kept entry (m is 4 or 8: O(m^2) ok) */
        size_t worst = m;
        for (size_t k = 0; k < m; ++k)
          if (mk[k] && (worst == m || s[k] < s[worst])) worst = k;
        if (worst == m) break;
        mk[worst] = false;
      }
    }
  }
}

/* ---- L2-norms calibration cache (same text format as oxidize-prune) ---- */

typedef struct { char *name; float *vals; size_t n; } calib_row;
typedef struct { calib_row *rows; size_t n; } calib_t;

static calib_t calib_load(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) oc_die("prune: cannot open calibration cache %s", path);
  calib_t c = {0};
  char *line = NULL;
  size_t cap = 0;
  ssize_t len;
  while ((len = getline(&line, &cap, f)) > 0) {
    char *p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '#' || *p == '\n' || *p == 0) continue;
    char *name = strtok(p, " \t\n");
    if (!name) continue;
    size_t vn = 0, vcap = 256;
    float *vals = malloc(vcap * sizeof(float));
    char *tok;
    while ((tok = strtok(NULL, " \t\n"))) {
      if (vn == vcap) vals = realloc(vals, (vcap *= 2) * sizeof(float));
      vals[vn++] = strtof(tok, NULL);
    }
    c.rows = realloc(c.rows, (c.n + 1) * sizeof(calib_row));
    c.rows[c.n].name = strdup(name);
    c.rows[c.n].vals = vals;
    c.rows[c.n].n = vn;
    c.n++;
  }
  free(line);
  fclose(f);
  return c;
}

static const calib_row *calib_get(const calib_t *c, const char *name) {
  for (size_t i = 0; i < c->n; ++i)
    if (strcmp(c->rows[i].name, name) == 0) return &c->rows[i];
  return NULL;
}

/* ---- GGUF writer ---- */

static void w32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void wstr(FILE *f, const char *s) {
  uint64_t n = strlen(s);
  w64(f, n);
  fwrite(s, 1, n, f);
}
static void wpad(FILE *f, uint64_t align) {
  long pos = ftell(f);
  size_t pad = (size_t)(((pos + (long)align - 1) / (long)align) * (long)align - pos);
  static const uint8_t zeros[64] = {0};
  while (pad) { size_t k = pad > 64 ? 64 : pad; fwrite(zeros, 1, k, f); pad -= k; }
}

typedef struct {
  const oc_tensor_info *src;   /* source info (name/dims) */
  oc_quant out_quant;
  const uint8_t *raw;          /* pass-through bytes (NULL if requantized) */
  size_t raw_bytes;
  uint8_t *owned;              /* requantized bytes (freed after write) */
  size_t owned_bytes;
} out_tensor;

static uint64_t tensor_elems(const oc_tensor_info *ti) {
  uint64_t n = 1;
  for (uint32_t d = 0; d < ti->n_dims; ++d) n *= ti->dims[d];
  return n;
}

/* exact byte size for known types; offset-delta fallback for unknown */
static size_t tensor_src_bytes(const oc_gguf *g, size_t idx) {
  const oc_tensor_info *ti = &g->tensors[idx];
  if (ti->quant != OC_UNKNOWN)
    return tensor_elems(ti) / oc_block_values(ti->quant) * oc_block_bytes(ti->quant);
  uint64_t next = g->map_size;
  for (size_t t = 0; t < g->n_tensors; ++t)
    if (g->tensors[t].offset > ti->offset && g->tensors[t].offset < next)
      next = g->tensors[t].offset;
  return (size_t)(next - ti->offset);
}

static void write_gguf(const char *path, const oc_gguf *g,
                       const out_tensor *ts, size_t n_ts) {
  FILE *f = fopen(path, "wb");
  if (!f) oc_die("prune: cannot open output %s", path);
  fwrite("GGUF", 1, 4, f);
  w32(f, g->version);
  w64(f, n_ts);
  w64(f, g->n_meta);
  fwrite(g->base + g->kv_off, 1, g->kv_end - g->kv_off, f); /* KV verbatim */
  uint64_t off = 0;
  for (size_t i = 0; i < n_ts; ++i) {
    const oc_tensor_info *ti = ts[i].src;
    wstr(f, ti->name);
    w32(f, ti->n_dims);
    for (uint32_t d = 0; d < ti->n_dims; ++d) w64(f, ti->dims[d]);
    w32(f, ts[i].raw ? ti->ggml_type : oc_to_ggml_type(ts[i].out_quant));
    w64(f, off);
    size_t bytes = ts[i].raw ? ts[i].raw_bytes : ts[i].owned_bytes;
    off = (off + bytes + g->align - 1) / g->align * g->align;
  }
  wpad(f, g->align);
  for (size_t i = 0; i < n_ts; ++i) {
    const uint8_t *data = ts[i].raw ? ts[i].raw : ts[i].owned;
    size_t bytes = ts[i].raw ? ts[i].raw_bytes : ts[i].owned_bytes;
    fwrite(data, 1, bytes, f);
    if (i + 1 < n_ts) wpad(f, g->align);
  }
  if (fclose(f) != 0) oc_die("prune: write failed for %s", path);
}

/* ---- CLI ---- */

typedef enum { M_NAME_FILTER, M_MAGNITUDE, M_WANDA } method_t;

static bool name_has_any(const char *name, char **subs, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (strstr(name, subs[i])) return true;
  return false;
}

static void prune_usage(void) {
  fprintf(stderr,
          "Usage: oxidize-c prune --input in.gguf --output out.gguf [options]\n"
          "  --method M          name-filter (default) | magnitude | wanda\n"
          "  --keep S            (name-filter) keep tensors containing S (repeatable)\n"
          "  --drop S            (name-filter) drop tensors containing S (repeatable)\n"
          "  --sparsity F        fraction to prune, default 0.5\n"
          "  --pattern P         unstructured (default) | n2of4 | n4of8\n"
          "  --calibration PATH  L2-norms cache (wanda)\n"
          "  --keep-name S       never prune tensors containing S (repeatable;\n"
          "                      default: token_embd output rope norm)\n"
          "  --joint-quantize Q  re-quantize survivors to Q (F32|F16|Q8_0|Q4_0)\n"
          "  --dry-run           print decisions without writing\n");
  exit(2);
}

int oc_prune_main(int argc, char **argv) {
  const char *input = NULL, *output = NULL, *calibration = NULL;
  method_t method = M_NAME_FILTER;
  float sparsity = 0.5f;
  size_t nm_n = 0, nm_m = 0;
  oc_quant joint = OC_UNKNOWN;
  bool dry_run = false;
  char *keeps[64], *drops[64], *keep_names[64];
  size_t n_keeps = 0, n_drops = 0, n_keep_names = 0;

  for (int i = 0; i < argc; ++i) {
#define PVAL() (i + 1 < argc ? argv[++i] : (prune_usage(), (char *)0))
    if (!strcmp(argv[i], "--input")) input = PVAL();
    else if (!strcmp(argv[i], "--output")) output = PVAL();
    else if (!strcmp(argv[i], "--calibration")) calibration = PVAL();
    else if (!strcmp(argv[i], "--sparsity")) sparsity = strtof(PVAL(), NULL);
    else if (!strcmp(argv[i], "--keep")) keeps[n_keeps++ % 64] = PVAL();
    else if (!strcmp(argv[i], "--drop")) drops[n_drops++ % 64] = PVAL();
    else if (!strcmp(argv[i], "--keep-name")) keep_names[n_keep_names++ % 64] = PVAL();
    else if (!strcmp(argv[i], "--dry-run")) dry_run = true;
    else if (!strcmp(argv[i], "--method")) {
      const char *v = PVAL();
      method = !strcmp(v, "wanda")     ? M_WANDA
             : !strcmp(v, "magnitude") ? M_MAGNITUDE
             : !strcmp(v, "name-filter") ? M_NAME_FILTER
             : (prune_usage(), M_NAME_FILTER);
    } else if (!strcmp(argv[i], "--pattern")) {
      const char *v = PVAL();
      if (!strcmp(v, "n2of4")) { nm_n = 2; nm_m = 4; }
      else if (!strcmp(v, "n4of8")) { nm_n = 4; nm_m = 8; }
      else if (strcmp(v, "unstructured")) prune_usage();
    } else if (!strcmp(argv[i], "--joint-quantize")) {
      joint = oc_quant_parse(PVAL());
      if (!oc_quantize_row(joint, (float[QK]){0}, (uint8_t[136]){0}, QK))
        oc_die("prune: unsupported --joint-quantize target (use F32|F16|Q8_0|Q4_0)");
    } else prune_usage();
#undef PVAL
  }
  if (!input || !output) prune_usage();
  if (sparsity < 0.0f || sparsity >= 1.0f) oc_die("prune: sparsity must be in [0,1)");

  if (n_keep_names == 0) {
    keep_names[n_keep_names++] = "token_embd";
    keep_names[n_keep_names++] = "output";
    keep_names[n_keep_names++] = "rope";
    keep_names[n_keep_names++] = "norm";
  }

  oc_gguf *g = oc_gguf_load(input);
  calib_t calib = {0};
  if (method == M_WANDA) {
    if (!calibration) oc_die("prune: --method wanda requires --calibration");
    calib = calib_load(calibration);
  }

  out_tensor *ts = calloc(g->n_tensors, sizeof(out_tensor));
  size_t n_ts = 0, pruned = 0;

  for (size_t t = 0; t < g->n_tensors; ++t) {
    const oc_tensor_info *ti = &g->tensors[t];
    if (method == M_NAME_FILTER) {
      bool drop = name_has_any(ti->name, drops, n_drops) ||
                  (n_keeps > 0 && !name_has_any(ti->name, keeps, n_keeps));
      if (drop) {
        printf("drop %s\n", ti->name);
        pruned++;
        continue;
      }
      printf("keep %s\n", ti->name);
      ts[n_ts++] = (out_tensor){.src = ti, .raw = g->base + ti->offset,
                                .raw_bytes = tensor_src_bytes(g, t)};
      continue;
    }

    /* magnitude / wanda: only 2D weight matrices, honoring keep-name */
    bool prunable = ti->n_dims == 2 && ti->quant != OC_UNKNOWN &&
                    !name_has_any(ti->name, keep_names, n_keep_names);
    if (!prunable) {
      ts[n_ts++] = (out_tensor){.src = ti, .raw = g->base + ti->offset,
                                .raw_bytes = tensor_src_bytes(g, t)};
      continue;
    }
    size_t cols = (size_t)ti->dims[0], rows = (size_t)ti->dims[1];
    const calib_row *cr = NULL;
    if (method == M_WANDA) {
      cr = calib_get(&calib, ti->name);
      if (!cr) oc_die("prune: calibration cache has no entry for %s", ti->name);
      if (cr->n != cols)
        oc_die("prune: calibration for %s has %zu norms, tensor has %zu cols",
               ti->name, cr->n, cols);
    }
    if (dry_run) {
      printf("prune %s [%zux%zu] %s\n", ti->name, rows, cols,
             oc_quant_name(ti->quant));
      pruned++;
      continue;
    }

    float *w = malloc(rows * cols * sizeof(float));
    size_t srow = oc_row_bytes(ti->quant, cols);
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r)
      oc_dequant_row(ti->quant, g->base + ti->offset + r * srow, w + r * cols, cols);

    bool *mask = malloc(rows * cols);
    if (method == M_WANDA) oc_wanda_mask(w, cr->vals, rows, cols, sparsity, mask);
    else oc_magnitude_mask(w, rows, cols, sparsity, mask);
    if (nm_m) {
      float *scores = malloc(rows * cols * sizeof(float));
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < rows * cols; ++i)
        scores[i] = fabsf(w[i]) * (method == M_WANDA ? (cr->vals[i % cols]) : 1.0f);
      oc_nm_mask(scores, rows, cols, nm_n, nm_m, mask);
      free(scores);
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows * cols; ++i)
      if (!mask[i]) w[i] = 0.0f;
    free(mask);

    oc_quant oq = joint != OC_UNKNOWN ? joint : ti->quant;
    { /* probe requant support; fall back to Q8_0 for K-quants etc. */
      float probe[QK] = {0};
      uint8_t pbuf[136];
      if (!oc_quantize_row(oq, probe, pbuf, QK)) {
        fprintf(stderr, "warning: no %s quantizer; re-emitting %s as Q8_0\n",
                oc_quant_name(oq), ti->name);
        oq = OC_Q8_0;
      }
    }
    size_t drow = oc_row_bytes(oq, cols);
    uint8_t *out = malloc(rows * drow);
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r)
      oc_quantize_row(oq, w + r * cols, out + r * drow, cols);
    free(w);
    printf("prune %s [%zux%zu] %s -> %s\n", ti->name, rows, cols,
           oc_quant_name(ti->quant), oc_quant_name(oq));
    pruned++;
    ts[n_ts++] = (out_tensor){.src = ti, .out_quant = oq, .owned = out,
                              .owned_bytes = rows * drow};
  }

  if (!dry_run) {
    write_gguf(output, g, ts, n_ts);
    printf("%s %zu of %zu tensors -> %s\n",
           method == M_NAME_FILTER ? "Pruned" :
           method == M_WANDA ? "Wanda-pruned" : "Magnitude-pruned",
           pruned, g->n_tensors, output);
  } else {
    printf("dry run: %zu of %zu tensors would be pruned\n", pruned, g->n_tensors);
  }

  for (size_t i = 0; i < n_ts; ++i) free(ts[i].owned);
  free(ts);
  for (size_t i = 0; i < calib.n; ++i) { free(calib.rows[i].name); free(calib.rows[i].vals); }
  free(calib.rows);
  oc_gguf_free(g);
  return 0;
}
