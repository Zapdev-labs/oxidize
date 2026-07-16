/* tools/ acceptance tests: quantize / prune / merge. Every produced file is
 * re-opened through gguf_open, so the writer is proven to emit valid GGUF. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/gguf.h"
#include "../src/model.h"
#include "../src/model_llama.h"
#include "../src/quant.h"
#include "../src/tensor.h"
#include "../src/tokenizer.h"
#include "../tools/convert.h"
#include "../tools/gguf_write.h"
#include "tests.h"

/* ---- tiny F32 GGUF builder ------------------------------------------------ */

typedef struct {
  const char* name;
  uint32_t nd;
  uint64_t d[2];
  const float* data;
} Tsp;

static void app(uint8_t** b, size_t* l, const void* p, size_t n) {
  *b = realloc(*b, *l + n);
  CHECK(*b != NULL);
  memcpy(*b + *l, p, n);
  *l += n;
}
static void app_u32(uint8_t** b, size_t* l, uint32_t v) { app(b, l, &v, 4); }
static void app_u64(uint8_t** b, size_t* l, uint64_t v) { app(b, l, &v, 8); }
static void app_str(uint8_t** b, size_t* l, const char* s) {
  app_u64(b, l, strlen(s));
  app(b, l, s, strlen(s));
}
static void pad32(uint8_t** b, size_t* l) {
  uint8_t z = 0;
  while (*l % 32) app(b, l, &z, 1);
}
static size_t nelem(const Tsp* t) {
  size_t n = 1;
  for (uint32_t i = 0; i < t->nd; ++i) n *= (size_t)t->d[i];
  return n;
}

/* Writes an F32-only GGUF to a temp file. Returns a malloc'd path. */
static char* build_gguf(const Tsp* ts, size_t nt) {
  uint8_t* b = NULL;
  size_t l = 0;
  app(&b, &l, "GGUF", 4);
  app_u32(&b, &l, 3);
  app_u64(&b, &l, nt);
  app_u64(&b, &l, 1);
  app_str(&b, &l, "general.architecture");
  app_u32(&b, &l, GGUF_T_STRING);
  app_str(&b, &l, "test");
  uint64_t off = 0;
  for (size_t i = 0; i < nt; ++i) {
    app_str(&b, &l, ts[i].name);
    app_u32(&b, &l, ts[i].nd);
    for (uint32_t d = 0; d < ts[i].nd; ++d) app_u64(&b, &l, ts[i].d[d]);
    app_u32(&b, &l, OC_F32);
    app_u64(&b, &l, off);
    off = (off + nelem(&ts[i]) * 4 + 31) / 32 * 32;
  }
  pad32(&b, &l);
  for (size_t i = 0; i < nt; ++i) {
    app(&b, &l, ts[i].data, nelem(&ts[i]) * 4);
    pad32(&b, &l);
  }

  char* path = strdup("/tmp/oc-tool-XXXXXX");
  CHECK(path != NULL);
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  CHECK(write(fd, b, l) == (ssize_t)l);
  CHECK(close(fd) == 0);
  free(b);
  return path;
}

static void fill(float* x, size_t n, unsigned seed, float amp) {
  for (size_t i = 0; i < n; ++i) {
    seed = seed * 1103515245u + 12345u;
    x[i] = ((float)((seed >> 16) & 0x7fff) / 16384.0f - 1.0f) * amp;
  }
}

/* Dequantized copy of a named tensor from a GGUF on disk. */
static float* load_dequant(const GgufFile* g, const char* name, size_t* n_out,
                           uint32_t* type_out) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  CHECK(t != NULL);
  size_t cols = (size_t)t->dims[0], rows = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) rows *= (size_t)t->dims[d];
  size_t rb = oc_row_bytes(t->ggml_type, cols);
  CHECK(rb != 0);
  float* out = malloc(rows * cols * sizeof(float));
  CHECK(out != NULL);
  for (size_t r = 0; r < rows; ++r)
    CHECK(oc_dequant_row(t->ggml_type, t->data + r * rb, out + r * cols, cols) == 0);
  *n_out = rows * cols;
  if (type_out) *type_out = t->ggml_type;
  return out;
}

/* ---- fixtures -------------------------------------------------------------- */

#define QCOLS 256 /* K-quants need cols % 256 == 0; also fine for Q4_0/AL5_XS */
#define QROWS 4
#define NCOLS 8

static float g_w[QCOLS * QROWS], g_norm[NCOLS];

static char* fixture_a(void) {
  fill(g_w, QCOLS * QROWS, 11, 1.0f);
  fill(g_norm, NCOLS, 22, 1.0f);
  Tsp ts[] = {
      {"blk.0.attn_q.weight", 2, {QCOLS, QROWS}, g_w},
      {"blk.0.attn_norm.weight", 1, {NCOLS, 0}, g_norm},
  };
  return build_gguf(ts, 2);
}

/* ---- quantize --------------------------------------------------------------
 * F32 source -> every implemented target; re-open the OUTPUT and check the
 * round-trip error against the type's bound. */
static void test_tool_quantize(const char* in) {
  static const struct {
    const char* name;
    uint32_t id;
    float bound; /* max abs error over weights in [-1, 1] */
  } targets[] = {
      {"F32", OC_F32, 1e-6f},
      {"F16", OC_F16, 1e-3f},
      {"Q8_0", OC_Q8_0, 0.01f},
      {"Q4_0", OC_Q4_0, 0.15f},
      {"Q4_K", OC_Q4_K, 0.25f},
      {"Q5_K", OC_Q5_K, 0.15f},
      {"Q6_K", OC_Q6_K, 0.10f},
      {"AL5_XS", OC_AL5_XS, 0.30f},
  };
  for (size_t k = 0; k < sizeof targets / sizeof *targets; ++k) {
    char out[64];
    snprintf(out, sizeof out, "%s.%s", in, targets[k].name);
    CHECK(tool_quantize(in, out, targets[k].name, 0) == 0);

    GgufFile g;
    char err[256];
    CHECK(gguf_open(&g, out, err, sizeof err) == 0);
    CHECK(g.n_tensors == 2);

    size_t n = 0;
    uint32_t ty = 0;
    float* dec = load_dequant(&g, "blk.0.attn_q.weight", &n, &ty);
    CHECK(ty == targets[k].id);
    CHECK(n == QCOLS * QROWS);
    float maxe = 0.0f, mse = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      float e = fabsf(dec[i] - g_w[i]);
      if (e > maxe) maxe = e;
      mse += e * e;
    }
    printf("   quantize %-6s max_err %.5f  rmse %.5f (bound %.3f)\n",
           targets[k].name, (double)maxe, sqrt((double)mse / (double)n),
           (double)targets[k].bound);
    CHECK(maxe <= targets[k].bound);
    free(dec);

    /* the 1-D norm tensor stays F32 and bit-exact */
    float* nm = load_dequant(&g, "blk.0.attn_norm.weight", &n, &ty);
    CHECK(ty == OC_F32 && n == NCOLS);
    for (size_t i = 0; i < n; ++i) CHECK(nm[i] == g_norm[i]);
    free(nm);
    gguf_close(&g);
    unlink(out);
  }
  /* unknown / dequant-only target refused, not silently mis-encoded */
  char out[64];
  snprintf(out, sizeof out, "%s.bad", in);
  CHECK(tool_quantize(in, out, "Q2_K", 0) != 0);
  unlink(out);
  printf("ok tools quantize (F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/AL5_XS round-trip)\n");
}

/* ---- prune ----------------------------------------------------------------- */
static void test_tool_prune(const char* in) {
  char out[64];
  snprintf(out, sizeof out, "%s.prune", in);
  const char* keep[] = {"attn_q"};
  CHECK(tool_prune(in, out, keep, 1, NULL, 0, 0) == 0);
  GgufFile g;
  char err[256];
  CHECK(gguf_open(&g, out, err, sizeof err) == 0);
  CHECK(g.n_tensors == 1);
  CHECK(gguf_tensor(&g, "blk.0.attn_q.weight") != NULL);
  CHECK(gguf_tensor(&g, "blk.0.attn_norm.weight") == NULL);
  /* payload survived the copy */
  size_t n = 0;
  float* w = load_dequant(&g, "blk.0.attn_q.weight", &n, NULL);
  CHECK(n == QCOLS * QROWS);
  for (size_t i = 0; i < n; ++i) CHECK(w[i] == g_w[i]);
  free(w);
  gguf_close(&g);
  unlink(out);

  const char* drop[] = {"norm"};
  CHECK(tool_prune(in, out, NULL, 0, drop, 1, 0) == 0);
  CHECK(gguf_open(&g, out, err, sizeof err) == 0);
  CHECK(g.n_tensors == 1);
  CHECK(gguf_tensor(&g, "blk.0.attn_q.weight") != NULL);
  gguf_close(&g);
  unlink(out);
  printf("ok tools prune (--keep / --drop)\n");
}

/* ---- sparse prune (magnitude / Wanda) -------------------------------------- */
#define SCOLS 8
#define SROWS 2

static void test_tool_prune_sparse(void) {
  /* Row 0: 0..7 (keep top half → cols 4..7). Row 1: 7..0 (keep cols 0..3). */
  static float sw[SCOLS * SROWS];
  static float snorm[NCOLS];
  for (int c = 0; c < SCOLS; ++c) {
    sw[c] = (float)c;
    sw[SCOLS + c] = (float)(SCOLS - 1 - c);
  }
  fill(snorm, NCOLS, 33, 1.0f);
  Tsp ts[] = {
      {"blk.0.attn_q.weight", 2, {SCOLS, SROWS}, sw},
      {"blk.0.attn_norm.weight", 1, {NCOLS, 0}, snorm},
  };
  char* in = build_gguf(ts, 2);
  char out[64];
  snprintf(out, sizeof out, "%s.sparse", in);

  CHECK(tool_prune_sparse(in, out, 0.5f, NULL, NULL, 0, NULL, 0, 0) == 0);
  GgufFile g;
  char err[256];
  CHECK(gguf_open(&g, out, err, sizeof err) == 0);
  CHECK(g.n_tensors == 2);
  size_t n = 0;
  uint32_t ty = 0;
  float* w = load_dequant(&g, "blk.0.attn_q.weight", &n, &ty);
  CHECK(ty == OC_F32 && n == SCOLS * SROWS);
  size_t zeros0 = 0, zeros1 = 0;
  for (size_t c = 0; c < SCOLS; ++c) {
    if (w[c] == 0.0f) zeros0++;
    else CHECK(w[c] == sw[c]);
    if (w[SCOLS + c] == 0.0f) zeros1++;
    else CHECK(w[SCOLS + c] == sw[SCOLS + c]);
  }
  CHECK(zeros0 == SCOLS / 2);
  CHECK(zeros1 == SCOLS / 2);
  for (size_t c = 0; c < 4; ++c) CHECK(w[c] == 0.0f);
  for (size_t c = 4; c < 8; ++c) CHECK(w[c] == sw[c]);
  /* 1-D norm copied verbatim */
  float* nm = load_dequant(&g, "blk.0.attn_norm.weight", &n, &ty);
  CHECK(ty == OC_F32 && n == NCOLS);
  for (size_t i = 0; i < n; ++i) CHECK(nm[i] == snorm[i]);
  free(nm);
  free(w);
  gguf_close(&g);
  unlink(out);

  /* Wanda: amplify right half so low-norm (left) cols are preferred for zeroing
   * even when |W| is larger on the left (row 1). */
  for (int c = 0; c < SCOLS; ++c) {
    sw[c] = 1.0f; /* uniform magnitudes */
    sw[SCOLS + c] = (c < 4) ? 10.0f : 1.0f; /* left half larger in |W| */
  }
  free(in);
  in = build_gguf(ts, 2);

  char* npath = strdup("/tmp/oc-norms-XXXXXX");
  CHECK(npath != NULL);
  int nfd = mkstemp(npath);
  CHECK(nfd >= 0);
  const char* ntxt =
      "# blk.0.attn_q.weight 8\n"
      "0.0 0.0 0.0 0.0 10.0 10.0 10.0 10.0\n";
  CHECK(write(nfd, ntxt, strlen(ntxt)) == (ssize_t)strlen(ntxt));
  CHECK(close(nfd) == 0);

  snprintf(out, sizeof out, "%s.wanda", in);
  CHECK(tool_prune_sparse(in, out, 0.5f, npath, NULL, 0, NULL, 0, 0) == 0);
  CHECK(gguf_open(&g, out, err, sizeof err) == 0);
  w = load_dequant(&g, "blk.0.attn_q.weight", &n, NULL);
  CHECK(n == SCOLS * SROWS);
  /* Both rows: Wanda zeros low-norm cols 0..3, keeps 4..7. */
  for (size_t r = 0; r < SROWS; ++r) {
    for (size_t c = 0; c < 4; ++c) CHECK(w[r * SCOLS + c] == 0.0f);
    for (size_t c = 4; c < 8; ++c) CHECK(w[r * SCOLS + c] != 0.0f);
  }
  free(w);
  gguf_close(&g);
  unlink(out);
  unlink(npath);
  free(npath);
  unlink(in);
  free(in);
  printf("ok tools prune sparse (magnitude 0.5 / Wanda norms)\n");
}

/* ---- merge ----------------------------------------------------------------- */
static void test_tool_merge(const char* a) {
  static float w2[QCOLS * QROWS], n2[NCOLS];
  fill(w2, QCOLS * QROWS, 99, 1.0f);
  fill(n2, NCOLS, 77, 1.0f);
  Tsp bt[] = {
      {"blk.0.attn_q.weight", 2, {QCOLS, QROWS}, w2},
      {"blk.0.attn_norm.weight", 1, {NCOLS, 0}, n2},
  };
  char* b = build_gguf(bt, 2);

  char out[64];
  snprintf(out, sizeof out, "%s.merge", a);
  const float alpha = 0.25f;
  CHECK(tool_merge(a, b, out, alpha, 0) == 0);

  GgufFile g;
  char err[256];
  CHECK(gguf_open(&g, out, err, sizeof err) == 0);
  CHECK(g.n_tensors == 2);
  size_t n = 0;
  uint32_t ty = 0;
  float* m = load_dequant(&g, "blk.0.attn_q.weight", &n, &ty);
  CHECK(ty == OC_F32 && n == QCOLS * QROWS);
  for (size_t i = 0; i < n; ++i)
    CHECK(fabsf(m[i] - (alpha * g_w[i] + (1.0f - alpha) * w2[i])) < 1e-6f);
  free(m);
  m = load_dequant(&g, "blk.0.attn_norm.weight", &n, &ty);
  CHECK(ty == OC_F32 && n == NCOLS);
  for (size_t i = 0; i < n; ++i)
    CHECK(fabsf(m[i] - (alpha * g_norm[i] + (1.0f - alpha) * n2[i])) < 1e-6f);
  free(m);
  gguf_close(&g);
  unlink(out);

  /* shape mismatch must fail loudly (nonzero), not blend garbage */
  static float w3[QCOLS * 2];
  fill(w3, QCOLS * 2, 5, 1.0f);
  Tsp ct[] = {
      {"blk.0.attn_q.weight", 2, {QCOLS, 2}, w3},
      {"blk.0.attn_norm.weight", 1, {NCOLS, 0}, n2},
  };
  char* c = build_gguf(ct, 2);
  fprintf(stderr, "-- expect one 'shape mismatch' line below --\n");
  CHECK(tool_merge(a, c, out, alpha, 0) != 0);
  unlink(out);

  /* tensor count mismatch too */
  Tsp dt[] = {{"blk.0.attn_q.weight", 2, {QCOLS, QROWS}, w2}};
  char* d = build_gguf(dt, 1);
  fprintf(stderr, "-- expect one 'count mismatch' line below --\n");
  CHECK(tool_merge(a, d, out, alpha, 0) != 0);
  unlink(out);

  unlink(b);
  unlink(c);
  unlink(d);
  free(b);
  free(c);
  free(d);
  printf("ok tools merge (soup + shape/count mismatch rejected)\n");
}

/* ==== convert (SafeTensors / HuggingFace -> GGUF) ==========================
 * The high-risk part is the llama/mistral q/k PERMUTE: model_llama.c runs ggml
 * NORMAL (adjacent-pair) RoPE for arch=llama, so the converter must permute q/k
 * exactly as llama.cpp does, or the model produces fluent GARBAGE (a bug no
 * crash and no shape error reveals). The proof here is independent of the
 * converter's own permute formula: convert ONE HF model two ways --
 *   (a) arch=llama  -> permuted q/k + NORMAL rope
 *   (b) arch=qwen2  -> unpermuted q/k + NeoX rope
 * Both must reproduce identical logits, because NORMAL(permute(w)) == NeoX(w) is
 * ggml ground truth (also pinned by test_model.c's check_rope_modes). Matching
 * logits <=> the permute is correct; a wrong permute makes (a) diverge from (b).
 */

/* tiny dense llama-shaped model: hidden 8, 2 heads, 1 kv head, head_dim 4 (even,
 * so q/k are permutable), ff 16, vocab 16, 2 layers. */
enum { CH = 8, CNH = 2, CNKV = 1, CHD = 4, CFF = 16, CV = 16, CLY = 2 };

typedef struct {
  char name[80];
  int ndim;
  int64_t d[3]; /* HF order [out, in] */
  float* data;
} StT;

static size_t st_nelem(const StT* t) {
  size_t n = 1;
  for (int i = 0; i < t->ndim; ++i) n *= (size_t)t->d[i];
  return n;
}

/* Serialize F32 tensors to a .safetensors file: u64 header length, a JSON header
 * {name:{dtype,shape,data_offsets}}, then the concatenated tensor bytes. */
static void write_safetensors(const char* path, const StT* ts, size_t nt) {
  size_t* off = malloc((nt + 1) * sizeof(size_t));
  CHECK(off != NULL);
  off[0] = 0;
  for (size_t i = 0; i < nt; ++i) off[i + 1] = off[i] + st_nelem(&ts[i]) * 4;

  size_t cap = 256 + nt * 160;
  char* h = malloc(cap);
  CHECK(h != NULL);
  size_t hl = 0;
  h[hl++] = '{';
  for (size_t i = 0; i < nt; ++i) {
    if (i) h[hl++] = ',';
    hl += (size_t)snprintf(h + hl, cap - hl,
                           "\"%s\":{\"dtype\":\"F32\",\"shape\":[", ts[i].name);
    for (int d = 0; d < ts[i].ndim; ++d)
      hl += (size_t)snprintf(h + hl, cap - hl, "%s%ld", d ? "," : "", (long)ts[i].d[d]);
    hl += (size_t)snprintf(h + hl, cap - hl, "],\"data_offsets\":[%zu,%zu]}", off[i], off[i + 1]);
  }
  h[hl++] = '}';

  FILE* f = fopen(path, "wb");
  CHECK(f != NULL);
  uint64_t hlen = hl;
  CHECK(fwrite(&hlen, 8, 1, f) == 1);
  CHECK(fwrite(h, 1, hl, f) == hl);
  for (size_t i = 0; i < nt; ++i) {
    size_t nb = st_nelem(&ts[i]) * 4;
    CHECK(fwrite(ts[i].data, 1, nb, f) == nb);
  }
  CHECK(fclose(f) == 0);
  free(h);
  free(off);
}

/* Build the HF tensor set (caller frees each .data). */
static size_t build_hf_model(StT* ts) {
  size_t n = 0;
  unsigned seed = 3;
#define ADD(NM, ND, D0, D1)                                        \
  do {                                                             \
    StT* t = &ts[n++];                                             \
    snprintf(t->name, sizeof t->name, "%s", (NM));                 \
    t->ndim = (ND);                                                \
    t->d[0] = (D0);                                                \
    t->d[1] = (D1);                                                \
    t->d[2] = 0;                                                   \
    size_t ne = st_nelem(t);                                       \
    t->data = malloc(ne * sizeof(float));                          \
    CHECK(t->data != NULL);                                        \
    fill(t->data, ne, seed++, 0.5f);                               \
  } while (0)

  ADD("model.embed_tokens.weight", 2, CV, CH);
  ADD("model.norm.weight", 1, CH, 0);
  ADD("lm_head.weight", 2, CV, CH);
  for (int l = 0; l < CLY; ++l) {
    char nm[80];
#define LADD(SUF, ND, D0, D1)                                      \
  do {                                                             \
    snprintf(nm, sizeof nm, "model.layers.%d.%s", l, (SUF));       \
    ADD(nm, ND, D0, D1);                                           \
  } while (0)
    LADD("input_layernorm.weight", 1, CH, 0);
    LADD("post_attention_layernorm.weight", 1, CH, 0);
    LADD("self_attn.q_proj.weight", 2, CNH * CHD, CH);
    LADD("self_attn.k_proj.weight", 2, CNKV * CHD, CH);
    LADD("self_attn.v_proj.weight", 2, CNKV * CHD, CH);
    LADD("self_attn.o_proj.weight", 2, CH, CNH * CHD);
    LADD("mlp.gate_proj.weight", 2, CFF, CH);
    LADD("mlp.up_proj.weight", 2, CFF, CH);
    LADD("mlp.down_proj.weight", 2, CH, CFF);
#undef LADD
  }
#undef ADD
  return n;
}

static void write_text(const char* path, const char* s) {
  FILE* f = fopen(path, "wb");
  CHECK(f != NULL);
  CHECK(fwrite(s, 1, strlen(s), f) == strlen(s));
  CHECK(fclose(f) == 0);
}

/* Assert blk.0.<name>.weight in `perm_g` (llama, permuted) equals the per-head
 * row permute of the same tensor in `orig_g` (qwen2, unpermuted). heads is the
 * head count for this projection (q: n_head, k: n_kv). Returns via CHECK. */
static void check_permuted(const GgufFile* perm_g, const GgufFile* orig_g,
                           const char* name, size_t heads) {
  const GgufTensorInfo* a = gguf_tensor(perm_g, name);
  const GgufTensorInfo* b = gguf_tensor(orig_g, name);
  CHECK(a && b && a->ggml_type == OC_F32 && b->ggml_type == OC_F32);
  size_t cols = (size_t)a->dims[0], rows = (size_t)a->dims[1];
  CHECK(cols == CH && rows == heads * CHD);
  const float* fa = (const float*)a->data; /* permuted */
  const float* fb = (const float*)b->data; /* original */
  size_t hd = CHD, half = hd / 2;
  int changed = 0;
  for (size_t hh = 0; hh < heads; ++hh)
    for (size_t dl = 0; dl < hd; ++dl) {
      size_t sl = (dl & 1) ? half + dl / 2 : dl / 2; /* NORMAL<-NeoX inverse */
      size_t drow = hh * hd + dl, srow = hh * hd + sl;
      for (size_t c = 0; c < cols; ++c)
        CHECK(fabsf(fa[drow * cols + c] - fb[srow * cols + c]) < 1e-6f);
      if (drow != srow) changed = 1;
    }
  CHECK(changed); /* the permute is non-identity, i.e. it actually happened */
}

static void test_convert(void) {
  char dir[] = "/tmp/oc-convert-XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char cfgp[256], tokp[256], stp[256], g_ll[256], g_qw[256], g_file[256];
  snprintf(cfgp, sizeof cfgp, "%s/config.json", dir);
  snprintf(tokp, sizeof tokp, "%s/tokenizer.json", dir);
  snprintf(stp, sizeof stp, "%s/model.safetensors", dir);
  snprintf(g_ll, sizeof g_ll, "%s/out-llama.gguf", dir);
  snprintf(g_qw, sizeof g_qw, "%s/out-qwen2.gguf", dir);
  snprintf(g_file, sizeof g_file, "%s/out-file.gguf", dir);

  write_text(cfgp,
             "{\"model_type\":\"llama\",\"hidden_size\":8,\"num_hidden_layers\":2,"
             "\"num_attention_heads\":2,\"num_key_value_heads\":1,"
             "\"intermediate_size\":16,\"rms_norm_eps\":1e-05,\"rope_theta\":10000.0,"
             "\"vocab_size\":16,\"max_position_embeddings\":32,"
             "\"bos_token_id\":1,\"eos_token_id\":2}");
  write_text(tokp,
             "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"<unk>\":0,\"<s>\":1,\"</s>\":2,"
             "\"a\":3,\"b\":4,\"ab\":5,\"c\":6},\"merges\":[\"a b\"]},"
             "\"added_tokens\":[{\"id\":1,\"content\":\"<s>\",\"special\":true},"
             "{\"id\":2,\"content\":\"</s>\",\"special\":true}]}");

  StT ts[64];
  size_t nt = build_hf_model(ts);
  write_safetensors(stp, ts, nt);

  /* Convert the SAME weights two ways (F32 output for an exact comparison). */
  ConvertOpts o_ll = {.arch_override = NULL, .outtype = "F32"}; /* config: llama */
  ConvertOpts o_qw = {.arch_override = "qwen2", .outtype = "F32"};
  CHECK(tool_convert(dir, g_ll, &o_ll, 0) == 0);
  CHECK(tool_convert(dir, g_qw, &o_qw, 0) == 0);
  /* single-FILE input finds the sibling config.json/tokenizer.json */
  ConvertOpts o_file = {.arch_override = NULL, .outtype = "F32"};
  CHECK(tool_convert(stp, g_file, &o_file, 0) == 0);

  GgufFile ga, gb;
  char err[256];
  CHECK(gguf_open(&ga, g_ll, err, sizeof err) == 0);
  CHECK(gguf_open(&gb, g_qw, err, sizeof err) == 0);

  /* arch string routed correctly and, critically, the RoPE mode differs. */
  char* arch_a = gguf_architecture(&ga);
  char* arch_b = gguf_architecture(&gb);
  CHECK(arch_a && strcmp(arch_a, "llama") == 0);
  CHECK(arch_b && strcmp(arch_b, "qwen2") == 0);
  free(arch_a);
  free(arch_b);

  /* Byte-level: llama q/k ARE the per-head permute of qwen2's (independent of
   * the logit test below). Both q (2 heads) and k (1 kv head) are permuted.
   * Done BEFORE model_load, which zeroes the caller's GgufFile on ownership. */
  check_permuted(&ga, &gb, "blk.0.attn_q.weight", CNH);
  check_permuted(&ga, &gb, "blk.0.attn_k.weight", CNKV);
  check_permuted(&ga, &gb, "blk.1.attn_q.weight", CNH);

  /* Embedded tokenizer loads and resolves known pieces + special ids. */
  Tokenizer tk;
  CHECK(tokenizer_init(&tk, &ga) == 0);
  CHECK(tokenizer_piece_id(&tk, "ab", 2) == 5);
  CHECK(tokenizer_piece_id(&tk, "a", 1) == 3);
  CHECK(tk.bos_id == 1 && tk.eos_id == 2);
  CHECK(tk.token_types[1] == 3 && tk.token_types[2] == 3); /* specials -> CONTROL */
  CHECK(tk.token_types[3] == 1);                            /* normal */
  tokenizer_free(&tk);

  Model ma, mb;
  CHECK(model_load(&ma, &ga, 0, false, err, sizeof err) == 0); /* consumes ga */
  CHECK(model_load(&mb, &gb, 0, false, err, sizeof err) == 0); /* consumes gb */
  CHECK(ma.family == MODEL_LLAMA && mb.family == MODEL_LLAMA);
  CHECK(((LlamaModel*)ma.handle)->rope_norm == true);  /* llama -> NORMAL */
  CHECK(((LlamaModel*)mb.handle)->rope_norm == false); /* qwen2 -> NeoX  */

  /* THE PERMUTE PROOF: identical logits over a short decode. */
  const int32_t toks[] = {3, 1, 4, 1, 5};
  const size_t ntok = sizeof toks / sizeof *toks;
  float* la = NULL;
  float* lb = NULL;
  for (size_t i = 0; i < ntok; ++i) la = ma.forward(ma.handle, toks[i], i, true);
  for (size_t i = 0; i < ntok; ++i) lb = mb.forward(mb.handle, toks[i], i, true);
  CHECK(la != NULL && lb != NULL);
  float maxd = 0.0f;
  int nonzero = 0;
  for (size_t i = 0; i < CV; ++i) {
    float d = fabsf(la[i] - lb[i]);
    if (d > maxd) maxd = d;
    if (fabsf(la[i]) > 1e-6f) nonzero = 1;
  }
  printf("   convert: llama(permute+NORMAL) vs qwen2(NeoX) logit maxdiff %.3e\n", (double)maxd);
  CHECK(nonzero);       /* logits are real, not a degenerate all-zero match */
  CHECK(maxd < 1e-3f);  /* permute exactly compensates the rope-mode change */

  model_free(&ma); /* frees ga's mmap + arrays */
  model_free(&mb); /* frees gb's mmap + arrays */

  /* ---- loud rejections (never a silently-wrong GGUF) ---- */
  ConvertOpts o_gemma_v1 = {.arch_override = "gemma", .outtype = "F32"};
  fprintf(stderr, "-- expect one 'architecture ... not supported' block below --\n");
  CHECK(tool_convert(dir, g_file, &o_gemma_v1, 0) != 0); /* gemma-v1 unsupported */
  ConvertOpts o_moe_arch = {.arch_override = "mixtral", .outtype = "F32"};
  CHECK(tool_convert(dir, g_file, &o_moe_arch, 0) != 0); /* MoE arch */
  ConvertOpts o_bad_type = {.arch_override = "llama", .outtype = "Q2_K"};
  CHECK(tool_convert(dir, g_file, &o_bad_type, 0) != 0); /* no encoder yet */

  /* MoE/expert tensors rejected even under a supported arch (map_name < 0). */
  char dir2[] = "/tmp/oc-convert-moe-XXXXXX";
  CHECK(mkdtemp(dir2) != NULL);
  char cfgp2[256], stp2[256], g2[256];
  snprintf(cfgp2, sizeof cfgp2, "%s/config.json", dir2);
  snprintf(stp2, sizeof stp2, "%s/model.safetensors", dir2);
  snprintf(g2, sizeof g2, "%s/out.gguf", dir2);
  write_text(cfgp2, "{\"model_type\":\"llama\",\"hidden_size\":8,\"num_attention_heads\":2}");
  StT moe[1];
  snprintf(moe[0].name, sizeof moe[0].name, "model.layers.0.mlp.experts.0.gate_proj.weight");
  moe[0].ndim = 2;
  moe[0].d[0] = CFF;
  moe[0].d[1] = CH;
  moe[0].d[2] = 0;
  size_t mne = st_nelem(&moe[0]);
  moe[0].data = malloc(mne * sizeof(float));
  CHECK(moe[0].data != NULL);
  fill(moe[0].data, mne, 9, 0.5f);
  write_safetensors(stp2, moe, 1);
  ConvertOpts o_ll2 = {.arch_override = "llama", .outtype = "F32"};
  fprintf(stderr, "-- expect one 'unsupported structure' line below --\n");
  CHECK(tool_convert(dir2, g2, &o_ll2, 0) != 0);
  free(moe[0].data);
  unlink(cfgp2);
  unlink(stp2);
  unlink(g2);
  rmdir(dir2);

  for (size_t i = 0; i < nt; ++i) free(ts[i].data);
  unlink(cfgp);
  unlink(tokp);
  unlink(stp);
  unlink(g_ll);
  unlink(g_qw);
  unlink(g_file);
  rmdir(dir);
  printf("ok tools convert (q/k permute proven; tokenizer; loud rejections)\n");
}

/* Gemma2+ sandwich: HF pre/post_feedforward norms map correctly, +1 is baked
 * into RMSNorm weights, and the resulting GGUF loads as MODEL_GEMMA4. */
static size_t build_hf_gemma(StT* ts) {
  size_t n = 0;
  unsigned seed = 11;
#define ADD(NM, ND, D0, D1)                                        \
  do {                                                             \
    StT* t = &ts[n++];                                             \
    snprintf(t->name, sizeof t->name, "%s", (NM));                 \
    t->ndim = (ND);                                                \
    t->d[0] = (D0);                                                \
    t->d[1] = (D1);                                                \
    t->d[2] = 0;                                                   \
    size_t ne = st_nelem(t);                                       \
    t->data = malloc(ne * sizeof(float));                          \
    CHECK(t->data != NULL);                                        \
    fill(t->data, ne, seed++, 0.5f);                               \
  } while (0)

  ADD("model.embed_tokens.weight", 2, CV, CH);
  ADD("model.norm.weight", 1, CH, 0);
  for (int l = 0; l < CLY; ++l) {
    char nm[96];
#define LADD(SUF, ND, D0, D1)                                      \
  do {                                                             \
    snprintf(nm, sizeof nm, "model.layers.%d.%s", l, (SUF));       \
    ADD(nm, ND, D0, D1);                                           \
  } while (0)
    LADD("input_layernorm.weight", 1, CH, 0);
    LADD("post_attention_layernorm.weight", 1, CH, 0);
    LADD("pre_feedforward_layernorm.weight", 1, CH, 0);
    LADD("post_feedforward_layernorm.weight", 1, CH, 0);
    LADD("self_attn.q_proj.weight", 2, CNH * CHD, CH);
    LADD("self_attn.k_proj.weight", 2, CNKV * CHD, CH);
    LADD("self_attn.v_proj.weight", 2, CNKV * CHD, CH);
    LADD("self_attn.o_proj.weight", 2, CH, CNH * CHD);
    LADD("self_attn.q_norm.weight", 1, CHD, 0);
    LADD("self_attn.k_norm.weight", 1, CHD, 0);
    LADD("mlp.gate_proj.weight", 2, CFF, CH);
    LADD("mlp.up_proj.weight", 2, CFF, CH);
    LADD("mlp.down_proj.weight", 2, CH, CFF);
#undef LADD
  }
#undef ADD
  return n;
}

static void test_convert_gemma4(void) {
  char dir[] = "/tmp/oc-convert-gemma-XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char cfgp[256], tokp[256], stp[256], gout[256];
  snprintf(cfgp, sizeof cfgp, "%s/config.json", dir);
  snprintf(tokp, sizeof tokp, "%s/tokenizer.json", dir);
  snprintf(stp, sizeof stp, "%s/model.safetensors", dir);
  snprintf(gout, sizeof gout, "%s/out-gemma4.gguf", dir);

  write_text(cfgp,
             "{\"model_type\":\"gemma4\",\"hidden_size\":8,\"num_hidden_layers\":2,"
             "\"num_attention_heads\":2,\"num_key_value_heads\":1,"
             "\"intermediate_size\":16,\"rms_norm_eps\":1e-06,\"rope_theta\":1000000.0,"
             "\"vocab_size\":16,\"max_position_embeddings\":64,"
             "\"sliding_window\":32,\"sliding_window_pattern\":2,"
             "\"final_logit_softcapping\":30.0,\"head_dim\":4,"
             "\"bos_token_id\":1,\"eos_token_id\":2,"
             "\"layer_types\":[\"sliding_attention\",\"full_attention\"]}");
  write_text(tokp,
             "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"<unk>\":0,\"<s>\":1,\"</s>\":2,"
             "\"a\":3,\"b\":4,\"ab\":5,\"c\":6},\"merges\":[\"a b\"]},"
             "\"added_tokens\":[{\"id\":1,\"content\":\"<s>\",\"special\":true},"
             "{\"id\":2,\"content\":\"</s>\",\"special\":true}]}");

  StT ts[64];
  size_t nt = build_hf_gemma(ts);
  write_safetensors(stp, ts, nt);

  ConvertOpts o = {.arch_override = NULL, .outtype = "F32"};
  CHECK(tool_convert(dir, gout, &o, 0) == 0);

  GgufFile g;
  char err[256];
  CHECK(gguf_open(&g, gout, err, sizeof err) == 0);
  char* arch = gguf_architecture(&g);
  CHECK(arch && strcmp(arch, "gemma4") == 0);
  free(arch);

  /* Sandwich mapping: post_attention != ffn_norm; pre_feedforward is ffn_norm. */
  CHECK(gguf_tensor(&g, "blk.0.post_attention_norm.weight") != NULL);
  CHECK(gguf_tensor(&g, "blk.0.ffn_norm.weight") != NULL);
  CHECK(gguf_tensor(&g, "blk.0.post_ffw_norm.weight") != NULL);
  CHECK(gguf_tensor(&g, "output.weight") == NULL); /* tied head dropped */

  /* +1 bake on RMSNorm weights (HF fill was ~[-0.5,0.5] centered; post-bake mean > 0.4). */
  const GgufTensorInfo* an = gguf_tensor(&g, "blk.0.attn_norm.weight");
  CHECK(an && an->ggml_type == OC_F32 && an->dims[0] == CH);
  const float* aw = (const float*)an->data;
  float mean = 0.0f;
  for (size_t i = 0; i < CH; ++i) mean += aw[i];
  mean /= (float)CH;
  CHECK(mean > 0.4f);

  uint32_t sw = 0, softcap_ok = 0;
  CHECK(gguf_get_u32(&g, "gemma4.attention.sliding_window", &sw) && sw == 32);
  float softcap = 0.0f;
  softcap_ok = gguf_get_f32(&g, "gemma4.final_logit_softcapping", &softcap);
  CHECK(softcap_ok && fabsf(softcap - 30.0f) < 1e-5f);
  const GgufValue* pat = gguf_get_arr(&g, "gemma4.attention.sliding_window_pattern");
  CHECK(pat && pat->v.arr.n == 2);
  CHECK(pat->v.arr.items[0].v.u == 1 && pat->v.arr.items[1].v.u == 0);

  Model m;
  CHECK(model_load(&m, &g, 0, false, err, sizeof err) == 0);
  CHECK(m.family == MODEL_GEMMA4);
  float* logits = m.forward(m.handle, 3, 0, true);
  CHECK(logits != NULL);
  int nonzero = 0;
  for (size_t i = 0; i < CV; ++i)
    if (fabsf(logits[i]) > 1e-8f) nonzero = 1;
  CHECK(nonzero);
  model_free(&m);

  for (size_t i = 0; i < nt; ++i) free(ts[i].data);
  unlink(cfgp);
  unlink(tokp);
  unlink(stp);
  unlink(gout);
  rmdir(dir);
  printf("ok tools convert gemma4 (sandwich map, +1 bake, loads)\n");
}

/* The JSON parser reads untrusted files (safetensors headers, config.json,
 * tokenizer.json), so it must reject malformed input without crashing (ASAN). */
static void test_convert_json(void) {
  static const char* bad[] = {
      "", " ", "{", "}", "[", "]", "{\"a\"}", "{\"a\":}", "{\"a\":1",
      "[1,2", "\"abc", "tru", "nul", "fals", "{,}", "[,]", "{\"a\":1,}",
      "[1,]", "12x", "1.2.3", "\"\\q\"", "\"\\u12\"", "\"\\uZZZZ\"",
      "{1:2}", "[\"a\":1]", "{\"k\" ", "nan", "0x1", NULL};
  for (size_t i = 0; bad[i]; ++i) {
    JNode* n = json_parse(bad[i], strlen(bad[i]));
    CHECK(n == NULL);
  }
  static const char* good[] = {
      "{}", "[]", "  123 ", "-4.5e10", "\"hi\\n\\t\\u0041\"", "true",
      "false", "null", "{\"k\":[1,2,{\"x\":\"y\"}]}", "[[[]]]", "0", NULL};
  for (size_t i = 0; good[i]; ++i) {
    JNode* n = json_parse(good[i], strlen(good[i]));
    CHECK(n != NULL);
    jfree(n);
  }
  /* deep nesting must fail via the depth cap, not overflow the stack */
  char deep[512];
  for (int i = 0; i < 400; ++i) deep[i] = '[';
  deep[400] = 0;
  CHECK(json_parse(deep, 400) == NULL);
  printf("ok tools convert json parser (rejects malformed input, ASAN-clean)\n");
}

void test_tools(void) {
  char* a = fixture_a();
  test_tool_quantize(a);
  test_tool_prune(a);
  test_tool_prune_sparse();
  test_tool_merge(a);
  unlink(a);
  free(a);
  test_convert_json();
  test_convert();
  test_convert_gemma4();
}
