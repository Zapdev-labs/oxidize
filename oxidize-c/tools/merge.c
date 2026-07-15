/* oxidize-c-merge: model soup. out = alpha*A + (1-alpha)*B, elementwise, over
 * same-named tensors.
 *   --a a.gguf --b b.gguf --output out.gguf [--alpha 0.5]
 * Tensor count, names, dims must match exactly; anything else is a hard error.
 * Both sides are dequantized to f32, blended, and re-encoded to A's type when
 * that type has an encoder (F32/F16/Q8_0/Q4_0/AL5_XS) — otherwise the tensor is
 * emitted as F32 (never silently re-encoded through a codec we cannot write).
 * KV metadata is A's.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/quant.h"
#include "../src/tensor.h"
#include "gguf_write.h"

typedef struct {
  const uint8_t *a, *b;
  uint32_t ta, tb, to;
  size_t cols, rba, rbb, rbo;
  float alpha;
  uint8_t* dst;
} BlendJob;

static void blend_rows(void* ctx, size_t r0, size_t r1) {
  BlendJob* j = (BlendJob*)ctx;
  float* ra = malloc(j->cols * sizeof(float));
  float* rb = malloc(j->cols * sizeof(float));
  if (!ra || !rb) abort();
  for (size_t r = r0; r < r1; ++r) {
    if (oc_dequant_row(j->ta, j->a + r * j->rba, ra, j->cols) != 0 ||
        oc_dequant_row(j->tb, j->b + r * j->rbb, rb, j->cols) != 0) {
      fprintf(stderr, "merge: dequant failed (types %u/%u)\n", j->ta, j->tb);
      exit(1);
    }
    for (size_t c = 0; c < j->cols; ++c)
      ra[c] = j->alpha * ra[c] + (1.0f - j->alpha) * rb[c];
    if (gw_encode_row(j->to, ra, j->dst + r * j->rbo, j->cols) != 0) {
      fprintf(stderr, "merge: encode failed (type %u)\n", j->to);
      exit(1);
    }
  }
  free(ra);
  free(rb);
}

int tool_merge(const char* pa, const char* pb, const char* out, float alpha,
               int verbose) {
  char err[256];
  GgufFile a, b;
  if (gguf_open(&a, pa, err, sizeof err) != 0) {
    fprintf(stderr, "gguf_open %s: %s\n", pa, err);
    return 1;
  }
  if (gguf_open(&b, pb, err, sizeof err) != 0) {
    fprintf(stderr, "gguf_open %s: %s\n", pb, err);
    gguf_close(&a);
    return 1;
  }

  GwWriter w = {0};
  GwTensor* ts = calloc(a.n_tensors ? a.n_tensors : 1, sizeof(GwTensor));
  const GgufTensorInfo** src_b =
      calloc(a.n_tensors ? a.n_tensors : 1, sizeof(*src_b));
  if (!ts || !src_b) {
    fprintf(stderr, "oom\n");
    return 1;
  }
  int rc = 1;

  if (a.n_tensors != b.n_tensors) {
    fprintf(stderr, "merge: tensor count mismatch: %zu vs %zu\n", a.n_tensors,
            b.n_tensors);
    goto done;
  }
  for (size_t i = 0; i < a.n_tensors; ++i) {
    const GgufTensorInfo* t = &a.tensors[i];
    const GgufTensorInfo* u = gguf_tensor(&b, t->name);
    if (!u) {
      fprintf(stderr, "merge: tensor '%s' missing from %s\n", t->name, pb);
      goto done;
    }
    if (u->n_dims != t->n_dims) {
      fprintf(stderr, "merge: '%s' rank mismatch: %u vs %u\n", t->name, t->n_dims,
              u->n_dims);
      goto done;
    }
    for (uint32_t d = 0; d < t->n_dims; ++d)
      if (t->dims[d] != u->dims[d]) {
        fprintf(stderr, "merge: '%s' shape mismatch at dim %u: %" PRIu64 " vs %" PRIu64
                        "\n",
                t->name, d, t->dims[d], u->dims[d]);
        goto done;
      }
    if (oc_row_bytes(t->ggml_type, t->dims[0]) == 0 ||
        oc_row_bytes(u->ggml_type, u->dims[0]) == 0) {
      fprintf(stderr, "merge: '%s' unsupported type (%u / %u)\n", t->name,
              t->ggml_type, u->ggml_type);
      goto done;
    }
    uint64_t rows = 1;
    for (uint32_t d = 1; d < t->n_dims; ++d) rows *= t->dims[d];
    if (!gw_data_ok(&a, t, rows * oc_row_bytes(t->ggml_type, t->dims[0])) ||
        !gw_data_ok(&b, u, rows * oc_row_bytes(u->ggml_type, u->dims[0]))) {
      fprintf(stderr, "merge: '%s' data runs past end of file (truncated GGUF?)\n",
              t->name);
      goto done;
    }
    uint32_t ot = gw_encodable(t->ggml_type) ? t->ggml_type : OC_F32;
    ts[i].name = t->name;
    ts[i].n_dims = t->n_dims;
    memcpy(ts[i].dims, t->dims, sizeof ts[i].dims);
    ts[i].type = ot;
    ts[i].size = rows * oc_row_bytes(ot, t->dims[0]);
    src_b[i] = u;
  }

  if (gw_open(&w, out, a.kvs, a.n_kv, a.alignment, ts, a.n_tensors, -1) != 0)
    goto done;
  oc_pool_init(0);
  for (size_t i = 0; i < a.n_tensors; ++i) {
    const GgufTensorInfo* t = &a.tensors[i];
    uint64_t rows = 1;
    for (uint32_t d = 1; d < t->n_dims; ++d) rows *= t->dims[d];
    uint8_t* dst = malloc(ts[i].size);
    if (!dst) {
      fprintf(stderr, "oom (%" PRIu64 " B)\n", ts[i].size);
      goto done;
    }
    BlendJob j = {.a = t->data,
                  .b = src_b[i]->data,
                  .ta = t->ggml_type,
                  .tb = src_b[i]->ggml_type,
                  .to = ts[i].type,
                  .cols = t->dims[0],
                  .rba = oc_row_bytes(t->ggml_type, t->dims[0]),
                  .rbb = oc_row_bytes(src_b[i]->ggml_type, t->dims[0]),
                  .rbo = oc_row_bytes(ts[i].type, t->dims[0]),
                  .alpha = alpha,
                  .dst = dst};
    oc_parallel_for(rows, blend_rows, &j);
    int e = gw_tensor(&w, dst, ts[i].size);
    free(dst);
    if (e != 0) goto done;
    if (verbose)
      printf("merged %-40s type %u (%" PRIu64 " B)\n", t->name, ts[i].type,
             ts[i].size);
  }
  rc = gw_close(&w) != 0;
  if (!rc && verbose) printf("done: %s (alpha %.3f)\n", out, (double)alpha);

done:
  if (w.f) gw_close(&w);
  free(ts);
  free(src_b);
  gguf_close(&a);
  gguf_close(&b);
  return rc;
}

#ifndef OC_TOOLS_LIB
int main(int argc, char** argv) {
  const char *pa = NULL, *pb = NULL, *out = NULL;
  float alpha = 0.5f;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--a") && i + 1 < argc) pa = argv[++i];
    else if (!strcmp(argv[i], "--b") && i + 1 < argc) pb = argv[++i];
    else if (!strcmp(argv[i], "--output") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) alpha = strtof(argv[++i], NULL);
    else {
      fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }
  if (!pa || !pb || !out) {
    fprintf(stderr,
            "usage: oxidize-c-merge --a a.gguf --b b.gguf --output out.gguf "
            "[--alpha 0.5]\n");
    return 1;
  }
  int rc = tool_merge(pa, pb, out, alpha, 1);
  oc_pool_free();
  return rc;
}
#endif
