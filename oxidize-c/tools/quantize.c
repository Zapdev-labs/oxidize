/* oxidize-c-quantize: GGUF -> GGUF re-quantization.
 *   --input in.gguf --output out.gguf
 *     [--target F32|F16|Q8_0|Q4_0|Q2_K|Q3_K|Q4_K|Q5_K|Q6_K|IQ4_XS|AL5_XS]
 * Source tensors are dequantized (any of the 22 readable types) and re-encoded.
 * Only 2-D tensors whose row length is encodable for the target are converted;
 * 1-D tensors (norms, rope_freqs) and anything else are copied verbatim.
 * Default target is AL5_XS (oxidize-c-requant is the same binary).
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/quant.h"
#include "../src/tensor.h"
#include "gguf_write.h"

static uint64_t n_rows(const GgufTensorInfo* t) {
  uint64_t r = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) r *= t->dims[d];
  return r;
}

int tool_quantize(const char* in, const char* out, const char* target, int verbose) {
  uint32_t tt = gw_type_id(target);
  if (tt == UINT32_MAX || !gw_encodable(tt)) {
    fprintf(stderr,
            "quantize: unsupported target '%s' "
            "(F32 F16 Q8_0 Q4_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ4_XS AL5_XS)\n",
            target);
    return 1;
  }

  char err[256];
  GgufFile f;
  if (gguf_open(&f, in, err, sizeof err) != 0) {
    fprintf(stderr, "gguf_open: %s\n", err);
    return 1;
  }

  size_t nt = f.n_tensors;
  GwWriter w = {0};
  GwTensor* ts = calloc(nt ? nt : 1, sizeof(GwTensor));
  int* conv = calloc(nt ? nt : 1, sizeof(int));
  if (!ts || !conv) {
    fprintf(stderr, "oom\n");
    return 1;
  }
  int rc = 1;
  for (size_t i = 0; i < nt; ++i) {
    const GgufTensorInfo* t = &f.tensors[i];
    uint64_t rows = n_rows(t);
    /* Convertible: 2-D, target-encodable row length, source readable, and not
     * already the target type. oc_row_bytes gates K-quant (cols % 256) etc. */
    conv[i] = t->n_dims == 2 && t->ggml_type != tt &&
              oc_row_bytes(tt, t->dims[0]) != 0 &&
              oc_row_bytes(t->ggml_type, t->dims[0]) != 0;
    uint32_t ot = conv[i] ? tt : t->ggml_type;
    size_t rb = oc_row_bytes(ot, t->dims[0]);
    if (rb == 0) {
      fprintf(stderr, "tensor %s: unsupported type %u\n", t->name, t->ggml_type);
      goto done;
    }
    if (!gw_data_ok(&f, t, rows * oc_row_bytes(t->ggml_type, t->dims[0]))) {
      fprintf(stderr, "tensor %s: data runs past end of file (truncated GGUF?)\n",
              t->name);
      goto done;
    }
    ts[i].name = t->name;
    ts[i].n_dims = t->n_dims;
    memcpy(ts[i].dims, t->dims, sizeof ts[i].dims);
    ts[i].type = ot;
    ts[i].size = rows * rb;
  }

  if (gw_open(&w, out, f.kvs, f.n_kv, f.alignment, ts, nt, (int)tt) != 0) goto done;

  for (size_t i = 0; i < nt; ++i) {
    const GgufTensorInfo* t = &f.tensors[i];
    if (!conv[i]) {
      if (gw_tensor(&w, t->data, ts[i].size) != 0) goto done;
      if (verbose)
        printf("[%3zu/%zu] %-40s copied (type %u, %" PRIu64 " B)\n", i + 1, nt,
               t->name, t->ggml_type, ts[i].size);
      continue;
    }
    uint8_t* dst = malloc(ts[i].size);
    if (!dst) {
      fprintf(stderr, "oom (%" PRIu64 " B)\n", ts[i].size);
      goto done;
    }
    gw_requant(t->data, t->ggml_type, tt, n_rows(t), t->dims[0], dst);
    int e = gw_tensor(&w, dst, ts[i].size);
    free(dst);
    if (e != 0) goto done;
    if (verbose)
      printf("[%3zu/%zu] %-40s %" PRIu64 "x%" PRIu64 " type %u -> %s\n", i + 1, nt,
             t->name, t->dims[0], n_rows(t), t->ggml_type, target);
    if (verbose) fflush(stdout);
  }
  rc = gw_close(&w) != 0;
  if (!rc && verbose) printf("done: %s\n", out);

done:
  if (w.f) gw_close(&w);
  free(ts);
  free(conv);
  gguf_close(&f);
  return rc;
}

#ifndef OC_TOOLS_LIB
int main(int argc, char** argv) {
  const char *in = NULL, *out = NULL, *target = "AL5_XS";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--input") && i + 1 < argc) in = argv[++i];
    else if (!strcmp(argv[i], "--output") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--target") && i + 1 < argc) target = argv[++i];
    else {
      fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }
  if (!in || !out) {
    fprintf(stderr,
            "usage: oxidize-c-quantize --input in.gguf --output out.gguf "
            "[--target F32|F16|Q8_0|Q4_0|Q2_K|Q3_K|Q4_K|Q5_K|Q6_K|IQ4_XS|AL5_XS]\n");
    return 1;
  }
  int rc = tool_quantize(in, out, target, 1);
  oc_pool_free();
  return rc;
}
#endif
