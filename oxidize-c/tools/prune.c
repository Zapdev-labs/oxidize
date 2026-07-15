/* oxidize-c-prune: GGUF -> GGUF, dropping tensors by name substring.
 *   --input in.gguf --output out.gguf [--keep SUB]... [--drop SUB]...
 * With any --keep, only tensors matching one of them survive; --drop then
 * removes matches from whatever is left. Tensor payloads are copied verbatim.
 * ponytail: name filter only — no magnitude/Wanda row pruning (that needs
 * calibration activations and a re-encode; out of scope here).
 */
#include <inttypes.h>
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
    uint64_t rows = 1;
    for (uint32_t d = 1; d < t->n_dims; ++d) rows *= t->dims[d];
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

#ifndef OC_TOOLS_LIB
int main(int argc, char** argv) {
  const char *in = NULL, *out = NULL;
  const char* keep[64];
  const char* drop[64];
  size_t nkeep = 0, ndrop = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--input") && i + 1 < argc) in = argv[++i];
    else if (!strcmp(argv[i], "--output") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--keep") && i + 1 < argc && nkeep < 64) keep[nkeep++] = argv[++i];
    else if (!strcmp(argv[i], "--drop") && i + 1 < argc && ndrop < 64) drop[ndrop++] = argv[++i];
    else {
      fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }
  if (!in || !out) {
    fprintf(stderr,
            "usage: oxidize-c-prune --input in.gguf --output out.gguf "
            "[--keep SUB]... [--drop SUB]...\n");
    return 1;
  }
  return tool_prune(in, out, keep, nkeep, drop, ndrop, 1);
}
#endif
