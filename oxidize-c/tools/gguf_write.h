/* Shared bits of the offline tools: a GGUF v3 writer (streaming: header first,
 * then tensor payloads in order), the row encoders, and the three tool cores.
 * Factored out of the old tools/requant.c — src/gguf.c is read-only. */
#ifndef OC_TOOLS_GGUF_WRITE_H
#define OC_TOOLS_GGUF_WRITE_H

#include <stdint.h>
#include <stdio.h>

#include "../src/gguf.h"

typedef struct {
  const char* name;
  uint32_t n_dims;
  uint64_t dims[GGUF_MAX_DIMS];
  uint32_t type; /* output ggml type */
  uint64_t size; /* output payload bytes */
} GwTensor;

typedef struct {
  FILE* f;
  uint64_t align;
  uint64_t pos; /* bytes written into the data section so far */
  int err;
} GwWriter;

/* Writes header + KVs + tensor infos + padding. `file_type` >= 0 patches (or
 * appends) general.file_type. Payloads must then be pushed with gw_tensor() in
 * the same order as `ts`. Returns 0 on success. */
int gw_open(GwWriter* w, const char* path, const GgufKv* kvs, size_t n_kv,
            uint64_t align, const GwTensor* ts, size_t nt, int file_type);
int gw_tensor(GwWriter* w, const void* data, uint64_t size); /* pads to align */
int gw_close(GwWriter* w);

/* 1 if `bytes` of this tensor's payload really lie inside the mapped file.
 * gguf.c does not bounds-check declared tensor sizes against the file length. */
int gw_data_ok(const GgufFile* f, const GgufTensorInfo* t, uint64_t bytes);

/* Encodable target types only: F32, F16, Q8_0, Q4_0, AL5_XS.
 * UINT32_MAX if the name is unknown / has no encoder. */
uint32_t gw_type_id(const char* name);
int gw_encodable(uint32_t type);
/* Encode n floats (n % 32 == 0 for the block types) into a row of `type`. */
int gw_encode_row(uint32_t type, const float* x, uint8_t* dst, size_t n);
/* rows x cols: dequant(src, src_type) -> encode(dst, dst_type), threaded. */
void gw_requant(const uint8_t* src, uint32_t src_type, uint32_t dst_type,
                size_t rows, size_t cols, uint8_t* dst);

/* Tool cores (the CLI mains are thin wrappers; the tests call these). Nonzero
 * return = failure, with a message on stderr. */
int tool_quantize(const char* in, const char* out, const char* target, int verbose);
int tool_prune(const char* in, const char* out, const char* const* keep,
               size_t nkeep, const char* const* drop, size_t ndrop, int verbose);
int tool_merge(const char* a, const char* b, const char* out, float alpha,
               int verbose);

#endif
