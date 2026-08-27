/*
 * gguf_writer.h — GGUF v3 file writer (serializer).
 *
 * Companion to the parser in gguf.h. Serializes tensors and metadata KV
 * pairs to the GGUF binary format on disk. The header (magic, version,
 * tensor_count, metadata_kv_count) is written up front with zero
 * placeholders; oc_gguf_writer_finalize() seeks back to patch the real
 * counts before closing the file.
 *
 * Port of the writer concepts in oxidize-core/src/format/gguf.rs to C11.
 *
 * On-disk layout (GGUF v3, little-endian):
 *   [ magic: u32 = 0x46554747 ("GGUF") ]
 *   [ version: u32 = 3 ]
 *   [ tensor_count: u64 ]
 *   [ metadata_kv_count: u64 ]
 *   [ metadata_kv ... ]          (each: key-string, value_type:u32, value)
 *   [ tensor_info ... ]          (each: name-string, n_dims:u32, dims:u64[],
 *                                 type:u32, offset:u64)
 *   [ padding to 32-byte alignment ]
 *   [ tensor_data ... ]
 *
 * String encoding: u64 length prefix + raw bytes (no NUL).
 *
 * Usage:
 *   OcGgufWriter w;
 *   oc_gguf_writer_init("out.gguf", "llama", &w);
 *   oc_gguf_writer_add_string(&w, "general.name", "my-model");
 *   oc_gguf_writer_add_uint32(&w, "llama.context_length", 4096);
 *   oc_gguf_writer_add_tensor(&w, "tok_embeddings.weight", 2, dims,
 *                             0, data, data_size);
 *   oc_gguf_writer_finalize(&w);
 *   oc_gguf_writer_free(&w);
 */
#ifndef OXIDIZE_GGUF_WRITER_H
#define OXIDIZE_GGUF_WRITER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"  /* OC_GGUF_MAGIC, OC_GGUF_VERSION_3, etc. */

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum tensor dims accepted by the writer. Matches OC_GGUF_MAX_DIMS. */
#define OC_GGUF_WRITER_MAX_DIMS 8

/* Writer state. All fields are private; callers pass a pointer to a
 * stack- or heap-allocated struct (as with OcGgufFile). */
typedef struct OcGgufWriter {
    FILE    *fp;               /* output file handle                          */
    uint64_t offset;           /* current write cursor (tracked manually)     */
    uint64_t tensor_count;     /* number of tensors added so far              */
    uint64_t metadata_count;   /* number of metadata KV pairs added           */
    /* Byte offsets into the file where the header counts live, so
     * finalize() can seek back and patch them. */
    long     tensor_count_off;
    long     metadata_count_off;
    /* Start of the tensor data section (after all tensor infos). */
    uint64_t data_section_start;
    /* Running offset of the next tensor's data (relative to file start). */
    uint64_t next_data_offset;
    /* Pending tensor data buffer (opaque GwPendingList*). Tensor data is
     * buffered in memory during add_tensor() and written at finalize()
     * time, after all tensor infos. This matches the GGUF layout where
     * all tensor info entries precede the data section. */
    void *pending;
    bool     finalized;        /* true after finalize() — further writes error */
    bool     owns_fp;          /* true when we fopen'd fp (not caller-provided) */
} OcGgufWriter;

/* Initialize a writer: open `path` for writing, emit the GGUF header (magic,
 * version=3, tensor_count=0, metadata_kv_count=0), and write the
 * `general.architecture` string metadata (as required by the GGUF spec).
 *
 * `path` may be NULL only when the caller intends to set `fp` manually
 * before adding metadata — the normal API path always passes a path.
 *
 * Returns OC_OK on success, OC_ERR_IO on file open/write failure,
 * OC_ERR_OOM on allocation failure, OC_ERR_INVALID_ARG on bad args.
 * On error, `*w` is zeroed. */
OcError oc_gguf_writer_init(const char *path, const char *arch_name,
                            OcGgufWriter *w);

/* ─── Metadata KV writers ────────────────────────────────────────────────
 *
 * Each writes a single metadata KV pair to the file and increments the
 * internal metadata count. Must be called AFTER init() and BEFORE
 * add_tensor() (metadata precedes tensor info in the GGUF layout).
 *
 * Returns OC_OK, OC_ERR_IO, OC_ERR_INVALID_ARG (NULL writer / NULL key /
 * writer finalized / tensors already added). */

OcError oc_gguf_writer_add_string(OcGgufWriter *w, const char *key,
                                  const char *value);

OcError oc_gguf_writer_add_uint32(OcGgufWriter *w, const char *key,
                                  uint32_t value);

OcError oc_gguf_writer_add_uint64(OcGgufWriter *w, const char *key,
                                  uint64_t value);

OcError oc_gguf_writer_add_float32(OcGgufWriter *w, const char *key,
                                  float value);

OcError oc_gguf_writer_add_array_string(OcGgufWriter *w, const char *key,
                                        const char *const *values, size_t count);

/* ─── Tensor writer ──────────────────────────────────────────────────────
 *
 * Writes a tensor info entry (name, n_dims, dims, type, offset) and the
 * raw tensor data. The data section is aligned to a 32-byte boundary; this
 * function inserts padding as needed. Must be called AFTER all metadata
 * writes and BEFORE finalize().
 *
 *   name      — NUL-terminated tensor name
 *   n_dims    — number of valid dimensions (1..OC_GGUF_WRITER_MAX_DIMS)
 *   dims      — array of `n_dims` dimension sizes
 *   type      — ggml dtype id (0=F32, 1=F16, 8=Q8_0, ...)
 *   data      — raw tensor bytes
 *   data_size — byte length of `data`
 *
 * Returns OC_OK, OC_ERR_IO, OC_ERR_INVALID_ARG, OC_ERR_OOM. */
OcError oc_gguf_writer_add_tensor(OcGgufWriter *w, const char *name,
                                  uint32_t n_dims, const uint64_t *dims,
                                  uint32_t type, const void *data,
                                  uint64_t data_size);

/* Finalize the writer: seek back to the header and patch the real
 * tensor_count and metadata_kv_count, then flush the file. After this
 * call, the writer is in "finalized" state and further add_* calls return
 * OC_ERR_INVALID_ARG. safe to call multiple times (subsequent calls are
 * no-ops).
 *
 * Returns OC_OK, OC_ERR_IO. */
OcError oc_gguf_writer_finalize(OcGgufWriter *w);

/* Close the file handle (if owned) and zero the struct. Safe on NULL or
 * already-freed writers. Does NOT free `w` itself (caller owns the struct).
 * Calls finalize() internally if not already finalized. */
void oc_gguf_writer_free(OcGgufWriter *w);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GGUF_WRITER_H */
