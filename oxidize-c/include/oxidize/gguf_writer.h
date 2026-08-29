/* gguf_writer.h — GGUF v3 file writer (serializer). */
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
    /* Pending tensor data buffer (opaque GwPendingList*). */
    void *pending;
    bool     finalized;        /* true after finalize() — further writes error */
    bool     owns_fp;          /* true when we fopen'd fp (not caller-provided) */
} OcGgufWriter;

/* Initialize a writer: open `path` for writing, emit the GGUF header (magic, version=3, tensor_count=0, metadata_kv_count=0), and write the `general.architecture` string metadata (as required by the GGUF spec). */
OcError oc_gguf_writer_init(const char *path, const char *arch_name,
                            OcGgufWriter *w);

OcError oc_gguf_writer_init_from_file(const char *path, const OcGgufFile *gf,
                                     OcGgufWriter *w);


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

OcError oc_gguf_writer_add_tensor(OcGgufWriter *w, const char *name,
                                  uint32_t n_dims, const uint64_t *dims,
                                  uint32_t type, const void *data,
                                  uint64_t data_size);

/* Finalize the writer: seek back to the header and patch the real tensor_count and metadata_kv_count, then flush the file. */
OcError oc_gguf_writer_finalize(OcGgufWriter *w);

/* Close the file handle (if owned) and zero the struct. Safe on NULL or
 * already-freed writers. Does NOT free `w` itself (caller owns the struct).
 * Calls finalize() internally if not already finalized. */
void oc_gguf_writer_free(OcGgufWriter *w);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GGUF_WRITER_H */
