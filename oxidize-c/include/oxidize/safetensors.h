/*
 * safetensors.h — HuggingFace SafeTensors file reader.
 *
 * Port of oxidize-core/src/format/safetensors.rs reader path to C11.
 *
 * The SafeTensors on-disk layout is:
 *   - 8-byte little-endian u64 header length (byte count of the JSON header)
 *   - JSON header object mapping tensor names to tensor descriptors
 *   - raw tensor data (concatenated, offsets are relative to this section)
 *
 * The JSON header looks like:
 *   {"tensor_name":{"dtype":"F32","shape":[1024,1024],
 *                   "data_offsets":[0,4194304]}, ...}
 *
 * This reader implements a minimal JSON parser that extracts only the keys
 * needed for tensor descriptors (dtype, shape, data_offsets). It supports
 * reading via mmap (preferred) or a malloc'd buffer on platforms where mmap
 * is unavailable.
 */
#ifndef OXIDIZE_SAFETENSORS_H
#define OXIDIZE_SAFETENSORS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of dimensions supported per tensor. */
#define OC_SAFETENSORS_MAX_DIMS 8

/* Maximum tensor-name length (NUL-terminated). */
#define OC_SAFETENSORS_NAME_LEN 128

/* Maximum dtype string length (NUL-terminated). */
#define OC_SAFETENSORS_DTYPE_LEN 16

/* A single parsed tensor descriptor. `data_offset` and `data_length` are byte
 * ranges relative to the start of the raw data section (i.e. they are the
 * raw values from the JSON `data_offsets` array, with data_length computed as
 * end - begin). `shape[i]` follows row-major order (shape[0] is the
 * outermost dimension). */
typedef struct OcSafetensorsTensor {
    char     name[OC_SAFETENSORS_NAME_LEN];
    char     dtype[OC_SAFETENSORS_DTYPE_LEN];
    uint64_t shape[OC_SAFETENSORS_MAX_DIMS];
    uint32_t n_dims;
    uint64_t data_offset;   /* byte offset from start of raw data section */
    uint64_t data_length;   /* byte length of this tensor's data */
} OcSafetensorsTensor;

/* Parsed SafeTensors file. `tensors` is a malloc'd array of `n_tensors`
 * entries owned by this struct and freed by oc_safetensors_close(). `raw_data`
 * points at the start of the raw tensor data section; it aliases either the
 * mmap'd region (when `mmapped` is true) or a malloc'd buffer. `data_start`
 * is the byte offset into the file where raw data begins (header_len + 8). */
typedef struct OcSafetensorsFile {
    OcSafetensorsTensor *tensors;
    size_t               n_tensors;
    uint64_t             data_start;   /* file offset of raw data section */
    void                *raw_data;     /* pointer to raw data section start */
    uint64_t             file_size;    /* total file size in bytes */
    bool                 mmapped;      /* true if raw_data is mmap-owned */
} OcSafetensorsFile;

/* Open and parse a .safetensors file from disk. The file is read into a
 * malloc'd buffer (the mmap fast path is used when available). On success
 * `*out` is populated and must be released via oc_safetensors_close(). On
 * error `*out` is zeroed.
 *
 * Returns OC_OK, OC_ERR_IO (file open/read/stat failure),
 * OC_ERR_FORMAT (truncated file, bad header length, or malformed JSON),
 * OC_ERR_OOM, or OC_ERR_INVALID_ARG (NULL args). */
OcError oc_safetensors_open(const char *path, OcSafetensorsFile *out);

/* Find a tensor by name. On success sets `*out` to point into `st->tensors`
 * (valid until oc_safetensors_close()). Returns OC_OK or
 * OC_ERR_INVALID_ARG (NULL args) / OC_ERR_TENSOR (not found). */
OcError oc_safetensors_get_tensor(const OcSafetensorsFile *st,
                                  const char *name,
                                  const OcSafetensorsTensor **out);

/* Return a pointer to a tensor's raw data within the file's raw data section.
 * The pointer aliases `st->raw_data` and is valid until oc_safetensors_close().
 * Returns OC_OK + `*out_data`, or OC_ERR_INVALID_ARG (NULL args) /
 * OC_ERR_FORMAT (tensor not backed by this file) / OC_ERR_TENSOR. */
OcError oc_safetensors_get_tensor_data(const OcSafetensorsFile *st,
                                       const OcSafetensorsTensor *tensor,
                                       const void **out_data);

/* Return the number of tensors in the file. Returns 0 if `st` is NULL. */
size_t oc_safetensors_n_tensors(const OcSafetensorsFile *st);

/* Free all resources owned by `st` (tensor array + raw data buffer / mmap).
 * Safe on NULL or zeroed OcSafetensorsFile. After this call, `*st` is
 * zeroed. Does NOT free `st` itself. */
void oc_safetensors_close(OcSafetensorsFile *st);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SAFETENSORS_H */
