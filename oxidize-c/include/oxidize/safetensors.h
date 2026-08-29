/* safetensors.h — HuggingFace SafeTensors file reader. */
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

typedef struct OcSafetensorsTensor {
    char     name[OC_SAFETENSORS_NAME_LEN];
    char     dtype[OC_SAFETENSORS_DTYPE_LEN];
    uint64_t shape[OC_SAFETENSORS_MAX_DIMS];
    uint32_t n_dims;
    uint64_t data_offset;   /* byte offset from start of raw data section */
    uint64_t data_length;   /* byte length of this tensor's data */
} OcSafetensorsTensor;

typedef struct OcSafetensorsFile {
    OcSafetensorsTensor *tensors;
    size_t               n_tensors;
    uint64_t             data_start;   /* file offset of raw data section */
    void                *raw_data;     /* pointer to raw data section start */
    uint64_t             file_size;    /* total file size in bytes */
    bool                 mmapped;      /* true if raw_data is mmap-owned */
} OcSafetensorsFile;

/* Open and parse a .safetensors file from disk. The file is read into a malloc'd buffer (the mmap fast path is used when available). On success */
OcError oc_safetensors_open(const char *path, OcSafetensorsFile *out);

/* Find a tensor by name. On success sets `*out` to point into `st->tensors`
 * (valid until oc_safetensors_close()). Returns OC_OK or
 * OC_ERR_INVALID_ARG (NULL args) / OC_ERR_TENSOR (not found). */
OcError oc_safetensors_get_tensor(const OcSafetensorsFile *st,
                                  const char *name,
                                  const OcSafetensorsTensor **out);

/* Return a pointer to a tensor's raw data within the file's raw data section. */
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
