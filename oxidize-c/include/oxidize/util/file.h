/* file.h — file utility helpers. */
#ifndef OXIDIZE_UTIL_FILE_H
#define OXIDIZE_UTIL_FILE_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Get file size in bytes. Returns OC_OK on success, OC_ERR_IO on failure
 * (e.g. file not found). Writes size to `*out_size`. */
OcError oc_file_size(const char *path, uint64_t *out_size);

/* Read the entire file into a freshly malloc'd buffer. Writes `*out_data` (caller frees) and `*out_size`; returns OC_OK, OC_ERR_IO, OC_ERR_OOM, or OC_ERR_INVALID_ARG. A zero-length file returns OC_OK with *out_size=0 and a non-NULL 1-byte malloc (distinguishes empty from OOM); the caller still frees it. */
OcError oc_file_read_all(const char *path, uint8_t **out_data, size_t *out_size);

/* Write `n` bytes to `path`. Truncates the file if it exists. Returns OC_OK
 * or OC_ERR_IO. */
OcError oc_file_write_all(const char *path, const uint8_t *data, size_t n);

/* Check if a file exists (and is a regular file). Returns true/false. */
bool oc_file_exists(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_UTIL_FILE_H */
