/*
 * bytes.h — OcBytes byte buffer utilities for serialization.
 *
 * Owning or borrowing byte buffer with append and read primitives
 * (little-endian). Port concept: oxidize-core/src/util/bytes.rs (the Rust
 * side centralizes mmap + byte reads; here OcBytes provides a growable
 * buffer for serialization / deserialization).
 */
#ifndef OXIDIZE_BYTES_H
#define OXIDIZE_BYTES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Byte buffer. May own its data (malloc'd, freed by oc_bytes_free) or borrow
 * it (wraps external memory; not freed). When `owned` is true, `data` was
 * allocated with malloc and must be released via oc_bytes_free(). */
typedef struct OcBytes {
    uint8_t *data;
    size_t   size;
    size_t   capacity;
    bool     owned;
} OcBytes;

/* Initialize an empty, zero-capacity buffer (no allocation). */
OcError oc_bytes_init(OcBytes *b);

/* Initialize a buffer by wrapping or copying external data. If `copy` is
 * true, allocates and copies; otherwise the caller retains data ownership.
 * Free an existing initialized buffer before reusing its storage here. */
OcError oc_bytes_from_data(OcBytes *b, const uint8_t *data, size_t size, bool copy);

/* Ensure at least `capacity` bytes of capacity. If the buffer borrows external
 * data (owned=false), it is converted to owned by copying. */
OcError oc_bytes_reserve(OcBytes *b, size_t capacity);

/* Append `size` bytes from `data`. Grows if needed. */
OcError oc_bytes_append(OcBytes *b, const void *data, size_t size);

/* Append a single byte. */
OcError oc_bytes_append_u8(OcBytes *b, uint8_t v);

/* Append a little-endian u16. */
OcError oc_bytes_append_u16_le(OcBytes *b, uint16_t v);

/* Append a little-endian u32. */
OcError oc_bytes_append_u32_le(OcBytes *b, uint32_t v);

/* Append a little-endian u64. */
OcError oc_bytes_append_u64_le(OcBytes *b, uint64_t v);

/* Append a NUL-terminated string (including the NUL terminator). */
OcError oc_bytes_append_str(OcBytes *b, const char *str);

/* Read a single byte at `offset`. Returns OC_ERR_INVALID_ARG if out of range. */
OcError oc_bytes_read_u8(const OcBytes *b, size_t offset, uint8_t *out);

/* Read a little-endian u32 at `offset`. Returns OC_ERR_INVALID_ARG if out
 * of range. */
OcError oc_bytes_read_u32_le(const OcBytes *b, size_t offset, uint32_t *out);

/* Read a little-endian u64 at `offset`. Returns OC_ERR_INVALID_ARG if out
 * of range. */
OcError oc_bytes_read_u64_le(const OcBytes *b, size_t offset, uint64_t *out);

/* Reset size to 0, keeping capacity (data pointer unchanged). */
OcError oc_bytes_clear(OcBytes *b);

/* Current size. Returns 0 if `b` is NULL. */
size_t oc_bytes_size(const OcBytes *b);

/* Data pointer. Returns NULL if `b` is NULL. */
const uint8_t *oc_bytes_data(const OcBytes *b);

/* Free owned data. Safe on NULL, borrowed, or already-freed buffers. */
void oc_bytes_free(OcBytes *b);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_BYTES_H */
