/* bytes.h — byte-level read primitives (little-endian). */
#ifndef OXIDIZE_UTIL_BYTES_H
#define OXIDIZE_UTIL_BYTES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bounds-checked little-endian reads. Return 0 if `offset` (plus the type's
 * size) exceeds `len`. */
uint8_t  oc_read_u8 (const uint8_t *buf, size_t len, size_t offset);
int8_t   oc_read_i8 (const uint8_t *buf, size_t len, size_t offset);
uint16_t oc_read_u16(const uint8_t *buf, size_t len, size_t offset);
int16_t  oc_read_i16(const uint8_t *buf, size_t len, size_t offset);
uint32_t oc_read_u32(const uint8_t *buf, size_t len, size_t offset);
int32_t  oc_read_i32(const uint8_t *buf, size_t len, size_t offset);
uint64_t oc_read_u64(const uint8_t *buf, size_t len, size_t offset);
int64_t  oc_read_i64(const uint8_t *buf, size_t len, size_t offset);
float    oc_read_f32(const uint8_t *buf, size_t len, size_t offset);
double   oc_read_f64(const uint8_t *buf, size_t len, size_t offset);

/* Little-endian writes into `buf` at `offset`. Returns true on success, false
 * if out of bounds. */
bool oc_write_u16(uint8_t *buf, size_t len, size_t offset, uint16_t v);
bool oc_write_u32(uint8_t *buf, size_t len, size_t offset, uint32_t v);
bool oc_write_u64(uint8_t *buf, size_t len, size_t offset, uint64_t v);

/* Volatile byte read for page prefaulting / NUMA warm-up. Returns 0 if
 * `offset >= len`. Mirrors oxidize-core::util::bytes::read_volatile_byte. */
uint8_t oc_read_volatile_byte(const uint8_t *buf, size_t len, size_t offset);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_UTIL_BYTES_H */
