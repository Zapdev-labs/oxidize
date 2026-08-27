/* bytes.c — byte-level read primitives (little-endian).
 *
 * Port of oxidize-core/src/util/bytes.rs (read_le_i16, read_volatile_byte, etc.)
 * to C. All reads are bounds-checked against `len`; out-of-range reads return
 * 0 and do not crash (no UB).
 */
#include "oxidize/util/bytes.h"

#include <string.h>

uint8_t oc_read_u8(const uint8_t *buf, size_t len, size_t offset)
{
    if (!buf || offset >= len) return 0;
    return buf[offset];
}

int8_t oc_read_i8(const uint8_t *buf, size_t len, size_t offset)
{
    return (int8_t)oc_read_u8(buf, len, offset);
}

uint16_t oc_read_u16(const uint8_t *buf, size_t len, size_t offset)
{
    if (!buf || offset + 2 > len || offset + 2 < offset) return 0;
    return (uint16_t)buf[offset]
         | ((uint16_t)buf[offset + 1] << 8);
}

int16_t oc_read_i16(const uint8_t *buf, size_t len, size_t offset)
{
    return (int16_t)oc_read_u16(buf, len, offset);
}

uint32_t oc_read_u32(const uint8_t *buf, size_t len, size_t offset)
{
    if (!buf || offset + 4 > len || offset + 4 < offset) return 0;
    return (uint32_t)buf[offset]
         | ((uint32_t)buf[offset + 1] << 8)
         | ((uint32_t)buf[offset + 2] << 16)
         | ((uint32_t)buf[offset + 3] << 24);
}

int32_t oc_read_i32(const uint8_t *buf, size_t len, size_t offset)
{
    return (int32_t)oc_read_u32(buf, len, offset);
}

uint64_t oc_read_u64(const uint8_t *buf, size_t len, size_t offset)
{
    if (!buf || offset + 8 > len || offset + 8 < offset) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= (uint64_t)buf[offset + i] << (8 * i);
    }
    return v;
}

int64_t oc_read_i64(const uint8_t *buf, size_t len, size_t offset)
{
    return (int64_t)oc_read_u64(buf, len, offset);
}

float oc_read_f32(const uint8_t *buf, size_t len, size_t offset)
{
    uint32_t u = oc_read_u32(buf, len, offset);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

double oc_read_f64(const uint8_t *buf, size_t len, size_t offset)
{
    uint64_t u = oc_read_u64(buf, len, offset);
    double d;
    memcpy(&d, &u, sizeof(d));
    return d;
}

bool oc_write_u16(uint8_t *buf, size_t len, size_t offset, uint16_t v)
{
    if (!buf || offset + 2 > len || offset + 2 < offset) return false;
    buf[offset]     = (uint8_t)(v & 0xff);
    buf[offset + 1] = (uint8_t)((v >> 8) & 0xff);
    return true;
}

bool oc_write_u32(uint8_t *buf, size_t len, size_t offset, uint32_t v)
{
    if (!buf || offset + 4 > len || offset + 4 < offset) return false;
    buf[offset]     = (uint8_t)(v & 0xff);
    buf[offset + 1] = (uint8_t)((v >> 8) & 0xff);
    buf[offset + 2] = (uint8_t)((v >> 16) & 0xff);
    buf[offset + 3] = (uint8_t)((v >> 24) & 0xff);
    return true;
}

bool oc_write_u64(uint8_t *buf, size_t len, size_t offset, uint64_t v)
{
    if (!buf || offset + 8 > len || offset + 8 < offset) return false;
    for (int i = 0; i < 8; i++) {
        buf[offset + i] = (uint8_t)((v >> (8 * i)) & 0xff);
    }
    return true;
}

uint8_t oc_read_volatile_byte(const uint8_t *buf, size_t len, size_t offset)
{
    if (!buf || offset >= len) return 0;
    /* Volatile read to force the page to be faulted in (NUMA warm-up /
     * prefault). Mirrors oxidize-core::util::bytes::read_volatile_byte. */
    return *(const volatile uint8_t *)(buf + offset);
}
