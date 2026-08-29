#include "oxidize/bytes.h"

#include <stdlib.h>
#include <string.h>

OcError oc_bytes_init(OcBytes *b)
{
    if (!b) return OC_ERR_INVALID_ARG;
    b->data     = NULL;
    b->size     = 0;
    b->capacity = 0;
    b->owned    = false;
    return OC_OK;
}

OcError oc_bytes_from_data(OcBytes *b, const uint8_t *data, size_t size, bool copy)
{
    if (!b) return OC_ERR_INVALID_ARG;
    oc_bytes_init(b);
    if (size > 0 && !data) return OC_ERR_INVALID_ARG;

    if (copy) {
        if (size == 0) {
            b->data     = NULL;
            b->size     = 0;
            b->capacity = 0;
            b->owned    = false;
            return OC_OK;
        }
        uint8_t *buf = (uint8_t *)malloc(size);
        if (!buf) return OC_ERR_OOM;
        memcpy(buf, data, size);
        b->data     = buf;
        b->size     = size;
        b->capacity = size;
        b->owned    = true;
    } else {
        /* Borrow the pointer; caller retains ownership. */
        b->data     = (uint8_t *)data;
        b->size     = size;
        b->capacity = size;
        b->owned    = false;
    }
    return OC_OK;
}

OcError oc_bytes_reserve(OcBytes *b, size_t capacity)
{
    if (!b) return OC_ERR_INVALID_ARG;
    if (capacity <= b->capacity) return OC_OK;

    if (!b->owned) {
        /* Convert borrow to owned by copying existing data. */
        size_t sz = b->size;
        uint8_t *buf = (uint8_t *)malloc(capacity);
        if (!buf) return OC_ERR_OOM;
        if (sz > 0 && b->data) {
            memcpy(buf, b->data, sz);
        }
        b->data     = buf;
        b->capacity = capacity;
        b->owned    = true;
        return OC_OK;
    }

    uint8_t *nd = (uint8_t *)realloc(b->data, capacity);
    if (!nd) return OC_ERR_OOM;
    b->data     = nd;
    b->capacity = capacity;
    return OC_OK;
}

OcError oc_bytes_append(OcBytes *b, const void *data, size_t size)
{
    if (!b) return OC_ERR_INVALID_ARG;
    if (size == 0) return OC_OK;
    if (!data) return OC_ERR_INVALID_ARG;

    if (b->size + size > b->capacity) {
        size_t need = b->size + size;
        size_t new_cap = b->capacity ? b->capacity : 16;
        while (new_cap < need) {
            if (new_cap > (SIZE_MAX / 2)) { new_cap = need; break; }
            new_cap *= 2;
        }
        OcError e = oc_bytes_reserve(b, new_cap);
        if (e != OC_OK) return e;
    }
    memcpy(b->data + b->size, data, size);
    b->size += size;
    return OC_OK;
}

OcError oc_bytes_append_u8(OcBytes *b, uint8_t v)
{
    return oc_bytes_append(b, &v, 1);
}

OcError oc_bytes_append_u16_le(OcBytes *b, uint16_t v)
{
    uint8_t buf[2];
    buf[0] = (uint8_t)(v & 0xFF);
    buf[1] = (uint8_t)((v >> 8) & 0xFF);
    return oc_bytes_append(b, buf, 2);
}

OcError oc_bytes_append_u32_le(OcBytes *b, uint32_t v)
{
    uint8_t buf[4];
    buf[0] = (uint8_t)(v & 0xFF);
    buf[1] = (uint8_t)((v >> 8) & 0xFF);
    buf[2] = (uint8_t)((v >> 16) & 0xFF);
    buf[3] = (uint8_t)((v >> 24) & 0xFF);
    return oc_bytes_append(b, buf, 4);
}

OcError oc_bytes_append_u64_le(OcBytes *b, uint64_t v)
{
    uint8_t buf[8];
    for (int i = 0; i < 8; i++) {
        buf[i] = (uint8_t)((v >> (i * 8)) & 0xFF);
    }
    return oc_bytes_append(b, buf, 8);
}

OcError oc_bytes_append_str(OcBytes *b, const char *str)
{
    if (!b) return OC_ERR_INVALID_ARG;
    if (!str) return OC_ERR_INVALID_ARG;
    /* Append including the NUL terminator. */
    size_t len = strlen(str) + 1;
    return oc_bytes_append(b, str, len);
}

OcError oc_bytes_read_u8(const OcBytes *b, size_t offset, uint8_t *out)
{
    if (!b || !out) return OC_ERR_INVALID_ARG;
    if (offset >= b->size) return OC_ERR_INVALID_ARG;
    *out = b->data[offset];
    return OC_OK;
}

OcError oc_bytes_read_u32_le(const OcBytes *b, size_t offset, uint32_t *out)
{
    if (!b || !out) return OC_ERR_INVALID_ARG;
    if (offset + 4 > b->size) return OC_ERR_INVALID_ARG;
    *out = (uint32_t)b->data[offset]
         | ((uint32_t)b->data[offset + 1] << 8)
         | ((uint32_t)b->data[offset + 2] << 16)
         | ((uint32_t)b->data[offset + 3] << 24);
    return OC_OK;
}

OcError oc_bytes_read_u64_le(const OcBytes *b, size_t offset, uint64_t *out)
{
    if (!b || !out) return OC_ERR_INVALID_ARG;
    if (offset + 8 > b->size) return OC_ERR_INVALID_ARG;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= ((uint64_t)b->data[offset + i]) << (i * 8);
    }
    *out = v;
    return OC_OK;
}

OcError oc_bytes_clear(OcBytes *b)
{
    if (!b) return OC_ERR_INVALID_ARG;
    b->size = 0;
    return OC_OK;
}

size_t oc_bytes_size(const OcBytes *b)
{
    return b ? b->size : 0;
}

const uint8_t *oc_bytes_data(const OcBytes *b)
{
    return b ? b->data : NULL;
}

void oc_bytes_free(OcBytes *b)
{
    if (!b) return;
    if (b->owned && b->data) {
        free(b->data);
    }
    b->data     = NULL;
    b->size     = 0;
    b->capacity = 0;
    b->owned    = false;
}
