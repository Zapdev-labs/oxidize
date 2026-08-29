/* gguf.c — GGUF v3/v2 binary format parser. logic 1:1 so bit-exact tensor inventory parity holds. Layout (little-endian throughout): */
#include "oxidize/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/log.h"
#include "oxidize/model.h"
#include "oxidize/util/bytes.h"
#include "oxidize/util/file.h"
#include "oxidize/util/mmap.h"
#include "oxidize/util/string.h"

typedef struct {
    const uint8_t *bytes;
    size_t        len;
    size_t        cursor;
} ByteReader;

static void reader_init(ByteReader *r, const uint8_t *bytes, size_t len)
{
    r->bytes  = bytes;
    r->len    = len;
    r->cursor = 0;
}

/* Returns the current cursor position. */
static size_t reader_pos(const ByteReader *r) { return r->cursor; }

/* Bounds-checked "read exactly n bytes". Returns a pointer into the buffer
 * (valid for the buffer's lifetime) on success, or NULL on EOF/overflow. */
static const uint8_t *reader_read_exact(ByteReader *r, size_t n, OcError *err)
{
    size_t end;
    if (__builtin_add_overflow(r->cursor, n, &end) || end > r->len) {
        *err = OC_ERR_FORMAT;
        return NULL;
    }
    const uint8_t *out = r->bytes + r->cursor;
    r->cursor = end;
    *err = OC_OK;
    return out;
}

static OcError reader_read_u8(ByteReader *r, uint8_t *out)
{
    OcError e; const uint8_t *p = reader_read_exact(r, 1, &e);
    if (!p) return e;
    *out = p[0]; return OC_OK;
}
static OcError reader_read_i8(ByteReader *r, int8_t *out)
{
    uint8_t u = 0; OcError e = reader_read_u8(r, &u); if (e != OC_OK) return e;
    *out = (int8_t)u; return OC_OK;
}
static OcError reader_read_u16(ByteReader *r, uint16_t *out)
{
    OcError e; const uint8_t *p = reader_read_exact(r, 2, &e);
    if (!p) return e;
    *out = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return OC_OK;
}
static OcError reader_read_i16(ByteReader *r, int16_t *out)
{
    uint16_t u = 0; OcError e = reader_read_u16(r, &u); if (e != OC_OK) return e;
    *out = (int16_t)u; return OC_OK;
}
static OcError reader_read_u32(ByteReader *r, uint32_t *out)
{
    OcError e; const uint8_t *p = reader_read_exact(r, 4, &e);
    if (!p) return e;
    *out = (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
    return OC_OK;
}
static OcError reader_read_i32(ByteReader *r, int32_t *out)
{
    uint32_t u = 0; OcError e = reader_read_u32(r, &u); if (e != OC_OK) return e;
    *out = (int32_t)u; return OC_OK;
}
static OcError reader_read_u64(ByteReader *r, uint64_t *out)
{
    OcError e; const uint8_t *p = reader_read_exact(r, 8, &e);
    if (!p) return e;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    *out = v; return OC_OK;
}
static OcError reader_read_i64(ByteReader *r, int64_t *out)
{
    uint64_t u = 0; OcError e = reader_read_u64(r, &u); if (e != OC_OK) return e;
    *out = (int64_t)u; return OC_OK;
}
static OcError reader_read_f32(ByteReader *r, float *out)
{
    uint32_t u = 0; OcError e = reader_read_u32(r, &u); if (e != OC_OK) return e;
    memcpy(out, &u, sizeof(*out));
    return OC_OK;
}
static OcError reader_read_f64(ByteReader *r, double *out)
{
    uint64_t u = 0; OcError e = reader_read_u64(r, &u); if (e != OC_OK) return e;
    memcpy(out, &u, sizeof(*out));
    return OC_OK;
}
static OcError reader_read_bool(ByteReader *r, bool *out)
{
    uint8_t u = 0; OcError e = reader_read_u8(r, &u); if (e != OC_OK) return e;
    *out = (u != 0); return OC_OK;
}

/* Read a GGUF string (length-prefixed u64, no NUL terminator on disk). */
static OcError reader_read_string(ByteReader *r, OcArena *arena,
                                  char **out_str, size_t *out_len)
{
    uint64_t len64 = 0;
    OcError e = reader_read_u64(r, &len64);
    if (e != OC_OK) return e;

    /* Guard against absurd lengths that would overflow size_t or OOM the box.
     * The remaining buffer is a hard upper bound. */
    if (len64 > r->len) return OC_ERR_FORMAT;
    size_t len = (size_t)len64;

    OcError e2;
    const uint8_t *p = reader_read_exact(r, len, &e2);
    if (!p) return e2;

    char *dst = oc_arena_dup_n(arena, (const char *)p, len);
    if (!dst) return OC_ERR_OOM;

    *out_str = dst;
    if (out_len) *out_len = len;
    return OC_OK;
}

static OcError read_value_of_type(ByteReader *r, OcArena *arena,
                                  OcGgufMetadataType type,
                                  OcGgufMetadataValue *out);

static OcError read_array(ByteReader *r, OcArena *arena,
                          OcGgufMetadataArray *out)
{
    uint32_t et_raw = 0;
    OcError e = reader_read_u32(r, &et_raw);
    if (e != OC_OK) return e;
    OcGgufMetadataType elem_type = oc_gguf_metadata_type_from_u32(et_raw);
    if (elem_type == OC_GGUF_MT_UNKNOWN) return OC_ERR_FORMAT;
    /* Reject nested ARRAY (ARRAY-of-ARRAY) — non-spec: the GGUF format only */
    if (elem_type == OC_GGUF_MT_ARRAY) {
        oc_log(OC_LOG_ERROR, "gguf: nested ARRAY metadata type rejected "
                "(non-spec, element_type=%u)", et_raw);
        return OC_ERR_FORMAT;
    }

    uint64_t len64 = 0;
    e = reader_read_u64(r, &len64);
    if (e != OC_OK) return e;
    if (len64 > r->len) return OC_ERR_FORMAT;   /* hard upper bound */
    size_t len = (size_t)len64;

    /* Allocate the values array from the arena. Each entry is a
     * OcGgufMetadataValue (fixed size). */
    OcGgufMetadataValue *values = (OcGgufMetadataValue *)oc_arena_alloc(
        arena, (len ? len : 1) * sizeof(OcGgufMetadataValue), 16);
    if (!values) return OC_ERR_OOM;

    for (size_t i = 0; i < len; i++) {
        e = read_value_of_type(r, arena, elem_type, &values[i]);
        if (e != OC_OK) return e;
    }

    out->elem_type = elem_type;
    out->len       = len;
    out->values    = values;
    return OC_OK;
}

static OcError read_value_of_type(ByteReader *r, OcArena *arena,
                                  OcGgufMetadataType type,
                                  OcGgufMetadataValue *out)
{
    memset(out, 0, sizeof(*out));
    out->type = type;
    switch (type) {
    case OC_GGUF_MT_UINT8:   return reader_read_u8 (r, &out->v.u8);
    case OC_GGUF_MT_INT8:    return reader_read_i8 (r, &out->v.i8);
    case OC_GGUF_MT_UINT16:  return reader_read_u16(r, &out->v.u16);
    case OC_GGUF_MT_INT16:   return reader_read_i16(r, &out->v.i16);
    case OC_GGUF_MT_UINT32:  return reader_read_u32(r, &out->v.u32);
    case OC_GGUF_MT_INT32:   return reader_read_i32(r, &out->v.i32);
    case OC_GGUF_MT_UINT64:  return reader_read_u64(r, &out->v.u64);
    case OC_GGUF_MT_INT64:   return reader_read_i64(r, &out->v.i64);
    case OC_GGUF_MT_FLOAT32: return reader_read_f32(r, &out->v.f32);
    case OC_GGUF_MT_FLOAT64: return reader_read_f64(r, &out->v.f64);
    case OC_GGUF_MT_BOOL:    return reader_read_bool(r, &out->v.b);
    case OC_GGUF_MT_STRING: {
        size_t len = 0;
        OcError e = reader_read_string(r, arena, &out->v.str.data, &len);
        if (e != OC_OK) return e;
        out->v.str.len = len;
        return OC_OK;
    }
    case OC_GGUF_MT_ARRAY:
        return read_array(r, arena, &out->v.arr);
    default:
        return OC_ERR_FORMAT;
    }
}


static bool is_power_of_two_u64(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

static uint64_t align_up_u64(uint64_t v, uint64_t alignment, bool *overflow)
{
    uint64_t mask = alignment - 1;
    uint64_t sum;
    if (__builtin_add_overflow(v, mask, &sum)) { *overflow = true; return 0; }
    *overflow = false;
    return sum & ~mask;
}

/* Resolve `general.alignment` from a metadata value, mirroring Rust
 * `alignment_from_metadata`. Accepts any unsigned type or a positive signed
 * type; rejects zero, negatives, and non-numeric types. */
static OcError alignment_from_value(const OcGgufMetadataValue *v, uint64_t *out)
{
    switch (v->type) {
    case OC_GGUF_MT_UINT8:  *out = (uint64_t)v->v.u8;  return OC_OK;
    case OC_GGUF_MT_UINT16: *out = (uint64_t)v->v.u16; return OC_OK;
    case OC_GGUF_MT_UINT32: *out = (uint64_t)v->v.u32; return OC_OK;
    case OC_GGUF_MT_UINT64: *out = v->v.u64;          return OC_OK;
    case OC_GGUF_MT_INT8:   if (v->v.i8  > 0) { *out = (uint64_t)v->v.i8;  return OC_OK; } break;
    case OC_GGUF_MT_INT16:  if (v->v.i16 > 0) { *out = (uint64_t)v->v.i16; return OC_OK; } break;
    case OC_GGUF_MT_INT32:  if (v->v.i32 > 0) { *out = (uint64_t)v->v.i32; return OC_OK; } break;
    case OC_GGUF_MT_INT64:  if (v->v.i64 > 0) { *out = (uint64_t)v->v.i64; return OC_OK; } break;
    default: break;
    }
    return OC_ERR_FORMAT;
}


OcGgufMetadataType oc_gguf_metadata_type_from_u32(uint32_t raw)
{
    if (raw < (uint32_t)OC_GGUF_MT__COUNT) return (OcGgufMetadataType)raw;
    return OC_GGUF_MT_UNKNOWN;
}

const char *oc_gguf_metadata_type_name(OcGgufMetadataType t)
{
    switch (t) {
    case OC_GGUF_MT_UINT8:   return "U8";
    case OC_GGUF_MT_INT8:    return "I8";
    case OC_GGUF_MT_UINT16:  return "U16";
    case OC_GGUF_MT_INT16:   return "I16";
    case OC_GGUF_MT_UINT32:  return "U32";
    case OC_GGUF_MT_INT32:   return "I32";
    case OC_GGUF_MT_UINT64:  return "U64";
    case OC_GGUF_MT_INT64:   return "I64";
    case OC_GGUF_MT_FLOAT32: return "F32";
    case OC_GGUF_MT_FLOAT64: return "F64";
    case OC_GGUF_MT_BOOL:    return "BOOL";
    case OC_GGUF_MT_STRING:  return "STRING";
    case OC_GGUF_MT_ARRAY:   return "ARRAY";
    default:                 return "UNKNOWN";
    }
}


OcError oc_gguf_parse(const uint8_t *buf, size_t len, OcGgufFile *out)
{
    if (!buf || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    ByteReader r;
    reader_init(&r, buf, len);

    /* Magic. */
    uint32_t magic = 0;
    OcError e = reader_read_u32(&r, &magic);
    if (e != OC_OK) {
        oc_log(OC_LOG_ERROR, "gguf: truncated header (magic): %s", oc_error_msg(e));
        return e;
    }
    if (magic != OC_GGUF_MAGIC) {
        oc_log(OC_LOG_ERROR, "gguf: bad magic 0x%08x", magic);
        return OC_ERR_FORMAT;
    }

    /* Version. */
    uint32_t version = 0;
    e = reader_read_u32(&r, &version);
    if (e != OC_OK) {
        oc_log(OC_LOG_ERROR, "gguf: truncated header (version): %s", oc_error_msg(e));
        return e;
    }
    if (version != OC_GGUF_VERSION_2 && version != OC_GGUF_VERSION_3) {
        oc_log(OC_LOG_ERROR, "gguf: unsupported version %u", version);
        return OC_ERR_FORMAT;
    }

    /* tensor_count + metadata_kv_count. */
    uint64_t tensor_count = 0, kv_count = 0;
    e = reader_read_u64(&r, &tensor_count);
    if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated tensor_count"); return e; }
    e = reader_read_u64(&r, &kv_count);
    if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated metadata_kv_count"); return e; }

    /* Cap absurd counts defensively (don't trust untrusted files to be sane).
     * Each KV/tensor entry must consume at least a few bytes, so any count
     * larger than `len` is necessarily truncated/corrupt. */
    if (kv_count > len || tensor_count > len) {
        oc_log(OC_LOG_ERROR, "gguf: header counts exceed file size");
        return OC_ERR_FORMAT;
    }

    /* Allocate the arena up-front. Most allocations here are small (keys,
     * values, names); the arena grows on demand. */
    OcArena *arena = oc_arena_new(0);
    if (!arena) return OC_ERR_OOM;

    /* Metadata KV array. */
    OcGgufMetadataKV *kv = NULL;
    if (kv_count > 0) {
        kv = (OcGgufMetadataKV *)oc_arena_alloc(arena,
            (size_t)kv_count * sizeof(OcGgufMetadataKV), 16);
        if (!kv) { oc_arena_free(arena); return OC_ERR_OOM; }
    }
    for (uint64_t i = 0; i < kv_count; i++) {
        char *key = NULL;
        e = reader_read_string(&r, arena, &key, NULL);
        if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated metadata key %llu", (unsigned long long)i); oc_arena_free(arena); return e; }
        uint32_t vtype_raw = 0;
        e = reader_read_u32(&r, &vtype_raw);
        if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated metadata value_type"); oc_arena_free(arena); return e; }
        OcGgufMetadataType vtype = oc_gguf_metadata_type_from_u32(vtype_raw);
        if (vtype == OC_GGUF_MT_UNKNOWN) { oc_log(OC_LOG_ERROR, "gguf: unknown metadata type %u", vtype_raw); oc_arena_free(arena); return OC_ERR_FORMAT; }
        e = read_value_of_type(&r, arena, vtype, &kv[i].value);
        if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated metadata value"); oc_arena_free(arena); return e; }
        kv[i].key = key;
    }

    /* Tensor info table. */
    OcGgufTensorInfo *tinfos = NULL;
    if (tensor_count > 0) {
        tinfos = (OcGgufTensorInfo *)oc_arena_alloc(arena,
            (size_t)tensor_count * sizeof(OcGgufTensorInfo), 16);
        if (!tinfos) { oc_arena_free(arena); return OC_ERR_OOM; }
        memset(tinfos, 0, (size_t)tensor_count * sizeof(*tinfos));
    }
    for (uint64_t i = 0; i < tensor_count; i++) {
        OcGgufTensorInfo *t = &tinfos[i];
        e = reader_read_string(&r, arena, (char **)&t->name, NULL);
        if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated tensor name %llu", (unsigned long long)i); oc_arena_free(arena); return e; }
        uint32_t n_dims = 0;
        e = reader_read_u32(&r, &n_dims);
        if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated tensor n_dims"); oc_arena_free(arena); return e; }
        if (n_dims > OC_GGUF_MAX_DIMS) {
            oc_log(OC_LOG_ERROR, "gguf: tensor %s has %u dims (max %d)", t->name, n_dims, OC_GGUF_MAX_DIMS);
            oc_arena_free(arena); return OC_ERR_FORMAT;
        }
        t->n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; d++) {
            e = reader_read_u64(&r, &t->dims[d]);
            if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated tensor dim"); oc_arena_free(arena); return e; }
        }
        e = reader_read_u32(&r, &t->ggml_type);
        if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated tensor ggml_type"); oc_arena_free(arena); return e; }
        e = reader_read_u64(&r, &t->relative_offset);
        if (e != OC_OK) { oc_log(OC_LOG_ERROR, "gguf: truncated tensor offset"); oc_arena_free(arena); return e; }
    }

    /* Alignment: optional `general.alignment` metadata key. */
    uint64_t alignment = OC_GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < kv_count; i++) {
        if (strcmp(kv[i].key, "general.alignment") == 0) {
            OcError ae = alignment_from_value(&kv[i].value, &alignment);
            if (ae != OC_OK) {
                oc_log(OC_LOG_ERROR, "gguf: invalid general.alignment value");
                oc_arena_free(arena);
                return OC_ERR_FORMAT;
            }
            break;
        }
    }
    if (!is_power_of_two_u64(alignment)) {
        oc_log(OC_LOG_ERROR, "gguf: alignment %llu is not a power of two",
                (unsigned long long)alignment);
        oc_arena_free(arena);
        return OC_ERR_FORMAT;
    }

    bool overflow = false;
    uint64_t data_section_start = align_up_u64((uint64_t)reader_pos(&r), alignment, &overflow);
    if (overflow || data_section_start > (uint64_t)len) {
        oc_log(OC_LOG_ERROR, "gguf: data_section_start %llu exceeds file size %zu",
                (unsigned long long)data_section_start, len);
        oc_arena_free(arena);
        return OC_ERR_FORMAT;
    }

    /* Resolve absolute offsets, sanity-check each is in-bounds. */
    for (uint64_t i = 0; i < tensor_count; i++) {
        OcGgufTensorInfo *t = &tinfos[i];
        if (__builtin_add_overflow(data_section_start, t->relative_offset, &t->absolute_offset)) {
            oc_log(OC_LOG_ERROR, "gguf: tensor %s offset overflow", t->name);
            oc_arena_free(arena);
            return OC_ERR_FORMAT;
        }
        if (t->absolute_offset > (uint64_t)len) {
            oc_log(OC_LOG_ERROR, "gguf: tensor %s absolute_offset %llu > file size %zu",
                    t->name, (unsigned long long)t->absolute_offset, len);
            oc_arena_free(arena);
            return OC_ERR_FORMAT;
        }
    }

    out->magic             = magic;
    out->version           = version;
    out->tensor_count      = tensor_count;
    out->metadata_kv_count  = kv_count;
    out->metadata          = kv;
    out->tensors           = tinfos;
    out->alignment         = alignment;
    out->data_section_start = data_section_start;
    out->arena             = arena;
    out->backing_buf        = NULL;
    out->backing_len       = 0;
    return OC_OK;
}

OcError oc_gguf_open(const char *path, OcGgufFile *out)
{
    if (!path || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    uint8_t *buf = NULL;
    size_t   len = 0;
    OcError  e   = oc_file_read_all(path, &buf, &len);
    if (e != OC_OK) {
        oc_log(OC_LOG_ERROR, "gguf: failed to open %s: %s", path, oc_error_msg(e));
        return e;
    }

    e = oc_gguf_parse(buf, len, out);
    if (e != OC_OK) {
        free(buf);
        /* Ensure no dangling arena. oc_gguf_parse failed before populating
         * out->arena on its error paths (it frees the arena itself). */
        memset(out, 0, sizeof(*out));
        return e;
    }

    /* On success, transfer ownership of the file bytes to the OcGgufFile. */
    out->backing_buf = buf;
    out->backing_len = len;
    return OC_OK;
}

void oc_gguf_free(OcGgufFile *out)
{
    if (!out) return;
    /* arena owns all parser allocations (kv array, kv keys, kv values,
     * tensors array, tensor names). One free. */
    if (out->arena) oc_arena_free(out->arena);
    if (out->backing_buf) free(out->backing_buf);
    memset(out, 0, sizeof(*out));
}


const OcGgufMetadataValue *oc_gguf_metadata_get(const OcGgufFile *f,
                                                const char *key)
{
    if (!f || !key) return NULL;
    for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
        if (strcmp(f->metadata[i].key, key) == 0) {
            return &f->metadata[i].value;
        }
    }
    return NULL;
}

/* Helper: extract a numeric value as a uint64 with range check. Returns true
 * on success; false if the type is not numeric or the value overflows `max`. */
static bool value_as_u64(const OcGgufMetadataValue *v, uint64_t max, uint64_t *out)
{
    switch (v->type) {
    case OC_GGUF_MT_UINT8:  *out = (uint64_t)v->v.u8;  return *out <= max;
    case OC_GGUF_MT_UINT16: *out = (uint64_t)v->v.u16; return *out <= max;
    case OC_GGUF_MT_UINT32: *out = (uint64_t)v->v.u32; return *out <= max;
    case OC_GGUF_MT_UINT64: *out = v->v.u64;           return *out <= max;
    case OC_GGUF_MT_INT8:   if (v->v.i8  < 0) return false; *out = (uint64_t)v->v.i8;  return *out <= max;
    case OC_GGUF_MT_INT16:  if (v->v.i16 < 0) return false; *out = (uint64_t)v->v.i16; return *out <= max;
    case OC_GGUF_MT_INT32:  if (v->v.i32 < 0) return false; *out = (uint64_t)v->v.i32; return *out <= max;
    case OC_GGUF_MT_INT64:  if (v->v.i64 < 0) return false; *out = (uint64_t)v->v.i64; return *out <= max;
    case OC_GGUF_MT_BOOL:   *out = v->v.b ? 1 : 0; return *out <= max;
    default: return false;
    }
}

bool oc_gguf_metadata_get_u8(const OcGgufFile *f, const char *key, uint8_t *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v) return false;
    uint64_t u; if (!value_as_u64(v, (uint64_t)UINT8_MAX, &u)) return false;
    *out = (uint8_t)u; return true;
}
bool oc_gguf_metadata_get_u16(const OcGgufFile *f, const char *key, uint16_t *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v) return false;
    uint64_t u; if (!value_as_u64(v, (uint64_t)UINT16_MAX, &u)) return false;
    *out = (uint16_t)u; return true;
}
bool oc_gguf_metadata_get_u32(const OcGgufFile *f, const char *key, uint32_t *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v) return false;
    uint64_t u; if (!value_as_u64(v, (uint64_t)UINT32_MAX, &u)) return false;
    *out = (uint32_t)u; return true;
}
bool oc_gguf_metadata_get_u64(const OcGgufFile *f, const char *key, uint64_t *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v) return false;
    uint64_t u; if (!value_as_u64(v, UINT64_MAX, &u)) return false;
    *out = u; return true;
}

bool oc_gguf_metadata_get_i32(const OcGgufFile *f, const char *key, int32_t *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v) return false;
    switch (v->type) {
    case OC_GGUF_MT_INT8:   *out = (int32_t)v->v.i8;  return true;
    case OC_GGUF_MT_INT16:  *out = (int32_t)v->v.i16; return true;
    case OC_GGUF_MT_INT32:  *out = v->v.i32;          return true;
    case OC_GGUF_MT_INT64:  if (v->v.i64 < INT32_MIN || v->v.i64 > INT32_MAX) return false; *out = (int32_t)v->v.i64; return true;
    case OC_GGUF_MT_UINT8:  *out = (int32_t)v->v.u8;  return true;
    case OC_GGUF_MT_UINT16: *out = (int32_t)v->v.u16; return true;
    case OC_GGUF_MT_UINT32: if (v->v.u32 > (uint32_t)INT32_MAX) return false; *out = (int32_t)v->v.u32; return true;
    case OC_GGUF_MT_UINT64: if (v->v.u64 > (uint64_t)INT32_MAX) return false; *out = (int32_t)v->v.u64; return true;
    case OC_GGUF_MT_BOOL:   *out = v->v.b ? 1 : 0; return true;
    default: return false;
    }
}

bool oc_gguf_metadata_get_f32(const OcGgufFile *f, const char *key, float *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v) return false;
    switch (v->type) {
    case OC_GGUF_MT_FLOAT32: *out = v->v.f32; return true;
    case OC_GGUF_MT_FLOAT64: *out = (float)v->v.f64; return true;
    default: {
        int32_t i;
        if (oc_gguf_metadata_get_i32(f, key, &i)) { *out = (float)i; return true; }
        return false;
    }
    }
}

bool oc_gguf_metadata_get_f64(const OcGgufFile *f, const char *key, double *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v) return false;
    switch (v->type) {
    case OC_GGUF_MT_FLOAT32: *out = (double)v->v.f32; return true;
    case OC_GGUF_MT_FLOAT64: *out = v->v.f64; return true;
    default: {
        int32_t i;
        if (oc_gguf_metadata_get_i32(f, key, &i)) { *out = (double)i; return true; }
        return false;
    }
    }
}

bool oc_gguf_metadata_get_bool(const OcGgufFile *f, const char *key, bool *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v) return false;
    if (v->type == OC_GGUF_MT_BOOL) { *out = v->v.b; return true; }
    /* Allow numeric → bool coercion (0=false, nonzero=true). */
    uint64_t u;
    if (value_as_u64(v, UINT64_MAX, &u)) { *out = (u != 0); return true; }
    return false;
}

bool oc_gguf_metadata_get_str(const OcGgufFile *f, const char *key,
                              const char **out_data, size_t *out_len)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(f, key);
    if (!v || v->type != OC_GGUF_MT_STRING) return false;
    if (out_data) *out_data = v->v.str.data;
    if (out_len)  *out_len  = v->v.str.len;
    return true;
}

const OcGgufTensorInfo *oc_gguf_tensor_get(const OcGgufFile *f, const char *name)
{
    if (!f || !name) return NULL;
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            return &f->tensors[i];
        }
    }
    return NULL;
}


void oc_gguf_dump(const OcGgufFile *f)
{
    if (!f) { fprintf(stderr, "(null OcGgufFile)\n"); return; }
    fprintf(stderr, "GGUF v%u  magic=0x%08x\n", f->version, f->magic);
    fprintf(stderr, "  tensor_count     = %llu\n", (unsigned long long)f->tensor_count);
    fprintf(stderr, "  metadata_kv_count = %llu\n", (unsigned long long)f->metadata_kv_count);
    fprintf(stderr, "  alignment        = %llu\n", (unsigned long long)f->alignment);
    fprintf(stderr, "  data_section_start = %llu\n", (unsigned long long)f->data_section_start);
    for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
        const OcGgufMetadataKV *kv = &f->metadata[i];
        fprintf(stderr, "  [%llu] %s = (%s)\n",
                (unsigned long long)i, kv->key,
                oc_gguf_metadata_type_name(kv->value.type));
    }
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        const OcGgufTensorInfo *t = &f->tensors[i];
        fprintf(stderr, "  tensor[%llu] %-40s dtype=%u n_dims=%u off=%llu\n",
                (unsigned long long)i, t->name ? t->name : "(null)",
                t->ggml_type, t->n_dims,
                (unsigned long long)t->absolute_offset);
    }
}

/* ─── Architecture detection (VAL-FOUND-012) ──────────────────────────────── */
OcModelArchitecture oc_gguf_arch_from_file(const OcGgufFile *f)
{
    if (!f) return OC_ARCH_UNKNOWN;

    /* Path 1: general.architecture STRING. */
    const char *arch_str = NULL;
    size_t      arch_len = 0;
    if (oc_gguf_metadata_get_str(f, "general.architecture", &arch_str, &arch_len)
        && arch_str && arch_len > 0) {
        OcModelArchitecture a = oc_model_arch_from_str(arch_str);
        if (a != OC_ARCH_UNKNOWN) return a;
    }

    /* Path 2: scan metadata keys for an `<arch>.*` namespace. */
    for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
        const char *key = f->metadata[i].key;
        if (!key) continue;
        const char *dot = strchr(key, '.');
        if (!dot || dot == key) continue;
        /* Extract the namespace (substring before the first '.'). */
        size_t ns_len = (size_t)(dot - key);
        if (ns_len >= 64) continue;   /* skip absurdly long namespaces */
        char ns[64];
        memcpy(ns, key, ns_len);
        ns[ns_len] = '\0';
        OcModelArchitecture a = oc_model_arch_from_str(ns);
        if (a != OC_ARCH_UNKNOWN) return a;
    }

    return OC_ARCH_UNKNOWN;
}


/* Check whether `filename` matches the split-GGUF pattern
 * `<base>-NNNNN-of-MMMMM.gguf` and, if so, populate `out_total` with the
 * parsed MMMMM value. Returns true on match. */
static bool parse_split_pattern(const char *filename, uint64_t *out_total)
{
    if (!filename || !*filename) return false;

    /* Must end in ".gguf". */
    const char *gguf_suffix = ".gguf";
    size_t fn_len = strlen(filename);
    size_t suf_len = strlen(gguf_suffix);
    if (fn_len <= suf_len) return false;
    if (strcmp(filename + fn_len - suf_len, gguf_suffix) != 0) return false;

    /* Strip the suffix → stem. */
    size_t stem_len = fn_len - suf_len;
    /* Find the last "-of-" in the stem. The "-of-" is 4 chars; we scan
     * the stem for the substring. */
    const char *of_tok = NULL;
    /* i is the index of the 'o' in "-of-" (so filename[i-1]='-'). We scan
     * from the end of the stem backward. */
    if (stem_len < 4) return false;
    for (size_t i = stem_len; i >= 4; i--) {
        /* filename[i-1] is '-' (checked below); need filename[i]='o',
         * filename[i+1]='f', filename[i+2]='-'. All within the stem
         * (i+2 <= stem_len-1, i.e. i <= stem_len-3). */
        if (i + 2 >= stem_len) continue;   /* "-of-" would cross into ".gguf" */
        if (filename[i - 1] == '-'
            && filename[i] == 'o'
            && filename[i + 1] == 'f'
            && filename[i + 2] == '-') {
            of_tok = filename + i - 1;
            break;
        }
    }
    if (!of_tok) return false;

    /* After "-of-" comes MMMMM (within the stem). total_str points at the
     * first digit; it ends at filename + stem_len. */
    const char *total_str = of_tok + 4;   /* skip "-of-" */
    const char *total_end = filename + stem_len;
    if (total_str >= total_end) return false;
    for (const char *p = total_str; p < total_end; p++) {
        if (*p < '0' || *p > '9') return false;
    }
    /* Parse total (must be >= 2 for a real split). */
    uint64_t total = 0;
    for (const char *p = total_str; p < total_end; p++) {
        total = total * 10 + (uint64_t)(*p - '0');
        if (total > 1000000) return false;   /* sanity cap */
    }
    if (total < 2) return false;

    /* Find the shard-number token before "-of-": "-NNNNN-". The shard number
     * must be all digits. */
    if (of_tok == filename) return false;
    const char *prev_dash = NULL;
    /* Index-based backwards scan: never forms a pointer before `filename`. */
    for (size_t k = (size_t)(of_tok - filename); k > 0; k--) {
        if (filename[k - 1] == '-') { prev_dash = filename + k - 1; break; }
    }
    if (!prev_dash) return false;
    /* Shard number is prev_dash+1 .. of_tok. Must be all digits, >= 1. */
    if (prev_dash + 1 >= of_tok) return false;
    for (const char *p = prev_dash + 1; p < of_tok; p++) {
        if (*p < '0' || *p > '9') return false;
    }

    if (out_total) *out_total = total;
    return true;
}

/* Build the i-th shard path: `<dir>/<base>-<i:05>-of-<total:05>.gguf`.
 * Returns a malloc'd string (caller frees) or NULL on OOM. */
static char *build_shard_path(const char *dir, const char *base,
                              uint64_t i, uint64_t total)
{
    /* Output layout: dir + "/" + base + "-" + NNNNN + "-of-" + MMMMM + ".gguf" + NUL.
   The literal parts sum to 1+1+5+4+5+5 = 21 chars. */
    size_t cap = strlen(dir) + 1 + strlen(base) + 21 + 1;   /* +1 for NUL */
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    int n = snprintf(buf, cap, "%s/%s-%05llu-of-%05llu.gguf",
                     dir, base,
                     (unsigned long long)i, (unsigned long long)total);
    if (n < 0 || (size_t)n >= cap) { free(buf); return NULL; }
    return buf;
}

/* Extract `<dir>/<base>` (without the `-NNNNN-of-MMMMM.gguf` suffix) from
 * `path`. Returns a malloc'd "<dir>/<base>" string (caller frees) or NULL on
 * OOM / no match. `out_total` receives MMMMM. */
static char *extract_split_base_and_dir(const char *path, uint64_t *out_total)
{
    if (!path) return NULL;
    /* Find the last '/' to split dir / filename. */
    const char *slash = strrchr(path, '/');
    const char *dir;
    const char *filename;
    size_t dir_len;
    if (slash) {
        dir = path;
        dir_len = (size_t)(slash - path) + 1;   /* include trailing '/' */
        filename = slash + 1;
    } else {
        dir = "";
        dir_len = 0;
        filename = path;
    }

    uint64_t total = 0;
    if (!parse_split_pattern(filename, &total)) return NULL;

    size_t fn_len = strlen(filename);
    const char *gguf_suffix = ".gguf";
    size_t suf_len = strlen(gguf_suffix);
    size_t stem_len = fn_len - suf_len;

    const char *of_tok = NULL;
    for (size_t i = stem_len; i >= 3; i--) {
        if (filename[i - 1] == '-'
            && filename[i] == 'o'
            && filename[i + 1] == 'f'
            && filename[i + 2] == '-') {
            of_tok = filename + i - 1;
            break;
        }
    }
    if (!of_tok) return NULL;

    /* Find the '-' before NNNNN (the shard number). Index-based backwards
     * scan: never forms a pointer before `filename`. */
    const char *prev_dash = NULL;
    for (size_t k = (size_t)(of_tok - filename); k > 0; k--) {
        if (filename[k - 1] == '-') { prev_dash = filename + k - 1; break; }
    }
    if (!prev_dash) return NULL;

    /* base = filename[0 .. prev_dash). */
    size_t base_len = (size_t)(prev_dash - filename);

    /* Compose <dir>/<base>. */
    size_t cap = dir_len + base_len + 1;
    char *base_path = (char *)malloc(cap);
    if (!base_path) return NULL;
    memcpy(base_path, dir, dir_len);
    memcpy(base_path + dir_len, filename, base_len);
    base_path[dir_len + base_len] = '\0';

    if (out_total) *out_total = total;
    return base_path;
}

/* Open + mmap a single shard. */
/* parsed from the mmap'd bytes. */
static OcError open_shard(const char *path, OcGgufShard *shard)
{
    memset(shard, 0, sizeof(*shard));
    OcMmap *m = NULL;
    OcError e = oc_mmap_open_readonly(path, &m);
    if (e != OC_OK) return e;

    /* Parse the mmap'd bytes. oc_gguf_parse dups the bytes it needs into the arena, so we can pass mmap'd bytes directly (the parse doesn't hold */
    OcGgufFile parsed;
    e = oc_gguf_parse(oc_mmap_bytes(m), oc_mmap_len(m), &parsed);
    if (e != OC_OK) {
        oc_mmap_close(m);
        return e;
    }

    shard->mmap  = m;
    shard->bytes = (uint8_t *)oc_mmap_bytes(m);
    shard->len   = oc_mmap_len(m);
    shard->parsed = parsed;
    return OC_OK;
}

/* Close a single shard (free mmap + parsed arena). */
static void close_shard(OcGgufShard *shard)
{
    if (!shard) return;
    if (shard->mmap) {
        oc_mmap_close(shard->mmap);
        shard->mmap = NULL;
    }
    /* parsed.arena is owned by the shard; free it. (parsed.backing_buf is
     * NULL because oc_gguf_parse doesn't take ownership.) */
    if (shard->parsed.arena) {
        oc_arena_free(shard->parsed.arena);
        shard->parsed.arena = NULL;
    }
    memset(shard, 0, sizeof(*shard));
}

OcError oc_gguf_map_open(const char *path, OcGgufMmappedFile *out)
{
    if (!path || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    /* Detect split pattern. */
    uint64_t total = 0;
    char *base_path = extract_split_base_and_dir(path, &total);

    if (base_path) {
        /* Split: split base_path into dir + base. */
        const char *slash = strrchr(base_path, '/');
        const char *dir;
        const char *base;
        char dir_buf[4096];
        if (slash) {
            size_t dir_len = (size_t)(slash - base_path) + 1;
            if (dir_len >= sizeof(dir_buf)) dir_len = sizeof(dir_buf) - 1;
            memcpy(dir_buf, base_path, dir_len);
            dir_buf[dir_len] = '\0';
            dir = dir_buf;
            base = slash + 1;
        } else {
            dir = "./";
            base = base_path;
        }

        /* Check all shards exist. */
        bool all_exist = true;
        for (uint64_t i = 1; i <= total; i++) {
            char *shard_path = build_shard_path(dir, base, i, total);
            if (!shard_path) { all_exist = false; break; }
            if (!oc_file_exists(shard_path)) {
                free(shard_path);
                all_exist = false;
                break;
            }
            free(shard_path);
        }

        if (all_exist && total >= 2) {
            /* Allocate shards array. */
            OcGgufShard *shards = (OcGgufShard *)calloc((size_t)total, sizeof(OcGgufShard));
            if (!shards) { free(base_path); return OC_ERR_OOM; }

            /* Open each shard. */
            for (uint64_t i = 0; i < total; i++) {
                char *shard_path = build_shard_path(dir, base, i + 1, total);
                if (!shard_path) {
                    for (uint64_t j = 0; j < i; j++) close_shard(&shards[j]);
                    free(shards);
                    free(base_path);
                    return OC_ERR_OOM;
                }
                OcError e = open_shard(shard_path, &shards[i]);
                free(shard_path);
                if (e != OC_OK) {
                    uint64_t shard_num = i + 1;
                    oc_log(OC_LOG_ERROR, "gguf: failed to open shard %llu: %s",
                            (unsigned long long)shard_num, oc_error_msg(e));
                    for (uint64_t j = 0; j < i; j++) close_shard(&shards[j]);
                    free(shards);
                    free(base_path);
                    return e;
                }
            }

            /* Build the unified view: take shard 0's parsed file as the
             * base, then append tensors from shards 1..N with shard_index
             * set per-tensor. */
            OcArena *unified_arena = oc_arena_new(0);
            if (!unified_arena) {
                for (uint64_t i = 0; i < total; i++) close_shard(&shards[i]);
                free(shards);
                free(base_path);
                return OC_ERR_OOM;
            }

            /* Total tensor count. */
            uint64_t total_tensors = 0;
            for (uint64_t i = 0; i < total; i++) {
                total_tensors += shards[i].parsed.tensor_count;
            }

            /* Allocate the merged tensor table in the unified arena. */
            OcGgufTensorInfo *merged = NULL;
            if (total_tensors > 0) {
                merged = (OcGgufTensorInfo *)oc_arena_alloc(unified_arena,
                    (size_t)total_tensors * sizeof(OcGgufTensorInfo), 16);
                if (!merged) {
                    oc_arena_free(unified_arena);
                    for (uint64_t i = 0; i < total; i++) close_shard(&shards[i]);
                    free(shards);
                    free(base_path);
                    return OC_ERR_OOM;
                }
                memset(merged, 0, (size_t)total_tensors * sizeof(*merged));
            }

            /* Copy shard 0's tensors (deep-copy names into the unified arena so they outlive the per-shard arena). */
            /* Invariant: if total_tensors == 0 then every shard's */
            uint64_t out_idx = 0;
            for (uint64_t s = 0; s < total; s++) {
                OcGgufFile *src = &shards[s].parsed;
                for (uint64_t i = 0; i < src->tensor_count; i++) {
                    /* total_tensors > 0 here (invariant above), so merged != NULL. */
                    OcGgufTensorInfo *dst = &merged[out_idx++];
                    const OcGgufTensorInfo *src_t = &src->tensors[i];
                    /* Deep-copy the name into the unified arena. */
                    dst->name = oc_arena_dup(unified_arena, src_t->name ? src_t->name : "");
                    if (!dst->name) goto oom_split;
                    dst->n_dims           = src_t->n_dims;
                    memcpy(dst->dims, src_t->dims, sizeof(dst->dims));
                    dst->ggml_type        = src_t->ggml_type;
                    dst->relative_offset  = src_t->relative_offset;
                    /* absolute_offset is per-shard: it's the offset within
                     * shard s's mmap'd bytes. shard_index selects the shard. */
                    dst->absolute_offset  = src_t->absolute_offset;
                    dst->shard_index     = (uint32_t)s;
                }
            }

            /* Unified OcGgufFile: copy shard 0's metadata into the unified
             * arena. The KV array + keys + values must outlive the per-shard
             * arena, so we deep-copy them. */
            OcGgufFile *unified = &out->unified;
            unified->magic              = shards[0].parsed.magic;
            unified->version            = shards[0].parsed.version;
            unified->tensor_count       = total_tensors;
            unified->metadata_kv_count  = shards[0].parsed.metadata_kv_count;
            unified->alignment          = shards[0].parsed.alignment;
            unified->data_section_start = shards[0].parsed.data_section_start;
            unified->arena             = unified_arena;
            /* backing_buf stays NULL: the bytes are mmap-backed and owned by
             * the shards. Putting shard bytes here would make oc_gguf_free()
             * call free() on mmap memory (invalid free). */
            unified->backing_buf        = NULL;
            unified->backing_len       = 0;
            unified->tensors            = merged;

            /* Deep-copy the metadata KV array + keys + values into the
             * unified arena. */
            if (shards[0].parsed.metadata_kv_count > 0) {
                size_t n_kv = (size_t)shards[0].parsed.metadata_kv_count;
                OcGgufMetadataKV *kv = (OcGgufMetadataKV *)oc_arena_alloc(
                    unified_arena, n_kv * sizeof(OcGgufMetadataKV), 16);
                if (!kv) {
                    oc_arena_free(unified_arena);
                    for (uint64_t i = 0; i < total; i++) close_shard(&shards[i]);
                    free(shards);
                    free(base_path);
                    return OC_ERR_OOM;
                }
                for (size_t i = 0; i < n_kv; i++) {
                    kv[i].key = oc_arena_dup(unified_arena, shards[0].parsed.metadata[i].key);
                    if (!kv[i].key) goto oom_split;
                    /* Deep-copy the value (handles strings + arrays). */
                    const OcGgufMetadataValue *sv = &shards[0].parsed.metadata[i].value;
                    OcGgufMetadataValue *dv = &kv[i].value;
                    dv->type = sv->type;
                    switch (sv->type) {
                    case OC_GGUF_MT_STRING:
                        dv->v.str.data = oc_arena_dup_n(unified_arena,
                            sv->v.str.data ? sv->v.str.data : "", sv->v.str.len);
                        dv->v.str.len = sv->v.str.len;
                        break;
                    case OC_GGUF_MT_ARRAY: {
                        /* Copy element values recursively (simple: just copy
                         * the union; for STRING elements we deep-copy). */
                        size_t alen = sv->v.arr.len;
                        dv->v.arr.elem_type = sv->v.arr.elem_type;
                        dv->v.arr.len       = alen;
                        if (alen > 0) {
                            dv->v.arr.values = (OcGgufMetadataValue *)oc_arena_alloc(
                                unified_arena, alen * sizeof(OcGgufMetadataValue), 16);
                            if (!dv->v.arr.values) {
                                oc_arena_free(unified_arena);
                                for (uint64_t k = 0; k < total; k++) close_shard(&shards[k]);
                                free(shards);
                                free(base_path);
                                return OC_ERR_OOM;
                            }
                            for (size_t j = 0; j < alen; j++) {
                                dv->v.arr.values[j].type = sv->v.arr.values[j].type;
                                if (sv->v.arr.elem_type == OC_GGUF_MT_STRING) {
                                    dv->v.arr.values[j].v.str.data = oc_arena_dup_n(
                                        unified_arena,
                                        sv->v.arr.values[j].v.str.data ? sv->v.arr.values[j].v.str.data : "",
                                        sv->v.arr.values[j].v.str.len);
                                    dv->v.arr.values[j].v.str.len = sv->v.arr.values[j].v.str.len;
                                } else {
                                    /* Numeric/bool: copy the union by value. */
                                    dv->v.arr.values[j].v = sv->v.arr.values[j].v;
                                }
                            }
                        } else {
                            dv->v.arr.values = NULL;
                        }
                        break;
                    }
                    default:
                        /* Numeric/bool: copy the union by value. */
                        dv->v = sv->v;
                        break;
                    }
                }
                unified->metadata = kv;
            } else {
                unified->metadata = NULL;
            }

            out->shards    = shards;
            out->n_shards  = (size_t)total;
            free(base_path);
            return OC_OK;

        oom_split:
            /* OOM while deep-copying names/keys: unwind everything so the
             * caller never sees a half-built unified view (NULL names would
             * crash strcmp lookups later). */
            memset(out, 0, sizeof(*out));
            oc_arena_free(unified_arena);
            for (uint64_t i = 0; i < total; i++) close_shard(&shards[i]);
            free(shards);
            free(base_path);
            return OC_ERR_OOM;
        }

        /* Split pattern matched but siblings missing: fall through to
         * single-file open. */
        free(base_path);
        base_path = NULL;
    }

    /* Single-file: open the path as a single shard. */
    {
        OcGgufShard *shards = (OcGgufShard *)calloc(1, sizeof(OcGgufShard));
        if (!shards) return OC_ERR_OOM;

        OcError e = open_shard(path, &shards[0]);
        if (e != OC_OK) {
            free(shards);
            return e;
        }

        /* Unified view = shard 0's parsed file, but with the tensor table */
        OcArena *unified_arena = oc_arena_new(0);
        if (!unified_arena) {
            close_shard(&shards[0]);
            free(shards);
            return OC_ERR_OOM;
        }

        uint64_t total_tensors = shards[0].parsed.tensor_count;
        OcGgufTensorInfo *merged = NULL;
        if (total_tensors > 0) {
            merged = (OcGgufTensorInfo *)oc_arena_alloc(unified_arena,
                (size_t)total_tensors * sizeof(OcGgufTensorInfo), 16);
            if (!merged) {
                oc_arena_free(unified_arena);
                close_shard(&shards[0]);
                free(shards);
                return OC_ERR_OOM;
            }
            memset(merged, 0, (size_t)total_tensors * sizeof(*merged));
            for (uint64_t i = 0; i < total_tensors; i++) {
                const OcGgufTensorInfo *src = &shards[0].parsed.tensors[i];
                merged[i].name = oc_arena_dup(unified_arena, src->name ? src->name : "");
                if (!merged[i].name) goto oom_single;
                merged[i].n_dims           = src->n_dims;
                memcpy(merged[i].dims, src->dims, sizeof(merged[i].dims));
                merged[i].ggml_type        = src->ggml_type;
                merged[i].relative_offset  = src->relative_offset;
                merged[i].absolute_offset  = src->absolute_offset;
                merged[i].shard_index     = 0;
            }
        }

        OcGgufFile *unified = &out->unified;
        unified->magic              = shards[0].parsed.magic;
        unified->version            = shards[0].parsed.version;
        unified->tensor_count       = total_tensors;
        unified->metadata_kv_count  = shards[0].parsed.metadata_kv_count;
        unified->alignment          = shards[0].parsed.alignment;
        unified->data_section_start = shards[0].parsed.data_section_start;
        unified->arena             = unified_arena;
        /* backing_buf stays NULL: bytes are mmap-backed, owned by the shard
         * (see multi-shard path). */
        unified->backing_buf        = NULL;
        unified->backing_len       = 0;
        unified->tensors            = merged;

        /* Deep-copy metadata KV (same as multi-shard path). */
        if (shards[0].parsed.metadata_kv_count > 0) {
            size_t n_kv = (size_t)shards[0].parsed.metadata_kv_count;
            OcGgufMetadataKV *kv = (OcGgufMetadataKV *)oc_arena_alloc(
                unified_arena, n_kv * sizeof(OcGgufMetadataKV), 16);
            if (!kv) {
                oc_arena_free(unified_arena);
                close_shard(&shards[0]);
                free(shards);
                return OC_ERR_OOM;
            }
            for (size_t i = 0; i < n_kv; i++) {
                kv[i].key = oc_arena_dup(unified_arena, shards[0].parsed.metadata[i].key);
                if (!kv[i].key) goto oom_single;
                const OcGgufMetadataValue *sv = &shards[0].parsed.metadata[i].value;
                OcGgufMetadataValue *dv = &kv[i].value;
                dv->type = sv->type;
                switch (sv->type) {
                case OC_GGUF_MT_STRING:
                    dv->v.str.data = oc_arena_dup_n(unified_arena,
                        sv->v.str.data ? sv->v.str.data : "", sv->v.str.len);
                    dv->v.str.len = sv->v.str.len;
                    break;
                case OC_GGUF_MT_ARRAY: {
                    size_t alen = sv->v.arr.len;
                    dv->v.arr.elem_type = sv->v.arr.elem_type;
                    dv->v.arr.len       = alen;
                    if (alen > 0) {
                        dv->v.arr.values = (OcGgufMetadataValue *)oc_arena_alloc(
                            unified_arena, alen * sizeof(OcGgufMetadataValue), 16);
                        if (!dv->v.arr.values) {
                            oc_arena_free(unified_arena);
                            close_shard(&shards[0]);
                            free(shards);
                            return OC_ERR_OOM;
                        }
                        for (size_t j = 0; j < alen; j++) {
                            dv->v.arr.values[j].type = sv->v.arr.values[j].type;
                            if (sv->v.arr.elem_type == OC_GGUF_MT_STRING) {
                                dv->v.arr.values[j].v.str.data = oc_arena_dup_n(
                                    unified_arena,
                                    sv->v.arr.values[j].v.str.data ? sv->v.arr.values[j].v.str.data : "",
                                    sv->v.arr.values[j].v.str.len);
                                dv->v.arr.values[j].v.str.len = sv->v.arr.values[j].v.str.len;
                            } else {
                                dv->v.arr.values[j].v = sv->v.arr.values[j].v;
                            }
                        }
                    } else {
                        dv->v.arr.values = NULL;
                    }
                    break;
                }
                default:
                    dv->v = sv->v;
                    break;
                }
            }
            unified->metadata = kv;
        } else {
            unified->metadata = NULL;
        }

        out->shards    = shards;
        out->n_shards  = 1;
        return OC_OK;

    oom_single:
        /* OOM while deep-copying names/keys (see oom_split above). */
        memset(out, 0, sizeof(*out));
        oc_arena_free(unified_arena);
        close_shard(&shards[0]);
        free(shards);
        return OC_ERR_OOM;
    }
}

OcError oc_gguf_map_advise_hugepage(OcGgufMmappedFile *m)
{
    if (!m || !m->shards) return OC_ERR_INVALID_ARG;
    bool all_ok = true;
    for (size_t i = 0; i < m->n_shards; i++) {
        if (oc_mmap_advise_hugepage(m->shards[i].mmap) != OC_OK) {
            all_ok = false;
        }
    }
    m->hugepage_advised = all_ok;
    return OC_OK;
}

bool oc_gguf_map_mlock_with_headroom(OcGgufMmappedFile *m)
{
    if (!m || !m->shards) return false;
    /* Check the AGGREGATE size against the headroom policy first: each shard's */
    uint64_t available = 0;
    if (oc_linux_mem_available_bytes(&available)
        && oc_gguf_map_total_bytes(m) >= (available * 7ull) / 10ull) {
        oc_log(OC_LOG_INFO,
                "gguf: skipping mlock (total %llu bytes vs %llu avail)",
                (unsigned long long)oc_gguf_map_total_bytes(m),
                (unsigned long long)available);
        for (size_t i = 0; i < m->n_shards; i++) {
            oc_mmap_advise_willneed(m->shards[i].mmap);
            oc_mmap_prefault(m->shards[i].mmap);
        }
        m->mlocked = false;
        return false;
    }
    bool all_locked = true;
    for (size_t i = 0; i < m->n_shards; i++) {
        if (!oc_mmap_mlock_with_headroom(m->shards[i].mmap)) {
            all_locked = false;
        }
    }
    m->mlocked = all_locked;
    return all_locked;
}

uint8_t oc_gguf_map_prefault(const OcGgufMmappedFile *m)
{
    if (!m || !m->shards) return 0;
    uint8_t checksum = 0;
    for (size_t i = 0; i < m->n_shards; i++) {
        checksum ^= oc_mmap_prefault(m->shards[i].mmap);
    }
    return checksum;
}

uint8_t oc_gguf_map_prefault_parallel(const OcGgufMmappedFile *m, size_t n_threads)
{
    if (!m || !m->shards) return 0;
    uint8_t checksum = 0;
    for (size_t i = 0; i < m->n_shards; i++) {
        checksum ^= oc_mmap_prefault_parallel(m->shards[i].mmap, n_threads);
    }
    return checksum;
}

uint64_t oc_gguf_map_total_bytes(const OcGgufMmappedFile *m)
{
    if (!m || !m->shards) return 0;
    uint64_t total = 0;
    for (size_t i = 0; i < m->n_shards; i++) {
        total += (uint64_t)m->shards[i].len;
    }
    return total;
}

const uint8_t *oc_gguf_map_tensor_data(const OcGgufMmappedFile *m,
                                       const OcGgufTensorInfo *info)
{
    if (!m || !info || !m->shards) return NULL;
    if (info->shard_index >= m->n_shards) return NULL;
    const OcGgufShard *s = &m->shards[info->shard_index];
    if (!s->bytes || info->absolute_offset > s->len) return NULL;
    return s->bytes + info->absolute_offset;
}

const OcGgufTensorInfo *oc_gguf_map_tensor_get(const OcGgufMmappedFile *m,
                                               const char *name)
{
    if (!m || !name) return NULL;
    return oc_gguf_tensor_get(&m->unified, name);
}

OcError oc_gguf_map_mapped_tensor_infos(const OcGgufMmappedFile *m,
                                        OcArena *arena,
                                        OcGgufTensorInfo **out_infos,
                                        size_t *out_count)
{
    if (!m || !arena || !out_infos || !out_count) return OC_ERR_INVALID_ARG;
    *out_infos = NULL;
    *out_count = 0;

    OcModelArchitecture arch = oc_gguf_arch_from_file(&m->unified);

    size_t n = (size_t)m->unified.tensor_count;
    if (n == 0) {
        return OC_OK;
    }

    OcGgufTensorInfo *infos = (OcGgufTensorInfo *)oc_arena_alloc(
        arena, n * sizeof(OcGgufTensorInfo), 16);
    if (!infos) return OC_ERR_OOM;
    memset(infos, 0, n * sizeof(*infos));

    for (size_t i = 0; i < n; i++) {
        const OcGgufTensorInfo *src = &m->unified.tensors[i];
        infos[i].name = oc_gguf_map_tensor_name(arch, src->name ? src->name : "", arena);
        if (!infos[i].name) return OC_ERR_OOM;
        infos[i].n_dims          = src->n_dims;
        memcpy(infos[i].dims, src->dims, sizeof(infos[i].dims));
        infos[i].ggml_type       = src->ggml_type;
        infos[i].relative_offset = src->relative_offset;
        infos[i].absolute_offset = src->absolute_offset;
        infos[i].shard_index    = src->shard_index;
    }

    *out_infos = infos;
    *out_count = n;
    return OC_OK;
}

void oc_gguf_map_free(OcGgufMmappedFile *out)
{
    if (!out) return;
    /* Free the unified arena (owns merged tensors + deep-copied metadata). */
    if (out->unified.arena) {
        oc_arena_free(out->unified.arena);
    }
    /* Free each shard (mmap + per-shard parsed arena). */
    if (out->shards) {
        for (size_t i = 0; i < out->n_shards; i++) {
            close_shard(&out->shards[i]);
        }
        free(out->shards);
    }
    memset(out, 0, sizeof(*out));
}
