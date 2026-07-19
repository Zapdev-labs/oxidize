/* gguf.c — GGUF v3/v2 binary format parser.
 *
 * Port of oxidize-core/src/format/gguf.rs (parse_gguf, ByteReader, alignment
 * handling, metadata KV value type dispatch). The C port mirrors the Rust
 * logic 1:1 so bit-exact tensor inventory parity holds.
 *
 * Layout (little-endian throughout):
 *   header:   magic[4] | version:u32 | tensor_count:u64 | metadata_kv_count:u64
 *   metadata: for each KV: key_len:u64 | key:bytes | value_type:u32 | value
 *   tensors:  for each tensor:
 *       name_len:u64 | name:bytes | n_dims:u32 | dims[n_dims]:u64 |
 *       ggml_type:u32 | relative_offset:u64
 *   data:     aligned to `general.alignment` (default 32)
 *
 * Malformed input (bad magic, unsupported version, truncated reads, invalid
 * alignment, offsets beyond EOF) returns OC_ERR_FORMAT. No segfaults, no UB:
 * all reads are bounds-checked against the buffer length.
 *
 * Allocation strategy: a single OcArena owns every parser-lifetime allocation
 * (KV keys, string values, array buffers, tensor names). oc_gguf_free()
 * releases the arena in one shot. The metadata/tensors arrays themselves are
 * also arena-allocated for simplicity (single free, no per-entry tracking).
 */
#include "oxidize/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/log.h"
#include "oxidize/util/bytes.h"
#include "oxidize/util/file.h"

/* ─── ByteReader: bounds-checked sequential cursor over a byte buffer ────────
 *
 * Mirrors the Rust `ByteReader` in gguf.rs. Every read checks bounds and
 * returns an OcError (OC_ERR_FORMAT on EOF/overflow) so the parser never
 * touches out-of-range memory. */
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

/* Read a GGUF string (length-prefixed u64, no NUL terminator on disk).
 * On success writes an arena-owned, NUL-terminated copy to `*out_str` and
 * the byte length (excluding the NUL) to `*out_len`. The arena owns the
 * returned pointer. Returns OC_ERR_FORMAT on truncation. */
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

    /* oc_arena_dup_n allocates len+1 bytes and NUL-terminates. */
    char *dst = oc_arena_dup_n(arena, (const char *)p, len);
    if (!dst) return OC_ERR_OOM;

    *out_str = dst;
    if (out_len) *out_len = len;
    return OC_OK;
}

/* ─── Metadata value dispatch ───────────────────────────────────────────────
 *
 * Mirrors Rust `read_value_of_type`. Recursively handles ARRAY (which reads
 * its element_type then `len` elements of that type). Allocates array
 * payloads in the arena. */
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

/* ─── Alignment ───────────────────────────────────────────────────────────── */

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

/* ─── Metadata type helpers ───────────────────────────────────────────────── */

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

/* ─── Parser core ─────────────────────────────────────────────────────────── */

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

    /* data_section_start = align_up(reader.cursor, alignment). */
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

/* ─── Lookups ─────────────────────────────────────────────────────────────── */

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

/* ─── Dump (debug aid for inspect_gguf) ───────────────────────────────────── */

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
