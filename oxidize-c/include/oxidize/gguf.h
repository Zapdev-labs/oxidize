/*
 * gguf.h — GGUF v3/v2 parser types and API.
 *
 * Port of oxidize-core/src/format/gguf.rs to C11. Parses the GGUF binary
 * format: header (magic, version, tensor_count, metadata_kv_count), all
 * metadata KV value types (U8/U16/U32/U64/I8/I16/I32/I64/F32/F64/BOOL/
 * STRING/ARRAY), and the tensor table (name, n_dims, dims, dtype, offset).
 *
 * Malformed input returns OC_ERR_FORMAT (no segfault). All parser-lifetime
 * allocations (metadata strings, arrays, tensor names) live in an OcArena
 * owned by OcGgufFile and freed by oc_gguf_free(). The backing file bytes
 * are owned by OcGgufFile when opened via oc_gguf_open(); when parsed via
 * oc_gguf_parse() the caller owns the buffer.
 */
#ifndef OXIDIZE_GGUF_H
#define OXIDIZE_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/arena.h"
#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF magic bytes "GGUF" interpreted as a little-endian u32.
 *   'G' = 0x47, 'G' = 0x47, 'U' = 0x55, 'F' = 0x46
 *   little-endian u32 = 0x46554747
 */
#define OC_GGUF_MAGIC 0x46554747u

/* Supported GGUF versions. v1 (very old) is rejected. */
#define OC_GGUF_VERSION_2 2u
#define OC_GGUF_VERSION_3 3u

/* Default alignment when "general.alignment" metadata is absent. */
#define OC_GGUF_DEFAULT_ALIGNMENT 32ull

/* GGUF does not hard-cap tensor dims; 8 is ample for all known models
 * (max used in practice is 4). */
#define OC_GGUF_MAX_DIMS 8

/* Metadata value type enum (GGUF spec §"metadata value type"). Numeric
 * values match the on-disk encoding and MUST NOT change. */
typedef enum {
    OC_GGUF_MT_UINT8   = 0,
    OC_GGUF_MT_INT8    = 1,
    OC_GGUF_MT_UINT16  = 2,
    OC_GGUF_MT_INT16   = 3,
    OC_GGUF_MT_UINT32  = 4,
    OC_GGUF_MT_INT32   = 5,
    OC_GGUF_MT_FLOAT32 = 6,
    OC_GGUF_MT_BOOL    = 7,
    OC_GGUF_MT_STRING  = 8,
    OC_GGUF_MT_ARRAY   = 9,
    OC_GGUF_MT_UINT64  = 10,
    OC_GGUF_MT_INT64   = 11,
    OC_GGUF_MT_FLOAT64 = 12,
    OC_GGUF_MT__COUNT,
    OC_GGUF_MT_UNKNOWN = 0xffffffffu,
} OcGgufMetadataType;

/* Forward declaration: arrays contain values, values may contain arrays. */
typedef struct OcGgufMetadataValue OcGgufMetadataValue;
typedef struct OcGgufMetadataArray OcGgufMetadataArray;

/* A metadata array: homogeneous sequence of values of `elem_type`. */
struct OcGgufMetadataArray {
    OcGgufMetadataType   elem_type;
    size_t               len;       /* number of elements                      */
    OcGgufMetadataValue *values;   /* arena-owned array of `len` values       */
};

/* Tagged union of all GGUF metadata value types. Strings are length-prefixed
 * (the spec permits embedded NULs); `str.len` is the byte length and `str.data`
 * is NOT guaranteed to be NUL-terminated. Use the convenience getters below
 * for NUL-terminated access where appropriate. */
struct OcGgufMetadataValue {
    OcGgufMetadataType type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        uint64_t u64;
        int64_t  i64;
        float    f32;
        double   f64;
        bool     b;
        struct { char *data; size_t len; } str;   /* arena-owned            */
        OcGgufMetadataArray arr;
    } v;
};

/* A single metadata KV pair. `key` is arena-owned and NUL-terminated. */
typedef struct OcGgufMetadataKV {
    const char            *key;
    OcGgufMetadataValue    value;
} OcGgufMetadataKV;

/* Tensor table entry. Matches Rust `GgufTensorInfo`.
 *   - `name` is arena-owned and NUL-terminated.
 *   - `dims[i]` follows GGUF order: dims[0] is the innermost (fastest-varying)
 *     dimension. `n_dims` is the count of valid dims entries.
 *   - `ggml_type` is the raw on-disk ggml dtype id (0=F32, 1=F16, 8=Q8_0, ...).
 *   - `relative_offset` is the byte offset from `data_section_start`.
 *   - `absolute_offset` is `data_section_start + relative_offset` (the byte
 *     offset into the file where this tensor's data begins). */
typedef struct OcGgufTensorInfo {
    const char *name;
    uint32_t    n_dims;
    uint64_t    dims[OC_GGUF_MAX_DIMS];
    uint32_t    ggml_type;
    uint64_t    relative_offset;
    uint64_t    absolute_offset;
} OcGgufTensorInfo;

/* Parsed GGUF file. All fields are read-only after a successful parse.
 * `arena` owns every parser-lifetime allocation (KV keys, string values,
 * array buffers, tensor names). `backing_buf` is the file bytes when opened
 * via oc_gguf_open(); NULL when parsed via oc_gguf_parse() (caller owns the
 * buffer). oc_gguf_free() releases the arena and (if present) the backing
 * buffer. */
typedef struct OcGgufFile {
    uint32_t            magic;               /* always OC_GGUF_MAGIC on success */
    uint32_t            version;             /* 2 or 3                           */
    uint64_t            tensor_count;       /* raw header field                 */
    uint64_t            metadata_kv_count;  /* raw header field                 */
    OcGgufMetadataKV  *metadata;           /* array of `metadata_kv_count` KV  */
    OcGgufTensorInfo   *tensors;            /* array of `tensor_count` entries  */
    uint64_t            alignment;          /* power of two                     */
    uint64_t            data_section_start; /* byte offset into the file        */
    OcArena            *arena;              /* owns all parser allocations       */
    uint8_t            *backing_buf;        /* NULL if caller owns the buffer    */
    size_t              backing_len;        /* valid iff backing_buf != NULL     */
} OcGgufFile;

/* Parse a GGUF file from an in-memory byte buffer. `buf` must remain valid
 * for the lifetime of the returned OcGgufFile (string and tensor-name
 * pointers alias into a dup'd copy stored in `out->arena`, NOT into `buf`
 * itself — so the caller may free `buf` immediately after a successful parse).
 * Returns OC_OK, OC_ERR_FORMAT (bad magic/version/alignment/truncated),
 * OC_ERR_OOM, or OC_ERR_INVALID_ARG (NULL args). On error, `*out` is zeroed. */
OcError oc_gguf_parse(const uint8_t *buf, size_t len, OcGgufFile *out);

/* Read a GGUF file from disk and parse it. The file bytes are read into a
 * freshly malloc'd buffer owned by `out->backing_buf` (freed by
 * oc_gguf_free()). Returns OC_OK, OC_ERR_IO, OC_ERR_FORMAT, OC_ERR_OOM, or
 * OC_ERR_INVALID_ARG. */
OcError oc_gguf_open(const char *path, OcGgufFile *out);

/* Free an OcGgufFile and all parser-lifetime allocations. Safe on NULL or
 * zeroed OcGgufFile. Does NOT free `out` itself (it's typically stack or
 * caller-owned). After this call, `*out` is zeroed. */
void oc_gguf_free(OcGgufFile *out);

/* Lookup a metadata KV by key. Returns a pointer to the value inside `f` (so
 * valid until oc_gguf_free()), or NULL if not found. */
const OcGgufMetadataValue *oc_gguf_metadata_get(const OcGgufFile *f,
                                                const char *key);

/* Typed convenience getters. Each returns true and writes `*out` if the key
 * exists and the value is convertible to the requested type. Numeric getters
 * accept any numeric value type (U8/U16/U32/U64/I8/I16/I32/I64/F32/F64/BOOL)
 * with range checks; on out-of-range they return false. */
bool oc_gguf_metadata_get_u8 (const OcGgufFile *f, const char *key, uint8_t  *out);
bool oc_gguf_metadata_get_u16(const OcGgufFile *f, const char *key, uint16_t *out);
bool oc_gguf_metadata_get_u32(const OcGgufFile *f, const char *key, uint32_t *out);
bool oc_gguf_metadata_get_u64(const OcGgufFile *f, const char *key, uint64_t *out);
bool oc_gguf_metadata_get_i32(const OcGgufFile *f, const char *key, int32_t  *out);
bool oc_gguf_metadata_get_f32(const OcGgufFile *f, const char *key, float    *out);
bool oc_gguf_metadata_get_f64(const OcGgufFile *f, const char *key, double   *out);
bool oc_gguf_metadata_get_bool(const OcGgufFile *f, const char *key, bool    *out);

/* String getter: writes the value's string payload to `*out_data` and
 * `*out_len` (if not NULL). The returned pointer aliases into `f` and is valid
 * until oc_gguf_free(). Returns true if the key exists and is a STRING. */
bool oc_gguf_metadata_get_str(const OcGgufFile *f, const char *key,
                              const char **out_data, size_t *out_len);

/* Lookup a tensor by name. Returns a pointer into `f` (valid until
 * oc_gguf_free()), or NULL if not found. */
const OcGgufTensorInfo *oc_gguf_tensor_get(const OcGgufFile *f, const char *name);

/* Pretty-print the GGUF file to stderr (for debugging / `inspect_gguf`).
 * Library code MUST prefer oc_log over this; this is a debug aid. */
void oc_gguf_dump(const OcGgufFile *f);

/* Convert an on-disk metadata type id to OcGgufMetadataType. Returns
 * OC_GGUF_MT_UNKNOWN for unrecognized ids. */
OcGgufMetadataType oc_gguf_metadata_type_from_u32(uint32_t raw);

/* Human-readable name for a metadata type ("U8", "STRING", "ARRAY", ...). */
const char *oc_gguf_metadata_type_name(OcGgufMetadataType t);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GGUF_H */
