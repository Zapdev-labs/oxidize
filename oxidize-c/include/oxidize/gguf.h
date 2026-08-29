#ifndef OXIDIZE_GGUF_H
#define OXIDIZE_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque forward declaration of OcMmap (defined in oxidize/util/mmap.h). */
/* public GGUF header; multi-shard callers that need raw mmap access should */
typedef struct OcMmap OcMmap;

/* GGUF magic bytes "GGUF" interpreted as a little-endian u32. little-endian u32 = 0x46554747 */
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

/* Tagged union of all GGUF metadata value types. Strings are length-prefixed */
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

/* Tensor table entry. Matches Rust `GgufTensorInfo`. */
typedef struct OcGgufTensorInfo {
    const char *name;
    uint32_t    n_dims;
    uint64_t    dims[OC_GGUF_MAX_DIMS];
    uint32_t    ggml_type;
    uint64_t    relative_offset;
    uint64_t    absolute_offset;
    uint32_t    shard_index;
} OcGgufTensorInfo;

/* Parsed GGUF file. All fields are read-only after a successful parse. */
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

/* Parse a GGUF file from an in-memory byte buffer. `buf` must remain valid */
OcError oc_gguf_parse(const uint8_t *buf, size_t len, OcGgufFile *out);

/* Read a GGUF file from disk and parse it. */
OcError oc_gguf_open(const char *path, OcGgufFile *out);

/* Free an OcGgufFile and all parser-lifetime allocations. Safe on NULL or
 * zeroed OcGgufFile. Does NOT free `out` itself (it's typically stack or
 * caller-owned). After this call, `*out` is zeroed. */
void oc_gguf_free(OcGgufFile *out);

/* Lookup a metadata KV by key. Returns a pointer to the value inside `f` (so
 * valid until oc_gguf_free()), or NULL if not found. */
const OcGgufMetadataValue *oc_gguf_metadata_get(const OcGgufFile *f,
                                                const char *key);

/* Typed convenience getters. Each returns true and writes `*out` if the key accept any numeric value type (U8/U16/U32/U64/I8/I16/I32/I64/F32/F64/BOOL) */
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

/* ─── Architecture detection (VAL-FOUND-012) ────────────────────────────── */
OcModelArchitecture oc_gguf_arch_from_file(const OcGgufFile *f);

/* ─── mmap-backed multi-shard loading (VAL-FOUND-005, 006, 015) ────────── `oc_gguf_map_open()` is the primary entry point for loading real model weights: it mmaps the file (PROT_READ, MAP_PRIVATE) and parses the GGUF header + metadata + tensor table without copying the weight bytes into userspace memory. */
/* shards are mmap'd and their tensor tables merged into a single unified */

/* A single shard within an OcGgufMmappedFile. */
typedef struct OcGgufShard {
    OcMmap     *mmap;     /* owned: mmap'd file bytes (PROT_READ)            */
    uint8_t    *bytes;    /* = (uint8_t *)oc_mmap_bytes(mmap); convenience    */
    size_t      len;      /* = oc_mmap_len(mmap); convenience                 */
    OcGgufFile  parsed;   /* per-shard parse result (owns per-shard arena)    */
} OcGgufShard;

/* Unified mmap-backed GGUF view. `unified` is the parsed file the caller */
typedef struct OcGgufMmappedFile {
    /* `unified.backing_buf` is always NULL: the file bytes are mmap-backed
     * and owned by `shards`. Free ONLY via oc_gguf_map_free(); never call
     * oc_gguf_free(&m->unified) directly. */
    OcGgufFile   unified;       /* merged view (owns the unified arena)     */
    OcGgufShard *shards;        /* array of `n_shards` shards (owned)       */
    size_t       n_shards;
    bool         hugepage_advised;  /* true if MADV_HUGEPAGE applied to all */
    bool         mlocked;           /* true if mlock() succeeded on all      */
} OcGgufMmappedFile;

/* Open a GGUF file via mmap (PROT_READ, MAP_PRIVATE). */
OcError oc_gguf_map_open(const char *path, OcGgufMmappedFile *out);

/* Apply MADV_HUGEPAGE to every shard (Linux only, best-effort). The caller */
OcError oc_gguf_map_advise_hugepage(OcGgufMmappedFile *out);

/* mlock every shard into physical RAM, but only if the total mapping fits in MemAvailable with >= 30% headroom (model_bytes < available * 7 / 10). */
bool oc_gguf_map_mlock_with_headroom(OcGgufMmappedFile *out);

/* Sequential prefault sweep across all shards (touch every 4 KiB page).
 * Returns the XOR checksum of all touched bytes. */
uint8_t oc_gguf_map_prefault(const OcGgufMmappedFile *m);

/* Parallel prefault sweep across all shards using `n_threads` pthreads.
 * Returns the XOR checksum. */
uint8_t oc_gguf_map_prefault_parallel(const OcGgufMmappedFile *m, size_t n_threads);

/* Total byte length across all shards (sum of mmap lengths). */
uint64_t oc_gguf_map_total_bytes(const OcGgufMmappedFile *m);

/* Returns a pointer to `info`'s tensor data within its shard's mmap. */
const uint8_t *oc_gguf_map_tensor_data(const OcGgufMmappedFile *m,
                                       const OcGgufTensorInfo *info);

/* Look up a tensor by mapped name in the unified view (VAL-FOUND-007..011). */
const OcGgufTensorInfo *oc_gguf_map_tensor_get(const OcGgufMmappedFile *m,
                                              const char *name);

/* Resolve the architecture from the unified metadata and return a freshly */
OcError oc_gguf_map_mapped_tensor_infos(const OcGgufMmappedFile *m,
                                        OcArena *arena,
                                        OcGgufTensorInfo **out_infos,
                                        size_t *out_count);

/* Free an OcGgufMmappedFile and all per-shard mmaps + arenas. Safe on NULL
 * or zeroed OcGgufMmappedFile. After this call, `*out` is zeroed. */
void oc_gguf_map_free(OcGgufMmappedFile *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GGUF_H */
