/* GGUF v2/v3 parser (mmap-based). Pure C11 port of oxidize-cpp/src/gguf.cpp.
 * Little-endian readers, default alignment 32, data section aligned up.
 * Unknown/custom tensor type ids (e.g. AL family 240-243) are accepted:
 * the raw ggml type id and raw data pointer are always exposed. */
#ifndef OC_GGUF_H
#define OC_GGUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum {
  GGUF_T_U8 = 0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16, GGUF_T_U32, GGUF_T_I32,
  GGUF_T_F32, GGUF_T_BOOL, GGUF_T_STRING, GGUF_T_ARRAY, GGUF_T_U64,
  GGUF_T_I64, GGUF_T_F64,
} GgufMetadataType;

typedef struct GgufValue {
  int kind; /* GgufMetadataType */
  union {
    uint64_t u;             /* U8/U16/U32/U64/BOOL */
    int64_t i;              /* I8/I16/I32/I64 */
    double f;               /* F32/F64 */
    struct { const char* ptr; size_t len; } str; /* points into mmap */
    struct { int elem_kind; struct GgufValue* items; size_t n; } arr;
  } v;
} GgufValue;

typedef struct {
  char* key;
  GgufValue val;
} GgufKv;

#define GGUF_MAX_DIMS 4

typedef struct {
  char* name;
  uint32_t n_dims;
  uint64_t dims[GGUF_MAX_DIMS];
  uint32_t ggml_type;      /* raw type id, may be unknown (240-243 AL family) */
  uint64_t offset;         /* absolute file offset */
  const uint8_t* data;     /* pointer into mmap */
} GgufTensorInfo;

/* One extra mmap for a split GGUF (shard 0 stays in map/size, like a single
 * file; shards 1..N-1 live here). A tensor's data points into whichever shard
 * holds it. */
typedef struct GgufShard {
  void* map;
  size_t size;
} GgufShard;

typedef struct {
  void* map;
  size_t size;
  uint32_t version;
  uint64_t alignment;
  uint64_t data_section_start;
  GgufKv* kvs;
  size_t n_kv;
  GgufTensorInfo* tensors;
  size_t n_tensors;
  /* O(1) name lookup: open-addressing tables over kvs[]/tensors[], each slot
   * holding index+1 (0 = empty). NULL if empty or the build failed, in which
   * case gguf_find/gguf_tensor fall back to a linear scan. */
  size_t* kv_hash;
  size_t kv_hash_cap;
  size_t* tensor_hash;
  size_t tensor_hash_cap;
  /* Extra shard mmaps for split GGUFs (count = n_shards). */
  GgufShard* shards;
  size_t n_shards;
} GgufFile;

/* Returns 0 on success; on failure writes a message into err (if non-NULL).
 * If `path` is one shard of a split model (<base>-NNNNN-of-MMMMM.gguf with
 * MMMMM>=2 and every sibling present on disk), all shards are mmap'd and merged
 * into one tensor table; shard 0 supplies the metadata. Otherwise a single file
 * is opened. Either way the resulting GgufFile is used identically. */
int gguf_open(GgufFile* f, const char* path, char* err, size_t errlen);
/* Parse from an in-memory buffer (buffer must outlive f; no mmap taken). */
int gguf_parse(GgufFile* f, const uint8_t* bytes, size_t len, char* err, size_t errlen);
void gguf_close(GgufFile* f);

const GgufValue* gguf_find(const GgufFile* f, const char* key);
bool gguf_get_u32(const GgufFile* f, const char* key, uint32_t* out);
bool gguf_get_f32(const GgufFile* f, const char* key, float* out);
/* Returned string is malloc'd (NUL-terminated copy); caller frees. NULL if absent. */
char* gguf_get_str(const GgufFile* f, const char* key);
const GgufValue* gguf_get_arr(const GgufFile* f, const char* key); /* NULL if not array */
const GgufTensorInfo* gguf_tensor(const GgufFile* f, const char* name);

/* "general.architecture" (malloc'd copy, caller frees; NULL if absent). */
char* gguf_architecture(const GgufFile* f);

#endif
