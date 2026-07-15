/* GGUF v2/v3 parser. See gguf.h. */
#include "gguf.h"

#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  const uint8_t* bytes;
  size_t len;
  size_t cur;
  int err; /* sticky */
} Reader;

static const uint8_t* rd_exact(Reader* r, size_t n) {
  if (r->err || r->cur > r->len || n > r->len - r->cur) {
    r->err = 1;
    return NULL;
  }
  const uint8_t* p = r->bytes + r->cur;
  r->cur += n;
  return p;
}

static uint8_t rd_u8(Reader* r) { const uint8_t* p = rd_exact(r, 1); return p ? p[0] : 0; }
static uint16_t rd_u16(Reader* r) {
  const uint8_t* p = rd_exact(r, 2);
  return p ? (uint16_t)(p[0] | (uint16_t)p[1] << 8) : 0;
}
static uint32_t rd_u32(Reader* r) {
  const uint8_t* p = rd_exact(r, 4);
  if (!p) return 0;
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t rd_u64(Reader* r) {
  const uint8_t* p = rd_exact(r, 8);
  if (!p) return 0;
  uint64_t v = 0;
  for (int b = 0; b < 8; ++b) v |= (uint64_t)p[b] << (8 * b);
  return v;
}
static float rd_f32(Reader* r) {
  uint32_t bits = rd_u32(r);
  float out;
  memcpy(&out, &bits, 4);
  return out;
}
static double rd_f64(Reader* r) {
  uint64_t bits = rd_u64(r);
  double out;
  memcpy(&out, &bits, 8);
  return out;
}

/* Reads a GGUF string; returns pointer into mmap (not NUL-terminated). */
static const char* rd_str(Reader* r, size_t* out_len) {
  uint64_t len = rd_u64(r);
  const uint8_t* p = rd_exact(r, (size_t)len);
  *out_len = (size_t)len;
  return (const char*)p;
}

static int rd_value(Reader* r, int kind, GgufValue* v) {
  v->kind = kind;
  switch (kind) {
    case GGUF_T_U8: v->v.u = rd_u8(r); break;
    case GGUF_T_I8: v->v.i = (int8_t)rd_u8(r); break;
    case GGUF_T_U16: v->v.u = rd_u16(r); break;
    case GGUF_T_I16: v->v.i = (int16_t)rd_u16(r); break;
    case GGUF_T_U32: v->v.u = rd_u32(r); break;
    case GGUF_T_I32: v->v.i = (int32_t)rd_u32(r); break;
    case GGUF_T_F32: v->v.f = (double)rd_f32(r); break;
    case GGUF_T_BOOL: v->v.u = rd_u8(r) != 0; break;
    case GGUF_T_U64: v->v.u = rd_u64(r); break;
    case GGUF_T_I64: v->v.i = (int64_t)rd_u64(r); break;
    case GGUF_T_F64: v->v.f = rd_f64(r); break;
    case GGUF_T_STRING: {
      size_t len;
      const char* s = rd_str(r, &len);
      v->v.str.ptr = s;
      v->v.str.len = len;
      break;
    }
    case GGUF_T_ARRAY: {
      uint32_t ek = rd_u32(r);
      uint64_t n = rd_u64(r);
      /* No arrays of arrays: llama.cpp does not emit them, and allowing them
       * lets a 12-bytes-per-level file recurse rd_value() until the stack
       * dies (and leaks the inner items on the failure path). */
      if (r->err || ek > GGUF_T_F64 || ek == GGUF_T_ARRAY) return -1;
      /* Sanity: each element consumes at least 1 byte. */
      if (n > r->len - r->cur) { r->err = 1; return -1; }
      GgufValue* items = n ? calloc((size_t)n, sizeof(GgufValue)) : NULL;
      if (n && !items) { r->err = 1; return -1; }
      for (uint64_t k = 0; k < n; ++k) {
        if (rd_value(r, (int)ek, &items[k]) != 0 || r->err) {
          free(items);
          return -1;
        }
      }
      v->v.arr.elem_kind = (int)ek;
      v->v.arr.items = items;
      v->v.arr.n = (size_t)n;
      break;
    }
    default:
      r->err = 1;
      return -1;
  }
  return r->err ? -1 : 0;
}

static void free_value(GgufValue* v) {
  if (v->kind == GGUF_T_ARRAY) {
    for (size_t i = 0; i < v->v.arr.n; ++i) free_value(&v->v.arr.items[i]);
    free(v->v.arr.items);
  }
}

static void set_err(char* err, size_t errlen, const char* msg) {
  if (err && errlen) snprintf(err, errlen, "%s", msg);
}

/* ---- O(1) name index (open addressing over the existing name arrays) -------
 * Model load does ~13 lookups over 60+ layers on an 800+ tensor table; a linear
 * scan makes that O(n^2). Slots hold index+1 (0 = empty). On a duplicate name
 * the first insertion wins, so a hash hit is byte-for-byte the linear scan's
 * first match. */
static uint64_t fnv1a(const char* s) {
  uint64_t h = 1469598103934665603ull;
  for (; *s; ++s) {
    h ^= (uint8_t)*s;
    h *= 1099511628211ull;
  }
  return h;
}

typedef const char* (*NameFn)(const void* base, size_t i);
static const char* kv_name(const void* b, size_t i) { return ((const GgufKv*)b)[i].key; }
static const char* tensor_name(const void* b, size_t i) {
  return ((const GgufTensorInfo*)b)[i].name;
}

static size_t* index_build(size_t n, const void* base, NameFn get, size_t* cap_out) {
  size_t cap = 8;
  while (cap < n * 2) {
    size_t next = cap << 1;
    if (next < cap) { *cap_out = 0; return NULL; } /* overflow: skip the index */
    cap = next;
  }
  size_t* tab = calloc(cap, sizeof(size_t));
  if (!tab) { *cap_out = 0; return NULL; }
  size_t mask = cap - 1;
  for (size_t i = 0; i < n; ++i) {
    const char* name = get(base, i);
    if (!name) continue;
    size_t j = (size_t)(fnv1a(name) & mask);
    int dup = 0;
    while (tab[j]) {
      if (strcmp(get(base, tab[j] - 1), name) == 0) { dup = 1; break; }
      j = (j + 1) & mask;
    }
    if (!dup) tab[j] = i + 1;
  }
  *cap_out = cap;
  return tab;
}

/* Returns index+1, or 0 if not found (or tab is NULL). */
static size_t index_lookup(const size_t* tab, size_t cap, const void* base, NameFn get,
                           const char* key) {
  if (!tab) return 0;
  size_t mask = cap - 1;
  size_t j = (size_t)(fnv1a(key) & mask);
  while (tab[j]) {
    if (strcmp(get(base, tab[j] - 1), key) == 0) return tab[j];
    j = (j + 1) & mask;
  }
  return 0;
}

static void build_indexes(GgufFile* f) {
  f->kv_hash = index_build(f->n_kv, f->kvs, kv_name, &f->kv_hash_cap);
  f->tensor_hash = index_build(f->n_tensors, f->tensors, tensor_name, &f->tensor_hash_cap);
}

/* Bytes on disk for one tensor, or 0 when the type's block geometry is unknown
 * (custom AL family etc.) and so cannot be bounds-checked; UINT64_MAX when the
 * declared geometry overflows a u64 (a corrupt dim set). Block sizes mirror the
 * fixed ggml wire-format constants in quant.h (kept local so this parser stays
 * dependency-free — it is also built standalone as the fuzz target). */
static uint64_t tensor_nbytes(const GgufTensorInfo* ti) {
  if (ti->n_dims == 0) return 0;
  uint64_t cols = ti->dims[0], rb;
  switch (ti->ggml_type) {
    case 0: /* F32  */ rb = cols > UINT64_MAX / 4 ? UINT64_MAX : cols * 4; break;
    case 1: /* F16  */ rb = cols > UINT64_MAX / 2 ? UINT64_MAX : cols * 2; break;
    case 2: /* Q4_0 */ rb = cols % 32 ? 0 : cols / 32 * 18; break;
    case 8: /* Q8_0 */ rb = cols % 32 ? 0 : cols / 32 * 34; break;
    case 12: /* Q4_K */ rb = cols % 256 ? 0 : cols / 256 * 144; break;
    case 13: /* Q5_K */ rb = cols % 256 ? 0 : cols / 256 * 176; break;
    case 14: /* Q6_K */ rb = cols % 256 ? 0 : cols / 256 * 210; break;
    case 243: /* AL5_XS */ rb = cols % 32 ? 0 : cols / 32 * 14; break;
    default: return 0; /* unknown geometry: not bounds-checkable */
  }
  if (rb == 0) return 0;
  if (rb == UINT64_MAX) return UINT64_MAX;
  uint64_t rows = 1;
  for (uint32_t d = 1; d < ti->n_dims; ++d) {
    uint64_t dim = ti->dims[d];
    if (dim && rows > UINT64_MAX / dim) return UINT64_MAX;
    rows *= dim;
  }
  if (rows && rb > UINT64_MAX / rows) return UINT64_MAX;
  return rb * rows;
}

/* Frees a parse (kvs, tensors, indexes) but touches no mmap — used both by
 * gguf_close and when merging per-shard parses whose maps outlive them. */
static void free_parse(GgufFile* f) {
  for (size_t i = 0; i < f->n_kv; ++i) {
    free(f->kvs[i].key);
    free_value(&f->kvs[i].val);
  }
  free(f->kvs);
  f->kvs = NULL;
  f->n_kv = 0;
  for (size_t i = 0; i < f->n_tensors; ++i) free(f->tensors[i].name);
  free(f->tensors);
  f->tensors = NULL;
  f->n_tensors = 0;
  free(f->kv_hash);
  f->kv_hash = NULL;
  f->kv_hash_cap = 0;
  free(f->tensor_hash);
  f->tensor_hash = NULL;
  f->tensor_hash_cap = 0;
}

int gguf_parse(GgufFile* f, const uint8_t* bytes, size_t len, char* err, size_t errlen) {
  memset(f, 0, sizeof(*f));
  Reader r = {bytes, len, 0, 0};
  const uint8_t* magic = rd_exact(&r, 4);
  if (!magic || memcmp(magic, "GGUF", 4) != 0) {
    set_err(err, errlen, "gguf: invalid gguf magic");
    return -1;
  }
  uint32_t version = rd_u32(&r);
  if (version != 2 && version != 3) {
    set_err(err, errlen, "gguf: unsupported gguf version");
    return -1;
  }
  uint64_t tensor_count = rd_u64(&r);
  uint64_t metadata_count = rd_u64(&r);
  if (r.err) {
    set_err(err, errlen, "gguf: unexpected end of file");
    return -1;
  }
  /* A kv costs >= 13 bytes on the wire and a tensor info >= 24: a count that
   * cannot fit in the file is a corrupt header, not a 100 GiB allocation. */
  if (metadata_count > len / 13 || tensor_count > len / 24) {
    set_err(err, errlen, "gguf: metadata/tensor count exceeds file size");
    return -1;
  }

  f->version = version;
  f->n_kv = 0;
  f->n_tensors = 0;
  f->kvs = metadata_count ? calloc((size_t)metadata_count, sizeof(GgufKv)) : NULL;
  f->tensors = tensor_count ? calloc((size_t)tensor_count, sizeof(GgufTensorInfo)) : NULL;
  if ((metadata_count && !f->kvs) || (tensor_count && !f->tensors)) {
    set_err(err, errlen, "gguf: out of memory");
    goto fail_parse2;
  }

  for (uint64_t m = 0; m < metadata_count; ++m) {
    size_t klen;
    const char* k = rd_str(&r, &klen);
    uint32_t vt = rd_u32(&r);
    if (r.err || vt > GGUF_T_F64) goto fail_parse;
    GgufKv* kv = &f->kvs[f->n_kv];
    kv->key = malloc(klen + 1);
    if (!kv->key) goto fail_parse;
    memcpy(kv->key, k, klen);
    kv->key[klen] = 0;
    if (rd_value(&r, (int)vt, &kv->val) != 0) {
      free(kv->key);
      kv->key = NULL;
      goto fail_parse;
    }
    f->n_kv++;
  }

  for (uint64_t t = 0; t < tensor_count; ++t) {
    GgufTensorInfo* ti = &f->tensors[f->n_tensors];
    size_t nlen;
    const char* n = rd_str(&r, &nlen);
    if (r.err) goto fail_parse;
    ti->name = malloc(nlen + 1);
    if (!ti->name) goto fail_parse;
    memcpy(ti->name, n, nlen);
    ti->name[nlen] = 0;
    f->n_tensors++; /* counted now: every goto below must free this name */
    uint32_t n_dims = rd_u32(&r);
    if (r.err || n_dims > GGUF_MAX_DIMS) goto fail_parse;
    ti->n_dims = n_dims;
    for (uint32_t d = 0; d < n_dims; ++d) ti->dims[d] = rd_u64(&r);
    /* Raw type id kept as-is: custom types (AL family 240-243) must load. */
    ti->ggml_type = rd_u32(&r);
    ti->offset = rd_u64(&r); /* relative until resolved below */
    if (r.err) goto fail_parse;
  }

  uint64_t alignment = 32;
  const GgufValue* av = gguf_find(f, "general.alignment");
  if (av) {
    if (av->kind == GGUF_T_U8 || av->kind == GGUF_T_U16 || av->kind == GGUF_T_U32 ||
        av->kind == GGUF_T_U64)
      alignment = av->v.u;
    else if (av->v.i > 0)
      alignment = (uint64_t)av->v.i;
  }
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    set_err(err, errlen, "gguf: invalid alignment");
    goto fail_parse2;
  }
  f->alignment = alignment;

  uint64_t data_start = ((uint64_t)r.cur + alignment - 1) & ~(alignment - 1);
  if (data_start > len) {
    if (f->n_tensors > 0) { /* tensor-less files may end unaligned */
      set_err(err, errlen, "gguf: unexpected end of file");
      goto fail_parse2;
    }
    data_start = len;
  }
  f->data_section_start = data_start;
  for (size_t t = 0; t < f->n_tensors; ++t) {
    GgufTensorInfo* ti = &f->tensors[t];
    if (ti->offset > UINT64_MAX - data_start) goto fail_parse;
    ti->offset += data_start;
    if (ti->offset > len) {
      set_err(err, errlen, "gguf: tensor offset past end of file");
      goto fail_parse2;
    }
    /* Reject a tensor whose declared size overflows a u64 (corrupt dims). The
     * full offset+nbytes<=len bound is enforced per-shard in the split loader
     * ("its shard's mmap"); single files stay lenient here to match the Rust
     * reference and accept metadata-only stub fixtures. */
    if (tensor_nbytes(ti) == UINT64_MAX) {
      set_err(err, errlen, "gguf: tensor size overflow");
      goto fail_parse2;
    }
    ti->data = bytes + ti->offset;
  }
  build_indexes(f);
  return 0;

fail_parse:
  set_err(err, errlen, "gguf: unexpected end of file / parse error");
fail_parse2:
  gguf_close(f);
  return -1;
}

/* mmap `path` and parse it. The map is returned to the caller (g->map is left
 * NULL); the caller owns munmap. */
static int map_parse(const char* path, void** map_out, size_t* size_out, GgufFile* g,
                     char* err, size_t errlen) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    set_err(err, errlen, "gguf: cannot open file");
    return -1;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size == 0) {
    close(fd);
    set_err(err, errlen, "gguf: cannot stat file / empty file");
    return -1;
  }
  size_t size = (size_t)st.st_size;
  void* map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    set_err(err, errlen, "gguf: mmap failed");
    return -1;
  }
  madvise(map, size, MADV_WILLNEED);
  if (gguf_parse(g, (const uint8_t*)map, size, err, errlen) != 0) {
    munmap(map, size);
    return -1;
  }
  *map_out = map;
  *size_out = size;
  return 0;
}

/* If `path` looks like a split shard (<prefix>-NNNNN-of-MMMMM<suffix>, both
 * groups exactly 5 digits, MMMMM>=2), reports the prefix length, MMMMM, and the
 * offset of <suffix> so sibling paths can be rebuilt. Returns 0 on match. */
static int split_info(const char* path, size_t* pfx_len, uint64_t* total, size_t* suf_off) {
  const char* found = NULL;
  for (const char* p = path; (p = strstr(p, "-of-")); p += 4) found = p;
  if (!found || (size_t)(found - path) < 6) return -1;
  for (int d = 1; d <= 5; ++d)
    if (!isdigit((unsigned char)found[-d])) return -1;
  if (found[-6] != '-') return -1;
  for (int d = 0; d < 5; ++d)
    if (!isdigit((unsigned char)found[4 + d])) return -1;
  uint64_t t = 0;
  for (int d = 0; d < 5; ++d) t = t * 10 + (uint64_t)(found[4 + d] - '0');
  if (t < 2) return -1;
  *pfx_len = (size_t)(found - path) - 6; /* excludes the '-' before NNNNN */
  *total = t;
  *suf_off = (size_t)(found - path) + 9; /* skip "-of-" (4) + MMMMM (5) */
  return 0;
}

/* Loads all `total` shards derived from `path` and merges them into `f`. Shard 0
 * supplies the metadata; every shard contributes its tensors, each pointing into
 * its own mmap. Returns 0 on success, +1 on a hard error (siblings existed but a
 * shard failed to load / was corrupt), -1 if the sibling set is incomplete on
 * disk (caller should fall back to opening `path` as a single file). */
static int open_shards(GgufFile* f, const char* path, uint64_t total, size_t pfx_len,
                       size_t suf_off, char* err, size_t errlen) {
  const char* suffix = path + suf_off;
  size_t bufsz = pfx_len + strlen(suffix) + 32;
  char* buf = malloc(bufsz);
  if (!buf) {
    set_err(err, errlen, "gguf: out of memory");
    return 1;
  }
  /* Any missing sibling means this is not a full split set: fall back. */
  for (uint64_t i = 1; i <= total; ++i) {
    snprintf(buf, bufsz, "%.*s-%05llu-of-%05llu%s", (int)pfx_len, path,
             (unsigned long long)i, (unsigned long long)total, suffix);
    if (access(buf, R_OK) != 0) {
      free(buf);
      return -1;
    }
  }

  int rc = 1;
  size_t loaded = 0, total_tensors = 0, w = 0;
  GgufFile* parts = calloc((size_t)total, sizeof(GgufFile));
  void** maps = calloc((size_t)total, sizeof(void*));
  size_t* sizes = calloc((size_t)total, sizeof(size_t));
  f->shards = calloc((size_t)(total - 1), sizeof(GgufShard));
  if (!parts || !maps || !sizes || !f->shards) {
    set_err(err, errlen, "gguf: out of memory");
    goto done;
  }

  for (uint64_t i = 0; i < total; ++i) {
    snprintf(buf, bufsz, "%.*s-%05llu-of-%05llu%s", (int)pfx_len, path,
             (unsigned long long)(i + 1), (unsigned long long)total, suffix);
    if (map_parse(buf, &maps[i], &sizes[i], &parts[i], err, errlen) != 0) goto done;
    loaded = (size_t)i + 1;
    /* Strict bounds: reject a tensor whose data would read past this shard's
     * mmap. Unknown-geometry types (nbytes==0) can't be checked and pass. */
    for (size_t t = 0; t < parts[i].n_tensors; ++t) {
      const GgufTensorInfo* ti = &parts[i].tensors[t];
      if (tensor_nbytes(ti) > sizes[i] - ti->offset) {
        set_err(err, errlen, "gguf: tensor data extends past end of shard");
        goto done;
      }
    }
    total_tensors += parts[i].n_tensors;
  }

  f->tensors = total_tensors ? calloc(total_tensors, sizeof(GgufTensorInfo)) : NULL;
  if (total_tensors && !f->tensors) {
    set_err(err, errlen, "gguf: out of memory");
    goto done;
  }
  /* shard 0 → primary map + metadata */
  f->map = maps[0];
  f->size = sizes[0];
  f->version = parts[0].version;
  f->alignment = parts[0].alignment;
  f->data_section_start = parts[0].data_section_start;
  f->kvs = parts[0].kvs;
  f->n_kv = parts[0].n_kv;
  parts[0].kvs = NULL; /* stolen: don't free in free_parse below */
  parts[0].n_kv = 0;
  for (uint64_t i = 1; i < total; ++i) {
    f->shards[i - 1].map = maps[i];
    f->shards[i - 1].size = sizes[i];
  }
  f->n_shards = (size_t)(total - 1);
  /* steal every shard's tensor infos (name + data pointer into its own map) */
  for (uint64_t i = 0; i < total; ++i) {
    for (size_t t = 0; t < parts[i].n_tensors; ++t) f->tensors[w++] = parts[i].tensors[t];
    free(parts[i].tensors);
    parts[i].tensors = NULL;
    parts[i].n_tensors = 0;
  }
  f->n_tensors = total_tensors;
  for (uint64_t i = 0; i < total; ++i) free_parse(&parts[i]); /* per-shard kvs/indexes */
  build_indexes(f);
  free(parts);
  free(maps);
  free(sizes);
  free(buf);
  return 0;

done: /* error: release everything loaded so far, leave *f empty */
  for (size_t i = 0; i < loaded; ++i) {
    free_parse(&parts[i]);
    if (maps[i]) munmap(maps[i], sizes[i]);
  }
  free(parts);
  free(maps);
  free(sizes);
  free(f->tensors);
  free(f->shards);
  free(buf);
  memset(f, 0, sizeof(*f));
  return rc;
}

int gguf_open(GgufFile* f, const char* path, char* err, size_t errlen) {
  memset(f, 0, sizeof(*f));
  size_t pfx_len, suf_off;
  uint64_t total;
  if (split_info(path, &pfx_len, &total, &suf_off) == 0) {
    int rc = open_shards(f, path, total, pfx_len, suf_off, err, errlen);
    if (rc == 0) return 0;
    if (rc > 0) return -1; /* real load error on an on-disk split set */
    memset(f, 0, sizeof(*f)); /* rc<0: incomplete set → single-file fallback */
  }
  void* map;
  size_t size;
  if (map_parse(path, &map, &size, f, err, errlen) != 0) {
    memset(f, 0, sizeof(*f));
    return -1;
  }
  f->map = map;
  f->size = size;
  return 0;
}

void gguf_close(GgufFile* f) {
  free_parse(f);
  if (f->map) munmap(f->map, f->size);
  for (size_t i = 0; i < f->n_shards; ++i)
    if (f->shards[i].map) munmap(f->shards[i].map, f->shards[i].size);
  free(f->shards);
  memset(f, 0, sizeof(*f));
}

const GgufValue* gguf_find(const GgufFile* f, const char* key) {
  if (f->kv_hash) {
    size_t idx = index_lookup(f->kv_hash, f->kv_hash_cap, f->kvs, kv_name, key);
    return idx ? &f->kvs[idx - 1].val : NULL;
  }
  /* index not built yet (alignment lookup during parse) or alloc failed */
  for (size_t i = 0; i < f->n_kv; ++i)
    if (strcmp(f->kvs[i].key, key) == 0) return &f->kvs[i].val;
  return NULL;
}

static bool value_as_u32(const GgufValue* v, uint32_t* out) {
  switch (v->kind) {
    case GGUF_T_BOOL:
    case GGUF_T_U8:
    case GGUF_T_U16:
    case GGUF_T_U32:
      *out = (uint32_t)v->v.u;
      return true;
    case GGUF_T_U64:
      if (v->v.u <= UINT32_MAX) { *out = (uint32_t)v->v.u; return true; }
      return false;
    case GGUF_T_I8:
    case GGUF_T_I16:
    case GGUF_T_I32:
    case GGUF_T_I64:
      if (v->v.i >= 0 && v->v.i <= (int64_t)UINT32_MAX) {
        *out = (uint32_t)v->v.i;
        return true;
      }
      return false;
    default:
      return false;
  }
}

bool gguf_get_u32(const GgufFile* f, const char* key, uint32_t* out) {
  const GgufValue* v = gguf_find(f, key);
  return v && value_as_u32(v, out);
}

bool gguf_get_f32(const GgufFile* f, const char* key, float* out) {
  const GgufValue* v = gguf_find(f, key);
  if (!v) return false;
  switch (v->kind) {
    case GGUF_T_F32:
    case GGUF_T_F64:
      *out = (float)v->v.f;
      return true;
    case GGUF_T_I8:
    case GGUF_T_I16:
    case GGUF_T_I32:
    case GGUF_T_I64:
      *out = (float)v->v.i;
      return true;
    case GGUF_T_U8:
    case GGUF_T_U16:
    case GGUF_T_U32:
    case GGUF_T_U64:
      *out = (float)v->v.u;
      return true;
    default:
      return false;
  }
}

char* gguf_get_str(const GgufFile* f, const char* key) {
  const GgufValue* v = gguf_find(f, key);
  if (!v || v->kind != GGUF_T_STRING) return NULL;
  char* s = malloc(v->v.str.len + 1);
  if (!s) return NULL;
  memcpy(s, v->v.str.ptr, v->v.str.len);
  s[v->v.str.len] = 0;
  return s;
}

const GgufValue* gguf_get_arr(const GgufFile* f, const char* key) {
  const GgufValue* v = gguf_find(f, key);
  return (v && v->kind == GGUF_T_ARRAY) ? v : NULL;
}

const GgufTensorInfo* gguf_tensor(const GgufFile* f, const char* name) {
  if (f->tensor_hash) {
    size_t idx = index_lookup(f->tensor_hash, f->tensor_hash_cap, f->tensors, tensor_name, name);
    return idx ? &f->tensors[idx - 1] : NULL;
  }
  for (size_t i = 0; i < f->n_tensors; ++i)
    if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
  return NULL;
}

char* gguf_architecture(const GgufFile* f) {
  return gguf_get_str(f, "general.architecture");
}

#ifdef GGUF_SELFTEST
/* Standalone checks for the P7a work (split shards, hash index, bounds). Not
 * wired into `make test` (owns only gguf.c/.h); build + run with:
 *   gcc -DGGUF_SELFTEST -std=c11 -D_DEFAULT_SOURCE src/gguf.c -o /tmp/gst && /tmp/gst
 */
#include <assert.h>

/* Build one single-file GGUF into a malloc'd buffer: optional arch string, then
 * one F32 tensor `tname` of `n` elements followed by `data_bytes` of real data
 * (pass n*4 for a well-formed tensor, or fewer to overrun for the bounds test). */
static uint8_t* st_build(const char* arch, const char* tname, uint64_t n,
                         size_t data_bytes, size_t* out_len) {
  uint8_t* b = NULL;
  size_t len = 0;
#define PUT(p, k)                          \
  do {                                     \
    b = realloc(b, len + (k));             \
    assert(b);                             \
    memcpy(b + len, (p), (k));             \
    len += (k);                            \
  } while (0)
#define PUTU32(v)                          \
  do {                                     \
    uint32_t _v = (uint32_t)(v);           \
    PUT(&_v, 4);                           \
  } while (0)
#define PUTU64(v)                          \
  do {                                     \
    uint64_t _v = (uint64_t)(v);           \
    PUT(&_v, 8);                           \
  } while (0)
#define PUTSTR(s)                          \
  do {                                     \
    size_t _l = strlen(s);                 \
    PUTU64(_l);                            \
    PUT((s), _l);                          \
  } while (0)
  PUT("GGUF", 4);
  PUTU32(3);
  PUTU64(1);              /* tensor count */
  PUTU64(arch ? 1u : 0u); /* kv count */
  if (arch) {
    PUTSTR("general.architecture");
    PUTU32(GGUF_T_STRING);
    PUTSTR(arch);
  }
  PUTSTR(tname);
  PUTU32(1);   /* n_dims */
  PUTU64(n);   /* dims[0] */
  PUTU32(0);   /* F32 */
  PUTU64(0);   /* relative offset */
  while (len % 32) {
    uint8_t z = 0;
    PUT(&z, 1);
  }
  for (size_t i = 0; i < data_bytes; ++i) {
    uint8_t v = (uint8_t)(i + (tname[0] << 1));
    PUT(&v, 1);
  }
#undef PUT
#undef PUTU32
#undef PUTU64
#undef PUTSTR
  *out_len = len;
  return b;
}

static void st_write(const char* path, const uint8_t* b, size_t len) {
  FILE* fp = fopen(path, "wb");
  assert(fp);
  assert(fwrite(b, 1, len, fp) == len);
  fclose(fp);
}

int main(void) {
  char dir[] = "/tmp/gguf_selftest_XXXXXX";
  assert(mkdtemp(dir));
  char p1[256], p2[256];
  snprintf(p1, sizeof p1, "%s/model-00001-of-00002.gguf", dir);
  snprintf(p2, sizeof p2, "%s/model-00002-of-00002.gguf", dir);

  /* 1. Two-shard round-trip: shard 0 carries the metadata + tensor.a, shard 1
   *    carries tensor.b. Both must be found from the unified table. */
  size_t l1, l2;
  uint8_t* b1 = st_build("split-test", "tensor.a", 8, 32, &l1);
  uint8_t* b2 = st_build(NULL, "tensor.b", 5, 20, &l2);
  st_write(p1, b1, l1);
  st_write(p2, b2, l2);

  GgufFile g;
  char err[256];
  assert(gguf_open(&g, p1, err, sizeof err) == 0);
  assert(g.n_tensors == 2);
  assert(g.n_shards == 1); /* one extra shard beyond the primary map */
  char* arch = gguf_architecture(&g);
  assert(arch && strcmp(arch, "split-test") == 0);
  free(arch);
  const GgufTensorInfo* ta = gguf_tensor(&g, "tensor.a");
  const GgufTensorInfo* tb = gguf_tensor(&g, "tensor.b");
  assert(ta && tb);
  /* each tensor's data points into its own shard's mmap and is readable */
  assert(ta->data[0] == (uint8_t)('t' << 1));
  assert(tb->data[0] == (uint8_t)('t' << 1));
  assert(ta->data != tb->data);
  /* opening via any shard yields the same set (metadata still from shard 0) */
  GgufFile g2;
  assert(gguf_open(&g2, p2, err, sizeof err) == 0);
  assert(g2.n_tensors == 2 && gguf_tensor(&g2, "tensor.a") && gguf_tensor(&g2, "tensor.b"));
  char* a2 = gguf_architecture(&g2);
  assert(a2 && strcmp(a2, "split-test") == 0);
  free(a2);
  gguf_close(&g2);

  /* 2. Hash index == linear scan for every tensor and every kv. */
  for (size_t i = 0; i < g.n_tensors; ++i) {
    const char* nm = g.tensors[i].name;
    const GgufTensorInfo* viahash = gguf_tensor(&g, nm); /* uses the index */
    const GgufTensorInfo* vialin = NULL;
    for (size_t j = 0; j < g.n_tensors; ++j)
      if (strcmp(g.tensors[j].name, nm) == 0) { vialin = &g.tensors[j]; break; }
    assert(viahash == vialin);
  }
  for (size_t i = 0; i < g.n_kv; ++i) {
    const char* k = g.kvs[i].key;
    const GgufValue* viahash = gguf_find(&g, k);
    const GgufValue* vialin = NULL;
    for (size_t j = 0; j < g.n_kv; ++j)
      if (strcmp(g.kvs[j].key, k) == 0) { vialin = &g.kvs[j].val; break; }
    assert(viahash == vialin);
  }
  assert(gguf_tensor(&g, "tensor.absent") == NULL); /* miss returns NULL */
  assert(gguf_find(&g, "no.such.key") == NULL);
  gguf_close(&g);
  free(b1);
  free(b2);

  /* 3a. A shard whose tensor's data runs past its mmap is rejected. Same shard 0
   *     but shard 1 declares 100 F32 (400 B) with only 8 B of data present. */
  uint8_t* r1 = st_build("split-test", "tensor.a", 8, 32, &l1);
  uint8_t* r2 = st_build(NULL, "tensor.b", 100, 8, &l2); /* 400 B claimed, 8 present */
  st_write(p1, r1, l1);
  st_write(p2, r2, l2);
  assert(gguf_open(&g, p1, err, sizeof err) != 0);
  assert(strstr(err, "past end of shard"));
  free(r1);
  free(r2);
  unlink(p1);
  unlink(p2);
  rmdir(dir);

  /* 3b. A single-file tensor whose declared geometry overflows a u64 is rejected
   *     by the universal overflow guard (in-memory parse, no mmap needed). */
  size_t lo;
  uint8_t* ob = st_build(NULL, "big", (uint64_t)1 << 62, 0, &lo); /* *4 overflows */
  assert(gguf_parse(&g, ob, lo, err, sizeof err) != 0);
  assert(strstr(err, "overflow"));
  free(ob);

  printf("gguf selftest: split shards + hash==linear + bounds/overflow OK\n");
  return 0;
}
#endif /* GGUF_SELFTEST */
