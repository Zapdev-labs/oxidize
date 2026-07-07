/* GGUF v2/v3 mmap parser. Port of oxidize-cpp/src/gguf.cpp minus split-shard
 * and HF tensor-name mapping (GGUF-native "blk.*" names only). Nested arrays
 * are skipped; scalar/string arrays are kept (tokenizer needs them). */
#include "oc.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  const uint8_t *p;
  size_t len, cur;
} rdr;

static const uint8_t *take(rdr *r, size_t n) {
  if (r->cur > r->len || n > r->len - r->cur) oc_die("gguf: unexpected EOF");
  const uint8_t *out = r->p + r->cur;
  r->cur += n;
  return out;
}
static uint8_t r8(rdr *r) { return take(r, 1)[0]; }
static uint16_t r16(rdr *r) {
  const uint8_t *p = take(r, 2);
  return (uint16_t)(p[0] | (p[1] << 8));
}
static uint32_t r32(rdr *r) {
  const uint8_t *p = take(r, 4);
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}
static uint64_t r64(rdr *r) {
  const uint8_t *p = take(r, 8);
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
  return v;
}
static char *rstr(rdr *r) {
  uint64_t n = r64(r);
  const uint8_t *p = take(r, (size_t)n);
  char *s = malloc((size_t)n + 1);
  memcpy(s, p, (size_t)n);
  s[n] = 0;
  return s;
}

/* read one scalar of gguf metadata type t, widened to double */
static double rnum(rdr *r, uint32_t t) {
  switch (t) {
    case 0: return (double)r8(r);                 /* u8 */
    case 1: return (double)(int8_t)r8(r);
    case 2: return (double)r16(r);
    case 3: return (double)(int16_t)r16(r);
    case 4: return (double)r32(r);
    case 5: return (double)(int32_t)r32(r);
    case 6: { uint32_t b = r32(r); float f; memcpy(&f, &b, 4); return (double)f; }
    case 7: return (double)(r8(r) != 0);          /* bool */
    case 10: return (double)r64(r);
    case 11: return (double)(int64_t)r64(r);
    case 12: { uint64_t b = r64(r); double f; memcpy(&f, &b, 8); return f; }
    default: oc_die("gguf: unexpected numeric metadata type %u", t);
  }
  return 0;
}

static void skip_meta_value(rdr *r, uint32_t t) {
  if (t == 8) {
    uint64_t n = r64(r);
    (void)take(r, (size_t)n);
    return;
  }
  if (t == 9) {
    uint32_t et = r32(r);
    uint64_t n = r64(r);
    for (uint64_t i = 0; i < n; ++i) skip_meta_value(r, et);
    return;
  }
  (void)rnum(r, t);
}

oc_gguf *oc_gguf_load(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) oc_die("gguf: cannot open %s", path);
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size == 0) oc_die("gguf: cannot stat %s", path);
  size_t size = (size_t)st.st_size;
  void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) oc_die("gguf: mmap failed for %s", path);
  madvise(map, size, MADV_WILLNEED);

  oc_gguf *g = calloc(1, sizeof(*g));
  g->map = map;
  g->map_size = size;
  g->base = (const uint8_t *)map;

  rdr r = {g->base, size, 0};
  if (memcmp(take(&r, 4), "GGUF", 4) != 0) oc_die("gguf: bad magic");
  uint32_t version = r32(&r);
  if (version != 2 && version != 3) oc_die("gguf: unsupported version %u", version);
  g->version = version;
  uint64_t n_tensors = r64(&r), n_meta = r64(&r);
  g->kv_off = r.cur;

  g->meta = calloc((size_t)n_meta, sizeof(oc_meta));
  g->n_meta = (size_t)n_meta;
  for (uint64_t m = 0; m < n_meta; ++m) {
    oc_meta *e = &g->meta[m];
    e->key = rstr(&r);
    uint32_t t = r32(&r);
    if (t == 8) {
      e->kind = 1;
      e->str = rstr(&r);
    } else if (t == 9) {
      e->kind = 2;
      uint32_t et = r32(&r);
      uint64_t n = r64(&r);
      e->count = (size_t)n;
      if (et == 8) {
        e->is_str = true;
        e->strs = malloc((size_t)n * sizeof(char *));
        for (uint64_t i = 0; i < n; ++i) e->strs[i] = rstr(&r);
      } else if (et == 9) {
        for (uint64_t i = 0; i < n; ++i) skip_meta_value(&r, et);
      } else {
        e->nums = malloc((size_t)n * sizeof(double));
        for (uint64_t i = 0; i < n; ++i) e->nums[i] = rnum(&r, et);
      }
    } else {
      e->kind = 0;
      e->num = rnum(&r, t);
    }
  }

  g->kv_end = r.cur;
  g->tensors = calloc((size_t)n_tensors, sizeof(oc_tensor_info));
  g->n_tensors = (size_t)n_tensors;
  for (uint64_t t = 0; t < n_tensors; ++t) {
    oc_tensor_info *ti = &g->tensors[t];
    ti->name = rstr(&r);
    ti->n_dims = r32(&r);
    if (ti->n_dims > 4) oc_die("gguf: tensor %s has %u dims", ti->name, ti->n_dims);
    for (uint32_t d = 0; d < ti->n_dims; ++d) ti->dims[d] = r64(&r);
    uint32_t ggml_type = r32(&r);
    ti->ggml_type = ggml_type;
    ti->quant = oc_from_ggml_type(ggml_type);
    ti->offset = r64(&r); /* relative; resolved below */
  }

  uint64_t align = 32;
  {
    const oc_meta *a = oc_meta_get(g, "general.alignment");
    if (a && a->kind == 0 && a->num > 0) align = (uint64_t)a->num;
  }
  if (align == 0 || (align & (align - 1)) != 0) oc_die("gguf: bad alignment %llu",
                                                       (unsigned long long)align);
  g->align = align;
  uint64_t data_start = ((uint64_t)r.cur + align - 1) & ~(align - 1);
  if (data_start > size) oc_die("gguf: unexpected EOF");
  for (size_t t = 0; t < g->n_tensors; ++t) {
    if (g->tensors[t].offset > UINT64_MAX - data_start)
      oc_die("gguf: tensor offset overflow");
    g->tensors[t].offset += data_start;
    if (g->tensors[t].offset > size) oc_die("gguf: tensor offset out of range");
  }
  return g;
}

void oc_gguf_free(oc_gguf *g) {
  if (!g) return;
  for (size_t i = 0; i < g->n_meta; ++i) {
    free(g->meta[i].key);
    free(g->meta[i].str);
    if (g->meta[i].strs) {
      for (size_t j = 0; j < g->meta[i].count; ++j) free(g->meta[i].strs[j]);
      free(g->meta[i].strs);
    }
    free(g->meta[i].nums);
  }
  free(g->meta);
  for (size_t i = 0; i < g->n_tensors; ++i) free(g->tensors[i].name);
  free(g->tensors);
  if (g->map) munmap(g->map, g->map_size);
  free(g);
}

const oc_meta *oc_meta_get(const oc_gguf *g, const char *key) {
  for (size_t i = 0; i < g->n_meta; ++i)
    if (strcmp(g->meta[i].key, key) == 0) return &g->meta[i];
  return NULL;
}

bool oc_meta_u32(const oc_gguf *g, const char *key, uint32_t *out) {
  const oc_meta *m = oc_meta_get(g, key);
  if (!m || m->kind != 0 || m->num < 0) return false;
  *out = (uint32_t)m->num;
  return true;
}

bool oc_meta_f32(const oc_gguf *g, const char *key, float *out) {
  const oc_meta *m = oc_meta_get(g, key);
  if (!m || m->kind != 0) return false;
  *out = (float)m->num;
  return true;
}

const char *oc_meta_str(const oc_gguf *g, const char *key) {
  const oc_meta *m = oc_meta_get(g, key);
  return m && m->kind == 1 ? m->str : NULL;
}

const oc_tensor_info *oc_find_tensor(const oc_gguf *g, const char *name) {
  for (size_t i = 0; i < g->n_tensors; ++i)
    if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
  return NULL;
}
