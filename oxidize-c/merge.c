/* oc-merge: SafeTensors checkpoint merger (linear / SLERP).
 *
 * C11 port of oxidize-merge (Rust). Dependency-free, POSIX mmap, little-endian
 * hosts only. Merges two HuggingFace SafeTensors checkpoints (single file,
 * sharded directory with model.safetensors.index.json, or plain directory of
 * *.safetensors) with per-tensor-category blend weights.
 *
 * Semantics mirror oxidize-merge/src/{merge,blend,index,recipe,writer}.rs:
 *   - blendable dtypes: F32 / F16 / BF16; everything else copies from A
 *   - SLERP treats the whole tensor as one vector, f64 accumulation,
 *     linear fallback for tiny / antipodal angles
 *   - missing-tensor policy: error | a | b
 *   - sharded output: model-NNNNN-of-?????.safetensors renamed at finish,
 *     plus model.safetensors.index.json with merged metadata
 *
 * Build: make merge   (produces ./oc-merge)
 * Test:  ./oc-merge --self-test
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* ---------------------------------------------------------------- fatal -- */

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "error: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) die("out of memory (%zu bytes)", n);
  return p;
}
static void *xrealloc(void *p, size_t n) {
  void *q = realloc(p, n ? n : 1);
  if (!q) die("out of memory (%zu bytes)", n);
  return q;
}
static char *xstrdup(const char *s) {
  char *d = xmalloc(strlen(s) + 1);
  strcpy(d, s);
  return d;
}

/* ------------------------------------------------------- f16 conversions -- */

static float f16_to_f32(uint16_t bits) {
  uint32_t sign = (bits >> 15) & 1;
  uint32_t exp = (bits >> 10) & 0x1f;
  uint32_t frac = bits & 0x3ff;
  uint32_t out;
  if (exp == 0) {
    if (frac == 0) {
      out = sign << 31;
    } else {
      int e = -1;
      uint32_t f = frac;
      while ((f & 0x400) == 0) {
        f <<= 1;
        e -= 1;
      }
      f &= 0x3ff;
      out = (sign << 31) | ((uint32_t)(127 - 15 + 1 + e) << 23) | (f << 13);
    }
  } else if (exp == 0x1f) {
    out = (sign << 31) | (0xffu << 23) | (frac << 13);
  } else {
    out = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
  }
  float r;
  memcpy(&r, &out, 4);
  return r;
}

static uint16_t f32_to_f16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, 4);
  uint16_t sign = (uint16_t)((bits >> 31) & 1);
  int exp = (int)((bits >> 23) & 0xff);
  uint32_t frac = bits & 0x7fffff;
  if (exp == 255)
    return (uint16_t)((sign << 15) | (0x1f << 10) | ((frac != 0 ? 1 : 0) << 9));
  int new_exp = exp - 127 + 15;
  uint32_t new_frac = frac >> 13;
  if (new_exp <= 0) {
    if (new_exp < -10) return (uint16_t)(sign << 15);
    new_frac |= 0x400;
    new_frac >>= 1 - new_exp;
    return (uint16_t)((sign << 15) | new_frac);
  }
  if (new_exp >= 0x1f) return (uint16_t)((sign << 15) | (0x1f << 10));
  if (((frac >> 12) & 1) == 1 && ((frac & 0xfff) != 0 || (new_frac & 1) == 1)) {
    new_frac += 1;
    if (new_frac == 0x400) {
      new_frac = 0;
      new_exp += 1;
      if (new_exp >= 0x1f) return (uint16_t)((sign << 15) | (0x1f << 10));
    }
  }
  return (uint16_t)((sign << 15) | ((uint32_t)new_exp << 10) | new_frac);
}

/* ------------------------------------------------------------ JSON parse -- */
/* Minimal targeted parser for safetensors headers and HF weight indexes. */

typedef struct {
  const char *p;
  const char *end;
  const char *what; /* for error messages */
} Json;

static void js_ws(Json *j) {
  while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r'))
    j->p++;
}
static void js_expect(Json *j, char c) {
  js_ws(j);
  if (j->p >= j->end || *j->p != c) die("%s: malformed JSON (expected '%c')", j->what, c);
  j->p++;
}
static bool js_peek(Json *j, char c) {
  js_ws(j);
  return j->p < j->end && *j->p == c;
}

/* Parse a JSON string, returning a malloc'd UTF-8 C string. */
static char *js_string(Json *j) {
  js_expect(j, '"');
  size_t cap = 32, n = 0;
  char *out = xmalloc(cap);
  while (j->p < j->end && *j->p != '"') {
    char c = *j->p++;
    if (c == '\\') {
      if (j->p >= j->end) die("%s: truncated escape", j->what);
      char e = *j->p++;
      switch (e) {
        case '"': c = '"'; break;
        case '\\': c = '\\'; break;
        case '/': c = '/'; break;
        case 'b': c = '\b'; break;
        case 'f': c = '\f'; break;
        case 'n': c = '\n'; break;
        case 'r': c = '\r'; break;
        case 't': c = '\t'; break;
        case 'u': {
          if (j->end - j->p < 4) die("%s: truncated \\u escape", j->what);
          unsigned cp = 0;
          for (int i = 0; i < 4; i++) {
            char h = *j->p++;
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
            else die("%s: bad \\u escape", j->what);
          }
          /* Encode BMP code point as UTF-8 (surrogate pairs unsupported;
           * tensor names / metadata never need them). */
          if (n + 4 >= cap) { cap *= 2; out = xrealloc(out, cap); }
          if (cp < 0x80) {
            out[n++] = (char)cp;
          } else if (cp < 0x800) {
            out[n++] = (char)(0xc0 | (cp >> 6));
            out[n++] = (char)(0x80 | (cp & 0x3f));
          } else {
            out[n++] = (char)(0xe0 | (cp >> 12));
            out[n++] = (char)(0x80 | ((cp >> 6) & 0x3f));
            out[n++] = (char)(0x80 | (cp & 0x3f));
          }
          continue;
        }
        default: die("%s: bad escape '\\%c'", j->what, e);
      }
    }
    if (n + 2 >= cap) { cap *= 2; out = xrealloc(out, cap); }
    out[n++] = c;
  }
  if (j->p >= j->end) die("%s: unterminated string", j->what);
  j->p++; /* closing quote */
  out[n] = 0;
  return out;
}

static uint64_t js_u64(Json *j) {
  js_ws(j);
  if (j->p >= j->end || *j->p < '0' || *j->p > '9')
    die("%s: expected number", j->what);
  uint64_t v = 0;
  while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
    v = v * 10 + (uint64_t)(*j->p - '0');
    j->p++;
  }
  return v;
}

static void js_skip_value(Json *j) {
  js_ws(j);
  if (j->p >= j->end) die("%s: truncated JSON", j->what);
  char c = *j->p;
  if (c == '"') {
    free(js_string(j));
  } else if (c == '{') {
    j->p++;
    js_ws(j);
    if (js_peek(j, '}')) { j->p++; return; }
    for (;;) {
      free(js_string(j));
      js_expect(j, ':');
      js_skip_value(j);
      js_ws(j);
      if (js_peek(j, ',')) { j->p++; continue; }
      js_expect(j, '}');
      return;
    }
  } else if (c == '[') {
    j->p++;
    if (js_peek(j, ']')) { j->p++; return; }
    for (;;) {
      js_skip_value(j);
      if (js_peek(j, ',')) { j->p++; continue; }
      js_expect(j, ']');
      return;
    }
  } else {
    /* number / true / false / null */
    while (j->p < j->end && *j->p != ',' && *j->p != '}' && *j->p != ']' &&
           *j->p != ' ' && *j->p != '\n' && *j->p != '\r' && *j->p != '\t')
      j->p++;
  }
}

/* ------------------------------------------------------ string-map (kv) -- */

typedef struct {
  char **keys;
  char **vals;
  size_t n, cap;
} KvMap;

static const char *kv_get(const KvMap *m, const char *key) {
  for (size_t i = 0; i < m->n; i++)
    if (strcmp(m->keys[i], key) == 0) return m->vals[i];
  return NULL;
}

/* Insert, overwriting an existing key (Rust BTreeMap::insert semantics). */
static void kv_put(KvMap *m, char *key, char *val) {
  for (size_t i = 0; i < m->n; i++) {
    if (strcmp(m->keys[i], key) == 0) {
      free(m->vals[i]);
      m->vals[i] = val;
      free(key);
      return;
    }
  }
  if (m->n == m->cap) {
    m->cap = m->cap ? m->cap * 2 : 8;
    m->keys = xrealloc(m->keys, m->cap * sizeof(char *));
    m->vals = xrealloc(m->vals, m->cap * sizeof(char *));
  }
  m->keys[m->n] = key;
  m->vals[m->n] = val;
  m->n++;
}

/* Merge per-shard metadata, erroring on conflicting values for the same key. */
static void kv_merge_strict(KvMap *into, const KvMap *from, const char *ctx) {
  for (size_t i = 0; i < from->n; i++) {
    const char *existing = kv_get(into, from->keys[i]);
    if (existing && strcmp(existing, from->vals[i]) != 0)
      die("%s: conflicting metadata for key \"%s\": \"%s\" vs \"%s\"", ctx,
          from->keys[i], existing, from->vals[i]);
    if (!existing)
      kv_put(into, xstrdup(from->keys[i]), xstrdup(from->vals[i]));
  }
}

static void kv_sort(KvMap *m) {
  /* insertion-sort by key so output JSON is deterministic like BTreeMap */
  for (size_t i = 1; i < m->n; i++) {
    char *k = m->keys[i], *v = m->vals[i];
    size_t jx = i;
    while (jx > 0 && strcmp(m->keys[jx - 1], k) > 0) {
      m->keys[jx] = m->keys[jx - 1];
      m->vals[jx] = m->vals[jx - 1];
      jx--;
    }
    m->keys[jx] = k;
    m->vals[jx] = v;
  }
}

/* ------------------------------------------------------------ model index -- */

typedef struct {
  char *name;
  char *dtype;      /* dtype string as it appears in the header, e.g. "BF16" */
  uint64_t *shape;
  size_t shape_n;
  size_t shard;     /* index into Model.shards */
  uint64_t begin;   /* offsets relative to shard data section */
  uint64_t end;
} TensorRef;

typedef struct {
  char *path;
  uint8_t *map;     /* whole file mmap */
  size_t map_len;
  size_t data_off;  /* 8 + header_len */
} Shard;

typedef struct {
  Shard *shards;
  size_t shards_n, shards_cap;
  TensorRef *tensors;
  size_t tensors_n, tensors_cap;
  KvMap metadata;
} Model;

static bool is_blendable(const char *dtype) {
  return strcmp(dtype, "F32") == 0 || strcmp(dtype, "F16") == 0 ||
         strcmp(dtype, "BF16") == 0;
}

static int tensor_cmp(const void *a, const void *b) {
  return strcmp(((const TensorRef *)a)->name, ((const TensorRef *)b)->name);
}

static TensorRef *model_find(Model *m, const char *name) {
  TensorRef key = {.name = (char *)name};
  return bsearch(&key, m->tensors, m->tensors_n, sizeof(TensorRef), tensor_cmp);
}

static const uint8_t *tensor_bytes(const Model *m, const TensorRef *t, size_t *len) {
  const Shard *s = &m->shards[t->shard];
  *len = (size_t)(t->end - t->begin);
  return s->map + s->data_off + t->begin;
}

static void model_push_tensor(Model *m, TensorRef t) {
  if (m->tensors_n == m->tensors_cap) {
    m->tensors_cap = m->tensors_cap ? m->tensors_cap * 2 : 256;
    m->tensors = xrealloc(m->tensors, m->tensors_cap * sizeof(TensorRef));
  }
  m->tensors[m->tensors_n++] = t;
}

/* mmap a shard, parse its safetensors header into `m`, return shard index. */
static size_t model_open_shard(Model *m, const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) die("failed to open %s: %s", path, strerror(errno));
  struct stat st;
  if (fstat(fd, &st) != 0) die("failed to stat %s: %s", path, strerror(errno));
  if (st.st_size < 8) die("%s: too small for a safetensors file", path);
  uint8_t *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) die("failed to mmap %s: %s", path, strerror(errno));
  close(fd);

  uint64_t header_len;
  memcpy(&header_len, map, 8); /* little-endian host assumed */
  if (8 + header_len > (uint64_t)st.st_size)
    die("%s: header length %llu exceeds file size", path, (unsigned long long)header_len);

  if (m->shards_n == m->shards_cap) {
    m->shards_cap = m->shards_cap ? m->shards_cap * 2 : 8;
    m->shards = xrealloc(m->shards, m->shards_cap * sizeof(Shard));
  }
  size_t shard_idx = m->shards_n++;
  m->shards[shard_idx] = (Shard){.path = xstrdup(path),
                                 .map = map,
                                 .map_len = (size_t)st.st_size,
                                 .data_off = (size_t)(8 + header_len)};

  Json j = {.p = (const char *)map + 8,
            .end = (const char *)map + 8 + header_len,
            .what = path};
  KvMap shard_meta = {0};
  js_expect(&j, '{');
  if (js_peek(&j, '}')) {
    j.p++;
  } else {
    for (;;) {
      char *key = js_string(&j);
      js_expect(&j, ':');
      if (strcmp(key, "__metadata__") == 0) {
        free(key);
        js_expect(&j, '{');
        if (js_peek(&j, '}')) {
          j.p++;
        } else {
          for (;;) {
            char *mk = js_string(&j);
            js_expect(&j, ':');
            if (js_peek(&j, '"')) {
              kv_put(&shard_meta, mk, js_string(&j));
            } else {
              free(mk);
              js_skip_value(&j);
            }
            if (js_peek(&j, ',')) { j.p++; continue; }
            js_expect(&j, '}');
            break;
          }
        }
      } else {
        TensorRef t = {.name = key, .shard = shard_idx};
        js_expect(&j, '{');
        for (;;) {
          char *fk = js_string(&j);
          js_expect(&j, ':');
          if (strcmp(fk, "dtype") == 0) {
            t.dtype = js_string(&j);
          } else if (strcmp(fk, "shape") == 0) {
            js_expect(&j, '[');
            size_t cap = 4;
            t.shape = xmalloc(cap * sizeof(uint64_t));
            t.shape_n = 0;
            if (js_peek(&j, ']')) {
              j.p++;
            } else {
              for (;;) {
                if (t.shape_n == cap) {
                  cap *= 2;
                  t.shape = xrealloc(t.shape, cap * sizeof(uint64_t));
                }
                t.shape[t.shape_n++] = js_u64(&j);
                if (js_peek(&j, ',')) { j.p++; continue; }
                js_expect(&j, ']');
                break;
              }
            }
          } else if (strcmp(fk, "data_offsets") == 0) {
            js_expect(&j, '[');
            t.begin = js_u64(&j);
            js_expect(&j, ',');
            t.end = js_u64(&j);
            js_expect(&j, ']');
          } else {
            js_skip_value(&j);
          }
          free(fk);
          if (js_peek(&j, ',')) { j.p++; continue; }
          js_expect(&j, '}');
          break;
        }
        if (!t.dtype) die("%s: tensor %s missing dtype", path, t.name);
        if (t.end < t.begin || 8 + header_len + t.end > (uint64_t)st.st_size)
          die("%s: tensor %s data_offsets out of range", path, t.name);
        model_push_tensor(m, t);
      }
      if (js_peek(&j, ',')) { j.p++; continue; }
      js_expect(&j, '}');
      break;
    }
  }
  kv_merge_strict(&m->metadata, &shard_meta, path);
  for (size_t i = 0; i < shard_meta.n; i++) {
    free(shard_meta.keys[i]);
    free(shard_meta.vals[i]);
  }
  free(shard_meta.keys);
  free(shard_meta.vals);
  return shard_idx;
}

/* Reject shard names that are not a plain file name (path traversal guard). */
static void validate_shard_name(const char *name) {
  if (!*name || strchr(name, '/') || strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
    die("invalid shard name \"%s\" in weight index (must be a plain file name)", name);
}

static char *path_join(const char *dir, const char *name) {
  size_t dl = strlen(dir);
  char *p = xmalloc(dl + strlen(name) + 2);
  sprintf(p, "%s%s%s", dir, (dl && dir[dl - 1] == '/') ? "" : "/", name);
  return p;
}

static int str_ptr_cmp(const void *a, const void *b) {
  return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void model_open(Model *m, const char *path) {
  memset(m, 0, sizeof(*m));
  struct stat st;
  if (stat(path, &st) != 0)
    die("model path %s does not exist: %s", path, strerror(errno));

  if (S_ISREG(st.st_mode)) {
    model_open_shard(m, path);
    qsort(m->tensors, m->tensors_n, sizeof(TensorRef), tensor_cmp);
    return;
  }
  if (!S_ISDIR(st.st_mode))
    die("model path %s is neither a file nor a directory", path);

  /* Collect *.safetensors.index.json candidates and *.safetensors files. */
  DIR *d = opendir(path);
  if (!d) die("failed to open directory %s: %s", path, strerror(errno));
  char *index_path = NULL;
  char **shard_files = NULL;
  size_t shard_n = 0, shard_cap = 0;
  struct dirent *de;
  while ((de = readdir(d))) {
    const char *n = de->d_name;
    size_t len = strlen(n);
    if (len > 23 && strcmp(n + len - 23, ".safetensors.index.json") == 0) {
      /* keep the lexicographically first index, like the Rust port */
      if (!index_path || strcmp(n, strrchr(index_path, '/') + 1) < 0) {
        free(index_path);
        index_path = path_join(path, n);
      }
    } else if (len > 12 && strcmp(n + len - 12, ".safetensors") == 0) {
      if (shard_n == shard_cap) {
        shard_cap = shard_cap ? shard_cap * 2 : 8;
        shard_files = xrealloc(shard_files, shard_cap * sizeof(char *));
      }
      shard_files[shard_n++] = xstrdup(n);
    }
  }
  closedir(d);

  if (index_path) {
    for (size_t i = 0; i < shard_n; i++) free(shard_files[i]);
    free(shard_files);

    FILE *f = fopen(index_path, "rb");
    if (!f) die("failed to read %s: %s", index_path, strerror(errno));
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *raw = xmalloc((size_t)len + 1);
    if (fread(raw, 1, (size_t)len, f) != (size_t)len)
      die("failed to read %s", index_path);
    fclose(f);
    raw[len] = 0;

    Json j = {.p = raw, .end = raw + len, .what = index_path};
    KvMap opened = {0}; /* shard names already mmapped */
    char **wm_tensor = NULL, **wm_shard = NULL;
    size_t wm_n = 0, wm_cap = 0;

    js_expect(&j, '{');
    if (!js_peek(&j, '}')) {
      for (;;) {
        char *key = js_string(&j);
        js_expect(&j, ':');
        if (strcmp(key, "metadata") == 0) {
          js_expect(&j, '{');
          if (js_peek(&j, '}')) {
            j.p++;
          } else {
            for (;;) {
              char *mk = js_string(&j);
              js_expect(&j, ':');
              if (js_peek(&j, '"')) kv_put(&m->metadata, mk, js_string(&j));
              else { free(mk); js_skip_value(&j); }
              if (js_peek(&j, ',')) { j.p++; continue; }
              js_expect(&j, '}');
              break;
            }
          }
        } else if (strcmp(key, "weight_map") == 0) {
          js_expect(&j, '{');
          if (js_peek(&j, '}')) {
            j.p++;
          } else {
            for (;;) {
              char *tn = js_string(&j);
              js_expect(&j, ':');
              if (!js_peek(&j, '"'))
                die("%s: weight_map entry for %s is not a string", index_path, tn);
              char *sn = js_string(&j);
              if (wm_n == wm_cap) {
                wm_cap = wm_cap ? wm_cap * 2 : 256;
                wm_tensor = xrealloc(wm_tensor, wm_cap * sizeof(char *));
                wm_shard = xrealloc(wm_shard, wm_cap * sizeof(char *));
              }
              wm_tensor[wm_n] = tn;
              wm_shard[wm_n] = sn;
              wm_n++;
              if (js_peek(&j, ',')) { j.p++; continue; }
              js_expect(&j, '}');
              break;
            }
          }
        } else {
          js_skip_value(&j);
        }
        free(key);
        if (js_peek(&j, ',')) { j.p++; continue; }
        js_expect(&j, '}');
        break;
      }
    } else {
      j.p++;
    }
    if (wm_n == 0) die("%s: weight index missing weight_map", index_path);

    /* Open each referenced shard once. Shard headers already carry full
     * tensor info; the weight_map just tells us which shards to open and
     * which tensors must exist. */
    for (size_t i = 0; i < wm_n; i++) {
      const char *sn = wm_shard[i];
      bool have = kv_get(&opened, sn) != NULL;
      if (!have) {
        validate_shard_name(sn);
        char *sp = path_join(path, sn);
        model_open_shard(m, sp);
        free(sp);
        kv_put(&opened, xstrdup(sn), xstrdup(""));
      }
    }
    qsort(m->tensors, m->tensors_n, sizeof(TensorRef), tensor_cmp);
    for (size_t i = 0; i < wm_n; i++) {
      if (!model_find(m, wm_tensor[i]))
        die("tensor %s missing from shard %s", wm_tensor[i], wm_shard[i]);
      free(wm_tensor[i]);
      free(wm_shard[i]);
    }
    free(wm_tensor);
    free(wm_shard);
    for (size_t i = 0; i < opened.n; i++) { free(opened.keys[i]); free(opened.vals[i]); }
    free(opened.keys);
    free(opened.vals);
    free(raw);
    free(index_path);
    return;
  }

  if (shard_n == 0) die("no .safetensors files found in %s", path);
  qsort(shard_files, shard_n, sizeof(char *), str_ptr_cmp);
  for (size_t i = 0; i < shard_n; i++) {
    size_t before = m->tensors_n;
    char *sp = path_join(path, shard_files[i]);
    model_open_shard(m, sp);
    free(sp);
    /* duplicate-tensor check across shards */
    for (size_t k = before; k < m->tensors_n; k++) {
      for (size_t l = 0; l < before; l++) {
        if (strcmp(m->tensors[k].name, m->tensors[l].name) == 0)
          die("duplicate tensor %s in directory %s", m->tensors[k].name, path);
      }
    }
    free(shard_files[i]);
  }
  free(shard_files);
  qsort(m->tensors, m->tensors_n, sizeof(TensorRef), tensor_cmp);
}

/* ---------------------------------------------------------------- recipe -- */

typedef struct {
  float attention_t, mlp_t, other_t;
  float default_t;
  bool has_default;
} Recipe;

static bool contains_ci(const char *haystack_lower, const char *needle) {
  return strstr(haystack_lower, needle) != NULL;
}

static float t_for_tensor(const Recipe *r, const char *name) {
  if (r->has_default) return r->default_t;
  char lower[512];
  size_t n = strlen(name);
  if (n >= sizeof(lower)) n = sizeof(lower) - 1;
  for (size_t i = 0; i < n; i++) {
    char c = name[i];
    lower[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
  }
  lower[n] = 0;
  if (contains_ci(lower, "self_attn") || contains_ci(lower, ".attn.") ||
      contains_ci(lower, "attention") || contains_ci(lower, "q_proj") ||
      contains_ci(lower, "k_proj") || contains_ci(lower, "v_proj") ||
      contains_ci(lower, "o_proj") || contains_ci(lower, "qkv") ||
      contains_ci(lower, "query_proj") || contains_ci(lower, "key_proj") ||
      contains_ci(lower, "value_proj"))
    return r->attention_t;
  if (contains_ci(lower, "mlp") || contains_ci(lower, "ffn") ||
      contains_ci(lower, "feed_forward") || contains_ci(lower, "expert") ||
      contains_ci(lower, "gate_proj") || contains_ci(lower, "up_proj") ||
      contains_ci(lower, "down_proj") || contains_ci(lower, "w1") ||
      contains_ci(lower, "w2") || contains_ci(lower, "w3"))
    return r->mlp_t;
  return r->other_t;
}

/* ----------------------------------------------------------------- blend -- */

static void linear_f32(const float *a, const float *b, size_t n, float t, float *out) {
  float one_minus_t = 1.0f - t;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) out[i] = fmaf(a[i], one_minus_t, b[i] * t);
}

static void slerp_f32(const float *a, const float *b, size_t n, float t, float *out) {
  if (n == 0) return;
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : dot, norm_a, norm_b)
  for (size_t i = 0; i < n; i++) {
    double l = (double)a[i], r = (double)b[i];
    dot += l * r;
    norm_a += l * l;
    norm_b += r * r;
  }
  if (norm_a == 0.0 && norm_b == 0.0) {
    memset(out, 0, n * sizeof(float));
    return;
  }
  if (norm_a == 0.0) { memcpy(out, b, n * sizeof(float)); return; }
  if (norm_b == 0.0) { memcpy(out, a, n * sizeof(float)); return; }
  double cos_theta = dot / (sqrt(norm_a) * sqrt(norm_b));
  if (cos_theta > 1.0) cos_theta = 1.0;
  if (cos_theta < -1.0) cos_theta = -1.0;
  double theta = acos(cos_theta);
  if (theta < 1e-8) { linear_f32(a, b, n, t, out); return; }
  double sin_theta = sin(theta);
  /* Near-antipodal: slerp weights blow up; fall back to linear. */
  if (sin_theta < 1e-8) { linear_f32(a, b, n, t, out); return; }
  double w0 = sin((1.0 - (double)t) * theta) / sin_theta;
  double w1 = sin((double)t * theta) / sin_theta;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
    out[i] = (float)(w0 * (double)a[i] + w1 * (double)b[i]);
}

typedef enum { METHOD_LINEAR, METHOD_SLERP } Method;

/* ------------------------------------------------------------- fp8 e4m3fn --
 * DeepSeek-style block-quantized checkpoints store F8_E4M3 weights plus a
 * per-128x128-block F32 `<name>_scale_inv` tensor (dequant = w * scale_inv).
 * Blending such pairs naively (copy fp8 from A, SLERP the scales) corrupts
 * the weights, so oc-merge dequantizes both sides to f32, blends, and
 * requantizes each block against a fresh amax/448 scale.
 * ponytail: block size hardcoded to 128 (the fp8 weight_block_size every
 * known checkpoint uses); read quantization_config if that ever varies. */
#define FP8_BLOCK 128
#define FP8_MAX 448.0f

static float g_fp8_table[256];

static void fp8_init_table(void) {
  for (int b = 0; b < 256; b++) {
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0xf;
    int man = b & 7;
    float v;
    if (exp == 0xf && man == 7) {
      v = NAN; /* e4m3fn: 0x7f/0xff are NaN, no infinities */
    } else if (exp == 0) {
      v = (float)man / 8.0f * (1.0f / 64.0f); /* subnormal: man/8 * 2^-6 */
    } else {
      v = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    }
    g_fp8_table[b] = sign ? -v : v;
  }
}

/* Encode f32 -> e4m3fn, round-to-nearest, ties-to-even, saturating.
 * Positive codes 0x00..0x7e decode monotonically, so binary-search them. */
static uint8_t fp8_encode(float v) {
  if (isnan(v)) return 0x7f;
  uint8_t sign = 0;
  if (v < 0.0f || (v == 0.0f && signbit(v))) {
    sign = 0x80;
    v = -v;
  }
  if (v >= FP8_MAX) return sign | 0x7e;
  int lo = 0, hi = 0x7e;
  while (lo < hi) { /* smallest code with table[code] >= v */
    int mid = (lo + hi) / 2;
    if (g_fp8_table[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0) {
    float d_hi = g_fp8_table[lo] - v;
    float d_lo = v - g_fp8_table[lo - 1];
    if (d_lo < d_hi || (d_lo == d_hi && ((lo - 1) & 1) == 0)) lo--;
  }
  return sign | (uint8_t)lo;
}

/* True when `weight_name` (dtype F8_E4M3, 2-D) has a matching F32
 * `<weight_name>_scale_inv` of block shape in both models. */
static bool fp8_pair_ok(Model *ma, Model *mb, const TensorRef *wa,
                        const TensorRef *wb, TensorRef **sa_out,
                        TensorRef **sb_out) {
  if (strcmp(wa->dtype, "F8_E4M3") != 0 || wa->shape_n != 2) return false;
  char scale_name[512];
  if (snprintf(scale_name, sizeof(scale_name), "%s_scale_inv", wa->name) >=
      (int)sizeof(scale_name))
    return false;
  TensorRef *sa = model_find(ma, scale_name);
  TensorRef *sb = model_find(mb, scale_name);
  if (!sa || !sb || strcmp(sa->dtype, "F32") != 0 ||
      strcmp(sb->dtype, "F32") != 0 || sa->shape_n != 2 || sb->shape_n != 2)
    return false;
  uint64_t br = (wa->shape[0] + FP8_BLOCK - 1) / FP8_BLOCK;
  uint64_t bc = (wa->shape[1] + FP8_BLOCK - 1) / FP8_BLOCK;
  if (sa->shape[0] != br || sa->shape[1] != bc || sb->shape[0] != br ||
      sb->shape[1] != bc)
    return false;
  (void)wb;
  if (sa_out) *sa_out = sa;
  if (sb_out) *sb_out = sb;
  return true;
}

/* Is `name` a scale tensor already emitted alongside its fp8 weight? */
static bool fp8_scale_consumed(Model *ma, Model *mb, const char *name) {
  size_t len = strlen(name);
  const char *suffix = "_scale_inv";
  size_t sl = strlen(suffix);
  if (len <= sl || strcmp(name + len - sl, suffix) != 0) return false;
  char weight_name[512];
  if (len - sl >= sizeof(weight_name)) return false;
  memcpy(weight_name, name, len - sl);
  weight_name[len - sl] = 0;
  TensorRef *wa = model_find(ma, weight_name);
  TensorRef *wb = model_find(mb, weight_name);
  if (!wa || !wb) return false;
  return fp8_pair_ok(ma, mb, wa, wb, NULL, NULL);
}

static void dequant_fp8(const uint8_t *w, const float *scale, uint64_t rows,
                        uint64_t cols, uint64_t bc, float *out) {
#pragma omp parallel for schedule(static)
  for (uint64_t r = 0; r < rows; r++) {
    const float *srow = scale + (r / FP8_BLOCK) * bc;
    for (uint64_t c = 0; c < cols; c++)
      out[r * cols + c] = g_fp8_table[w[r * cols + c]] * srow[c / FP8_BLOCK];
  }
}

/* Requantize blended f32 values to fp8 + fresh per-block scales. */
static void requant_fp8(const float *vals, uint64_t rows, uint64_t cols,
                        uint64_t br, uint64_t bc, uint8_t *w_out,
                        float *scale_out) {
#pragma omp parallel for schedule(static) collapse(2)
  for (uint64_t rb = 0; rb < br; rb++) {
    for (uint64_t cb = 0; cb < bc; cb++) {
      uint64_t r0 = rb * FP8_BLOCK, r1 = r0 + FP8_BLOCK;
      uint64_t c0 = cb * FP8_BLOCK, c1 = c0 + FP8_BLOCK;
      if (r1 > rows) r1 = rows;
      if (c1 > cols) c1 = cols;
      float amax = 0.0f;
      for (uint64_t r = r0; r < r1; r++)
        for (uint64_t c = c0; c < c1; c++) {
          float a = fabsf(vals[r * cols + c]);
          if (a > amax) amax = a;
        }
      float scale = amax > 0.0f ? amax / FP8_MAX : 1.0f;
      scale_out[rb * bc + cb] = scale;
      float inv = 1.0f / scale;
      for (uint64_t r = r0; r < r1; r++)
        for (uint64_t c = c0; c < c1; c++)
          w_out[r * cols + c] = fp8_encode(vals[r * cols + c] * inv);
    }
  }
}

/* Blend two raw tensors of a blendable dtype into `out` (same dtype). */
static void blend_bytes(const char *dtype, const uint8_t *a, const uint8_t *b,
                        size_t len, float t, Method method, uint8_t *out,
                        const char *name) {
  size_t elem = strcmp(dtype, "F32") == 0 ? 4 : 2;
  if (len % elem != 0) die("tensor %s byte length not a multiple of element size", name);
  size_t n = len / elem;
  float *fa = xmalloc(n * sizeof(float));
  float *fb = xmalloc(n * sizeof(float));
  float *fo = xmalloc(n * sizeof(float));
  if (elem == 4) {
    memcpy(fa, a, len);
    memcpy(fb, b, len);
  } else if (strcmp(dtype, "F16") == 0) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      uint16_t ua, ub;
      memcpy(&ua, a + 2 * i, 2);
      memcpy(&ub, b + 2 * i, 2);
      fa[i] = f16_to_f32(ua);
      fb[i] = f16_to_f32(ub);
    }
  } else { /* BF16 */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      uint16_t ua, ub;
      memcpy(&ua, a + 2 * i, 2);
      memcpy(&ub, b + 2 * i, 2);
      uint32_t xa = (uint32_t)ua << 16, xb = (uint32_t)ub << 16;
      memcpy(&fa[i], &xa, 4);
      memcpy(&fb[i], &xb, 4);
    }
  }
  if (method == METHOD_LINEAR) linear_f32(fa, fb, n, t, fo);
  else slerp_f32(fa, fb, n, t, fo);
  if (elem == 4) {
    memcpy(out, fo, len);
  } else if (strcmp(dtype, "F16") == 0) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      uint16_t u = f32_to_f16(fo[i]);
      memcpy(out + 2 * i, &u, 2);
    }
  } else { /* BF16: truncate like the Rust port */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      uint32_t bits;
      memcpy(&bits, &fo[i], 4);
      uint16_t u = (uint16_t)(bits >> 16);
      memcpy(out + 2 * i, &u, 2);
    }
  }
  free(fa);
  free(fb);
  free(fo);
}

/* ---------------------------------------------------------------- writer -- */

typedef struct {
  char *name;
  char *dtype;
  uint64_t *shape;
  size_t shape_n;
  uint8_t *data; /* owned */
  size_t len;
} OutTensor;

typedef struct {
  bool single;           /* output is a lone .safetensors file */
  char *output;          /* file path (single) or directory */
  uint64_t max_shard_bytes;
  KvMap metadata;
  OutTensor *pend;       /* current shard (sharded) or all tensors (single) */
  size_t pend_n, pend_cap;
  uint64_t pend_bytes;
  size_t shard_index;
  char **wm_tensor;      /* accumulated weight_map */
  char **wm_shard;
  size_t wm_n, wm_cap;
  size_t total_tensors;
} Writer;

static void mkdir_p(const char *path) {
  char *tmp = xstrdup(path);
  for (char *p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
        die("failed to create directory %s: %s", tmp, strerror(errno));
      *p = '/';
    }
  }
  if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
    die("failed to create directory %s: %s", tmp, strerror(errno));
  free(tmp);
}

static void json_escape_to(FILE *f, const char *s) {
  for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
    if (*p == '"' || *p == '\\') fprintf(f, "\\%c", *p);
    else if (*p < 0x20) fprintf(f, "\\u%04x", *p);
    else fputc(*p, f);
  }
}

static void write_safetensors_file(const char *path, OutTensor *tensors,
                                   size_t n, const KvMap *metadata) {
  /* Build header JSON in memory. */
  size_t cap = 4096, hn = 0;
  char *hdr = xmalloc(cap);
#define HCAT(...)                                                        \
  do {                                                                   \
    int need = snprintf(NULL, 0, __VA_ARGS__);                           \
    while (hn + (size_t)need + 1 > cap) { cap *= 2; hdr = xrealloc(hdr, cap); } \
    hn += (size_t)snprintf(hdr + hn, cap - hn, __VA_ARGS__);             \
  } while (0)

  /* Escape helper for in-memory building (names/metadata rarely need it). */
  HCAT("{");
  bool first = true;
  if (metadata->n) {
    HCAT("\"__metadata__\":{");
    for (size_t i = 0; i < metadata->n; i++) {
      /* metadata keys/values escaped conservatively via a temp file-less path:
       * assume no embedded quotes is wrong — escape manually */
      HCAT("%s\"", i ? "," : "");
      for (const unsigned char *p = (const unsigned char *)metadata->keys[i]; *p; p++) {
        if (*p == '"' || *p == '\\') HCAT("\\%c", *p);
        else if (*p < 0x20) HCAT("\\u%04x", *p);
        else HCAT("%c", *p);
      }
      HCAT("\":\"");
      for (const unsigned char *p = (const unsigned char *)metadata->vals[i]; *p; p++) {
        if (*p == '"' || *p == '\\') HCAT("\\%c", *p);
        else if (*p < 0x20) HCAT("\\u%04x", *p);
        else HCAT("%c", *p);
      }
      HCAT("\"");
    }
    HCAT("}");
    first = false;
  }
  uint64_t off = 0;
  for (size_t i = 0; i < n; i++) {
    OutTensor *t = &tensors[i];
    HCAT("%s\"", first ? "" : ",");
    for (const unsigned char *p = (const unsigned char *)t->name; *p; p++) {
      if (*p == '"' || *p == '\\') HCAT("\\%c", *p);
      else if (*p < 0x20) HCAT("\\u%04x", *p);
      else HCAT("%c", *p);
    }
    HCAT("\":{\"dtype\":\"%s\",\"shape\":[", t->dtype);
    for (size_t s = 0; s < t->shape_n; s++)
      HCAT("%s%llu", s ? "," : "", (unsigned long long)t->shape[s]);
    HCAT("],\"data_offsets\":[%llu,%llu]}", (unsigned long long)off,
         (unsigned long long)(off + t->len));
    off += t->len;
    first = false;
  }
  HCAT("}");
#undef HCAT

  /* Pad header with spaces to an 8-byte multiple (matches safetensors-rs). */
  size_t padded = (hn + 7) & ~(size_t)7;
  while (hn + 1 > cap || padded + 1 > cap) { cap *= 2; hdr = xrealloc(hdr, cap); }
  for (size_t i = hn; i < padded; i++) hdr[i] = ' ';

  FILE *f = fopen(path, "wb");
  if (!f) die("failed to create %s: %s", path, strerror(errno));
  uint64_t hlen = padded;
  if (fwrite(&hlen, 8, 1, f) != 1 || fwrite(hdr, 1, padded, f) != padded)
    die("failed to write %s: %s", path, strerror(errno));
  for (size_t i = 0; i < n; i++) {
    if (tensors[i].len &&
        fwrite(tensors[i].data, 1, tensors[i].len, f) != tensors[i].len)
      die("failed to write %s: %s", path, strerror(errno));
  }
  if (fclose(f) != 0) die("failed to write %s: %s", path, strerror(errno));
  free(hdr);
}

static void writer_init(Writer *w, const char *output, uint64_t max_shard_bytes,
                        KvMap metadata) {
  memset(w, 0, sizeof(*w));
  w->output = xstrdup(output);
  w->max_shard_bytes = max_shard_bytes;
  w->metadata = metadata;
  size_t len = strlen(output);
  w->single = len > 12 && strcmp(output + len - 12, ".safetensors") == 0;
  if (w->single) {
    char *dir = xstrdup(output);
    char *slash = strrchr(dir, '/');
    if (slash && slash != dir) {
      *slash = 0;
      mkdir_p(dir);
    }
    free(dir);
  } else {
    if (max_shard_bytes == 0) die("max shard size must be greater than zero");
    mkdir_p(output);
  }
}

static void writer_flush_shard(Writer *w) {
  char shard_name[64];
  snprintf(shard_name, sizeof(shard_name), "model-%05zu-of-?????.safetensors",
           w->shard_index);
  char *shard_path = path_join(w->output, shard_name);
  write_safetensors_file(shard_path, w->pend, w->pend_n, &w->metadata);
  free(shard_path);
  for (size_t i = 0; i < w->pend_n; i++) {
    if (w->wm_n == w->wm_cap) {
      w->wm_cap = w->wm_cap ? w->wm_cap * 2 : 256;
      w->wm_tensor = xrealloc(w->wm_tensor, w->wm_cap * sizeof(char *));
      w->wm_shard = xrealloc(w->wm_shard, w->wm_cap * sizeof(char *));
    }
    w->wm_tensor[w->wm_n] = w->pend[i].name; /* take ownership */
    w->wm_shard[w->wm_n] = xstrdup(shard_name);
    w->wm_n++;
    w->total_tensors++;
    free(w->pend[i].dtype);
    free(w->pend[i].shape);
    free(w->pend[i].data);
  }
  w->shard_index++;
  w->pend_n = 0;
  w->pend_bytes = 0;
}

static void writer_push(Writer *w, OutTensor t) {
  if (!w->single && w->pend_n > 0 &&
      w->pend_bytes + (uint64_t)t.len > w->max_shard_bytes)
    writer_flush_shard(w);
  if (w->pend_n == w->pend_cap) {
    w->pend_cap = w->pend_cap ? w->pend_cap * 2 : 64;
    w->pend = xrealloc(w->pend, w->pend_cap * sizeof(OutTensor));
  }
  w->pend[w->pend_n++] = t;
  w->pend_bytes += t.len;
}

static size_t writer_finish(Writer *w) {
  if (w->single) {
    if (w->pend_n == 0) die("no tensors were written");
    write_safetensors_file(w->output, w->pend, w->pend_n, &w->metadata);
    return w->pend_n;
  }
  if (w->pend_n > 0) writer_flush_shard(w);
  if (w->wm_n == 0) die("no tensors were written");

  size_t total = w->shard_index;
  /* Rename model-NNNNN-of-?????.safetensors -> -of-TOTAL. */
  for (size_t i = 0; i < total; i++) {
    char oldn[64], newn[64];
    snprintf(oldn, sizeof(oldn), "model-%05zu-of-?????.safetensors", i);
    snprintf(newn, sizeof(newn), "model-%05zu-of-%05zu.safetensors", i, total);
    char *oldp = path_join(w->output, oldn);
    char *newp = path_join(w->output, newn);
    struct stat st;
    if (stat(oldp, &st) == 0) {
      if (rename(oldp, newp) != 0)
        die("failed to rename %s: %s", oldp, strerror(errno));
    } else if (stat(newp, &st) != 0) {
      die("shard %s missing while finalizing index (expected %s or %s)", oldn,
          oldp, newp);
    }
    free(oldp);
    free(newp);
  }

  /* weight_map, sorted by tensor name for a deterministic index. */
  size_t *order = xmalloc(w->wm_n * sizeof(size_t));
  for (size_t i = 0; i < w->wm_n; i++) order[i] = i;
  for (size_t i = 1; i < w->wm_n; i++) { /* insertion sort on names */
    size_t k = order[i], jx = i;
    while (jx > 0 && strcmp(w->wm_tensor[order[jx - 1]], w->wm_tensor[k]) > 0) {
      order[jx] = order[jx - 1];
      jx--;
    }
    order[jx] = k;
  }

  char *index_path = path_join(w->output, "model.safetensors.index.json");
  FILE *f = fopen(index_path, "wb");
  if (!f) die("failed to create %s: %s", index_path, strerror(errno));
  fprintf(f, "{\n  \"metadata\": {\n");
  kv_sort(&w->metadata);
  for (size_t i = 0; i < w->metadata.n; i++) {
    fprintf(f, "    \"");
    json_escape_to(f, w->metadata.keys[i]);
    fprintf(f, "\": \"");
    json_escape_to(f, w->metadata.vals[i]);
    fprintf(f, "\"%s\n", i + 1 < w->metadata.n ? "," : "");
  }
  fprintf(f, "  },\n  \"weight_map\": {\n");
  for (size_t i = 0; i < w->wm_n; i++) {
    char fixed[64];
    /* stored names still contain of-????? — recompute final name */
    snprintf(fixed, sizeof(fixed), "%s", w->wm_shard[order[i]]);
    char *q = strstr(fixed, "of-?????");
    if (q) snprintf(q, sizeof(fixed) - (size_t)(q - fixed), "of-%05zu.safetensors", total);
    fprintf(f, "    \"");
    json_escape_to(f, w->wm_tensor[order[i]]);
    fprintf(f, "\": \"%s\"%s\n", fixed, i + 1 < w->wm_n ? "," : "");
  }
  fprintf(f, "  }\n}\n");
  if (fclose(f) != 0) die("failed to write %s", index_path);
  free(index_path);
  free(order);
  return w->total_tensors;
}

/* ----------------------------------------------------------------- merge -- */

typedef enum { MISSING_ERROR, MISSING_A, MISSING_B } MissingPolicy;

static void resolve_single_side(MissingPolicy policy, bool missing_from_b,
                                const char *name) {
  if (policy == MISSING_ERROR) {
    if (missing_from_b) die("tensor %s exists only in model A", name);
    die("tensor %s exists only in model B", name);
  }
  if (policy == MISSING_A && !missing_from_b)
    die("tensor %s missing from model A", name);
  if (policy == MISSING_B && missing_from_b)
    die("tensor %s missing from model B", name);
}

static void validate_compatible(const TensorRef *a, const TensorRef *b) {
  if (strcmp(a->dtype, b->dtype) != 0)
    die("dtype mismatch for %s: %s vs %s", a->name, a->dtype, b->dtype);
  if (a->shape_n != b->shape_n ||
      memcmp(a->shape, b->shape, a->shape_n * sizeof(uint64_t)) != 0)
    die("shape mismatch for %s", a->name);
}

static uint64_t *shape_dup(const uint64_t *shape, size_t n) {
  uint64_t *s = xmalloc((n ? n : 1) * sizeof(uint64_t));
  memcpy(s, shape, n * sizeof(uint64_t));
  return s;
}

/* -------------------------------------------------------------- self-test -- */

static void self_test(void);

/* ------------------------------------------------------------------ main -- */

static void usage(void) {
  fprintf(stderr,
          "oc-merge: merge two SafeTensors checkpoints (linear or SLERP)\n"
          "usage: oc-merge --a <model> --b <model> --output <path>\n"
          "  --a / --b        .safetensors file or HuggingFace model directory\n"
          "  --output         .safetensors file or directory for sharded output\n"
          "  --method         linear | slerp            (default slerp)\n"
          "  --preset         kimi-k275\n"
          "  --t              global blend weight in [0,1] toward B\n"
          "  --attention-t    attention blend weight     (default 0.3)\n"
          "  --mlp-t          MLP/expert blend weight    (default 0.5)\n"
          "  --other-t        other-tensor blend weight  (default 0.4)\n"
          "  --missing        error | a | b              (default error)\n"
          "  --max-shard-gib  max shard size in GiB      (default 5)\n"
          "  --dry-run        validate without writing\n"
          "  --self-test      run built-in correctness checks\n");
  exit(1);
}

int main(int argc, char **argv) {
  const char *a_path = NULL, *b_path = NULL, *out_path = NULL;
  Method method = METHOD_SLERP;
  MissingPolicy missing = MISSING_ERROR;
  Recipe recipe = {.attention_t = 0.3f, .mlp_t = 0.5f, .other_t = 0.4f};
  bool preset_kimi = false, has_t = false, dry_run = false;
  float global_t = 0.5f;
  uint64_t max_shard_gib = 5;

  fp8_init_table();
  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    const char *val = (i + 1 < argc) ? argv[i + 1] : NULL;
    if (strcmp(arg, "--self-test") == 0) { self_test(); return 0; }
    if (strcmp(arg, "--dry-run") == 0) { dry_run = true; continue; }
    if (strcmp(arg, "--a") == 0 && val) { a_path = val; i++; }
    else if (strcmp(arg, "--b") == 0 && val) { b_path = val; i++; }
    else if (strcmp(arg, "--output") == 0 && val) { out_path = val; i++; }
    else if (strcmp(arg, "--method") == 0 && val) {
      if (strcmp(val, "linear") == 0) method = METHOD_LINEAR;
      else if (strcmp(val, "slerp") == 0) method = METHOD_SLERP;
      else die("--method must be linear or slerp");
      i++;
    } else if (strcmp(arg, "--preset") == 0 && val) {
      if (strcmp(val, "kimi-k275") != 0) die("unknown preset %s", val);
      preset_kimi = true;
      i++;
    } else if (strcmp(arg, "--t") == 0 && val) {
      global_t = strtof(val, NULL);
      has_t = true;
      i++;
    } else if (strcmp(arg, "--attention-t") == 0 && val) {
      recipe.attention_t = strtof(val, NULL); i++;
    } else if (strcmp(arg, "--mlp-t") == 0 && val) {
      recipe.mlp_t = strtof(val, NULL); i++;
    } else if (strcmp(arg, "--other-t") == 0 && val) {
      recipe.other_t = strtof(val, NULL); i++;
    } else if (strcmp(arg, "--missing") == 0 && val) {
      if (strcmp(val, "error") == 0) missing = MISSING_ERROR;
      else if (strcmp(val, "a") == 0) missing = MISSING_A;
      else if (strcmp(val, "b") == 0) missing = MISSING_B;
      else die("--missing must be error, a, or b");
      i++;
    } else if (strcmp(arg, "--max-shard-gib") == 0 && val) {
      max_shard_gib = strtoull(val, NULL, 10); i++;
    } else {
      usage();
    }
  }
  if (!a_path || !b_path || !out_path) usage();
  if (has_t && (global_t < 0.0f || global_t > 1.0f)) die("--t must be in [0, 1]");
  if (recipe.attention_t < 0.0f || recipe.attention_t > 1.0f)
    die("--attention-t must be in [0, 1]");
  if (recipe.mlp_t < 0.0f || recipe.mlp_t > 1.0f) die("--mlp-t must be in [0, 1]");
  if (recipe.other_t < 0.0f || recipe.other_t > 1.0f) die("--other-t must be in [0, 1]");

  if (has_t) {
    recipe.attention_t = recipe.mlp_t = recipe.other_t = global_t;
    recipe.default_t = global_t;
    recipe.has_default = true;
  } else if (preset_kimi) {
    recipe.attention_t = 0.3f;
    recipe.mlp_t = 0.5f;
    recipe.other_t = 0.4f;
  }

  Model ma, mb;
  model_open(&ma, a_path);
  model_open(&mb, b_path);

  /* Union of tensor names, lexicographically sorted (both tables sorted). */
  size_t na = ma.tensors_n, nb = mb.tensors_n;
  size_t ia = 0, ib = 0;
  size_t merged = 0, copied_a = 0, copied_b = 0;

  Writer writer;
  if (!dry_run) {
    KvMap meta = {0};
    for (size_t i = 0; i < ma.metadata.n; i++)
      kv_put(&meta, xstrdup(ma.metadata.keys[i]), xstrdup(ma.metadata.vals[i]));
    for (size_t i = 0; i < mb.metadata.n; i++)
      kv_put(&meta, xstrdup(mb.metadata.keys[i]), xstrdup(mb.metadata.vals[i]));
    char buf[64];
    kv_put(&meta, xstrdup("oxidize-merge.method"),
           xstrdup(method == METHOD_LINEAR ? "linear" : "slerp"));
    snprintf(buf, sizeof(buf), "%g", (double)recipe.attention_t);
    kv_put(&meta, xstrdup("oxidize-merge.attention_t"), xstrdup(buf));
    snprintf(buf, sizeof(buf), "%g", (double)recipe.mlp_t);
    kv_put(&meta, xstrdup("oxidize-merge.mlp_t"), xstrdup(buf));
    snprintf(buf, sizeof(buf), "%g", (double)recipe.other_t);
    kv_put(&meta, xstrdup("oxidize-merge.other_t"), xstrdup(buf));
    if (recipe.has_default) {
      snprintf(buf, sizeof(buf), "%g", (double)recipe.default_t);
      kv_put(&meta, xstrdup("oxidize-merge.default_t"), xstrdup(buf));
    }
    kv_put(&meta, xstrdup("oxidize-merge.model_a"), xstrdup(a_path));
    kv_put(&meta, xstrdup("oxidize-merge.model_b"), xstrdup(b_path));
    uint64_t max_bytes = max_shard_gib > UINT64_MAX / (1024ull * 1024 * 1024)
                             ? UINT64_MAX
                             : max_shard_gib * 1024ull * 1024 * 1024;
    writer_init(&writer, out_path, max_bytes, meta);
  }

  while (ia < na || ib < nb) {
    TensorRef *ta = ia < na ? &ma.tensors[ia] : NULL;
    TensorRef *tb = ib < nb ? &mb.tensors[ib] : NULL;
    int cmp = ta && tb ? strcmp(ta->name, tb->name) : (ta ? -1 : 1);
    if (cmp == 0) {
      validate_compatible(ta, tb);
      TensorRef *sa = NULL, *sb = NULL;
      if (fp8_pair_ok(&ma, &mb, ta, tb, &sa, &sb)) {
        merged++;
        if (!dry_run) {
          uint64_t rows = ta->shape[0], cols = ta->shape[1];
          uint64_t br = (rows + FP8_BLOCK - 1) / FP8_BLOCK;
          uint64_t bc = (cols + FP8_BLOCK - 1) / FP8_BLOCK;
          size_t lwa, lwb, lsa, lsb;
          const uint8_t *wa_b = tensor_bytes(&ma, ta, &lwa);
          const uint8_t *wb_b = tensor_bytes(&mb, tb, &lwb);
          const uint8_t *sa_b = tensor_bytes(&ma, sa, &lsa);
          const uint8_t *sb_b = tensor_bytes(&mb, sb, &lsb);
          if (lwa != rows * cols || lwb != lwa || lsa != br * bc * 4 || lsb != lsa)
            die("tensor %s fp8/scale byte length mismatch", ta->name);
          /* scales may be unaligned in the mmap; copy before float access */
          float *sa_f = xmalloc(lsa);
          float *sb_f = xmalloc(lsb);
          memcpy(sa_f, sa_b, lsa);
          memcpy(sb_f, sb_b, lsb);
          size_t n = (size_t)rows * cols;
          float *fa = xmalloc(n * sizeof(float));
          float *fb = xmalloc(n * sizeof(float));
          float *fo = xmalloc(n * sizeof(float));
          dequant_fp8(wa_b, sa_f, rows, cols, bc, fa);
          dequant_fp8(wb_b, sb_f, rows, cols, bc, fb);
          float t = t_for_tensor(&recipe, ta->name);
          if (method == METHOD_LINEAR) linear_f32(fa, fb, n, t, fo);
          else slerp_f32(fa, fb, n, t, fo);
          uint8_t *w_out = xmalloc(n);
          float *scale_out = xmalloc(br * bc * sizeof(float));
          requant_fp8(fo, rows, cols, br, bc, w_out, scale_out);
          writer_push(&writer, (OutTensor){.name = xstrdup(ta->name),
                                           .dtype = xstrdup(ta->dtype),
                                           .shape = shape_dup(ta->shape, 2),
                                           .shape_n = 2,
                                           .data = w_out,
                                           .len = n});
          writer_push(&writer, (OutTensor){.name = xstrdup(sa->name),
                                           .dtype = xstrdup("F32"),
                                           .shape = shape_dup(sa->shape, 2),
                                           .shape_n = 2,
                                           .data = (uint8_t *)scale_out,
                                           .len = br * bc * 4});
          free(sa_f);
          free(sb_f);
          free(fa);
          free(fb);
          free(fo);
        }
        ia++;
        ib++;
        continue;
      }
      if (fp8_scale_consumed(&ma, &mb, ta->name)) {
        merged++; /* emitted together with its fp8 weight */
        ia++;
        ib++;
        continue;
      }
      if (is_blendable(ta->dtype)) {
        merged++;
        if (!dry_run) {
          size_t la, lb;
          const uint8_t *ba = tensor_bytes(&ma, ta, &la);
          const uint8_t *bb = tensor_bytes(&mb, tb, &lb);
          if (la != lb) die("tensor %s byte length mismatch", ta->name);
          uint8_t *out = xmalloc(la ? la : 1);
          blend_bytes(ta->dtype, ba, bb, la, t_for_tensor(&recipe, ta->name),
                      method, out, ta->name);
          writer_push(&writer, (OutTensor){.name = xstrdup(ta->name),
                                           .dtype = xstrdup(ta->dtype),
                                           .shape = shape_dup(ta->shape, ta->shape_n),
                                           .shape_n = ta->shape_n,
                                           .data = out,
                                           .len = la});
        }
      } else {
        copied_a++;
        if (!dry_run) {
          size_t la;
          const uint8_t *ba = tensor_bytes(&ma, ta, &la);
          uint8_t *out = xmalloc(la ? la : 1);
          memcpy(out, ba, la);
          writer_push(&writer, (OutTensor){.name = xstrdup(ta->name),
                                           .dtype = xstrdup(ta->dtype),
                                           .shape = shape_dup(ta->shape, ta->shape_n),
                                           .shape_n = ta->shape_n,
                                           .data = out,
                                           .len = la});
        }
      }
      ia++;
      ib++;
    } else if (cmp < 0) {
      resolve_single_side(missing, true, ta->name);
      copied_a++;
      if (!dry_run) {
        size_t la;
        const uint8_t *ba = tensor_bytes(&ma, ta, &la);
        uint8_t *out = xmalloc(la ? la : 1);
        memcpy(out, ba, la);
        writer_push(&writer, (OutTensor){.name = xstrdup(ta->name),
                                         .dtype = xstrdup(ta->dtype),
                                         .shape = shape_dup(ta->shape, ta->shape_n),
                                         .shape_n = ta->shape_n,
                                         .data = out,
                                         .len = la});
      }
      ia++;
    } else {
      resolve_single_side(missing, false, tb->name);
      copied_b++;
      if (!dry_run) {
        size_t lb;
        const uint8_t *bb = tensor_bytes(&mb, tb, &lb);
        uint8_t *out = xmalloc(lb ? lb : 1);
        memcpy(out, bb, lb);
        writer_push(&writer, (OutTensor){.name = xstrdup(tb->name),
                                         .dtype = xstrdup(tb->dtype),
                                         .shape = shape_dup(tb->shape, tb->shape_n),
                                         .shape_n = tb->shape_n,
                                         .data = out,
                                         .len = lb});
      }
      ib++;
    }
  }

  if (dry_run) {
    printf("Dry run: would blend %zu tensors, copy %zu from A, copy %zu from B -> %s\n",
           merged, copied_a, copied_b, out_path);
  } else {
    writer_finish(&writer);
    printf("Merged %zu tensors (%zu copied from A, %zu copied from B) -> %s\n",
           merged, copied_a, copied_b, out_path);
  }
  return 0;
}

/* -------------------------------------------------------------- self-test -- */

#include <assert.h>

static void st_write_f32_model(const char *path, const char *name,
                               const float *vals, size_t n) {
  OutTensor t = {.name = xstrdup(name),
                 .dtype = xstrdup("F32"),
                 .shape = xmalloc(sizeof(uint64_t)),
                 .shape_n = 1,
                 .data = xmalloc(n * 4),
                 .len = n * 4};
  t.shape[0] = n;
  memcpy(t.data, vals, n * 4);
  KvMap meta = {0};
  write_safetensors_file(path, &t, 1, &meta);
  free(t.name);
  free(t.dtype);
  free(t.shape);
  free(t.data);
}

static void self_test(void) {
  const char *dir = "/tmp/oc-merge-selftest";
  mkdir_p(dir);
  char pa[256], pb[256], pout[256];
  snprintf(pa, sizeof(pa), "%s/a.safetensors", dir);
  snprintf(pb, sizeof(pb), "%s/b.safetensors", dir);

  /* linear midpoint */
  float va[2] = {0.0f, 2.0f}, vb[2] = {2.0f, 4.0f};
  st_write_f32_model(pa, "weight", va, 2);
  st_write_f32_model(pb, "weight", vb, 2);
  Model ma, mb;
  model_open(&ma, pa);
  model_open(&mb, pb);
  assert(ma.tensors_n == 1 && mb.tensors_n == 1);
  TensorRef *ta = model_find(&ma, "weight");
  TensorRef *tb = model_find(&mb, "weight");
  assert(ta && tb && strcmp(ta->dtype, "F32") == 0);
  size_t la, lb;
  const uint8_t *ba = tensor_bytes(&ma, ta, &la);
  const uint8_t *bb = tensor_bytes(&mb, tb, &lb);
  assert(la == 8 && lb == 8);
  uint8_t out[8];
  blend_bytes("F32", ba, bb, 8, 0.5f, METHOD_LINEAR, out, "weight");
  float fo[2];
  memcpy(fo, out, 8);
  assert(fabsf(fo[0] - 1.0f) < 1e-5f && fabsf(fo[1] - 3.0f) < 1e-5f);

  /* slerp endpoints + 45° midpoint on orthogonal unit vectors */
  float ua[2] = {1.0f, 0.0f}, ub[2] = {0.0f, 1.0f};
  float so[2];
  slerp_f32(ua, ub, 2, 0.0f, so);
  assert(fabsf(so[0] - 1.0f) < 1e-5f && fabsf(so[1]) < 1e-5f);
  slerp_f32(ua, ub, 2, 1.0f, so);
  assert(fabsf(so[0]) < 1e-5f && fabsf(so[1] - 1.0f) < 1e-5f);
  slerp_f32(ua, ub, 2, 0.5f, so);
  float half = 0.70710678f;
  assert(fabsf(so[0] - half) < 1e-4f && fabsf(so[1] - half) < 1e-4f);

  /* bf16 roundtrip blend: 1.0 and 3.0 -> 2.0 exactly representable */
  uint16_t b16a[2] = {0x3f80, 0x4040}; /* 1.0, 3.0 */
  uint16_t b16b[2] = {0x4040, 0x40a0}; /* 3.0, 5.0 */
  uint8_t bo[4];
  blend_bytes("BF16", (uint8_t *)b16a, (uint8_t *)b16b, 4, 0.5f, METHOD_LINEAR,
              bo, "t");
  uint16_t r0, r1;
  memcpy(&r0, bo, 2);
  memcpy(&r1, bo + 2, 2);
  assert(r0 == 0x4000 /* 2.0 */ && r1 == 0x4080 /* 4.0 */);

  /* f16 conversion roundtrip on a few values */
  for (float v = -4.0f; v <= 4.0f; v += 0.25f)
    assert(f16_to_f32(f32_to_f16(v)) == v);

  /* sharded writer: force 2 shards, reopen via index, verify values */
  snprintf(pout, sizeof(pout), "%s/sharded", dir);
  KvMap meta = {0};
  kv_put(&meta, xstrdup("format"), xstrdup("pt"));
  Writer w;
  writer_init(&w, pout, 8 /* bytes: one 2-float tensor per shard */, meta);
  float t1[2] = {1.0f, 2.0f}, t2[2] = {3.0f, 4.0f};
  OutTensor o1 = {.name = xstrdup("alpha"), .dtype = xstrdup("F32"),
                  .shape = xmalloc(8), .shape_n = 1, .data = xmalloc(8), .len = 8};
  o1.shape[0] = 2;
  memcpy(o1.data, t1, 8);
  OutTensor o2 = {.name = xstrdup("beta"), .dtype = xstrdup("F32"),
                  .shape = xmalloc(8), .shape_n = 1, .data = xmalloc(8), .len = 8};
  o2.shape[0] = 2;
  memcpy(o2.data, t2, 8);
  writer_push(&w, o1);
  writer_push(&w, o2);
  size_t count = writer_finish(&w);
  assert(count == 2);
  Model ms;
  model_open(&ms, pout);
  assert(ms.tensors_n == 2);
  TensorRef *t_alpha = model_find(&ms, "alpha");
  TensorRef *t_beta = model_find(&ms, "beta");
  assert(t_alpha && t_beta);
  size_t l;
  const uint8_t *pbytes = tensor_bytes(&ms, t_alpha, &l);
  float rv[2];
  memcpy(rv, pbytes, 8);
  assert(l == 8 && rv[0] == 1.0f && rv[1] == 2.0f);
  pbytes = tensor_bytes(&ms, t_beta, &l);
  memcpy(rv, pbytes, 8);
  assert(l == 8 && rv[0] == 3.0f && rv[1] == 4.0f);
  assert(strcmp(kv_get(&ms.metadata, "format"), "pt") == 0);

  /* fp8 e4m3fn: encode(decode) roundtrip for every non-NaN code */
  for (int b = 0; b < 256; b++) {
    if (b == 0x7f || b == 0xff) continue; /* NaN */
    uint8_t rt = fp8_encode(g_fp8_table[b]);
    assert(g_fp8_table[rt] == g_fp8_table[b]);
  }
  assert(fp8_encode(1e9f) == 0x7e && fp8_encode(-1e9f) == 0xfe); /* saturate */

  /* fp8 dequant -> blend(t=0) -> requant identity within one quant step,
   * with partial blocks (rows=130 -> 2 row blocks) */
  {
    enum { ROWS = 130, COLS = 5 };
    uint64_t br = 2, bc = 1;
    uint8_t w[ROWS * COLS];
    float scale[2] = {0.011f, 3.7f};
    for (size_t i = 0; i < sizeof(w); i++) {
      uint8_t code = (uint8_t)((i * 37) & 0xff);
      if (code == 0x7f || code == 0xff) code = 0x10;
      w[i] = code;
    }
    float deq[ROWS * COLS], out[ROWS * COLS], deq2[ROWS * COLS];
    dequant_fp8(w, scale, ROWS, COLS, bc, deq);
    linear_f32(deq, deq, ROWS * COLS, 0.0f, out); /* t=0 -> A */
    uint8_t w2[ROWS * COLS];
    float scale2[2];
    requant_fp8(out, ROWS, COLS, br, bc, w2, scale2);
    dequant_fp8(w2, scale2, ROWS, COLS, bc, deq2);
    for (size_t i = 0; i < (size_t)ROWS * COLS; i++) {
      float ref = deq[i];
      float tol = fabsf(ref) * 0.0725f + 1e-6f; /* 3-bit mantissa grid shift */
      assert(fabsf(deq2[i] - ref) <= tol);
    }
  }

  /* recipe classification */
  Recipe r = {.attention_t = 0.3f, .mlp_t = 0.5f, .other_t = 0.4f};
  assert(fabsf(t_for_tensor(&r, "model.layers.0.self_attn.q_proj.weight") - 0.3f) < 1e-6f);
  assert(fabsf(t_for_tensor(&r, "model.layers.3.mlp.experts.0.gate_proj.weight") - 0.5f) < 1e-6f);
  assert(fabsf(t_for_tensor(&r, "model.embed_tokens.weight") - 0.4f) < 1e-6f);

  printf("oc-merge self-test: all checks passed\n");
}
