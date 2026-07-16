/* oxidize-c-convert: HuggingFace SafeTensors -> GGUF v3.
 *
 * Reads a single .safetensors file OR a HF model directory (config.json +
 * one-or-more *.safetensors + optional tokenizer.json), maps HF tensor names to
 * the ggml/llama.cpp convention the C loaders expect (token_embd.weight,
 * blk.N.attn_q.weight, ...), copies the config.json geometry into {arch}.* GGUF
 * KVs, embeds the tokenizer, and re-encodes every weight to --outtype.
 *
 * SCOPE (honest): dense decoder-only.
 *   permuted-q/k archs (ggml NORMAL rope): llama, mistral, yi
 *   NeoX-rope archs (unpermuted q/k):       qwen2, qwen3, phi3/phi
 *   Gemma sandwich (GeGLU + 4 norms):      gemma2/gemma3/gemma4 (written as
 *     GGUF arch "gemma4" for the C loader; RMSNorm weights get +1 baked in)
 * Phi-3 ships fused HF tensors (self_attn.qkv_proj, mlp.gate_up_proj); this
 * converter SPLITS them into the separate attn_q/k/v + ffn_gate/up that
 * model_llama.c already loads. SuRoPE / sliding-window 128k factors are not
 * written — fine for 4k Phi-3-mini; long-context SuRoPE GGUFs need more work.
 * Everything else -- MoE, gemma-v1 without sandwich norms, deepseek (MLA),
 * wordpiece tokenizers, non-F32/F16/BF16 safetensors -- is REJECTED LOUDLY.
 * A slower/rejecting correct converter beats a fast wrong one: a subtly-wrong
 * GGUF (esp. a bad q/k permute) is silent fluent garbage.
 *
 * THE PERMUTE (highest-risk part). llama.cpp permutes q/k so that ggml NORMAL
 * (adjacent-pair) rope on the permuted weights reproduces HF/NeoX rope on the
 * natural layout. model_llama.c runs NORMAL rope for llama/mistral/yi, so their
 * q/k MUST be permuted here. Per head (head_dim = rows/heads, must be even):
 * dest row 2i <- src row i, dest row 2i+1 <- src row i+head_dim/2. This is the
 * inverse of oc_rope_normal's pairing, verified against tests/test_model.c's
 * check_rope_modes() and, end to end, by converting one model as both llama
 * (permute+NORMAL) and qwen2 (no-permute+NeoX): identical logits <=> permute ok.
 *
 * Reuses tools/gguf_write.{h,c} for the GGUF writer + row encoders. Reference:
 * oxidize-convert + oxidize-core/src/format/safetensors_to_gguf.rs (which maps
 * to oxidize's OWN tensor names and does NOT permute -- a latent bug there).
 */
#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../src/quant.h"
#include "../src/tensor.h"
#include "convert.h"
#include "gguf_write.h"

/* ======================================================================== *
 *  Minimal strict JSON parser (no deps).  Robust to malformed input: every
 *  read is bounded, nesting depth is capped, and any error unwinds to NULL
 *  without OOB. It parses untrusted files, so this must never crash.
 * ======================================================================== */

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JKind;
/* JNode is forward-declared in convert.h; defined here. */
struct JNode {
  JKind kind;
  double num;   /* J_NUM */
  int boolean;  /* J_BOOL */
  char* str;    /* J_STR (decoded, NUL-terminated) */
  size_t slen;
  JNode** items; /* J_ARR / J_OBJ values */
  char** keys;   /* J_OBJ keys (NUL-terminated), parallel to items */
  size_t n;
};

typedef struct {
  const char* p;
  const char* end;
  int err;
  int depth;
} JP;

#define J_MAX_DEPTH 200

static JNode* jparse_value(JP* j);

static void jskip_ws(JP* j) {
  while (j->p < j->end) {
    char c = *j->p;
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') j->p++;
    else break;
  }
}

static JNode* jnode(JKind k) {
  JNode* n = calloc(1, sizeof(JNode));
  if (n) n->kind = k;
  return n;
}

void jfree(JNode* n) {
  if (!n) return;
  free(n->str);
  for (size_t i = 0; i < n->n; ++i) {
    if (n->keys) free(n->keys[i]);
    jfree(n->items[i]);
  }
  free(n->keys);
  free(n->items);
  free(n);
}

/* Append UTF-8 of codepoint cp to buf[*len] (buf has room for 4). */
static void jput_utf8(char* buf, size_t* len, unsigned cp) {
  if (cp < 0x80) {
    buf[(*len)++] = (char)cp;
  } else if (cp < 0x800) {
    buf[(*len)++] = (char)(0xC0 | (cp >> 6));
    buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    buf[(*len)++] = (char)(0xE0 | (cp >> 12));
    buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
  } else {
    buf[(*len)++] = (char)(0xF0 | (cp >> 18));
    buf[(*len)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
  }
}

static int jhex4(JP* j, unsigned* out) {
  if (j->end - j->p < 4) return -1;
  unsigned v = 0;
  for (int i = 0; i < 4; ++i) {
    char c = j->p[i];
    v <<= 4;
    if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
    else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
    else return -1;
  }
  j->p += 4;
  *out = v;
  return 0;
}

/* Parse a JSON string (j->p is at the opening quote). Returns malloc'd decoded
 * bytes (NUL-terminated) and sets *len; NULL on error. */
static char* jparse_string_raw(JP* j, size_t* len) {
  if (j->p >= j->end || *j->p != '"') { j->err = 1; return NULL; }
  j->p++;
  size_t cap = 16, n = 0;
  char* out = malloc(cap);
  if (!out) { j->err = 1; return NULL; }
  while (j->p < j->end) {
    unsigned char c = (unsigned char)*j->p++;
    if (c == '"') {
      if (n + 1 > cap) { char* t = realloc(out, n + 1); if (!t) goto oom; out = t; }
      out[n] = 0;
      *len = n;
      return out;
    }
    /* ensure room for up to 4 bytes + NUL */
    if (n + 5 > cap) { cap = cap * 2 + 8; char* t = realloc(out, cap); if (!t) goto oom; out = t; }
    if (c == '\\') {
      if (j->p >= j->end) break;
      char e = *j->p++;
      switch (e) {
        case '"': out[n++] = '"'; break;
        case '\\': out[n++] = '\\'; break;
        case '/': out[n++] = '/'; break;
        case 'n': out[n++] = '\n'; break;
        case 't': out[n++] = '\t'; break;
        case 'r': out[n++] = '\r'; break;
        case 'b': out[n++] = '\b'; break;
        case 'f': out[n++] = '\f'; break;
        case 'u': {
          unsigned cp;
          if (jhex4(j, &cp) != 0) goto err;
          if (cp >= 0xD800 && cp <= 0xDBFF) { /* high surrogate: need low */
            if (j->end - j->p >= 2 && j->p[0] == '\\' && j->p[1] == 'u') {
              j->p += 2;
              unsigned lo;
              if (jhex4(j, &lo) != 0) goto err;
              if (lo >= 0xDC00 && lo <= 0xDFFF)
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              else { cp = 0xFFFD; j->p -= 6; } /* not a low surrogate: emit U+FFFD */
            } else {
              cp = 0xFFFD;
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD; /* lone low surrogate */
          }
          jput_utf8(out, &n, cp);
          break;
        }
        default: goto err; /* invalid escape */
      }
    } else if (c < 0x20) {
      goto err; /* control char must be escaped */
    } else {
      out[n++] = (char)c;
    }
  }
err:
  free(out);
  j->err = 1;
  return NULL;
oom:
  free(out);
  j->err = 1;
  return NULL;
}

static JNode* jparse_string(JP* j) {
  size_t len = 0;
  char* s = jparse_string_raw(j, &len);
  if (!s) return NULL;
  JNode* n = jnode(J_STR);
  if (!n) { free(s); j->err = 1; return NULL; }
  n->str = s;
  n->slen = len;
  return n;
}

static JNode* jparse_number(JP* j) {
  const char* start = j->p;
  /* accept  -?digits(.digits)?([eE][+-]?digits)?  */
  if (j->p < j->end && (*j->p == '-' || *j->p == '+')) j->p++;
  int any = 0;
  while (j->p < j->end && isdigit((unsigned char)*j->p)) { j->p++; any = 1; }
  if (j->p < j->end && *j->p == '.') {
    j->p++;
    while (j->p < j->end && isdigit((unsigned char)*j->p)) { j->p++; any = 1; }
  }
  if (j->p < j->end && (*j->p == 'e' || *j->p == 'E')) {
    j->p++;
    if (j->p < j->end && (*j->p == '-' || *j->p == '+')) j->p++;
    while (j->p < j->end && isdigit((unsigned char)*j->p)) j->p++;
  }
  if (!any) { j->err = 1; return NULL; }
  JNode* n = jnode(J_NUM);
  if (!n) { j->err = 1; return NULL; }
  n->num = strtod(start, NULL); /* buffer is NUL-terminated by the caller */
  return n;
}

static JNode* jparse_lit(JP* j, const char* lit, JKind k, int boolean) {
  size_t l = strlen(lit);
  if ((size_t)(j->end - j->p) < l || memcmp(j->p, lit, l) != 0) { j->err = 1; return NULL; }
  j->p += l;
  JNode* n = jnode(k);
  if (!n) { j->err = 1; return NULL; }
  n->boolean = boolean;
  return n;
}

static int jpush(JNode* n, char* key, JNode* val) {
  JNode** ni = realloc(n->items, (n->n + 1) * sizeof(JNode*));
  if (!ni) return -1;
  n->items = ni;
  if (key || n->keys) { /* object */
    char** nk = realloc(n->keys, (n->n + 1) * sizeof(char*));
    if (!nk) return -1;
    n->keys = nk;
    n->keys[n->n] = key;
  }
  n->items[n->n] = val;
  n->n++;
  return 0;
}

static JNode* jparse_array(JP* j) {
  j->p++; /* [ */
  JNode* n = jnode(J_ARR);
  if (!n) { j->err = 1; return NULL; }
  jskip_ws(j);
  if (j->p < j->end && *j->p == ']') { j->p++; return n; }
  for (;;) {
    JNode* v = jparse_value(j);
    if (!v) { jfree(n); return NULL; }
    if (jpush(n, NULL, v) != 0) { jfree(v); jfree(n); j->err = 1; return NULL; }
    jskip_ws(j);
    if (j->p >= j->end) { jfree(n); j->err = 1; return NULL; }
    if (*j->p == ',') { j->p++; jskip_ws(j); continue; }
    if (*j->p == ']') { j->p++; return n; }
    jfree(n); j->err = 1; return NULL;
  }
}

static JNode* jparse_object(JP* j) {
  j->p++; /* { */
  JNode* n = jnode(J_OBJ);
  if (!n) { j->err = 1; return NULL; }
  jskip_ws(j);
  if (j->p < j->end && *j->p == '}') { j->p++; return n; }
  for (;;) {
    jskip_ws(j);
    if (j->p >= j->end || *j->p != '"') { jfree(n); j->err = 1; return NULL; }
    size_t klen = 0;
    char* key = jparse_string_raw(j, &klen);
    if (!key) { jfree(n); return NULL; }
    jskip_ws(j);
    if (j->p >= j->end || *j->p != ':') { free(key); jfree(n); j->err = 1; return NULL; }
    j->p++;
    JNode* v = jparse_value(j);
    if (!v) { free(key); jfree(n); return NULL; }
    if (jpush(n, key, v) != 0) { free(key); jfree(v); jfree(n); j->err = 1; return NULL; }
    jskip_ws(j);
    if (j->p >= j->end) { jfree(n); j->err = 1; return NULL; }
    if (*j->p == ',') { j->p++; continue; }
    if (*j->p == '}') { j->p++; return n; }
    jfree(n); j->err = 1; return NULL;
  }
}

static JNode* jparse_value(JP* j) {
  if (++j->depth > J_MAX_DEPTH) { j->err = 1; j->depth--; return NULL; }
  jskip_ws(j);
  JNode* r = NULL;
  if (j->p >= j->end) { j->err = 1; }
  else {
    char c = *j->p;
    if (c == '"') r = jparse_string(j);
    else if (c == '{') r = jparse_object(j);
    else if (c == '[') r = jparse_array(j);
    else if (c == 't') r = jparse_lit(j, "true", J_BOOL, 1);
    else if (c == 'f') r = jparse_lit(j, "false", J_BOOL, 0);
    else if (c == 'n') r = jparse_lit(j, "null", J_NULL, 0);
    else if (c == '-' || c == '+' || isdigit((unsigned char)c)) r = jparse_number(j);
    else j->err = 1;
  }
  j->depth--;
  return r;
}

/* Parse a whole NUL-terminated buffer; trailing whitespace allowed. */
JNode* json_parse(const char* buf, size_t len) {
  JP j = {buf, buf + len, 0, 0};
  JNode* n = jparse_value(&j);
  if (!n) return NULL;
  jskip_ws(&j);
  if (j.err || j.p != j.end) { jfree(n); return NULL; }
  return n;
}

/* ---- DOM accessors -------------------------------------------------------- */

static JNode* jobj_get(const JNode* o, const char* key) {
  if (!o || o->kind != J_OBJ) return NULL;
  for (size_t i = 0; i < o->n; ++i)
    if (o->keys[i] && strcmp(o->keys[i], key) == 0) return o->items[i];
  return NULL;
}
static int jget_u32(const JNode* o, const char* key, uint32_t* out) {
  JNode* v = jobj_get(o, key);
  if (!v || v->kind != J_NUM || v->num < 0) return 0;
  *out = (uint32_t)(v->num + 0.5);
  return 1;
}
static int jget_f32(const JNode* o, const char* key, float* out) {
  JNode* v = jobj_get(o, key);
  if (!v || v->kind != J_NUM) return 0;
  *out = (float)v->num;
  return 1;
}
static const char* jget_str(const JNode* o, const char* key) {
  JNode* v = jobj_get(o, key);
  return (v && v->kind == J_STR) ? v->str : NULL;
}

/* ======================================================================== *
 *  File helpers
 * ======================================================================== */

/* Read an entire file into a malloc'd, NUL-terminated buffer. NULL on error. */
static char* read_file(const char* path, size_t* len_out) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long sz = ftell(f);
  if (sz < 0) { fclose(f); return NULL; }
  if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
  char* buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
  buf[sz] = 0;
  fclose(f);
  if (len_out) *len_out = (size_t)sz;
  return buf;
}

static int is_dir(const char* path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}
static int is_file(const char* path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}
static int ends_with(const char* s, const char* suf) {
  size_t ls = strlen(s), lf = strlen(suf);
  return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

/* ======================================================================== *
 *  SafeTensors reader (mmap the data; parse the JSON header)
 * ======================================================================== */

typedef struct {
  char* name;          /* owned */
  uint32_t src_type;   /* OC_F32 / OC_F16 / OC_BF16 */
  uint32_t ndim;
  uint64_t shape[GGUF_MAX_DIMS]; /* HF order [out, in, ...] */
  const uint8_t* data; /* into an mmap */
  uint64_t nbytes;
} StTensor;

typedef struct {
  void* map;
  size_t size;
} StMap;

typedef struct {
  StTensor* t;
  size_t n, cap;
  StMap* maps;
  size_t n_maps, cap_maps;
} StModel;

static int st_dtype(const char* d, uint32_t* out) {
  if (!strcmp(d, "F32")) { *out = OC_F32; return 0; }
  if (!strcmp(d, "F16")) { *out = OC_F16; return 0; }
  if (!strcmp(d, "BF16")) { *out = OC_BF16; return 0; }
  return -1; /* F8/I64/BOOL/... not supported */
}

static void st_free(StModel* s) {
  for (size_t i = 0; i < s->n; ++i) free(s->t[i].name);
  free(s->t);
  for (size_t i = 0; i < s->n_maps; ++i)
    if (s->maps[i].map) munmap(s->maps[i].map, s->maps[i].size);
  free(s->maps);
  memset(s, 0, sizeof(*s));
}

/* mmap one .safetensors file and append its tensor views to s. */
static int st_load_file(StModel* s, const char* path, char* err, size_t errlen) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) { snprintf(err, errlen, "convert: cannot open %s", path); return -1; }
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 8) {
    snprintf(err, errlen, "convert: %s too small to be safetensors", path);
    close(fd);
    return -1;
  }
  size_t fsize = (size_t)st.st_size;
  void* map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) { snprintf(err, errlen, "convert: mmap %s failed", path); return -1; }

  const uint8_t* base = map;
  uint64_t hlen;
  memcpy(&hlen, base, 8); /* little-endian host assumed (x86-64) */
  if (hlen > fsize - 8) {
    snprintf(err, errlen, "convert: %s header length %" PRIu64 " exceeds file", path, hlen);
    munmap(map, fsize);
    return -1;
  }
  const uint8_t* data_base = base + 8 + hlen;
  uint64_t data_avail = fsize - 8 - hlen;

  /* header JSON -> NUL-terminated copy for the parser */
  char* hj = malloc((size_t)hlen + 1);
  if (!hj) { snprintf(err, errlen, "convert: oom"); munmap(map, fsize); return -1; }
  memcpy(hj, base + 8, (size_t)hlen);
  hj[hlen] = 0;
  JNode* root = json_parse(hj, (size_t)hlen);
  free(hj);
  if (!root || root->kind != J_OBJ) {
    snprintf(err, errlen, "convert: %s has a malformed safetensors header", path);
    jfree(root);
    munmap(map, fsize);
    return -1;
  }

  /* register the mmap so it outlives the write loop */
  if (s->n_maps == s->cap_maps) {
    s->cap_maps = s->cap_maps ? s->cap_maps * 2 : 4;
    StMap* nm = realloc(s->maps, s->cap_maps * sizeof(StMap));
    if (!nm) { snprintf(err, errlen, "convert: oom"); jfree(root); munmap(map, fsize); return -1; }
    s->maps = nm;
  }
  s->maps[s->n_maps].map = map;
  s->maps[s->n_maps].size = fsize;
  s->n_maps++;

  int rc = -1;
  for (size_t i = 0; i < root->n; ++i) {
    const char* name = root->keys[i];
    JNode* meta = root->items[i];
    if (strcmp(name, "__metadata__") == 0) continue;
    if (meta->kind != J_OBJ) { snprintf(err, errlen, "convert: %s tensor entry not an object", name); goto done; }
    const char* dtype = jget_str(meta, "dtype");
    JNode* shape = jobj_get(meta, "shape");
    JNode* offs = jobj_get(meta, "data_offsets");
    if (!dtype || !shape || shape->kind != J_ARR || !offs || offs->kind != J_ARR || offs->n != 2) {
      snprintf(err, errlen, "convert: %s missing dtype/shape/data_offsets", name);
      goto done;
    }
    uint32_t ty;
    if (st_dtype(dtype, &ty) != 0) {
      snprintf(err, errlen, "convert: %s has unsupported dtype %s (only F32/F16/BF16)", name, dtype);
      goto done;
    }
    if (shape->n < 1 || shape->n > GGUF_MAX_DIMS) {
      snprintf(err, errlen, "convert: %s has %zu dims (1..%d supported)", name, shape->n, GGUF_MAX_DIMS);
      goto done;
    }
    if (offs->items[0]->kind != J_NUM || offs->items[1]->kind != J_NUM) {
      snprintf(err, errlen, "convert: %s data_offsets not numbers", name);
      goto done;
    }
    double d0 = offs->items[0]->num, d1 = offs->items[1]->num;
    if (d0 < 0 || d1 < d0 || d1 > (double)data_avail) {
      snprintf(err, errlen, "convert: %s data_offsets out of range", name);
      goto done;
    }
    uint64_t start = (uint64_t)d0, endo = (uint64_t)d1, nb = endo - start;

    /* element count from shape, and byte-length cross-check.
       Both multiplies are overflow-checked: a hostile header (e.g.
       shape=[2^62,4], or [2^40,2^23] which wraps only in the *elem_bytes
       step) can otherwise wrap the uint64 product to a small value / 0,
       passing this byte check and corrupting the heap in emit_payload. Once
       nelem*elem_bytes cannot wrap and must equal nb (bounded by data_avail),
       every downstream product (rows, out_bytes) is bounded by the real file. */
    uint64_t nelem = 1;
    uint64_t shp[GGUF_MAX_DIMS];
    for (size_t d = 0; d < shape->n; ++d) {
      if (shape->items[d]->kind != J_NUM || shape->items[d]->num < 0) {
        snprintf(err, errlen, "convert: %s has a bad shape entry", name);
        goto done;
      }
      shp[d] = (uint64_t)shape->items[d]->num;
      uint64_t dim = shp[d] ? shp[d] : 1;
      if (nelem > UINT64_MAX / dim) {
        snprintf(err, errlen, "convert: %s shape overflows uint64", name);
        goto done;
      }
      nelem *= dim;
    }
    uint64_t elem_bytes = (ty == OC_F32) ? 4 : 2;
    if (nelem > UINT64_MAX / elem_bytes) {
      snprintf(err, errlen, "convert: %s shape overflows uint64", name);
      goto done;
    }
    if (nb != nelem * elem_bytes) {
      snprintf(err, errlen, "convert: %s byte length %" PRIu64 " != %" PRIu64 " (shape*dtype)",
               name, nb, nelem * elem_bytes);
      goto done;
    }

    /* dedup: a name may legitimately appear only once across all shards */
    for (size_t k = 0; k < s->n; ++k)
      if (strcmp(s->t[k].name, name) == 0) {
        snprintf(err, errlen, "convert: duplicate tensor %s across shards", name);
        goto done;
      }

    if (s->n == s->cap) {
      s->cap = s->cap ? s->cap * 2 : 64;
      StTensor* nt = realloc(s->t, s->cap * sizeof(StTensor));
      if (!nt) { snprintf(err, errlen, "convert: oom"); goto done; }
      s->t = nt;
    }
    StTensor* t = &s->t[s->n++];
    memset(t, 0, sizeof(*t));
    t->name = strdup(name);
    if (!t->name) { s->n--; snprintf(err, errlen, "convert: oom"); goto done; }
    t->src_type = ty;
    t->ndim = (uint32_t)shape->n;
    for (size_t d = 0; d < shape->n; ++d) t->shape[d] = shp[d];
    t->data = data_base + start;
    t->nbytes = nb;
  }
  rc = 0;
done:
  jfree(root);
  return rc;
}

/* Load a single file or every *.safetensors in a directory. */
static int st_load(StModel* s, const char* input, char* err, size_t errlen) {
  memset(s, 0, sizeof(*s));
  if (is_file(input)) return st_load_file(s, input, err, errlen);
  if (!is_dir(input)) { snprintf(err, errlen, "convert: %s is neither file nor dir", input); return -1; }

  DIR* d = opendir(input);
  if (!d) { snprintf(err, errlen, "convert: cannot read dir %s", input); return -1; }
  /* collect + sort names for a deterministic, shard-order-independent read */
  char** names = NULL;
  size_t nn = 0, cn = 0;
  struct dirent* e;
  while ((e = readdir(d)) != NULL) {
    if (!ends_with(e->d_name, ".safetensors")) continue;
    if (nn == cn) { cn = cn ? cn * 2 : 8; names = realloc(names, cn * sizeof(char*)); if (!names) { closedir(d); snprintf(err, errlen, "convert: oom"); return -1; } }
    names[nn++] = strdup(e->d_name);
  }
  closedir(d);
  if (nn == 0) { free(names); snprintf(err, errlen, "convert: no .safetensors in %s", input); return -1; }
  for (size_t i = 0; i + 1 < nn; ++i) /* tiny insertion sort */
    for (size_t k = i + 1; k < nn; ++k)
      if (strcmp(names[k], names[i]) < 0) { char* t = names[i]; names[i] = names[k]; names[k] = t; }

  int rc = 0;
  for (size_t i = 0; i < nn; ++i) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", input, names[i]);
    if (st_load_file(s, path, err, errlen) != 0) { rc = -1; break; }
  }
  for (size_t i = 0; i < nn; ++i) free(names[i]);
  free(names);
  return rc;
}

/* ======================================================================== *
 *  Architecture whitelist + HF -> ggml tensor-name map
 * ======================================================================== */

/* Normalize an HF model_type to a supported ggml arch string, or NULL if the
 * arch is out of scope. `*permute` is set for the ggml-NORMAL-rope families.
 * `*sandwich` is set for Gemma2+ (4 RMSNorms per layer); those are always
 * written as GGUF arch "gemma4" so model_gemma4.c can load them. */
static const char* arch_map(const char* mt, int* permute, int* sandwich) {
  char lc[64];
  size_t i = 0;
  for (; mt[i] && i < sizeof lc - 1; ++i) lc[i] = (char)tolower((unsigned char)mt[i]);
  lc[i] = 0;
  *sandwich = 0;
  if (!strcmp(lc, "llama") || !strcmp(lc, "mistral") || !strcmp(lc, "yi")) {
    *permute = 1;
    return strcmp(lc, "yi") == 0 ? "yi" : (strcmp(lc, "mistral") == 0 ? "mistral" : "llama");
  }
  if (!strcmp(lc, "qwen2")) { *permute = 0; return "qwen2"; }
  if (!strcmp(lc, "qwen3")) { *permute = 0; return "qwen3"; }
  /* phi / phi3: NeoX rope, unpermuted; fused HF weights are split below. */
  if (!strcmp(lc, "phi") || !strcmp(lc, "phi3")) {
    *permute = 0;
    return "phi3";
  }
  /* gemma2/3/4 share the sandwich layout the C gemma4 loader expects. Plain
   * "gemma" (v1) is 2-norm Llama-like and is rejected — no silently-wrong map. */
  if (!strcmp(lc, "gemma2") || !strcmp(lc, "gemma3") || !strcmp(lc, "gemma4")) {
    *permute = 0;
    *sandwich = 1;
    return "gemma4";
  }
  return NULL;
}

/* Multimodal HF checkpoints nest text weights under model.language_model.*. */
static void normalize_hf_name(const char* hf, char* out, size_t outlen) {
  if (strncmp(hf, "model.language_model.", 21) == 0)
    snprintf(out, outlen, "model.%s", hf + 21);
  else
    snprintf(out, outlen, "%s", hf);
}

/* HF tensor name -> ggml/llama.cpp name. `sandwich` selects Gemma2+ norm map.
 * Returns 1 and fills `out` (>= 160 bytes) if emitted; 0 if dropped; -1 if the
 * name is an unsupported structure (MoE / vision) so the convert fails.
 * Fused phi3 qkv_proj / gate_up_proj are NOT rejected here — the plan loop
 * splits them into separate OutTs before calling map_name. */
static int map_name(const char* hf_raw, char* out, size_t outlen, int sandwich) {
  char hf[256];
  normalize_hf_name(hf_raw, hf, sizeof hf);

  if (strstr(hf, ".experts.") || strstr(hf, "block_sparse_moe") ||
      strstr(hf, ".mtp.") || strstr(hf, "linear_attn") ||
      strstr(hf, "vision_tower") || strstr(hf, "multi_modal") ||
      strstr(hf, "audio_tower"))
    return -1;

  /* Fused tensors handled by the plan loop; ignore here if seen. */
  if (strstr(hf, "self_attn.qkv_proj") || strstr(hf, "gate_up_proj"))
    return 0;

  if (!strcmp(hf, "model.embed_tokens.weight")) {
    snprintf(out, outlen, "token_embd.weight");
    return 1;
  }
  /* Gemma ties the head to the embedding; emitting output.weight is redundant. */
  if (!strcmp(hf, "lm_head.weight")) {
    if (sandwich) return 0;
    snprintf(out, outlen, "output.weight");
    return 1;
  }
  if (!strcmp(hf, "model.norm.weight")) {
    snprintf(out, outlen, "output_norm.weight");
    return 1;
  }
  if (strstr(hf, "rotary_emb.inv_freq")) return 0;

  const char* rest = NULL;
  if (strncmp(hf, "model.layers.", 13) == 0) rest = hf + 13;
  else return 0;

  char* dot = strchr(rest, '.');
  if (!dot) return 0;
  char layer[16];
  size_t ll = (size_t)(dot - rest);
  if (ll == 0 || ll >= sizeof layer) return 0;
  memcpy(layer, rest, ll);
  layer[ll] = 0;
  for (size_t i = 0; i < ll; ++i) if (!isdigit((unsigned char)layer[i])) return 0;
  const char* suf = dot + 1;

  /* Llama-style: post_attention_layernorm is the FFN pre-norm.
   * Gemma sandwich: that tensor is post-attn residual norm; FFN pre-norm is
   * pre_feedforward_layernorm; post_feedforward is the residual FFN norm. */
  static const struct { const char* hf; const char* gg; } M_LLAMA[] = {
      {"input_layernorm.weight", "attn_norm.weight"},
      {"post_attention_layernorm.weight", "ffn_norm.weight"},
      {"self_attn.q_proj.weight", "attn_q.weight"},
      {"self_attn.k_proj.weight", "attn_k.weight"},
      {"self_attn.v_proj.weight", "attn_v.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      {"self_attn.q_proj.bias", "attn_q.bias"},
      {"self_attn.k_proj.bias", "attn_k.bias"},
      {"self_attn.v_proj.bias", "attn_v.bias"},
      {"self_attn.o_proj.bias", "attn_output.bias"},
      {"self_attn.q_norm.weight", "attn_q_norm.weight"},
      {"self_attn.k_norm.weight", "attn_k_norm.weight"},
      {"mlp.gate_proj.weight", "ffn_gate.weight"},
      {"mlp.up_proj.weight", "ffn_up.weight"},
      {"mlp.down_proj.weight", "ffn_down.weight"},
  };
  static const struct { const char* hf; const char* gg; } M_GEMMA[] = {
      {"input_layernorm.weight", "attn_norm.weight"},
      {"post_attention_layernorm.weight", "post_attention_norm.weight"},
      {"pre_feedforward_layernorm.weight", "ffn_norm.weight"},
      {"post_feedforward_layernorm.weight", "post_ffw_norm.weight"},
      {"layer_scalar", "layer_output_scale.weight"},
      {"layer_scalar.weight", "layer_output_scale.weight"},
      {"self_attn.q_proj.weight", "attn_q.weight"},
      {"self_attn.k_proj.weight", "attn_k.weight"},
      {"self_attn.v_proj.weight", "attn_v.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      {"self_attn.q_norm.weight", "attn_q_norm.weight"},
      {"self_attn.k_norm.weight", "attn_k_norm.weight"},
      {"mlp.gate_proj.weight", "ffn_gate.weight"},
      {"mlp.up_proj.weight", "ffn_up.weight"},
      {"mlp.down_proj.weight", "ffn_down.weight"},
  };
  if (sandwich) {
    for (size_t i = 0; i < sizeof M_GEMMA / sizeof *M_GEMMA; ++i)
      if (!strcmp(suf, M_GEMMA[i].hf)) {
        snprintf(out, outlen, "blk.%s.%s", layer, M_GEMMA[i].gg);
        return 1;
      }
  } else {
    for (size_t i = 0; i < sizeof M_LLAMA / sizeof *M_LLAMA; ++i)
      if (!strcmp(suf, M_LLAMA[i].hf)) {
        snprintf(out, outlen, "blk.%s.%s", layer, M_LLAMA[i].gg);
        return 1;
      }
  }
  return 0;
}

static int is_norm_weight(const char* gg) {
  return strstr(gg, "_norm.weight") != NULL || !strcmp(gg, "output_norm.weight");
}

/* Is this ggml name a q/k projection weight that needs the permute? */
static int is_qk_weight(const char* gg, int* is_k) {
  if (ends_with(gg, ".attn_q.weight")) { *is_k = 0; return 1; }
  if (ends_with(gg, ".attn_k.weight")) { *is_k = 1; return 1; }
  return 0;
}

/* ======================================================================== *
 *  GGUF KV metadata builder (owns keys + string/array values)
 * ======================================================================== */

typedef struct {
  GgufKv* kv;
  size_t n, cap;
} KvBuf;

static int kv_reserve(KvBuf* b) {
  if (b->n == b->cap) {
    b->cap = b->cap ? b->cap * 2 : 16;
    GgufKv* nk = realloc(b->kv, b->cap * sizeof(GgufKv));
    if (!nk) return -1;
    b->kv = nk;
  }
  return 0;
}
static void kv_u32(KvBuf* b, const char* k, uint32_t v) {
  if (kv_reserve(b)) return;
  b->kv[b->n].key = strdup(k);
  b->kv[b->n].val.kind = GGUF_T_U32;
  b->kv[b->n].val.v.u = v;
  b->n++;
}
static void kv_f32(KvBuf* b, const char* k, float v) {
  if (kv_reserve(b)) return;
  b->kv[b->n].key = strdup(k);
  b->kv[b->n].val.kind = GGUF_T_F32;
  b->kv[b->n].val.v.f = v;
  b->n++;
}
static void kv_str(KvBuf* b, const char* k, const char* v) {
  if (kv_reserve(b)) return;
  b->kv[b->n].key = strdup(k);
  b->kv[b->n].val.kind = GGUF_T_STRING;
  char* c = strdup(v);
  b->kv[b->n].val.v.str.ptr = c;
  b->kv[b->n].val.v.str.len = c ? strlen(c) : 0;
  b->n++;
}
/* Takes ownership of items[] and each item's malloc'd string payload. */
static void kv_arr(KvBuf* b, const char* k, int elem_kind, GgufValue* items, size_t n) {
  if (kv_reserve(b)) return;
  b->kv[b->n].key = strdup(k);
  b->kv[b->n].val.kind = GGUF_T_ARRAY;
  b->kv[b->n].val.v.arr.elem_kind = elem_kind;
  b->kv[b->n].val.v.arr.items = items;
  b->kv[b->n].val.v.arr.n = n;
  b->n++;
}

static void kv_value_free(GgufValue* v) {
  if (v->kind == GGUF_T_STRING) {
    free((char*)v->v.str.ptr);
  } else if (v->kind == GGUF_T_ARRAY) {
    for (size_t i = 0; i < v->v.arr.n; ++i) kv_value_free(&v->v.arr.items[i]);
    free(v->v.arr.items);
  }
}
static void kv_free(KvBuf* b) {
  for (size_t i = 0; i < b->n; ++i) {
    free(b->kv[i].key);
    kv_value_free(&b->kv[i].val);
  }
  free(b->kv);
  memset(b, 0, sizeof(*b));
}

/* ======================================================================== *
 *  Tokenizer embedding (tokenizer.json -> tokenizer.ggml.*)
 * ======================================================================== */

/* Build a GGUF string-array value set from n owned C strings. */
static GgufValue* mk_str_items(char** strs, const size_t* lens, size_t n) {
  GgufValue* items = calloc(n ? n : 1, sizeof(GgufValue));
  if (!items) return NULL;
  for (size_t i = 0; i < n; ++i) {
    items[i].kind = GGUF_T_STRING;
    items[i].v.str.ptr = strs[i]; /* ownership moves into the array */
    items[i].v.str.len = lens[i];
  }
  return items;
}

/* Read bos/eos/unk/pad ids from config.json / tokenizer_config.json. */
static void tok_special_ids(KvBuf* kb, const JNode* cfg, const JNode* tcfg) {
  const JNode* src[2] = {cfg, tcfg};
  const struct { const char* field; const char* key; } ids[] = {
      {"bos_token_id", "tokenizer.ggml.bos_token_id"},
      {"eos_token_id", "tokenizer.ggml.eos_token_id"},
      {"pad_token_id", "tokenizer.ggml.padding_token_id"},
      {"unk_token_id", "tokenizer.ggml.unknown_token_id"},
  };
  for (size_t s = 0; s < 2; ++s) {
    if (!src[s]) continue;
    for (size_t i = 0; i < sizeof ids / sizeof *ids; ++i) {
      uint32_t v;
      if (jget_u32(src[s], ids[i].field, &v)) kv_u32(kb, ids[i].key, v);
    }
  }
}

/* Returns 0 if a tokenizer was embedded, -1 on a hard error (unsupported type
 * or malformed vocab -> the whole convert must fail, never a broken vocab). */
static int embed_tokenizer(KvBuf* kb, const char* dir, const JNode* cfg,
                           char* err, size_t errlen) {
  char tp[4128];
  snprintf(tp, sizeof tp, "%s/tokenizer.json", dir);
  if (!is_file(tp)) {
    fprintf(stderr, "convert: warning: no tokenizer.json in %.400s; GGUF will have no embedded tokenizer\n", dir);
    return 0; /* absent tokenizer is a warning, not a failure */
  }
  size_t len = 0;
  char* raw = read_file(tp, &len);
  if (!raw) { snprintf(err, errlen, "convert: cannot read %.400s", tp); return -1; }
  JNode* tok = json_parse(raw, len);
  free(raw);
  if (!tok) { snprintf(err, errlen, "convert: malformed tokenizer.json"); return -1; }

  int rc = -1;
  JNode* model = jobj_get(tok, "model");
  const char* mtype = model ? jget_str(model, "type") : NULL;
  JNode* vocab = model ? jobj_get(model, "vocab") : NULL;
  JNode* added = jobj_get(tok, "added_tokens");
  if (!model || !mtype || !vocab) { snprintf(err, errlen, "convert: tokenizer.json missing model/type/vocab"); goto done; }

  int is_bpe = !strcasecmp(mtype, "bpe");
  int is_unigram = !strcasecmp(mtype, "unigram");
  if (!is_bpe && !is_unigram) {
    snprintf(err, errlen, "convert: tokenizer type '%s' unsupported (only BPE and Unigram)", mtype);
    goto done;
  }

  /* Determine vocab size (max id + 1), including added_tokens. */
  uint64_t maxid = 0;
  if (is_bpe) {
    if (vocab->kind != J_OBJ) { snprintf(err, errlen, "convert: BPE vocab is not an object"); goto done; }
    for (size_t i = 0; i < vocab->n; ++i)
      if (vocab->items[i]->kind == J_NUM && vocab->items[i]->num >= 0) {
        uint64_t id = (uint64_t)vocab->items[i]->num;
        if (id > maxid) maxid = id;
      }
  } else {
    if (vocab->kind != J_ARR) { snprintf(err, errlen, "convert: Unigram vocab is not an array"); goto done; }
    if (vocab->n > 0) maxid = vocab->n - 1;
  }
  if (added && added->kind == J_ARR)
    for (size_t i = 0; i < added->n; ++i) {
      uint32_t id;
      if (jget_u32(added->items[i], "id", &id) && id > maxid) maxid = id;
    }
  size_t nvocab = (size_t)maxid + 1;
  if (nvocab == 0 || nvocab > 5000000) { snprintf(err, errlen, "convert: implausible vocab size %zu", nvocab); goto done; }

  /* dense token/type/score arrays */
  char** toks = calloc(nvocab, sizeof(char*));
  size_t* tlens = calloc(nvocab, sizeof(size_t));
  int32_t* ttype = calloc(nvocab, sizeof(int32_t));
  float* score = calloc(nvocab, sizeof(float));
  if (!toks || !tlens || !ttype || !score) { snprintf(err, errlen, "convert: oom"); free(toks); free(tlens); free(ttype); free(score); goto done; }
  for (size_t i = 0; i < nvocab; ++i) { toks[i] = NULL; ttype[i] = 1; } /* NORMAL */

  if (is_bpe) {
    for (size_t i = 0; i < vocab->n; ++i) {
      if (vocab->items[i]->kind != J_NUM || vocab->items[i]->num < 0) continue;
      uint64_t id = (uint64_t)vocab->items[i]->num;
      if (id >= nvocab) continue;
      const char* key = vocab->keys[i];
      free(toks[id]);
      toks[id] = strdup(key);
      tlens[id] = strlen(key);
    }
  } else {
    for (size_t i = 0; i < vocab->n && i < nvocab; ++i) {
      JNode* pair = vocab->items[i];
      if (pair->kind != J_ARR || pair->n < 2) continue;
      if (pair->items[0]->kind == J_STR) {
        free(toks[i]);
        toks[i] = malloc(pair->items[0]->slen + 1);
        if (toks[i]) { memcpy(toks[i], pair->items[0]->str, pair->items[0]->slen + 1); tlens[i] = pair->items[0]->slen; }
      }
      if (pair->items[1]->kind == J_NUM) score[i] = (float)pair->items[1]->num;
    }
  }

  /* added_tokens override content + mark specials as CONTROL(3) */
  if (added && added->kind == J_ARR)
    for (size_t i = 0; i < added->n; ++i) {
      JNode* a = added->items[i];
      uint32_t id;
      if (!jget_u32(a, "id", &id) || id >= nvocab) continue;
      const char* content = jget_str(a, "content");
      if (content) { free(toks[id]); toks[id] = strdup(content); tlens[id] = strlen(content); }
      JNode* sp = jobj_get(a, "special");
      if (sp && sp->kind == J_BOOL && sp->boolean) ttype[id] = 3;
    }

  /* any gaps (unfilled ids) become empty strings so id==index stays dense */
  for (size_t i = 0; i < nvocab; ++i)
    if (!toks[i]) { toks[i] = strdup(""); tlens[i] = 0; }

  /* -> GGUF KV arrays */
  GgufValue* tok_items = mk_str_items(toks, tlens, nvocab);
  free(toks); /* pointers moved into tok_items */
  free(tlens);
  if (!tok_items) { snprintf(err, errlen, "convert: oom"); free(ttype); free(score); goto done; }
  kv_str(kb, "tokenizer.ggml.model", is_bpe ? "gpt2" : "llama");
  kv_arr(kb, "tokenizer.ggml.tokens", GGUF_T_STRING, tok_items, nvocab);

  GgufValue* type_items = calloc(nvocab, sizeof(GgufValue));
  if (type_items) {
    for (size_t i = 0; i < nvocab; ++i) { type_items[i].kind = GGUF_T_I32; type_items[i].v.i = ttype[i]; }
    kv_arr(kb, "tokenizer.ggml.token_type", GGUF_T_I32, type_items, nvocab);
  }
  free(ttype);

  if (is_unigram) {
    GgufValue* score_items = calloc(nvocab, sizeof(GgufValue));
    if (score_items) {
      for (size_t i = 0; i < nvocab; ++i) { score_items[i].kind = GGUF_T_F32; score_items[i].v.f = score[i]; }
      kv_arr(kb, "tokenizer.ggml.scores", GGUF_T_F32, score_items, nvocab);
    }
  }
  free(score);

  if (is_bpe) {
    JNode* merges = jobj_get(model, "merges");
    if (merges && merges->kind == J_ARR) {
      size_t nm = merges->n;
      char** ms = calloc(nm ? nm : 1, sizeof(char*));
      size_t* mlen = calloc(nm ? nm : 1, sizeof(size_t));
      size_t mc = 0;
      if (ms && mlen) {
        for (size_t i = 0; i < nm; ++i) {
          JNode* m = merges->items[i];
          if (m->kind == J_STR) {
            ms[mc] = strdup(m->str);
            mlen[mc] = m->slen;
            mc++;
          } else if (m->kind == J_ARR && m->n >= 2 && m->items[0]->kind == J_STR && m->items[1]->kind == J_STR) {
            size_t l = m->items[0]->slen + 1 + m->items[1]->slen;
            char* s = malloc(l + 1);
            if (s) {
              memcpy(s, m->items[0]->str, m->items[0]->slen);
              s[m->items[0]->slen] = ' ';
              memcpy(s + m->items[0]->slen + 1, m->items[1]->str, m->items[1]->slen + 1);
              ms[mc] = s;
              mlen[mc] = l;
              mc++;
            }
          }
        }
        GgufValue* merge_items = mk_str_items(ms, mlen, mc);
        if (merge_items) kv_arr(kb, "tokenizer.ggml.merges", GGUF_T_STRING, merge_items, mc);
        else for (size_t i = 0; i < mc; ++i) free(ms[i]);
      }
      free(ms);
      free(mlen);
    }
  }

  /* special ids from config.json + tokenizer_config.json */
  char tcp[4128];
  snprintf(tcp, sizeof tcp, "%s/tokenizer_config.json", dir);
  JNode* tcfg = NULL;
  size_t tclen = 0;
  char* tcraw = read_file(tcp, &tclen);
  if (tcraw) { tcfg = json_parse(tcraw, tclen); free(tcraw); }
  tok_special_ids(kb, cfg, tcfg);
  jfree(tcfg);

  rc = 0;
done:
  jfree(tok);
  return rc;
}

/* ======================================================================== *
 *  Output plan + write
 * ======================================================================== */

typedef struct {
  const StTensor* src;
  char gg[160];       /* ggml name */
  uint32_t out_type;  /* ggml output type id */
  uint32_t ndim;
  uint64_t dims[GGUF_MAX_DIMS]; /* ggml order = reverse(HF shape) */
  uint64_t cols, rows;
  uint64_t out_bytes;
  int permute;        /* 1 for q/k on permuted archs */
  int norm_plus1;     /* 1 => bake +1 into RMSNorm weights (gemma GGUF contract) */
  uint64_t head_dim;  /* per-head size when permuting */
  uint64_t src_row0;  /* row offset into fused HF source (0 for normal tensors) */
} OutT;

/* choose ggml output type: 1-D tensors stay F32; 2-D uses outtype when the
 * row length is compatible, else F32. */
static uint32_t pick_type(uint32_t outtype, uint32_t ndim, uint64_t cols) {
  if (ndim < 2) return OC_F32; /* norms / biases */
  if (outtype == OC_F32 || outtype == OC_F16) return outtype;
  if (cols % 32 == 0 && oc_row_bytes(outtype, cols) != 0) return outtype;
  return OC_F32; /* fall back rather than reject a non-block-aligned row */
}

/* Encode one output tensor's payload into `dst` (out_bytes), decoding from the
 * mmap source and applying the row permute when set. */
static int emit_payload(const OutT* o, uint8_t* dst, char* err, size_t errlen) {
  uint32_t st = o->src->src_type;
  size_t src_rb = oc_row_bytes(st, o->cols);
  size_t dst_rb = oc_row_bytes(o->out_type, o->cols);
  if (src_rb == 0 || dst_rb == 0) { snprintf(err, errlen, "convert: %s bad row bytes", o->gg); return -1; }
  float* row = malloc(o->cols * sizeof(float));
  if (!row) { snprintf(err, errlen, "convert: oom"); return -1; }
  int rc = -1;
  for (uint64_t d = 0; d < o->rows; ++d) {
    uint64_t s = o->src_row0 + d;
    if (o->permute) {
      uint64_t hd = o->head_dim, half = hd / 2;
      uint64_t head = d / hd, dl = d % hd;
      uint64_t sl = (dl & 1) ? half + dl / 2 : dl / 2;
      s = o->src_row0 + head * hd + sl;
    }
    if (oc_dequant_row(st, o->src->data + s * src_rb, row, o->cols) != 0) {
      snprintf(err, errlen, "convert: %s dequant failed", o->gg);
      goto out;
    }
    if (o->norm_plus1) {
      for (uint64_t c = 0; c < o->cols; ++c) row[c] += 1.0f;
    }
    if (gw_encode_row(o->out_type, row, dst + d * dst_rb, o->cols) != 0) {
      snprintf(err, errlen, "convert: %s encode failed", o->gg);
      goto out;
    }
  }
  rc = 0;
out:
  free(row);
  return rc;
}

int tool_convert(const char* input, const char* output, const ConvertOpts* opts,
                 int verbose) {
  char err[512];
  err[0] = 0;
  int rc = 1;
  StModel st;
  KvBuf kb = {0};
  OutT* outs = NULL;
  GwTensor* gts = NULL;
  GwWriter w = {0};
  JNode* cfg_root = NULL;

  uint32_t outtype = OC_F16;
  if (opts->outtype) {
    outtype = gw_type_id(opts->outtype); /* case-sensitive: caller upper-cases */
    if (outtype == UINT32_MAX || !gw_encodable(outtype)) {
      fprintf(stderr,
              "convert: unsupported --outtype '%s' "
              "(F32 F16 BF16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q2_K Q3_K Q4_K Q5_K Q6_K IQ4_XS IQ2_XXS AL5_XS)\n",
              opts->outtype);
      return 1;
    }
  }

  if (st_load(&st, input, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); goto done; }

  /* config.json lives in the input dir (or the file's dir). */
  char cfgdir[4096];
  if (is_dir(input)) snprintf(cfgdir, sizeof cfgdir, "%s", input);
  else {
    snprintf(cfgdir, sizeof cfgdir, "%s", input);
    char* slash = strrchr(cfgdir, '/');
    if (slash) *slash = 0; else snprintf(cfgdir, sizeof cfgdir, ".");
  }
  char cfgpath[4200];
  snprintf(cfgpath, sizeof cfgpath, "%s/config.json", cfgdir);
  size_t cflen = 0;
  char* cfgraw = read_file(cfgpath, &cflen);
  if (cfgraw) { cfg_root = json_parse(cfgraw, cflen); free(cfgraw); if (!cfg_root) { fprintf(stderr, "convert: malformed config.json\n"); goto done; } }
  else fprintf(stderr, "convert: warning: no config.json at %.400s; the GGUF will lack "
                       "geometry metadata and may fail to load\n", cfgpath);

  /* geometry config object (text_config wins if present) */
  const JNode* cfg = cfg_root;
  JNode* tc = jobj_get(cfg_root, "text_config");
  if (tc && tc->kind == J_OBJ) cfg = tc;

  /* resolve arch */
  const char* model_type = opts->arch_override;
  if (!model_type) model_type = jget_str(cfg, "model_type");
  if (!model_type && cfg_root) model_type = jget_str(cfg_root, "model_type");
  if (!model_type) { fprintf(stderr, "convert: no arch (need config.json model_type or --arch)\n"); goto done; }
  int permute = 0, sandwich = 0;
  const char* arch = arch_map(model_type, &permute, &sandwich);
  if (!arch) {
    fprintf(stderr,
            "convert: architecture '%s' is not supported.\n"
            "  Supported (dense): llama, mistral, yi, qwen2, qwen3, phi3, gemma2, gemma3, gemma4.\n"
            "  Rejected: MoE, gemma-v1, deepseek(MLA), vision/audio towers, etc.\n"
            "  Use llama.cpp's convert_hf_to_gguf.py for those.\n",
            model_type);
    goto done;
  }
  if (sandwich && strcmp(model_type, "gemma4") != 0 && verbose)
    fprintf(stderr, "convert: HF model_type=%s -> GGUF arch gemma4 (C loader family)\n",
            model_type);

  /* head counts (needed for the permute, and emitted as metadata) */
  uint32_t n_head = 0, n_kv = 0;
  jget_u32(cfg, "num_attention_heads", &n_head);
  if (!jget_u32(cfg, "num_key_value_heads", &n_kv)) n_kv = n_head;
  if (permute && n_head == 0) {
    fprintf(stderr, "convert: %s needs num_attention_heads in config.json for the q/k permute\n", arch);
    goto done;
  }

  /* ---- metadata KVs ---- */
  kv_str(&kb, "general.architecture", arch);
  const char* nm = jget_str(cfg_root, "name");
  if (nm) kv_str(&kb, "general.name", nm);
  {
    char pk[96];
    uint32_t u;
    float f;
#define PU(suffix, field) do { snprintf(pk, sizeof pk, "%s." suffix, arch); if (jget_u32(cfg, field, &u)) kv_u32(&kb, pk, u); } while (0)
#define PF(suffix, field) do { snprintf(pk, sizeof pk, "%s." suffix, arch); if (jget_f32(cfg, field, &f)) kv_f32(&kb, pk, f); } while (0)
    PU("embedding_length", "hidden_size");
    PU("block_count", "num_hidden_layers");
    PU("feed_forward_length", "intermediate_size");
    PU("context_length", "max_position_embeddings");
    PU("vocab_size", "vocab_size");
    if (n_head) { snprintf(pk, sizeof pk, "%s.attention.head_count", arch); kv_u32(&kb, pk, n_head); }
    /* Prefer explicit global kv heads when present (gemma4 dual GQA). */
    uint32_t n_kv_emit = n_kv;
    if (sandwich) {
      uint32_t ng = 0;
      if (jget_u32(cfg, "num_global_key_value_heads", &ng) && ng) n_kv_emit = ng;
    }
    { snprintf(pk, sizeof pk, "%s.attention.head_count_kv", arch); kv_u32(&kb, pk, n_kv_emit); }
    PF("attention.layer_norm_rms_epsilon", "rms_norm_eps");
    PF("rope.freq_base", "rope_theta");
    uint32_t hd = 0, hidden = 0;
    if (!jget_u32(cfg, "head_dim", &hd) && !jget_u32(cfg, "global_head_dim", &hd) &&
        jget_u32(cfg, "hidden_size", &hidden) && n_head)
      hd = hidden / n_head;
    if (hd) {
      snprintf(pk, sizeof pk, "%s.attention.key_length", arch); kv_u32(&kb, pk, hd);
      snprintf(pk, sizeof pk, "%s.attention.value_length", arch); kv_u32(&kb, pk, hd);
      if (sandwich) {
        snprintf(pk, sizeof pk, "%s.rope.dimension_count", arch); kv_u32(&kb, pk, hd);
      }
    }
    if (sandwich) {
      uint32_t sw = 0;
      if (jget_u32(cfg, "sliding_window", &sw)) {
        snprintf(pk, sizeof pk, "%s.attention.sliding_window", arch);
        kv_u32(&kb, pk, sw);
      }
      float softcap = 0.0f;
      if (jget_f32(cfg, "final_logit_softcapping", &softcap)) {
        snprintf(pk, sizeof pk, "%s.final_logit_softcapping", arch);
        kv_f32(&kb, pk, softcap);
      }
      float attn_scale = 0.0f;
      if (jget_f32(cfg, "query_pre_attn_scalar", &attn_scale) && attn_scale > 0.0f) {
        /* HF stores the pre-attn scalar; ggml wants 1/sqrt(scale) style — store as-is
         * when present under attention.scale (loader default is 1.0). */
        snprintf(pk, sizeof pk, "%s.attention.scale", arch);
        kv_f32(&kb, pk, 1.0f / sqrtf(attn_scale));
      }
      float prf = 0.0f;
      if (jget_f32(cfg, "partial_rotary_factor", &prf) && prf > 0.0f) {
        snprintf(pk, sizeof pk, "%s.rope.partial_rotary_factor", arch);
        kv_f32(&kb, pk, prf);
      }
      float swa_theta = 0.0f;
      if (jget_f32(cfg, "rope_local_base_freq", &swa_theta) ||
          jget_f32(cfg, "rope_theta_local", &swa_theta)) {
        snprintf(pk, sizeof pk, "%s.rope.freq_base_swa", arch);
        kv_f32(&kb, pk, swa_theta);
      }
      JNode* rope = jobj_get(cfg, "rope_parameters");
      if (rope && rope->kind == J_OBJ) {
        float rb = 0.0f;
        if (jget_f32(rope, "rope_theta", &rb) || jget_f32(rope, "theta", &rb)) {
          snprintf(pk, sizeof pk, "%s.rope.freq_base", arch);
          kv_f32(&kb, pk, rb);
        }
        if (jget_f32(rope, "local_base", &swa_theta) ||
            jget_f32(rope, "rope_theta_local", &swa_theta)) {
          snprintf(pk, sizeof pk, "%s.rope.freq_base_swa", arch);
          kv_f32(&kb, pk, swa_theta);
        }
      }
      /* layer_types: ["sliding_attention","full_attention",...] -> u32 pattern */
      JNode* lt = jobj_get(cfg, "layer_types");
      if (lt && lt->kind == J_ARR && lt->n > 0) {
        GgufValue* items = calloc(lt->n, sizeof(GgufValue));
        if (items) {
          for (size_t i = 0; i < lt->n; ++i) {
            items[i].kind = GGUF_T_U32;
            const char* s = (lt->items[i] && lt->items[i]->kind == J_STR)
                                ? lt->items[i]->str
                                : NULL;
            items[i].v.u = (s && strstr(s, "sliding")) ? 1u : 0u;
          }
          snprintf(pk, sizeof pk, "%s.attention.sliding_window_pattern", arch);
          kv_arr(&kb, pk, GGUF_T_U32, items, lt->n);
        }
      } else {
        uint32_t pat = 0;
        if (jget_u32(cfg, "sliding_window_pattern", &pat)) {
          snprintf(pk, sizeof pk, "%s.attention.sliding_window_pattern", arch);
          kv_u32(&kb, pk, pat);
        }
      }
      uint32_t hd_swa = 0;
      if (jget_u32(cfg, "head_dim_sliding", &hd_swa) ||
          jget_u32(cfg, "sliding_window_head_dim", &hd_swa)) {
        snprintf(pk, sizeof pk, "%s.rope.dimension_count_swa", arch);
        kv_u32(&kb, pk, hd_swa);
      }
    }
#undef PU
#undef PF
  }

  /* ---- tokenizer ---- */
  if (embed_tokenizer(&kb, cfgdir, cfg_root, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); goto done; }

  /* ---- plan output tensors ----
   * Phi3 fused weights expand 1->3 (qkv) or 1->2 (gate_up), so allocate headroom. */
  size_t ocap = st.n * 3 + 8;
  outs = calloc(ocap, sizeof(OutT));
  if (!outs) { fprintf(stderr, "convert: oom\n"); goto done; }
  size_t no = 0;

  uint32_t head_dim_cfg = 0, hidden_cfg = 0, ff_cfg = 0;
  jget_u32(cfg, "intermediate_size", &ff_cfg);
  if (!jget_u32(cfg, "head_dim", &head_dim_cfg) &&
      jget_u32(cfg, "hidden_size", &hidden_cfg) && n_head)
    head_dim_cfg = hidden_cfg / n_head;

  for (size_t i = 0; i < st.n; ++i) {
    const StTensor* t = &st.t[i];
    char hf[256];
    normalize_hf_name(t->name, hf, sizeof hf);

    /* ---- fused phi3 splits (before map_name) ---- */
    const char* rest = NULL;
    if (strncmp(hf, "model.layers.", 13) == 0) rest = hf + 13;
    if (rest) {
      char* dot = strchr(rest, '.');
      if (dot) {
        char layer[16];
        size_t ll = (size_t)(dot - rest);
        if (ll > 0 && ll < sizeof layer) {
          memcpy(layer, rest, ll);
          layer[ll] = 0;
          int layer_ok = 1;
          for (size_t k = 0; k < ll; ++k)
            if (!isdigit((unsigned char)layer[k])) layer_ok = 0;
          const char* suf = dot + 1;
          if (layer_ok && t->ndim == 2) {
            uint64_t cols = t->shape[t->ndim - 1]; /* HF [out, in] -> in */
            uint64_t rows_tot = t->shape[0];

            if (!strcmp(suf, "self_attn.qkv_proj.weight")) {
              if (n_head == 0 || n_kv == 0 || head_dim_cfg == 0) {
                fprintf(stderr,
                        "convert: %s needs num_attention_heads / "
                        "num_key_value_heads / head_dim (or hidden_size) to split qkv\n",
                        t->name);
                goto done;
              }
              uint64_t q_rows = (uint64_t)n_head * head_dim_cfg;
              uint64_t kv_rows = (uint64_t)n_kv * head_dim_cfg;
              if (rows_tot != q_rows + 2 * kv_rows) {
                fprintf(stderr,
                        "convert: %s rows %" PRIu64 " != q(%" PRIu64 ")+2*kv(%" PRIu64 ")\n",
                        t->name, rows_tot, q_rows, kv_rows);
                goto done;
              }
              const char* names[3] = {"attn_q.weight", "attn_k.weight", "attn_v.weight"};
              uint64_t bases[3] = {0, q_rows, q_rows + kv_rows};
              uint64_t rs[3] = {q_rows, kv_rows, kv_rows};
              for (int p = 0; p < 3; ++p) {
                if (no >= ocap) {
                  fprintf(stderr, "convert: output plan overflow\n");
                  goto done;
                }
                OutT* o = &outs[no++];
                memset(o, 0, sizeof *o);
                o->src = t;
                snprintf(o->gg, sizeof o->gg, "blk.%s.%s", layer, names[p]);
                o->ndim = 2;
                o->cols = cols;
                o->rows = rs[p];
                o->src_row0 = bases[p];
                o->dims[0] = cols;
                o->dims[1] = rs[p];
                o->out_type = pick_type(outtype, 2, o->cols);
                size_t rb = oc_row_bytes(o->out_type, o->cols);
                if (rb == 0) {
                  fprintf(stderr, "convert: %s: cannot encode row of %" PRIu64 "\n", o->gg,
                          o->cols);
                  goto done;
                }
                o->out_bytes = o->rows * rb;
              }
              continue;
            }
            if (!strcmp(suf, "self_attn.qkv_proj.bias")) {
              fprintf(stderr, "convert: fused qkv bias not supported (%s)\n", t->name);
              goto done;
            }

            if (!strcmp(suf, "mlp.gate_up_proj.weight")) {
              uint64_t half = rows_tot / 2;
              if (rows_tot % 2 != 0 || (ff_cfg && half != ff_cfg)) {
                fprintf(stderr,
                        "convert: %s rows %" PRIu64 " not 2*intermediate_size",
                        t->name, rows_tot);
                if (ff_cfg) fprintf(stderr, " (%u)", ff_cfg);
                fprintf(stderr, "\n");
                goto done;
              }
              const char* names[2] = {"ffn_gate.weight", "ffn_up.weight"};
              for (int p = 0; p < 2; ++p) {
                if (no >= ocap) {
                  fprintf(stderr, "convert: output plan overflow\n");
                  goto done;
                }
                OutT* o = &outs[no++];
                memset(o, 0, sizeof *o);
                o->src = t;
                snprintf(o->gg, sizeof o->gg, "blk.%s.%s", layer, names[p]);
                o->ndim = 2;
                o->cols = cols;
                o->rows = half;
                o->src_row0 = (uint64_t)p * half;
                o->dims[0] = cols;
                o->dims[1] = half;
                o->out_type = pick_type(outtype, 2, o->cols);
                size_t rb = oc_row_bytes(o->out_type, o->cols);
                if (rb == 0) {
                  fprintf(stderr, "convert: %s: cannot encode row of %" PRIu64 "\n", o->gg,
                          o->cols);
                  goto done;
                }
                o->out_bytes = o->rows * rb;
              }
              continue;
            }
          }
        }
      }
    }

    char gg[160];
    int m = map_name(t->name, gg, sizeof gg, sandwich);
    if (m < 0) {
      fprintf(stderr, "convert: tensor '%s' is part of an unsupported structure "
                      "(MoE/MTP/vision); this converter handles dense text only\n", t->name);
      goto done;
    }
    if (m == 0) continue; /* dropped */
    if (no >= ocap) {
      fprintf(stderr, "convert: output plan overflow\n");
      goto done;
    }
    OutT* o = &outs[no];
    o->src = t;
    memcpy(o->gg, gg, sizeof o->gg);
    o->ndim = t->ndim;
    /* ggml dims = reverse(HF shape); cols = dims[0], rows = product(dims[1..]) */
    for (uint32_t d = 0; d < t->ndim; ++d) o->dims[d] = t->shape[t->ndim - 1 - d];
    o->cols = o->dims[0];
    o->rows = 1;
    for (uint32_t d = 1; d < t->ndim; ++d) o->rows *= o->dims[d];
    o->src_row0 = 0;
    o->out_type = pick_type(outtype, t->ndim, o->cols);
    o->norm_plus1 = sandwich && is_norm_weight(gg);
    o->permute = 0;
    o->head_dim = 0;
    size_t rb = oc_row_bytes(o->out_type, o->cols);
    if (rb == 0) { fprintf(stderr, "convert: %s: cannot encode row of %" PRIu64 "\n", gg, o->cols); goto done; }
    o->out_bytes = o->rows * rb;

    int is_k = 0;
    if (permute && is_qk_weight(gg, &is_k)) {
      uint64_t heads = is_k ? n_kv : n_head;
      if (heads == 0 || o->rows % heads != 0) {
        fprintf(stderr, "convert: %s rows %" PRIu64 " not divisible by heads %" PRIu64 "\n", gg, o->rows, heads);
        goto done;
      }
      o->head_dim = o->rows / heads;
      if (o->head_dim % 2 != 0) {
        fprintf(stderr, "convert: %s head_dim %" PRIu64 " is odd; cannot permute for NORMAL rope\n", gg, o->head_dim);
        goto done;
      }
      o->permute = 1;
    }
    no++;
  }
  if (no == 0) { fprintf(stderr, "convert: no recognizable tensors produced\n"); goto done; }

  /* Gemma sandwich: every layer that has attn_norm must also have ffn_norm
   * (pre_feedforward). Plain gemma-v1 checkpoints fail here instead of loading
   * with a silently-wrong ffn_norm. */
  if (sandwich) {
    int have_attn_norm = 0, have_ffn_norm = 0;
    for (size_t i = 0; i < no; ++i) {
      if (strstr(outs[i].gg, ".attn_norm.weight")) have_attn_norm = 1;
      if (strstr(outs[i].gg, ".ffn_norm.weight")) have_ffn_norm = 1;
    }
    if (have_attn_norm && !have_ffn_norm) {
      fprintf(stderr,
              "convert: gemma sandwich norms missing (need pre_feedforward_layernorm).\n"
              "  gemma-v1 (2-norm) is not supported; use gemma2/3/4 or llama.cpp.\n");
      goto done;
    }
  }

  /* sort by name for a deterministic file (loader is order-independent) */
  for (size_t i = 0; i + 1 < no; ++i)
    for (size_t k = i + 1; k < no; ++k)
      if (strcmp(outs[k].gg, outs[i].gg) < 0) { OutT tmp = outs[i]; outs[i] = outs[k]; outs[k] = tmp; }

  gts = calloc(no, sizeof(GwTensor));
  if (!gts) { fprintf(stderr, "convert: oom\n"); goto done; }
  for (size_t i = 0; i < no; ++i) {
    gts[i].name = outs[i].gg;
    gts[i].n_dims = outs[i].ndim;
    for (uint32_t d = 0; d < outs[i].ndim; ++d) gts[i].dims[d] = outs[i].dims[d];
    gts[i].type = outs[i].out_type;
    gts[i].size = outs[i].out_bytes;
  }

  if (gw_open(&w, output, kb.kv, kb.n, 32, gts, no, (int)outtype) != 0) {
    fprintf(stderr, "convert: cannot write %s\n", output);
    goto done;
  }
  for (size_t i = 0; i < no; ++i) {
    uint8_t* dst = malloc(outs[i].out_bytes ? outs[i].out_bytes : 1);
    if (!dst) { fprintf(stderr, "convert: oom\n"); goto done; }
    if (emit_payload(&outs[i], dst, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); free(dst); goto done; }
    int e = gw_tensor(&w, dst, outs[i].out_bytes);
    free(dst);
    if (e != 0) { fprintf(stderr, "convert: write error on %s\n", outs[i].gg); goto done; }
    if (verbose && i % 25 == 0)
      fprintf(stderr, "  [%zu/%zu] %s\n", i + 1, no, outs[i].gg);
  }
  if (gw_close(&w) != 0) { fprintf(stderr, "convert: close error\n"); goto done; }
  w.f = NULL;
  if (verbose)
    fprintf(stderr, "convert: wrote %s (%zu tensors, arch=%s%s, %s)\n", output, no, arch,
            permute ? ", q/k permuted" : "", opts->outtype ? opts->outtype : "F16");
  rc = 0;

done:
  if (w.f) gw_close(&w);
  free(gts);
  free(outs);
  kv_free(&kb);
  jfree(cfg_root);
  st_free(&st);
  return rc;
}

#ifndef OC_TOOLS_LIB
static void usage(void) {
  fprintf(stderr,
          "usage: oxidize-c-convert --input <dir|file.safetensors> --output out.gguf\n"
          "                         [--arch llama|mistral|yi|qwen2|qwen3|phi3|gemma2|gemma3|gemma4]\n"
          "                         [--outtype f16|f32|q8_0|q4_0|q2_k|q3_k|q4_k|q5_k|q6_k|iq4_xs]\n"
          "Converts a dense HuggingFace SafeTensors model to GGUF v3.\n"
          "Supported: llama/mistral/yi (q/k permute), qwen2/qwen3, phi3 (fused qkv split),\n"
          "           gemma2/3/4 (sandwich).\n"
          "MoE, gemma-v1, deepseek, vision/audio and non-BPE/Unigram tokenizers are rejected.\n");
}

int main(int argc, char** argv) {
  const char *in = NULL, *out = NULL, *arch = NULL, *outtype = NULL;
  static char ot[16];
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--input") && i + 1 < argc) in = argv[++i];
    else if (!strcmp(argv[i], "--output") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--arch") && i + 1 < argc) arch = argv[++i];
    else if (!strcmp(argv[i], "--outtype") && i + 1 < argc) {
      const char* v = argv[++i];
      size_t k = 0;
      for (; v[k] && k < sizeof ot - 1; ++k) ot[k] = (char)toupper((unsigned char)v[k]);
      ot[k] = 0;
      outtype = ot;
    } else { usage(); return 1; }
  }
  if (!in || !out) { usage(); return 1; }
  ConvertOpts o = {.arch_override = arch, .outtype = outtype};
  int rc = tool_convert(in, out, &o, 1);
  oc_pool_free();
  return rc;
}
#endif
