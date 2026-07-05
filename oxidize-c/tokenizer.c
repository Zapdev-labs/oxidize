/* GGUF tokenizer: SentencePiece (llama) + byte-level BPE (gpt2/qwen).
 * Port of oxidize-cpp/src/tokenizer.cpp. */
#include "oc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- tiny string->int hash map (FNV-1a, open addressing) ---- */
typedef struct {
  char **keys;
  int32_t *vals;
  size_t cap, len;
} strmap;

static uint64_t fnv(const char *s, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ (uint8_t)s[i]) * 1099511628211ull;
  return h;
}

static void map_init(strmap *m, size_t want) {
  size_t cap = 16;
  while (cap < want * 2) cap <<= 1;
  m->cap = cap;
  m->len = 0;
  m->keys = calloc(cap, sizeof(char *));
  m->vals = malloc(cap * sizeof(int32_t));
}

static void map_put(strmap *m, const char *key, int32_t val) {
  size_t i = fnv(key, strlen(key)) & (m->cap - 1);
  while (m->keys[i]) {
    if (strcmp(m->keys[i], key) == 0) return; /* first wins (mirror emplace) */
    i = (i + 1) & (m->cap - 1);
  }
  m->keys[i] = strdup(key);
  m->vals[i] = val;
  m->len++;
}

static int32_t map_get(const strmap *m, const char *key, size_t klen) {
  size_t i = fnv(key, klen) & (m->cap - 1);
  while (m->keys[i]) {
    if (strncmp(m->keys[i], key, klen) == 0 && m->keys[i][klen] == 0)
      return m->vals[i];
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}

static void map_free(strmap *m) {
  for (size_t i = 0; i < m->cap; ++i) free(m->keys[i]);
  free(m->keys);
  free(m->vals);
}

struct oc_tokenizer {
  bool bpe;
  char **pieces;
  float *scores;
  size_t n;
  strmap piece_to_id;
  strmap merge_ranks; /* "a b" -> rank (BPE) */
  int64_t bos, eos, unk, eot;
  bool add_space_prefix, wants_bos;
  /* GPT-2 byte<->unicode maps */
  char b2u[256][4];
  uint8_t b2u_len[256];
};

static size_t utf8_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x6) return 2;
  if ((c >> 4) == 0xe) return 3;
  if ((c >> 3) == 0x1e) return 4;
  return 1;
}

static size_t cp_to_utf8(uint32_t cp, char *out) {
  if (cp < 0x80) { out[0] = (char)cp; return 1; }
  if (cp < 0x800) {
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  }
  out[0] = (char)(0xE0 | (cp >> 12));
  out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
  out[2] = (char)(0x80 | (cp & 0x3F));
  return 3;
}

static void init_b2u(oc_tokenizer *t) {
  bool printable[256] = {0};
  for (int b = '!'; b <= '~'; ++b) printable[b] = true;
  for (int b = 0xA1; b <= 0xAC; ++b) printable[b] = true;
  for (int b = 0xAE; b <= 0xFF; ++b) printable[b] = true;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    uint32_t cp = printable[b] ? (uint32_t)b : (uint32_t)(256 + n++);
    t->b2u_len[b] = (uint8_t)cp_to_utf8(cp, t->b2u[b]);
  }
}

oc_tokenizer *oc_tokenizer_load(const oc_gguf *g) {
  const char *model = oc_meta_str(g, "tokenizer.ggml.model");
  if (!model) model = "llama";
  bool bpe;
  if (strcmp(model, "gpt2") == 0 || strcmp(model, "bpe") == 0) bpe = true;
  else if (strcmp(model, "llama") == 0) bpe = false;
  else {
    fprintf(stderr, "warning: unsupported tokenizer '%s'\n", model);
    return NULL;
  }
  const oc_meta *toks = oc_meta_get(g, "tokenizer.ggml.tokens");
  if (!toks || toks->kind != 2 || !toks->is_str) return NULL;

  oc_tokenizer *t = calloc(1, sizeof(*t));
  t->bpe = bpe;
  t->n = toks->count;
  t->pieces = toks->strs; /* borrowed from gguf (outlives tokenizer) */
  t->scores = calloc(t->n, sizeof(float));
  const oc_meta *sc = oc_meta_get(g, "tokenizer.ggml.scores");
  if (sc && sc->kind == 2 && !sc->is_str)
    for (size_t i = 0; i < sc->count && i < t->n; ++i)
      t->scores[i] = (float)sc->nums[i];
  map_init(&t->piece_to_id, t->n);
  for (size_t i = 0; i < t->n; ++i) map_put(&t->piece_to_id, t->pieces[i], (int32_t)i);

  map_init(&t->merge_ranks, 16);
  if (bpe) {
    const oc_meta *mg = oc_meta_get(g, "tokenizer.ggml.merges");
    if (mg && mg->kind == 2 && mg->is_str) {
      map_free(&t->merge_ranks);
      map_init(&t->merge_ranks, mg->count);
      for (size_t r = 0; r < mg->count; ++r)
        map_put(&t->merge_ranks, mg->strs[r], (int32_t)r);
    }
  }

  uint32_t u;
  t->bos = oc_meta_u32(g, "tokenizer.ggml.bos_token_id", &u) ? (int64_t)u : -1;
  t->eos = oc_meta_u32(g, "tokenizer.ggml.eos_token_id", &u) ? (int64_t)u : -1;
  t->unk = oc_meta_u32(g, "tokenizer.ggml.unknown_token_id", &u) ? (int64_t)u : -1;
  t->eot = oc_meta_u32(g, "tokenizer.ggml.eot_token_id", &u) ? (int64_t)u : -1;
  t->add_space_prefix = !bpe;
  t->wants_bos = !bpe;
  if (oc_meta_u32(g, "tokenizer.ggml.add_bos_token", &u)) t->wants_bos = u != 0;
  init_b2u(t);
  return t;
}

void oc_tokenizer_free(oc_tokenizer *t) {
  if (!t) return;
  free(t->scores);
  map_free(&t->piece_to_id);
  map_free(&t->merge_ranks);
  free(t);
}

bool oc_is_eog(const oc_tokenizer *t, uint32_t id) {
  return (int64_t)id == t->eos || (t->eot >= 0 && (int64_t)id == t->eot);
}

/* ---- token id output buffer ---- */
typedef struct { uint32_t *v; size_t n, cap; } idvec;
static void idpush(idvec *o, uint32_t id) {
  if (o->n == o->cap) {
    o->cap = o->cap ? o->cap * 2 : 64;
    o->v = realloc(o->v, o->cap * sizeof(uint32_t));
  }
  o->v[o->n++] = id;
}

/* ---- SPM (SentencePiece) ---- */
#define SPACE "\xe2\x96\x81" /* U+2581 */

typedef struct { int left, right; float score; size_t size; } bigram;
typedef struct { bigram *v; size_t n, cap; } bheap; /* max-heap on (score, -left) */

static bool bg_less(const bigram *a, const bigram *b) {
  return a->score < b->score || (a->score == b->score && a->left > b->left);
}
static void heap_push(bheap *h, bigram b) {
  if (h->n == h->cap) {
    h->cap = h->cap ? h->cap * 2 : 64;
    h->v = realloc(h->v, h->cap * sizeof(bigram));
  }
  size_t i = h->n++;
  h->v[i] = b;
  while (i > 0) {
    size_t p = (i - 1) / 2;
    if (!bg_less(&h->v[p], &h->v[i])) break;
    bigram tmp = h->v[p]; h->v[p] = h->v[i]; h->v[i] = tmp;
    i = p;
  }
}
static bigram heap_pop(bheap *h) {
  bigram top = h->v[0];
  h->v[0] = h->v[--h->n];
  size_t i = 0;
  for (;;) {
    size_t l = 2 * i + 1, r = l + 1, m = i;
    if (l < h->n && bg_less(&h->v[m], &h->v[l])) m = l;
    if (r < h->n && bg_less(&h->v[m], &h->v[r])) m = r;
    if (m == i) break;
    bigram tmp = h->v[m]; h->v[m] = h->v[i]; h->v[i] = tmp;
    i = m;
  }
  return top;
}

typedef struct { size_t off, len; int prev, next; } sym_t;

static void spm_try_add(const oc_tokenizer *t, const char *norm, sym_t *syms,
                        int left, int right, bheap *q) {
  if (left < 0 || right < 0) return;
  char buf[256];
  size_t ln = syms[left].len + syms[right].len;
  if (ln >= sizeof buf) return;
  memcpy(buf, norm + syms[left].off, syms[left].len);
  memcpy(buf + syms[left].len, norm + syms[right].off, syms[right].len);
  int32_t pid = map_get(&t->piece_to_id, buf, ln);
  if (pid < 0) return;
  heap_push(q, (bigram){left, right, t->scores[pid], ln});
}

static void encode_spm(const oc_tokenizer *t, const char *text, idvec *out) {
  size_t tlen = strlen(text);
  char *norm = malloc(tlen * 3 + 4);
  size_t nn = 0;
  if (t->add_space_prefix) { memcpy(norm + nn, SPACE, 3); nn += 3; }
  for (size_t i = 0; i < tlen; ++i) {
    if (text[i] == ' ') { memcpy(norm + nn, SPACE, 3); nn += 3; }
    else norm[nn++] = text[i];
  }
  norm[nn] = 0;

  size_t max_syms = nn + 1;
  sym_t *syms = malloc(max_syms * sizeof(sym_t));
  size_t ns = 0;
  for (size_t i = 0; i < nn;) {
    size_t l = utf8_len((unsigned char)norm[i]);
    if (i + l > nn) l = 1;
    syms[ns] = (sym_t){i, l, (int)ns - 1, -1};
    ns++;
    i += l;
  }
  for (size_t i = 0; i + 1 < ns; ++i) syms[i].next = (int)(i + 1);

  bheap q = {0};
  for (size_t i = 0; i + 1 < ns; ++i)
    spm_try_add(t, norm, syms, (int)i, (int)(i + 1), &q);

  while (q.n > 0) {
    bigram b = heap_pop(&q);
    sym_t *l = &syms[b.left];
    if (l->len == 0 || b.right < 0 || syms[b.right].len == 0) continue;
    if (l->len + syms[b.right].len != b.size) continue; /* stale */
    l->len += syms[b.right].len;
    syms[b.right].len = 0;
    l->next = syms[b.right].next;
    if (syms[b.right].next >= 0) syms[syms[b.right].next].prev = b.left;
    spm_try_add(t, norm, syms, l->prev, b.left, &q);
    spm_try_add(t, norm, syms, b.left, l->next, &q);
  }

  for (int s = ns > 0 ? 0 : -1; s >= 0; s = syms[s].next) {
    if (syms[s].len == 0) continue;
    int32_t pid = map_get(&t->piece_to_id, norm + syms[s].off, syms[s].len);
    if (pid >= 0) {
      idpush(out, (uint32_t)pid);
    } else {
      for (size_t j = 0; j < syms[s].len; ++j) {
        char buf[8];
        snprintf(buf, sizeof buf, "<0x%02X>", (unsigned char)norm[syms[s].off + j]);
        int32_t bid = map_get(&t->piece_to_id, buf, 6);
        if (bid >= 0) idpush(out, (uint32_t)bid);
        else if (t->unk >= 0) idpush(out, (uint32_t)t->unk);
      }
    }
  }
  free(q.v);
  free(syms);
  free(norm);
}

/* ---- byte-level BPE (gpt2/qwen) ---- */

typedef enum { CL_SPACE, CL_LETTER, CL_DIGIT, CL_OTHER } cls_t;
static cls_t classify(uint32_t cp) {
  if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0b || cp == 0x0c)
    return CL_SPACE;
  if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp >= 0x80)
    return CL_LETTER;
  if (cp >= '0' && cp <= '9') return CL_DIGIT;
  return CL_OTHER;
}

static uint32_t decode_cp(const char *s, size_t n, size_t i, size_t *len) {
  unsigned char c = (unsigned char)s[i];
  *len = utf8_len(c);
  if (*len == 1 || i + *len > n) { *len = 1; return c; }
  uint32_t cp;
  if (*len == 2) cp = (uint32_t)(c & 0x1f) << 6 | (s[i + 1] & 0x3f);
  else if (*len == 3)
    cp = (uint32_t)(c & 0x0f) << 12 | (uint32_t)(s[i + 1] & 0x3f) << 6 | (s[i + 2] & 0x3f);
  else
    cp = (uint32_t)(c & 0x07) << 18 | (uint32_t)(s[i + 1] & 0x3f) << 12 |
         (uint32_t)(s[i + 2] & 0x3f) << 6 | (s[i + 3] & 0x3f);
  return cp;
}

/* BPE-encode one pretokenized word (bytes -> unicode symbols -> merges). */
static void bpe_word(const oc_tokenizer *t, const char *word, size_t wn, idvec *out) {
  if (wn == 0) return;
  /* symbols as (ptr,len) into a working buffer; symbols merge left-to-right */
  size_t cap = wn * 4 + 1;
  char *buf = malloc(cap);
  size_t *off = malloc((wn + 1) * sizeof(size_t)); /* symbol boundaries in buf */
  size_t nb = 0, nsym = 0;
  for (size_t i = 0; i < wn; ++i) {
    off[nsym++] = nb;
    unsigned char c = (unsigned char)word[i];
    memcpy(buf + nb, t->b2u[c], t->b2u_len[c]);
    nb += t->b2u_len[c];
  }
  off[nsym] = nb;

  char pair[512];
  while (nsym > 1) {
    int best_rank = INT32_MAX;
    size_t best = SIZE_MAX;
    for (size_t k = 0; k + 1 < nsym; ++k) {
      size_t l1 = off[k + 1] - off[k], l2 = off[k + 2] - off[k + 1];
      if (l1 + l2 + 2 > sizeof pair) continue;
      memcpy(pair, buf + off[k], l1);
      pair[l1] = ' ';
      memcpy(pair + l1 + 1, buf + off[k + 1], l2);
      int32_t rank = map_get(&t->merge_ranks, pair, l1 + 1 + l2);
      if (rank >= 0 && rank < best_rank) { best_rank = rank; best = k; }
    }
    if (best == SIZE_MAX) break;
    /* merge best & best+1: just drop the boundary */
    memmove(off + best + 1, off + best + 2, (nsym - best - 1) * sizeof(size_t));
    nsym--;
  }
  for (size_t k = 0; k < nsym; ++k) {
    int32_t pid = map_get(&t->piece_to_id, buf + off[k], off[k + 1] - off[k]);
    if (pid >= 0) idpush(out, (uint32_t)pid);
    else if (t->unk >= 0) idpush(out, (uint32_t)t->unk);
  }
  free(buf);
  free(off);
}

static void encode_bpe(const oc_tokenizer *t, const char *text, idvec *out) {
  size_t n = strlen(text), i = 0;
  static const char *ctr[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
  while (i < n) {
    if (text[i] == '\'') {
      bool matched = false;
      for (size_t c = 0; c < 7; ++c) {
        size_t cl = strlen(ctr[c]);
        if (i + cl <= n && strncmp(text + i, ctr[c], cl) == 0) {
          bpe_word(t, text + i, cl, out);
          i += cl;
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }
    size_t p = text[i] == ' ' ? i + 1 : i;
    if (p < n) {
      size_t cl;
      cls_t cls = classify(decode_cp(text, n, p, &cl));
      if (cls != CL_SPACE) {
        size_t start = i;
        i = p;
        while (i < n) {
          size_t l2;
          if (classify(decode_cp(text, n, i, &l2)) != cls) break;
          i += l2;
        }
        bpe_word(t, text + start, i - start, out);
        continue;
      }
    }
    size_t k = i;
    while (k < n) {
      size_t l3;
      if (classify(decode_cp(text, n, k, &l3)) != CL_SPACE) break;
      k += l3;
    }
    bpe_word(t, text + i, k - i, out);
    i = k;
  }
}

/* encode one plain-text span (no special tokens) */
static void encode_span(const oc_tokenizer *t, const char *text, idvec *out) {
  if (t->bpe) encode_bpe(t, text, out);
  else encode_spm(t, text, out);
}

uint32_t *oc_tokenize(const oc_tokenizer *t, const char *text, bool add_bos,
                      size_t *n_out) {
  idvec out = {0};
  if (add_bos && t->wants_bos && t->bos >= 0) idpush(&out, (uint32_t)t->bos);
  /* Split on "<|...|>" special tokens that exist verbatim in the vocab
   * (ChatML markers etc.); BPE/SPM the plain spans between them. */
  const char *p = text;
  char span[4096];
  size_t sn = 0;
  while (*p) {
    if (p[0] == '<' && p[1] == '|') {
      const char *end = strstr(p + 2, "|>");
      if (end && (size_t)(end + 2 - p) <= 64) {
        int32_t id = map_get(&t->piece_to_id, p, (size_t)(end + 2 - p));
        if (id >= 0) {
          if (sn) { span[sn] = 0; encode_span(t, span, &out); sn = 0; }
          idpush(&out, (uint32_t)id);
          p = end + 2;
          continue;
        }
      }
    }
    if (sn + 1 >= sizeof span) { span[sn] = 0; encode_span(t, span, &out); sn = 0; }
    span[sn++] = *p++;
  }
  if (sn) { span[sn] = 0; encode_span(t, span, &out); }
  *n_out = out.n;
  return out.v;
}

size_t oc_detokenize(const oc_tokenizer *t, uint32_t id, char *buf, size_t cap) {
  if (id >= t->n) return 0;
  const char *p = t->pieces[id];
  size_t pn = strlen(p), o = 0;
  if (!t->bpe) {
    /* SPM: "<0xXX>" byte tokens, "▁" -> space */
    if (pn == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
      unsigned v;
      if (sscanf(p + 3, "%02X", &v) == 1 && o < cap) buf[o++] = (char)v;
      return o;
    }
    for (size_t i = 0; i < pn && o < cap;) {
      if (pn - i >= 3 && memcmp(p + i, SPACE, 3) == 0) { buf[o++] = ' '; i += 3; }
      else buf[o++] = p[i++];
    }
    return o;
  }
  /* BPE: reverse the byte<->unicode map */
  for (size_t i = 0; i < pn && o < cap;) {
    size_t l = utf8_len((unsigned char)p[i]);
    if (i + l > pn) l = 1;
    int byte = -1;
    for (int b = 0; b < 256; ++b) {
      if (t->b2u_len[b] == l && memcmp(t->b2u[b], p + i, l) == 0) { byte = b; break; }
    }
    if (byte >= 0) buf[o++] = (char)byte;
    else {
      for (size_t j = 0; j < l && o < cap; ++j) buf[o++] = p[i + j];
    }
    i += l;
  }
  return o;
}
