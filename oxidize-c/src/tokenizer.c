#include "tokenizer.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPM_SPACE "\xe2\x96\x81" /* U+2581 */

static uint64_t fnv1a(const char* s, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= (uint8_t)s[i];
    h *= 1099511628211ull;
  }
  return h;
}

static size_t utf8_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x6) return 2;
  if ((c >> 4) == 0xe) return 3;
  if ((c >> 3) == 0x1e) return 4;
  return 1;
}

int32_t tokenizer_piece_id(const Tokenizer* t, const char* piece, size_t len) {
  uint64_t h = fnv1a(piece, len);
  size_t mask = t->ht_size - 1;
  for (size_t i = h & mask;; i = (i + 1) & mask) {
    int32_t id = t->ht[i];
    if (id < 0) return -1;
    if (t->pieces[id].len == len && memcmp(t->pieces[id].ptr, piece, len) == 0)
      return id;
  }
}

static void ht_insert(Tokenizer* t, int32_t id) {
  uint64_t h = fnv1a(t->pieces[id].ptr, t->pieces[id].len);
  size_t mask = t->ht_size - 1;
  size_t i = h & mask;
  while (t->ht[i] >= 0) {
    /* keep first occurrence for duplicate pieces */
    int32_t e = t->ht[i];
    if (t->pieces[e].len == t->pieces[id].len &&
        memcmp(t->pieces[e].ptr, t->pieces[id].ptr, t->pieces[id].len) == 0)
      return;
    i = (i + 1) & mask;
  }
  t->ht[i] = id;
}

static uint64_t pair_hash(int32_t l, int32_t r) {
  return (uint64_t)(uint32_t)l * 2654435761u + (uint32_t)r * 40503u + 0x9e3779b9u;
}

static const BpeMerge* merge_find(const Tokenizer* t, int32_t l, int32_t r) {
  if (!t->merge_ht_size) return NULL;
  size_t mask = t->merge_ht_size - 1;
  for (size_t i = pair_hash(l, r) & mask;; i = (i + 1) & mask) {
    const BpeMerge* m = &t->merge_ht[i];
    if (m->left < 0) return NULL;
    if (m->left == l && m->right == r) return m;
  }
}

static void merge_insert(Tokenizer* t, BpeMerge e) {
  size_t mask = t->merge_ht_size - 1;
  size_t i = pair_hash(e.left, e.right) & mask;
  while (t->merge_ht[i].left >= 0) {
    /* first (lowest-rank) rule for a pair wins; ignore later duplicates */
    if (t->merge_ht[i].left == e.left && t->merge_ht[i].right == e.right) return;
    i = (i + 1) & mask;
  }
  t->merge_ht[i] = e;
}

/* Load tokenizer.ggml.merges: each "a b" rule joins pieces a,b; store keyed by
 * (id(a), id(b)) -> rank, id(ab). Rules whose left/right/merged aren't in the
 * vocab are skipped (matches the Rust loader). Leaves merge_ht_size 0 if absent
 * so bpe_encode keeps the greedy fallback. */
static void bpe_load_merges(Tokenizer* t, const GgufFile* g) {
  const GgufValue* mg = gguf_get_arr(g, "tokenizer.ggml.merges");
  if (!mg || mg->v.arr.elem_kind != GGUF_T_STRING || mg->v.arr.n == 0) return;
  size_t nm = mg->v.arr.n;
  size_t sz = 1;
  while (sz < nm * 2) sz <<= 1;
  t->merge_ht = malloc(sz * sizeof(BpeMerge));
  if (!t->merge_ht) return;
  for (size_t i = 0; i < sz; ++i) t->merge_ht[i].left = -1;
  t->merge_ht_size = sz;

  char* buf = malloc(2 * t->max_piece_len + 2); /* left ++ right (space removed) */
  if (!buf) { free(t->merge_ht); t->merge_ht = NULL; t->merge_ht_size = 0; return; }
  for (size_t rank = 0; rank < nm; ++rank) {
    const char* s = mg->v.arr.items[rank].v.str.ptr;
    size_t slen = mg->v.arr.items[rank].v.str.len;
    size_t sp = 0;
    while (sp < slen && s[sp] != ' ') ++sp;
    if (sp == 0 || sp >= slen) continue; /* malformed: no single space split */
    size_t rl = slen - sp - 1;
    int32_t left = tokenizer_piece_id(t, s, sp);
    int32_t right = tokenizer_piece_id(t, s + sp + 1, rl);
    if (left < 0 || right < 0) continue;
    if (sp + rl > 2 * t->max_piece_len) continue; /* can't be a vocab piece */
    memcpy(buf, s, sp);
    memcpy(buf + sp, s + sp + 1, rl); /* concatenation = merge string sans space */
    int32_t merged = tokenizer_piece_id(t, buf, sp + rl);
    if (merged < 0) continue;
    merge_insert(t, (BpeMerge){left, right, (int32_t)rank, merged});
  }
  free(buf);
}

/* Collect CONTROL(3)/USER_DEFINED(4) non-empty token ids, sorted longest piece
 * first, so chat markers pre-split ahead of surrounding text. */
static void bpe_collect_special(Tokenizer* t) {
  t->special_ids = malloc(t->n_vocab * sizeof(int32_t));
  if (!t->special_ids) return;
  size_t k = 0;
  for (size_t i = 0; i < t->n_vocab; ++i) {
    int ty = t->token_types[i];
    if ((ty == 3 || ty == 4) && t->pieces[i].len > 0) {
      /* insertion sort by piece length descending (n_special is small) */
      size_t j = k++;
      while (j > 0 && t->pieces[t->special_ids[j - 1]].len < t->pieces[i].len) {
        t->special_ids[j] = t->special_ids[j - 1];
        --j;
      }
      t->special_ids[j] = (int32_t)i;
    }
  }
  t->n_special = k;
}

int tokenizer_init(Tokenizer* t, const GgufFile* g) {
  memset(t, 0, sizeof(*t));
  const GgufValue* toks = gguf_get_arr(g, "tokenizer.ggml.tokens");
  if (!toks || toks->v.arr.elem_kind != GGUF_T_STRING) {
    fprintf(stderr, "tokenizer: missing tokenizer.ggml.tokens\n");
    return -1;
  }
  size_t n = toks->v.arr.n;
  t->n_vocab = n;
  t->pieces = calloc(n, sizeof(TokPiece));
  t->scores = calloc(n, sizeof(float));
  t->token_types = calloc(n, sizeof(int32_t));
  if (!t->pieces || !t->scores || !t->token_types) return -1;
  for (size_t i = 0; i < n; ++i) {
    t->pieces[i].ptr = toks->v.arr.items[i].v.str.ptr;
    t->pieces[i].len = toks->v.arr.items[i].v.str.len;
    if (t->pieces[i].len > t->max_piece_len) t->max_piece_len = t->pieces[i].len;
    t->token_types[i] = 1;
  }
  const GgufValue* sc = gguf_get_arr(g, "tokenizer.ggml.scores");
  if (sc)
    for (size_t i = 0; i < n && i < sc->v.arr.n; ++i)
      t->scores[i] = (float)sc->v.arr.items[i].v.f;
  const GgufValue* tt = gguf_get_arr(g, "tokenizer.ggml.token_type");
  if (tt)
    for (size_t i = 0; i < n && i < tt->v.arr.n; ++i) {
      const GgufValue* e = &tt->v.arr.items[i];
      t->token_types[i] = (e->kind == GGUF_T_I8 || e->kind == GGUF_T_I16 ||
                           e->kind == GGUF_T_I32 || e->kind == GGUF_T_I64)
                              ? (int32_t)e->v.i
                              : (int32_t)e->v.u;
    }

  size_t sz = 1;
  while (sz < n * 2) sz <<= 1;
  t->ht_size = sz;
  t->ht = malloc(sz * sizeof(int32_t));
  if (!t->ht) return -1;
  for (size_t i = 0; i < sz; ++i) t->ht[i] = -1;
  for (size_t i = 0; i < n; ++i) ht_insert(t, (int32_t)i);

  uint32_t u;
  t->bos_id = gguf_get_u32(g, "tokenizer.ggml.bos_token_id", &u) ? (int64_t)u : -1;
  t->eos_id = gguf_get_u32(g, "tokenizer.ggml.eos_token_id", &u) ? (int64_t)u : -1;
  t->unk_id = gguf_get_u32(g, "tokenizer.ggml.unknown_token_id", &u) ? (int64_t)u : -1;
  t->eot_id = gguf_get_u32(g, "tokenizer.ggml.eot_token_id", &u) ? (int64_t)u : -1;
  t->add_bos = true;
  if (gguf_get_u32(g, "tokenizer.ggml.add_bos_token", &u)) t->add_bos = u != 0;
  t->add_space_prefix = true;
  if (gguf_get_u32(g, "tokenizer.ggml.add_space_prefix", &u))
    t->add_space_prefix = u != 0;

  char* model = gguf_get_str(g, "tokenizer.ggml.model");
  if (model && strcmp(model, "gpt2") == 0) {
    t->is_bpe = true;
    t->add_space_prefix = false;
    /* GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to
     * 256+n in order. */
    int n_extra = 0;
    for (int b = 0; b < 256; ++b) {
      int printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                      (b >= 0xAE && b <= 0xFF);
      t->byte_to_cp[b] = printable ? (uint16_t)b : (uint16_t)(256 + n_extra++);
    }
    memset(t->cp_to_byte, 0, sizeof(t->cp_to_byte));
    for (int b = 0; b < 256; ++b) t->cp_to_byte[t->byte_to_cp[b]] = (uint8_t)b;
    bpe_load_merges(t, g);
    bpe_collect_special(t);
  }
  free(model);
  return 0;
}

/* Append UTF-8 encoding of cp (< 0x800 here) to dst; returns bytes written. */
static size_t put_cp(char* dst, unsigned cp) {
  if (cp < 0x80) {
    dst[0] = (char)cp;
    return 1;
  }
  dst[0] = (char)(0xC0 | (cp >> 6));
  dst[1] = (char)(0x80 | (cp & 0x3F));
  return 2;
}

/* Decode one UTF-8 codepoint at p (len avail); writes cp, returns bytes read. */
static size_t get_cp(const char* p, size_t avail, unsigned* cp) {
  unsigned char c = (unsigned char)p[0];
  if (c < 0x80 || avail < 2) {
    *cp = c;
    return 1;
  }
  if ((c >> 5) == 0x6) {
    *cp = ((unsigned)(c & 0x1F) << 6) | ((unsigned char)p[1] & 0x3F);
    return 2;
  }
  if ((c >> 4) == 0xE && avail >= 3) {
    *cp = ((unsigned)(c & 0x0F) << 12) | (((unsigned char)p[1] & 0x3F) << 6) |
          ((unsigned char)p[2] & 0x3F);
    return 3;
  }
  *cp = c;
  return 1;
}

/* Byte-level BPE, greedy longest-match against the vocab. Fallback used only
 * when the model ships no merge table: any valid segmentation decodes back to
 * the same text, and without merges canonical BPE isn't defined. */
static int32_t* bpe_encode_greedy(const Tokenizer* t, const char* text, size_t* n_out) {
  size_t tlen = strlen(text);
  char* norm = malloc(2 * tlen + 1);
  if (!norm) { *n_out = 0; return NULL; }
  size_t nn = 0;
  for (size_t i = 0; i < tlen; ++i)
    nn += put_cp(norm + nn, t->byte_to_cp[(unsigned char)text[i]]);

  int32_t* out = malloc((nn + 1) * sizeof(int32_t));
  size_t cnt = 0, i = 0;
  size_t maxlen = t->max_piece_len;
  while (i < nn) {
    size_t lim = nn - i < maxlen ? nn - i : maxlen;
    int32_t best = -1;
    size_t best_l = 0;
    for (size_t l = lim; l >= 1; --l) {
      int32_t id = tokenizer_piece_id(t, norm + i, l);
      if (id >= 0) { best = id; best_l = l; break; }
    }
    if (best < 0) { i += 1; continue; /* unmappable byte: skip */ }
    out[cnt++] = best;
    i += best_l;
  }
  free(norm);
  *n_out = cnt;
  return out;
}

/* Rank-based byte-level BPE of one plain segment (no special tokens). Maps each
 * byte to its GPT-2 code-point piece, then repeatedly joins the adjacent pair
 * with the lowest merge rank until none remain — canonical BPE, so ids match
 * llama.cpp. Appends to out[*cnt]; out must hold >= (segment length) more ids. */
static void bpe_encode_segment(const Tokenizer* t, const char* text, size_t tlen,
                               int32_t* out, size_t* cnt) {
  if (tlen == 0) return;
  int32_t* seq = malloc(tlen * sizeof(int32_t));
  if (!seq) return;
  size_t n = 0;
  for (size_t i = 0; i < tlen; ++i) {
    char cb[4];
    size_t w = put_cp(cb, t->byte_to_cp[(unsigned char)text[i]]);
    int32_t id = tokenizer_piece_id(t, cb, w);
    if (id < 0) id = (int32_t)t->unk_id;
    if (id >= 0) seq[n++] = id; /* else drop the byte (no piece, no unk) */
  }

  while (n >= 2) {
    int best_rank = INT_MAX;
    int32_t merged = -1, bl = -1, br = -1;
    for (size_t i = 0; i + 1 < n; ++i) {
      const BpeMerge* m = merge_find(t, seq[i], seq[i + 1]);
      if (m && m->rank < best_rank) {
        best_rank = m->rank;
        merged = m->merged;
        bl = seq[i];
        br = seq[i + 1];
      }
    }
    if (best_rank == INT_MAX) break;
    size_t w = 0; /* collapse every occurrence of (bl,br), left to right */
    for (size_t i = 0; i < n;) {
      if (i + 1 < n && seq[i] == bl && seq[i + 1] == br) {
        seq[w++] = merged;
        i += 2;
      } else {
        seq[w++] = seq[i++];
      }
    }
    n = w;
  }
  for (size_t i = 0; i < n; ++i) out[(*cnt)++] = seq[i];
  free(seq);
}

/* Byte-level BPE encode. With a merge table: split the input on special tokens
 * (chat markers) first so they emit their exact id, then apply rank-based BPE to
 * the text between. Without merges: greedy longest-match fallback. */
static int32_t* bpe_encode(const Tokenizer* t, const char* text, size_t* n_out) {
  if (t->merge_ht_size == 0) return bpe_encode_greedy(t, text, n_out);

  size_t tlen = strlen(text);
  /* Every byte yields at most one token and each special token consumes >=1
   * byte, so tlen ids (+1) is always enough — no growth needed. */
  int32_t* out = malloc((tlen + 1) * sizeof(int32_t));
  if (!out) { *n_out = 0; return NULL; }
  size_t cnt = 0, pos = 0;
  while (pos < tlen) {
    size_t best_pos = tlen, best_len = 0;
    int32_t best_id = -1;
    for (size_t s = 0; s < t->n_special; ++s) {
      int32_t id = t->special_ids[s];
      const char* sp = t->pieces[id].ptr;
      size_t sl = t->pieces[id].len;
      for (size_t i = pos; i + sl <= tlen; ++i) {
        if (memcmp(text + i, sp, sl) == 0) {
          if (i < best_pos) { best_pos = i; best_len = sl; best_id = id; }
          break; /* leftmost occurrence of this piece is enough */
        }
      }
    }
    if (best_id < 0) {
      bpe_encode_segment(t, text + pos, tlen - pos, out, &cnt);
      break;
    }
    if (best_pos > pos)
      bpe_encode_segment(t, text + pos, best_pos - pos, out, &cnt);
    out[cnt++] = best_id;
    pos = best_pos + best_len;
  }
  *n_out = cnt;
  return out;
}

void tokenizer_free(Tokenizer* t) {
  free(t->pieces);
  free(t->scores);
  free(t->token_types);
  free(t->ht);
  free(t->merge_ht);
  free(t->special_ids);
  memset(t, 0, sizeof(*t));
}

/* True if `text` begins with a CONTROL(3)/USER_DEFINED(4) token piece. The SPM
 * path (unlike bpe_encode) does not partition on specials, so it otherwise can't
 * tell a chat-formatted prompt — which opens with a control token (<|im_start|>,
 * <start_of_turn>, ...) — from plain text. Used only to suppress the leading
 * add_space_prefix ▁: that dummy space must never precede an opening control
 * token (no model is trained with a lone ▁ before its turn marker; llama.cpp
 * applies the prefix to raw text, never ahead of a leading special). */
static bool spm_starts_with_special(const Tokenizer* t, const char* text,
                                    size_t tlen) {
  for (size_t i = 0; i < t->n_vocab; ++i) {
    int ty = t->token_types[i];
    size_t pl = t->pieces[i].len;
    if ((ty == 3 || ty == 4) && pl > 0 && pl <= tlen &&
        memcmp(text, t->pieces[i].ptr, pl) == 0)
      return true;
  }
  return false;
}

/* Viterbi best-path over the SentencePiece unigram scores, with a heavily
 * penalized byte/char fallback edge so unmatched input still segments. */
int32_t* tokenizer_encode(const Tokenizer* t, const char* text, bool add_bos,
                          size_t* n_out) {
  if (t->is_bpe) {
    (void)add_bos; /* qwen35: add_bos_token = false */
    return bpe_encode(t, text, n_out);
  }
  size_t tlen = strlen(text);
  /* Normalize: leading "▁", ' ' -> "▁" (each space grows by 2 bytes). */
  char* norm = malloc(3 * tlen + 4);
  if (!norm) { *n_out = 0; return NULL; }
  size_t nn = 0;
  /* Dummy space prefix, but not before a leading control token (chat prompts).
   * ponytail: only the *leading* special is guarded here — interior specials do
   * not re-trigger a following ▁ (llama.cpp's is_prev_special). Upgrade path:
   * full special-token partition in this path if a model needs ▁role after an
   * interior marker; today no supported SPM chat template depends on that. */
  if (t->add_space_prefix && !spm_starts_with_special(t, text, tlen)) {
    memcpy(norm + nn, SPM_SPACE, 3);
    nn += 3;
  }
  for (size_t i = 0; i < tlen; ++i) {
    if (text[i] == ' ') {
      memcpy(norm + nn, SPM_SPACE, 3);
      nn += 3;
    } else {
      norm[nn++] = text[i];
    }
  }

  float* dp = malloc((nn + 1) * sizeof(float));
  int32_t* tok = malloc((nn + 1) * sizeof(int32_t)); /* token ending at i */
  size_t* back = malloc((nn + 1) * sizeof(size_t));
  if (!dp || !tok || !back) { free(norm); free(dp); free(tok); free(back); *n_out = 0; return NULL; }
  const float NEG = -1e30f;
  for (size_t i = 0; i <= nn; ++i) dp[i] = NEG;
  dp[0] = 0.0f;
  size_t maxlen = t->max_piece_len < 64 ? t->max_piece_len : 64;

  for (size_t i = 0; i < nn; ++i) {
    if (dp[i] <= NEG) continue;
    size_t lim = nn - i < maxlen ? nn - i : maxlen;
    for (size_t l = 1; l <= lim; ++l) {
      int32_t id = tokenizer_piece_id(t, norm + i, l);
      if (id < 0) continue;
      float s = dp[i] + t->scores[id];
      if (s > dp[i + l]) {
        dp[i + l] = s;
        tok[i + l] = id;
        back[i + l] = i;
      }
    }
    /* Fallback edge: one UTF-8 char, expanded to <0xXX> byte tokens (or unk)
     * at emission. Sentinel id -2. */
    size_t cl = utf8_len((unsigned char)norm[i]);
    if (i + cl > nn) cl = 1;
    float s = dp[i] - 1e6f;
    if (s > dp[i + cl]) {
      dp[i + cl] = s;
      tok[i + cl] = -2;
      back[i + cl] = i;
    }
  }

  /* Walk back, then reverse. */
  size_t cap = nn + 8;
  int32_t* out = malloc(cap * sizeof(int32_t));
  size_t cnt = 0;
  for (size_t i = nn; i > 0; i = back[i]) {
    int32_t id = tok[i];
    if (id == -2) {
      for (size_t b = i; b > back[i]; --b) { /* per raw byte, reversed */
        char bt[8];
        snprintf(bt, sizeof(bt), "<0x%02X>", (unsigned char)norm[b - 1]);
        int32_t bid = tokenizer_piece_id(t, bt, 6);
        out[cnt++] = bid >= 0 ? bid : (int32_t)t->unk_id;
      }
    } else {
      out[cnt++] = id;
    }
  }
  if (add_bos && t->add_bos && t->bos_id >= 0) out[cnt++] = (int32_t)t->bos_id;
  for (size_t i = 0; i < cnt / 2; ++i) {
    int32_t tmp = out[i];
    out[i] = out[cnt - 1 - i];
    out[cnt - 1 - i] = tmp;
  }
  free(norm);
  free(dp);
  free(tok);
  free(back);
  *n_out = cnt;
  return out;
}

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

size_t tokenizer_decode_token(const Tokenizer* t, int32_t id, char* buf, size_t cap) {
  if (id < 0 || (size_t)id >= t->n_vocab || cap == 0) return 0;
  const char* p = t->pieces[id].ptr;
  size_t len = t->pieces[id].len;
  if (t->is_bpe) {
    if (t->token_types[id] == 3) return 0; /* control token: don't print */
    size_t w = 0;
    for (size_t i = 0; i < len && w < cap;) {
      unsigned cp;
      i += get_cp(p + i, len - i, &cp);
      buf[w++] = cp < 512 ? (char)t->cp_to_byte[cp] : '?';
    }
    return w;
  }
  /* Byte token "<0xXX>". */
  if (len == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
    int hi = hexval(p[3]), lo = hexval(p[4]);
    if (hi >= 0 && lo >= 0) {
      buf[0] = (char)(hi * 16 + lo);
      return 1;
    }
  }
  size_t w = 0;
  for (size_t i = 0; i < len && w < cap;) {
    if (i + 3 <= len && memcmp(p + i, SPM_SPACE, 3) == 0) {
      buf[w++] = ' ';
      i += 3;
    } else {
      buf[w++] = p[i++];
    }
  }
  return w;
}

/* ---- chat templates -------------------------------------------------------
 * Fixed per-family control tokens (llama.cpp style). A turn is assembled from
 * these pieces: an optional system section, the user section, then the
 * assistant opener (the generation prompt). `sys_open == NULL` means the family
 * has no system role, so the system text is folded into the first user turn as
 * "{user_open}{system}\n\n{user}{user_close}" (Gemma/Mistral do this). For a
 * follow-up turn the previous assistant turn is closed with `asst_close` before
 * a fresh user turn — this string appends cleanly onto a live KV cache. */
typedef struct {
  const char* name;
  const char* sys_open;   /* NULL => fold system into the user turn */
  const char* sys_close;
  const char* user_open;
  const char* user_close;
  const char* asst_open;  /* generation opener (assistant-turn start) */
  const char* asst_close; /* closes the assistant turn (for continuation) */
  const char* stop;       /* bare turn-terminating token (extra stop id) */
} ChatTable;

/* Indexed by ChatFamily. ponytail: CHATML's opener carries Qwen's pre-closed
 * <think> block (non-thinking prefill) to preserve current qwen36 behavior; a
 * true generic-ChatML/enable_thinking split can key off the template later. */
static const ChatTable CHAT_TABLE[] = {
    [CHAT_CHATML] = {"chatml", "<|im_start|>system\n", "<|im_end|>\n",
                     "<|im_start|>user\n", "<|im_end|>\n",
                     "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                     "<|im_end|>\n", "<|im_end|>"},
    [CHAT_LLAMA3] = {"llama3",
                     "<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>",
                     "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>",
                     "<|start_header_id|>assistant<|end_header_id|>\n\n",
                     "<|eot_id|>", "<|eot_id|>"},
    [CHAT_MISTRAL] = {"mistral", NULL, NULL, "[INST] ", " [/INST]", "",
                      "</s>", "</s>"},
    [CHAT_GEMMA] = {"gemma", NULL, NULL, "<start_of_turn>user\n",
                    "<end_of_turn>\n", "<start_of_turn>model\n", "<end_of_turn>\n",
                    "<end_of_turn>"},
    [CHAT_PHI3] = {"phi3", "<|system|>\n", "<|end|>\n", "<|user|>\n", "<|end|>\n",
                   "<|assistant|>\n", "<|end|>\n", "<|end|>"},
    [CHAT_GEMMA4] = {"gemma4", NULL, NULL, "<|turn>user\n", "<turn|>\n",
                     "<|turn>model\n<|channel>thought\n<channel|>", "<turn|>\n",
                     "<turn|>"},
};

const char* chat_family_name(ChatFamily fam) { return CHAT_TABLE[fam].name; }
const char* chat_stop_token(ChatFamily fam) { return CHAT_TABLE[fam].stop; }

ChatFamily chat_detect(const Tokenizer* t, const char* tmpl) {
  /* Strong signal: substrings of the model's own chat_template. Order matters —
   * check the most specific markers first. */
  if (tmpl) {
    if (strstr(tmpl, "<|turn>")) return CHAT_GEMMA4;
    if (strstr(tmpl, "<start_of_turn>")) return CHAT_GEMMA;
    if (strstr(tmpl, "<|start_header_id|>")) return CHAT_LLAMA3;
    if (strstr(tmpl, "<|im_start|>")) return CHAT_CHATML;
    if (strstr(tmpl, "[INST]")) return CHAT_MISTRAL;
    if (strstr(tmpl, "<|assistant|>")) return CHAT_PHI3;
  }
  /* No/unknown template: probe the vocab for a family's marker token. */
  if (t) {
    static const struct {
      const char* tok;
      ChatFamily fam;
    } probe[] = {
        {"<|turn>", CHAT_GEMMA4},         {"<start_of_turn>", CHAT_GEMMA},
        {"<|start_header_id|>", CHAT_LLAMA3}, {"<|im_start|>", CHAT_CHATML},
        {"<|assistant|>", CHAT_PHI3},
    };
    for (size_t i = 0; i < sizeof probe / sizeof *probe; ++i)
      if (tokenizer_piece_id(t, probe[i].tok, strlen(probe[i].tok)) >= 0)
        return probe[i].fam;
  }
  return CHAT_CHATML; /* generic fallback */
}

size_t chat_format_turn(ChatFamily fam, const char* system, const char* user,
                        bool first_turn, char* buf, size_t cap) {
  const ChatTable* c = &CHAT_TABLE[fam];
  int n;
  if (!first_turn) {
    /* close the prior assistant turn, then a fresh user turn + opener */
    n = snprintf(buf, cap, "%s%s%s%s%s", c->asst_close, c->user_open, user,
                 c->user_close, c->asst_open);
  } else if (system && c->sys_open) {
    /* dedicated system turn */
    n = snprintf(buf, cap, "%s%s%s%s%s%s%s", c->sys_open, system, c->sys_close,
                 c->user_open, user, c->user_close, c->asst_open);
  } else if (system) {
    /* no system role: fold it into the user turn */
    n = snprintf(buf, cap, "%s%s\n\n%s%s%s", c->user_open, system, user,
                 c->user_close, c->asst_open);
  } else {
    n = snprintf(buf, cap, "%s%s%s%s", c->user_open, user, c->user_close,
                 c->asst_open);
  }
  if (n < 0 || (size_t)n >= cap) return 0;
  return (size_t)n;
}

/* ---- self-test ------------------------------------------------------------
 * Proves the merge policy is rank-based, not greedy-longest, on a vocab where
 * the two DIVERGE, and that special tokens pre-split. See tokenizer.h. */
static void st_put(uint8_t** b, size_t* l, const void* p, size_t n) {
  *b = realloc(*b, *l + n);
  if (!*b) abort();
  memcpy(*b + *l, p, n);
  *l += n;
}
static void st_u32(uint8_t** b, size_t* l, uint32_t v) { st_put(b, l, &v, 4); }
static void st_u64(uint8_t** b, size_t* l, uint64_t v) { st_put(b, l, &v, 8); }
static void st_str(uint8_t** b, size_t* l, const char* s) {
  st_u64(b, l, strlen(s));
  st_put(b, l, s, strlen(s));
}

int tokenizer_selftest(void) {
  /* Vocab where greedy-longest and rank BPE disagree on "abc":
   *   ids: 0 "a" 1 "b" 2 "c" 3 "ab" 4 "abc" 5 "<|end|>"(control)
   *   merges: rank0 "a b" -> "ab"  (the ONLY merge)
   * greedy-longest: "abc" -> [4]        (one token, wrong vs llama.cpp)
   * rank-based:     "abc" -> [3,2]=ab,c (a+b merges first; abc never forms) */
  const char* toks[] = {"a", "b", "c", "ab", "abc", "<|end|>"};
  const int32_t types[] = {1, 1, 1, 1, 1, 3};
  const size_t nv = 6;
  const char* merges[] = {"a b"};
  const size_t nm = 1;

  uint8_t* b = NULL;
  size_t l = 0;
  st_put(&b, &l, "GGUF", 4);
  st_u32(&b, &l, 3);           /* version */
  st_u64(&b, &l, 0);           /* tensor count */
  st_u64(&b, &l, 5);           /* kv count: tokens, token_type, model, merges, scores */

  st_str(&b, &l, "tokenizer.ggml.tokens");
  st_u32(&b, &l, GGUF_T_ARRAY);
  st_u32(&b, &l, GGUF_T_STRING);
  st_u64(&b, &l, nv);
  for (size_t i = 0; i < nv; ++i) st_str(&b, &l, toks[i]);

  st_str(&b, &l, "tokenizer.ggml.scores");
  st_u32(&b, &l, GGUF_T_ARRAY);
  st_u32(&b, &l, GGUF_T_F32);
  st_u64(&b, &l, nv);
  for (size_t i = 0; i < nv; ++i) { float s = 0.0f; st_put(&b, &l, &s, 4); }

  st_str(&b, &l, "tokenizer.ggml.token_type");
  st_u32(&b, &l, GGUF_T_ARRAY);
  st_u32(&b, &l, GGUF_T_I32);
  st_u64(&b, &l, nv);
  for (size_t i = 0; i < nv; ++i) st_u32(&b, &l, (uint32_t)types[i]);

  st_str(&b, &l, "tokenizer.ggml.merges");
  st_u32(&b, &l, GGUF_T_ARRAY);
  st_u32(&b, &l, GGUF_T_STRING);
  st_u64(&b, &l, nm);
  for (size_t i = 0; i < nm; ++i) st_str(&b, &l, merges[i]);

  st_str(&b, &l, "tokenizer.ggml.model");
  st_u32(&b, &l, GGUF_T_STRING);
  st_str(&b, &l, "gpt2");

  GgufFile g;
  char err[256];
  if (gguf_parse(&g, b, l, err, sizeof err) != 0) {
    fprintf(stderr, "selftest: gguf_parse: %s\n", err);
    abort();
  }
  Tokenizer t;
  if (tokenizer_init(&t, &g) != 0) abort();
  if (!t.is_bpe || t.merge_ht_size == 0 || t.n_special != 1) abort();

  size_t n = 0;
  int32_t* ids = tokenizer_encode(&t, "abc", false, &n);
  /* rank-based: [ab, c] = [3, 2]; greedy would be [abc] = [4] */
  if (!ids || n != 2 || ids[0] != 3 || ids[1] != 2) {
    fprintf(stderr, "selftest: rank BPE of \"abc\" gave");
    for (size_t i = 0; i < n; ++i) fprintf(stderr, " %d", ids ? ids[i] : -1);
    fprintf(stderr, ", want 3 2 ([ab,c]) not 4 ([abc])\n");
    abort();
  }
  /* round-trip */
  char out[64];
  size_t w = 0;
  for (size_t i = 0; i < n; ++i) w += tokenizer_decode_token(&t, ids[i], out + w, sizeof out - w);
  out[w] = 0;
  if (strcmp(out, "abc") != 0) { fprintf(stderr, "selftest: roundtrip \"%s\"\n", out); abort(); }
  free(ids);

  /* special-token pre-split: "ab<|end|>c" -> [ab, <|end|>, c] = [3, 5, 2] */
  ids = tokenizer_encode(&t, "ab<|end|>c", false, &n);
  if (!ids || n != 3 || ids[0] != 3 || ids[1] != 5 || ids[2] != 2) {
    fprintf(stderr, "selftest: special-split gave");
    for (size_t i = 0; i < n; ++i) fprintf(stderr, " %d", ids ? ids[i] : -1);
    fprintf(stderr, ", want 3 5 2 ([ab,<|end|>,c])\n");
    abort();
  }
  free(ids);

  tokenizer_free(&t);
  gguf_close(&g);
  free(b);
  printf("ok tokenizer rank-BPE merges + special pre-split "
         "(greedy \"abc\"->[abc], rank->[ab,c]; \"ab<|end|>c\"->[ab,<|end|>,c])\n");
  return chat_selftest();
}

/* ---- chat-template self-test ----------------------------------------------
 * Builds a byte-level BPE vocab carrying several families' control tokens, then
 * for each family checks that (a) chat_detect() picks it from a representative
 * chat_template string and (b) the formatted turn tokenizes to the EXPECTED
 * control-token ids at the expected positions — not merely matching text. */
static size_t st_count(const int32_t* ids, size_t n, int32_t id) {
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) k += ids[i] == id;
  return k;
}

int chat_selftest(void) {
  /* Byte pieces (GPT-2 byte->unicode) so any text round-trips, plus each
   * family's control tokens as CONTROL(3) so they pre-split to one id. */
  static char bytep[256][8];
  uint16_t b2c[256];
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    int printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                    (b >= 0xAE && b <= 0xFF);
    b2c[b] = printable ? (uint16_t)b : (uint16_t)(256 + extra++);
  }
  const char* ctl[] = {"<|im_start|>", "<|im_end|>", "<|start_header_id|>",
                       "<|end_header_id|>", "<|eot_id|>", "<start_of_turn>",
                       "<end_of_turn>"};
  const size_t nctl = sizeof ctl / sizeof *ctl;
  const size_t nv = 256 + nctl;

  uint8_t* b = NULL;
  size_t l = 0;
  st_put(&b, &l, "GGUF", 4);
  st_u32(&b, &l, 3);
  st_u64(&b, &l, 0);
  st_u64(&b, &l, 3); /* tokens, token_type, model */

  st_str(&b, &l, "tokenizer.ggml.tokens");
  st_u32(&b, &l, GGUF_T_ARRAY);
  st_u32(&b, &l, GGUF_T_STRING);
  st_u64(&b, &l, nv);
  for (int by = 0; by < 256; ++by) {
    unsigned cp = b2c[by];
    size_t w = 0;
    if (cp < 0x80) {
      bytep[by][w++] = (char)cp;
    } else {
      bytep[by][w++] = (char)(0xC0 | (cp >> 6));
      bytep[by][w++] = (char)(0x80 | (cp & 0x3F));
    }
    bytep[by][w] = 0;
    st_str(&b, &l, bytep[by]);
  }
  for (size_t i = 0; i < nctl; ++i) st_str(&b, &l, ctl[i]);

  st_str(&b, &l, "tokenizer.ggml.token_type");
  st_u32(&b, &l, GGUF_T_ARRAY);
  st_u32(&b, &l, GGUF_T_I32);
  st_u64(&b, &l, nv);
  for (int i = 0; i < 256; ++i) st_u32(&b, &l, 1);      /* NORMAL */
  for (size_t i = 0; i < nctl; ++i) st_u32(&b, &l, 3);  /* CONTROL */

  st_str(&b, &l, "tokenizer.ggml.model");
  st_u32(&b, &l, GGUF_T_STRING);
  st_str(&b, &l, "gpt2");

  GgufFile g;
  char err[256];
  if (gguf_parse(&g, b, l, err, sizeof err) != 0) {
    fprintf(stderr, "chat_selftest: gguf_parse: %s\n", err);
    abort();
  }
  Tokenizer t;
  if (tokenizer_init(&t, &g) != 0) abort();

  int32_t im_s = tokenizer_piece_id(&t, "<|im_start|>", 12);
  int32_t im_e = tokenizer_piece_id(&t, "<|im_end|>", 10);
  int32_t sh_s = tokenizer_piece_id(&t, "<|start_header_id|>", 19);
  int32_t sh_e = tokenizer_piece_id(&t, "<|end_header_id|>", 17);
  int32_t eot = tokenizer_piece_id(&t, "<|eot_id|>", 10);
  int32_t got = tokenizer_piece_id(&t, "<start_of_turn>", 15);
  int32_t eom = tokenizer_piece_id(&t, "<end_of_turn>", 13);
  if (im_s < 0 || im_e < 0 || sh_s < 0 || sh_e < 0 || eot < 0 || got < 0 || eom < 0)
    abort();

  char buf[512];
  size_t n = 0;
  int32_t* ids;

  /* --- ChatML: detect + first turn control-token layout -----------------
   * <|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n
   * <|im_start|>assistant\n... => im_start x3, im_end x2, opener leads. */
  if (chat_detect(&t, "{{'<|im_start|>' + role}}") != CHAT_CHATML) abort();
  if (chat_family_name(CHAT_CHATML)[0] != 'c') abort();
  if (chat_format_turn(CHAT_CHATML, "sys", "hi", true, buf, sizeof buf) == 0) abort();
  ids = tokenizer_encode(&t, buf, false, &n);
  if (!ids || ids[0] != im_s || st_count(ids, n, im_s) != 3 ||
      st_count(ids, n, im_e) != 2) {
    fprintf(stderr, "chat_selftest: chatml first-turn control tokens wrong\n");
    abort();
  }
  free(ids);

  /* ChatML follow-up turn closes the prior assistant turn first. */
  if (chat_format_turn(CHAT_CHATML, NULL, "hi", false, buf, sizeof buf) == 0) abort();
  ids = tokenizer_encode(&t, buf, false, &n);
  if (!ids || ids[0] != im_e || st_count(ids, n, im_s) != 2) {
    fprintf(stderr, "chat_selftest: chatml follow-up open wrong\n");
    abort();
  }
  free(ids);

  /* --- Llama-3: dedicated system turn, <|eot_id|> closes each turn ------- */
  if (chat_detect(&t, "...<|start_header_id|>...") != CHAT_LLAMA3) abort();
  if (chat_format_turn(CHAT_LLAMA3, "sys", "hi", true, buf, sizeof buf) == 0) abort();
  ids = tokenizer_encode(&t, buf, false, &n);
  if (!ids || ids[0] != sh_s || st_count(ids, n, sh_s) != 3 ||
      st_count(ids, n, sh_e) != 3 || st_count(ids, n, eot) != 2) {
    fprintf(stderr, "chat_selftest: llama3 first-turn control tokens wrong\n");
    abort();
  }
  free(ids);

  /* --- Gemma: no system role, folds system into the user turn ----------- */
  if (chat_detect(&t, "...<start_of_turn>...") != CHAT_GEMMA) abort();
  if (chat_format_turn(CHAT_GEMMA, "sys", "hi", true, buf, sizeof buf) == 0) abort();
  ids = tokenizer_encode(&t, buf, false, &n);
  if (!ids || ids[0] != got || st_count(ids, n, got) != 2 ||
      st_count(ids, n, eom) != 1) {
    fprintf(stderr, "chat_selftest: gemma first-turn control tokens wrong\n");
    abort();
  }
  free(ids);

  /* --- fallback: no template, no known marker in this vocab probe -------- *
   * (vocab has im_start etc., so probe would find ChatML — that is correct
   * generic behavior.) An empty template string with unknown markers falls
   * through to the vocab probe; a NULL tokenizer forces the ChatML default. */
  if (chat_detect(NULL, NULL) != CHAT_CHATML) abort();
  if (chat_detect(NULL, "no known markers here") != CHAT_CHATML) abort();

  tokenizer_free(&t);
  gguf_close(&g);
  free(b);
  printf("ok chat templates: detect+format+tokenize "
         "(chatml/llama3/gemma control ids verified; mistral/phi3/gemma4 tabled)\n");
  return 0;
}
