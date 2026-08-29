/* tokenizer_bpe.c — byte-level BPE tokenizer (tiktoken-style) for reference exactly (VAL-TOK-001 / VAL-TOK-011 require bit-exact is NOT applied here because the Rust reference does not apply it either. */

#define _POSIX_C_SOURCE 200809L  /* strdup, strndup */

#include "oxidize/tokenizer.h"

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/hashtable.h"
#include "oxidize/log.h"
#include "oxidize/vector.h"

#include "utf8_utils.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


/* Printable byte ranges (match Rust `gpt2_byte_is_printable`). A byte is */
static bool gpt2_byte_is_printable(uint32_t b)
{
    return (b >= 33 && b <= 126)
        || (b >= 161 && b <= 172)
        || (b >= 174 && b <= 255);
}

/* Precomputed: byte b → codepoint. Codepoints 0..255 are printable bytes
 * (direct identity) or 256..323 (the 68 non-printable bytes). */
static uint32_t byte_to_gpt2_codepoint(uint8_t b)
{
    if (gpt2_byte_is_printable(b)) {
        return (uint32_t)b;
    }
    /* Find the index of b among the non-printable bytes in ascending order. */
    uint32_t index = 0;
    for (uint32_t x = 0; x <= 255; ++x) {
        if (gpt2_byte_is_printable(x)) {
            continue;
        }
        if (x == b) {
            return 256u + index;
        }
        ++index;
    }
    return 0xFFFDu;  /* unreachable for b in 0..=255 */
}

/* Precomputed byte → GPT-2 UTF-8 string. */
static char byte_to_gpt2_str[256][5];
static atomic_int byte_to_gpt2_str_state;  /* zero-initialized */

static void init_byte_to_gpt2_str(void)
{
    int expected = 0;
    if (atomic_load_explicit(&byte_to_gpt2_str_state, memory_order_acquire) == 2) {
        return;
    }
    if (atomic_compare_exchange_strong_explicit(&byte_to_gpt2_str_state,
                                                &expected, 1,
                                                memory_order_acquire,
                                                memory_order_acquire)) {
        for (uint32_t b = 0; b < 256; ++b) {
            uint32_t cp = byte_to_gpt2_codepoint((uint8_t)b);
            char buf[5] = {0};
            size_t n = oc_utf8_encode_cp(cp, buf);
            memcpy(byte_to_gpt2_str[b], buf, n);
            byte_to_gpt2_str[b][n] = '\0';
        }
        atomic_store_explicit(&byte_to_gpt2_str_state, 2, memory_order_release);
        return;
    }
    /* Another thread is initializing — spin until it publishes. The table
     * build is a few microseconds, so a plain spin is fine. */
    while (atomic_load_explicit(&byte_to_gpt2_str_state, memory_order_acquire) != 2) {
        /* spin */
    }
}

/* Reverse map: GPT-2 codepoint → original byte. Returns true and writes 256..=323). Returns false otherwise (caller should emit the char's UTF-8 */
static bool gpt2_codepoint_to_byte(uint32_t cp, uint8_t *out_byte)
{
    if ((cp >= 33 && cp <= 126)
        || (cp >= 161 && cp <= 172)
        || (cp >= 174 && cp <= 255)) {
        *out_byte = (uint8_t)cp;
        return true;
    }
    if (cp >= 256 && cp <= 323) {
        /* Walk the non-printable bytes in ascending order to find the one
         * at index `cp - 256`. */
        uint32_t index = cp - 256;
        uint32_t seen = 0;
        for (uint32_t x = 0; x <= 255; ++x) {
            if (gpt2_byte_is_printable(x)) {
                continue;
            }
            if (seen == index) {
                *out_byte = (uint8_t)x;
                return true;
            }
            ++seen;
        }
    }
    return false;
}


/* FNV-1a 64-bit hash of a u64 key (mixes the bits; the raw key is already
 * well-distributed from token-id pairs so this mainly avoids pathological
 * clustering for sequential ids). */
static uint64_t fnv1a_u64(uint64_t key)
{
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 8; ++i) {
        h ^= (key >> (i * 8)) & 0xFF;
        h *= 1099511628211ull;
    }
    return h;
}

/* Open-addressing u64→u32 map. Stores values in-line (no pointer indirection).
 * Grows when load factor > 0.7. Uses linear probing. Tombstone-free: we
 * never delete entries during the tokenizer's lifetime. */
typedef struct {
    uint64_t *keys;     /* malloc'd, OC_U64_EMPTY = UINT64_MAX marks empty */
    uint32_t *values;   /* malloc'd, parallel to keys */
    size_t    cap;       /* power of two */
    size_t    count;     /* live entries */
} OcU64Map;

#define OC_U64_EMPTY UINT64_MAX

static OcU64Map *u64map_new(size_t initial_cap)
{
    OcU64Map *m = (OcU64Map *)calloc(1, sizeof(OcU64Map));
    if (!m) return NULL;
    if (initial_cap < 16) initial_cap = 16;
    /* Round up to power of two. */
    size_t cap = 1;
    while (cap < initial_cap) cap <<= 1;
    m->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    m->values = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!m->keys || !m->values) {
        free(m->keys); free(m->values); free(m);
        return NULL;
    }
    for (size_t i = 0; i < cap; ++i) m->keys[i] = OC_U64_EMPTY;
    m->cap = cap;
    m->count = 0;
    return m;
}

static void u64map_free(OcU64Map *m)
{
    if (!m) return;
    free(m->keys);
    free(m->values);
    free(m);
}

static bool u64map_grow(OcU64Map *m, size_t new_cap)
{
    uint64_t *old_keys = m->keys;
    uint32_t *old_values = m->values;
    size_t old_cap = m->cap;

    m->keys = (uint64_t *)malloc(new_cap * sizeof(uint64_t));
    m->values = (uint32_t *)malloc(new_cap * sizeof(uint32_t));
    if (!m->keys || !m->values) {
        free(m->keys); free(m->values);
        m->keys = old_keys; m->values = old_values;
        return false;
    }
    for (size_t i = 0; i < new_cap; ++i) m->keys[i] = OC_U64_EMPTY;
    m->cap = new_cap;
    m->count = 0;
    for (size_t i = 0; i < old_cap; ++i) {
        if (old_keys[i] != OC_U64_EMPTY) {
            uint64_t key = old_keys[i];
            uint32_t hash = (uint32_t)fnv1a_u64(key) & (m->cap - 1);
            while (m->keys[hash] != OC_U64_EMPTY) {
                hash = (hash + 1) & (m->cap - 1);
            }
            m->keys[hash] = key;
            m->values[hash] = old_values[i];
            m->count++;
        }
    }
    free(old_keys);
    free(old_values);
    return true;
}

/* Insert key→value. Returns OC_OK or OC_ERR_OOM. */
static OcError u64map_put(OcU64Map *m, uint64_t key, uint32_t value)
{
    if (key == OC_U64_EMPTY) {
        /* 0xFFFFFFFFFFFFFFFF is reserved as the empty sentinel; remap it
         * to a safe placeholder (this never happens in practice because
         * token ids are u32 and packed pairs never reach UINT64_MAX). */
        return OC_ERR_INVALID_ARG;
    }
    if ((m->count + 1) * 10 >= m->cap * 7) {
        if (!u64map_grow(m, m->cap << 1)) {
            return OC_ERR_OOM;
        }
    }
    uint32_t hash = (uint32_t)fnv1a_u64(key) & (m->cap - 1);
    while (m->keys[hash] != OC_U64_EMPTY) {
        if (m->keys[hash] == key) {
            m->values[hash] = value;
            return OC_OK;
        }
        hash = (hash + 1) & (m->cap - 1);
    }
    m->keys[hash] = key;
    m->values[hash] = value;
    m->count++;
    return OC_OK;
}

/* Lookup. Returns true and writes `*out` if found, false otherwise. */
static bool u64map_get(const OcU64Map *m, uint64_t key, uint32_t *out)
{
    if (key == OC_U64_EMPTY) return false;
    uint32_t hash = (uint32_t)fnv1a_u64(key) & (m->cap - 1);
    while (m->keys[hash] != OC_U64_EMPTY) {
        if (m->keys[hash] == key) {
            *out = m->values[hash];
            return true;
        }
        hash = (hash + 1) & (m->cap - 1);
    }
    return false;
}

/* Pack two u32 token ids into a u64 key (left in high 32 bits). */
static inline uint64_t pair_key(uint32_t left, uint32_t right)
{
    return ((uint64_t)left << 32) | (uint64_t)right;
}


/* A CONTROL / USER_DEFINED special token piece (e.g. `<|im_start|>`). */
struct OcBpeSpecialPiece {
    char    *piece;   /* arena-owned, NUL-terminated */
    uint32_t id;
    size_t   len;     /* byte length of piece (for O(n) matching) */
};

struct OcBpeTokenizer {
    /* vocab: GPT-2-encoded token string → id. Keys are arena-owned
     * NUL-terminated strings (the GPT-2 mapping never produces embedded
     * NULs — codepoints 256..323 encode as multi-byte UTF-8). */
    OcHashtable *vocab;          /* string → (void*)(uintptr_t) id */
    char    **id_to_token;
    size_t    vocab_size;
    OcU64Map *merge_ranks;
    /* merged ids: pair_key → merged_token_id. */
    OcU64Map *merged_ids;
    /* Special-token ids from `tokenizer.ggml.*_token_id` metadata. */
    uint32_t  unknown_id;  bool has_unknown;
    uint32_t  bos_id;      bool has_bos;
    uint32_t  eos_id;      bool has_eos;
    uint32_t  pad_id;      bool has_pad;
    uint32_t  separator_id; bool has_separator;
    uint32_t  cls_id;      bool has_cls;
    uint32_t  mask_id;     bool has_mask;
    /* Whether to map input bytes through the GPT-2 byte_to_unicode table
     * (true for Qwen/GPT2 GGUF-loaded tokenizers; false for the toy
     * `oc_bpe_train` constructor). */
    bool      use_byte_fallback;
    /* CONTROL (3) / USER_DEFINED (4) token pieces, sorted by descending
     * length so overlapping markers match greedily (mirrors Rust). */
    struct OcBpeSpecialPiece *special_pieces;
    size_t    n_special_pieces;
};


/* Count adjacent (left, right) id pairs across all sequences and return the most frequent pair. */
static OcError find_most_frequent_pair(const OcVector *sequences,
                                       uint32_t *out_left,
                                       uint32_t *out_right,
                                       bool *out_found)
{
    *out_found = false;
    /* Use a temporary u64→u32 map for pair counts. */
    OcU64Map *counts = u64map_new(64);
    if (!counts) return OC_ERR_OOM;

    uint32_t best_count = 0;
    uint64_t best_key = 0;
    bool any = false;

    for (size_t s = 0; s < oc_vector_len(sequences); ++s) {
        const OcVector *seq = (const OcVector *)oc_vector_get(sequences, s);
        /* `seq` is an OcVector of u32 ids. */
        size_t n = oc_vector_len(seq);
        if (n < 2) continue;
        for (size_t i = 0; i + 1 < n; ++i) {
            uint32_t l = *(const uint32_t *)oc_vector_get(seq, i);
            uint32_t r = *(const uint32_t *)oc_vector_get(seq, i + 1);
            uint64_t key = pair_key(l, r);
            uint32_t cnt = 0;
            u64map_get(counts, key, &cnt);
            cnt += 1;
            OcError e = u64map_put(counts, key, cnt);
            if (e != OC_OK) {
                u64map_free(counts);
                return e;
            }
            if (cnt > best_count || (!any && cnt > 0)) {
                best_count = cnt;
                best_key = key;
                any = true;
            }
        }
    }
    u64map_free(counts);
    if (!any) return OC_OK;
    *out_left = (uint32_t)(best_key >> 32);
    *out_right = (uint32_t)(best_key & 0xFFFFFFFFu);
    *out_found = true;
    return OC_OK;
}

/* Apply a single merge to a sequence: replace every non-overlapping
 * occurrence of (left, right) with merged_id. Mirrors Rust `apply_merge`. */
static OcError apply_merge_to_vec(OcVector *seq, uint32_t left,
                                  uint32_t right, uint32_t merged_id)
{
    size_t n = oc_vector_len(seq);
    OcVector out;
    OcError e = oc_vector_init(&out, sizeof(uint32_t));
    if (e != OC_OK) return e;
    e = oc_vector_reserve(&out, n);
    if (e != OC_OK) { oc_vector_free(&out); return e; }

    size_t i = 0;
    while (i < n) {
        if (i + 1 < n) {
            uint32_t l = *(const uint32_t *)oc_vector_get(seq, i);
            uint32_t r = *(const uint32_t *)oc_vector_get(seq, i + 1);
            if (l == left && r == right) {
                e = oc_vector_push(&out, &merged_id);
                if (e != OC_OK) { oc_vector_free(&out); return e; }
                i += 2;
                continue;
            }
        }
        uint32_t v = *(const uint32_t *)oc_vector_get(seq, i);
        e = oc_vector_push(&out, &v);
        if (e != OC_OK) { oc_vector_free(&out); return e; }
        i += 1;
    }
    /* Replace seq contents with out. */
    oc_vector_free(seq);
    *seq = out;
    return OC_OK;
}

OcError oc_bpe_train(const char *const *corpus, size_t n_corpus,
                     size_t merge_limit, OcArena *arena,
                     OcBpeTokenizer **out)
{
    if (!corpus || !arena || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;
    init_byte_to_gpt2_str();

    OcBpeTokenizer *bpe = (OcBpeTokenizer *)oc_arena_alloc(arena, sizeof(*bpe), sizeof(void*));
    if (!bpe) return OC_ERR_OOM;
    memset(bpe, 0, sizeof(*bpe));
    bpe->vocab = oc_hashtable_new(256);
    bpe->merge_ranks = u64map_new(64);
    bpe->merged_ids = u64map_new(64);
    bpe->use_byte_fallback = false;
    if (!bpe->vocab || !bpe->merge_ranks || !bpe->merged_ids) {
        oc_bpe_free(bpe);
        return OC_ERR_OOM;
    }

    /* Build initial vocab from characters in the corpus. Rust iterates
     * `sample.chars()` — Unicode codepoints. We iterate UTF-8 codepoints. */
    OcVector id_to_token;
    OcError e = oc_vector_init(&id_to_token, sizeof(char *));
    if (e != OC_OK) { oc_bpe_free(bpe); return e; }

    /* Helper: look up or insert a char-string token, returning its id.
     * Sets `e` on allocation failure (caller must check `e` afterwards) so
     * the hashtable and id_to_token never go out of sync. */
    #define INTERN_CHAR(str, len) do {                              \
        void *vp;                                                   \
        if (oc_hashtable_get(bpe->vocab, (str), &vp)) {            \
            *(uint32_t *)&(id) = (uint32_t)(uintptr_t)vp;           \
        } else {                                                    \
            (id) = (uint32_t)oc_vector_len(&id_to_token);          \
            char *dup = oc_arena_dup_n(arena, (str), (len));      \
            if (!dup) { e = OC_ERR_OOM; break; }                    \
            e = oc_vector_push(&id_to_token, &dup);                \
            if (e != OC_OK) break;                                  \
            if (oc_hashtable_put(bpe->vocab, dup,                   \
                                 (void *)(uintptr_t)(id), NULL) != OC_OK) { \
                e = OC_ERR_OOM; break;                              \
            }                                                       \
        }                                                           \
    } while (0)

    /* Build per-sample id sequences. */
    OcVector sequences;
    e = oc_vector_init(&sequences, sizeof(OcVector));
    if (e != OC_OK) {
        oc_vector_free(&id_to_token);
        oc_bpe_free(bpe);
        return e;
    }

    for (size_t s = 0; s < n_corpus; ++s) {
        const char *sample = corpus[s];
        OcVector seq;
        e = oc_vector_init(&seq, sizeof(uint32_t));
        if (e != OC_OK) goto fail;
        const char *p = sample;
        const char *sample_end = sample + strlen(sample);
        while (*p) {
            uint32_t cp;
            size_t adv = oc_utf8_decode_cp(p, (size_t)(sample_end - p), &cp);
            if (adv == 0) adv = 1;
            char buf[5];
            size_t n = oc_utf8_encode_cp(cp, buf);
            buf[n] = '\0';
            uint32_t id;
            INTERN_CHAR(buf, n);
            if (e != OC_OK) { oc_vector_free(&seq); goto fail; }
            e = oc_vector_push(&seq, &id);
            if (e != OC_OK) { oc_vector_free(&seq); goto fail; }
            p += adv;
        }
        e = oc_vector_push(&sequences, &seq);
        if (e != OC_OK) { oc_vector_free(&seq); goto fail; }
    }

    /* Run up to `merge_limit` merge rounds. */
    for (size_t rank = 0; rank < merge_limit; ++rank) {
        uint32_t left, right;
        bool found = false;
        e = find_most_frequent_pair(&sequences, &left, &right, &found);
        if (e != OC_OK) goto fail;
        if (!found) break;
        /* Build the merged token string: id_to_token[left] + id_to_token[right]. */
        const char *ls = bpe->id_to_token ? bpe->id_to_token[left] : NULL;
        const char *rs = bpe->id_to_token ? bpe->id_to_token[right] : NULL;
        /* Fallback to the id_to_token vector if the array isn't set yet. */
        if (!ls) ls = *(char *const *)oc_vector_get(&id_to_token, left);
        if (!rs) rs = *(char *const *)oc_vector_get(&id_to_token, right);
        size_t ll = strlen(ls);
        size_t lr = strlen(rs);
        char *merged = oc_arena_alloc(arena, ll + lr + 1, 1);
        if (!merged) { e = OC_ERR_OOM; goto fail; }
        memcpy(merged, ls, ll);
        memcpy(merged + ll, rs, lr);
        merged[ll + lr] = '\0';

        /* If the merged token is already in the vocab, skip (mirrors Rust). */
        void *existing;
        if (oc_hashtable_get(bpe->vocab, merged, &existing)) {
            /* Still apply the merge to the sequences so counts are correct */
            /* Wait — re-reading Rust: when `vocab.contains_key(&merged)`,
             * it `continue`s WITHOUT applying the merge. So the pair stays
             * unmerged in the sequences. Match that exactly. */
            continue;
        }

        uint32_t merged_id = (uint32_t)oc_vector_len(&id_to_token);
        e = oc_vector_push(&id_to_token, &merged);
        if (e != OC_OK) goto fail;
        if (oc_hashtable_put(bpe->vocab, merged, (void *)(uintptr_t)merged_id, NULL) != OC_OK) {
            e = OC_ERR_OOM;
            goto fail;
        }
        e = u64map_put(bpe->merge_ranks, pair_key(left, right), (uint32_t)rank);
        if (e != OC_OK) goto fail;
        e = u64map_put(bpe->merged_ids, pair_key(left, right), merged_id);
        if (e != OC_OK) goto fail;

        for (size_t s = 0; s < oc_vector_len(&sequences); ++s) {
            OcVector *seq = (OcVector *)oc_vector_get_mut(&sequences, s);
            e = apply_merge_to_vec(seq, left, right, merged_id);
            if (e != OC_OK) goto fail;
        }
    }

    /* Materialize id_to_token as a dense array. */
    bpe->vocab_size = oc_vector_len(&id_to_token);
    bpe->id_to_token = oc_arena_alloc(arena, bpe->vocab_size * sizeof(char *), sizeof(void *));
    if (!bpe->id_to_token) {
        e = OC_ERR_OOM;
        goto fail;
    }
    for (size_t i = 0; i < bpe->vocab_size; ++i) {
        bpe->id_to_token[i] = *(char *const *)oc_vector_get(&id_to_token, i);
    }
    oc_vector_free(&id_to_token);

    /* Free per-sample sequences. */
    for (size_t s = 0; s < oc_vector_len(&sequences); ++s) {
        OcVector *seq = (OcVector *)oc_vector_get_mut(&sequences, s);
        oc_vector_free(seq);
    }
    oc_vector_free(&sequences);

    *out = bpe;
    return OC_OK;

fail:
    oc_vector_free(&id_to_token);
    for (size_t s = 0; s < oc_vector_len(&sequences); ++s) {
        OcVector *seq = (OcVector *)oc_vector_get_mut(&sequences, s);
        oc_vector_free(seq);
    }
    oc_vector_free(&sequences);
    oc_bpe_free(bpe);
    return e;
}

OcError oc_bpe_with_unknown_token(OcBpeTokenizer *bpe, OcArena *arena,
                                  const char *token)
{
    if (!bpe || !arena || !token) return OC_ERR_INVALID_ARG;
    void *vp;
    uint32_t id;
    if (oc_hashtable_get(bpe->vocab, token, &vp)) {
        id = (uint32_t)(uintptr_t)vp;
    } else {
        id = (uint32_t)bpe->vocab_size;
        char *dup = oc_arena_dup(arena, token);
        if (!dup) return OC_ERR_OOM;
        oc_hashtable_put(bpe->vocab, dup, (void *)(uintptr_t)id, NULL);
        /* Grow id_to_token array. We can't realloc arena memory, so allocate
         * a new one and copy. */
        char **new_arr = oc_arena_alloc(arena, (id + 1) * sizeof(char *), sizeof(void *));
        if (!new_arr) return OC_ERR_OOM;
        for (size_t i = 0; i < bpe->vocab_size; ++i) {
            new_arr[i] = bpe->id_to_token[i];
        }
        new_arr[id] = dup;
        bpe->id_to_token = new_arr;
        bpe->vocab_size = id + 1;
    }
    bpe->has_unknown = true;
    bpe->unknown_id = id;
    return OC_OK;
}


/* Find the best merge for a sequence: the (left, right) pair present in the sequence with the lowest rank. */
static bool find_best_merge(const OcBpeTokenizer *bpe,
                            const uint32_t *seq, size_t n,
                            uint32_t *out_left, uint32_t *out_right,
                            uint32_t *out_merged_id)
{
    uint32_t best_rank = UINT32_MAX;
    bool found = false;
    uint64_t best_key = 0;
    /* We must find the pair present in the sequence with the lowest rank.
     * Rust scans the merges map and filters by `present_pairs`. We do the
     * same: for each adjacent pair in the sequence, look up its rank. */
    for (size_t i = 0; i + 1 < n; ++i) {
        uint64_t key = pair_key(seq[i], seq[i + 1]);
        uint32_t rank;
        if (u64map_get(bpe->merge_ranks, key, &rank)) {
            if (rank < best_rank) {
                best_rank = rank;
                best_key = key;
                found = true;
            }
        }
    }
    if (!found) return false;
    uint32_t merged_id;
    if (!u64map_get(bpe->merged_ids, best_key, &merged_id)) {
        return false;
    }
    *out_left = (uint32_t)(best_key >> 32);
    *out_right = (uint32_t)(best_key & 0xFFFFFFFFu);
    *out_merged_id = merged_id;
    return true;
}

/* Apply a merge in-place: replace every non-overlapping (left, right) with
 * merged_id. Mirrors Rust `apply_merge`. Writes into `out` (caller-allocated,
 * at least `n` entries). Returns the new length. */
static size_t apply_merge(const uint32_t *seq, size_t n,
                          uint32_t left, uint32_t right, uint32_t merged_id,
                          uint32_t *out)
{
    size_t j = 0;
    size_t i = 0;
    while (i < n) {
        if (i + 1 < n && seq[i] == left && seq[i + 1] == right) {
            out[j++] = merged_id;
            i += 2;
        } else {
            out[j++] = seq[i];
            i += 1;
        }
    }
    return j;
}

/* Encode a single segment (no special pieces) via byte-level BPE.
 * Mirrors Rust `BpeTokenizer::encode_segment`. */
static OcError bpe_encode_segment(const OcBpeTokenizer *bpe, const char *text,
                                  uint32_t **out_ids, size_t *out_count)
{
    /* Step 1: build the initial id sequence. */
    size_t text_len = strlen(text);

    /* Upper bound on the id sequence length: one id per byte (byte-fallback)
     * or one id per codepoint (char mode). Use text_len as the bound. */
    size_t cap = text_len + 1;
    uint32_t *seq = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!seq && cap > 0) return OC_ERR_OOM;
    size_t n = 0;

    if (bpe->use_byte_fallback) {
        const uint8_t *bytes = (const uint8_t *)text;
        for (size_t i = 0; i < text_len; ++i) {
            const char *key = byte_to_gpt2_str[bytes[i]];
            void *vp;
            uint32_t id;
            if (oc_hashtable_get(bpe->vocab, key, &vp)) {
                id = (uint32_t)(uintptr_t)vp;
            } else if (bpe->has_unknown) {
                id = bpe->unknown_id;
            } else {
                /* No unknown token: skip this byte (Rust's filter_map drops it). */
                continue;
            }
            seq[n++] = id;
        }
    } else {
        const char *p = text;
        const char *text_end = text + text_len;
        while (*p) {
            uint32_t cp;
            size_t adv = oc_utf8_decode_cp(p, (size_t)(text_end - p), &cp);
            if (adv == 0) adv = 1;
            char buf[5];
            size_t bn = oc_utf8_encode_cp(cp, buf);
            buf[bn] = '\0';
            void *vp;
            uint32_t id;
            if (oc_hashtable_get(bpe->vocab, buf, &vp)) {
                id = (uint32_t)(uintptr_t)vp;
            } else if (bpe->has_unknown) {
                id = bpe->unknown_id;
            } else {
                p += adv;
                continue;
            }
            seq[n++] = id;
            p += adv;
        }
    }

    /* Step 2: BPE merge loop. */
    while (n >= 2) {
        uint32_t left, right, merged_id;
        if (!find_best_merge(bpe, seq, n, &left, &right, &merged_id)) {
            break;
        }
        uint32_t *next = (uint32_t *)malloc(n * sizeof(uint32_t));
        if (!next) { free(seq); return OC_ERR_OOM; }
        n = apply_merge(seq, n, left, right, merged_id, next);
        free(seq);
        seq = next;
    }

    *out_ids = seq;
    *out_count = n;
    return OC_OK;
}

/* Encode with special-piece pre-split. Mirrors Rust `BpeTokenizer::encode`. */
OcError oc_bpe_encode(const OcBpeTokenizer *bpe, const char *text,
                      uint32_t **out_ids, size_t *out_count)
{
    if (!bpe || !text || !out_ids || !out_count) return OC_ERR_INVALID_ARG;
    *out_ids = NULL;
    *out_count = 0;

    if (bpe->n_special_pieces == 0) {
        return bpe_encode_segment(bpe, text, out_ids, out_count);
    }

    /* Scan for the earliest-matching special piece (by byte position).
     * Rust finds the minimum `pos` across all pieces; we do the same. */
    OcVector result;
    OcError e = oc_vector_init(&result, sizeof(uint32_t));
    if (e != OC_OK) return e;

    const char *rest = text;
    while (*rest) {
        size_t best_pos = SIZE_MAX;
        size_t best_len = 0;
        uint32_t best_id = 0;
        for (size_t i = 0; i < bpe->n_special_pieces; ++i) {
            const char *piece = bpe->special_pieces[i].piece;
            size_t plen = bpe->special_pieces[i].len;
            /* Find the first occurrence of `piece` in `rest`. */
            const char *found = strstr(rest, piece);
            if (found) {
                size_t pos = (size_t)(found - rest);
                if (pos < best_pos) {
                    best_pos = pos;
                    best_len = plen;
                    best_id = bpe->special_pieces[i].id;
                }
            }
        }
        if (best_pos == SIZE_MAX) {
            /* No more special pieces — encode the rest as a segment. */
            uint32_t *seg_ids;
            size_t seg_count;
            e = bpe_encode_segment(bpe, rest, &seg_ids, &seg_count);
            if (e != OC_OK) { oc_vector_free(&result); return e; }
            if (seg_count > 0) {
                e = oc_vector_push_n(&result, seg_ids, seg_count);
            }
            free(seg_ids);
            if (e != OC_OK) { oc_vector_free(&result); return e; }
            break;
        }
        /* Encode the text before the special piece. */
        if (best_pos > 0) {
            /* Temporarily NUL-terminate at best_pos. We copy to avoid
             * modifying the caller's string. */
            char *prefix = (char *)malloc(best_pos + 1);
            if (!prefix) { oc_vector_free(&result); return OC_ERR_OOM; }
            memcpy(prefix, rest, best_pos);
            prefix[best_pos] = '\0';
            uint32_t *seg_ids;
            size_t seg_count;
            e = bpe_encode_segment(bpe, prefix, &seg_ids, &seg_count);
            free(prefix);
            if (e != OC_OK) { oc_vector_free(&result); return e; }
            if (seg_count > 0) {
                e = oc_vector_push_n(&result, seg_ids, seg_count);
            }
            free(seg_ids);
            if (e != OC_OK) { oc_vector_free(&result); return e; }
        }
        /* Emit the special piece id. */
        e = oc_vector_push(&result, &best_id);
        if (e != OC_OK) { oc_vector_free(&result); return e; }
        rest += best_pos + best_len;
    }

    /* Materialize the result as a malloc'd array. */
    size_t count = oc_vector_len(&result);
    uint32_t *ids = (uint32_t *)malloc((count ? count : 1) * sizeof(uint32_t));
    if (!ids) { oc_vector_free(&result); return OC_ERR_OOM; }
    for (size_t i = 0; i < count; ++i) {
        ids[i] = *(const uint32_t *)oc_vector_get(&result, i);
    }
    oc_vector_free(&result);
    *out_ids = ids;
    *out_count = count;
    return OC_OK;
}


OcError oc_bpe_decode_raw(const OcBpeTokenizer *bpe, const uint32_t *ids,
                          size_t count, uint8_t **out_bytes, size_t *out_len)
{
    if (!bpe || !ids || !out_bytes || !out_len) return OC_ERR_INVALID_ARG;
    *out_bytes = NULL;
    *out_len = 0;

    /* First pass: compute the total byte length of the concatenated token
     * strings. */
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t id = ids[i];
        if (id >= bpe->vocab_size || !bpe->id_to_token[id]) {
            return OC_ERR_TOKENIZER;
        }
        total += strlen(bpe->id_to_token[id]);
    }

    /* Allocate the concatenated string. */
    char *concat = (char *)malloc(total + 1);
    if (!concat) return OC_ERR_OOM;
    size_t off = 0;
    for (size_t i = 0; i < count; ++i) {
        const char *tok = bpe->id_to_token[ids[i]];
        size_t len = strlen(tok);
        memcpy(concat + off, tok, len);
        off += len;
    }
    concat[off] = '\0';

    if (!bpe->use_byte_fallback) {
        *out_bytes = (uint8_t *)concat;
        *out_len = off;
        return OC_OK;
    }

    /* Reverse the GPT-2 byte mapping. */
    /* Upper bound: each UTF-8 char (1-4 bytes) maps to 1 byte, so the output
     * is at most `off` bytes. */
    uint8_t *bytes = (uint8_t *)malloc(off + 1);
    if (!bytes) { free(concat); return OC_ERR_OOM; }
    size_t byte_len = 0;
    const char *p = concat;
    const char *end = concat + off;
    while (p < end) {
        uint32_t cp;
        size_t adv = oc_utf8_decode_cp(p, (size_t)(end - p), &cp);
        if (adv == 0) { p += 1; continue; }
        uint8_t b;
        if (gpt2_codepoint_to_byte(cp, &b)) {
            bytes[byte_len++] = b;
        } else {
            /* Not in the byte mapping — emit the UTF-8 bytes verbatim
             * (mirrors Rust's `None` branch). */
            for (size_t k = 0; k < adv; ++k) {
                bytes[byte_len++] = (uint8_t)p[k];
            }
        }
        p += adv;
    }
    free(concat);

    *out_bytes = bytes;
    *out_len = byte_len;
    return OC_OK;
}

OcError oc_bpe_decode(const OcBpeTokenizer *bpe, const uint32_t *ids,
                      size_t count, char **out_text)
{
    if (!out_text) return OC_ERR_INVALID_ARG;
    *out_text = NULL;
    uint8_t *bytes = NULL;
    size_t byte_len = 0;
    OcError e = oc_bpe_decode_raw(bpe, ids, count, &bytes, &byte_len);
    if (e != OC_OK) return e;

    /* The bytes may not be valid UTF-8 (e.g. mid-multibyte sequences split */
    uint8_t *lossy = (uint8_t *)malloc(byte_len * 3 + 1);
    if (!lossy) { free(bytes); return OC_ERR_OOM; }
    size_t lossy_len = oc_utf8_lossy(bytes, byte_len, lossy);
    lossy[lossy_len] = '\0';
    free(bytes);

    *out_text = (char *)lossy;
    return OC_OK;
}


OcError oc_tokenizer_apply_chat_template(const OcChatMessage *messages,
                                         size_t n_messages,
                                         OcTemplateKind kind,
                                         bool add_generation_prompt,
                                         char **out_text)
{
    if (!out_text || (n_messages > 0 && !messages)) return OC_ERR_INVALID_ARG;
    if (kind != OC_TEMPLATE_CHATML) return OC_ERR_INVALID_ARG;
    *out_text = NULL;

    /* Compute total length. Each message contributes:
     *   "<|im_start|>" (12) + role + "\n" + content + "<|im_end|>\n" (11)
     * Plus optional "<|im_start|>assistant\n" (22). */
    size_t total = 0;
    for (size_t i = 0; i < n_messages; ++i) {
        if (!messages[i].role || !messages[i].content) {
            return OC_ERR_INVALID_ARG;
        }
        total += 12 + strlen(messages[i].role) + 1
               + strlen(messages[i].content) + 11;
    }
    if (add_generation_prompt) {
        total += 22;
    }

    char *out = (char *)malloc(total + 1);
    if (!out) return OC_ERR_OOM;
    size_t off = 0;
    for (size_t i = 0; i < n_messages; ++i) {
        memcpy(out + off, "<|im_start|>", 12); off += 12;
        size_t rl = strlen(messages[i].role);
        memcpy(out + off, messages[i].role, rl); off += rl;
        out[off++] = '\n';
        size_t cl = strlen(messages[i].content);
        memcpy(out + off, messages[i].content, cl); off += cl;
        memcpy(out + off, "<|im_end|>\n", 11); off += 11;
    }
    if (add_generation_prompt) {
        memcpy(out + off, "<|im_start|>assistant\n", 22); off += 22;
    }
    out[off] = '\0';
    *out_text = out;
    return OC_OK;
}


/* Helper: get a metadata string array as a vector of arena-owned strings.
 * Returns OC_OK, OC_ERR_TOKENIZER (missing/invalid), or OC_ERR_OOM. */
static OcError get_string_array(const OcGgufFile *gguf, const char *key,
                                OcArena *arena, OcVector *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(gguf, key);
    if (!v || v->type != OC_GGUF_MT_ARRAY) {
        return OC_ERR_TOKENIZER;
    }
    OcError e = oc_vector_init(out, sizeof(char *));
    if (e != OC_OK) return e;
    for (size_t i = 0; i < v->v.arr.len; ++i) {
        const OcGgufMetadataValue *elem = &v->v.arr.values[i];
        if (elem->type != OC_GGUF_MT_STRING) {
            oc_vector_free(out);
            return OC_ERR_TOKENIZER;
        }
        char *dup = oc_arena_dup_n(arena, elem->v.str.data, elem->v.str.len);
        if (!dup) { oc_vector_free(out); return OC_ERR_OOM; }
        e = oc_vector_push(out, &dup);
        if (e != OC_OK) { oc_vector_free(out); return e; }
    }
    return OC_OK;
}

/* Load a BPE tokenizer from GGUF metadata. Mirrors Rust `load_bpe`. */
OcError oc_bpe_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                              OcBpeTokenizer **out)
{
    if (!gguf || !arena || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;
    init_byte_to_gpt2_str();

    /* Read tokenizer.ggml.tokens (required) and tokenizer.ggml.merges (optional). */
    OcVector tokens;
    OcError e = get_string_array(gguf, "tokenizer.ggml.tokens", arena, &tokens);
    if (e != OC_OK) return e;

    OcVector merges;
    const OcGgufMetadataValue *merges_val = oc_gguf_metadata_get(gguf, "tokenizer.ggml.merges");
    if (merges_val && merges_val->type == OC_GGUF_MT_ARRAY) {
        e = get_string_array(gguf, "tokenizer.ggml.merges", arena, &merges);
        if (e != OC_OK) { oc_vector_free(&tokens); return e; }
    } else {
        e = oc_vector_init(&merges, sizeof(char *));
        if (e != OC_OK) { oc_vector_free(&tokens); return e; }
    }

    size_t vocab_size = oc_vector_len(&tokens);
    OcBpeTokenizer *bpe = (OcBpeTokenizer *)oc_arena_alloc(arena, sizeof(*bpe), sizeof(void*));
    if (!bpe) { oc_vector_free(&tokens); oc_vector_free(&merges); return OC_ERR_OOM; }
    memset(bpe, 0, sizeof(*bpe));
    bpe->vocab = oc_hashtable_new(vocab_size * 2 > 0 ? vocab_size * 2 : 256);
    bpe->merge_ranks = u64map_new(oc_vector_len(&merges) > 0 ? oc_vector_len(&merges) * 2 : 64);
    bpe->merged_ids = u64map_new(oc_vector_len(&merges) > 0 ? oc_vector_len(&merges) * 2 : 64);
    bpe->use_byte_fallback = true;
    bpe->vocab_size = vocab_size;
    bpe->id_to_token = oc_arena_alloc(arena, vocab_size * sizeof(char *), sizeof(void *));
    if (!bpe->vocab || !bpe->merge_ranks || !bpe->merged_ids || !bpe->id_to_token) {
        oc_vector_free(&tokens); oc_vector_free(&merges);
        oc_bpe_free(bpe);
        return OC_ERR_OOM;
    }

    /* Build vocab + id_to_token. */
    for (size_t id = 0; id < vocab_size; ++id) {
        char *tok = *(char *const *)oc_vector_get(&tokens, id);
        bpe->id_to_token[id] = tok;
        oc_hashtable_put(bpe->vocab, tok, (void *)(uintptr_t)id, NULL);
    }

    /* Build merge ranks + merged ids. */
    for (size_t rank = 0; rank < oc_vector_len(&merges); ++rank) {
        const char *merge = *(char *const *)oc_vector_get(&merges, rank);
        /* Split on the first space. */
        const char *sp = strchr(merge, ' ');
        if (!sp) {
            /* Invalid merge entry — mirrors Rust's `InvalidMergeEntry`
             * error in `load_bpe`. */
            oc_log_error("tokenizer_bpe: invalid merge entry \"%s\" (no space)",
                         merge);
            oc_vector_free(&tokens); oc_vector_free(&merges);
            oc_bpe_free(bpe);
            return OC_ERR_TOKENIZER;
        }
        size_t left_len = (size_t)(sp - merge);
        const char *right = sp + 1;
        /* Look up left and right in vocab. We need NUL-terminated keys, so
         * we dup the left part. */
        char *left = oc_arena_dup_n(arena, merge, left_len);
        if (!left) {
            oc_vector_free(&tokens); oc_vector_free(&merges);
            oc_bpe_free(bpe);
            return OC_ERR_OOM;
        }
        void *lvp, *rvp, *mvp;
        if (!oc_hashtable_get(bpe->vocab, left, &lvp)) continue;
        if (!oc_hashtable_get(bpe->vocab, right, &rvp)) continue;
        /* Build merged token string: left + right. */
        size_t right_len = strlen(right);
        char *merged = oc_arena_alloc(arena, left_len + right_len + 1, 1);
        if (!merged) {
            oc_vector_free(&tokens); oc_vector_free(&merges);
            oc_bpe_free(bpe);
            return OC_ERR_OOM;
        }
        memcpy(merged, left, left_len);
        memcpy(merged + left_len, right, right_len);
        merged[left_len + right_len] = '\0';
        if (!oc_hashtable_get(bpe->vocab, merged, &mvp)) continue;
        uint32_t left_id = (uint32_t)(uintptr_t)lvp;
        uint32_t right_id = (uint32_t)(uintptr_t)rvp;
        uint32_t merged_id = (uint32_t)(uintptr_t)mvp;
        u64map_put(bpe->merge_ranks, pair_key(left_id, right_id), (uint32_t)rank);
        u64map_put(bpe->merged_ids, pair_key(left_id, right_id), merged_id);
    }

    /* Collect CONTROL (3) / USER_DEFINED (4) special pieces. */
    const OcGgufMetadataValue *tt = oc_gguf_metadata_get(gguf, "tokenizer.ggml.token_type");
    if (tt && tt->type == OC_GGUF_MT_ARRAY) {
        /* First pass: count special pieces. */
        size_t n_special = 0;
        for (size_t i = 0; i < tt->v.arr.len; ++i) {
            const OcGgufMetadataValue *elem = &tt->v.arr.values[i];
            int32_t t = 0;
            if (elem->type == OC_GGUF_MT_INT32) t = elem->v.i32;
            else if (elem->type == OC_GGUF_MT_UINT32) t = (int32_t)elem->v.u32;
            else continue;
            if ((t == 3 || t == 4) && i < vocab_size) {
                const char *piece = bpe->id_to_token[i];
                if (piece && piece[0] != '\0') {
                    n_special++;
                }
            }
        }
        if (n_special > 0) {
            bpe->special_pieces = oc_arena_alloc(arena,
                                                 n_special * sizeof(*bpe->special_pieces),
                                                 sizeof(void *));
            if (!bpe->special_pieces) {
                oc_vector_free(&tokens); oc_vector_free(&merges);
                oc_bpe_free(bpe);
                return OC_ERR_OOM;
            }
            size_t idx = 0;
            for (size_t i = 0; i < tt->v.arr.len; ++i) {
                const OcGgufMetadataValue *elem = &tt->v.arr.values[i];
                int32_t t = 0;
                if (elem->type == OC_GGUF_MT_INT32) t = elem->v.i32;
                else if (elem->type == OC_GGUF_MT_UINT32) t = (int32_t)elem->v.u32;
                else continue;
                if ((t == 3 || t == 4) && i < vocab_size) {
                    const char *piece = bpe->id_to_token[i];
                    if (piece && piece[0] != '\0') {
                        bpe->special_pieces[idx].piece = (char *)piece;
                        bpe->special_pieces[idx].id = (uint32_t)i;
                        bpe->special_pieces[idx].len = strlen(piece);
                        idx++;
                    }
                }
            }
            bpe->n_special_pieces = idx;
            /* Sort by descending length (mirrors Rust). Use simple insertion
             * sort — the special-piece count is small (< 100 typically). */
            for (size_t i = 1; i < bpe->n_special_pieces; ++i) {
                struct OcBpeSpecialPiece tmp = bpe->special_pieces[i];
                size_t j = i;
                while (j > 0 && bpe->special_pieces[j - 1].len < tmp.len) {
                    bpe->special_pieces[j] = bpe->special_pieces[j - 1];
                    j--;
                }
                bpe->special_pieces[j] = tmp;
            }
        }
    }

    /* Read special-token ids from metadata. */
    uint32_t v;
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.unknown_token_id", &v)) {
        bpe->has_unknown = true; bpe->unknown_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.bos_token_id", &v)) {
        bpe->has_bos = true; bpe->bos_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.eos_token_id", &v)) {
        bpe->has_eos = true; bpe->eos_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.padding_token_id", &v)
        || oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.pad_token_id", &v)) {
        bpe->has_pad = true; bpe->pad_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.separator_token_id", &v)
        || oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.sep_token_id", &v)) {
        bpe->has_separator = true; bpe->separator_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.cls_token_id", &v)) {
        bpe->has_cls = true; bpe->cls_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.mask_token_id", &v)) {
        bpe->has_mask = true; bpe->mask_id = v;
    }

    oc_vector_free(&tokens);
    oc_vector_free(&merges);
    *out = bpe;
    return OC_OK;
}

void oc_bpe_fill_special_tokens(const OcBpeTokenizer *bpe, OcTokenizer *out)
{
    if (!bpe || !out) return;
    out->has_unknown = bpe->has_unknown;     out->unknown_id = bpe->unknown_id;
    out->has_bos = bpe->has_bos;            out->bos_id = bpe->bos_id;
    out->has_eos = bpe->has_eos;            out->eos_id = bpe->eos_id;
    out->has_pad = bpe->has_pad;             out->pad_id = bpe->pad_id;
    out->has_separator = bpe->has_separator; out->separator_id = bpe->separator_id;
    out->has_cls = bpe->has_cls;             out->cls_id = bpe->cls_id;
    out->has_mask = bpe->has_mask;           out->mask_id = bpe->mask_id;
}


OcError oc_tokenizer_encode(const OcTokenizer *t, const char *text,
                            OcSpecialTokenPolicy policy,
                            uint32_t **out_ids, size_t *out_count)
{
    if (!t || !text || !out_ids || !out_count) return OC_ERR_INVALID_ARG;
    *out_ids = NULL;
    *out_count = 0;

    /* Determine whether BOS should be prepended after the kind-specific
     * encode runs. OC_TOK_ADD_BOS mirrors Rust `EncodeOptions { add_bos:
     * true }` — the BOS id is prepended only when the tokenizer has one. */
    bool prepend_bos = (policy == OC_TOK_ADD_BOS) && t->has_bos;

    /* For BPE, OC_TOK_DISALLOW_SPECIAL suppresses special-piece matching. injection-prevention promise (VAL-TOK-004) holds for every kind. */
    OcError e;
    if (t->kind == OC_TOK_KIND_BPE && t->bpe) {
        const OcBpeTokenizer *bpe = t->bpe;
        if (policy == OC_TOK_DISALLOW_SPECIAL) {
            OcBpeTokenizer tmp = *bpe;
            tmp.n_special_pieces = 0;
            e = oc_bpe_encode(&tmp, text, out_ids, out_count);
        } else {
            e = oc_bpe_encode(bpe, text, out_ids, out_count);
        }
    } else if (t->kind == OC_TOK_KIND_SENTENCEPIECE && t->sp) {
        e = oc_sp_encode(t->sp, text, out_ids, out_count);
    } else if (t->kind == OC_TOK_KIND_WORDPIECE && t->wp) {
        e = oc_wp_encode(t->wp, text, out_ids, out_count);
    } else if (t->kind == OC_TOK_KIND_TIKTOKEN && t->tiktoken) {
        e = oc_tiktoken_encode(t->tiktoken, text, out_ids, out_count);
    } else {
        return OC_ERR_TOKENIZER;
    }
    if (e != OC_OK) return e;

    /* Enforce OC_TOK_DISALLOW_SPECIAL for the non-BPE kinds: drop any */
    if (policy == OC_TOK_DISALLOW_SPECIAL && t->kind != OC_TOK_KIND_BPE
        && *out_ids) {
        uint32_t *ids = *out_ids;
        size_t kept = 0;
        for (size_t i = 0; i < *out_count; ++i) {
            if (!oc_tokenizer_is_special(t, ids[i])) {
                ids[kept++] = ids[i];
            }
        }
        *out_count = kept;
    }

    /* Prepend BOS if requested and the tokenizer has one (VAL-TOK-007). */
    if (prepend_bos && out_ids) {
        size_t n = *out_count;
        uint32_t *with_bos = (uint32_t *)malloc((n + 1) * sizeof(uint32_t));
        if (!with_bos) {
            free(*out_ids);
            *out_ids = NULL;
            *out_count = 0;
            return OC_ERR_OOM;
        }
        with_bos[0] = t->bos_id;
        if (n > 0) {
            memcpy(with_bos + 1, *out_ids, n * sizeof(uint32_t));
        }
        free(*out_ids);
        *out_ids = with_bos;
        *out_count = n + 1;
    }
    return OC_OK;
}

OcError oc_tokenizer_decode(const OcTokenizer *t, const uint32_t *ids,
                            size_t count, char **out_text)
{
    if (!t || !out_text) return OC_ERR_INVALID_ARG;
    *out_text = NULL;
    if (count == 0 || !ids) {
        char *empty = (char *)malloc(1);
        if (!empty) return OC_ERR_OOM;
        empty[0] = '\0';
        *out_text = empty;
        return OC_OK;
    }
    if (t->kind == OC_TOK_KIND_BPE && t->bpe) {
        return oc_bpe_decode(t->bpe, ids, count, out_text);
    }
    if (t->kind == OC_TOK_KIND_SENTENCEPIECE && t->sp) {
        return oc_sp_decode(t->sp, ids, count, out_text);
    }
    if (t->kind == OC_TOK_KIND_WORDPIECE && t->wp) {
        return oc_wp_decode(t->wp, ids, count, out_text);
    }
    if (t->kind == OC_TOK_KIND_TIKTOKEN && t->tiktoken) {
        return oc_tiktoken_decode(t->tiktoken, ids, count, out_text);
    }
    return OC_ERR_TOKENIZER;
}

bool oc_tokenizer_is_special(const OcTokenizer *t, uint32_t id)
{
    if (!t) return false;
    return (t->has_unknown && t->unknown_id == id)
        || (t->has_bos && t->bos_id == id)
        || (t->has_eos && t->eos_id == id)
        || (t->has_pad && t->pad_id == id)
        || (t->has_separator && t->separator_id == id)
        || (t->has_cls && t->cls_id == id)
        || (t->has_mask && t->mask_id == id);
}

bool oc_tokenizer_add_bos_default(const OcTokenizer *t)
{
    if (!t) return false;
    if (t->has_add_bos_token) {
        return t->add_bos_token;
    }
    /* Rust default: only SentencePiece adds BOS. BPE does not. */
    return t->kind == OC_TOK_KIND_SENTENCEPIECE;
}


void oc_bpe_free(OcBpeTokenizer *bpe)
{
    if (!bpe) return;
    /* Free the malloc'd u64 maps. All other allocations (strings,
     * id_to_token array, special_pieces) are arena-owned. */
    u64map_free(bpe->merge_ranks);
    u64map_free(bpe->merged_ids);
    oc_hashtable_free(bpe->vocab);
    /* Clear the pointers so a double-free is safe. */
    bpe->merge_ranks = NULL;
    bpe->merged_ids = NULL;
    bpe->vocab = NULL;
}

void oc_tokenizer_free(OcTokenizer *t)
{
    if (!t) return;
    /* Free the per-kind malloc'd internals. The arena owns all other
     * allocations (token strings, id_to_token arrays, scores, etc.). */
    if (t->bpe) {
        oc_bpe_free(t->bpe);
    }
    if (t->sp) {
        oc_sp_free(t->sp);
    }
    if (t->wp) {
        oc_wp_free(t->wp);
    }
    if (t->tiktoken) {
        oc_tiktoken_free(t->tiktoken);
    }
    if (t->arena) {
        oc_arena_free(t->arena);
    }
    memset(t, 0, sizeof(*t));
}
