/* tokenizer_tiktoken.c — raw Tiktoken tokenizer (byte-level, no GPT-2
 * byte_to_unicode mapping), for models that declare `tokenizer.ggml.model
 * == "tiktoken"`.
 *
 * Faithful port of oxidize-core/src/format/tokenizer.rs::
 *   - `load_tiktoken`                          (load from GGUF metadata)
 *   - `TiktokenTokenizer::new`                 (test constructor)
 *   - `TiktokenTokenizer::with_unknown_token`
 *   - `TiktokenTokenizer::encode`              (byte-level BPE merge loop)
 *   - `TiktokenTokenizer::decode`              (concatenate byte sequences)
 *
 * Differences from the BPE path (tokenizer_bpe.c):
 *   - Vocab keys are raw byte sequences (Vec<u8> in Rust), NOT GPT-2-mapped
 *     strings. A tiktoken token may contain any byte value including NUL,
 *     so the vocab uses a byte-sequence hash map (FNV-1a + linear probing)
 *     rather than the NUL-terminated OcHashtable used by BPE.
 *   - No `special_pieces` pre-split (raw tiktoken GGUFs do not carry
 *     CONTROL/USER_DEFINED piece metadata the way Qwen GGUFs do).
 *   - No `byte_to_unicode` mapping on encode or decode — input bytes are
 *     looked up directly in the vocab.
 *
 * Encode algorithm (mirrors Rust `TiktokenTokenizer::encode`):
 *   1. For each input byte, look up the single-byte token in the vocab;
 *      emit its id, or the `<unk>` id when absent (VAL-TOK-009:
 *      "<unk> fallback for OOV characters matches Rust").
 *   2. Run the standard BPE merge loop: repeatedly find the adjacent
 *      (left, right) id pair with the lowest merge rank, replace every
 *      non-overlapping occurrence with its merged id, and repeat until
 *      no merge applies. Identical to the BPE merge loop except the
 *      vocab is byte-keyed.
 *
 * Decode (mirrors Rust `TiktokenTokenizer::decode`): concatenate the byte
 * sequences of each id, then apply `String::from_utf8_lossy` (replace
 * invalid UTF-8 with U+FFFD).
 */

#define _POSIX_C_SOURCE 200809L  /* strdup */

#include "oxidize/tokenizer.h"

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/log.h"
#include "oxidize/vector.h"

#include "utf8_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── Byte-sequence open-addressing hash map ─────────────────────────── */

/* FNV-1a 64-bit hash of a byte sequence. */
static uint64_t tikt_fnv1a_bytes(const uint8_t *data, size_t len)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* A byte-sequence → u32 id open-addressing hash map. Keys are stored as
 * (data, len) pairs owned by the map (malloc'd). Grows when load factor
 * > 0.7. Tombstone-free. */
typedef struct {
    uint8_t *keys_data;   /* concatenated key bytes (all keys packed)        */
    size_t  *keys_off;     /* offset into keys_data for key i (cap+1 sentinels) */
    size_t  *keys_len;     /* byte length of key i                            */
    uint32_t *values;      /* id for key i                                    */
    bool    *occupied;     /* slot is live                                    */
    size_t   cap;          /* power of two                                    */
    size_t   count;        /* live entries                                    */
    /* Backing storage for the concatenated key bytes; grown as needed. */
    uint8_t *byte_buf;
    size_t   byte_buf_len;
    size_t   byte_buf_cap;
} OcByteMap;

static OcByteMap *tikt_bytemap_new(size_t initial_cap)
{
    if (initial_cap < 16) initial_cap = 16;
    size_t cap = 1;
    while (cap < initial_cap) cap <<= 1;
    OcByteMap *m = (OcByteMap *)calloc(1, sizeof(OcByteMap));
    if (!m) return NULL;
    m->keys_data = NULL;  /* unused; we store per-slot key pointers */
    m->keys_off = (size_t *)calloc(cap, sizeof(size_t));
    m->keys_len = (size_t *)calloc(cap, sizeof(size_t));
    m->values   = (uint32_t *)calloc(cap, sizeof(uint32_t));
    m->occupied = (bool *)calloc(cap, sizeof(bool));
    m->byte_buf = (uint8_t *)malloc(256);
    m->byte_buf_cap = 256;
    m->byte_buf_len = 0;
    m->cap = cap;
    m->count = 0;
    if (!m->keys_off || !m->keys_len || !m->values || !m->occupied || !m->byte_buf) {
        free(m->keys_off); free(m->keys_len); free(m->values);
        free(m->occupied); free(m->byte_buf); free(m);
        return NULL;
    }
    return m;
}

static void tikt_bytemap_free(OcByteMap *m)
{
    if (!m) return;
    free(m->keys_off);
    free(m->keys_len);
    free(m->values);
    free(m->occupied);
    free(m->byte_buf);
    free(m);
}

/* Ensure byte_buf has at least `need` more bytes available. */
static bool tikt_bytemap_reserve_bytes(OcByteMap *m, size_t need)
{
    if (m->byte_buf_len + need <= m->byte_buf_cap) return true;
    size_t new_cap = m->byte_buf_cap;
    while (new_cap < m->byte_buf_len + need) new_cap <<= 1;
    uint8_t *p = (uint8_t *)realloc(m->byte_buf, new_cap);
    if (!p) return false;
    m->byte_buf = p;
    m->byte_buf_cap = new_cap;
    return true;
}

static bool tikt_bytemap_grow(OcByteMap *m, size_t new_cap)
{
    size_t *old_off = m->keys_off;
    size_t *old_len = m->keys_len;
    uint32_t *old_values = m->values;
    bool *old_occupied = m->occupied;
    size_t old_cap = m->cap;

    m->keys_off = (size_t *)calloc(new_cap, sizeof(size_t));
    m->keys_len = (size_t *)calloc(new_cap, sizeof(size_t));
    m->values   = (uint32_t *)calloc(new_cap, sizeof(uint32_t));
    m->occupied = (bool *)calloc(new_cap, sizeof(bool));
    if (!m->keys_off || !m->keys_len || !m->values || !m->occupied) {
        free(m->keys_off); free(m->keys_len); free(m->values); free(m->occupied);
        m->keys_off = old_off; m->keys_len = old_len;
        m->values = old_values; m->occupied = old_occupied;
        return false;
    }
    m->cap = new_cap;
    m->count = 0;
    for (size_t i = 0; i < old_cap; ++i) {
        if (old_occupied[i]) {
            const uint8_t *key = m->byte_buf + old_off[i];
            size_t key_len = old_len[i];
            uint64_t hash = tikt_fnv1a_bytes(key, key_len) & (m->cap - 1);
            while (m->occupied[hash]) {
                hash = (hash + 1) & (m->cap - 1);
            }
            m->keys_off[hash] = old_off[i];
            m->keys_len[hash] = key_len;
            m->values[hash] = old_values[i];
            m->occupied[hash] = true;
            m->count++;
        }
    }
    free(old_off); free(old_len); free(old_values); free(old_occupied);
    return true;
}

/* Insert key→value. The key bytes are copied into the map's byte_buf. */
static OcError tikt_bytemap_put(OcByteMap *m, const uint8_t *key, size_t key_len,
                                uint32_t value)
{
    if ((m->count + 1) * 10 >= m->cap * 7) {
        if (!tikt_bytemap_grow(m, m->cap << 1)) return OC_ERR_OOM;
    }
    if (!tikt_bytemap_reserve_bytes(m, key_len)) return OC_ERR_OOM;
    size_t off = m->byte_buf_len;
    memcpy(m->byte_buf + off, key, key_len);
    m->byte_buf_len += key_len;

    uint64_t hash = tikt_fnv1a_bytes(key, key_len) & (m->cap - 1);
    while (m->occupied[hash]) {
        /* Compare existing key. */
        if (m->keys_len[hash] == key_len &&
            memcmp(m->byte_buf + m->keys_off[hash], key, key_len) == 0) {
            /* Replace value. The old offset's bytes are now orphaned in
             * byte_buf but that's fine — byte_buf is never compacted. */
            m->values[hash] = value;
            return OC_OK;
        }
        hash = (hash + 1) & (m->cap - 1);
    }
    m->keys_off[hash] = off;
    m->keys_len[hash] = key_len;
    m->values[hash] = value;
    m->occupied[hash] = true;
    m->count++;
    return OC_OK;
}

/* Lookup. Returns true and writes `*out` if found. */
static bool tikt_bytemap_get(const OcByteMap *m, const uint8_t *key, size_t key_len,
                             uint32_t *out)
{
    uint64_t hash = tikt_fnv1a_bytes(key, key_len) & (m->cap - 1);
    while (m->occupied[hash]) {
        if (m->keys_len[hash] == key_len &&
            memcmp(m->byte_buf + m->keys_off[hash], key, key_len) == 0) {
            *out = m->values[hash];
            return true;
        }
        hash = (hash + 1) & (m->cap - 1);
    }
    return false;
}

/* ─── u64 → u32 open-addressing map (merge ranks / merged ids) ────────
 * Duplicated from tokenizer_bpe.c (the BPE u64map is file-static there).
 * Kept local so this file is self-contained. */

static uint64_t tikt_fnv1a_u64(uint64_t key)
{
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 8; ++i) {
        h ^= (key >> (i * 8)) & 0xFF;
        h *= 1099511628211ull;
    }
    return h;
}

typedef struct {
    uint64_t *keys;
    uint32_t *values;
    size_t    cap;
    size_t    count;
} TiktU64Map;

#define TIKT_U64_EMPTY UINT64_MAX

static TiktU64Map *tikt_u64map_new(size_t initial_cap)
{
    if (initial_cap < 16) initial_cap = 16;
    size_t cap = 1;
    while (cap < initial_cap) cap <<= 1;
    TiktU64Map *m = (TiktU64Map *)calloc(1, sizeof(TiktU64Map));
    if (!m) return NULL;
    m->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    m->values = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!m->keys || !m->values) {
        free(m->keys); free(m->values); free(m);
        return NULL;
    }
    for (size_t i = 0; i < cap; ++i) m->keys[i] = TIKT_U64_EMPTY;
    m->cap = cap; m->count = 0;
    return m;
}

static void tikt_u64map_free(TiktU64Map *m)
{
    if (!m) return;
    free(m->keys); free(m->values); free(m);
}

static bool tikt_u64map_grow(TiktU64Map *m, size_t new_cap)
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
    for (size_t i = 0; i < new_cap; ++i) m->keys[i] = TIKT_U64_EMPTY;
    m->cap = new_cap; m->count = 0;
    for (size_t i = 0; i < old_cap; ++i) {
        if (old_keys[i] != TIKT_U64_EMPTY) {
            uint64_t key = old_keys[i];
            uint32_t hash = (uint32_t)tikt_fnv1a_u64(key) & (m->cap - 1);
            while (m->keys[hash] != TIKT_U64_EMPTY) {
                hash = (hash + 1) & (m->cap - 1);
            }
            m->keys[hash] = key;
            m->values[hash] = old_values[i];
            m->count++;
        }
    }
    free(old_keys); free(old_values);
    return true;
}

static OcError tikt_u64map_put(TiktU64Map *m, uint64_t key, uint32_t value)
{
    if (key == TIKT_U64_EMPTY) return OC_ERR_INVALID_ARG;
    if ((m->count + 1) * 10 >= m->cap * 7) {
        if (!tikt_u64map_grow(m, m->cap << 1)) return OC_ERR_OOM;
    }
    uint32_t hash = (uint32_t)tikt_fnv1a_u64(key) & (m->cap - 1);
    while (m->keys[hash] != TIKT_U64_EMPTY) {
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

static bool tikt_u64map_get(const TiktU64Map *m, uint64_t key, uint32_t *out)
{
    if (key == TIKT_U64_EMPTY) return false;
    uint32_t hash = (uint32_t)tikt_fnv1a_u64(key) & (m->cap - 1);
    while (m->keys[hash] != TIKT_U64_EMPTY) {
        if (m->keys[hash] == key) {
            *out = m->values[hash];
            return true;
        }
        hash = (hash + 1) & (m->cap - 1);
    }
    return false;
}

static inline uint64_t tikt_pair_key(uint32_t left, uint32_t right)
{
    return ((uint64_t)left << 32) | (uint64_t)right;
}

/* ─── Tiktoken tokenizer state ─────────────────────────────────────── */

struct OcTiktokenTokenizer {
    /* vocab: byte sequence → id. */
    OcByteMap *vocab;
    /* id → byte slice (dense array indexed by id). Bytes are arena-owned. */
    uint8_t **id_to_token_data;
    size_t   *id_to_token_len;
    size_t    vocab_size;
    /* merge ranks: pair_key → rank (lowest rank wins). */
    TiktU64Map *merge_ranks;
    /* merged ids: pair_key → merged_token_id. */
    TiktU64Map *merged_ids;
    /* Fast single-byte lookup: byte → id (or UINT32_MAX if absent).
     * Built at load time so the hot encode path avoids hash lookups. */
    uint32_t single_byte_id[256];
    /* Special-token ids (also mirrored in the OcTokenizer wrapper). */
    uint32_t  unknown_id;  bool has_unknown;
    uint32_t  bos_id;      bool has_bos;
    uint32_t  eos_id;      bool has_eos;
    uint32_t  pad_id;      bool has_pad;
    uint32_t  separator_id; bool has_separator;
    uint32_t  cls_id;      bool has_cls;
    uint32_t  mask_id;     bool has_mask;
};

/* ─── Constructor (mirrors Rust `TiktokenTokenizer::new`) ──────────── */

OcError oc_tiktoken_new(const OcByteSlice *vocab_tokens, size_t n_vocab,
                        const OcByteSlicePair *merge_pairs, size_t n_merges,
                        OcArena *arena, OcTiktokenTokenizer **out)
{
    if (!vocab_tokens || !arena || !out) return OC_ERR_INVALID_ARG;
    if (n_merges > 0 && !merge_pairs) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcTiktokenTokenizer *t = (OcTiktokenTokenizer *)
        oc_arena_alloc(arena, sizeof(*t), sizeof(void *));
    if (!t) return OC_ERR_OOM;
    memset(t, 0, sizeof(*t));

    t->vocab = tikt_bytemap_new(n_vocab * 2 > 0 ? n_vocab * 2 : 16);
    t->merge_ranks = tikt_u64map_new(n_merges > 0 ? n_merges * 2 : 64);
    t->merged_ids = tikt_u64map_new(n_merges > 0 ? n_merges * 2 : 64);
    t->vocab_size = n_vocab;
    t->id_to_token_data = oc_arena_alloc(arena, n_vocab * sizeof(uint8_t *), sizeof(void *));
    t->id_to_token_len = oc_arena_alloc(arena, n_vocab * sizeof(size_t), sizeof(void *));
    if (!t->vocab || !t->merge_ranks || !t->merged_ids
        || !t->id_to_token_data || !t->id_to_token_len) {
        return OC_ERR_OOM;
    }

    /* Init single-byte table to "absent". */
    for (int i = 0; i < 256; ++i) t->single_byte_id[i] = UINT32_MAX;

    /* Build vocab + id_to_token. Copy each token's bytes into the arena so
     * they outlive the caller's input arrays. */
    for (size_t id = 0; id < n_vocab; ++id) {
        const uint8_t *data = vocab_tokens[id].data;
        size_t len = vocab_tokens[id].len;
        uint8_t *copy = oc_arena_alloc(arena, len ? len : 1, 1);
        if (!copy) return OC_ERR_OOM;
        if (len) memcpy(copy, data, len);
        t->id_to_token_data[id] = copy;
        t->id_to_token_len[id] = len;
        OcError e = tikt_bytemap_put(t->vocab, copy, len, (uint32_t)id);
        if (e != OC_OK) return e;
        if (len == 1) {
            t->single_byte_id[data[0]] = (uint32_t)id;
        }
    }

    /* Build merge ranks + merged ids. */
    for (size_t rank = 0; rank < n_merges; ++rank) {
        const uint8_t *ldata = merge_pairs[rank].left.data;
        size_t llen = merge_pairs[rank].left.len;
        const uint8_t *rdata = merge_pairs[rank].right.data;
        size_t rlen = merge_pairs[rank].right.len;

        uint32_t left_id, right_id, merged_id;
        if (!tikt_bytemap_get(t->vocab, ldata, llen, &left_id)) continue;
        if (!tikt_bytemap_get(t->vocab, rdata, rlen, &right_id)) continue;
        /* Build the merged token: left + right. */
        size_t merged_len = llen + rlen;
        uint8_t *merged = (uint8_t *)malloc(merged_len ? merged_len : 1);
        if (!merged) return OC_ERR_OOM;
        if (llen) memcpy(merged, ldata, llen);
        if (rlen) memcpy(merged + llen, rdata, rlen);
        bool found = tikt_bytemap_get(t->vocab, merged, merged_len, &merged_id);
        free(merged);
        if (!found) continue;

        tikt_u64map_put(t->merge_ranks, tikt_pair_key(left_id, right_id), (uint32_t)rank);
        tikt_u64map_put(t->merged_ids, tikt_pair_key(left_id, right_id), merged_id);
    }

    *out = t;
    return OC_OK;
}

OcError oc_tiktoken_with_unknown_token(OcTiktokenTokenizer *t, OcArena *arena,
                                       const char *token)
{
    if (!t || !arena || !token) return OC_ERR_INVALID_ARG;
    size_t token_len = strlen(token);
    uint32_t id;
    if (tikt_bytemap_get(t->vocab, (const uint8_t *)token, token_len, &id)) {
        /* Already in vocab. */
    } else {
        /* Rust allocates a new id = (max existing id) + 1. We grow the
         * id_to_token arrays by allocating new ones in the arena. */
        id = (uint32_t)t->vocab_size;
        uint8_t *copy = oc_arena_alloc(arena, token_len ? token_len : 1, 1);
        if (!copy) return OC_ERR_OOM;
        if (token_len) memcpy(copy, (const uint8_t *)token, token_len);
        uint8_t **new_data = oc_arena_alloc(arena, (id + 1) * sizeof(uint8_t *), sizeof(void *));
        size_t   *new_len = oc_arena_alloc(arena, (id + 1) * sizeof(size_t), sizeof(void *));
        if (!new_data || !new_len) return OC_ERR_OOM;
        for (size_t i = 0; i < t->vocab_size; ++i) {
            new_data[i] = t->id_to_token_data[i];
            new_len[i] = t->id_to_token_len[i];
        }
        new_data[id] = copy;
        new_len[id] = token_len;
        t->id_to_token_data = new_data;
        t->id_to_token_len = new_len;
        t->vocab_size = id + 1;
        tikt_bytemap_put(t->vocab, copy, token_len, id);
    }
    t->has_unknown = true;
    t->unknown_id = id;
    return OC_OK;
}

/* ─── Encode ─────────────────────────────────────────────────────────── */

/* Find the (left, right) pair present in the sequence with the lowest rank.
 * Mirrors Rust `best_merge_for_sequence`. */
static bool tikt_find_best_merge(const OcTiktokenTokenizer *t,
                                const uint32_t *seq, size_t n,
                                uint32_t *out_left, uint32_t *out_right,
                                uint32_t *out_merged_id)
{
    uint32_t best_rank = UINT32_MAX;
    bool found = false;
    uint64_t best_key = 0;
    for (size_t i = 0; i + 1 < n; ++i) {
        uint64_t key = tikt_pair_key(seq[i], seq[i + 1]);
        uint32_t rank;
        if (tikt_u64map_get(t->merge_ranks, key, &rank)) {
            if (rank < best_rank) {
                best_rank = rank;
                best_key = key;
                found = true;
            }
        }
    }
    if (!found) return false;
    uint32_t merged_id;
    if (!tikt_u64map_get(t->merged_ids, best_key, &merged_id)) return false;
    *out_left = (uint32_t)(best_key >> 32);
    *out_right = (uint32_t)(best_key & 0xFFFFFFFFu);
    *out_merged_id = merged_id;
    return true;
}

/* Apply a merge in-place: replace every non-overlapping (left, right) with
 * merged_id. Writes into `out` (caller-allocated, at least `n` entries).
 * Returns the new length. */
static size_t tikt_apply_merge(const uint32_t *seq, size_t n,
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

OcError oc_tiktoken_encode(const OcTiktokenTokenizer *t, const char *text,
                           uint32_t **out_ids, size_t *out_count)
{
    if (!t || !text || !out_ids || !out_count) return OC_ERR_INVALID_ARG;
    *out_ids = NULL;
    *out_count = 0;

    size_t text_len = strlen(text);
    if (text_len == 0) return OC_OK;

    /* Step 1: build the initial id sequence — one id per input byte. */
    size_t cap = text_len;
    uint32_t *seq = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!seq) return OC_ERR_OOM;
    size_t n = 0;
    const uint8_t *bytes = (const uint8_t *)text;
    for (size_t i = 0; i < text_len; ++i) {
        uint32_t id = t->single_byte_id[bytes[i]];
        if (id == UINT32_MAX) {
            /* Single-byte token not in vocab. Rust uses `.or(unknown)`. */
            if (t->has_unknown) {
                id = t->unknown_id;
            } else {
                /* No unknown token configured: skip this byte (mirrors Rust
                 * `filter_map` dropping None). */
                continue;
            }
        }
        seq[n++] = id;
    }

    /* Step 2: BPE merge loop. */
    while (n >= 2) {
        uint32_t left, right, merged_id;
        if (!tikt_find_best_merge(t, seq, n, &left, &right, &merged_id)) {
            break;
        }
        uint32_t *next = (uint32_t *)malloc(n * sizeof(uint32_t));
        if (!next) { free(seq); return OC_ERR_OOM; }
        n = tikt_apply_merge(seq, n, left, right, merged_id, next);
        free(seq);
        seq = next;
    }

    *out_ids = seq;
    *out_count = n;
    return OC_OK;
}

/* ─── Decode ─────────────────────────────────────────────────────────── */

/* Lossy UTF-8 conversion is shared (utf8_utils.h::oc_utf8_lossy) and matches
 * Rust `String::from_utf8_lossy`: one U+FFFD per maximal invalid
 * subsequence (a lead byte plus its longest valid continuation prefix). */

OcError oc_tiktoken_decode(const OcTiktokenTokenizer *t, const uint32_t *ids,
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

    /* First pass: total byte length. */
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t id = ids[i];
        if (id >= t->vocab_size || !t->id_to_token_data[id]) {
            return OC_ERR_TOKENIZER;
        }
        total += t->id_to_token_len[id];
    }

    /* Concatenate all token byte sequences. */
    uint8_t *bytes = (uint8_t *)malloc(total ? total : 1);
    if (!bytes) return OC_ERR_OOM;
    size_t off = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t len = t->id_to_token_len[ids[i]];
        if (len) memcpy(bytes + off, t->id_to_token_data[ids[i]], len);
        off += len;
    }

    /* Apply lossy UTF-8 conversion (mirrors Rust `from_utf8_lossy`). Worst
     * case: every byte expands to the 3-byte U+FFFD replacement. */
    size_t out_cap = total * 3 + 1;
    uint8_t *out = (uint8_t *)malloc(out_cap);
    if (!out) { free(bytes); return OC_ERR_OOM; }
    size_t out_len = oc_utf8_lossy(bytes, total, out);
    out[out_len] = '\0';
    free(bytes);

    *out_text = (char *)out;
    return OC_OK;
}

/* ─── Load from GGUF metadata ───────────────────────────────────────── */

/* Helper: read a string array from GGUF metadata into a vector of
 * (data, len) slices. The slices alias into the GGUF's arena (caller
 * guarantees arena lifetime). */
static OcError tikt_get_string_slices(const OcGgufFile *gguf, const char *key,
                                     OcVector *out_slices)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(gguf, key);
    if (!v || v->type != OC_GGUF_MT_ARRAY) {
        return OC_ERR_TOKENIZER;
    }
    OcError e = oc_vector_init(out_slices, sizeof(OcByteSlice));
    if (e != OC_OK) return e;
    for (size_t i = 0; i < v->v.arr.len; ++i) {
        const OcGgufMetadataValue *elem = &v->v.arr.values[i];
        if (elem->type != OC_GGUF_MT_STRING) {
            oc_vector_free(out_slices);
            return OC_ERR_TOKENIZER;
        }
        OcByteSlice slice;
        slice.data = (const uint8_t *)elem->v.str.data;
        slice.len = elem->v.str.len;
        e = oc_vector_push(out_slices, &slice);
        if (e != OC_OK) { oc_vector_free(out_slices); return e; }
    }
    return OC_OK;
}

OcError oc_tiktoken_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                                   OcTiktokenTokenizer **out)
{
    if (!gguf || !arena || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    /* Read tokens (required) + merges (optional). */
    OcVector tokens;
    OcError e = tikt_get_string_slices(gguf, "tokenizer.ggml.tokens", &tokens);
    if (e != OC_OK) return e;

    OcVector merges;
    const OcGgufMetadataValue *merges_val =
        oc_gguf_metadata_get(gguf, "tokenizer.ggml.merges");
    if (merges_val && merges_val->type == OC_GGUF_MT_ARRAY) {
        e = tikt_get_string_slices(gguf, "tokenizer.ggml.merges", &merges);
        if (e != OC_OK) { oc_vector_free(&tokens); return e; }
    } else {
        e = oc_vector_init(&merges, sizeof(OcByteSlice));
        if (e != OC_OK) { oc_vector_free(&tokens); return e; }
    }

    size_t vocab_size = oc_vector_len(&tokens);
    size_t n_merges = oc_vector_len(&merges);

    OcTiktokenTokenizer *t = (OcTiktokenTokenizer *)
        oc_arena_alloc(arena, sizeof(*t), sizeof(void *));
    if (!t) { oc_vector_free(&tokens); oc_vector_free(&merges); return OC_ERR_OOM; }
    memset(t, 0, sizeof(*t));

    t->vocab = tikt_bytemap_new(vocab_size * 2 > 0 ? vocab_size * 2 : 16);
    t->merge_ranks = tikt_u64map_new(n_merges > 0 ? n_merges * 2 : 64);
    t->merged_ids = tikt_u64map_new(n_merges > 0 ? n_merges * 2 : 64);
    t->vocab_size = vocab_size;
    t->id_to_token_data = oc_arena_alloc(arena, vocab_size * sizeof(uint8_t *), sizeof(void *));
    t->id_to_token_len = oc_arena_alloc(arena, vocab_size * sizeof(size_t), sizeof(void *));
    if (!t->vocab || !t->merge_ranks || !t->merged_ids
        || !t->id_to_token_data || !t->id_to_token_len) {
        oc_vector_free(&tokens); oc_vector_free(&merges);
        oc_tiktoken_free(t);
        return OC_ERR_OOM;
    }

    for (int i = 0; i < 256; ++i) t->single_byte_id[i] = UINT32_MAX;

    /* Build vocab + id_to_token. Dup each token's bytes into the arena so
     * they outlive the GGUF parse (the GGUF arena may be freed before the
     * tokenizer is). */
    for (size_t id = 0; id < vocab_size; ++id) {
        OcByteSlice *slice = (OcByteSlice *)oc_vector_get_mut(&tokens, id);
        uint8_t *copy = oc_arena_alloc(arena, slice->len ? slice->len : 1, 1);
        if (!copy) {
            oc_vector_free(&tokens); oc_vector_free(&merges);
            oc_tiktoken_free(t);
            return OC_ERR_OOM;
        }
        if (slice->len) memcpy(copy, slice->data, slice->len);
        t->id_to_token_data[id] = copy;
        t->id_to_token_len[id] = slice->len;
        e = tikt_bytemap_put(t->vocab, copy, slice->len, (uint32_t)id);
        if (e != OC_OK) {
            oc_vector_free(&tokens); oc_vector_free(&merges);
            oc_tiktoken_free(t);
            return e;
        }
        if (slice->len == 1) {
            t->single_byte_id[slice->data[0]] = (uint32_t)id;
        }
    }

    /* Build merge ranks + merged ids. */
    for (size_t rank = 0; rank < n_merges; ++rank) {
        OcByteSlice *merge = (OcByteSlice *)oc_vector_get_mut(&merges, rank);
        /* Split on first space byte. */
        const uint8_t *mb = merge->data;
        size_t mlen = merge->len;
        const uint8_t *sp = NULL;
        for (size_t k = 0; k < mlen; ++k) {
            if (mb[k] == ' ') { sp = mb + k; break; }
        }
        if (!sp) {
            /* Invalid merge entry — mirrors Rust's `InvalidMergeEntry`
             * error in `load_tiktoken`. */
            oc_log_error("tokenizer_tiktoken: invalid merge entry at rank %zu "
                         "(no space)", rank);
            oc_vector_free(&tokens); oc_vector_free(&merges);
            oc_tiktoken_free(t);
            return OC_ERR_TOKENIZER;
        }
        size_t llen = (size_t)(sp - mb);
        const uint8_t *rdata = sp + 1;
        size_t rlen = mlen - llen - 1;

        uint32_t left_id, right_id, merged_id;
        if (!tikt_bytemap_get(t->vocab, mb, llen, &left_id)) continue;
        if (!tikt_bytemap_get(t->vocab, rdata, rlen, &right_id)) continue;
        /* Build merged token. */
        size_t merged_len = llen + rlen;
        uint8_t *merged = (uint8_t *)malloc(merged_len ? merged_len : 1);
        if (!merged) {
            oc_vector_free(&tokens); oc_vector_free(&merges);
            oc_tiktoken_free(t);
            return OC_ERR_OOM;
        }
        if (llen) memcpy(merged, mb, llen);
        if (rlen) memcpy(merged + llen, rdata, rlen);
        bool found = tikt_bytemap_get(t->vocab, merged, merged_len, &merged_id);
        free(merged);
        if (!found) continue;

        tikt_u64map_put(t->merge_ranks,
                       tikt_pair_key(left_id, right_id), (uint32_t)rank);
        tikt_u64map_put(t->merged_ids,
                       tikt_pair_key(left_id, right_id), merged_id);
    }

    oc_vector_free(&tokens);
    oc_vector_free(&merges);

    /* Read special-token ids from metadata. */
    uint32_t v;
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.unknown_token_id", &v)) {
        t->has_unknown = true; t->unknown_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.bos_token_id", &v)) {
        t->has_bos = true; t->bos_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.eos_token_id", &v)) {
        t->has_eos = true; t->eos_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.padding_token_id", &v)
        || oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.pad_token_id", &v)) {
        t->has_pad = true; t->pad_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.separator_token_id", &v)
        || oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.sep_token_id", &v)) {
        t->has_separator = true; t->separator_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.cls_token_id", &v)) {
        t->has_cls = true; t->cls_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.mask_token_id", &v)) {
        t->has_mask = true; t->mask_id = v;
    }

    *out = t;
    return OC_OK;
}

void oc_tiktoken_fill_special_tokens(const OcTiktokenTokenizer *t, OcTokenizer *out)
{
    if (!t || !out) return;
    out->has_unknown = t->has_unknown;     out->unknown_id = t->unknown_id;
    out->has_bos = t->has_bos;            out->bos_id = t->bos_id;
    out->has_eos = t->has_eos;            out->eos_id = t->eos_id;
    out->has_pad = t->has_pad;             out->pad_id = t->pad_id;
    out->has_separator = t->has_separator; out->separator_id = t->separator_id;
    out->has_cls = t->has_cls;             out->cls_id = t->cls_id;
    out->has_mask = t->has_mask;           out->mask_id = t->mask_id;
}

void oc_tiktoken_free(OcTiktokenTokenizer *t)
{
    if (!t) return;
    tikt_bytemap_free(t->vocab);
    tikt_u64map_free(t->merge_ranks);
    tikt_u64map_free(t->merged_ids);
    t->vocab = NULL;
    t->merge_ranks = NULL;
    t->merged_ids = NULL;
}
