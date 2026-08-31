/* tokenizer_train.c — BPE tokenizer trainer implementation. */

#define _POSIX_C_SOURCE 200809L  /* strdup, strndup */

#include "oxidize/tokenizer_train.h"

#include "oxidize/error.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define DA_INIT_CAP 16

#define DEFINE_DA(name, type)                                              \
    typedef struct name {                                                  \
        type *data;                                                        \
        size_t len;                                                       \
        size_t cap;                                                       \
    } name;                                                               \
    static OcError name##_push(name *da, type val)                         \
    {                                                                      \
        if (da->len >= da->cap) {                                         \
            size_t ncap = da->cap > 0 ? da->cap * 2 : DA_INIT_CAP;        \
            type *nd = realloc(da->data, ncap * sizeof(type));            \
            if (nd == NULL) {                                             \
                return OC_ERR_OOM;                                       \
            }                                                            \
            da->data = nd;                                               \
            da->cap = ncap;                                              \
        }                                                                \
        da->data[da->len++] = val;                                       \
        return OC_OK;                                                    \
    }                                                                    \
    static void name##_free(name *da)                                    \
    {                                                                    \
        free(da->data);                                                  \
        da->data = NULL;                                                 \
        da->len = 0;                                                     \
        da->cap = 0;                                                     \
    }

DEFINE_DA(OcUint32Array, uint32_t)
DEFINE_DA(OcWordArray, OcUint32Array)

/* Vocab dynamic array. */
typedef struct OcVocabArray {
    OcBpeVocabEntry *data;
    size_t len;
    size_t cap;
} OcVocabArray;

static OcError vocab_push(OcVocabArray *va, const char *token, uint32_t id)
{
    if (va->len >= va->cap) {
        size_t ncap = va->cap > 0 ? va->cap * 2 : DA_INIT_CAP;
        OcBpeVocabEntry *nd = realloc(va->data, ncap * sizeof(OcBpeVocabEntry));
        if (nd == NULL) {
            return OC_ERR_OOM;
        }
        va->data = nd;
        va->cap = ncap;
    }
    char *dup = strdup(token);
    if (dup == NULL) {
        return OC_ERR_OOM;
    }
    va->data[va->len].token = dup;
    va->data[va->len].id = id;
    va->len++;
    return OC_OK;
}

static void vocab_free(OcVocabArray *va)
{
    for (size_t i = 0; i < va->len; i++) {
        free(va->data[i].token);
    }
    free(va->data);
    va->data = NULL;
    va->len = 0;
    va->cap = 0;
}

/* Merge rules dynamic array. */
typedef struct OcMergeArray {
    OcBpeMerge *data;
    size_t len;
    size_t cap;
} OcMergeArray;

static OcError merge_push(OcMergeArray *ma, uint32_t left, uint32_t right,
                         uint32_t merged)
{
    if (ma->len >= ma->cap) {
        size_t ncap = ma->cap > 0 ? ma->cap * 2 : DA_INIT_CAP;
        OcBpeMerge *nd = realloc(ma->data, ncap * sizeof(OcBpeMerge));
        if (nd == NULL) {
            return OC_ERR_OOM;
        }
        ma->data = nd;
        ma->cap = ncap;
    }
    ma->data[ma->len].left = left;
    ma->data[ma->len].right = right;
    ma->data[ma->len].merged = merged;
    ma->len++;
    return OC_OK;
}

static void merge_free(OcMergeArray *ma)
{
    free(ma->data);
    ma->data = NULL;
    ma->len = 0;
    ma->cap = 0;
}


/* Simple open-addressing hash map: string key -> uint32_t value.
 * Uses FNV-1a hashing + linear probing. */

#define TOKEN_MAP_INIT_CAP 256

typedef struct {
    char    *key;     /* NULL = empty slot; "\0__DELETED__" = tombstone */
    uint32_t value;
} TokenMapEntry;

typedef struct {
    TokenMapEntry *entries;
    size_t cap;
    size_t count;
} TokenMap;

static uint64_t fnv1a(const char *s)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t *p = (const uint8_t *)s;
    while (*p) {
        h ^= (uint64_t)*p;
        h *= 0x100000001b3ULL;
        p++;
    }
    return h;
}

static OcError tokenmap_init(TokenMap *m)
{
    m->cap = TOKEN_MAP_INIT_CAP;
    m->count = 0;
    m->entries = calloc(m->cap, sizeof(TokenMapEntry));
    if (m->entries == NULL) {
        return OC_ERR_OOM;
    }
    return OC_OK;
}

static void tokenmap_free(TokenMap *m)
{
    for (size_t i = 0; i < m->cap; i++) {
        free(m->entries[i].key);
    }
    free(m->entries);
    m->entries = NULL;
    m->cap = 0;
    m->count = 0;
}

static OcError tokenmap_grow(TokenMap *m)
{
    size_t ncap = m->cap * 2;
    TokenMapEntry *nentries = calloc(ncap, sizeof(TokenMapEntry));
    if (nentries == NULL) {
        return OC_ERR_OOM;
    }
    /* Rehash all live entries. */
    for (size_t i = 0; i < m->cap; i++) {
        if (m->entries[i].key != NULL && m->entries[i].key[0] != '\0') {
            /* For simplicity, we never insert empty strings as keys. */
            /* For simplicity, we never insert empty strings as keys. */
            uint64_t h = fnv1a(m->entries[i].key);
            size_t idx = (size_t)(h & (ncap - 1));
            while (nentries[idx].key != NULL) {
                idx = (idx + 1) & (ncap - 1);
            }
            nentries[idx].key = m->entries[i].key;
            nentries[idx].value = m->entries[i].value;
        }
    }
    free(m->entries);
    m->entries = nentries;
    m->cap = ncap;
    return OC_OK;
}

/* Tombstone sentinel: we mark deleted entries with a special key value.
 * Since empty slots have key == NULL, we use a non-NULL sentinel. */
static char tombstone_sentinel = 1; /* address used as marker */

#define IS_TOMBSTONE(e) ((e)->key == &tombstone_sentinel)

static OcError tokenmap_put(TokenMap *m, const char *key, uint32_t value)
{
    if (key == NULL || key[0] == '\0') {
        return OC_ERR_INVALID_ARG;
    }
    /* Grow if load factor > 0.7. */
    if (m->count * 10 >= m->cap * 7) {
        OcError e = tokenmap_grow(m);
        if (e != OC_OK) {
            return e;
        }
    }
    uint64_t h = fnv1a(key);
    size_t idx = (size_t)(h & (m->cap - 1));
    while (m->entries[idx].key != NULL && !IS_TOMBSTONE(&m->entries[idx])) {
        if (strcmp(m->entries[idx].key, key) == 0) {
            /* Key exists — update value. */
            m->entries[idx].value = value;
            return OC_OK;
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    /* Insert. */
    if (IS_TOMBSTONE(&m->entries[idx])) {
        /* Reuse tombstone slot. */
        m->entries[idx].key = strdup(key);
        if (m->entries[idx].key == NULL) {
            return OC_ERR_OOM;
        }
        m->entries[idx].value = value;
    } else {
        m->entries[idx].key = strdup(key);
        if (m->entries[idx].key == NULL) {
            return OC_ERR_OOM;
        }
        m->entries[idx].value = value;
        m->count++;
    }
    return OC_OK;
}

static bool tokenmap_get(const TokenMap *m, const char *key, uint32_t *out)
{
    if (key == NULL || key[0] == '\0') {
        return false;
    }
    uint64_t h = fnv1a(key);
    size_t idx = (size_t)(h & (m->cap - 1));
    while (m->entries[idx].key != NULL) {
        if (!IS_TOMBSTONE(&m->entries[idx]) &&
            strcmp(m->entries[idx].key, key) == 0) {
            *out = m->entries[idx].value;
            return true;
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    return false;
}


struct OcBpeTrainer {
    OcBpeTrainConfig config;
    OcVocabArray     vocab;
    OcMergeArray    merges;
    TokenMap         token_to_id;   /* token string -> vocab id            */
    bool             trained;
};


/* Decode the next UTF-8 codepoint from `s` starting at byte `*i`. Writes
 * the codepoint to `*cp` and advances `*i` by the number of bytes consumed.
 * Returns OC_OK or OC_ERR_FORMAT for invalid UTF-8. */
static OcError utf8_decode(const char *s, size_t len, size_t *i, uint32_t *cp)
{
    if (*i >= len) {
        return OC_ERR_FORMAT;
    }
    uint8_t b0 = (uint8_t)s[*i];
    if (b0 < 0x80) {
        *cp = b0;
        *i += 1;
    } else if ((b0 & 0xE0) == 0xC0) {
        if (*i + 1 >= len) {
            *cp = b0;
            *i += 1;
            return OC_OK;
        }
        *cp = ((uint32_t)(b0 & 0x1F) << 6) | (uint8_t)(s[*i + 1] & 0x3F);
        *i += 2;
    } else if ((b0 & 0xF0) == 0xE0) {
        if (*i + 2 >= len) {
            *cp = b0;
            *i += 1;
            return OC_OK;
        }
        *cp = ((uint32_t)(b0 & 0x0F) << 12) |
              ((uint32_t)(s[*i + 1] & 0x3F) << 6) |
              (uint8_t)(s[*i + 2] & 0x3F);
        *i += 3;
    } else if ((b0 & 0xF8) == 0xF0) {
        if (*i + 3 >= len) {
            *cp = b0;
            *i += 1;
            return OC_OK;
        }
        *cp = ((uint32_t)(b0 & 0x07) << 18) |
              ((uint32_t)(s[*i + 1] & 0x3F) << 12) |
              ((uint32_t)(s[*i + 2] & 0x3F) << 6) |
              (uint8_t)(s[*i + 3] & 0x3F);
        *i += 4;
    } else {
        /* Invalid leading byte — treat as a single byte. */
        *cp = b0;
        *i += 1;
    }
    return OC_OK;
}

/* Encode a Unicode codepoint as UTF-8 into `buf` (must be at least 5 bytes).
 * Returns the number of bytes written. */
static int utf8_encode(uint32_t cp, char *buf)
{
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}


/* Check if a byte is whitespace or punctuation (for pre-tokenization). */
static bool is_word_separator(char c)
{
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
        c == '\f') {
        return true;
    }
    /* Treat common punctuation as separators (and include them as their own
     * tokens so they're not lost). We split on punctuation but keep the
     * punctuation as a separate word. */
    /* For simplicity: punctuation is a separator. Each run of punctuation
     * chars forms a single word. */
    return false;
}

static bool is_punctuation(char c)
{
    /* ASCII punctuation ranges. */
    return (c >= 0x21 && c <= 0x2F) ||
           (c >= 0x3A && c <= 0x40) ||
           (c >= 0x5B && c <= 0x60) ||
           (c >= 0x7B && c <= 0x7E);
}


/* Reset the trainer to a clean state (free existing data, reinit maps). */
static void trainer_reset(OcBpeTrainer *t)
{
    vocab_free(&t->vocab);
    merge_free(&t->merges);
    tokenmap_free(&t->token_to_id);
    /* Zero out the structs so they can be re-initialized. */
    t->vocab = (OcVocabArray){0};
    t->merges = (OcMergeArray){0};
    t->token_to_id = (TokenMap){0};
    t->trained = false;
}

/* Add a token to the vocab if it doesn't already exist. Returns the vocab id
 * via `*id`. Returns OC_OK or OC_ERR_OOM. */
static OcError vocab_add(OcBpeTrainer *t, const char *token, uint32_t *id)
{
    if (tokenmap_get(&t->token_to_id, token, id)) {
        return OC_OK; /* already exists */
    }
    uint32_t new_id = (uint32_t)t->vocab.len;
    OcError e = vocab_push(&t->vocab, token, new_id);
    if (e != OC_OK) {
        return e;
    }
    e = tokenmap_put(&t->token_to_id, token, new_id);
    if (e != OC_OK) {
        return e;
    }
    *id = new_id;
    return OC_OK;
}

/* Build a merged token string from two vocab tokens. Returns a malloc'd
 * string (caller must free) or NULL on OOM. */
static char *build_merged_token(const char *left, const char *right)
{
    size_t llen = strlen(left);
    size_t rlen = strlen(right);
    char *result = malloc(llen + rlen + 1);
    if (result == NULL) {
        return NULL;
    }
    memcpy(result, left, llen);
    memcpy(result + llen, right, rlen);
    result[llen + rlen] = '\0';
    return result;
}

/* Count the most frequent adjacent pair across all words. */
static OcError find_best_pair(const OcWordArray *words, uint32_t *best_left,
                             uint32_t *best_right, size_t *best_count)
{
    typedef struct {
        uint32_t left;
        uint32_t right;
        size_t count;
    } PairCount;

    /* Pair hash map entry. `occupied` distinguishes empty slots from
     * valid key=0 entries (left=0, right=0). */
    typedef struct {
        uint64_t key;
        size_t   idx;       /* index into pairs array (0-based) */
        bool     occupied;
    } PairMapEntry;

    size_t pairs_cap = 256;
    PairCount *pairs = malloc(pairs_cap * sizeof(PairCount));
    if (!pairs) return OC_ERR_OOM;
    size_t npairs = 0;

    size_t map_cap = 256;
    PairMapEntry *map = calloc(map_cap, sizeof(PairMapEntry));
    if (!map) { free(pairs); return OC_ERR_OOM; }

    *best_count = 0;
    *best_left = 0;
    *best_right = 0;

    for (size_t wi = 0; wi < words->len; wi++) {
        const OcUint32Array *word = &words->data[wi];
        for (size_t si = 0; si + 1 < word->len; si++) {
            uint32_t l = word->data[si];
            uint32_t r = word->data[si + 1];
            uint64_t key = ((uint64_t)l << 32) | r;

            /* Probe the hash map. */
            uint64_t h = key * 0x9E3779B97F4A7C15ULL;
            size_t idx = (size_t)(h & (map_cap - 1));

            while (map[idx].occupied) {
                if (map[idx].key == key) {
                    /* Found existing pair — increment count. */
                    pairs[map[idx].idx].count++;
                    goto next_pair;
                }
                idx = (idx + 1) & (map_cap - 1);
            }

            /* Not found — need to insert. Check load factor first. */
            if ((npairs + 1) * 2 >= map_cap) {
                /* Grow the map. */
                size_t ncap = map_cap * 2;
                PairMapEntry *nmap = calloc(ncap, sizeof(PairMapEntry));
                if (!nmap) { free(pairs); free(map); return OC_ERR_OOM; }
                for (size_t j = 0; j < map_cap; j++) {
                    if (map[j].occupied) {
                        uint64_t h2 = map[j].key * 0x9E3779B97F4A7C15ULL;
                        size_t nidx = (size_t)(h2 & (ncap - 1));
                        while (nmap[nidx].occupied) {
                            nidx = (nidx + 1) & (ncap - 1);
                        }
                        nmap[nidx] = map[j];
                    }
                }
                free(map);
                map = nmap;
                map_cap = ncap;

                /* Re-probe for the insertion slot. */
                h = key * 0x9E3779B97F4A7C15ULL;
                idx = (size_t)(h & (map_cap - 1));
                while (map[idx].occupied) {
                    idx = (idx + 1) & (map_cap - 1);
                }
            }

            /* Grow pairs array if needed. */
            if (npairs >= pairs_cap) {
                size_t ncap = pairs_cap * 2;
                PairCount *np = realloc(pairs, ncap * sizeof(PairCount));
                if (!np) { free(pairs); free(map); return OC_ERR_OOM; }
                pairs = np;
                pairs_cap = ncap;
            }

            /* Insert new pair. */
            pairs[npairs].left = l;
            pairs[npairs].right = r;
            pairs[npairs].count = 1;
            map[idx].key = key;
            map[idx].idx = npairs;
            map[idx].occupied = true;
            npairs++;

        next_pair:;
        }
    }

    /* Find the best pair. */
    for (size_t i = 0; i < npairs; i++) {
        if (pairs[i].count > *best_count) {
            *best_count = pairs[i].count;
            *best_left = pairs[i].left;
            *best_right = pairs[i].right;
        }
    }

    free(pairs);
    free(map);
    return OC_OK;
}

/* Apply a merge: replace all occurrences of (left, right) with `merged` in
 * all words. */
static void apply_merge(OcWordArray *words, uint32_t left, uint32_t right,
                       uint32_t merged)
{
    for (size_t wi = 0; wi < words->len; wi++) {
        OcUint32Array *word = &words->data[wi];
        if (word->len < 2) {
            continue;
        }
        /* In-place merge: scan left to right, when we find (left, right),
         * replace with merged and shift the rest. */
        size_t write = 0;
        size_t read = 0;
        while (read < word->len) {
            if (read + 1 < word->len &&
                word->data[read] == left &&
                word->data[read + 1] == right) {
                word->data[write] = merged;
                read += 2;
            } else {
                word->data[write] = word->data[read];
                read += 1;
            }
            write += 1;
        }
        word->len = write;
    }
}

OcError oc_bpe_trainer_train(OcBpeTrainer *t, const char *corpus, size_t corpus_len)
{
    if (t == NULL || corpus == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (corpus_len == 0) {
        corpus_len = strlen(corpus);
    }
    if (corpus_len == 0) {
        return OC_ERR_INVALID_ARG;
    }

    /* Reset trainer state. */
    trainer_reset(t);

    /* Re-init the token map. */
    OcError e = tokenmap_init(&t->token_to_id);
    if (e != OC_OK) {
        return e;
    }

    /* Split into words (whitespace + punctuation boundaries), then split
     * each word into individual UTF-8 codepoint symbols. */

    OcWordArray words = {0};

    size_t i = 0;
    while (i < corpus_len) {
        /* Skip whitespace. */
        while (i < corpus_len && is_word_separator(corpus[i])) {
            i++;
        }
        if (i >= corpus_len) {
            break;
        }

        OcUint32Array word = {0};

        if (is_punctuation(corpus[i])) {
            /* A run of punctuation forms a single word. */
            while (i < corpus_len && is_punctuation(corpus[i])) {
                /* Each punctuation char is its own symbol. */
                uint32_t cp;
                e = utf8_decode(corpus, corpus_len, &i, &cp);
                if (e != OC_OK) {
                    OcUint32Array_free(&word);
                    goto train_fail_words;
                }
                char buf[5];
                int blen = utf8_encode(cp, buf);
                buf[blen] = '\0';
                uint32_t id;
                e = vocab_add(t, buf, &id);
                if (e != OC_OK) {
                    OcUint32Array_free(&word);
                    goto train_fail_words;
                }
                e = OcUint32Array_push(&word, id);
                if (e != OC_OK) {
                    OcUint32Array_free(&word);
                    goto train_fail_words;
                }
            }
        } else {
            /* A run of non-whitespace, non-punctuation chars. */
            while (i < corpus_len && !is_word_separator(corpus[i]) &&
                   !is_punctuation(corpus[i])) {
                uint32_t cp;
                e = utf8_decode(corpus, corpus_len, &i, &cp);
                if (e != OC_OK) {
                    OcUint32Array_free(&word);
                    goto train_fail_words;
                }
                char buf[5];
                int blen = utf8_encode(cp, buf);
                buf[blen] = '\0';
                uint32_t id;
                e = vocab_add(t, buf, &id);
                if (e != OC_OK) {
                    OcUint32Array_free(&word);
                    goto train_fail_words;
                }
                e = OcUint32Array_push(&word, id);
                if (e != OC_OK) {
                    OcUint32Array_free(&word);
                    goto train_fail_words;
                }
            }
        }

        if (word.len > 0) {
            e = OcWordArray_push(&words, word);
            if (e != OC_OK) {
                OcUint32Array_free(&word);
                goto train_fail_words;
            }
        } else {
            OcUint32Array_free(&word);
        }
    }

    size_t merges_done = 0;
    size_t max_merges = t->config.max_merges;
    size_t max_vocab = t->config.max_vocab_size;
    size_t min_freq = t->config.min_frequency;

    while (true) {
        /* Check termination conditions. */
        if (max_merges > 0 && merges_done >= max_merges) {
            break;
        }
        if (max_vocab > 0 && t->vocab.len >= max_vocab) {
            break;
        }

        /* Find the best pair. */
        uint32_t best_left, best_right;
        size_t best_count;
        e = find_best_pair(&words, &best_left, &best_right, &best_count);
        if (e != OC_OK) {
            goto train_fail_words;
        }

        /* Check if the best pair meets the min frequency threshold. */
        if (best_count < min_freq) {
            break;
        }

        /* Build the merged token string. */
        const char *left_str = t->vocab.data[best_left].token;
        const char *right_str = t->vocab.data[best_right].token;
        char *merged_str = build_merged_token(left_str, right_str);
        if (merged_str == NULL) {
            e = OC_ERR_OOM;
            goto train_fail_words;
        }

        /* Add the merged token to the vocab. */
        uint32_t merged_id;
        /* Check if it already exists (shouldn't normally, but be safe). */
        if (tokenmap_get(&t->token_to_id, merged_str, &merged_id)) {
            free(merged_str);
            /* This token already exists — skip this merge to avoid
             * infinite loop. */
            break;
        }

        e = vocab_push(&t->vocab, merged_str, (uint32_t)t->vocab.len);
        free(merged_str);
        if (e != OC_OK) {
            goto train_fail_words;
        }
        merged_id = (uint32_t)(t->vocab.len - 1);
        e = tokenmap_put(&t->token_to_id, t->vocab.data[merged_id].token,
                         merged_id);
        if (e != OC_OK) {
            goto train_fail_words;
        }

        /* Record the merge rule. */
        e = merge_push(&t->merges, best_left, best_right, merged_id);
        if (e != OC_OK) {
            goto train_fail_words;
        }

        /* Apply the merge to all words. */
        apply_merge(&words, best_left, best_right, merged_id);

        merges_done++;
    }

    /* Free the words array (symbols are just ids, no owned strings). */
    for (size_t wi = 0; wi < words.len; wi++) {
        OcUint32Array_free(&words.data[wi]);
    }
    OcWordArray_free(&words);

    t->trained = true;
    return OC_OK;

train_fail_words:
    for (size_t wi = 0; wi < words.len; wi++) {
        OcUint32Array_free(&words.data[wi]);
    }
    OcWordArray_free(&words);
    trainer_reset(t);
    return e;
}


OcBpeTrainer *oc_bpe_trainer_init(OcBpeTrainConfig config)
{
    OcBpeTrainer *t = calloc(1, sizeof(OcBpeTrainer));
    if (t == NULL) {
        return NULL;
    }
    if (config.max_vocab_size == 0) {
        config.max_vocab_size = OC_BPE_DEFAULT_MAX_VOCAB;
    }
    if (config.min_frequency == 0) {
        config.min_frequency = OC_BPE_DEFAULT_MIN_FREQ;
    }
    if (config.max_merges == 0) {
        config.max_merges = OC_BPE_DEFAULT_MAX_MERGES;
    }
    t->config = config;
    t->trained = false;
    return t;
}

OcError oc_bpe_trainer_vocab(const OcBpeTrainer *t,
                             const OcBpeVocabEntry **out_entries,
                             size_t *out_count)
{
    if (t == NULL || out_entries == NULL || out_count == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    *out_entries = t->vocab.data;
    *out_count = t->vocab.len;
    return OC_OK;
}

OcError oc_bpe_trainer_merges(const OcBpeTrainer *t,
                              const OcBpeMerge **out_merges, size_t *out_count)
{
    if (t == NULL || out_merges == NULL || out_count == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    *out_merges = t->merges.data;
    *out_count = t->merges.len;
    return OC_OK;
}

OcError oc_bpe_trainer_save(const OcBpeTrainer *t, const char *path)
{
    if (t == NULL || path == NULL) {
        return OC_ERR_INVALID_ARG;
    }

    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        return OC_ERR_IO;
    }

    /* Write JSON. We manually escape strings. */
    fprintf(fp, "{\n");

    /* Vocab. */
    fprintf(fp, "  \"vocab\": [\n");
    for (size_t i = 0; i < t->vocab.len; i++) {
        fprintf(fp, "    {\"id\": %u, \"token\": \"", (unsigned)t->vocab.data[i].id);
        /* Escape the token string. */
        const char *tok = t->vocab.data[i].token;
        for (const char *p = tok; *p; p++) {
            unsigned char c = (unsigned char)*p;
            if (c == '\\' || c == '"') {
                fprintf(fp, "\\%c", c);
            } else if (c >= 0x20 && c < 0x7F) {
                fputc(c, fp);
            } else {
                fprintf(fp, "\\u%04x", c);
            }
        }
        fprintf(fp, "\"}");
        if (i + 1 < t->vocab.len) {
            fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "  ],\n");

    /* Merges. */
    fprintf(fp, "  \"merges\": [\n");
    for (size_t i = 0; i < t->merges.len; i++) {
        fprintf(fp, "    {\"left\": %u, \"right\": %u, \"merged\": %u}",
                (unsigned)t->merges.data[i].left,
                (unsigned)t->merges.data[i].right,
                (unsigned)t->merges.data[i].merged);
        if (i + 1 < t->merges.len) {
            fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "  ]\n");

    fprintf(fp, "}\n");

    if (fclose(fp) != 0) {
        return OC_ERR_IO;
    }
    return OC_OK;
}

size_t oc_bpe_trainer_vocab_size(const OcBpeTrainer *t)
{
    if (t == NULL) {
        return 0;
    }
    return t->vocab.len;
}

size_t oc_bpe_trainer_merge_count(const OcBpeTrainer *t)
{
    if (t == NULL) {
        return 0;
    }
    return t->merges.len;
}

void oc_bpe_trainer_free(OcBpeTrainer *t)
{
    if (t == NULL) {
        return;
    }
    trainer_reset(t);
    free(t);
}
