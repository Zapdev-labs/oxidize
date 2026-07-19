/* hashtable.c — OcHashtable open-addressing string-keyed map (FNV-1a).
 *
 * Implementation:
 * - Buckets are an array of {key, value, hash} slots. A NULL key marks an
 *   empty slot. A deleted slot uses a sentinel key (HT_TOMBSTONE) so probe
 *   sequences continue past deletions.
 * - Hash: FNV-1a 64-bit, masked by (cap - 1) (cap is always a power of two).
 * - Collision resolution: linear probing.
 * - Grow: when load (size + tombstones) / cap > 0.7, double the bucket count
 *   and reinsert live entries (tombstones dropped).
 *
 * The hashtable does NOT own keys or values. The typical pattern is to dup
 * keys into an OcArena at insertion time.
 */
#include "oxidize/hashtable.h"

#include <stdlib.h>
#include <string.h>

/* Tombstone marker for deleted slots. We use a distinct non-NULL pointer that
 * cannot collide with a real user key (users pass real C strings, never this
 * sentinel). */
static char g_tombstone_buf[1] = {0};
#define HT_TOMBSTONE ((const char *)g_tombstone_buf)

typedef struct {
    const char *key;     /* NULL = empty; HT_TOMBSTONE = deleted       */
    void       *value;
    uint64_t    hash;
} OcHtSlot;

struct OcHashtable {
    OcHtSlot *slots;
    size_t    cap;        /* always a power of two                       */
    size_t    size;       /* live entries                                 */
    size_t    tombstones; /* deleted entries not yet reclaimed            */
};

uint64_t oc_fnv1a_hash(const char *s)
{
    if (!s) return 0;
    uint64_t h = 0xcbf29ce484222325ULL;  /* FNV-1a 64-bit offset basis */
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        h ^= (uint64_t)*p;
        h *= 0x100000001b3ULL;            /* FNV-1a 64-bit prime         */
    }
    return h;
}

static size_t round_up_pow2(size_t n)
{
    size_t r = 1;
    while (r < n) {
        r <<= 1;
        if (r == 0) return n;  /* overflow guard */
    }
    return r ? r : 1;
}

OcHashtable *oc_hashtable_new(size_t initial_cap)
{
    if (initial_cap == 0) initial_cap = OC_HT_DEFAULT_CAP;
    initial_cap = round_up_pow2(initial_cap);
    if (initial_cap < 4) initial_cap = 4;

    OcHashtable *ht = (OcHashtable *)malloc(sizeof(OcHashtable));
    if (!ht) return NULL;
    ht->slots = (OcHtSlot *)calloc(initial_cap, sizeof(OcHtSlot));
    if (!ht->slots) { free(ht); return NULL; }
    ht->cap        = initial_cap;
    ht->size       = 0;
    ht->tombstones = 0;
    return ht;
}

void oc_hashtable_free(OcHashtable *ht)
{
    if (!ht) return;
    free(ht->slots);
    free(ht);
}

static size_t probe_index(uint64_t hash, size_t cap, size_t i)
{
    return (size_t)((hash + (uint64_t)i) & (cap - 1));
}

static OcHtSlot *find_slot(OcHashtable *ht, const char *key, uint64_t hash,
                           bool for_insert)
{
    /* for_insert: returns the slot to write into (may be empty or tombstone).
     * for lookup: returns the slot if found, else NULL. */
    size_t first_tombstone = (size_t)-1;
    for (size_t i = 0; i < ht->cap; i++) {
        size_t idx = probe_index(hash, ht->cap, i);
        OcHtSlot *s = &ht->slots[idx];
        if (s->key == NULL) {
            /* Empty: not present. For insert, return this (or an earlier tomb). */
            if (for_insert) {
                if (first_tombstone != (size_t)-1)
                    return &ht->slots[first_tombstone];
                return s;
            }
            return NULL;
        }
        if (s->key == HT_TOMBSTONE) {
            if (for_insert && first_tombstone == (size_t)-1)
                first_tombstone = idx;
            continue;
        }
        if (s->hash == hash && strcmp(s->key, key) == 0) {
            return s;
        }
    }
    if (for_insert && first_tombstone != (size_t)-1)
        return &ht->slots[first_tombstone];
    return NULL;
}

static OcError grow_if_needed(OcHashtable *ht)
{
    size_t load = ht->size + ht->tombstones;
    if (load * 10 < ht->cap * 7) return OC_OK;  /* load factor < 0.7 */

    size_t new_cap = ht->cap * 2;
    if (new_cap < ht->cap) return OC_ERR_OOM;  /* overflow */

    OcHtSlot *old = ht->slots;
    size_t old_cap = ht->cap;

    OcHtSlot *nw = (OcHtSlot *)calloc(new_cap, sizeof(OcHtSlot));
    if (!nw) return OC_ERR_OOM;

    ht->slots = nw;
    ht->cap = new_cap;
    ht->size = 0;
    ht->tombstones = 0;

    for (size_t i = 0; i < old_cap; i++) {
        if (old[i].key != NULL && old[i].key != HT_TOMBSTONE) {
            OcHtSlot *s = find_slot(ht, old[i].key, old[i].hash, true);
            /* new array is empty so find_slot always succeeds */
            s->key   = old[i].key;
            s->value = old[i].value;
            s->hash  = old[i].hash;
            ht->size++;
        }
    }
    free(old);
    return OC_OK;
}

OcError oc_hashtable_put(OcHashtable *ht, const char *key, void *value,
                         void **prev_value)
{
    if (!ht || !key) return OC_ERR_INVALID_ARG;
    if (prev_value) *prev_value = NULL;

    OcError e = grow_if_needed(ht);
    if (e != OC_OK) return e;

    uint64_t hash = oc_fnv1a_hash(key);
    OcHtSlot *s = find_slot(ht, key, hash, true);
    if (!s) {
        /* Table full of tombstones (shouldn't happen due to grow). Force a grow. */
        e = grow_if_needed(ht);
        if (e != OC_OK) return e;
        s = find_slot(ht, key, hash, true);
        if (!s) return OC_ERR_INTERNAL;
    }

    if (s->key == NULL || s->key == HT_TOMBSTONE) {
        /* New insertion. */
        if (s->key == HT_TOMBSTONE) ht->tombstones--;
        s->key   = key;
        s->value = value;
        s->hash  = hash;
        ht->size++;
        return OC_OK;
    }
    /* Existing entry: replace value. */
    if (prev_value) *prev_value = s->value;
    s->value = value;
    return OC_OK;
}

bool oc_hashtable_get(const OcHashtable *ht, const char *key, void **out_value)
{
    if (!ht || !key) return false;
    uint64_t hash = oc_fnv1a_hash(key);
    /* const cast: find_slot only reads/mutates on insert path; here for_insert
     * is false so no mutation occurs. */
    OcHtSlot *s = find_slot((OcHashtable *)ht, key, hash, false);
    if (!s) return false;
    if (out_value) *out_value = s->value;
    return true;
}

bool oc_hashtable_remove(OcHashtable *ht, const char *key, void **prev_value)
{
    if (!ht || !key) return false;
    uint64_t hash = oc_fnv1a_hash(key);
    OcHtSlot *s = find_slot(ht, key, hash, false);
    if (!s) return false;
    if (prev_value) *prev_value = s->value;
    s->key = HT_TOMBSTONE;
    s->value = NULL;
    s->hash = 0;
    ht->size--;
    ht->tombstones++;
    return true;
}

size_t oc_hashtable_size(const OcHashtable *ht) { return ht ? ht->size : 0; }
size_t oc_hashtable_capacity(const OcHashtable *ht) { return ht ? ht->cap : 0; }

bool oc_hashtable_next(OcHashtable *ht, size_t *iter, const char **out_key,
                       void **out_value)
{
    if (!ht || !iter) return false;
    for (; *iter < ht->cap; (*iter)++) {
        OcHtSlot *s = &ht->slots[*iter];
        if (s->key != NULL && s->key != HT_TOMBSTONE) {
            if (out_key)   *out_key   = s->key;
            if (out_value) *out_value = s->value;
            (*iter)++;
            return true;
        }
    }
    return false;
}
