/*
 * hashtable.h — OcHashtable open-addressing string-keyed hash map.
 *
 * Uses FNV-1a hashing of NUL-terminated string keys + linear probing.
 * Values are `void *` (caller owns the pointed-to objects). Grows automatically
 * when load factor exceeds 0.7. Does NOT own keys (they are caller-owned; the
 * typical pattern is to dup keys into an OcArena) nor values.
 *
 * Port concept: architecture.md §2 (src/core/hashtable.c).
 */
#ifndef OXIDIZE_HASHTABLE_H
#define OXIDIZE_HASHTABLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcHashtable OcHashtable;

/* Default initial bucket count (MUST be a power of two). */
#define OC_HT_DEFAULT_CAP 16

/* Create a new hashtable with at least `initial_cap` buckets (rounded up to a
 * power of two). If `initial_cap == 0`, uses OC_HT_DEFAULT_CAP. Returns NULL
 * on OOM. */
OcHashtable *oc_hashtable_new(size_t initial_cap);

/* Free the hashtable (does NOT free keys or values). Safe on NULL. */
void oc_hashtable_free(OcHashtable *ht);

/* FNV-1a 64-bit hash of a NUL-terminated string. Exposed for testing. */
uint64_t oc_fnv1a_hash(const char *s);

/* Insert/replace `key` -> `value`. If `key` already exists, replaces the value
 * and (if `prev_value != NULL`) writes the previous value to `*prev_value`.
 * `key` is NOT copied; caller must keep it alive for the hashtable's lifetime
 * (typically by duping into an OcArena). Returns OC_OK on success, OC_ERR_OOM
 * on allocation failure during grow. */
OcError oc_hashtable_put(OcHashtable *ht, const char *key, void *value,
                         void **prev_value);

/* Lookup `key`. If found, writes the value to `*out_value` (if `out_value !=
 * NULL`) and returns true. Returns false if not found. */
bool oc_hashtable_get(const OcHashtable *ht, const char *key, void **out_value);

/* Remove `key`. If found, writes the previous value to `*prev_value` (if not
 * NULL) and returns true. Returns false if not found. */
bool oc_hashtable_remove(OcHashtable *ht, const char *key, void **prev_value);

/* Number of entries currently stored. */
size_t oc_hashtable_size(const OcHashtable *ht);

/* Number of buckets (capacity). */
size_t oc_hashtable_capacity(const OcHashtable *ht);

/* Iterator. Caller passes `iter = 0` initially; function writes the next
 * key/value to `*out_key` / `*out_value` and returns true, or returns false
 * when iteration is complete. The order is unspecified. Mutating the
 * hashtable during iteration is undefined. */
bool oc_hashtable_next(OcHashtable *ht, size_t *iter, const char **out_key,
                       void **out_value);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_HASHTABLE_H */
