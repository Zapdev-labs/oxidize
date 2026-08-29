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

/* Insert/replace `key` -> `value`. If `key` already exists, replaces the value */
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

/* Iterator. Caller passes `iter = 0` initially; function writes the next */
bool oc_hashtable_next(OcHashtable *ht, size_t *iter, const char **out_key,
                       void **out_value);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_HASHTABLE_H */
