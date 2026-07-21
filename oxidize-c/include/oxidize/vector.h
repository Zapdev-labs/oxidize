/*
 * vector.h — OcVector dynamic array of arbitrary element size.
 *
 * Backed by a contiguous `malloc`'d buffer that grows geometrically (factor 2).
 * Supports push/get/len/free. Elements are copied by value (memcpy) at the
 * caller-specified element size. The vector owns its backing buffer; callers
 * own any pointers stored as elements.
 *
 * Port concept: architecture.md §2 (src/core/vector.c).
 */
#ifndef OXIDIZE_VECTOR_H
#define OXIDIZE_VECTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcVector {
    void  *data;          /* backing buffer (malloc'd)            */
    size_t elem_size;     /* bytes per element                    */
    size_t len;           /* current element count                */
    size_t cap;           /* current capacity (in elements)        */
} OcVector;

/* Initialize an OcVector on the given stack/heap struct. `elem_size` must be
 * > 0. Returns OC_OK on success, OC_ERR_INVALID_ARG if elem_size == 0.
 * Initial capacity is 0 (no allocation until first push). */
OcError oc_vector_init(OcVector *v, size_t elem_size);

/* Free the backing buffer (does NOT touch element payloads). Safe on NULL or
 * already-freed vectors (sets data=NULL, len=cap=0). */
void oc_vector_free(OcVector *v);

/* Append `elem` (must point to `v->elem_size` bytes). Grows if needed.
 * Returns OC_OK on success, OC_ERR_OOM on allocation failure, OC_ERR_INVALID_ARG
 * on NULL args. */
OcError oc_vector_push(OcVector *v, const void *elem);

/* Append `n` elements from `elems`. Returns OC_OK or an error. */
OcError oc_vector_push_n(OcVector *v, const void *elems, size_t n);

/* Pointer to element at `i` (mutable; the const applies to the vector
 * structure, not the element storage). Returns NULL if out of range. */
void *oc_vector_get(const OcVector *v, size_t i);

/* Current length. */
static inline size_t oc_vector_len(const OcVector *v)
{
    return v ? v->len : 0;
}

/* Reserve at least `n` elements of capacity. Returns OC_OK or OC_ERR_OOM. */
OcError oc_vector_reserve(OcVector *v, size_t n);

/* Pop the last element into `*out` (if not NULL). Returns true if popped,
 * false if empty. */
bool oc_vector_pop(OcVector *v, void *out);

/* Reset length to 0 WITHOUT freeing backing buffer (capacity unchanged). */
void oc_vector_clear(OcVector *v);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VECTOR_H */
