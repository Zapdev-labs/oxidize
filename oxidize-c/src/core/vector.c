/* vector.c — OcVector dynamic array. */
#include "oxidize/vector.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

OcError oc_vector_init(OcVector *v, size_t elem_size)
{
    if (!v) return OC_ERR_INVALID_ARG;
    if (elem_size == 0) return OC_ERR_INVALID_ARG;
    v->data      = NULL;
    v->elem_size = elem_size;
    v->len       = 0;
    v->cap       = 0;
    return OC_OK;
}

void oc_vector_free(OcVector *v)
{
    if (!v) return;
    free(v->data);
    v->data = NULL;
    v->len  = 0;
    v->cap  = 0;
}

OcError oc_vector_reserve(OcVector *v, size_t n)
{
    if (!v) return OC_ERR_INVALID_ARG;
    if (n <= v->cap) return OC_OK;
    /* Geometric growth: at least double, but at least n. */
    size_t new_cap = v->cap ? v->cap : 4;
    while (new_cap < n) {
        if (new_cap > (SIZE_MAX / 2)) { new_cap = n; break; }
        new_cap *= 2;
    }
    if (new_cap > SIZE_MAX / v->elem_size) return OC_ERR_OOM; /* byte size would wrap */
    void *nd = realloc(v->data, new_cap * v->elem_size);
    if (!nd) return OC_ERR_OOM;
    v->data = nd;
    v->cap  = new_cap;
    return OC_OK;
}

OcError oc_vector_push(OcVector *v, const void *elem)
{
    if (!v || !elem) return OC_ERR_INVALID_ARG;
    if (v->len == v->cap) {
        /* `elem` may alias an existing element; realloc would invalidate it.
         * Remember its offset and rebase after growth. */
        size_t elem_off = SIZE_MAX;
        const uint8_t *e8 = (const uint8_t *)elem;
        if (v->data && e8 >= (uint8_t *)v->data &&
            e8 < (uint8_t *)v->data + v->cap * v->elem_size)
            elem_off = (size_t)(e8 - (uint8_t *)v->data);
        OcError e = oc_vector_reserve(v, v->len + 1);
        if (e != OC_OK) return e;
        if (elem_off != SIZE_MAX) elem = (uint8_t *)v->data + elem_off;
    }
    memcpy((uint8_t *)v->data + v->len * v->elem_size, elem, v->elem_size);
    v->len++;
    return OC_OK;
}

OcError oc_vector_push_n(OcVector *v, const void *elems, size_t n)
{
    if (!v) return OC_ERR_INVALID_ARG;
    if (n == 0) return OC_OK;
    if (!elems) return OC_ERR_INVALID_ARG;
    if (n > SIZE_MAX - v->len) return OC_ERR_OOM; /* len + n would wrap */
    if (v->len + n > v->cap) {
        /* `elems` may alias the vector's own buffer; rebase across realloc. */
        size_t elems_off = SIZE_MAX;
        const uint8_t *e8 = (const uint8_t *)elems;
        if (v->data && e8 >= (uint8_t *)v->data &&
            e8 < (uint8_t *)v->data + v->cap * v->elem_size)
            elems_off = (size_t)(e8 - (uint8_t *)v->data);
        OcError e = oc_vector_reserve(v, v->len + n);
        if (e != OC_OK) return e;
        if (elems_off != SIZE_MAX) elems = (uint8_t *)v->data + elems_off;
    }
    memcpy((uint8_t *)v->data + v->len * v->elem_size, elems, n * v->elem_size);
    v->len += n;
    return OC_OK;
}

const void *oc_vector_get(const OcVector *v, size_t i)
{
    if (!v || i >= v->len) return NULL;
    return (uint8_t *)v->data + i * v->elem_size;
}

void *oc_vector_get_mut(OcVector *v, size_t i)
{
    if (!v || i >= v->len) return NULL;
    return (uint8_t *)v->data + i * v->elem_size;
}

bool oc_vector_pop(OcVector *v, void *out)
{
    if (!v || v->len == 0) return false;
    v->len--;
    if (out) memcpy(out, (uint8_t *)v->data + v->len * v->elem_size, v->elem_size);
    return true;
}

void oc_vector_clear(OcVector *v)
{
    if (!v) return;
    v->len = 0;
}
