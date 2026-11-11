/*
 * checked.h — overflow-checked size_t arithmetic for allocation sizing.
 *
 * Each function returns false and leaves `*out` untouched on overflow.
 */
#ifndef OXIDIZE_UTIL_CHECKED_H
#define OXIDIZE_UTIL_CHECKED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

static inline bool oc_size_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static inline bool oc_size_add(size_t a, size_t b, size_t *out)
{
    if (a > SIZE_MAX - b) return false;
    *out = a + b;
    return true;
}

#endif /* OXIDIZE_UTIL_CHECKED_H */
