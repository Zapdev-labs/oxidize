/* string.c — string utility helpers. */
#define _POSIX_C_SOURCE 200809L  /* strcasecmp / strndup */

#include "oxidize/util/string.h"

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

char *oc_strdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s);
    char *r = (char *)malloc(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n + 1);
    return r;
}

char *oc_strndup(const char *s, size_t n)
{
    if (!s) return NULL;
    /* Find actual length up to n. */
    size_t actual = strnlen(s, n);
    char *r = (char *)malloc(actual + 1);
    if (!r) return NULL;
    memcpy(r, s, actual);
    r[actual] = '\0';
    return r;
}

size_t oc_strlen(const char *s)
{
    return s ? strlen(s) : 0;
}

int oc_strcmp(const char *a, const char *b)
{
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    return strcmp(a, b);
}

static int ci_cmp(const char *a, const char *b)
{
    while (*a && *b) {
        int ca = tolower((unsigned char)*a);
        int cb = tolower((unsigned char)*b);
        if (ca != cb) return ca - cb;
        a++; b++;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

int oc_strcasecmp(const char *a, const char *b)
{
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    return ci_cmp(a, b);
}

bool oc_starts_with(const char *s, const char *prefix)
{
    if (!s || !prefix) return false;
    while (*prefix) {
        if (*s != *prefix) return false;
        s++; prefix++;
    }
    return true;
}

bool oc_ends_with(const char *s, const char *suffix)
{
    if (!s || !suffix) return false;
    size_t ls = strlen(s);
    size_t lf = strlen(suffix);
    if (lf > ls) return false;
    return memcmp(s + (ls - lf), suffix, lf) == 0;
}

const char *oc_split_once(const char *s, char delim, char **out_left)
{
    if (!s || !out_left) return NULL;
    *out_left = NULL;
    const char *p = s;
    while (*p && *p != delim) p++;
    if (*p == '\0') {
        return s;  /* delim not found */
    }
    size_t left_len = (size_t)(p - s);
    char *left = (char *)malloc(left_len + 1);
    if (!left) return NULL;
    memcpy(left, s, left_len);
    left[left_len] = '\0';
    *out_left = left;
    return p + 1;
}

char *oc_trim(char *s)
{
    if (!s) return NULL;
    /* Trim leading. */
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    /* Trim trailing. */
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) end--;
    *(end + 1) = '\0';
    return s;
}

bool oc_parse_i64(const char *s, long long *out)
{
    if (!s || !*s || !out) return false;
    char *end = NULL;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (errno != 0) return false;
    if (!end || *end != '\0') return false;
    *out = v;
    return true;
}

bool oc_parse_f64(const char *s, double *out)
{
    if (!s || !*s || !out) return false;
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0) return false;
    if (!end || *end != '\0') return false;
    *out = v;
    return true;
}
