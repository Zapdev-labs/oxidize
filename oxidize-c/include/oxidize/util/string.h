/* string.h — string utility helpers. */
#ifndef OXIDIZE_UTIL_STRING_H
#define OXIDIZE_UTIL_STRING_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Duplicate a NUL-terminated string via malloc. Returns NULL on NULL input
 * or OOM. Caller must free(). */
char *oc_strdup(const char *s);

/* Duplicate at most `n` bytes of `s`, NUL-terminating the result. Returns
 * NULL on NULL input or OOM. Caller must free(). */
char *oc_strndup(const char *s, size_t n);

/* Length of a NUL-terminated string. Returns 0 on NULL. */
size_t oc_strlen(const char *s);

/* Compare two NUL-terminated strings. NULL is treated as less than any
 * non-NULL string. Returns <0, 0, or >0 like strcmp. */
int oc_strcmp(const char *a, const char *b);

/* Case-insensitive comparison. NULL handling same as oc_strcmp. */
int oc_strcasecmp(const char *a, const char *b);

/* Check if `s` starts with `prefix`. NULL `s` or `prefix` returns false. */
bool oc_starts_with(const char *s, const char *prefix);

/* Check if `s` ends with `suffix`. NULL `s` or `suffix` returns false. */
bool oc_ends_with(const char *s, const char *suffix);

/* Split `s` on the first occurrence of `delim`. */
const char *oc_split_once(const char *s, char delim, char **out_left);

/* Trim leading + trailing ASCII whitespace in place. Returns a pointer into
 * `s` (not a new allocation). The NUL terminator is moved to the new end. */
char *oc_trim(char *s);

/* Parse a base-10 integer. Returns true on success, false on parse error or
 * overflow. On success writes the value to `*out`. */
bool oc_parse_i64(const char *s, long long *out);

/* Parse a base-10 float (double). Returns true on success. */
bool oc_parse_f64(const char *s, double *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_UTIL_STRING_H */
