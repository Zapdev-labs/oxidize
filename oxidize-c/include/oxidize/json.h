#ifndef OXIDIZE_JSON_H
#define OXIDIZE_JSON_H
#include "oxidize/arena.h"
#include "oxidize/error.h"
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { OC_JSON_NULL, OC_JSON_BOOL, OC_JSON_NUMBER, OC_JSON_STRING, OC_JSON_ARRAY, OC_JSON_OBJECT } OcJsonType;
typedef struct OcJson {
    OcJsonType type;
    char *key, *string;
    double number;
    bool boolean;
    size_t count;
    struct OcJson *child, *next;
} OcJson;
/* Strict JSON, UTF-8 strings, unique object keys, maximum depth 128.
 * All nodes and strings belong to arena. Rejects embedded NUL strings. */
OcError oc_json_parse(const char *data, size_t len, OcArena *arena, OcJson **out);
const OcJson *oc_json_get(const OcJson *object, const char *key);
bool oc_json_u32(const OcJson *value, uint32_t *out);
#ifdef __cplusplus
}
#endif
#endif
