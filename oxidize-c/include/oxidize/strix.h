#ifndef OXIDIZE_STRIX_H
#define OXIDIZE_STRIX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_STRIX_MAX_SCHEMA 4096

typedef enum {
    OC_STRIX_MODE_NONE = 0,
    OC_STRIX_MODE_JSON = 1,
    OC_STRIX_MODE_REGEX = 2,
    OC_STRIX_MODE_GRAMMAR = 3,
} OcStrixMode;

typedef enum {
    OC_STRIX_JSON_OBJECT = 0,
    OC_STRIX_JSON_ARRAY = 1,
    OC_STRIX_JSON_STRING = 2,
    OC_STRIX_JSON_NUMBER = 3,
    OC_STRIX_JSON_BOOL = 4,
    OC_STRIX_JSON_NULL = 5,
} OcStrixJsonType;

typedef struct {
    OcStrixMode mode;
    char schema[OC_STRIX_MAX_SCHEMA];
    bool strict;
    uint32_t max_depth;
} OcStrixConfig;

typedef struct {
    OcStrixConfig config;
    bool in_string;
    int32_t depth;
    OcStrixJsonType current_type;
    bool expecting_key;
    bool expecting_colon;
    bool expecting_value;
    bool expecting_comma;
    char *output;
    size_t output_len;
    size_t output_cap;
} OcStrixState;

OcError oc_strix_config_init(OcStrixConfig *cfg);
OcError oc_strix_config_json(OcStrixConfig *cfg, const char *schema, bool strict);
OcError oc_strix_config_regex(OcStrixConfig *cfg, const char *pattern);
OcError oc_strix_state_init(OcStrixState *state, const OcStrixConfig *cfg,
                           char *output, size_t output_cap);
OcError oc_strix_accept_token(OcStrixState *state, uint32_t token_id,
                             const char *token_text);
OcError oc_strix_accept_char(OcStrixState *state, char c);
bool oc_strix_is_valid_token(const OcStrixState *state, uint32_t token_id,
                            const char *token_text);
OcError oc_strix_filter_logits(const OcStrixState *state, float *logits,
                              size_t vocab_size, const char **token_texts,
                              size_t n_tokens);
OcError oc_strix_finalize(OcStrixState *state);
bool oc_strix_is_complete(const OcStrixState *state);
const char *oc_strix_mode_name(OcStrixMode mode);
const char *oc_strix_json_type_name(OcStrixJsonType type);
void oc_strix_free(OcStrixState *state);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_STRIX_H */
