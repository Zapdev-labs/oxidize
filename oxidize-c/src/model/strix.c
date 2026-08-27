/*
 * strix.c — Structured output constrained generation implementation.
 */
#include "oxidize/strix.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

OcError oc_strix_config_init(OcStrixConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->mode = OC_STRIX_MODE_NONE;
    cfg->strict = false;
    cfg->max_depth = 32;
    return OC_OK;
}

OcError oc_strix_config_json(OcStrixConfig *cfg, const char *schema, bool strict)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->mode = OC_STRIX_MODE_JSON;
    cfg->strict = strict;
    cfg->max_depth = 32;
    if (schema) {
        size_t n = strlen(schema);
        if (n >= OC_STRIX_MAX_SCHEMA) n = OC_STRIX_MAX_SCHEMA - 1;
        memcpy(cfg->schema, schema, n);
        cfg->schema[n] = '\0';
    }
    return OC_OK;
}

OcError oc_strix_config_regex(OcStrixConfig *cfg, const char *pattern)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->mode = OC_STRIX_MODE_REGEX;
    cfg->strict = true;
    cfg->max_depth = 0;
    if (pattern) {
        size_t n = strlen(pattern);
        if (n >= OC_STRIX_MAX_SCHEMA) n = OC_STRIX_MAX_SCHEMA - 1;
        memcpy(cfg->schema, pattern, n);
        cfg->schema[n] = '\0';
    }
    return OC_OK;
}

OcError oc_strix_state_init(OcStrixState *state, const OcStrixConfig *cfg,
                           char *output, size_t output_cap)
{
    if (!state || !output || output_cap == 0) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    if (cfg) state->config = *cfg;
    else oc_strix_config_init(&state->config);
    state->output = output;
    state->output_cap = output_cap;
    state->output_len = 0;
    state->output[0] = '\0';
    state->depth = 0;
    state->expecting_value = true;
    return OC_OK;
}

OcError oc_strix_accept_char(OcStrixState *state, char c)
{
    if (!state) return OC_ERR_INVALID_ARG;
    if (state->output_len + 1 >= state->output_cap) return OC_ERR_OOM;

    state->output[state->output_len++] = c;
    state->output[state->output_len] = '\0';

    /* Track JSON state. */
    if (state->config.mode == OC_STRIX_MODE_JSON) {
        if (c == '{') {
            state->depth++;
            state->current_type = OC_STRIX_JSON_OBJECT;
            state->expecting_key = true;
            state->expecting_value = false;
        } else if (c == '}') {
            state->depth--;
            if (state->depth <= 0) {
                state->expecting_value = false;
            }
        } else if (c == '[') {
            state->depth++;
            state->current_type = OC_STRIX_JSON_ARRAY;
        } else if (c == ']') {
            state->depth--;
        } else if (c == '"') {
            state->in_string = !state->in_string;
            if (state->in_string) {
                state->current_type = OC_STRIX_JSON_STRING;
            }
        } else if (c == ':' && !state->in_string) {
            state->expecting_colon = false;
            state->expecting_value = true;
        } else if (c == ',' && !state->in_string) {
            if (state->current_type == OC_STRIX_JSON_OBJECT) {
                state->expecting_key = true;
            }
            state->expecting_comma = false;
        }
    }

    return OC_OK;
}

OcError oc_strix_accept_token(OcStrixState *state, uint32_t token_id,
                             const char *token_text)
{
    if (!state || !token_text) return OC_ERR_INVALID_ARG;
    (void)token_id;
    for (const char *p = token_text; *p; p++) {
        OcError e = oc_strix_accept_char(state, *p);
        if (e != OC_OK) return e;
    }
    return OC_OK;
}

bool oc_strix_is_valid_token(const OcStrixState *state, uint32_t token_id,
                            const char *token_text)
{
    if (!state || !token_text) return false;
    (void)token_id;
    if (state->config.mode == OC_STRIX_MODE_NONE) return true;

    /* For JSON mode, check if accepting this token would produce valid JSON.
     * This is a simplified check. */
    if (state->config.mode == OC_STRIX_MODE_JSON) {
        /* If complete (depth == 0 and output not empty), reject further tokens. */
        if (oc_strix_is_complete(state)) return false;
        /* If expecting a key, only allow strings or }. */
        if (state->expecting_key && !state->in_string) {
            if (token_text[0] == '"' || token_text[0] == '}') return true;
            return false;
        }
        return true;
    }

    return true;
}

OcError oc_strix_filter_logits(const OcStrixState *state, float *logits,
                              size_t vocab_size, const char **token_texts,
                              size_t n_tokens)
{
    if (!state || !logits || !token_texts) return OC_ERR_INVALID_ARG;
    if (state->config.mode == OC_STRIX_MODE_NONE) return OC_OK;

    size_t n = n_tokens < vocab_size ? n_tokens : vocab_size;
    for (size_t i = 0; i < n; i++) {
        if (!token_texts[i]) continue;
        if (!oc_strix_is_valid_token(state, (uint32_t)i, token_texts[i])) {
            logits[i] = -INFINITY;
        }
    }
    return OC_OK;
}

OcError oc_strix_finalize(OcStrixState *state)
{
    if (!state) return OC_ERR_INVALID_ARG;
    state->expecting_value = false;
    return OC_OK;
}

bool oc_strix_is_complete(const OcStrixState *state)
{
    if (!state) return false;
    if (state->config.mode == OC_STRIX_MODE_NONE) return true;
    if (state->output_len == 0) return false;
    if (state->config.mode == OC_STRIX_MODE_JSON) {
        return state->depth == 0 && !state->in_string && state->output_len > 0;
    }
    return true;
}

const char *oc_strix_mode_name(OcStrixMode mode)
{
    switch (mode) {
    case OC_STRIX_MODE_NONE:    return "none";
    case OC_STRIX_MODE_JSON:    return "json";
    case OC_STRIX_MODE_REGEX:   return "regex";
    case OC_STRIX_MODE_GRAMMAR: return "grammar";
    default: return "unknown";
    }
}

const char *oc_strix_json_type_name(OcStrixJsonType type)
{
    switch (type) {
    case OC_STRIX_JSON_OBJECT: return "object";
    case OC_STRIX_JSON_ARRAY:  return "array";
    case OC_STRIX_JSON_STRING: return "string";
    case OC_STRIX_JSON_NUMBER: return "number";
    case OC_STRIX_JSON_BOOL:   return "bool";
    case OC_STRIX_JSON_NULL:   return "null";
    default: return "unknown";
    }
}

void oc_strix_free(OcStrixState *state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
}
