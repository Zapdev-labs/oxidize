/*
 * scrutiny.c — Output scrutiny implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/scrutiny.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) { if (dst && cap > 0) dst[0] = '\0'; return; }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static bool contains_phrase(const char *text, const char *phrase)
{
    if (!text || !phrase) return false;
    size_t plen = strlen(phrase);
    if (plen == 0) return false;
    const char *p = text;
    while (*p) {
        if (strncasecmp(p, phrase, plen) == 0) return true;
        p++;
    }
    return false;
}

static bool looks_like_pii(const char *text)
{
    if (!text) return false;
    /* Check for common PII patterns: SSN-like (XXX-XX-XXXX), email, phone. */
    const char *p = text;
    while (*p) {
        /* SSN pattern: 3 digits, dash, 2 digits, dash, 4 digits. */
        if (p[0] >= '0' && p[0] <= '9' && p[1] >= '0' && p[1] <= '9' &&
            p[2] >= '0' && p[2] <= '9' && p[3] == '-' &&
            p[4] >= '0' && p[4] <= '9' && p[5] >= '0' && p[5] <= '9' &&
            p[6] == '-' &&
            p[7] >= '0' && p[7] <= '9' && p[8] >= '0' && p[8] <= '9' &&
            p[9] >= '0' && p[9] <= '9' && p[10] >= '0' && p[10] <= '9')
            return true;
        p++;
    }
    return false;
}

OcError oc_scrutiny_config_init(OcScrutinyConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->block_on_violation = true;
    return OC_OK;
}

OcError oc_scrutiny_add_banned_phrase(OcScrutinyConfig *cfg, const char *phrase)
{
    if (!cfg || !phrase) return OC_ERR_INVALID_ARG;
    if (cfg->n_rules >= OC_SCRUTINY_MAX_RULES) return OC_ERR_OOM;
    OcScrutinyRule *r = &cfg->rules[cfg->n_rules++];
    memset(r, 0, sizeof(*r));
    r->type = OC_SCRUTINY_RULE_BANNED_PHRASE;
    copy_str(r->pattern, sizeof(r->pattern), phrase);
    r->enabled = true;
    return OC_OK;
}

OcError oc_scrutiny_add_max_length(OcScrutinyConfig *cfg, uint32_t max_len)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    if (cfg->n_rules >= OC_SCRUTINY_MAX_RULES) return OC_ERR_OOM;
    OcScrutinyRule *r = &cfg->rules[cfg->n_rules++];
    memset(r, 0, sizeof(*r));
    r->type = OC_SCRUTINY_RULE_MAX_LENGTH;
    r->max_value = max_len;
    r->enabled = true;
    return OC_OK;
}

OcError oc_scrutiny_add_min_length(OcScrutinyConfig *cfg, uint32_t min_len)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    if (cfg->n_rules >= OC_SCRUTINY_MAX_RULES) return OC_ERR_OOM;
    OcScrutinyRule *r = &cfg->rules[cfg->n_rules++];
    memset(r, 0, sizeof(*r));
    r->type = OC_SCRUTINY_RULE_MIN_LENGTH;
    r->min_value = min_len;
    r->enabled = true;
    return OC_OK;
}

OcError oc_scrutiny_add_pii_rule(OcScrutinyConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    if (cfg->n_rules >= OC_SCRUTINY_MAX_RULES) return OC_ERR_OOM;
    OcScrutinyRule *r = &cfg->rules[cfg->n_rules++];
    memset(r, 0, sizeof(*r));
    r->type = OC_SCRUTINY_RULE_PII;
    r->enabled = true;
    return OC_OK;
}

OcError oc_scrutiny_add_toxicity_rule(OcScrutinyConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    if (cfg->n_rules >= OC_SCRUTINY_MAX_RULES) return OC_ERR_OOM;
    OcScrutinyRule *r = &cfg->rules[cfg->n_rules++];
    memset(r, 0, sizeof(*r));
    r->type = OC_SCRUTINY_RULE_TOXICITY;
    r->enabled = true;
    return OC_OK;
}

OcError oc_scrutiny_check(const OcScrutinyConfig *cfg, const char *text,
                          OcScrutinyResult *out)
{
    if (!cfg || !text || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->passed = true;

    size_t text_len = strlen(text);

    for (uint32_t i = 0; i < cfg->n_rules; i++) {
        if (!cfg->rules[i].enabled) continue;
        const OcScrutinyRule *r = &cfg->rules[i];
        bool violated = false;

        switch (r->type) {
        case OC_SCRUTINY_RULE_BANNED_PHRASE:
            if (contains_phrase(text, r->pattern)) violated = true;
            break;
        case OC_SCRUTINY_RULE_MAX_LENGTH:
            if (text_len > r->max_value) violated = true;
            break;
        case OC_SCRUTINY_RULE_MIN_LENGTH:
            if (text_len < r->min_value) violated = true;
            break;
        case OC_SCRUTINY_RULE_PII:
            if (looks_like_pii(text)) violated = true;
            break;
        case OC_SCRUTINY_RULE_TOXICITY:
            /* Stub: no real toxicity detection. */
            break;
        default:
            break;
        }

        if (violated) {
            out->passed = false;
            out->n_violations++;
            if (out->n_violations == 1) {
                out->violation_type = r->type;
                copy_str(out->first_violation, sizeof(out->first_violation),
                         r->type == OC_SCRUTINY_RULE_BANNED_PHRASE ? r->pattern : "");
            }
            if (cfg->block_on_violation) return OC_OK;
        }
    }
    return OC_OK;
}

OcError oc_scrutiny_check_tokens(const OcScrutinyConfig *cfg,
                                const uint32_t *tokens, size_t n,
                                OcScrutinyResult *out)
{
    if (!cfg || !tokens || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->passed = true;

    for (uint32_t i = 0; i < cfg->n_rules; i++) {
        if (!cfg->rules[i].enabled) continue;
        const OcScrutinyRule *r = &cfg->rules[i];
        bool violated = false;

        switch (r->type) {
        case OC_SCRUTINY_RULE_MAX_LENGTH:
            if (n > r->max_value) violated = true;
            break;
        case OC_SCRUTINY_RULE_MIN_LENGTH:
            if (n < r->min_value) violated = true;
            break;
        default:
            break;
        }

        if (violated) {
            out->passed = false;
            out->n_violations++;
            if (out->n_violations == 1)
                out->violation_type = r->type;
        }
    }
    return OC_OK;
}

bool oc_scrutiny_result_passed(const OcScrutinyResult *result)
{
    return result ? result->passed : true;
}

const char *oc_scrutiny_rule_type_name(OcScrutinyRuleType type)
{
    switch (type) {
    case OC_SCRUTINY_RULE_BANNED_PHRASE: return "banned_phrase";
    case OC_SCRUTINY_RULE_MAX_LENGTH:   return "max_length";
    case OC_SCRUTINY_RULE_MIN_LENGTH:   return "min_length";
    case OC_SCRUTINY_RULE_REGEX:        return "regex";
    case OC_SCRUTINY_RULE_PII:          return "pii";
    case OC_SCRUTINY_RULE_TOXICITY:     return "toxicity";
    default: return "unknown";
    }
}

void oc_scrutiny_result_free(OcScrutinyResult *result)
{
    if (!result) return;
    memset(result, 0, sizeof(*result));
}
