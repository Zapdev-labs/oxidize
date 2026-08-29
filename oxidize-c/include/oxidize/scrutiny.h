#ifndef OXIDIZE_SCRUTINY_H
#define OXIDIZE_SCRUTINY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_SCRUTINY_MAX_RULES 64
#define OC_SCRUTINY_MAX_PATTERNS 32
#define OC_SCRUTINY_MAX_PATTERN_LEN 256

typedef enum {
    OC_SCRUTINY_RULE_BANNED_PHRASE = 0,
    OC_SCRUTINY_RULE_MAX_LENGTH    = 1,
    OC_SCRUTINY_RULE_MIN_LENGTH   = 2,
    OC_SCRUTINY_RULE_REGEX         = 3,
    OC_SCRUTINY_RULE_PII           = 4,
    OC_SCRUTINY_RULE_TOXICITY      = 5,
} OcScrutinyRuleType;

typedef struct {
    OcScrutinyRuleType type;
    char pattern[OC_SCRUTINY_MAX_PATTERN_LEN];
    uint32_t max_value;
    uint32_t min_value;
    bool enabled;
} OcScrutinyRule;

typedef struct {
    OcScrutinyRule rules[OC_SCRUTINY_MAX_RULES];
    uint32_t n_rules;
    bool block_on_violation;
} OcScrutinyConfig;

typedef struct {
    bool passed;
    uint32_t n_violations;
    char first_violation[OC_SCRUTINY_MAX_PATTERN_LEN];
    OcScrutinyRuleType violation_type;
} OcScrutinyResult;

OcError oc_scrutiny_config_init(OcScrutinyConfig *cfg);
OcError oc_scrutiny_add_banned_phrase(OcScrutinyConfig *cfg, const char *phrase);
OcError oc_scrutiny_add_max_length(OcScrutinyConfig *cfg, uint32_t max_len);
OcError oc_scrutiny_add_min_length(OcScrutinyConfig *cfg, uint32_t min_len);
OcError oc_scrutiny_add_pii_rule(OcScrutinyConfig *cfg);
OcError oc_scrutiny_add_toxicity_rule(OcScrutinyConfig *cfg);
OcError oc_scrutiny_check(const OcScrutinyConfig *cfg, const char *text,
                          OcScrutinyResult *out);
OcError oc_scrutiny_check_tokens(const OcScrutinyConfig *cfg,
                                const uint32_t *tokens, size_t n,
                                OcScrutinyResult *out);
bool oc_scrutiny_result_passed(const OcScrutinyResult *result);
const char *oc_scrutiny_rule_type_name(OcScrutinyRuleType type);
void oc_scrutiny_result_free(OcScrutinyResult *result);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SCRUTINY_H */
