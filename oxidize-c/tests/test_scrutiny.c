/* test_scrutiny.c — Scrutiny module tests. */
#include <criterion/criterion.h>
#include "oxidize/scrutiny.h"
#include <string.h>

Test(scrutiny, config_init)
{
    OcScrutinyConfig cfg;
    cr_assert_eq(oc_scrutiny_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_rules, 0);
    cr_assert(cfg.block_on_violation);
}

Test(scrutiny, config_init_null)
{
    cr_assert_neq(oc_scrutiny_config_init(NULL), OC_OK);
}

Test(scrutiny, add_banned_phrase)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    cr_assert_eq(oc_scrutiny_add_banned_phrase(&cfg, "bomb"), OC_OK);
    cr_assert_eq(cfg.n_rules, 1);
    cr_assert_str_eq(cfg.rules[0].pattern, "bomb");
    cr_assert(cfg.rules[0].enabled);
}

Test(scrutiny, add_max_length)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    cr_assert_eq(oc_scrutiny_add_max_length(&cfg, 100), OC_OK);
    cr_assert_eq(cfg.rules[0].max_value, 100);
}

Test(scrutiny, add_min_length)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    cr_assert_eq(oc_scrutiny_add_min_length(&cfg, 5), OC_OK);
    cr_assert_eq(cfg.rules[0].min_value, 5);
}

Test(scrutiny, add_pii_rule)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    cr_assert_eq(oc_scrutiny_add_pii_rule(&cfg), OC_OK);
    cr_assert_eq(cfg.rules[0].type, OC_SCRUTINY_RULE_PII);
}

Test(scrutiny, add_toxicity_rule)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    cr_assert_eq(oc_scrutiny_add_toxicity_rule(&cfg), OC_OK);
    cr_assert_eq(cfg.rules[0].type, OC_SCRUTINY_RULE_TOXICITY);
}

Test(scrutiny, check_clean_text)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_banned_phrase(&cfg, "secret");
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check(&cfg, "hello world", &result), OC_OK);
    cr_assert(result.passed);
    cr_assert_eq(result.n_violations, 0);
}

Test(scrutiny, check_banned_phrase)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_banned_phrase(&cfg, "secret");
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check(&cfg, "this is a secret message", &result), OC_OK);
    cr_assert(!result.passed);
    cr_assert_eq(result.n_violations, 1);
    cr_assert_eq(result.violation_type, OC_SCRUTINY_RULE_BANNED_PHRASE);
}

Test(scrutiny, check_max_length)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_max_length(&cfg, 10);
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check(&cfg, "short", &result), OC_OK);
    cr_assert(result.passed);
    cr_assert_eq(oc_scrutiny_check(&cfg, "this is a very long text that exceeds limit", &result), OC_OK);
    cr_assert(!result.passed);
}

Test(scrutiny, check_min_length)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_min_length(&cfg, 10);
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check(&cfg, "short", &result), OC_OK);
    cr_assert(!result.passed);
    cr_assert_eq(oc_scrutiny_check(&cfg, "this is long enough text", &result), OC_OK);
    cr_assert(result.passed);
}

Test(scrutiny, check_pii_ssn)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_pii_rule(&cfg);
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check(&cfg, "SSN: 123-45-6789", &result), OC_OK);
    cr_assert(!result.passed);
    cr_assert_eq(result.violation_type, OC_SCRUTINY_RULE_PII);
}

Test(scrutiny, check_pii_clean)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_pii_rule(&cfg);
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check(&cfg, "no PII here", &result), OC_OK);
    cr_assert(result.passed);
}

Test(scrutiny, check_null)
{
    cr_assert_neq(oc_scrutiny_check(NULL, NULL, NULL), OC_OK);
}

Test(scrutiny, check_tokens_max)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_max_length(&cfg, 5);
    uint32_t tokens[] = {1, 2, 3, 4, 5, 6, 7};
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check_tokens(&cfg, tokens, 7, &result), OC_OK);
    cr_assert(!result.passed);
}

Test(scrutiny, check_tokens_min)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_min_length(&cfg, 5);
    uint32_t tokens[] = {1, 2};
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check_tokens(&cfg, tokens, 2, &result), OC_OK);
    cr_assert(!result.passed);
}

Test(scrutiny, result_passed_null)
{
    cr_assert(oc_scrutiny_result_passed(NULL));
}

Test(scrutiny, rule_type_name)
{
    cr_assert_str_eq(oc_scrutiny_rule_type_name(OC_SCRUTINY_RULE_BANNED_PHRASE), "banned_phrase");
    cr_assert_str_eq(oc_scrutiny_rule_type_name(OC_SCRUTINY_RULE_MAX_LENGTH), "max_length");
    cr_assert_str_eq(oc_scrutiny_rule_type_name(OC_SCRUTINY_RULE_PII), "pii");
    cr_assert_str_eq(oc_scrutiny_rule_type_name(OC_SCRUTINY_RULE_TOXICITY), "toxicity");
}

Test(scrutiny, result_free_null)
{
    oc_scrutiny_result_free(NULL);
}

Test(scrutiny, multiple_violations)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_banned_phrase(&cfg, "secret");
    oc_scrutiny_add_banned_phrase(&cfg, "password");
    cfg.block_on_violation = false;
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check(&cfg, "secret password here", &result), OC_OK);
    cr_assert(!result.passed);
    cr_assert_eq(result.n_violations, 2);
}

Test(scrutiny, case_insensitive)
{
    OcScrutinyConfig cfg;
    oc_scrutiny_config_init(&cfg);
    oc_scrutiny_add_banned_phrase(&cfg, "secret");
    OcScrutinyResult result;
    cr_assert_eq(oc_scrutiny_check(&cfg, "this is a SECRET message", &result), OC_OK);
    cr_assert(!result.passed);
}
