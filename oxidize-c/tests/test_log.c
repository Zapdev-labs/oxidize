/* test_log.c — oc_log leveled logging + OX_LOG_LEVEL tests. */
#define _POSIX_C_SOURCE 200809L  /* setenv/unsetenv */

#include <criterion/criterion.h>
#include "oxidize/log.h"

#include <stdlib.h>
#include <string.h>

Test(log, level_name_round_trip)
{
    cr_assert_str_eq(oc_log_level_name(OC_LOG_DEBUG), "DEBUG", "");
    cr_assert_str_eq(oc_log_level_name(OC_LOG_INFO),  "INFO",  "");
    cr_assert_str_eq(oc_log_level_name(OC_LOG_WARN),  "WARN",  "");
    cr_assert_str_eq(oc_log_level_name(OC_LOG_ERROR), "ERROR", "");

    cr_assert_eq(oc_log_level_from_str("DEBUG"), OC_LOG_DEBUG, "");
    cr_assert_eq(oc_log_level_from_str("info"),  OC_LOG_INFO,  "");
    cr_assert_eq(oc_log_level_from_str("Warn"), OC_LOG_WARN,  "");
    cr_assert_eq(oc_log_level_from_str("WARNING"), OC_LOG_WARN, "");
    cr_assert_eq(oc_log_level_from_str("error"), OC_LOG_ERROR, "");
    cr_assert_eq(oc_log_level_from_str("garbage"), OC_LOG_INFO, "unknown -> INFO");
}

Test(log, set_level_filters)
{
    oc_log_set_level(OC_LOG_ERROR);
    cr_assert_eq(oc_log_get_level(), OC_LOG_ERROR, "level should be ERROR");
    oc_log_set_level(OC_LOG_DEBUG);
    cr_assert_eq(oc_log_get_level(), OC_LOG_DEBUG, "level should be DEBUG");
    /* Restore default. */
    oc_log_set_level(OC_LOG_INFO);
}

Test(log, env_filter_debug)
{
    /* Criterion forks each test into its own process, so log.c's statics
     * (g_level_set / g_env_checked) are fresh here: no set_level has run,
     * and the env is read on first init. */
    setenv("OX_LOG_LEVEL", "DEBUG", 1);
    oc_log_init_from_env();
    cr_assert_eq(oc_log_get_level(), OC_LOG_DEBUG, "env DEBUG applied");
    /* Explicit set_level overrides the env-derived level. */
    oc_log_set_level(OC_LOG_WARN);
    cr_assert_eq(oc_log_get_level(), OC_LOG_WARN, "set_level overrides env");
    unsetenv("OX_LOG_LEVEL");
}

Test(log, emits_at_or_above)
{
    /* Just verify no crash; output goes to stderr. */
    oc_log_set_level(OC_LOG_WARN);
    oc_log_error("test error message: %d", 42);
    oc_log_warn("test warn");
    oc_log_info("should be filtered out");  /* below WARN */
    oc_log_debug("should be filtered out");
    oc_log_set_level(OC_LOG_INFO);
}

