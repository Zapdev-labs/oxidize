#define _POSIX_C_SOURCE 200809L  /* for strcasecmp */

#include "oxidize/log.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ci_eq(const char *a, const char *b)
{
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}

static OcLogLevel g_level      = OC_LOG_INFO;
static int        g_level_set   = 0;   /* 0 = not yet initialized */
static int        g_env_checked = 0;

static OcLogLevel parse_env_level(void)
{
    const char *env = getenv("OX_LOG_LEVEL");
    if (!env || !*env) return OC_LOG_INFO;
    /* Accept either numeric (0..3) or symbolic names. */
    if (env[0] >= '0' && env[0] <= '9' && env[1] == '\0') {
        int v = env[0] - '0';
        if (v < 0) v = 0;
        if (v > (int)OC_LOG_ERROR) v = (int)OC_LOG_ERROR;
        return (OcLogLevel)v;
    }
    return oc_log_level_from_str(env);
}

void oc_log_init_from_env(void)
{
    if (g_env_checked) return;
    g_env_checked = 1;
    if (!g_level_set) {
        g_level = parse_env_level();
    }
}

void oc_log_set_level(OcLogLevel level)
{
    if ((unsigned)level >= (unsigned)OC_LOG__COUNT) return;
    g_level     = level;
    g_level_set = 1;
}

OcLogLevel oc_log_get_level(void)
{
    if (!g_env_checked && !g_level_set) {
        oc_log_init_from_env();
    }
    return g_level;
}

static const char *level_str(OcLogLevel l)
{
    switch (l) {
    case OC_LOG_DEBUG: return "DEBUG";
    case OC_LOG_INFO:  return "INFO";
    case OC_LOG_WARN:  return "WARN";
    case OC_LOG_ERROR: return "ERROR";
    default:           return "?";
    }
}

OcLogLevel oc_log_level_from_str(const char *s)
{
    if (!s) return OC_LOG_INFO;
    /* Case-insensitive compare */
    if (ci_eq(s, "DEBUG")) return OC_LOG_DEBUG;
    if (ci_eq(s, "INFO"))  return OC_LOG_INFO;
    if (ci_eq(s, "WARN") || ci_eq(s, "WARNING")) return OC_LOG_WARN;
    if (ci_eq(s, "ERROR")) return OC_LOG_ERROR;
    return OC_LOG_INFO;
}

const char *oc_log_level_name(OcLogLevel level)
{
    if ((unsigned)level >= (unsigned)OC_LOG__COUNT) return "?";
    return level_str(level);
}

void oc_log(OcLogLevel level, const char *fmt, ...)
{
    if ((unsigned)level >= (unsigned)OC_LOG__COUNT) return;
    if (level < oc_log_get_level()) return;

    flockfile(stderr);
    fputs(level_str(level), stderr);
    fputc(':', stderr);
    fputc(' ', stderr);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    /* Ensure a trailing newline. */
    fputc('\n', stderr);
    funlockfile(stderr);
}
