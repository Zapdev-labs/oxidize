/* log.h — oc_log leveled logging with OX_LOG_LEVEL env filter. Concurrent oc_log() writes are flockfile-safe; g_level and lazy-init are not atomic — set the level once before worker threads. */
#ifndef OXIDIZE_LOG_H
#define OXIDIZE_LOG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_LOG_DEBUG = 0,
    OC_LOG_INFO  = 1,
    OC_LOG_WARN  = 2,
    OC_LOG_ERROR = 3,
    OC_LOG__COUNT,
} OcLogLevel;

/* Set the minimum level to emit (overrides OX_LOG_LEVEL for this process).
 * Call once from a single thread before worker threads spawn; g_level is not atomic. */
void oc_log_set_level(OcLogLevel level);

/* Get the current minimum level. Safe to call from any thread once init
 * has completed (i.e. after the first oc_log() / oc_log_init_from_env()
 * call from the main thread). */
OcLogLevel oc_log_get_level(void);

/* Re-read OX_LOG_LEVEL from the environment and apply it. Called once at
 * library init; callers normally don't need this. Must be invoked from a
 * single thread before worker threads start (same contract as oc_log_set_level). */
void oc_log_init_from_env(void);

/* Log a message at the given level. `fmt` is a printf-style format string.
 * If the message's level is below the configured minimum, this is a no-op. */
void oc_log(OcLogLevel level, const char *fmt, ...);

/* Convenience macros. Pass a single printf-style format string + args: oc_log_info("loaded %zu tensors", n); oc_log_warn("config missing, using default"); */
#define oc_log_debug(...) oc_log(OC_LOG_DEBUG, __VA_ARGS__)
#define oc_log_info(...)  oc_log(OC_LOG_INFO,  __VA_ARGS__)
#define oc_log_warn(...)  oc_log(OC_LOG_WARN,  __VA_ARGS__)
#define oc_log_error(...) oc_log(OC_LOG_ERROR, __VA_ARGS__)

/* Convert level <-> string. */
const char *oc_log_level_name(OcLogLevel level);
OcLogLevel oc_log_level_from_str(const char *s);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LOG_H */
