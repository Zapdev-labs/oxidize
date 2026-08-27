/*
 * log.h — oc_log leveled logging with OX_LOG_LEVEL env filter.
 *
 * Writes to stderr with a level prefix ("DEBUG", "INFO", "WARN", "ERROR").
 * The OX_LOG_LEVEL environment variable filters messages: only messages at
 * or above the configured level are emitted. Levels (numeric):
 *   0=DEBUG  1=INFO  2=WARN  3=ERROR
 * Default (env unset) is INFO. Library code MUST use oc_log, never printf.
 *
 * Threading contract (SINGLE-THREADED INIT):
 *   The global `g_level` flag (and the `g_level_set` / `g_env_checked`
 *   cached flags in log.c) are NOT atomic and NOT protected by a mutex.
 *   Callers MUST set the log level exactly once, from a single thread,
 *   BEFORE spawning any worker threads that may call oc_log(). Concretely:
 *     - Either call oc_log_init_from_env() once at process startup (the
 *       first oc_log() call lazy-initializes from OX_LOG_LEVEL if not done
 *       explicitly), OR call oc_log_set_level(level) once at startup.
 *     - After threads are spawned, treat g_level as read-only. Mutating
 *       oc_log_set_level() at runtime from multiple threads is a data race
 *       (UB under C11) — wrap level changes in a single-threaded
 *       coordinator if dynamic reconfiguration is genuinely needed.
 *   The oc_log() emit path itself uses flockfile(stderr) so concurrent log
 *   writes from multiple threads are safe; only the level-probe +
 *   lazy-init read of g_level is the racy part covered by this contract.
 *
 * Port concept: architecture.md §9 conventions.
 */
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
 * Default is OC_LOG_INFO unless OX_LOG_LEVEL is set.
 *
 * THREADING: must be called once at startup BEFORE spawning threads that
 * call oc_log(); g_level is not atomic (see file-level comment). */
void oc_log_set_level(OcLogLevel level);

/* Get the current minimum level. Safe to call from any thread once init
 * has completed (i.e. after the first oc_log() / oc_log_init_from_env()
 * call from the main thread). */
OcLogLevel oc_log_get_level(void);

/* Re-read OX_LOG_LEVEL from the environment and apply it. Called once at
 * library init; callers normally don't need this. Must be invoked from a
 * single thread before worker threads start (see file-level comment). */
void oc_log_init_from_env(void);

/* Log a message at the given level. `fmt` is a printf-style format string.
 * If the message's level is below the configured minimum, this is a no-op. */
void oc_log(OcLogLevel level, const char *fmt, ...);

/* Convenience macros. Pass a single printf-style format string + args:
 *   oc_log_info("loaded %zu tensors", n);
 *   oc_log_warn("config missing, using default");
 */
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
