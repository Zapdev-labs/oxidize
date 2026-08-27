/*
 * middleware.h — server middleware stack (auth, rate limit, metrics, audit, CORS).
 *
 * Implements the `server-middleware` feature on top of server-http-core.
 * Each incoming request passes through the configured middleware chain before
 * the route handler is dispatched, and each response passes back through for
 * post-processing (audit recording, metrics, CORS headers).
 *
 * Middleware is dependency-free (libc + libpthread only), matching the
 * project's convention. Atomic counters are used for metrics; a mutex
 * protects the per-IP rate-limit table and the audit ring buffer.
 */
#ifndef OXIDIZE_MIDDLEWARE_H
#define OXIDIZE_MIDDLEWARE_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/http.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Middleware types ──────────────────────────────────────────────────── */

typedef enum {
    OC_MW_AUTH       = 1u << 0,
    OC_MW_RATE_LIMIT = 1u << 1,
    OC_MW_METRICS    = 1u << 2,
    OC_MW_AUDIT      = 1u << 3,
    OC_MW_CORS       = 1u << 4,
    OC_MW_ALL        = 0x1Fu,
} OcMiddlewareType;

/* ─── Auth config ─────────────────────────────────────────────────────────
 *
 * When `enabled` is true, requests must carry an `Authorization: Bearer
 * <api_key>` header whose value matches `api_key`. Requests without a
 * matching header are rejected with 401 Unauthorized. */
typedef struct OcAuthConfig {
    char  *api_key;     /* owned, NUL-terminated; NULL disables auth     */
    bool   enabled;
} OcAuthConfig;

/* ─── Rate-limit config ───────────────────────────────────────────────────
 *
 * Token-bucket per client IP. `requests_per_minute` is the steady-state
 * refill rate; `burst_size` is the maximum tokens the bucket can hold.
 * When `per_ip` is true, each IP has its own bucket; when false, a single
 * global bucket is used. Requests that find an empty bucket are rejected
 * with 429 Too Many Requests. */
typedef struct OcRateLimitConfig {
    uint32_t requests_per_minute;
    uint32_t burst_size;
    bool     per_ip;
} OcRateLimitConfig;

/* ─── Metrics ─────────────────────────────────────────────────────────────
 *
 * Atomic counters updated by every request. `avg_latency_ms` is a running
 * average maintained with a compare-and-swap loop. The latency histogram
 * has fixed buckets covering common latency bands. */
#define OC_METRICS_HIST_BUCKETS 8u

typedef struct OcMetrics {
    _Atomic uint64_t request_count;
    _Atomic uint64_t error_count;       /* status >= 400                  */
    _Atomic uint64_t tokens_generated;
    _Atomic uint64_t total_latency_ms;   /* sum for average                */
    _Atomic uint64_t latency_hist[OC_METRICS_HIST_BUCKETS];
} OcMetrics;

/* ─── Audit entry + ring buffer ───────────────────────────────────────────
 *
 * The audit log is a fixed-size ring buffer of the most recent N entries.
 * It is queryable via the /admin/audit endpoint (handler provided by the
 * caller). Entries are timestamped with `time(NULL)` (seconds since epoch). */
#define OC_AUDIT_RING_SIZE 256u

typedef struct OcAuditEntry {
    uint64_t timestamp;     /* seconds since epoch                       */
    char     method[8];      /* "GET", "POST", ...                       */
    char     path[256];
    int      status;
    uint64_t duration_ms;
    char     client_ip[64];
} OcAuditEntry;

typedef struct OcAuditLog {
    OcAuditEntry ring[OC_AUDIT_RING_SIZE];
    _Atomic uint64_t head;       /* next write index (mod RING_SIZE)     */
    _Atomic uint64_t count;       /* total entries ever written           */
    pthread_mutex_t lock;
} OcAuditLog;

/* ─── Per-IP rate-limit bucket table ──────────────────────────────────────
 *
 * Fixed-size open-addressing table keyed by client IP string. New IPs
 * evict the least-recently-used entry when the table is full. The token
 * count + last-refill timestamp live per bucket. */
#define OC_RL_TABLE_SIZE 256u

typedef struct OcRateBucket {
    char     ip[64];
    double   tokens;       /* current token count (fractional)         */
    uint64_t last_refill;  /* ms since epoch of last refill            */
    bool     in_use;
} OcRateBucket;

typedef struct OcRateLimiter {
    OcRateLimitConfig cfg;
    OcRateBucket      global_bucket;
    OcRateBucket      buckets[OC_RL_TABLE_SIZE];
    pthread_mutex_t   lock;
} OcRateLimiter;

/* ─── Request/response context (passed through the chain) ───────────────── */
typedef struct OcRequestContext {
    OcHttpMethod method;
    const char   *path;            /* NUL-terminated                      */
    const char   *auth_header;     /* Authorization value or NULL         */
    const char   *client_ip;       /* "1.2.3.4" or NULL                  */
} OcRequestContext;

typedef struct OcResponseContext {
    int      status;
    uint64_t duration_ms;
    size_t   tokens_generated;
} OcResponseContext;

/* ─── Middleware stack ──────────────────────────────────────────────────── */
typedef struct OcMiddleware {
    uint32_t         enabled;       /* bitmask of OcMiddlewareType       */
    OcAuthConfig     auth;
    OcRateLimiter    rate_limiter;
    OcMetrics        metrics;
    OcAuditLog       audit;
    /* CORS config */
    char             cors_origin[128];
    char             cors_methods[128];
    char             cors_headers[256];
    bool             cors_enabled;
} OcMiddleware;

/* Initialize a middleware stack with the given enabled flags + configs.
 * `api_key` may be NULL (disables auth). `cors_origin` may be NULL
 * (defaults to "*"). Returns OC_OK on success, OC_ERR_INVALID_ARG if `mw`
 * is NULL, OC_ERR_OOM if the API key copy fails, or OC_ERR_INTERNAL if
 * mutex initialization fails. */
OcError oc_middleware_init(OcMiddleware *mw,
                           uint32_t enabled,
                           const char *api_key,
                           const OcRateLimitConfig *rl_cfg,
                           const char *cors_origin);

/* Release all heap allocations owned by the middleware stack. Safe on NULL. */
void oc_middleware_free(OcMiddleware *mw);

/* Process an incoming request through the pre-handler middleware chain.
 * Returns 0 if the request is allowed to proceed, or an HTTP status code
 * (401, 429) if a middleware rejects it. */
int oc_middleware_process_request(OcMiddleware *mw, const OcRequestContext *req);

/* Process a response through the post-handler middleware chain (metrics,
 * audit). Records the metrics + audit entry; called by the server after
 * the route handler returns. */
void oc_middleware_process_response(OcMiddleware *mw,
                                    const OcRequestContext *req,
                                    const OcResponseContext *resp);

/* Append CORS headers to a response buffer. Returns bytes written (0 on
 * overflow). No-op if CORS is disabled (writes nothing). */
size_t oc_middleware_cors_headers(const OcMiddleware *mw, char *buf, size_t cap);

/* ─── Metrics helpers ──────────────────────────────────────────────────── */

/* Record a single request's metrics atomically. */
void oc_metrics_record(OcMetrics *m, int status, uint64_t duration_ms,
                       size_t tokens_generated);

/* Format metrics as a JSON string into `buf` (NUL-terminated). Returns
 * bytes written excluding the NUL, or 0 on overflow. */
size_t oc_metrics_format(const OcMetrics *m, char *buf, size_t cap);

/* Format metrics in the Prometheus text exposition format (version 0.0.4)
 * into `buf` (NUL-terminated). This is what `GET /metrics` serves, matching
 * the Rust server's `TextEncoder` output so existing scrape configs and
 * dashboards keep working. The latency histogram is emitted as a proper
 * Prometheus cumulative histogram (`_bucket`/`_sum`/`_count` with `le`
 * labels), which requires running the bucket counts into a running total.
 * Returns bytes written excluding the NUL, or 0 on overflow. */
size_t oc_metrics_format_prometheus(const OcMetrics *m, char *buf, size_t cap);

/* ─── Audit helpers ─────────────────────────────────────────────────────── */

/* Append an entry to the audit ring buffer. */
void oc_audit_record(OcAuditLog *log, const OcRequestContext *req,
                     const OcResponseContext *resp);

/* Copy up to `count` most-recent entries into `out` (caller-allocated).
 * Entries are returned newest-first. Returns the number copied. */
size_t oc_audit_get(const OcAuditLog *log, OcAuditEntry *out, size_t count);

/* Format the audit log as a JSON array into `buf`. Returns bytes written
 * excluding the NUL, or 0 on overflow. */
size_t oc_audit_format(const OcAuditLog *log, char *buf, size_t cap);

/* ─── Rate-limiter helpers ────────────────────────────────────────────────
 *
 * Returns true if the request is allowed (and consumes a token), false if
 * the rate limit is exceeded. Thread-safe. */
bool oc_rate_limiter_allow(OcRateLimiter *rl, const char *client_ip);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MIDDLEWARE_H */
