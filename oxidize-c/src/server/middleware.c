/* middleware.c — server middleware stack implementation. */
#define _POSIX_C_SOURCE 200809L   /* strdup, snprintf */
#include "oxidize/middleware.h"

#include "oxidize/log.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>


static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

/* Bucket for a latency value into histogram index 0..N-1.
 * Bucket edges (ms): [0,10), [10,50), [50,100), [100,250), [250,500),
 * [500,1000), [1000,5000), [5000,+inf). */
static size_t latency_bucket(uint64_t ms)
{
    if (ms < 10u)   return 0;
    if (ms < 50u)   return 1;
    if (ms < 100u)  return 2;
    if (ms < 250u)  return 3;
    if (ms < 500u)  return 4;
    if (ms < 1000u) return 5;
    if (ms < 5000u) return 6;
    return 7;
}


/* Returns 0 if authorized, 401 otherwise. */
static int auth_check(const OcAuthConfig *auth, const char *auth_header)
{
    if (!auth->enabled || auth->api_key == NULL) return 0;
    if (auth_header == NULL) return 401;
    /* Expect "Bearer <key>" (case-insensitive scheme). */
    const char *p = auth_header;
    while (*p == ' ' || *p == '\t') p++;
    if (strncasecmp(p, "Bearer", 6) != 0) return 401;
    p += 6;
    while (*p == ' ' || *p == '\t') p++;
    if (strcmp(p, auth->api_key) != 0) return 401;
    return 0;
}


static void bucket_refill(OcRateBucket *b, const OcRateLimitConfig *cfg,
                          uint64_t now)
{
    if (b->last_refill == 0) {
        b->last_refill = now;
        b->tokens = (double)cfg->burst_size;
        return;
    }
    if (now <= b->last_refill) return;
    double add = (double)(now - b->last_refill) *
                 (double)cfg->requests_per_minute / 60000.0;
    b->tokens += add;
    if (b->tokens > (double)cfg->burst_size) {
        b->tokens = (double)cfg->burst_size;
    }
    b->last_refill = now;
}

static size_t rl_find_slot(OcRateLimiter *rl, const char *ip)
{
    /* Linear scan; returns index of matching slot, or first empty slot,
     * or the LRU slot (lowest last_refill) if full. */
    size_t first_empty = OC_RL_TABLE_SIZE;
    size_t lru = 0;
    uint64_t lru_ts = UINT64_MAX;
    for (size_t i = 0; i < OC_RL_TABLE_SIZE; i++) {
        if (!rl->buckets[i].in_use) {
            if (first_empty == OC_RL_TABLE_SIZE) first_empty = i;
            continue;
        }
        if (strncmp(rl->buckets[i].ip, ip, sizeof(rl->buckets[i].ip) - 1) == 0) {
            return i;
        }
        if (rl->buckets[i].last_refill < lru_ts) {
            lru_ts = rl->buckets[i].last_refill;
            lru = i;
        }
    }
    return first_empty != OC_RL_TABLE_SIZE ? first_empty : lru;
}

bool oc_rate_limiter_allow(OcRateLimiter *rl, const char *client_ip)
{
    if (rl->cfg.requests_per_minute == 0 && rl->cfg.burst_size == 0) {
        return true;
    }
    const char *ip = (client_ip != NULL) ? client_ip : "0.0.0.0";
    uint64_t now = now_ms();
    pthread_mutex_lock(&rl->lock);
    OcRateBucket *b;
    if (rl->cfg.per_ip) {
        size_t idx = rl_find_slot(rl, ip);
        b = &rl->buckets[idx];
        if (!b->in_use || strncmp(b->ip, ip, sizeof(b->ip) - 1) != 0) {
            memset(b, 0, sizeof(*b));
            strncpy(b->ip, ip, sizeof(b->ip) - 1);
            b->ip[sizeof(b->ip) - 1] = '\0';
            b->in_use = true;
            b->tokens = (double)rl->cfg.burst_size;
            b->last_refill = now;
        } else {
            bucket_refill(b, &rl->cfg, now);
        }
    } else {
        b = &rl->global_bucket;
        if (b->last_refill == 0) {
            strncpy(b->ip, "global", sizeof(b->ip) - 1);
            b->in_use = true;
            b->tokens = (double)rl->cfg.burst_size;
            b->last_refill = now;
        } else {
            bucket_refill(b, &rl->cfg, now);
        }
    }
    bool allow = b->tokens >= 1.0;
    if (allow) {
        b->tokens -= 1.0;
    }
    pthread_mutex_unlock(&rl->lock);
    return allow;
}


void oc_metrics_record(OcMetrics *m, int status, uint64_t duration_ms,
                       size_t tokens_generated)
{
    atomic_fetch_add_explicit(&m->request_count, 1u, memory_order_relaxed);
    if (status >= 400) {
        atomic_fetch_add_explicit(&m->error_count, 1u, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&m->tokens_generated,
                              (uint64_t)tokens_generated, memory_order_relaxed);
    atomic_fetch_add_explicit(&m->total_latency_ms, duration_ms,
                              memory_order_relaxed);
    size_t bucket = latency_bucket(duration_ms);
    atomic_fetch_add_explicit(&m->latency_hist[bucket], 1u,
                              memory_order_relaxed);
}

size_t oc_metrics_format(const OcMetrics *m, char *buf, size_t cap)
{
    uint64_t req   = atomic_load_explicit(&m->request_count, memory_order_relaxed);
    uint64_t err   = atomic_load_explicit(&m->error_count, memory_order_relaxed);
    uint64_t toks  = atomic_load_explicit(&m->tokens_generated, memory_order_relaxed);
    uint64_t sum   = atomic_load_explicit(&m->total_latency_ms, memory_order_relaxed);
    double avg = (req > 0) ? (double)sum / (double)req : 0.0;

    int n = snprintf(buf, cap,
        "{\"request_count\":%llu,\"error_count\":%llu,"
        "\"avg_latency_ms\":%.2f,\"tokens_generated\":%llu,"
        "\"latency_histogram\":[",
        (unsigned long long)req, (unsigned long long)err,
        avg, (unsigned long long)toks);
    if (n < 0 || (size_t)n >= cap) return 0;
    size_t off = (size_t)n;
    for (size_t i = 0; i < OC_METRICS_HIST_BUCKETS; i++) {
        uint64_t v = atomic_load_explicit(&m->latency_hist[i],
                                          memory_order_relaxed);
        const char *fmt = (i + 1 < OC_METRICS_HIST_BUCKETS) ? "%llu," : "%llu]";
        int w = snprintf(buf + off, cap - off, fmt, (unsigned long long)v);
        if (w < 0 || (size_t)w >= cap - off) return 0;
        off += (size_t)w;
    }
    int w2 = snprintf(buf + off, cap - off, "}");
    if (w2 < 0 || (size_t)w2 >= cap - off) return 0;
    off += (size_t)w2;
    return off;
}

/* Upper edge of each latency bucket in seconds, matching latency_bucket()
 * above. The final bucket is +Inf. */
static const char *const k_bucket_le[OC_METRICS_HIST_BUCKETS] = {
    "0.01", "0.05", "0.1", "0.25", "0.5", "1", "5", "+Inf"
};

/* Append to `buf` at `*off`, returning false on overflow. */
static bool pfmt(char *buf, size_t cap, size_t *off, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *off, cap - *off, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= cap - *off) return false;
    *off += (size_t)n;
    return true;
}

size_t oc_metrics_format_prometheus(const OcMetrics *m, char *buf, size_t cap)
{
    if (!m || !buf || cap == 0) return 0;
    uint64_t req  = atomic_load_explicit(&m->request_count, memory_order_relaxed);
    uint64_t err  = atomic_load_explicit(&m->error_count, memory_order_relaxed);
    uint64_t toks = atomic_load_explicit(&m->tokens_generated, memory_order_relaxed);
    uint64_t sum  = atomic_load_explicit(&m->total_latency_ms, memory_order_relaxed);

    size_t off = 0;
    if (!pfmt(buf, cap, &off,
              "# HELP oxidize_requests_total Total HTTP requests handled.\n"
              "# TYPE oxidize_requests_total counter\n"
              "oxidize_requests_total %llu\n"
              "# HELP oxidize_request_errors_total Requests answered with status >= 400.\n"
              "# TYPE oxidize_request_errors_total counter\n"
              "oxidize_request_errors_total %llu\n"
              "# HELP oxidize_tokens_generated_total Tokens produced by the model.\n"
              "# TYPE oxidize_tokens_generated_total counter\n"
              "oxidize_tokens_generated_total %llu\n",
              (unsigned long long)req, (unsigned long long)err,
              (unsigned long long)toks)) {
        return 0;
    }

    if (!pfmt(buf, cap, &off,
              "# HELP oxidize_request_duration_seconds Request latency.\n"
              "# TYPE oxidize_request_duration_seconds histogram\n")) {
        return 0;
    }
    /* Prometheus buckets are cumulative: each `le` bucket counts every
     * observation at or below that edge, so run a total across the
     * non-cumulative internal histogram. */
    uint64_t cumulative = 0;
    for (size_t i = 0; i < OC_METRICS_HIST_BUCKETS; i++) {
        cumulative += atomic_load_explicit(&m->latency_hist[i],
                                           memory_order_relaxed);
        if (!pfmt(buf, cap, &off,
                  "oxidize_request_duration_seconds_bucket{le=\"%s\"} %llu\n",
                  k_bucket_le[i], (unsigned long long)cumulative)) {
            return 0;
        }
    }
    /* `_count` must equal the +Inf bucket; both derive from the same
     * histogram rather than from request_count, which a concurrent
     * oc_metrics_record() could have advanced past the histogram write. */
    if (!pfmt(buf, cap, &off,
              "oxidize_request_duration_seconds_sum %.3f\n"
              "oxidize_request_duration_seconds_count %llu\n",
              (double)sum / 1000.0, (unsigned long long)cumulative)) {
        return 0;
    }
    return off;
}


void oc_audit_record(OcAuditLog *log, const OcRequestContext *req,
                     const OcResponseContext *resp)
{
    pthread_mutex_lock(&log->lock);
    uint64_t idx = atomic_load_explicit(&log->head, memory_order_relaxed);
    OcAuditEntry *e = &log->ring[idx % OC_AUDIT_RING_SIZE];
    e->timestamp = (uint64_t)time(NULL);
    e->status = resp->status;
    e->duration_ms = resp->duration_ms;
    const char *method_str = (req->method == OC_HTTP_GET)   ? "GET" :
                             (req->method == OC_HTTP_POST)  ? "POST" :
                             (req->method == OC_HTTP_OPTIONS) ? "OPTIONS" : "OTHER";
    size_t ml = strlen(method_str);
    if (ml >= sizeof(e->method)) ml = sizeof(e->method) - 1;
    memcpy(e->method, method_str, ml);
    e->method[ml] = '\0';
    if (req->path != NULL) {
        size_t pl = strlen(req->path);
        if (pl >= sizeof(e->path)) pl = sizeof(e->path) - 1;
        memcpy(e->path, req->path, pl);
        e->path[pl] = '\0';
    } else {
        e->path[0] = '\0';
    }
    if (req->client_ip != NULL) {
        size_t il = strlen(req->client_ip);
        if (il >= sizeof(e->client_ip)) il = sizeof(e->client_ip) - 1;
        memcpy(e->client_ip, req->client_ip, il);
        e->client_ip[il] = '\0';
    } else {
        e->client_ip[0] = '\0';
    }
    atomic_store_explicit(&log->head, idx + 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&log->count, 1u, memory_order_relaxed);
    pthread_mutex_unlock(&log->lock);
}

size_t oc_audit_get(const OcAuditLog *log, OcAuditEntry *out, size_t count)
{
    OcAuditLog *mutlog = (OcAuditLog *)log; /* const-safe: lock only */
    pthread_mutex_lock(&mutlog->lock);
    uint64_t total = atomic_load_explicit(&log->count, memory_order_relaxed);
    uint64_t head = atomic_load_explicit(&log->head, memory_order_relaxed);
    size_t available = (total < OC_AUDIT_RING_SIZE) ? (size_t)total
                                                    : OC_AUDIT_RING_SIZE;
    size_t copy_n = (count < available) ? count : available;
    /* newest first: head-1, head-2, ... */
    for (size_t i = 0; i < copy_n; i++) {
        uint64_t src_idx = (head + OC_AUDIT_RING_SIZE - 1u - (uint64_t)i)
                          % OC_AUDIT_RING_SIZE;
        out[i] = log->ring[src_idx];
    }
    pthread_mutex_unlock(&mutlog->lock);
    return copy_n;
}

/* JSON-escape `src` into `dst` (cap bytes, NUL-terminated). Escapes quote,
 * backslash, and control characters so audit output stays valid JSON. */
static void json_escape_into(const char *src, char *dst, size_t cap)
{
    size_t o = 0;
    for (const char *s = src; *s != '\0' && o + 7 < cap; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            dst[o++] = '\\';
            dst[o++] = (char)c;
        } else if (c < 0x20u) {
            int w = snprintf(dst + o, cap - o, "\\u%04x", (unsigned)c);
            if (w < 0) break;
            o += (size_t)w;
        } else {
            dst[o++] = (char)c;
        }
    }
    dst[o] = '\0';
}

size_t oc_audit_format(const OcAuditLog *log, char *buf, size_t cap)
{
    OcAuditEntry entries[OC_AUDIT_RING_SIZE];
    size_t n = oc_audit_get(log, entries, OC_AUDIT_RING_SIZE);
    int w = snprintf(buf, cap, "[");
    if (w < 0 || (size_t)w >= cap) return 0;
    size_t off = (size_t)w;
    for (size_t i = 0; i < n; i++) {
        const OcAuditEntry *e = &entries[i];
        char esc_method[sizeof(e->method) * 6];
        char esc_path[sizeof(e->path) * 6];
        char esc_ip[sizeof(e->client_ip) * 6];
        json_escape_into(e->method, esc_method, sizeof(esc_method));
        json_escape_into(e->path, esc_path, sizeof(esc_path));
        json_escape_into(e->client_ip, esc_ip, sizeof(esc_ip));
        int w2 = snprintf(buf + off, cap - off,
            "%s{\"timestamp\":%llu,\"method\":\"%s\",\"path\":\"%s\","
            "\"status\":%d,\"duration_ms\":%llu,\"client_ip\":\"%s\"}",
            (i == 0) ? "" : ",",
            (unsigned long long)e->timestamp, esc_method, esc_path,
            e->status, (unsigned long long)e->duration_ms, esc_ip);
        if (w2 < 0 || (size_t)w2 >= cap - off) return 0;
        off += (size_t)w2;
    }
    int w3 = snprintf(buf + off, cap - off, "]");
    if (w3 < 0 || (size_t)w3 >= cap - off) return 0;
    off += (size_t)w3;
    return off;
}


OcError oc_middleware_init(OcMiddleware *mw, uint32_t enabled,
                          const char *api_key,
                          const OcRateLimitConfig *rl_cfg,
                          const char *cors_origin)
{
    if (mw == NULL) return OC_ERR_INVALID_ARG;
    memset(mw, 0, sizeof(*mw));
    mw->enabled = enabled;

    /* Auth */
    if (api_key != NULL && api_key[0] != '\0') {
        mw->auth.api_key = strdup(api_key);
        if (mw->auth.api_key == NULL) return OC_ERR_OOM;
        mw->auth.enabled = true;
    } else {
        mw->auth.api_key = NULL;
        mw->auth.enabled = false;
    }

    /* Rate limiter */
    if (rl_cfg != NULL) {
        mw->rate_limiter.cfg = *rl_cfg;
    } else {
        mw->rate_limiter.cfg.requests_per_minute = 60;
        mw->rate_limiter.cfg.burst_size = 10;
        mw->rate_limiter.cfg.per_ip = true;
    }
    if (pthread_mutex_init(&mw->rate_limiter.lock, NULL) != 0) {
        free(mw->auth.api_key);
        return OC_ERR_INTERNAL;
    }
    mw->rate_limiter.global_bucket.last_refill = 0;

    /* Audit */
    if (pthread_mutex_init(&mw->audit.lock, NULL) != 0) {
        pthread_mutex_destroy(&mw->rate_limiter.lock);
        free(mw->auth.api_key);
        return OC_ERR_INTERNAL;
    }

    /* CORS */
    mw->cors_enabled = (enabled & OC_MW_CORS) != 0;
    const char *origin = (cors_origin != NULL) ? cors_origin : "*";
    strncpy(mw->cors_origin, origin, sizeof(mw->cors_origin) - 1);
    strncpy(mw->cors_methods, "GET, POST, OPTIONS",
            sizeof(mw->cors_methods) - 1);
    strncpy(mw->cors_headers, "Content-Type, Authorization",
            sizeof(mw->cors_headers) - 1);

    return OC_OK;
}

void oc_middleware_free(OcMiddleware *mw)
{
    if (mw == NULL) return;
    free(mw->auth.api_key);
    mw->auth.api_key = NULL;
    pthread_mutex_destroy(&mw->rate_limiter.lock);
    pthread_mutex_destroy(&mw->audit.lock);
}


int oc_middleware_process_request(OcMiddleware *mw, const OcRequestContext *req)
{
    if (mw == NULL || req == NULL) return 500;

    /* CORS preflight: OPTIONS requests are allowed through (no auth). */
    if ((mw->enabled & OC_MW_CORS) != 0 && req->method == OC_HTTP_OPTIONS) {
        return 0;
    }

    if ((mw->enabled & OC_MW_AUTH) != 0) {
        int r = auth_check(&mw->auth, req->auth_header);
        if (r != 0) return r;
    }

    if ((mw->enabled & OC_MW_RATE_LIMIT) != 0) {
        if (!oc_rate_limiter_allow(&mw->rate_limiter, req->client_ip)) {
            return 429;
        }
    }

    return 0;
}

void oc_middleware_process_response(OcMiddleware *mw,
                                    const OcRequestContext *req,
                                    const OcResponseContext *resp)
{
    if (mw == NULL || req == NULL || resp == NULL) return;
    if ((mw->enabled & OC_MW_METRICS) != 0) {
        oc_metrics_record(&mw->metrics, resp->status,
                          resp->duration_ms, resp->tokens_generated);
    }
    if ((mw->enabled & OC_MW_AUDIT) != 0) {
        oc_audit_record(&mw->audit, req, resp);
    }
}

size_t oc_middleware_cors_headers(const OcMiddleware *mw, char *buf, size_t cap)
{
    if (mw == NULL || !mw->cors_enabled) return 0;
    int n = snprintf(buf, cap,
        "Access-Control-Allow-Origin: %s\r\n"
        "Access-Control-Allow-Methods: %s\r\n"
        "Access-Control-Allow-Headers: %s\r\n",
        mw->cors_origin, mw->cors_methods, mw->cors_headers);
    if (n < 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}
