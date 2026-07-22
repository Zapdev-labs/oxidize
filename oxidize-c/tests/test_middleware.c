/*
 * test_middleware.c — middleware stack tests.
 *
 * VAL-MW-001..005 cover:
 *   1. Auth: valid key accepted, invalid key rejected (401), missing header
 *      rejected (401).
 *   2. Rate limit: allows burst, rejects when exceeded (429).
 *   3. Metrics: counters increment correctly + JSON format.
 *   4. Audit: entries recorded and retrievable (newest-first).
 *   5. CORS: headers set correctly.
 */
#define _GNU_SOURCE 1
#include <criterion/criterion.h>

#include "oxidize/middleware.h"

#include <string.h>

/* ─── Auth ────────────────────────────────────────────────────────────────── */

Test(middleware, auth_valid_key_accepted)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_AUTH, "secret123", &rl, "*"),
                 OC_OK);
    OcRequestContext req = {
        .method = OC_HTTP_POST, .path = "/v1/chat",
        .auth_header = "Bearer secret123", .client_ip = "1.2.3.4",
    };
    cr_assert_eq(oc_middleware_process_request(&mw, &req), 0,
                 "valid key should be accepted");
    oc_middleware_free(&mw);
}

Test(middleware, auth_invalid_key_rejected)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_AUTH, "secret123", &rl, "*"),
                 OC_OK);
    OcRequestContext req = {
        .method = OC_HTTP_POST, .path = "/v1/chat",
        .auth_header = "Bearer wrong", .client_ip = "1.2.3.4",
    };
    cr_assert_eq(oc_middleware_process_request(&mw, &req), 401,
                 "invalid key should be rejected with 401");
    oc_middleware_free(&mw);
}

Test(middleware, auth_missing_header_rejected)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_AUTH, "secret123", &rl, "*"),
                 OC_OK);
    OcRequestContext req = {
        .method = OC_HTTP_POST, .path = "/v1/chat",
        .auth_header = NULL, .client_ip = "1.2.3.4",
    };
    cr_assert_eq(oc_middleware_process_request(&mw, &req), 401,
                 "missing header should be rejected with 401");
    oc_middleware_free(&mw);
}

Test(middleware, auth_disabled_when_no_key)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_AUTH, NULL, &rl, "*"), OC_OK);
    OcRequestContext req = {
        .method = OC_HTTP_POST, .path = "/v1/chat",
        .auth_header = NULL, .client_ip = "1.2.3.4",
    };
    cr_assert_eq(oc_middleware_process_request(&mw, &req), 0,
                 "no key configured = auth disabled");
    oc_middleware_free(&mw);
}

Test(middleware, cors_preflight_bypasses_auth)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_AUTH | OC_MW_CORS,
                                    "secret123", &rl, "*"), OC_OK);
    OcRequestContext req = {
        .method = OC_HTTP_OPTIONS, .path = "/v1/chat",
        .auth_header = NULL, .client_ip = "1.2.3.4",
    };
    cr_assert_eq(oc_middleware_process_request(&mw, &req), 0,
                 "OPTIONS preflight should bypass auth");
    oc_middleware_free(&mw);
}

/* ─── Rate limit ──────────────────────────────────────────────────────────── */

Test(middleware, rate_limit_allows_burst)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 3,
                             .per_ip = true };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_RATE_LIMIT, NULL, &rl, "*"),
                 OC_OK);
    OcRequestContext req = {
        .method = OC_HTTP_GET, .path = "/v1/models",
        .auth_header = NULL, .client_ip = "10.0.0.1",
    };
    for (int i = 0; i < 3; i++) {
        cr_assert_eq(oc_middleware_process_request(&mw, &req), 0,
                     "request %d within burst should be allowed", i);
    }
    oc_middleware_free(&mw);
}

Test(middleware, rate_limit_rejects_when_exceeded)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 2,
                             .per_ip = true };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_RATE_LIMIT, NULL, &rl, "*"),
                 OC_OK);
    OcRequestContext req = {
        .method = OC_HTTP_GET, .path = "/v1/models",
        .auth_header = NULL, .client_ip = "10.0.0.2",
    };
    cr_assert_eq(oc_middleware_process_request(&mw, &req), 0);
    cr_assert_eq(oc_middleware_process_request(&mw, &req), 0);
    cr_assert_eq(oc_middleware_process_request(&mw, &req), 429,
                 "third request should be rejected with 429");
    oc_middleware_free(&mw);
}

Test(middleware, rate_limit_per_ip_isolation)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 1,
                             .per_ip = true };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_RATE_LIMIT, NULL, &rl, "*"),
                 OC_OK);
    OcRequestContext a = { .method = OC_HTTP_GET, .path = "/",
                           .auth_header = NULL, .client_ip = "10.0.0.3" };
    OcRequestContext b = { .method = OC_HTTP_GET, .path = "/",
                           .auth_header = NULL, .client_ip = "10.0.0.4" };
    cr_assert_eq(oc_middleware_process_request(&mw, &a), 0);
    cr_assert_eq(oc_middleware_process_request(&mw, &a), 429,
                 "second request from same IP rejected");
    cr_assert_eq(oc_middleware_process_request(&mw, &b), 0,
                 "different IP has its own bucket");
    oc_middleware_free(&mw);
}

/* ─── Metrics ─────────────────────────────────────────────────────────────── */

Test(middleware, metrics_counters_increment)
{
    OcMetrics m;
    memset(&m, 0, sizeof(m));
    oc_metrics_record(&m, 200, 15, 10);
    oc_metrics_record(&m, 200, 50, 20);
    oc_metrics_record(&m, 500, 200, 0);
    char buf[1024];
    size_t n = oc_metrics_format(&m, buf, sizeof(buf));
    cr_assert_gt(n, 0, "format should produce output");
    cr_assert(strstr(buf, "\"request_count\":3") != NULL, "3 requests");
    cr_assert(strstr(buf, "\"error_count\":1") != NULL, "1 error");
    cr_assert(strstr(buf, "\"tokens_generated\":30") != NULL, "30 tokens");
    cr_assert(strstr(buf, "\"avg_latency_ms\":") != NULL, "avg latency");
}

Test(middleware, metrics_process_response_increments)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_METRICS, NULL, &rl, "*"),
                 OC_OK);
    OcRequestContext req = { .method = OC_HTTP_POST, .path = "/v1/chat",
                              .auth_header = NULL, .client_ip = "1.2.3.4" };
    OcResponseContext resp = { .status = 200, .duration_ms = 10, .tokens_generated = 5 };
    oc_middleware_process_response(&mw, &req, &resp);
    char buf[1024];
    cr_assert_gt(oc_metrics_format(&mw.metrics, buf, sizeof(buf)), 0);
    cr_assert(strstr(buf, "\"request_count\":1") != NULL);
    cr_assert(strstr(buf, "\"tokens_generated\":5") != NULL);
    oc_middleware_free(&mw);
}

/* ─── Audit ────────────────────────────────────────────────────────────────── */

Test(middleware, audit_entries_recorded_and_retrievable)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_AUDIT, NULL, &rl, "*"), OC_OK);
    OcRequestContext req = { .method = OC_HTTP_POST, .path = "/v1/chat",
                             .auth_header = NULL, .client_ip = "9.9.9.9" };
    for (int i = 0; i < 5; i++) {
        OcResponseContext resp = { .status = (i == 2) ? 500 : 200,
                                   .duration_ms = (uint64_t)(i * 10), .tokens_generated = 0 };
        oc_middleware_process_response(&mw, &req, &resp);
    }
    OcAuditEntry entries[5];
    size_t n = oc_audit_get(&mw.audit, entries, 5);
    cr_assert_eq(n, 5u, "should retrieve 5 entries");
    /* newest first: entries[0]=i=4 (status 200, dur 40), entries[4]=i=0 (dur 0).
     * The 500-status entry was i=2, which lands at index 2 in newest-first. */
    cr_assert_eq(entries[0].status, 200, "newest entry was i=4 (status 200)");
    cr_assert_eq(entries[0].duration_ms, 40u);
    cr_assert_str_eq(entries[0].method, "POST");
    cr_assert_str_eq(entries[0].client_ip, "9.9.9.9");
    cr_assert_eq(entries[2].status, 500, "i=2 (status 500) at index 2");
    cr_assert_eq(entries[4].duration_ms, 0u, "oldest entry was i=0 (dur 0)");
    oc_middleware_free(&mw);
}

Test(middleware, audit_format_json)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_AUDIT, NULL, &rl, "*"), OC_OK);
    OcRequestContext req = { .method = OC_HTTP_GET, .path = "/v1/models",
                             .auth_header = NULL, .client_ip = "1.1.1.1" };
    OcResponseContext resp = { .status = 200, .duration_ms = 5, .tokens_generated = 0 };
    oc_middleware_process_response(&mw, &req, &resp);
    char buf[2048];
    size_t n = oc_audit_format(&mw.audit, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(buf[0] == '[', "JSON array starts with [");
    cr_assert(strstr(buf, "\"method\":\"GET\"") != NULL);
    cr_assert(strstr(buf, "\"path\":\"/v1/models\"") != NULL);
    cr_assert(strstr(buf, "\"client_ip\":\"1.1.1.1\"") != NULL);
    oc_middleware_free(&mw);
}

/* ─── CORS ────────────────────────────────────────────────────────────────── */

Test(middleware, cors_headers_set_correctly)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_CORS, NULL, &rl,
                                    "https://app.example.com"), OC_OK);
    char buf[512];
    size_t n = oc_middleware_cors_headers(&mw, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "Access-Control-Allow-Origin: https://app.example.com") != NULL);
    cr_assert(strstr(buf, "Access-Control-Allow-Methods: GET, POST, OPTIONS") != NULL);
    cr_assert(strstr(buf, "Access-Control-Allow-Headers: Content-Type, Authorization") != NULL);
    oc_middleware_free(&mw);
}

Test(middleware, cors_disabled_writes_nothing)
{
    OcMiddleware mw;
    OcRateLimitConfig rl = { .requests_per_minute = 0, .burst_size = 0,
                             .per_ip = false };
    cr_assert_eq(oc_middleware_init(&mw, OC_MW_AUTH, NULL, &rl, "*"), OC_OK);
    char buf[64];
    cr_assert_eq(oc_middleware_cors_headers(&mw, buf, sizeof(buf)), 0u,
                 "CORS disabled should write nothing");
    oc_middleware_free(&mw);
}
