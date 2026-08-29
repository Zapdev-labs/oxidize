/*
 * test_http.c — HTTP server core tests.
 *
 * VAL-HTTP-001..005 cover:
 *   1. oc_http_format_response produces a valid HTTP/1.1 response.
 *   2. oc_http_status_line covers common codes.
 *   3. End-to-end: start a server on a random port, send a GET, verify the
 *      handler is invoked and the response parses correctly.
 *   4. POST with a JSON body reaches the handler intact.
 *   5. Server stop + join cleanly shuts down.
 *
 * The end-to-end test uses a real loopback socket; no mocking. The server
 * runs on port 0 (kernel-assigned) to avoid collisions.
 */
#define _GNU_SOURCE 1
#include "framework.h"

#include "oxidize/http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

static void http_shutdown(OcHttpServer *srv, int fd)
{
    if (fd >= 0)
        close(fd);
    oc_http_server_stop(srv);
    oc_http_server_join(srv);
}

/* ─── Response formatting ──────────────────────────────────────────────── */

Test(http, format_response_basic)
{
    char buf[1024];
    const char *body = "{\"hello\":\"world\"}";
    size_t n = oc_http_format_response(buf, sizeof(buf), 200,
                                       "application/json", body, strlen(body),
                                       NULL);
    cr_assert(n > 0, "should produce a response");
    cr_assert(strstr(buf, "HTTP/1.1 200 OK") == buf, "starts with status line");
    cr_assert(strstr(buf, "Content-Type: application/json") != NULL,
              "includes content-type");
    cr_assert(strstr(buf, "Content-Length: 17") != NULL, "includes length");
    cr_assert(strstr(buf, "Connection: close") != NULL, "connection close");
    cr_assert(strstr(buf, "Access-Control-Allow-Origin") == NULL,
              "CORS omitted unless extra headers are supplied");
    cr_assert(strstr(buf + n - 17, body) != NULL, "body is at the end");
}

Test(http, format_response_handles_null_body)
{
    char buf[256];
    size_t n = oc_http_format_response(buf, sizeof(buf), 204, NULL, NULL, 0,
                                       NULL);
    cr_assert(n > 0, "should produce a 204 response");
    cr_assert(strstr(buf, "204 No Content") != NULL, "204 status line");
    cr_assert(strstr(buf, "Content-Length: 0") != NULL, "zero length");
}

Test(http, format_response_overflow_returns_zero)
{
    char buf[32];   /* too small for any response */
    size_t n = oc_http_format_response(buf, sizeof(buf), 200, "application/json",
                                       "{}", 2, NULL);
    cr_assert_eq(n, 0u, "overflow returns 0");
}

Test(http, json_bool_skips_string_values_named_like_the_key)
{
    cr_assert(oc_http_json_bool_field(
        "{\"prompt\":\"stream\",\"stream\":true}", "stream", false));
    cr_assert_not(oc_http_json_bool_field(
        "{\"prompt\":\"stream\"}", "stream", false));
}

Test(http, status_line_covers_common_codes)
{
    cr_assert_str_eq(oc_http_status_line(200), "200 OK");
    cr_assert_str_eq(oc_http_status_line(404), "404 Not Found");
    cr_assert_str_eq(oc_http_status_line(413), "413 Payload Too Large");
    cr_assert_str_eq(oc_http_status_line(500), "500 Internal Server Error");
    /* Unknown code falls back to 500. */
    cr_assert_str_eq(oc_http_status_line(999), "500 Internal Server Error");
}

/* ─── Handler for end-to-end tests ─────────────────────────────────────── */

typedef struct TestState {
    int last_status;
    char last_body[256];
    int call_count;
    bool non_stream_callbacks_null;
} TestState;

static void test_handler(const OcHttpRequest *req,
                         int *out_status, const char **out_content_type,
                         const char **out_body, size_t *out_body_len,
                         void *user_data)
{
    TestState *st = (TestState *)user_data;
    st->call_count++;
    if (strcmp(req->path, "/v1/models") == 0 && req->method == OC_HTTP_GET) {
        *out_status = 200;
        *out_content_type = "application/json";
        const char *body = "{\"data\":[{\"id\":\"test-model\"}]}";
        *out_body = strdup(body);
        *out_body_len = strlen(body);
        st->last_status = 200;
    } else if (strcmp(req->path, "/v1/echo") == 0 && req->method == OC_HTTP_POST) {
        *out_status = 200;
        *out_content_type = "application/json";
        /* Echo the body back (NUL-terminated by the parser). */
        *out_body = strdup(req->body);
        *out_body_len = req->content_length;
        st->last_status = 200;
        st->non_stream_callbacks_null = req->stream_write == NULL &&
                                        req->stream_context == NULL;
        size_t cpy = req->content_length < sizeof(st->last_body) - 1
                   ? req->content_length : sizeof(st->last_body) - 1;
        memcpy(st->last_body, req->body, cpy);
        st->last_body[cpy] = '\0';
    } else if (strcmp(req->path, "/v1/chat/completions") == 0 &&
               req->method == OC_HTTP_POST) {
        usleep(200 * 1000);
        if (req->stream_write != NULL)
            (void)req->stream_write(req->stream_context,
                                    "data: token\n\n", 13);
        usleep(1300 * 1000);
        *out_status = 200;
        *out_content_type = "text/event-stream";
        *out_body = strdup("data: [DONE]\n\n");
        *out_body_len = strlen(*out_body);
        st->last_status = 200;
    } else {
        *out_status = 404;
        *out_content_type = "application/json";
        const char *body = "{\"error\":\"not found\"}";
        *out_body = strdup(body);
        *out_body_len = strlen(body);
        st->last_status = 404;
    }
}

/* Helper: connect to the server and send a raw request, read response. */
static char *send_request(uint16_t port, const char *raw, size_t raw_len,
                          size_t *out_resp_len)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    cr_assert_geq(fd, 0, "socket() failed");
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    /* Retry connect briefly — the worker may not have called accept() yet. */
    int connected = 0;
    for (int attempt = 0; attempt < 50; attempt++) {
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            connected = 1;
            break;
        }
        usleep(10 * 1000);   /* 10 ms */
    }
    cr_assert(connected, "connect() failed after retries (port %u)", port);
    ssize_t wr = write(fd, raw, raw_len);
    cr_assert_eq(wr, (ssize_t)raw_len, "write failed");
    shutdown(fd, SHUT_WR);
    char *resp = malloc(8192);
    cr_assert_not_null(resp, "OOM");
    ssize_t rd = read(fd, resp, 8191);
    close(fd);
    cr_assert_geq(rd, 1, "read failed");
    *out_resp_len = (size_t)rd;
    resp[rd] = '\0';
    return resp;
}

Test(http, end_to_end_get_returns_json)
{
    TestState st = {0};
    OcHttpServer srv;
    OcError e = oc_http_server_start("127.0.0.1", 0, 2, test_handler, &st, &srv);
    cr_assert_eq(e, OC_OK, "server should start");
    usleep(50 * 1000);   /* give workers time to enter accept() */
    size_t resp_len;
    const char *req = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    char *resp = send_request(srv.port, req, strlen(req), &resp_len);
    cr_assert_not_null(resp);
    cr_assert(strstr(resp, "200 OK") != NULL, "should be 200");
    cr_assert(strstr(resp, "test-model") != NULL, "body should contain model");
    cr_assert_eq(st.call_count, 1, "handler called once");
    free(resp);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(http, end_to_end_post_body_reaches_handler)
{
    TestState st = {0};
    OcHttpServer srv;
    OcError e = oc_http_server_start("127.0.0.1", 0, 1, test_handler, &st, &srv);
    cr_assert_eq(e, OC_OK, "server should start");
    const char *body = "{\"prompt\":\"hello\"}";
    char req[256];
    int n = snprintf(req, sizeof(req),
        "POST /v1/echo HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
        strlen(body), body);
    size_t resp_len;
    char *resp = send_request(srv.port, req, (size_t)n, &resp_len);
    cr_assert_not_null(resp);
    cr_assert(strstr(resp, "200 OK") != NULL, "should be 200");
    cr_assert(strstr(resp, "hello") != NULL, "echo body should be in response");
    cr_assert_str_eq(st.last_body, body, "handler saw the body intact");
    cr_assert(st.non_stream_callbacks_null,
              "non-stream requests must not expose stream callbacks");
    free(resp);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(http, streaming_response_sends_heartbeats_while_handler_runs)
{
    TestState st = {0};
    OcHttpServer srv;
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 1,
                                      test_handler, &st, &srv), OC_OK);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    cr_assert_geq(fd, 0);
    struct timeval timeout = { .tv_sec = 1, .tv_usec = 200 * 1000 };
    cr_assert_eq(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                            sizeof(timeout)), 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(srv.port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    cr_assert_eq(connect(fd, (struct sockaddr *)&addr, sizeof(addr)), 0);
    const char *body = "{\"stream\":true}";
    char request[256];
    int n = snprintf(request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
        strlen(body), body);
    cr_assert_eq(write(fd, request, (size_t)n), n);
    char response[1024] = {0};
    size_t used = 0;
    size_t heartbeats = 0;
    while (heartbeats < 2 && used < sizeof(response) - 1) {
        ssize_t got = read(fd, response + used, sizeof(response) - used - 1);
        if (got <= 0)
            break;
        used += (size_t)got;
        response[used] = '\0';
        heartbeats = 0;
        const char *p = response;
        while ((p = strstr(p, ": oxidize\n\n")) != NULL) {
            heartbeats++;
            p += strlen(": oxidize\n\n");
        }
    }
    cr_expect(strstr(response, "text/event-stream") != NULL);
    cr_expect(strstr(response, "Access-Control-Allow-Origin") == NULL);
    cr_expect_geq(heartbeats, 2,
                  "stream needs repeated heartbeats during generation");
    http_shutdown(&srv, fd);
}

Test(http, streaming_handler_can_send_data_before_returning)
{
    TestState st = {0};
    OcHttpServer srv;
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 1,
                                      test_handler, &st, &srv), OC_OK);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    cr_assert_geq(fd, 0);
    struct timeval timeout = { .tv_sec = 0, .tv_usec = 500 * 1000 };
    cr_assert_eq(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                            sizeof(timeout)), 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(srv.port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    cr_assert_eq(connect(fd, (struct sockaddr *)&addr, sizeof(addr)), 0);
    const char *body = "{\"stream\":true}";
    char request[256];
    int n = snprintf(request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
        strlen(body), body);
    cr_assert_eq(write(fd, request, (size_t)n), n);
    char response[1024] = {0};
    size_t used = 0;
    while (strstr(response, "data: token\n\n") == NULL &&
           used < sizeof(response) - 1) {
        ssize_t got = read(fd, response + used, sizeof(response) - used - 1);
        if (got <= 0)
            break;
        used += (size_t)got;
        response[used] = '\0';
    }
    cr_expect(strstr(response, "data: token\n\n") != NULL,
              "streamed handler data must arrive before return");
    http_shutdown(&srv, fd);
}

Test(http, reads_body_split_across_packets)
{
    TestState st = {0};
    OcHttpServer srv;
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 1,
                                      test_handler, &st, &srv), OC_OK);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    cr_assert_geq(fd, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(srv.port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    int connected = 0;
    for (int attempt = 0; attempt < 50; attempt++) {
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            connected = 1;
            break;
        }
        usleep(10 * 1000);
    }
    cr_assert(connected, "connect() failed after retries");
    const char *headers =
        "POST /v1/echo HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\n\r\n";
    cr_assert_eq(write(fd, headers, strlen(headers)), (ssize_t)strlen(headers));
    usleep(20 * 1000);
    cr_assert_eq(write(fd, "hello", 5), 5);
    shutdown(fd, SHUT_WR);
    char response[1024] = {0};
    cr_assert_gt(read(fd, response, sizeof(response) - 1), 0);
    cr_assert(strstr(response, "200 OK") != NULL);
    cr_assert_str_eq(st.last_body, "hello");
    close(fd);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(http, options_is_handled_without_dispatch)
{
    TestState st = {0};
    OcHttpServer srv;
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 1,
                                      test_handler, &st, &srv), OC_OK);
    const char *request = "OPTIONS /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    size_t response_len;
    char *response = send_request(srv.port, request, strlen(request), &response_len);
    cr_assert(strstr(response, "204 No Content") != NULL);
    cr_assert_eq(st.call_count, 0);
    free(response);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(http, chunked_requests_are_rejected)
{
    TestState st = {0};
    OcHttpServer srv;
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 1,
                                      test_handler, &st, &srv), OC_OK);
    const char *request =
        "POST /v1/echo HTTP/1.1\r\nHost: localhost\r\n"
        "Transfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
    size_t response_len;
    char *response = send_request(srv.port, request, strlen(request), &response_len);
    cr_assert(strstr(response, "411 Length Required") != NULL);
    cr_assert_eq(st.call_count, 0);
    free(response);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(http, start_rejects_bad_args)
{
    OcHttpServer srv;
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 0, test_handler, NULL, &srv),
                 OC_ERR_INVALID_ARG, "n_threads=0 rejected");
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 1, NULL, NULL, &srv),
                 OC_ERR_INVALID_ARG, "null handler rejected");
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 1, test_handler, NULL, NULL),
                 OC_ERR_INVALID_ARG, "null out rejected");
}

Test(http, start_does_not_read_uninitialized_configuration)
{
    TestState st = {0};
    OcHttpServer srv;
    memset(&srv, 0xa5, sizeof(srv));
    cr_assert_eq(oc_http_server_start("127.0.0.1", 0, 1,
                                      test_handler, &st, &srv), OC_OK);
    cr_assert_null(srv.stream_authorize);
    cr_assert_eq(srv.extra_headers[0], '\0');
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(http, stop_and_join_are_idempotent)
{
    TestState st = {0};
    OcHttpServer srv;
    oc_http_server_start("127.0.0.1", 0, 1, test_handler, &st, &srv);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
    /* Second stop/join should be no-ops. */
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
    cr_assert(true, "no crash on double stop/join");
}

Test(http, unknown_path_returns_404)
{
    TestState st = {0};
    OcHttpServer srv;
    oc_http_server_start("127.0.0.1", 0, 1, test_handler, &st, &srv);
    size_t resp_len;
    char *resp = send_request(srv.port,
        "GET /nonexistent HTTP/1.1\r\nHost: localhost\r\n\r\n",
        strlen("GET /nonexistent HTTP/1.1\r\nHost: localhost\r\n\r\n"), &resp_len);
    cr_assert(strstr(resp, "404 Not Found") != NULL, "should be 404");
    free(resp);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}
