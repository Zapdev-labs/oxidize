/*
 * test_openai.c — OpenAI-compatible route tests.
 *
 * VAL-OPENAI-001..005 cover:
 *   1. GET /v1/models returns 200 + a model list (placeholder when no model).
 *   2. POST /v1/completions returns 503 when no model is loaded.
 *   3. POST /v1/chat/completions returns 503 when no model is loaded.
 *   4. Unknown path returns 404 with a JSON error.
 *   5. oc_openai_error_json produces valid JSON.
 *
 * Full end-to-end generation tests (with a real loaded model) run on the
 * remote NUMA box as part of the cpu-qwen-benchmark-121 feature; here we
 * verify the routing + error contract only.
 */
#define _GNU_SOURCE 1
#include <criterion/criterion.h>

#include "oxidize/http.h"
#include "oxidize/openai.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static char *send_raw(uint16_t port, const char *raw, size_t raw_len,
                      size_t *out_len)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    cr_assert_geq(fd, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    int connected = 0;
    for (int i = 0; i < 50; i++) {
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            connected = 1; break;
        }
        usleep(10 * 1000);
    }
    cr_assert(connected, "connect failed");
    write(fd, raw, raw_len);
    shutdown(fd, SHUT_WR);
    char *resp = malloc(8192);
    ssize_t rd = read(fd, resp, 8192);
    close(fd);
    cr_assert_geq(rd, 1);
    *out_len = (size_t)rd;
    resp[rd] = '\0';
    return resp;
}

/* ─── error_json ────────────────────────────────────────────────────────── */

Test(openai, error_json_is_valid)
{
    char *e = oc_openai_error_json("bad request", "invalid_request_error");
    cr_assert_not_null(e);
    cr_assert(strstr(e, "\"message\":\"bad request\"") != NULL, "has message");
    cr_assert(strstr(e, "\"type\":\"invalid_request_error\"") != NULL, "has type");
    free(e);
}

Test(openai, error_json_default_type)
{
    char *e = oc_openai_error_json("oops", NULL);
    cr_assert_not_null(e);
    cr_assert(strstr(e, "invalid_request_error") != NULL, "default type");
    free(e);
}

/* ─── Routing with no model loaded ─────────────────────────────────────── */

static OcOpenaiState g_state = {0};

static void start_server(OcHttpServer *srv)
{
    OcError e = oc_http_server_start("127.0.0.1", 0, 1, oc_openai_handler,
                                     &g_state, srv);
    cr_assert_eq(e, OC_OK);
    usleep(50 * 1000);
}

Test(openai, list_models_returns_placeholder_when_no_model)
{
    OcHttpServer srv;
    memset(&g_state, 0, sizeof(g_state));
    g_state.model_loaded = false;
    start_server(&srv);
    const char *req = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    size_t len;
    char *resp = send_raw(srv.port, req, strlen(req), &len);
    cr_assert(strstr(resp, "200 OK") != NULL, "should be 200");
    cr_assert(strstr(resp, "placeholder") != NULL, "should list placeholder model");
    free(resp);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(openai, completions_returns_503_when_no_model)
{
    OcHttpServer srv;
    memset(&g_state, 0, sizeof(g_state));
    g_state.model_loaded = false;
    start_server(&srv);
    const char *body = "{\"prompt\":\"hello\",\"max_tokens\":10}";
    char req[256];
    int n = snprintf(req, sizeof(req),
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
        strlen(body), body);
    size_t len;
    char *resp = send_raw(srv.port, req, (size_t)n, &len);
    cr_assert(strstr(resp, "503") != NULL, "should be 503");
    cr_assert(strstr(resp, "no model loaded") != NULL, "error message");
    free(resp);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(openai, chat_completions_returns_503_when_no_model)
{
    OcHttpServer srv;
    memset(&g_state, 0, sizeof(g_state));
    g_state.model_loaded = false;
    start_server(&srv);
    const char *body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    char req[256];
    int n = snprintf(req, sizeof(req),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
        strlen(body), body);
    size_t len;
    char *resp = send_raw(srv.port, req, (size_t)n, &len);
    cr_assert(strstr(resp, "503") != NULL, "should be 503");
    free(resp);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(openai, unknown_path_returns_404)
{
    OcHttpServer srv;
    memset(&g_state, 0, sizeof(g_state));
    start_server(&srv);
    const char *req = "GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n";
    size_t len;
    char *resp = send_raw(srv.port, req, strlen(req), &len);
    cr_assert(strstr(resp, "404") != NULL, "should be 404");
    cr_assert(strstr(resp, "not found") != NULL, "error message");
    free(resp);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}

Test(openai, completions_missing_prompt_returns_400)
{
    /* Even with no model, a missing 'prompt' field is a 400 (client error)
     * before the 503 model check. */
    OcHttpServer srv;
    memset(&g_state, 0, sizeof(g_state));
    g_state.model_loaded = true;   /* skip the 503 path to reach the 400 */
    g_state.model_id = "test";
    start_server(&srv);
    const char *body = "{\"max_tokens\":10}";
    char req[256];
    int n = snprintf(req, sizeof(req),
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
        strlen(body), body);
    size_t len;
    char *resp = send_raw(srv.port, req, (size_t)n, &len);
    cr_assert(strstr(resp, "400") != NULL, "should be 400");
    cr_assert(strstr(resp, "missing") != NULL, "error mentions missing field");
    free(resp);
    oc_http_server_stop(&srv);
    oc_http_server_join(&srv);
}
