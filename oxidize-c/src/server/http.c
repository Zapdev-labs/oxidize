/*
 * http.c — minimal dependency-free HTTP/1.1 server.
 *
 * Uses POSIX sockets (libc + libpthread only). Each worker thread runs
 * accept() in a loop, reads one request, dispatches to the handler, writes
 * the response, and closes the connection (Connection: close — no
 * keep-alive). This is sufficient for single-client dev use and the
 * `oxidize run`/`--serve-api` workflow.
 *
 * Request parsing: reads up to OC_HTTP_MAX_REQUEST_BYTES, locates the
 * "\r\n\r\n" header/body separator, parses the request line, extracts
 * Content-Length, and NUL-terminates the body. Does not support chunked
 * transfer-encoding (rejects with 411).
 */
#define _GNU_SOURCE 1   /* accept4, SO_REUSEPORT on Linux */
#include "oxidize/http.h"

#include "oxidize/log.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

/* ─── Status lines ────────────────────────────────────────────────────── */

const char *oc_http_status_line(int status)
{
    switch (status) {
    case 200: return "200 OK";
    case 201: return "201 Created";
    case 204: return "204 No Content";
    case 400: return "400 Bad Request";
    case 401: return "401 Unauthorized";
    case 403: return "403 Forbidden";
    case 404: return "404 Not Found";
    case 405: return "405 Method Not Allowed";
    case 408: return "408 Request Timeout";
    case 411: return "411 Length Required";
    case 413: return "413 Payload Too Large";
    case 422: return "422 Unprocessable Entity";
    case 429: return "429 Too Many Requests";
    case 500: return "500 Internal Server Error";
    case 501: return "501 Not Implemented";
    case 503: return "503 Service Unavailable";
    default:  return "500 Internal Server Error";
    }
}

/* ─── Response formatting ──────────────────────────────────────────────── */

size_t oc_http_format_response(char *buf, size_t cap,
                               int status, const char *content_type,
                               const char *body, size_t body_len)
{
    if (content_type == NULL) content_type = "application/json";
    if (body == NULL) body_len = 0;
    int n = snprintf(buf, cap,
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "\r\n",
        oc_http_status_line(status), content_type, body_len);
    if (n < 0 || (size_t)n >= cap) return 0;
    size_t header_len = (size_t)n;
    if (header_len + body_len > cap) return 0;
    if (body_len > 0) memcpy(buf + header_len, body, body_len);
    return header_len + body_len;
}

/* ─── Request parsing ────────────────────────────────────────────────────
 *
 * Returns true on success; fills `req`. On failure, returns false and sets
 * `*out_status` to the appropriate HTTP error code (400, 411, 413). */

static bool parse_request(char *buf, size_t buf_len, OcHttpRequest *req,
                          int *out_status)
{
    *out_status = 400;
    /* Find the header/body separator. */
    char *sep = memmem(buf, buf_len, "\r\n\r\n", 4);
    if (sep == NULL) {
        *out_status = 411;   /* no separator → need more / Length required */
        return false;
    }
    *sep = '\0';   /* terminate the header block for ease of parsing */
    size_t header_len = (size_t)(sep - buf);
    /* Parse the request line: METHOD /path?query HTTP/1.1 */
    char *line_end = strstr(buf, "\r\n");
    if (line_end == NULL) return false;
    *line_end = '\0';
    char *method = buf;
    char *sp1 = strchr(method, ' ');
    if (sp1 == NULL) return false;
    *sp1 = '\0';
    char *path = sp1 + 1;
    char *sp2 = strchr(path, ' ');
    if (sp2 == NULL) return false;
    *sp2 = '\0';

    if (strcmp(method, "GET") == 0)       req->method = OC_HTTP_GET;
    else if (strcmp(method, "POST") == 0) req->method = OC_HTTP_POST;
    else if (strcmp(method, "OPTIONS") == 0) req->method = OC_HTTP_OPTIONS;
    else                                   req->method = OC_HTTP_OTHER;

    /* Split path / query. */
    char *q = strchr(path, '?');
    if (q != NULL) {
        *q = '\0';
        req->query = q + 1;
    } else {
        req->query = NULL;
    }
    req->path = path;

    char *te = strstr(line_end + 2, "Transfer-Encoding:");
    if (te == NULL) te = strstr(line_end + 2, "transfer-encoding:");
    if (te != NULL && (strstr(te, "chunked") != NULL ||
                       strstr(te, "Chunked") != NULL)) {
        *out_status = 411;
        return false;
    }

    /* Find Content-Length. */
    char *cl = strstr(line_end + 2, "Content-Length:");
    if (cl == NULL) cl = strstr(line_end + 2, "content-length:");
    size_t content_length = 0;
    if (cl != NULL) {
        cl += strlen("Content-Length:");
        while (*cl == ' ' || *cl == '\t') cl++;
        content_length = (size_t)strtoull(cl, NULL, 10);
    }

    /* Body starts after the "\r\n\r\n" separator. */
    char *body = sep + 4;
    size_t body_avail = buf_len - header_len - 4;
    if (content_length > body_avail) {
        *out_status = 400;   /* truncated body */
        return false;
    }
    body[content_length] = '\0';   /* NUL-terminate for JSON parsing */
    req->body = body;
    req->content_length = content_length;
    *out_status = 200;
    return true;
}

/* ─── Worker thread ────────────────────────────────────────────────────── */

typedef struct OcWorkerCtx {
    OcHttpServer *srv;
} OcWorkerCtx;

static void send_simple_response(int fd, int status, const char *body)
{
    char hdr[512];
    size_t body_len = body ? strlen(body) : 0;
    size_t n = oc_http_format_response(hdr, sizeof(hdr), status,
                                       "application/json", body, body_len);
    if (n == 0) return;
    size_t sent = 0;
    while (sent < n) {
#ifdef MSG_NOSIGNAL
        ssize_t wr = send(fd, hdr + sent, n - sent, MSG_NOSIGNAL);
#else
        ssize_t wr = send(fd, hdr + sent, n - sent, 0);
#endif
        if (wr > 0) {
            sent += (size_t)wr;
        } else if (wr < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
}

static size_t expected_request_size(char *buf, size_t len)
{
    char *sep = memmem(buf, len, "\r\n\r\n", 4);
    if (sep == NULL) return 0;
    size_t header_bytes = (size_t)(sep - buf) + 4;
    char saved = *sep;
    *sep = '\0';
    char *cl = strstr(buf, "Content-Length:");
    if (cl == NULL) cl = strstr(buf, "content-length:");
    unsigned long long body_bytes = 0;
    errno = 0;
    if (cl != NULL) {
        cl += strlen("Content-Length:");
        while (*cl == ' ' || *cl == '\t') cl++;
        body_bytes = strtoull(cl, NULL, 10);
    }
    *sep = saved;
    if (errno == ERANGE || body_bytes > OC_HTTP_MAX_REQUEST_BYTES - header_bytes)
        return SIZE_MAX;
    return header_bytes + (size_t)body_bytes;
}

static bool send_all(int fd, const char *buf, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
#ifdef MSG_NOSIGNAL
        ssize_t wr = send(fd, buf + sent, len - sent, MSG_NOSIGNAL);
#else
        ssize_t wr = send(fd, buf + sent, len - sent, 0);
#endif
        if (wr > 0) sent += (size_t)wr;
        else if (wr < 0 && errno == EINTR) continue;
        else return false;
    }
    return true;
}

static void *worker_main(void *arg)
{
    OcWorkerCtx *wc = (OcWorkerCtx *)arg;
    OcHttpServer *srv = wc->srv;
    char *buf = malloc(OC_HTTP_MAX_REQUEST_BYTES + 1);
    if (buf == NULL) {
        oc_log(OC_LOG_ERROR, "http: worker OOM, exiting");
        return NULL;
    }
    while (!atomic_load_explicit(&srv->stop,
                                memory_order_acquire)) {
        int fd = accept(srv->listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (atomic_load_explicit(&srv->stop,
                                     memory_order_acquire)) break;
            oc_log(OC_LOG_WARN, "http: accept failed: %s", strerror(errno));
            continue;
        }
        struct timeval read_timeout = { .tv_sec = 5, .tv_usec = 0 };
        (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                         &read_timeout, sizeof(read_timeout));
#ifdef SO_NOSIGPIPE
        int no_sigpipe = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE,
                         &no_sigpipe, sizeof(no_sigpipe));
#endif
        /* Read until we have the full request or hit the cap. */
        size_t total = 0;
        ssize_t rd;
        bool too_large = false;
        bool timed_out = false;
        while (total < OC_HTTP_MAX_REQUEST_BYTES) {
            rd = read(fd, buf + total, OC_HTTP_MAX_REQUEST_BYTES - total);
            if (rd <= 0) {
                timed_out = rd < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
                break;
            }
            total += (size_t)rd;
            buf[total] = '\0';
            size_t expected = expected_request_size(buf, total);
            if (expected == SIZE_MAX) { too_large = true; break; }
            if (expected > 0 && total >= expected) break;
        }
        if (total == 0) {
            close(fd);
            continue;
        }
        if (too_large || total >= OC_HTTP_MAX_REQUEST_BYTES) {
            send_simple_response(fd, 413, "{\"error\":\"payload too large\"}");
            close(fd);
            continue;
        }
        if (timed_out) {
            send_simple_response(fd, 408, "{\"error\":\"request timeout\"}");
            close(fd);
            continue;
        }
        buf[total] = '\0';

        OcHttpRequest req;
        int parse_status = 400;
        if (!parse_request(buf, total, &req, &parse_status)) {
            send_simple_response(fd, parse_status,
                                 "{\"error\":\"malformed request\"}");
            close(fd);
            continue;
        }

        /* Handle CORS preflight (OPTIONS) inline. */
        if (req.method == OC_HTTP_OPTIONS) {
            send_simple_response(fd, 204, "");
            close(fd);
            continue;
        }

        /* Dispatch to caller. */
        int status = 404;
        const char *ctype = NULL;
        const char *body = NULL;
        size_t body_len = 0;
        srv->handler(&req, &status, &ctype, &body, &body_len, srv->user_data);

        char resp[8192];
        size_t resp_len = oc_http_format_response(resp, sizeof(resp), status,
                                                  ctype, body, body_len);
        if (resp_len > 0) {
            (void)send_all(fd, resp, resp_len);
        } else if (body_len > 0) {
            /* Body too big for the inline buffer — fall back to a malloc'd
             * response. */
            size_t cap = body_len + 512;
            char *big = malloc(cap);
            if (big != NULL) {
                size_t n = oc_http_format_response(big, cap, status, ctype,
                                                  body, body_len);
                if (n > 0) {
                    (void)send_all(fd, big, n);
                }
                free(big);
            }
        }
        /* Free the handler-returned body. Handlers malloc their JSON
         * responses (see oc_openai_handler); the server core owns freeing
         * them after the response is written. String literals (used by
         * some test handlers) are also safe to pass to free() per C11
         * because free(NULL) is a no-op and we never return a literal
         * when body_len > 0 in production code — but to be safe we only
         * free when the handler indicated a non-empty malloc'd body.
         * NOTE: handlers that return string literals with body_len > 0
         * MUST instead return a malloc'd copy (oc_openai_handler does). */
        if (body != NULL && body_len > 0) {
            free((void *)body);
        }
        close(fd);
    }
    free(buf);
    return NULL;
}

/* ─── Server lifecycle ────────────────────────────────────────────────── */

OcError oc_http_server_start(const char *host, uint16_t port, size_t n_threads,
                             OcHttpHandler handler, void *user_data,
                             OcHttpServer *out)
{
    if (handler == NULL || out == NULL || n_threads == 0) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->handler   = handler;
    out->user_data = user_data;
    out->n_threads = n_threads;
    out->stop      = false;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return OC_ERR_NETWORK;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#ifdef SO_REUSEPORT
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (host == NULL || strcmp(host, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
            close(fd);
            return OC_ERR_NETWORK;
        }
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        oc_log(OC_LOG_ERROR, "http: bind %s:%u failed: %s",
               host ? host : "0.0.0.0", port, strerror(errno));
        close(fd);
        return OC_ERR_NETWORK;
    }
    if (listen(fd, 64) < 0) {
        close(fd);
        return OC_ERR_NETWORK;
    }
    out->listen_fd = fd;
    out->port = port;

    /* Get the actual bound port (if port==0 was requested). */
    socklen_t alen = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &alen) == 0) {
        out->port = ntohs(addr.sin_port);
    }

    out->threads = calloc(n_threads, sizeof(pthread_t));
    if (out->threads == NULL) {
        close(fd);
        return OC_ERR_OOM;
    }
    static OcWorkerCtx ctx;   /* single server per process for now */
    ctx.srv = out;
    for (size_t i = 0; i < n_threads; i++) {
        if (pthread_create(&out->threads[i], NULL, worker_main, &ctx) != 0) {
            out->n_threads = i;   /* only join the ones we started */
            return OC_ERR_NETWORK;
        }
    }
    oc_log(OC_LOG_INFO, "http: server listening on %s:%u (%zu threads)",
           host ? host : "0.0.0.0", out->port, n_threads);
    return OC_OK;
}

OcError oc_http_server_join(OcHttpServer *s)
{
    if (s == NULL) return OC_ERR_INVALID_ARG;
    if (s->joined) return OC_OK;   /* idempotent */
    for (size_t i = 0; i < s->n_threads; i++) {
        pthread_join(s->threads[i], NULL);
    }
    s->joined = true;
    free(s->threads);
    s->threads = NULL;
    return OC_OK;
}

OcError oc_http_server_stop(OcHttpServer *s)
{
    if (s == NULL) return OC_ERR_INVALID_ARG;
    atomic_store_explicit(&s->stop, true, memory_order_release);
    if (s->listen_fd >= 0) {
        shutdown(s->listen_fd, SHUT_RDWR);
        close(s->listen_fd);
        s->listen_fd = -1;
    }
    return OC_OK;
}
