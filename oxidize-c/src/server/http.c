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
#include <poll.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
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
                               const char *body, size_t body_len,
                               const char *extra_headers)
{
    if (content_type == NULL) content_type = "application/json";
    if (body == NULL) body_len = 0;
    if (extra_headers == NULL) extra_headers = "";
    int n = snprintf(buf, cap,
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "%s"
        "\r\n",
        oc_http_status_line(status), content_type, body_len, extra_headers);
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

typedef struct HeaderValue {
    char *data;
    size_t len;
} HeaderValue;

static HeaderValue find_header(char *start, char *end, const char *name)
{
    size_t name_len = strlen(name);
    char *line = start;
    while (line < end) {
        char *line_end = memmem(line, (size_t)(end - line), "\r\n", 2);
        if (!line_end) line_end = end;
        if ((size_t)(line_end - line) > name_len && line[name_len] == ':' &&
            strncasecmp(line, name, name_len) == 0) {
            char *value = line + name_len + 1;
            while (value < line_end && (*value == ' ' || *value == '\t')) value++;
            return (HeaderValue){value, (size_t)(line_end - value)};
        }
        line = line_end < end ? line_end + 2 : end;
    }
    return (HeaderValue){NULL, 0};
}

static bool parse_request(char *buf, size_t buf_len, OcHttpRequest *req,
                          int *out_status)
{
    *out_status = 400;
    req->stream_write = NULL;
    req->stream_context = NULL;
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

    HeaderValue te = find_header(line_end + 2, sep, "Transfer-Encoding");
    if (te.data) {
        *out_status = 411;
        return false;
    }

    /* Find Content-Length. */
    HeaderValue cl = find_header(line_end + 2, sep, "Content-Length");
    size_t content_length = 0;
    if (cl.data != NULL) {
        content_length = (size_t)strtoull(cl.data, NULL, 10);
    }

    /* Authorization, for the auth middleware. Extracted LAST among the
     * headers: terminating the value in place clobbers that line's '\r', so
     * find_header's line walk must already be done. */
    HeaderValue auth = find_header(line_end + 2, sep, "Authorization");
    if (auth.data != NULL) {
        auth.data[auth.len] = '\0';
        req->auth_header = auth.data;
    } else {
        req->auth_header = NULL;
    }
    req->client_ip = NULL;   /* filled in by the worker from the peer addr */

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

bool oc_http_json_bool_field(const char *json, const char *key, bool def)
{
    size_t key_len = strlen(key);
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (const char *p = json; *p; p++) {
        if (in_string) {
            if (escaped) escaped = false;
            else if (*p == '\\') escaped = true;
            else if (*p == '"') in_string = false;
            continue;
        }
        if (*p == '{' || *p == '[') { depth++; continue; }
        if (*p == '}' || *p == ']') { if (depth > 0) depth--; continue; }
        if (*p != '"') continue;
        if (depth == 1 && strncmp(p + 1, key, key_len) == 0 &&
            p[1 + key_len] == '"') {
            const char *value = p + key_len + 2;
            while (*value == ' ' || *value == '\t' || *value == '\r' || *value == '\n') value++;
            if (*value++ != ':') {
                in_string = true;
                continue;
            }
            while (*value == ' ' || *value == '\t' || *value == '\r' || *value == '\n') value++;
            if (strncmp(value, "true", 4) == 0) return true;
            if (strncmp(value, "false", 5) == 0) return false;
            return def;
        }
        in_string = true;
    }
    return def;
}

static bool request_wants_completion_stream(const OcHttpRequest *req)
{
    if (req->method != OC_HTTP_POST || req->body == NULL) return false;
    if (strcmp(req->path, "/v1/completions") != 0 &&
        strcmp(req->path, "/v1/chat/completions") != 0)
        return false;
    return oc_http_json_bool_field(req->body, "stream", false);
}

/* ─── Worker thread ────────────────────────────────────────────────────── */

typedef struct OcWorkerCtx {
    OcHttpServer *srv;
} OcWorkerCtx;

static const char *server_extra(const OcHttpServer *srv)
{
    if (srv == NULL || srv->extra_headers[0] == '\0') return "";
    return srv->extra_headers;
}

static void send_simple_response(OcHttpServer *srv, int fd, int status,
                                 const char *body)
{
    char hdr[768];
    size_t body_len = body ? strlen(body) : 0;
    size_t n = oc_http_format_response(hdr, sizeof(hdr), status,
                                       "application/json", body, body_len,
                                       server_extra(srv));
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
    char *line = memmem(buf, (size_t)(sep - buf), "\r\n", 2);
    if (line != NULL) line += 2;
    HeaderValue cl = line ? find_header(line, sep, "Content-Length") :
                            (HeaderValue){NULL, 0};
    unsigned long long body_bytes = 0;
    errno = 0;
    if (cl.data != NULL) {
        body_bytes = strtoull(cl.data, NULL, 10);
    }
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

typedef struct OcStreamHeartbeat {
    int fd;
    pthread_mutex_t write_mutex;
    atomic_bool stop;
    atomic_bool disconnected;
} OcStreamHeartbeat;

static bool stream_write(void *context, const char *data, size_t len)
{
    OcStreamHeartbeat *heartbeat = context;
    pthread_mutex_lock(&heartbeat->write_mutex);
    bool sent = !atomic_load_explicit(&heartbeat->disconnected,
                                      memory_order_acquire) &&
                send_all(heartbeat->fd, data, len);
    if (!sent)
        atomic_store_explicit(&heartbeat->disconnected, true,
                              memory_order_release);
    pthread_mutex_unlock(&heartbeat->write_mutex);
    return sent;
}

static void *stream_heartbeat_main(void *arg)
{
    OcStreamHeartbeat *heartbeat = arg;
    static const char event[] = ": oxidize\n\n";
    const struct timespec interval = { .tv_sec = 0, .tv_nsec = 100000000L };
    for (;;) {
        for (int tick = 0; tick < 10; tick++) {
            if (atomic_load_explicit(&heartbeat->stop, memory_order_acquire))
                return NULL;
            (void)nanosleep(&interval, NULL);
        }
        if (!stream_write(heartbeat, event, sizeof(event) - 1u)) {
            return NULL;
        }
    }
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
        struct sockaddr_storage peer;
        socklen_t peer_len = sizeof(peer);
        char peer_ip[INET6_ADDRSTRLEN] = {0};
        int fd = accept(srv->listen_fd, (struct sockaddr *)&peer, &peer_len);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (atomic_load_explicit(&srv->stop,
                                     memory_order_acquire)) break;
            oc_log(OC_LOG_WARN, "http: accept failed: %s", strerror(errno));
            continue;
        }
        /* Numeric peer address, for the rate limiter's per-IP buckets and the
         * audit log. Best-effort: an unsupported family leaves it empty. */
        if (peer.ss_family == AF_INET) {
            (void)inet_ntop(AF_INET, &((struct sockaddr_in *)&peer)->sin_addr,
                            peer_ip, sizeof(peer_ip));
        } else if (peer.ss_family == AF_INET6) {
            (void)inet_ntop(AF_INET6, &((struct sockaddr_in6 *)&peer)->sin6_addr,
                            peer_ip, sizeof(peer_ip));
        }
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
        struct timespec deadline;
        (void)clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += 5;
        while (total < OC_HTTP_MAX_REQUEST_BYTES) {
            struct timespec now;
            (void)clock_gettime(CLOCK_MONOTONIC, &now);
            int64_t remaining_ms = (int64_t)(deadline.tv_sec - now.tv_sec) * 1000 +
                                   (deadline.tv_nsec - now.tv_nsec) / 1000000;
            if (remaining_ms <= 0) { timed_out = true; break; }
            struct pollfd pfd = { .fd = fd, .events = POLLIN };
            int ready;
            do {
                ready = poll(&pfd, 1, remaining_ms > INT32_MAX ? INT32_MAX : (int)remaining_ms);
            } while (ready < 0 && errno == EINTR);
            if (ready == 0) { timed_out = true; break; }
            if (ready < 0) break;
            rd = read(fd, buf + total, OC_HTTP_MAX_REQUEST_BYTES - total);
            if (rd <= 0) {
                timed_out = rd < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
                break;
            }
            total += (size_t)rd;
            (void)clock_gettime(CLOCK_MONOTONIC, &now);
            if (now.tv_sec > deadline.tv_sec ||
                (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec)) {
                timed_out = true;
                break;
            }
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
            send_simple_response(srv, fd, 413, "{\"error\":\"payload too large\"}");
            close(fd);
            continue;
        }
        if (timed_out) {
            send_simple_response(srv, fd, 408, "{\"error\":\"request timeout\"}");
            close(fd);
            continue;
        }
        buf[total] = '\0';

        OcHttpRequest req;
        int parse_status = 400;
        if (!parse_request(buf, total, &req, &parse_status)) {
            send_simple_response(srv, fd, parse_status,
                                 "{\"error\":\"malformed request\"}");
            close(fd);
            continue;
        }
        req.client_ip = peer_ip[0] != '\0' ? peer_ip : NULL;

        /* Handle CORS preflight (OPTIONS) inline. */
        if (req.method == OC_HTTP_OPTIONS) {
            send_simple_response(srv, fd, 204, "");
            close(fd);
            continue;
        }

        const bool stream_started = request_wants_completion_stream(&req);
        if (stream_started && srv->stream_authorize != NULL) {
            int status = 400;
            const char *body = NULL;
            if (!srv->stream_authorize(&req, &status, &body, srv->user_data)) {
                send_simple_response(srv, fd, status, body);
                free((void *)body);
                close(fd);
                continue;
            }
        }
        OcStreamHeartbeat heartbeat = { .fd = fd };
        pthread_t heartbeat_thread;
        bool heartbeat_running = false;
        if (stream_started) {
            char handshake[768];
            int hn = snprintf(handshake, sizeof(handshake),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: close\r\n"
                "%s"
                "\r\n"
                ": oxidize\n\n",
                server_extra(srv));
            atomic_init(&heartbeat.stop, false);
            atomic_init(&heartbeat.disconnected, false);
            pthread_mutex_init(&heartbeat.write_mutex, NULL);
            if (hn < 0 || (size_t)hn >= sizeof(handshake) ||
                !stream_write(&heartbeat, handshake, (size_t)hn)) {
                pthread_mutex_destroy(&heartbeat.write_mutex);
                close(fd);
                continue;
            }
            req.stream_write = stream_write;
            req.stream_context = &heartbeat;
            heartbeat_running =
                pthread_create(&heartbeat_thread, NULL, stream_heartbeat_main,
                               &heartbeat) == 0;
        }

        int status = 404;
        const char *ctype = NULL;
        const char *body = NULL;
        size_t body_len = 0;
        srv->handler(&req, &status, &ctype, &body, &body_len, srv->user_data);
        if (heartbeat_running) {
            atomic_store_explicit(&heartbeat.stop, true, memory_order_release);
            pthread_join(heartbeat_thread, NULL);
        }

        char resp[8192];
        size_t resp_len = 0;
        if (stream_started) {
            const bool disconnected =
                atomic_load_explicit(&heartbeat.disconnected,
                                     memory_order_acquire);
            if (!disconnected && status == 200 && body != NULL && body_len > 0)
                (void)stream_write(&heartbeat, body, body_len);
        } else {
            resp_len = oc_http_format_response(resp, sizeof(resp), status,
                                               ctype, body, body_len,
                                               server_extra(srv));
        }
        if (!stream_started && resp_len > 0) {
            (void)send_all(fd, resp, resp_len);
        } else if (!stream_started && body_len > 0) {
            /* Body too big for the inline buffer — fall back to a malloc'd
             * response. */
            size_t cap = body_len + 512;
            char *big = malloc(cap);
            if (big != NULL) {
                size_t n = oc_http_format_response(big, cap, status, ctype,
                                                  body, body_len,
                                                  server_extra(srv));
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
        if (stream_started)
            pthread_mutex_destroy(&heartbeat.write_mutex);
        close(fd);
    }
    free(buf);
    return NULL;
}

/* ─── Server lifecycle ────────────────────────────────────────────────── */

static OcError http_server_start(const char *host, uint16_t port,
                                 size_t n_threads, OcHttpHandler handler,
                                 void *user_data, OcHttpServer *out,
                                 bool preserve_config)
{
    if (handler == NULL || out == NULL || n_threads == 0) return OC_ERR_INVALID_ARG;
    char extra_headers[sizeof(out->extra_headers)];
    OcHttpStreamAuthorize stream_authorize = NULL;
    if (preserve_config) {
        stream_authorize = out->stream_authorize;
        memcpy(extra_headers, out->extra_headers, sizeof(extra_headers));
    }
    memset(out, 0, sizeof(*out));
    if (preserve_config) {
        memcpy(out->extra_headers, extra_headers, sizeof(extra_headers));
        out->stream_authorize = stream_authorize;
    }
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
            out->n_threads = i;
            oc_http_server_stop(out);
            if (i > 0) {
                oc_http_server_join(out);
            } else {
                free(out->threads);
                out->threads = NULL;
            }
            return OC_ERR_NETWORK;
        }
    }
    oc_log(OC_LOG_INFO, "http: server listening on %s:%u (%zu threads)",
           host ? host : "0.0.0.0", out->port, n_threads);
    return OC_OK;
}

OcError oc_http_server_start(const char *host, uint16_t port, size_t n_threads,
                             OcHttpHandler handler, void *user_data,
                             OcHttpServer *out)
{
    return http_server_start(host, port, n_threads, handler, user_data, out,
                             false);
}

OcError oc_http_server_start_configured(const char *host, uint16_t port,
                                        size_t n_threads,
                                        OcHttpHandler handler,
                                        void *user_data, OcHttpServer *out)
{
    return http_server_start(host, port, n_threads, handler, user_data, out,
                             true);
}

void oc_http_server_set_stream_authorizer(OcHttpServer *s,
                                          OcHttpStreamAuthorize authorize)
{
    if (s != NULL) s->stream_authorize = authorize;
}

OcError oc_http_server_set_extra_headers(OcHttpServer *s, const char *headers)
{
    if (s == NULL) return OC_ERR_INVALID_ARG;
    if (headers == NULL) headers = "";
    size_t n = strlen(headers);
    if (n >= sizeof(s->extra_headers)) return OC_ERR_INVALID_ARG;
    memcpy(s->extra_headers, headers, n + 1);
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
