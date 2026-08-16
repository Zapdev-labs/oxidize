/*
 * http.h — minimal dependency-free HTTP/1.1 server.
 *
 * Implements the `server-http-core` feature: a thread-pooled TCP server that
 * parses HTTP/1.1 requests and dispatches them to a caller-provided handler.
 * Uses only libc + libpthread (no external HTTP library), matching the
 * project's dependency-free convention.
 *
 * Scope: enough to serve the OpenAI-compatible routes (POST JSON, GET
 * /v1/models). NOT a general-purpose HTTP server — no keep-alive pipelining,
 * no chunked transfer-encoding (responses are sent with Content-Length and
 * Connection: close). Sufficient for single-client dev use and the
 * `oxidize run`/`--serve-api` workflow.
 *
 * Thread model: a fixed-size worker pool accepts connections from a listening
 * socket. Each request is handled synchronously in the worker thread. The
 * caller's handler receives the parsed method, path, headers, and body, and
 * writes the response status + JSON body.
 */
#ifndef OXIDIZE_HTTP_H
#define OXIDIZE_HTTP_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum supported request size (header + body). Requests larger than this
 * are rejected with 413 Payload Too Large. 16 MiB covers the largest OpenAI
 * request we expect (a long prompt + image). */
#define OC_HTTP_MAX_REQUEST_BYTES (16u * 1024u * 1024u)

typedef enum {
    OC_HTTP_GET,
    OC_HTTP_POST,
    OC_HTTP_OPTIONS,
    OC_HTTP_OTHER,
} OcHttpMethod;

typedef bool (*OcHttpStreamWrite)(void *context, const char *data, size_t len);

/* A parsed HTTP request. `body` is NUL-terminated for convenience (the
 * additional NUL is appended beyond content_length). All string fields
 * alias into the request buffer owned by the server (valid for the duration
 * of the handler call only). */
typedef struct OcHttpRequest {
    OcHttpMethod method;
    const char *path;             /* e.g. "/v1/chat/completions"            */
    const char *query;            /* after '?' or NULL                       */
    const char *body;             /* NUL-terminated request body             */
    size_t      content_length;
    /* Authorization header value ("Bearer sk-..."), or NULL when absent.
     * Points into the request buffer — do not retain past the handler. */
    const char *auth_header;
    /* Peer address as a numeric string ("127.0.0.1"), or NULL if it could
     * not be determined. Points at worker-thread stack storage. */
    const char *client_ip;
    OcHttpStreamWrite stream_write;
    void             *stream_context;
} OcHttpRequest;

/* Caller-provided handler. Writes the response status code, Content-Type
 * (defaults to application/json), and JSON body. `body` may be NULL for an
 * empty 204 response. The handler must NOT retain pointers into `req` after
 * returning (the buffer is reused for the next request). */
typedef void (*OcHttpHandler)(const OcHttpRequest *req,
                              int *out_status,
                              const char **out_content_type,
                              const char **out_body,
                              size_t *out_body_len,
                              void *user_data);

typedef bool (*OcHttpStreamAuthorize)(const OcHttpRequest *req,
                                      int *out_status,
                                      const char **out_body,
                                      void *user_data);

typedef struct OcHttpServer {
    int          listen_fd;
    uint16_t     port;
    size_t       n_threads;
    pthread_t   *threads;       /* worker pool                            */
    OcHttpHandler handler;
    OcHttpStreamAuthorize stream_authorize;
    void        *user_data;
    char         extra_headers[384];
    _Atomic bool stop;           /* set by oc_http_server_stop              */
    bool         joined;        /* true after oc_http_server_join          */
} OcHttpServer;

/* Start a server bound to `host`:`port` with `n_threads` worker threads.
 * `host` may be NULL (binds to 0.0.0.0) or "127.0.0.1". Returns OC_OK and
 * fills `*out`, or OC_ERR_NETWORK (bind/listen failed), OC_ERR_OOM, or
 * OC_ERR_INVALID_ARG. The server runs until oc_http_server_stop(). */
OcError oc_http_server_start(const char *host, uint16_t port, size_t n_threads,
                             OcHttpHandler handler, void *user_data,
                             OcHttpServer *out);

void oc_http_server_set_stream_authorizer(OcHttpServer *s,
                                          OcHttpStreamAuthorize authorize);

/* Extra response headers (typically CORS). Empty string omits them.
 * Configure a zeroed OcHttpServer, then start; start preserves this field
 * and stream_authorize so workers never race with setup. Returns
 * OC_ERR_INVALID_ARG if the block does not fit (no silent truncation). */
OcError oc_http_server_set_extra_headers(OcHttpServer *s, const char *headers);

/* Block until the server stops (or returns immediately if already stopped).
 * Internally joins all worker threads. */
OcError oc_http_server_join(OcHttpServer *s);

/* Signal the server to stop accepting new connections and shut down. Safe
 * to call from any thread. */
OcError oc_http_server_stop(OcHttpServer *s);

/* Build a canonical HTTP/1.1 response string for `status` + `body`. Writes
 * into `buf` (cap bytes); returns the bytes written (0 on overflow). The
 * caller owns `buf`. Used internally by the worker; exposed for tests. */
size_t oc_http_format_response(char *buf, size_t cap,
                               int status, const char *content_type,
                               const char *body, size_t body_len,
                               const char *extra_headers);

/* Read a boolean at the top level of a JSON object. Keys inside strings or
 * nested values are ignored. Returns `def` when the key is absent or is not
 * a JSON boolean. */
bool oc_http_json_bool_field(const char *json, const char *key, bool def);

/* Human-readable status line for a code ("200 OK", "404 Not Found", ...). */
const char *oc_http_status_line(int status);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_HTTP_H */
