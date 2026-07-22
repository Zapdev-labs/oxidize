/*
 * sse.h — Server-Sent Events (SSE) for the C port.
 *
 * Implements the `server-sse` feature: the text/event-stream wire format
 * used to push incremental updates (token streaming, progress events) to
 * HTTP clients. Built on top of server-http-core: a client is an accepted
 * HTTP connection that has been upgraded to an event stream by writing the
 * SSE content-type and then a sequence of "event: ...\ndata: ...\n\n"
 * blocks.
 *
 * Wire format (per the HTML spec, https://html.spec.whatwg.org/#server-sent-events):
 *   event: <name>\r\n
 *   data: <data>\r\n
 *   id: <id>\r\n
 *   \r\n
 * We emit CRLF line endings per spec but accept both CRLF and LF on parse.
 * Multi-line `data` is emitted as multiple `data:` lines; the client
 * reassembles them with "\n" joins.
 *
 * Concurrency: the OcSseServer is NOT thread-safe. Callers that broadcast
 * from multiple threads must serialize externally.
 */
#ifndef OXIDIZE_SSE_H
#define OXIDIZE_SSE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of concurrent SSE clients a server will track. Sized for
 * single-instance dev use; production deployments would shard by fd-set. */
#define OC_SSE_MAX_CLIENTS 64

/* Maximum length of a single SSE event name (including NUL). */
#define OC_SSE_MAX_EVENT_NAME 64

/* Maximum length of an SSE event id (including NUL). */
#define OC_SSE_MAX_EVENT_ID 64

/* Maximum formatted size of a single SSE event block. Sized to fit a
 * generous token delta + metadata; callers streaming very large JSON
 * objects per event should chunk them. */
#define OC_SSE_MAX_EVENT_BYTES 8192

/* A single SSE event. `event` and `id` may be NULL/empty (omitted on the
 * wire). `data` is the payload; if it contains newlines, each line is
 * emitted as its own `data:` line per the spec. None of the pointers are
 * owned; the caller must keep them alive for the duration of the format /
 * send call. */
typedef struct OcSseEvent {
    const char *event;   /* may be NULL or "" → no event: line       */
    const char *data;    /* may be NULL or "" → no data: line        */
    const char *id;      /* may be NULL or "" → no id: line          */
} OcSseEvent;

/* A connected SSE client. The fd is owned by the caller (the HTTP server);
 * the SSE layer does NOT close it on disconnect — it just marks the slot
 * free so the caller can close. `last_event_id` is the most recent id
 * sent to (or received from) this client; used for reconnection support. */
typedef struct OcSseClient {
    int    fd;                       /* socket fd, or -1 if slot is free */
    char   last_event_id[OC_SSE_MAX_EVENT_ID];
    time_t connected_at;             /* time(NULL) at accept              */
    bool   in_use;                   /* true if this slot holds a client  */
} OcSseClient;

/* The SSE server: a fixed-capacity table of clients. The server does not
 * own the listening socket; the caller accepts HTTP connections, upgrades
 * them (writes the SSE headers), then hands the fd to
 * oc_sse_server_accept(). This keeps the SSE layer decoupled from the HTTP
 * request parser. */
typedef struct OcSseServer {
    OcSseClient clients[OC_SSE_MAX_CLIENTS];
    size_t      n_clients;            /* count of in_use slots            */
    size_t      max_clients;          /* soft cap (<= OC_SSE_MAX_CLIENTS) */
} OcSseServer;

/* ─── Per-client helpers (no server required) ──────────────────────────── */

/* Initialize an SSE client struct for an already-accepted fd. Returns
 * OC_OK or OC_ERR_INVALID_ARG (NULL client). */
OcError oc_sse_init(OcSseClient *c, int fd);

/* Release client-owned state. Does NOT close `fd` (caller responsibility).
 * Safe on NULL. Zeroes the struct. */
void oc_sse_free(OcSseClient *c);

/* Format an SSE event into `buf` (cap bytes) per the wire format. Handles
 * multi-line `data` by emitting one `data:` line per source line. Writes
 * CRLF line endings and a trailing blank line. Returns bytes written
 * (excluding NUL), or 0 on overflow / NULL args. */
size_t oc_sse_format_event(const OcSseEvent *ev, char *buf, size_t cap);

/* Convenience: format an event from raw fields (avoids constructing an
 * OcSseEvent struct). Any of event/data/id may be NULL. */
size_t oc_sse_format_event_raw(const char *event, const char *data,
                               const char *id, char *buf, size_t cap);

/* Parse incoming SSE data (e.g. from a POST body or upstream stream).
 * `buf` is the raw bytes (may contain multiple events separated by blank
 * lines). On success fills `out_event` (pointers alias into `buf`) and
 * returns the number of bytes consumed from `buf` (including the trailing
 * blank line). Returns 0 if no complete event is present. The caller
 * typically calls this in a loop, advancing `buf` by the returned count.
 *
 * On a parse error (malformed field) returns 0 and does not modify
 * `out_event`; the caller may skip a byte and retry. */
size_t oc_sse_parse_event(const char *buf, size_t len, OcSseEvent *out_event);

/* Send an SSE event to a connected client by writing the formatted block
 * to `c->fd`. On success returns OC_OK and updates `c->last_event_id` if
 * `ev->id` is non-NULL/non-empty. On broken pipe / connection reset
 * returns OC_ERR_NETWORK (the caller should call oc_sse_free on the
 * client). `ev` may have NULL fields. */
OcError oc_sse_send_event(OcSseClient *c, const OcSseEvent *ev);

/* Send a heartbeat comment line (": keepalive\n\n") to keep the connection
 * alive through proxies. Returns OC_OK or OC_ERR_NETWORK. */
OcError oc_sse_send_heartbeat(OcSseClient *c);

/* ─── Server ─────────────────────────────────────────────────────────────── */

/* Initialize an SSE server with capacity `max_clients` (clamped to
 * OC_SSE_MAX_CLIENTS). Returns OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_sse_server_init(OcSseServer *s, size_t max_clients);

/* Release all client slots. Does NOT close client fds (caller must close
 * before freeing, or leak them). Safe on NULL. Zeroes the struct. */
void oc_sse_server_free(OcSseServer *s);

/* Register an already-accepted fd as a new SSE client. Returns OC_OK, or
 * OC_ERR_INVALID_ARG, or OC_ERR_OOM (capacity reached; caller should
 * close the fd). On success the new client slot is marked in_use and its
 * `connected_at` is set. */
OcError oc_sse_server_accept(OcSseServer *s, int fd);

/* Broadcast an event to all connected clients. Clients that fail to receive
 * (broken pipe) are disconnected (slot freed; fd NOT closed by this layer
 * — the caller is responsible for closing fds it owns). Returns OC_OK if
 * at least one client received the event, OC_ERR_NETWORK if all clients
 * failed, OC_ERR_INVALID_ARG on NULL args. */
OcError oc_sse_server_broadcast(OcSseServer *s, const OcSseEvent *ev);

/* Disconnect a client by fd: marks the slot free. Does NOT close `fd`.
 * Returns OC_OK if a client was disconnected, OC_ERR_INVALID_ARG if no
 * matching client was found. */
OcError oc_sse_server_disconnect(OcSseServer *s, int fd);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SSE_H */
