/* sse.h — Server-Sent Events (SSE) for the C port. */
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

/* A single SSE event. */
typedef struct OcSseEvent {
    const char *event;   /* may be NULL or "" → no event: line       */
    const char *data;    /* may be NULL or "" → no data: line        */
    const char *id;      /* may be NULL or "" → no id: line          */
} OcSseEvent;

/* A connected SSE client. */
typedef struct OcSseClient {
    int    fd;                       /* socket fd, or -1 if slot is free */
    char   last_event_id[OC_SSE_MAX_EVENT_ID];
    time_t connected_at;             /* time(NULL) at accept              */
    bool   in_use;                   /* true if this slot holds a client  */
} OcSseClient;

/* The SSE server: a fixed-capacity table of clients. The server does not */
typedef struct OcSseServer {
    OcSseClient clients[OC_SSE_MAX_CLIENTS];
    size_t      n_clients;            /* count of in_use slots            */
    size_t      max_clients;          /* soft cap (<= OC_SSE_MAX_CLIENTS) */
} OcSseServer;


/* Initialize an SSE client struct for an already-accepted fd. Returns
 * OC_OK or OC_ERR_INVALID_ARG (NULL client). */
OcError oc_sse_init(OcSseClient *c, int fd);

/* Release client-owned state. Does NOT close `fd` (caller responsibility).
 * Safe on NULL. Zeroes the struct. */
void oc_sse_free(OcSseClient *c);

/* Format an SSE event into `buf` (cap bytes) per the wire format. Handles */
size_t oc_sse_format_event(const OcSseEvent *ev, char *buf, size_t cap);

/* Convenience: format an event from raw fields (avoids constructing an
 * OcSseEvent struct). Any of event/data/id may be NULL. */
size_t oc_sse_format_event_raw(const char *event, const char *data,
                               const char *id, char *buf, size_t cap);

/* Parse incoming SSE data (e.g. from a POST body or upstream stream). */
size_t oc_sse_parse_event(const char *buf, size_t len, OcSseEvent *out_event);

/* Send an SSE event to a connected client by writing the formatted block */
OcError oc_sse_send_event(OcSseClient *c, const OcSseEvent *ev);

/* Send a heartbeat comment line (": keepalive\n\n") to keep the connection
 * alive through proxies. Returns OC_OK or OC_ERR_NETWORK. */
OcError oc_sse_send_heartbeat(OcSseClient *c);


/* Initialize an SSE server with capacity `max_clients` (clamped to
 * OC_SSE_MAX_CLIENTS). Returns OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_sse_server_init(OcSseServer *s, size_t max_clients);

/* Release all client slots. Does NOT close client fds (caller must close
 * before freeing, or leak them). Safe on NULL. Zeroes the struct. */
void oc_sse_server_free(OcSseServer *s);

/* Register an already-accepted fd as a new SSE client. */
OcError oc_sse_server_accept(OcSseServer *s, int fd);

/* Broadcast an event to all connected clients. Clients that fail to receive */
OcError oc_sse_server_broadcast(OcSseServer *s, const OcSseEvent *ev);

/* Disconnect a client by fd: marks the slot free. Does NOT close `fd`.
 * Returns OC_OK if a client was disconnected, OC_ERR_INVALID_ARG if no
 * matching client was found. */
OcError oc_sse_server_disconnect(OcSseServer *s, int fd);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SSE_H */
