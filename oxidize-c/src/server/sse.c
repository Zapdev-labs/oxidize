/* sse.c — Server-Sent Events implementation. */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/sse.h"

#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


/* Append `s` (NUL-terminated) to `buf` at offset `*off`, advancing *off.
 * Returns true on success, false on overflow. */
static bool buf_append(char *buf, size_t cap, size_t *off, const char *s)
{
    size_t slen = strlen(s);
    if (*off + slen + 1 > cap) return false;
    memcpy(buf + *off, s, slen);
    *off += slen;
    return true;
}

/* Write all bytes to fd, retrying on EINTR. Returns true if all bytes were written, false on a hard error (broken pipe, connection reset, etc.). Blocks SIGPIPE around the write so a dead client does not kill the process; instead write() returns -1/EPIPE and we treat it as a disconnect. */
static bool write_all(int fd, const char *buf, size_t len)
{
    /* Block SIGPIPE for the lifetime of this process so a dead client
     * surfaces as EPIPE instead of terminating the server. This is a
     * one-time setup; subsequent calls are no-ops. */
    static bool sigpipe_blocked = false;
    if (!sigpipe_blocked) {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = SIG_IGN;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGPIPE, &sa, NULL);
        sigpipe_blocked = true;
    }

    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, buf + off, len - off);
        if (n > 0) {
            off += (size_t)n;
            continue;
        }
        if (n < 0 && (errno == EINTR)) continue;
        /* EPIPE / ECONNRESET / ENOTCONN / EBADF: caller should disconnect. */
        return false;
    }
    return true;
}

/* Skip CRLF or LF at `buf[*pos]`, advancing *pos. Returns the number of
 * bytes consumed (0, 1, or 2). */
static size_t skip_eol(const char *buf, size_t len, size_t *pos)
{
    size_t consumed = 0;
    if (*pos < len && buf[*pos] == '\r') { (*pos)++; consumed++; }
    if (*pos < len && buf[*pos] == '\n') { (*pos)++; consumed++; }
    return consumed;
}


OcError oc_sse_init(OcSseClient *c, int fd)
{
    if (!c) return OC_ERR_INVALID_ARG;
    memset(c, 0, sizeof(*c));
    c->fd = fd;
    c->connected_at = time(NULL);
    c->in_use = (fd >= 0);
    return OC_OK;
}

void oc_sse_free(OcSseClient *c)
{
    if (!c) return;
    memset(c, 0, sizeof(*c));
    c->fd = -1;
}

size_t oc_sse_format_event(const OcSseEvent *ev, char *buf, size_t cap)
{
    if (!ev || !buf || cap == 0) return 0;
    size_t off = 0;

    if (ev->event && *ev->event) {
        if (!buf_append(buf, cap, &off, "event: ")) return 0;
        if (!buf_append(buf, cap, &off, ev->event)) return 0;
        if (!buf_append(buf, cap, &off, "\r\n")) return 0;
    }

    if (ev->data && *ev->data) {
        const char *p = ev->data;
        while (1) {
            const char *nl = strchr(p, '\n');
            size_t line_len = nl ? (size_t)(nl - p) : strlen(p);
            /* Strip a trailing \r if present (CRLF source data). */
            if (line_len > 0 && p[line_len - 1] == '\r') line_len--;
            if (!buf_append(buf, cap, &off, "data: ")) return 0;
            if (off + line_len + 1 > cap) return 0;
            memcpy(buf + off, p, line_len);
            off += line_len;
            if (!buf_append(buf, cap, &off, "\r\n")) return 0;
            if (!nl) break;
            p = nl + 1;
            if (*p == '\0') break;
        }
    }

    if (ev->id && *ev->id) {
        if (!buf_append(buf, cap, &off, "id: ")) return 0;
        if (!buf_append(buf, cap, &off, ev->id)) return 0;
        if (!buf_append(buf, cap, &off, "\r\n")) return 0;
    }

    /* Terminating blank line. */
    if (!buf_append(buf, cap, &off, "\r\n")) return 0;

    buf[off] = '\0';
    return off;
}

size_t oc_sse_format_event_raw(const char *event, const char *data,
                               const char *id, char *buf, size_t cap)
{
    OcSseEvent ev = { .event = event, .data = data, .id = id };
    return oc_sse_format_event(&ev, buf, cap);
}

size_t oc_sse_parse_event(const char *buf, size_t len, OcSseEvent *out_event)
{
    if (!buf || !out_event || len == 0) return 0;
    /* The parser mutates the caller's buffer in place to NUL-terminate field values and to join multi-line `data:` payloads with '\n'. The buffer is treated as writable per the documented contract (callers pass a writable recv buffer). `buf` is `const char *` in the signature for API ergonomics; we cast away const here. */
    char *wbuf = (char *)buf;

    memset(out_event, 0, sizeof(*out_event));
    size_t pos = 0;
    bool have_field = false;

    /* For multi-line `data:` we rewrite the buffer in place: the first data value stays where it is, and subsequent values are moved right after it (joined by '\n'). `data_start` is the anchor offset of the joined data region; `data_write` is the current write cursor. */
    size_t data_start = 0;
    char *data_write = NULL;
    bool data_started = false;

    while (pos < len) {
        /* Detect end-of-event: a blank line (just CRLF or just LF). */
        size_t line_start = pos;
        size_t eol_consumed = skip_eol(buf, len, &pos);
        if (eol_consumed > 0 && line_start == pos - eol_consumed) {
            /* Blank line: event terminator (only if we've seen a field). */
            if (!have_field) continue;  /* ignore leading blank lines */
            /* Finalize: NUL-terminate the data region if we built one. */
            if (data_write) *data_write = '\0';
            return pos;
        }

        /* Not a blank line: parse a "field: value" line. */
        /* Find the colon (if any) before the line end. */
        size_t colon = line_start;
        while (colon < len && buf[colon] != '\n' && buf[colon] != '\r' &&
               buf[colon] != ':') {
            colon++;
        }
        size_t field_end = colon;
        size_t value_start = colon;
        size_t value_end;
        if (colon < len && buf[colon] == ':') {
            value_start = colon + 1;
            /* Per spec: if the first char after ':' is a space, skip it. */
            if (value_start < len && buf[value_start] == ' ') value_start++;
            value_end = value_start;
            while (value_end < len && buf[value_end] != '\n' &&
                   buf[value_end] != '\r') {
                value_end++;
            }
        } else {
            value_end = field_end;
        }

        /* Match the field name (buf[line_start .. field_end)). */
        size_t fnlen = field_end - line_start;
        if (fnlen == 4 && memcmp(buf + line_start, "data", 4) == 0) {
            size_t vlen = value_end - value_start;
            if (!data_started) {
                /* First data line: anchor the region at the value start. */
                data_start = value_start;
                data_write = wbuf + value_start + vlen;
                data_started = true;
                out_event->data = wbuf + data_start;
            } else {
                /* Subsequent data line: join with '\n' then append value. */
                *data_write++ = '\n';
                if (vlen > 0) {
                    memmove(data_write, wbuf + value_start, vlen);
                    data_write += vlen;
                }
            }
            have_field = true;
        } else if (fnlen == 5 && memcmp(buf + line_start, "event", 5) == 0) {
            have_field = true;
            /* Advance pos past the EOL BEFORE NUL-terminating in place,
             * so we don't destroy the EOL bytes we need to skip. */
            pos = value_end;
            if (pos < len && buf[pos] == '\r') pos++;
            if (pos < len && buf[pos] == '\n') pos++;
            wbuf[value_end] = '\0';
            out_event->event = wbuf + value_start;
            continue;
        } else if (fnlen == 2 && memcmp(buf + line_start, "id", 2) == 0) {
            have_field = true;
            pos = value_end;
            if (pos < len && buf[pos] == '\r') pos++;
            if (pos < len && buf[pos] == '\n') pos++;
            wbuf[value_end] = '\0';
            out_event->id = wbuf + value_start;
            continue;
        } else {
            /* Unknown field or comment line; ignore but mark have_field. */
            have_field = true;
        }

        /* Advance past the line end. */
        pos = value_end;
        if (pos < len && buf[pos] == '\r') pos++;
        if (pos < len && buf[pos] == '\n') pos++;
    }

    /* No blank line found: event is incomplete. */
    return 0;
}

OcError oc_sse_send_event(OcSseClient *c, const OcSseEvent *ev)
{
    if (!c || !ev || c->fd < 0) return OC_ERR_INVALID_ARG;
    char buf[OC_SSE_MAX_EVENT_BYTES];
    size_t n = oc_sse_format_event(ev, buf, sizeof(buf));
    if (n == 0) return OC_ERR_INTERNAL;
    if (!write_all(c->fd, buf, n)) {
        return OC_ERR_NETWORK;
    }
    if (ev->id && *ev->id) {
        size_t ilen = strlen(ev->id);
        if (ilen >= sizeof(c->last_event_id)) ilen = sizeof(c->last_event_id) - 1;
        memcpy(c->last_event_id, ev->id, ilen);
        c->last_event_id[ilen] = '\0';
    }
    return OC_OK;
}

OcError oc_sse_send_heartbeat(OcSseClient *c)
{
    if (!c || c->fd < 0) return OC_ERR_INVALID_ARG;
    const char *heartbeat = ": keepalive\r\n\r\n";
    if (!write_all(c->fd, heartbeat, strlen(heartbeat))) {
        return OC_ERR_NETWORK;
    }
    return OC_OK;
}


OcError oc_sse_server_init(OcSseServer *s, size_t max_clients)
{
    if (!s) return OC_ERR_INVALID_ARG;
    memset(s, 0, sizeof(*s));
    if (max_clients == 0) max_clients = OC_SSE_MAX_CLIENTS;
    if (max_clients > OC_SSE_MAX_CLIENTS) max_clients = OC_SSE_MAX_CLIENTS;
    s->max_clients = max_clients;
    /* All slots start with fd = -1, in_use = false (memset to 0 already). */
    for (size_t i = 0; i < OC_SSE_MAX_CLIENTS; i++) {
        s->clients[i].fd = -1;
    }
    return OC_OK;
}

void oc_sse_server_free(OcSseServer *s)
{
    if (!s) return;
    /* The server does not own client fds; the caller is responsible for
     * closing them. We just mark slots free. */
    for (size_t i = 0; i < OC_SSE_MAX_CLIENTS; i++) {
        s->clients[i].fd = -1;
        s->clients[i].in_use = false;
    }
    s->n_clients = 0;
}

OcError oc_sse_server_accept(OcSseServer *s, int fd)
{
    if (!s || fd < 0) return OC_ERR_INVALID_ARG;
    if (s->n_clients >= s->max_clients) return OC_ERR_OOM;
    for (size_t i = 0; i < OC_SSE_MAX_CLIENTS; i++) {
        if (!s->clients[i].in_use) {
            oc_sse_init(&s->clients[i], fd);
            s->n_clients++;
            return OC_OK;
        }
    }
    return OC_ERR_OOM;
}

OcError oc_sse_server_broadcast(OcSseServer *s, const OcSseEvent *ev)
{
    if (!s || !ev) return OC_ERR_INVALID_ARG;
    if (s->n_clients == 0) return OC_ERR_NETWORK;

    /* Format once, send to each client. */
    char buf[OC_SSE_MAX_EVENT_BYTES];
    size_t n = oc_sse_format_event(ev, buf, sizeof(buf));
    if (n == 0) return OC_ERR_INTERNAL;

    size_t ok = 0;
    size_t failed = 0;
    for (size_t i = 0; i < OC_SSE_MAX_CLIENTS; i++) {
        if (!s->clients[i].in_use) continue;
        if (!write_all(s->clients[i].fd, buf, n)) {
            /* Disconnect this client: mark slot free (do NOT close fd). */
            s->clients[i].in_use = false;
            s->clients[i].fd = -1;
            s->n_clients--;
            failed++;
        } else {
            if (ev->id && *ev->id) {
                size_t ilen = strlen(ev->id);
                if (ilen >= sizeof(s->clients[i].last_event_id))
                    ilen = sizeof(s->clients[i].last_event_id) - 1;
                memcpy(s->clients[i].last_event_id, ev->id, ilen);
                s->clients[i].last_event_id[ilen] = '\0';
            }
            ok++;
        }
    }
    if (ok == 0) return OC_ERR_NETWORK;
    (void)failed;
    return OC_OK;
}

OcError oc_sse_server_disconnect(OcSseServer *s, int fd)
{
    if (!s) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < OC_SSE_MAX_CLIENTS; i++) {
        if (s->clients[i].in_use && s->clients[i].fd == fd) {
            s->clients[i].in_use = false;
            s->clients[i].fd = -1;
            if (s->n_clients > 0) s->n_clients--;
            return OC_OK;
        }
    }
    return OC_ERR_INVALID_ARG;
}
