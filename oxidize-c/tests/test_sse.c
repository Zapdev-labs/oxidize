/* test_sse.c — SSE streaming tests. */
#define _POSIX_C_SOURCE 200809L

#include "framework.h"
#include "oxidize/sse.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* ─── Per-client: init / free ───────────────────────────────────────────── */

Test(sse, init_free)
{
    OcSseClient c;
    cr_assert_eq(oc_sse_init(&c, 5), OC_OK);
    cr_assert_eq(c.fd, 5);
    cr_assert(c.in_use);
    cr_assert_eq(c.last_event_id[0], '\0');
    oc_sse_free(&c);
    cr_assert_eq(c.fd, -1);
    cr_assert(!c.in_use);
}

Test(sse, init_null)
{
    cr_assert_eq(oc_sse_init(NULL, 1), OC_ERR_INVALID_ARG);
}

Test(sse, free_null)
{
    oc_sse_free(NULL);
}

/* ─── format_event ───────────────────────────────────────────────────────── */

Test(sse, format_event_basic)
{
    OcSseEvent ev = { .event = "token", .data = "hello", .id = "1" };
    char buf[256];
    size_t n = oc_sse_format_event(&ev, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "event: token\r\n") != NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "data: hello\r\n") != NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "id: 1\r\n") != NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "\r\n\r\n") != NULL, "buf=%s", buf);
}

Test(sse, format_event_no_event)
{
    OcSseEvent ev = { .event = NULL, .data = "payload", .id = NULL };
    char buf[128];
    size_t n = oc_sse_format_event(&ev, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "event:") == NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "data: payload\r\n") != NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "id:") == NULL, "buf=%s", buf);
}

Test(sse, format_event_empty_fields)
{
    OcSseEvent ev = { .event = "", .data = "", .id = "" };
    char buf[64];
    size_t n = oc_sse_format_event(&ev, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    /* All fields empty: just the terminating blank line. */
    cr_assert_str_eq(buf, "\r\n");
    (void)n;
}

Test(sse, format_event_multiline_data)
{
    OcSseEvent ev = { .event = NULL, .data = "line1\nline2\nline3", .id = NULL };
    char buf[256];
    size_t n = oc_sse_format_event(&ev, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "data: line1\r\ndata: line2\r\ndata: line3\r\n") != NULL,
              "buf=%s", buf);
}

Test(sse, format_event_crlf_data)
{
    /* Data with embedded CRLF should still produce one `data:` line per
     * logical line, with the trailing \r stripped. */
    OcSseEvent ev = { .event = NULL, .data = "a\r\nb", .id = NULL };
    char buf[128];
    size_t n = oc_sse_format_event(&ev, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "data: a\r\ndata: b\r\n") != NULL, "buf=%s", buf);
    (void)n;
}

Test(sse, format_event_raw)
{
    char buf[128];
    size_t n = oc_sse_format_event_raw("update", "hi", "7", buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "event: update") != NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "data: hi") != NULL, "buf=%s", buf);
    cr_assert(strstr(buf, "id: 7") != NULL, "buf=%s", buf);
}

Test(sse, format_event_null)
{
    char buf[64];
    cr_assert_eq(oc_sse_format_event(NULL, buf, sizeof(buf)), 0);
    OcSseEvent ev = { .event = "x", .data = "y", .id = "z" };
    cr_assert_eq(oc_sse_format_event(&ev, NULL, 0), 0);
}

Test(sse, format_event_overflow)
{
    /* A tiny buffer cannot hold the formatted event; should return 0. */
    OcSseEvent ev = { .event = "verylongeventname", .data = "data", .id = "1" };
    char buf[4];
    cr_assert_eq(oc_sse_format_event(&ev, buf, sizeof(buf)), 0);
}

/* ─── parse_event ───────────────────────────────────────────────────────── */

Test(sse, parse_event_basic)
{
    /* Writable buffer (parse may NUL-terminate in place). */
    char buf[] = "event: token\r\ndata: hello\r\nid: 1\r\n\r\n";
    OcSseEvent ev;
    size_t n = oc_sse_parse_event(buf, strlen(buf), &ev);
    cr_assert_gt(n, 0);
    cr_assert_str_eq(ev.event, "token");
    cr_assert_str_eq(ev.data, "hello");
    cr_assert_str_eq(ev.id, "1");
}

Test(sse, parse_event_lf_only)
{
    char buf[] = "event: token\ndata: hi\n\n";
    OcSseEvent ev;
    size_t n = oc_sse_parse_event(buf, strlen(buf), &ev);
    cr_assert_gt(n, 0);
    cr_assert_str_eq(ev.event, "token");
    cr_assert_str_eq(ev.data, "hi");
    cr_assert_null(ev.id);
}

Test(sse, parse_event_data_only)
{
    char buf[] = "data: payload\r\n\r\n";
    OcSseEvent ev;
    size_t n = oc_sse_parse_event(buf, strlen(buf), &ev);
    cr_assert_gt(n, 0);
    cr_assert_null(ev.event);
    cr_assert_str_eq(ev.data, "payload");
}

Test(sse, parse_event_multiline_data)
{
    char buf[] = "data: a\r\ndata: b\r\n\r\n";
    OcSseEvent ev;
    size_t n = oc_sse_parse_event(buf, strlen(buf), &ev);
    cr_assert_gt(n, 0);
    cr_assert_str_eq(ev.data, "a\nb");
}

Test(sse, parse_event_space_after_colon)
{
    char buf[] = "data: spaced\r\n\r\n";
    OcSseEvent ev;
    size_t n = oc_sse_parse_event(buf, strlen(buf), &ev);
    cr_assert_gt(n, 0);
    /* Leading space after the colon should be stripped. */
    cr_assert_str_eq(ev.data, "spaced");
}

Test(sse, parse_event_incomplete)
{
    char buf[] = "data: incomplete";
    OcSseEvent ev;
    size_t n = oc_sse_parse_event(buf, strlen(buf), &ev);
    cr_assert_eq(n, 0);
}

Test(sse, parse_event_null)
{
    OcSseEvent ev;
    cr_assert_eq(oc_sse_parse_event(NULL, 0, &ev), 0);
    char buf[] = "data: x\r\n\r\n";
    cr_assert_eq(oc_sse_parse_event(buf, strlen(buf), NULL), 0);
}

/* ─── send_event (via pipe) ──────────────────────────────────────────────── */

Test(sse, send_event_pipe)
{
    int fds[2];
    cr_assert_eq(pipe(fds), 0);

    OcSseClient c;
    cr_assert_eq(oc_sse_init(&c, fds[1]), OC_OK);

    OcSseEvent ev = { .event = "token", .data = "hello", .id = "5" };
    cr_assert_eq(oc_sse_send_event(&c, &ev), OC_OK);
    cr_assert_str_eq(c.last_event_id, "5");

    /* Read back from the pipe. */
    char rbuf[256];
    ssize_t r = read(fds[0], rbuf, sizeof(rbuf) - 1);
    cr_assert_gt(r, 0);
    rbuf[r] = '\0';
    cr_assert(strstr(rbuf, "event: token") != NULL, "rbuf=%s", rbuf);
    cr_assert(strstr(rbuf, "data: hello") != NULL, "rbuf=%s", rbuf);
    cr_assert(strstr(rbuf, "id: 5") != NULL, "rbuf=%s", rbuf);

    close(fds[0]);
    close(fds[1]);
}

Test(sse, send_heartbeat_pipe)
{
    int fds[2];
    cr_assert_eq(pipe(fds), 0);

    OcSseClient c;
    cr_assert_eq(oc_sse_init(&c, fds[1]), OC_OK);

    cr_assert_eq(oc_sse_send_heartbeat(&c), OC_OK);

    char rbuf[64];
    ssize_t r = read(fds[0], rbuf, sizeof(rbuf) - 1);
    cr_assert_gt(r, 0);
    rbuf[r] = '\0';
    cr_assert(strstr(rbuf, ": keepalive") != NULL, "rbuf=%s", rbuf);

    close(fds[0]);
    close(fds[1]);
}

Test(sse, send_event_null)
{
    OcSseClient c;
    cr_assert_eq(oc_sse_init(&c, -1), OC_OK);
    OcSseEvent ev = { .event = "x", .data = "y", .id = "z" };
    cr_assert_eq(oc_sse_send_event(NULL, &ev), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_sse_send_event(&c, NULL), OC_ERR_INVALID_ARG);
    oc_sse_free(&c);
}

/* ─── Server: init / free ────────────────────────────────────────────────── */

Test(sse_server, init_free)
{
    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 8), OC_OK);
    cr_assert_eq(s.max_clients, 8);
    cr_assert_eq(s.n_clients, 0);
    oc_sse_server_free(&s);
    cr_assert_eq(s.n_clients, 0);
}

Test(sse_server, init_default_max)
{
    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 0), OC_OK);
    cr_assert_eq(s.max_clients, OC_SSE_MAX_CLIENTS);
    oc_sse_server_free(&s);
}

Test(sse_server, init_clamps_max)
{
    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 99999), OC_OK);
    cr_assert_eq(s.max_clients, OC_SSE_MAX_CLIENTS);
    oc_sse_server_free(&s);
}

Test(sse_server, init_null)
{
    cr_assert_eq(oc_sse_server_init(NULL, 8), OC_ERR_INVALID_ARG);
}

Test(sse_server, free_null)
{
    oc_sse_server_free(NULL);
}

/* ─── accept / disconnect ────────────────────────────────────────────────── */

Test(sse_server, accept_disconnect)
{
    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 4), OC_OK);
    cr_assert_eq(oc_sse_server_accept(&s, 100), OC_OK);
    cr_assert_eq(oc_sse_server_accept(&s, 101), OC_OK);
    cr_assert_eq(s.n_clients, 2);

    cr_assert_eq(oc_sse_server_disconnect(&s, 100), OC_OK);
    cr_assert_eq(s.n_clients, 1);

    /* Disconnect again: not found. */
    cr_assert_eq(oc_sse_server_disconnect(&s, 100), OC_ERR_INVALID_ARG);
    /* Disconnect unknown fd. */
    cr_assert_eq(oc_sse_server_disconnect(&s, 999), OC_ERR_INVALID_ARG);

    oc_sse_server_free(&s);
}

Test(sse_server, accept_null)
{
    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 4), OC_OK);
    cr_assert_eq(oc_sse_server_accept(NULL, 1), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_sse_server_accept(&s, -1), OC_ERR_INVALID_ARG);
    oc_sse_server_free(&s);
}

Test(sse_server, accept_at_capacity)
{
    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 2), OC_OK);
    cr_assert_eq(oc_sse_server_accept(&s, 1), OC_OK);
    cr_assert_eq(oc_sse_server_accept(&s, 2), OC_OK);
    cr_assert_eq(oc_sse_server_accept(&s, 3), OC_ERR_OOM);
    cr_assert_eq(s.n_clients, 2);
    oc_sse_server_free(&s);
}

/* ─── broadcast ─────────────────────────────────────────────────────────── */

Test(sse_server, broadcast_to_clients)
{
    int fds_a[2], fds_b[2];
    cr_assert_eq(pipe(fds_a), 0);
    cr_assert_eq(pipe(fds_b), 0);

    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 4), OC_OK);
    cr_assert_eq(oc_sse_server_accept(&s, fds_a[1]), OC_OK);
    cr_assert_eq(oc_sse_server_accept(&s, fds_b[1]), OC_OK);

    OcSseEvent ev = { .event = "tick", .data = "ping", .id = "1" };
    cr_assert_eq(oc_sse_server_broadcast(&s, &ev), OC_OK);

    /* Both clients should have received the event. */
    char ra[128], rb[128];
    ssize_t na = read(fds_a[0], ra, sizeof(ra) - 1);
    ssize_t nb = read(fds_b[0], rb, sizeof(rb) - 1);
    cr_assert_gt(na, 0);
    cr_assert_gt(nb, 0);
    ra[na] = '\0';
    rb[nb] = '\0';
    cr_assert(strstr(ra, "event: tick") != NULL, "ra=%s", ra);
    cr_assert(strstr(rb, "event: tick") != NULL, "rb=%s", rb);

    close(fds_a[0]); close(fds_a[1]);
    close(fds_b[0]); close(fds_b[1]);
    oc_sse_server_free(&s);
}

Test(sse_server, broadcast_no_clients)
{
    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 4), OC_OK);
    OcSseEvent ev = { .event = "x", .data = "y", .id = "z" };
    cr_assert_eq(oc_sse_server_broadcast(&s, &ev), OC_ERR_NETWORK);
    oc_sse_server_free(&s);
}

Test(sse_server, broadcast_null)
{
    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 4), OC_OK);
    OcSseEvent ev = { .event = "x", .data = "y", .id = "z" };
    cr_assert_eq(oc_sse_server_broadcast(NULL, &ev), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_sse_server_broadcast(&s, NULL), OC_ERR_INVALID_ARG);
    oc_sse_server_free(&s);
}

Test(sse_server, broadcast_disconnects_dead_client)
{
    /* Accept a client whose fd is the read end of a closed pipe: writes fail. */
    int fds[2];
    cr_assert_eq(pipe(fds), 0);
    close(fds[0]);  /* close read end → writes to fds[1] will get EPIPE */

    OcSseServer s;
    cr_assert_eq(oc_sse_server_init(&s, 4), OC_OK);
    cr_assert_eq(oc_sse_server_accept(&s, fds[1]), OC_OK);
    cr_assert_eq(s.n_clients, 1);

    OcSseEvent ev = { .event = "x", .data = "y", .id = "z" };
    /* broadcast should fail to write and disconnect the client. */
    OcError rc = oc_sse_server_broadcast(&s, &ev);
    /* Either all clients failed (OC_ERR_NETWORK) or the SIGPIPE was caught. */
    cr_assert(rc == OC_ERR_NETWORK, "rc=%d", rc);
    cr_assert_eq(s.n_clients, 0);

    close(fds[1]);
    oc_sse_server_free(&s);
}
