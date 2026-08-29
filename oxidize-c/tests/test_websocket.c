/*
 * test_websocket.c — WebSocket protocol tests.
 *
 * VAL-WS-001..005 cover:
 *   1. SHA-1 computation (FIPS test vector "abc").
 *   2. Base64 encoding (known vector "Man" -> "TWFu").
 *   3. WebSocket accept key computation (RFC 6455 §1.3 test vector).
 *   4. Frame construction (text, binary, close).
 *   5. Frame parsing (unmasked text, masked text, extended payload length).
 */
#define _GNU_SOURCE 1
#include "framework.h"

#include "oxidize/websocket.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ─── SHA-1 ──────────────────────────────────────────────────────────────── */

Test(websocket, sha1_known_vector)
{
    /* FIPS 180-2: SHA-1("abc") = a9993e36 4706816a ba3e2571 7850c26c 9cd0d89d */
    uint8_t out[20];
    oc_sha1((const uint8_t *)"abc", 3, out);
    static const uint8_t expected[20] = {
        0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a,
        0xba, 0x3e, 0x25, 0x71, 0x78, 0x50, 0xc2, 0x6c,
        0x9c, 0xd0, 0xd8, 0x9d,
    };
    cr_assert(memcmp(out, expected, 20) == 0, "SHA-1(\"abc\") mismatch");
}

Test(websocket, sha1_empty_string)
{
    /* SHA-1("") = da39a3ee 5e6b4b0d 3255bfef 95601890 afd80709 */
    uint8_t out[20];
    oc_sha1((const uint8_t *)"", 0, out);
    static const uint8_t expected[20] = {
        0xda, 0x39, 0xa3, 0xee, 0x5e, 0x6b, 0x4b, 0x0d,
        0x32, 0x55, 0xbf, 0xef, 0x95, 0x60, 0x18, 0x90,
        0xaf, 0xd8, 0x07, 0x09,
    };
    cr_assert(memcmp(out, expected, 20) == 0, "SHA-1(\"\") mismatch");
}

Test(websocket, sha1_longer_input)
{
    /* SHA-1("The quick brown fox jumps over the lazy dog") =
     * 2fd4e1c6 7a2d28fc ed849ee1 bb76e739 1b93eb12 */
    const char *msg = "The quick brown fox jumps over the lazy dog";
    uint8_t out[20];
    oc_sha1((const uint8_t *)msg, strlen(msg), out);
    static const uint8_t expected[20] = {
        0x2f, 0xd4, 0xe1, 0xc6, 0x7a, 0x2d, 0x28, 0xfc,
        0xed, 0x84, 0x9e, 0xe1, 0xbb, 0x76, 0xe7, 0x39,
        0x1b, 0x93, 0xeb, 0x12,
    };
    cr_assert(memcmp(out, expected, 20) == 0, "SHA-1(fox) mismatch");
}

/* ─── Base64 ─────────────────────────────────────────────────────────────── */

Test(websocket, base64_known_vector)
{
    /* "Man" -> "TWFu" (RFC 4648 §10 test vector) */
    char out[32];
    size_t n = oc_base64_encode((const uint8_t *)"Man", 3, out, sizeof(out));
    cr_assert_eq(n, 4u);
    cr_assert_str_eq(out, "TWFu");
}

Test(websocket, base64_padding_one_byte)
{
    /* "Ma" -> "TWE=" (2 bytes -> 1 padding) */
    char out[32];
    size_t n = oc_base64_encode((const uint8_t *)"Ma", 2, out, sizeof(out));
    cr_assert_eq(n, 4u);
    cr_assert_str_eq(out, "TWE=");
}

Test(websocket, base64_padding_two_bytes)
{
    /* "M" -> "TQ==" (1 byte -> 2 padding) */
    char out[32];
    size_t n = oc_base64_encode((const uint8_t *)"M", 1, out, sizeof(out));
    cr_assert_eq(n, 4u);
    cr_assert_str_eq(out, "TQ==");
}

Test(websocket, base64_empty)
{
    char out[8];
    size_t n = oc_base64_encode((const uint8_t *)"", 0, out, sizeof(out));
    cr_assert_eq(n, 0u);
    cr_assert_str_eq(out, "");
}

Test(websocket, base64_overflow_returns_zero)
{
    char out[3];   /* too small for any output */
    size_t n = oc_base64_encode((const uint8_t *)"Man", 3, out, sizeof(out));
    cr_assert_eq(n, 0u);
}

/* ─── WebSocket accept key ─────────────────────────────────────────────────── */

Test(websocket, accept_key_rfc6455_vector)
{
    /* RFC 6455 §1.3: client key "dGhlIHNhbXBsZSBub25jZQ==" -> server accept
     * "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" */
    char out[64];
    OcError e = oc_ws_compute_accept("dGhlIHNhbXBsZSBub25jZQ==",
                                     out, sizeof(out));
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(out, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
}

Test(websocket, accept_key_short_buffer_rejected)
{
    char out[5];
    OcError e = oc_ws_compute_accept("dGhlIHNhbXBsZSBub25jZQ==",
                                     out, sizeof(out));
    cr_assert_eq(e, OC_ERR_INVALID_ARG);
}

/* ─── Frame construction ──────────────────────────────────────────────────── */

Test(websocket, build_text_frame_short)
{
    /* A short text frame "hi", FIN=1, unmasked (server->client).
     * Expected: 0x81 0x02 'h' 'i' */
    uint8_t out[16];
    size_t n = oc_ws_build_frame(OC_WS_OPCODE_TEXT, true, false, NULL,
                                 (const uint8_t *)"hi", 2, out, sizeof(out));
    cr_assert_eq(n, 4u);
    cr_assert_eq(out[0], 0x81u, "FIN=1, opcode=1");
    cr_assert_eq(out[1], 0x02u, "payload len 2, no mask");
    cr_assert_eq(out[2], 'h');
    cr_assert_eq(out[3], 'i');
}

Test(websocket, build_binary_frame_masked)
{
    /* Masked binary frame, payload "AB" with mask 0x01 0x02 0x03 0x04.
     * Layout: [0]=0x82 [1]=0x82 [2..5]=mask [6]='A'^0x01 [7]='B'^0x02 */
    uint8_t mask[4] = { 0x01, 0x02, 0x03, 0x04 };
    uint8_t out[16];
    size_t n = oc_ws_build_frame(OC_WS_OPCODE_BINARY, true, true, mask,
                                 (const uint8_t *)"AB", 2, out, sizeof(out));
    cr_assert_eq(n, 8u);
    cr_assert_eq(out[0], 0x82u, "FIN=1, opcode=2");
    cr_assert_eq(out[1], 0x82u, "payload len 2, mask bit set");
    cr_assert_eq(out[2], 0x01u);
    cr_assert_eq(out[3], 0x02u);
    cr_assert_eq(out[4], 0x03u);
    cr_assert_eq(out[5], 0x04u);
    cr_assert_eq(out[6], 'A' ^ 0x01);
    cr_assert_eq(out[7], 'B' ^ 0x02);
}

Test(websocket, build_close_frame)
{
    /* Close frame with status 1000 (0x03E8). FIN=1, opcode=8, payload 2 bytes. */
    uint8_t payload[2] = { 0x03, 0xE8 };
    uint8_t out[16];
    size_t n = oc_ws_build_frame(OC_WS_OPCODE_CLOSE, true, false, NULL,
                                 payload, 2, out, sizeof(out));
    cr_assert_eq(n, 4u);
    cr_assert_eq(out[0], 0x88u, "FIN=1, opcode=8 (close)");
    cr_assert_eq(out[1], 0x02u, "payload len 2");
    cr_assert_eq(out[2], 0x03u);
    cr_assert_eq(out[3], 0xE8u);
}

Test(websocket, build_extended_length_16bit)
{
    /* 126-byte payload uses 16-bit extended length. */
    uint8_t payload[126];
    memset(payload, 'x', sizeof(payload));
    uint8_t out[256];
    size_t n = oc_ws_build_frame(OC_WS_OPCODE_TEXT, true, false, NULL,
                                 payload, 126, out, sizeof(out));
    cr_assert_eq(n, 130u);
    cr_assert_eq(out[0], 0x81u);
    cr_assert_eq(out[1], 126u, "extended length marker");
    cr_assert_eq(out[2], 0x00u, "high byte");
    cr_assert_eq(out[3], 126u, "low byte");
    cr_assert_eq(out[4], 'x');
}

Test(websocket, build_overflow_returns_zero)
{
    uint8_t out[2];   /* too small */
    size_t n = oc_ws_build_frame(OC_WS_OPCODE_TEXT, true, false, NULL,
                                 (const uint8_t *)"hi", 2, out, sizeof(out));
    cr_assert_eq(n, 0u);
}

/* ─── Frame parsing ───────────────────────────────────────────────────────── */

Test(websocket, parse_unmasked_text_frame)
{
    uint8_t buf[] = { 0x81, 0x05, 'h', 'e', 'l', 'l', 'o' };
    OcWsFrame f;
    size_t n = oc_ws_parse_frame(buf, sizeof(buf), &f);
    cr_assert_eq(n, 7u);
    cr_assert(f.fin);
    cr_assert_eq(f.opcode, OC_WS_OPCODE_TEXT);
    cr_assert_not(f.masked);
    cr_assert_eq(f.payload_len, 5u);
    cr_assert(memcmp(f.payload, "hello", 5) == 0);
}

Test(websocket, parse_masked_text_frame)
{
    /* Mask 0x37 0xfa 0x21 0x3d, payload "Hello" masked.
     * "Hello" ^ mask: H^0x37=0x7F, e^0xfa=0x9F, l^0x21=0x4D, l^0x3d=0x51, o^0x37=0x58 */
    uint8_t buf[] = {
        0x81, 0x85,
        0x37, 0xfa, 0x21, 0x3d,
        0x7f, 0x9f, 0x4d, 0x51, 0x58,
    };
    OcWsFrame f;
    size_t n = oc_ws_parse_frame(buf, sizeof(buf), &f);
    cr_assert_eq(n, 11u);
    cr_assert(f.fin);
    cr_assert_eq(f.opcode, OC_WS_OPCODE_TEXT);
    cr_assert(f.masked);
    cr_assert_eq(f.mask[0], 0x37u);
    cr_assert_eq(f.mask[3], 0x3du);
    cr_assert_eq(f.payload_len, 5u);
    /* payload aliases the masked bytes (parser does not unmask; caller does) */
    cr_assert_eq(f.payload[0], 0x7fu);
}

Test(websocket, parse_extended_payload_16bit)
{
    /* 130-byte payload: marker 126, length 130 = 0x00 0x82. */
    uint8_t buf[140];
    buf[0] = 0x82;   /* FIN=1, binary */
    buf[1] = 126u;   /* 16-bit extended */
    buf[2] = 0x00;
    buf[3] = 130u;
    memset(buf + 4, 'y', 130);
    OcWsFrame f;
    size_t n = oc_ws_parse_frame(buf, sizeof(buf), &f);
    cr_assert_eq(n, 134u);
    cr_assert_eq(f.opcode, OC_WS_OPCODE_BINARY);
    cr_assert_eq(f.payload_len, 130u);
    cr_assert_eq(f.frame_len, 134u);
}

Test(websocket, parse_extended_payload_64bit)
{
    /* 300-byte payload via 64-bit extended length. */
    size_t plen = 300;
    uint8_t buf[320];
    buf[0] = 0x81;
    buf[1] = 127u;
    memset(buf + 2, 0, 8);
    buf[9] = (uint8_t)(plen & 0xFFu);
    buf[8] = (uint8_t)((plen >> 8) & 0xFFu);
    memset(buf + 10, 'z', plen);
    OcWsFrame f;
    size_t n = oc_ws_parse_frame(buf, 10 + plen, &f);
    cr_assert_eq(n, 10 + plen);
    cr_assert_eq(f.payload_len, plen);
    cr_assert_eq(f.frame_len, 10 + plen);
}

Test(websocket, parse_returns_zero_for_incomplete)
{
    uint8_t buf[] = { 0x81, 0x05, 'h', 'e' };   /* claims 5 bytes, only 2 */
    OcWsFrame f;
    size_t n = oc_ws_parse_frame(buf, sizeof(buf), &f);
    cr_assert_eq(n, 0u, "incomplete frame should return 0");
}

Test(websocket, parse_close_frame_with_status)
{
    uint8_t buf[] = { 0x88, 0x02, 0x03, 0xE8 };
    OcWsFrame f;
    size_t n = oc_ws_parse_frame(buf, sizeof(buf), &f);
    cr_assert_eq(n, 4u);
    cr_assert_eq(f.opcode, OC_WS_OPCODE_CLOSE);
    cr_assert_eq(f.payload_len, 2u);
    cr_assert_eq(f.payload[0], 0x03u);
    cr_assert_eq(f.payload[1], 0xE8u);
}

Test(websocket, roundtrip_text_frame)
{
    const char *msg = "realtime test payload";
    uint8_t built[64];
    size_t bn = oc_ws_build_frame(OC_WS_OPCODE_TEXT, true, false, NULL,
                                  (const uint8_t *)msg, strlen(msg),
                                  built, sizeof(built));
    cr_assert_gt(bn, 0);
    OcWsFrame f;
    size_t pn = oc_ws_parse_frame(built, bn, &f);
    cr_assert_eq(pn, bn);
    cr_assert_eq(f.payload_len, strlen(msg));
    cr_assert(memcmp(f.payload, msg, strlen(msg)) == 0);
}

/* ─── Session init/free ──────────────────────────────────────────────────── */

Test(websocket, session_init_sets_open_state)
{
    OcWsSession sess;
    cr_assert_eq(oc_ws_session_init(&sess, -1), OC_OK);
    cr_assert_eq(sess.fd, -1);
    cr_assert_eq(sess.state, OC_WS_OPEN);
    cr_assert(sess.recv_buf != NULL);
    cr_assert(sess.recv_cap > 0);
    oc_ws_session_free(&sess);
    cr_assert(sess.recv_buf == NULL);
    cr_assert_eq(sess.recv_cap, 0u);
}

/* ─── Socket-level read: fragmentation + coalescing + unmasked rejection ─── */

#include <sys/socket.h>
#include <unistd.h>

Test(websocket, read_frame_reassembles_fragments)
{
    int fds[2];
    cr_assert_eq(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
    OcWsSession sess;
    cr_assert_eq(oc_ws_session_init(&sess, fds[0]), OC_OK);

    const uint8_t mask[4] = { 0x11, 0x22, 0x33, 0x44 };
    uint8_t wire[64];
    /* TEXT "Hel" FIN=0, then CONTINUATION "lo" FIN=1, both masked;
     * written in one coalesced burst. */
    size_t n1 = oc_ws_build_frame(OC_WS_OPCODE_TEXT, false, true, mask,
                                  (const uint8_t *)"Hel", 3, wire, sizeof(wire));
    cr_assert_gt(n1, 0);
    size_t n2 = oc_ws_build_frame(OC_WS_OPCODE_CONTINUATION, true, true, mask,
                                  (const uint8_t *)"lo", 2,
                                  wire + n1, sizeof(wire) - n1);
    cr_assert_gt(n2, 0);
    cr_assert_eq(write(fds[1], wire, n1 + n2), (ssize_t)(n1 + n2));

    OcWsFrame f;
    cr_assert_eq(oc_ws_read_frame(&sess, &f), OC_OK);
    cr_assert_eq(f.opcode, OC_WS_OPCODE_TEXT);
    cr_assert_eq(f.payload_len, 5u);
    cr_assert(memcmp(f.payload, "Hello", 5) == 0);

    oc_ws_session_free(&sess);
    close(fds[0]);
    close(fds[1]);
}

Test(websocket, read_frame_rejects_unmasked_client_frame)
{
    int fds[2];
    cr_assert_eq(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
    OcWsSession sess;
    cr_assert_eq(oc_ws_session_init(&sess, fds[0]), OC_OK);

    uint8_t wire[16];
    size_t n = oc_ws_build_frame(OC_WS_OPCODE_TEXT, true, false, NULL,
                                 (const uint8_t *)"hi", 2, wire, sizeof(wire));
    cr_assert_eq(write(fds[1], wire, n), (ssize_t)n);

    OcWsFrame f;
    cr_assert_eq(oc_ws_read_frame(&sess, &f), OC_ERR_FORMAT);

    oc_ws_session_free(&sess);
    close(fds[0]);
    close(fds[1]);
}

Test(websocket, parse_rejects_u64_length_wrap)
{
    /* 127-marker with a length near UINT64_MAX must not wrap into a tiny
     * frame; parser must not report a complete frame. */
    uint8_t buf[16] = { 0x81, 0xFF,
                        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xF6 };
    OcWsFrame f;
    cr_assert_eq(oc_ws_parse_frame(buf, sizeof(buf), &f), 0u);
}

Test(websocket, build_rejects_null_payload_with_length)
{
    uint8_t out[32];
    cr_assert_eq(oc_ws_build_frame(OC_WS_OPCODE_TEXT, true, false, NULL,
                                   NULL, 3, out, sizeof(out)), 0u);
}
