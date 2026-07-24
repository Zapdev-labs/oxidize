/*
 * websocket.c — RFC 6455 WebSocket protocol implementation.
 *
 * Contains a dependency-free SHA-1 + Base64 (used only for the
 * Sec-WebSocket-Accept computation), frame parsing + construction
 * (pure, no socket I/O), and thin socket-level wrappers that drive the
 * session state machine.
 *
 * Frame parsing handles:
 *   - 7-bit, 16-bit, and 64-bit payload lengths (extended length fields)
 *   - Client-to-server masking (4-byte XOR)
 *   - Fragmented frames (FIN=0 continuation; the caller reassembles via
 *     the session's frag buffer)
 *
 * Frame construction produces unmasked server frames (RFC 6455 §5.1:
 * servers MUST NOT mask frames to clients).
 */
#define _POSIX_C_SOURCE 200809L   /* snprintf */
#include "oxidize/websocket.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* ─── SHA-1 (FIPS 180-4) ──────────────────────────────────────────────────── */

typedef struct OcSha1Ctx {
    uint32_t state[5];
    uint64_t bit_count;
    uint8_t  buffer[64];
    size_t   buffer_len;
} OcSha1Ctx;

static uint32_t rotl(uint32_t x, uint32_t n)
{
    return (x << n) | (x >> (32u - n));
}

static void sha1_init(OcSha1Ctx *ctx)
{
    ctx->state[0] = 0x67452301u;
    ctx->state[1] = 0xEFCDAB89u;
    ctx->state[2] = 0x98BADCFEu;
    ctx->state[3] = 0x10325476u;
    ctx->state[4] = 0xC3D2E1F0u;
    ctx->bit_count = 0;
    ctx->buffer_len = 0;
}

static void sha1_block(OcSha1Ctx *ctx, const uint8_t block[64])
{
    uint32_t w[80];
    for (size_t i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4]     << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8)  |
               ((uint32_t)block[i * 4 + 3]);
    }
    for (size_t i = 16; i < 80; i++) {
        w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1u);
    }
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2];
    uint32_t d = ctx->state[3], e = ctx->state[4];
    for (size_t i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20)       { f = (b & c) | ((~b) & d);      k = 0x5A827999u; }
        else if (i < 40)  { f = b ^ c ^ d;                 k = 0x6ED9EBA1u; }
        else if (i < 60)  { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
        else              { f = b ^ c ^ d;                 k = 0xCA62C1D6u; }
        uint32_t t = rotl(a, 5u) + f + e + k + w[i];
        e = d; d = c; c = rotl(b, 30u); b = a; a = t;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c;
    ctx->state[3] += d; ctx->state[4] += e;
}

static void sha1_update(OcSha1Ctx *ctx, const uint8_t *data, size_t len)
{
    ctx->bit_count += (uint64_t)len * 8u;
    while (len > 0) {
        size_t copy = 64u - ctx->buffer_len;
        if (copy > len) copy = len;
        memcpy(ctx->buffer + ctx->buffer_len, data, copy);
        ctx->buffer_len += copy;
        data += copy;
        len -= copy;
        if (ctx->buffer_len == 64u) {
            sha1_block(ctx, ctx->buffer);
            ctx->buffer_len = 0;
        }
    }
}

static void sha1_final(OcSha1Ctx *ctx, uint8_t out[20])
{
    /* Append 0x80. */
    ctx->buffer[ctx->buffer_len++] = 0x80u;
    if (ctx->buffer_len > 56u) {
        while (ctx->buffer_len < 64u) ctx->buffer[ctx->buffer_len++] = 0;
        sha1_block(ctx, ctx->buffer);
        ctx->buffer_len = 0;
    }
    while (ctx->buffer_len < 56u) ctx->buffer[ctx->buffer_len++] = 0;
    /* Append 64-bit big-endian bit count. */
    uint64_t bits = ctx->bit_count;
    for (int i = 7; i >= 0; i--) {
        ctx->buffer[56 + i] = (uint8_t)(bits & 0xFFu);
        bits >>= 8;
    }
    sha1_block(ctx, ctx->buffer);
    for (size_t i = 0; i < 5; i++) {
        out[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

void oc_sha1(const uint8_t *data, size_t len, uint8_t out[20])
{
    OcSha1Ctx ctx;
    sha1_init(&ctx);
    sha1_update(&ctx, data, len);
    sha1_final(&ctx, out);
}

/* ─── Base64 ──────────────────────────────────────────────────────────────── */

static const char B64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t oc_base64_encode(const uint8_t *data, size_t len, char *out, size_t cap)
{
    size_t needed = ((len + 2u) / 3u) * 4u + 1u;
    if (out == NULL || cap < needed) return 0;
    size_t o = 0;
    size_t i = 0;
    while (i + 2 < len) {
        uint32_t v = ((uint32_t)data[i] << 16) |
                     ((uint32_t)data[i + 1] << 8) |
                     ((uint32_t)data[i + 2]);
        out[o++] = B64_ALPHABET[(v >> 18) & 0x3F];
        out[o++] = B64_ALPHABET[(v >> 12) & 0x3F];
        out[o++] = B64_ALPHABET[(v >> 6) & 0x3F];
        out[o++] = B64_ALPHABET[v & 0x3F];
        i += 3;
    }
    if (i < len) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        out[o++] = B64_ALPHABET[(v >> 18) & 0x3F];
        out[o++] = B64_ALPHABET[(v >> 12) & 0x3F];
        out[o++] = (i + 1 < len) ? B64_ALPHABET[(v >> 6) & 0x3F] : '=';
        out[o++] = '=';
    }
    out[o] = '\0';
    return o;
}

/* ─── WebSocket accept computation ─────────────────────────────────────────── */

#define OC_WS_MAGIC "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OcError oc_ws_compute_accept(const char *key, char *out, size_t cap)
{
    if (key == NULL || out == NULL || cap < 29u) return OC_ERR_INVALID_ARG;
    /* Concatenate key + magic GUID. */
    size_t key_len = strlen(key);
    size_t magic_len = sizeof(OC_WS_MAGIC) - 1u;
    size_t total = key_len + magic_len;
    uint8_t *buf = malloc(total);
    if (buf == NULL) return OC_ERR_OOM;
    memcpy(buf, key, key_len);
    memcpy(buf + key_len, OC_WS_MAGIC, magic_len);
    uint8_t digest[20];
    oc_sha1(buf, total, digest);
    free(buf);
    size_t n = oc_base64_encode(digest, 20, out, cap);
    if (n == 0) return OC_ERR_INTERNAL;
    return OC_OK;
}

/* ─── Frame parsing ────────────────────────────────────────────────────────── */

size_t oc_ws_parse_frame(const uint8_t *buf, size_t len, OcWsFrame *frame)
{
    if (frame == NULL) return 0;
    memset(frame, 0, sizeof(*frame));
    if (len < 2u) return 0;

    uint8_t b0 = buf[0];
    uint8_t b1 = buf[1];
    frame->fin = (b0 & 0x80u) != 0;
    frame->opcode = b0 & 0x0Fu;
    frame->masked = (b1 & 0x80u) != 0;
    uint8_t len7 = b1 & 0x7Fu;

    size_t hdr = 2u;
    uint64_t payload_len;
    if (len7 < 126u) {
        payload_len = len7;
    } else if (len7 == 126u) {
        if (len < 4u) return 0;
        payload_len = ((uint64_t)buf[2] << 8) | buf[3];
        hdr = 4u;
    } else { /* 127 */
        if (len < 10u) return 0;
        payload_len = 0;
        for (size_t i = 0; i < 8; i++) {
            payload_len = (payload_len << 8) | buf[2 + i];
        }
        hdr = 10u;
    }
    /* Reject lengths that cannot fit size_t / would wrap hdr + payload_len. */
    if (payload_len > (uint64_t)(SIZE_MAX - hdr - 4u)) return 0;
    if (frame->masked) {
        if (len < hdr + 4u) return 0;
        memcpy(frame->mask, buf + hdr, 4);
        hdr += 4u;
    }
    if (len < hdr + (size_t)payload_len) return 0;
    frame->payload = buf + hdr;
    frame->payload_len = (size_t)payload_len;
    frame->frame_len = hdr + (size_t)payload_len;
    return frame->frame_len;
}

/* ─── Frame construction ──────────────────────────────────────────────────── */

size_t oc_ws_build_frame(uint8_t opcode, bool fin, bool masked,
                         const uint8_t *mask, const uint8_t *payload,
                         size_t payload_len, uint8_t *out, size_t cap)
{
    if (out == NULL) return 0;
    if (payload == NULL && payload_len > 0) return 0;
    size_t hdr = 2u;
    if (payload_len < 126u) {
        /* nothing */
    } else if (payload_len < 65536u) {
        hdr += 2u;
    } else {
        hdr += 8u;
    }
    if (masked) hdr += 4u;
    size_t total = hdr + payload_len;
    if (total > cap) return 0;

    size_t off = 0;
    out[off++] = (uint8_t)((fin ? 0x80u : 0x00u) | (opcode & 0x0Fu));
    uint8_t len_flag = masked ? 0x80u : 0x00u;
    if (payload_len < 126u) {
        out[off++] = len_flag | (uint8_t)payload_len;
    } else if (payload_len < 65536u) {
        out[off++] = len_flag | 126u;
        out[off++] = (uint8_t)(payload_len >> 8);
        out[off++] = (uint8_t)(payload_len & 0xFFu);
    } else {
        out[off++] = len_flag | 127u;
        uint64_t l = payload_len;
        for (int i = 7; i >= 0; i--) {
            out[off++] = (uint8_t)((l >> (i * 8)) & 0xFFu);
        }
    }
    if (masked) {
        const uint8_t *m = (mask != NULL) ? mask : (const uint8_t *)"\0\0\0\0";
        memcpy(out + off, m, 4);
        off += 4;
        for (size_t i = 0; i < payload_len; i++) {
            out[off + i] = payload[i] ^ m[i % 4];
        }
    } else if (payload_len > 0) {
        memcpy(out + off, payload, payload_len);
    }
    return total;
}

/* ─── Session lifecycle ─────────────────────────────────────────────────────── */

OcError oc_ws_session_init(OcWsSession *sess, int fd)
{
    if (sess == NULL) return OC_ERR_INVALID_ARG;
    memset(sess, 0, sizeof(*sess));
    sess->fd = fd;
    sess->state = OC_WS_OPEN;
    sess->recv_cap = 65536u;
    sess->recv_buf = malloc(sess->recv_cap);
    if (sess->recv_buf == NULL) {
        sess->recv_cap = 0;
        return OC_ERR_OOM;
    }
    sess->frag_cap = 65536u;
    sess->frag_buf = malloc(sess->frag_cap);
    if (sess->frag_buf == NULL) {
        free(sess->recv_buf);
        sess->recv_buf = NULL;
        sess->recv_cap = 0;
        sess->frag_cap = 0;
        return OC_ERR_OOM;
    }
    return OC_OK;
}

void oc_ws_session_free(OcWsSession *sess)
{
    if (sess == NULL) return;
    free(sess->recv_buf);
    free(sess->frag_buf);
    sess->recv_buf = NULL;
    sess->frag_buf = NULL;
    sess->recv_cap = 0;
    sess->frag_cap = 0;
}

/* ─── Socket-level read/send ───────────────────────────────────────────────── */

/* ponytail: 16 MiB cap on a single (possibly fragmented) message — bump if a
 * legitimate use case ever needs more. */
#define OC_WS_MAX_MESSAGE (16u * 1024u * 1024u)

/* Ensure frag_buf can hold at least `need` bytes. */
static OcError frag_reserve(OcWsSession *sess, size_t need)
{
    if (need <= sess->frag_cap) return OC_OK;
    size_t cap = sess->frag_cap ? sess->frag_cap : 65536u;
    while (cap < need) cap *= 2u;
    uint8_t *nb = realloc(sess->frag_buf, cap);
    if (nb == NULL) return OC_ERR_OOM;
    sess->frag_buf = nb;
    sess->frag_cap = cap;
    return OC_OK;
}

/* Read exactly one complete wire frame into *frame, buffering partial and
 * coalesced TCP reads in recv_buf. On return the frame's raw bytes are
 * still at the front of recv_buf. */
static OcError read_one_frame(OcWsSession *sess, OcWsFrame *frame)
{
    for (;;) {
        if (sess->recv_len >= 2u) {
            size_t consumed = oc_ws_parse_frame(sess->recv_buf, sess->recv_len,
                                                frame);
            if (consumed > 0) return OC_OK;
        }
        if (sess->recv_len >= OC_WS_MAX_MESSAGE) return OC_ERR_FORMAT;
        if (sess->recv_len == sess->recv_cap) {
            size_t cap = sess->recv_cap * 2u;
            uint8_t *nb = realloc(sess->recv_buf, cap);
            if (nb == NULL) return OC_ERR_OOM;
            sess->recv_buf = nb;
            sess->recv_cap = cap;
        }
        ssize_t n = read(sess->fd, sess->recv_buf + sess->recv_len,
                         sess->recv_cap - sess->recv_len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return OC_ERR_IO;
        sess->recv_len += (size_t)n;
    }
}

/* Drop the first `consumed` bytes of recv_buf (compact leftover bytes). */
static void recv_consume(OcWsSession *sess, size_t consumed)
{
    memmove(sess->recv_buf, sess->recv_buf + consumed,
            sess->recv_len - consumed);
    sess->recv_len -= consumed;
}

OcError oc_ws_read_frame(OcWsSession *sess, OcWsFrame *frame)
{
    if (sess == NULL || frame == NULL) return OC_ERR_INVALID_ARG;
    if (sess->state == OC_WS_CLOSED) return OC_ERR_IO;

    for (;;) {
        OcError e = read_one_frame(sess, frame);
        if (e != OC_OK) return e;
        /* RFC 6455 §5.1: a server MUST reject unmasked client frames. */
        if (!frame->masked) return OC_ERR_FORMAT;

        /* Copy + unmask the payload out of recv_buf into frag_buf so the
         * receive buffer can be compacted immediately. Fragment payloads
         * accumulate at frag_len; control/whole-message payloads land just
         * past any in-progress fragment data. */
        bool is_control = (frame->opcode & 0x8u) != 0;
        bool is_data_start = frame->opcode == OC_WS_OPCODE_TEXT ||
                             frame->opcode == OC_WS_OPCODE_BINARY;
        bool is_cont = frame->opcode == OC_WS_OPCODE_CONTINUATION;
        size_t plen = frame->payload_len;

        if (is_control) {
            e = frag_reserve(sess, sess->frag_len + plen);
            if (e != OC_OK) return e;
            uint8_t *dst = sess->frag_buf + sess->frag_len;
            for (size_t i = 0; i < plen; i++) {
                dst[i] = frame->payload[i] ^ frame->mask[i % 4];
            }
            recv_consume(sess, frame->frame_len);
            frame->payload = dst;
            frame->masked = false;
            return OC_OK;
        }

        if (is_data_start && frame->fin && !sess->frag_in_progress) {
            /* Whole unfragmented message. */
            e = frag_reserve(sess, plen);
            if (e != OC_OK) return e;
            for (size_t i = 0; i < plen; i++) {
                sess->frag_buf[i] = frame->payload[i] ^ frame->mask[i % 4];
            }
            recv_consume(sess, frame->frame_len);
            frame->payload = sess->frag_buf;
            frame->masked = false;
            return OC_OK;
        }

        /* Fragmented path. */
        if (is_data_start) {
            if (sess->frag_in_progress) return OC_ERR_FORMAT;
            sess->frag_in_progress = true;
            sess->frag_opcode = frame->opcode;
            sess->frag_len = 0;
        } else if (is_cont) {
            if (!sess->frag_in_progress) return OC_ERR_FORMAT;
        } else {
            return OC_ERR_FORMAT;   /* reserved opcode */
        }
        if (sess->frag_len + plen > OC_WS_MAX_MESSAGE) return OC_ERR_FORMAT;
        e = frag_reserve(sess, sess->frag_len + plen);
        if (e != OC_OK) return e;
        for (size_t i = 0; i < plen; i++) {
            sess->frag_buf[sess->frag_len + i] =
                frame->payload[i] ^ frame->mask[i % 4];
        }
        sess->frag_len += plen;
        recv_consume(sess, frame->frame_len);
        if (frame->fin) {
            frame->opcode = sess->frag_opcode;
            frame->payload = sess->frag_buf;
            frame->payload_len = sess->frag_len;
            frame->masked = false;
            sess->frag_in_progress = false;
            sess->frag_len = 0;   /* buffer contents stay valid until next read */
            return OC_OK;
        }
        /* Not final: keep reading frames until the message completes. */
    }
}

OcError oc_ws_send_frame(OcWsSession *sess, uint8_t opcode, bool fin,
                         const uint8_t *payload, size_t payload_len)
{
    if (sess == NULL) return OC_ERR_INVALID_ARG;
    if (sess->state == OC_WS_CLOSED) return OC_ERR_IO;
    /* Servers MUST NOT mask frames. */
    size_t needed = 14u + payload_len;
    uint8_t *buf = malloc(needed);
    if (buf == NULL) return OC_ERR_OOM;
    size_t n = oc_ws_build_frame(opcode, fin, false, NULL,
                                 payload, payload_len, buf, needed);
    if (n == 0) {
        free(buf);
        return OC_ERR_INTERNAL;
    }
    size_t off = 0;
    while (off < n) {
        /* Use send() so a peer disconnect yields EPIPE instead of SIGPIPE. */
#ifdef MSG_NOSIGNAL
        ssize_t w = send(sess->fd, buf + off, n - off, MSG_NOSIGNAL);
#else
        ssize_t w = send(sess->fd, buf + off, n - off, 0);
#endif
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) {
            free(buf);
            return OC_ERR_IO;
        }
        off += (size_t)w;
    }
    free(buf);
    if (opcode == OC_WS_OPCODE_CLOSE) {
        sess->state = OC_WS_CLOSING;
    }
    return OC_OK;
}

OcError oc_ws_send_text(OcWsSession *sess, const char *text)
{
    if (text == NULL) text = "";
    return oc_ws_send_frame(sess, OC_WS_OPCODE_TEXT, true,
                            (const uint8_t *)text, strlen(text));
}

OcError oc_ws_close(OcWsSession *sess, uint16_t status_code)
{
    if (sess == NULL) return OC_ERR_INVALID_ARG;
    if (sess->state == OC_WS_CLOSED) return OC_OK;
    uint16_t code = (status_code == 0) ? 1000u : status_code;
    uint8_t payload[2];
    payload[0] = (uint8_t)(code >> 8);
    payload[1] = (uint8_t)(code & 0xFFu);
    OcError e = oc_ws_send_frame(sess, OC_WS_OPCODE_CLOSE, true, payload, 2);
    if (e == OC_OK) {
        sess->state = OC_WS_CLOSED;
    }
    return e;
}
