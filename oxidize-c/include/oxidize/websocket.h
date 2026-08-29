/* websocket.h — RFC 6455 WebSocket protocol (server side). */
#ifndef OXIDIZE_WEBSOCKET_H
#define OXIDIZE_WEBSOCKET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_WS_OPCODE_CONTINUATION = 0x0,
    OC_WS_OPCODE_TEXT         = 0x1,
    OC_WS_OPCODE_BINARY       = 0x2,
    OC_WS_OPCODE_CLOSE        = 0x8,
    OC_WS_OPCODE_PING         = 0x9,
    OC_WS_OPCODE_PONG         = 0xA,
} OcWsOpcode;

typedef enum {
    OC_WS_CONNECTING = 0,
    OC_WS_OPEN,
    OC_WS_CLOSING,
    OC_WS_CLOSED,
} OcWsState;

/* A parsed WebSocket frame. `payload` aliases the input buffer (valid only
 * for the lifetime of that buffer) unless the frame was fragmented, in
 * which case the caller reassembles into its own buffer. */
typedef struct OcWsFrame {
    uint8_t  opcode;
    bool     fin;
    bool     masked;
    uint8_t  mask[4];
    const uint8_t *payload;     /* aliases input buffer or NULL if empty  */
    size_t   payload_len;
    size_t   frame_len;         /* total bytes consumed from input       */
} OcWsFrame;

typedef struct OcWsSession {
    int        fd;
    OcWsState  state;
    /* receive buffer for partial reads */
    uint8_t   *recv_buf;
    size_t     recv_len;
    size_t     recv_cap;
    /* fragmentation reassembly */
    uint8_t    frag_opcode;
    bool       frag_in_progress;
    uint8_t   *frag_buf;
    size_t     frag_len;
    size_t     frag_cap;
} OcWsSession;


/* Compute the 20-byte SHA-1 digest of `data` (len bytes) into `out[20]`. */
void oc_sha1(const uint8_t *data, size_t len, uint8_t out[20]);

/* Base64-encode `data` (len bytes) into `out` (NUL-terminated). Returns
 * the number of base64 chars written (excluding NUL), or 0 if `cap` is
 * too small. `out` must be at least 4*ceil(len/3)+1 bytes. */
size_t oc_base64_encode(const uint8_t *data, size_t len, char *out, size_t cap);

/* Compute the WebSocket accept value for a client-provided Sec-WebSocket-Key.
 * Writes 28 chars + NUL into `out` (cap must be >= 29). Returns OC_OK or
 * OC_ERR_INVALID_ARG (NULL/short buffer). */
OcError oc_ws_compute_accept(const char *key, char *out, size_t cap);

size_t oc_ws_parse_frame(const uint8_t *buf, size_t len, OcWsFrame *frame);

/* Construct a WebSocket frame into `out` (cap bytes). `mask` is used only
 * when `masked` is true (must be 4 bytes). Returns bytes written or 0 on
 * overflow. */
size_t oc_ws_build_frame(uint8_t opcode, bool fin, bool masked,
                         const uint8_t *mask, const uint8_t *payload,
                         size_t payload_len, uint8_t *out, size_t cap);


/* Initialize a session for an already-accepted socket fd. Returns OC_OK
 * or OC_ERR_OOM. Caller owns `sess`. */
OcError oc_ws_session_init(OcWsSession *sess, int fd);

/* Release session-owned buffers. Does NOT close `fd` (caller's responsibility).
 * Safe on NULL. */
void oc_ws_session_free(OcWsSession *sess);


/* Read a complete frame from the session's socket (blocking). Returns OC_OK */
OcError oc_ws_read_frame(OcWsSession *sess, OcWsFrame *frame);

/* Send a frame over the session's socket. Returns OC_OK or OC_ERR_IO. */
OcError oc_ws_send_frame(OcWsSession *sess, uint8_t opcode, bool fin,
                         const uint8_t *payload, size_t payload_len);

/* Convenience: send a NUL-terminated text frame. */
OcError oc_ws_send_text(OcWsSession *sess, const char *text);

/* Send a close frame with the given status code (or 1000 if 0) and mark
 * the session as CLOSING. Returns OC_OK or OC_ERR_IO. */
OcError oc_ws_close(OcWsSession *sess, uint16_t status_code);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_WEBSOCKET_H */
