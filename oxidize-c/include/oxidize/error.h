/* error.h — OcError codes + OcErrorCtx chain. */
#ifndef OXIDIZE_ERROR_H
#define OXIDIZE_ERROR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes. Order MUST stay stable (callers compare against these enums).
 * Adding new codes is append-only. */
typedef enum {
    OC_OK = 0,
    OC_ERR_IO,            /* file/socket/read/write failure            */
    OC_ERR_FORMAT,       /* GGUF/SafeTensors parse failure            */
    OC_ERR_QUANT,        /* unknown/unsupported quantization type     */
    OC_ERR_TENSOR,       /* shape/dtype mismatch                      */
    OC_ERR_TOKENIZER,    /* tokenizer load/encode/decode failure       */
    OC_ERR_MODEL,        /* unsupported architecture, layer mismatch   */
    OC_ERR_BACKEND,      /* CUDA/Vulkan/Metal backend init failure    */
    OC_ERR_OOM,          /* out of memory                             */
    OC_ERR_INVALID_ARG,  /* bad argument (caller programming error)    */
    OC_ERR_NETWORK,      /* network/connect failure                   */
    OC_ERR_AUTH,         /* authentication/authorization failure       */
    OC_ERR_INTERNAL,     /* internal invariant violation              */
    OC_ERR__COUNT,       /* sentinel; not a valid error code          */
} OcError;

/* Human-readable, NUL-terminated message for the given code. Returns
 * "unknown" for codes outside the enum range. Never returns NULL. */
const char *oc_error_msg(OcError e);

/* True if the code is OC_OK. Convenience for callers. */
bool oc_error_is_ok(OcError e);

/* Rich error context with optional message + cause chain. `msg` is owned by
 * an OcArena (caller-managed lifetime); `cause` is an owned pointer chain
 * freed via oc_error_ctx_free(). */
typedef struct OcErrorCtx {
    OcError code;
    const char *msg;             /* owned by an arena, not freed by ctx */
    struct OcErrorCtx *cause;   /* owned, freed by oc_error_ctx_free    */
} OcErrorCtx;

/* Allocate a new error context on the heap (malloc). `msg` may be NULL.
 * `cause` may be NULL (no chain). Returns NULL on OOM. */
OcErrorCtx *oc_error_ctx_new(OcError code, const char *msg, OcErrorCtx *cause);

/* Free an error context chain (frees this node + all causes recursively).
 * Does NOT free `msg` (arena-owned). Safe on NULL. */
void oc_error_ctx_free(OcErrorCtx *ctx);

/* Format the full error chain into `buf` (up to `cap-1` chars, NUL-terminated).
 * Returns the number of bytes written (excluding NUL). If `buf` is NULL or
 * cap==0, returns the length that would have been written. */
size_t oc_error_ctx_format(const OcErrorCtx *ctx, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ERROR_H */
