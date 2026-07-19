/* error.c — OcError codes + OcErrorCtx chain. */
#include "oxidize/error.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const kMessages[OC_ERR__COUNT] = {
    [OC_OK]              = "ok",
    [OC_ERR_IO]          = "I/O error",
    [OC_ERR_FORMAT]      = "format error",
    [OC_ERR_QUANT]       = "quantization error",
    [OC_ERR_TENSOR]      = "tensor error",
    [OC_ERR_TOKENIZER]   = "tokenizer error",
    [OC_ERR_MODEL]       = "model error",
    [OC_ERR_BACKEND]     = "backend error",
    [OC_ERR_OOM]         = "out of memory",
    [OC_ERR_INVALID_ARG] = "invalid argument",
    [OC_ERR_NETWORK]     = "network error",
    [OC_ERR_AUTH]        = "authentication error",
    [OC_ERR_INTERNAL]    = "internal error",
};

const char *oc_error_msg(OcError e)
{
    if ((unsigned)e < (unsigned)OC_ERR__COUNT) {
        const char *m = kMessages[(unsigned)e];
        if (m) return m;
    }
    return "unknown error";
}

bool oc_error_is_ok(OcError e)
{
    return e == OC_OK;
}

OcErrorCtx *oc_error_ctx_new(OcError code, const char *msg, OcErrorCtx *cause)
{
    OcErrorCtx *ctx = (OcErrorCtx *)malloc(sizeof(OcErrorCtx));
    if (!ctx) return NULL;
    ctx->code  = code;
    ctx->msg   = msg;
    ctx->cause = cause;
    return ctx;
}

void oc_error_ctx_free(OcErrorCtx *ctx)
{
    while (ctx) {
        OcErrorCtx *next = ctx->cause;
        /* msg is arena-owned; do not free. */
        free(ctx);
        ctx = next;
    }
}

size_t oc_error_ctx_format(const OcErrorCtx *ctx, char *buf, size_t cap)
{
    if (!buf || cap == 0) {
        /* Compute required length. */
        size_t total = 0;
        const OcErrorCtx *c = ctx;
        bool first = true;
        while (c) {
            const char *msg = c->msg ? c->msg : "";
            /* crude: use snprintf to a throwaway buffer to measure */
            char tmp[1];
            int n = snprintf(tmp, 0, "%s%s: %s",
                              first ? "" : " caused by: ",
                              oc_error_msg(c->code), msg);
            if (n < 0) n = 0;
            total += (size_t)n;
            first = false;
            c = c->cause;
        }
        return total;
    }

    size_t written = 0;
    bool first = true;
    const OcErrorCtx *c = ctx;
    while (c && written + 1 < cap) {
        const char *msg = c->msg ? c->msg : "";
        int n = snprintf(buf + written, cap - written, "%s%s: %s",
                         first ? "" : " caused by: ",
                         oc_error_msg(c->code), msg);
        if (n < 0) break;
        written += (size_t)n;
        first = false;
        c = c->cause;
    }
    if (cap > 0) buf[written < cap ? written : cap - 1] = '\0';
    return written;
}
