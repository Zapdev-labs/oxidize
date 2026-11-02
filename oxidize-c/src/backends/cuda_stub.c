/*
 * cuda_stub.c — stub implementation of CUDA API when compiled without OC_CUDA.
 *
 * Returns OC_ERR_UNSUPPORTED for all functions, so the CPU forward path
 * is used. This file is compiled instead of cuda.cu when OC_CUDA is not
 * defined, keeping the build dependency-free.
 */
#include "oxidize/cuda.h"

#include <string.h>

bool oc_cuda_available(void) { return false; }

OcError oc_cuda_init(OcCudaContext *ctx, const OcLlamaModel *model)
{
    (void)model;
    if (ctx) memset(ctx, 0, sizeof(*ctx));
    return OC_ERR_BACKEND;
}

OcError oc_cuda_forward(OcCudaContext *ctx, uint32_t token, uint32_t pos,
                        float *logits_out)
{
    (void)ctx; (void)token; (void)pos; (void)logits_out;
    return OC_ERR_BACKEND;
}

void oc_cuda_reset(OcCudaContext *ctx) { (void)ctx; }

void oc_cuda_free(OcCudaContext *ctx) { (void)ctx; }

OcError oc_cuda_selftest(void) { return OC_ERR_BACKEND; }

bool oc_cuda_fast_supported(const OcCudaContext *ctx)
{
    (void)ctx; return false;
}

OcError oc_cuda_fast_prefill(OcCudaContext *ctx, const uint32_t *ids,
                             size_t n_ids, uint32_t pos0, bool emit_first)
{
    (void)ctx; (void)ids; (void)n_ids; (void)pos0; (void)emit_first;
    return OC_ERR_BACKEND;
}

OcError oc_cuda_fast_decode(OcCudaContext *ctx, uint32_t n)
{
    (void)ctx; (void)n; return OC_ERR_BACKEND;
}

OcError oc_cuda_fast_take(OcCudaContext *ctx, uint32_t n, uint32_t *out)
{
    (void)ctx; (void)n; (void)out; return OC_ERR_BACKEND;
}

OcError oc_cuda_fast_logits(OcCudaContext *ctx, float *out)
{
    (void)ctx; (void)out; return OC_ERR_BACKEND;
}

OcError oc_cuda_fast_step(OcCudaContext *ctx)
{
    (void)ctx; return OC_ERR_BACKEND;
}

OcError oc_cuda_fast_feed(OcCudaContext *ctx, uint32_t token)
{
    (void)ctx; (void)token; return OC_ERR_BACKEND;
}

OcError oc_cuda_fast_wait(OcCudaContext *ctx)
{
    (void)ctx; return OC_ERR_BACKEND;
}

void oc_cuda_fast_attach(OcCudaContext *ctx) { (void)ctx; }
void oc_cuda_fast_detach(OcCudaContext *ctx) { (void)ctx; }
