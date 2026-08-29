/* cuda_stub.c — stub implementation of CUDA API when compiled without OC_CUDA. */
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
