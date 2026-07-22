/*
 * offload.c — CPU/GPU layer offload pipeline implementation.
 */
#include "oxidize/offload.h"

#include <stdlib.h>
#include <string.h>

OcError oc_offload_init(OcOffloadPipeline *pipe, OcLlamaModel *model,
                        const OcOffloadConfig *cfg)
{
    if (!pipe || !model || !cfg) return OC_ERR_INVALID_ARG;
    memset(pipe, 0, sizeof(*pipe));
    pipe->config = *cfg;
    pipe->model = model;

    /* Allocate transfer buffer (n_embd floats). */
    pipe->transfer_buf = calloc(cfg->n_embd, sizeof(float));
    if (!pipe->transfer_buf) return OC_ERR_OOM;

    /* Check if CUDA is available. */
    pipe->cuda_active = oc_cuda_available();

    if (pipe->cuda_active && cfg->gpu_layers > 0) {
        /* Initialize CUDA context for GPU layers. */
        OcError e = oc_cuda_init(&pipe->cuda_ctx, model);
        if (e != OC_OK) {
            /* Fall back to CPU-only. */
            pipe->cuda_active = false;
            pipe->cuda_ptr = NULL;
        } else {
            pipe->cuda_ptr = &pipe->cuda_ctx;
        }
    }

    /* Initialize CPU session for CPU layers (or full model if no GPU). */
    pipe->cpu_sess = calloc(1, sizeof(OcLlamaSession));
    if (!pipe->cpu_sess) {
        free(pipe->transfer_buf);
        if (pipe->cuda_ptr) oc_cuda_free(pipe->cuda_ptr);
        return OC_ERR_OOM;
    }
    OcError e = oc_llama_session_init(model, pipe->cpu_sess);
    if (e != OC_OK) {
        free(pipe->cpu_sess);
        free(pipe->transfer_buf);
        if (pipe->cuda_ptr) oc_cuda_free(pipe->cuda_ptr);
        return e;
    }

    pipe->initialized = true;
    return OC_OK;
}

OcError oc_offload_forward(OcOffloadPipeline *pipe, uint32_t token,
                           float *logits_out)
{
    if (!pipe || !pipe->initialized) return OC_ERR_INVALID_ARG;

    uint32_t gpu_layers = pipe->config.gpu_layers;
    if (pipe->cuda_active && gpu_layers > 0) {
        /* Run GPU layers first. */
        OcError e = oc_cuda_forward(pipe->cuda_ptr, token,
                                    (uint32_t)pipe->cpu_sess->pos,
                                    pipe->transfer_buf);
        if (e != OC_OK) return e;

        /* Run remaining CPU layers starting from the GPU output. */
        /* Copy transfer_buf into CPU session's x. */
        memcpy(pipe->cpu_sess->x, pipe->transfer_buf,
               pipe->config.n_embd * sizeof(float));

        /* Run CPU layers [gpu_layers, n_layer). */
        for (uint32_t l = gpu_layers; l < pipe->config.n_layer; l++) {
            /* We need to call forward_layer, but it's static in llama.c.
             * For now, we use oc_llama_forward which processes all layers.
             * A proper implementation would expose per-layer forward. */
            /* TODO: expose per-layer forward for offload. */
        }

        /* Fall through to CPU-only path for now. */
    }

    /* CPU-only path: run full forward. */
    pipe->cpu_sess->pos++;
    return oc_llama_forward(pipe->cpu_sess, token, logits_out);
}

void oc_offload_reset(OcOffloadPipeline *pipe)
{
    if (!pipe || !pipe->initialized) return;
    oc_llama_session_reset(pipe->cpu_sess);
    if (pipe->cuda_ptr) oc_cuda_reset(pipe->cuda_ptr);
}

void oc_offload_free(OcOffloadPipeline *pipe)
{
    if (!pipe) return;
    if (pipe->cpu_sess) {
        oc_llama_session_free(pipe->cpu_sess);
        free(pipe->cpu_sess);
    }
    if (pipe->cuda_ptr) oc_cuda_free(pipe->cuda_ptr);
    free(pipe->transfer_buf);
    memset(pipe, 0, sizeof(*pipe));
}

bool oc_offload_cuda_available(void)
{
    return oc_cuda_available();
}

uint32_t oc_offload_suggest_gpu_layers(uint64_t model_size_bytes,
                                       uint64_t available_vram_bytes)
{
    if (available_vram_bytes == 0) return 0;
    /* Rough estimate: each layer uses model_size / n_layer bytes.
     * We want to fit gpu_layers * (model_size / n_layer) + KV cache in VRAM.
     * For simplicity, assume 20% overhead for KV cache + workspace. */
    uint64_t usable = (uint64_t)(available_vram_bytes * 0.8);
    /* Assume 32 layers as default estimate. */
    uint64_t per_layer = model_size_bytes / 32;
    if (per_layer == 0) return 0;
    uint32_t n = (uint32_t)(usable / per_layer);
    return n;
}
