/*
 * offload.h — CPU/GPU layer offload pipeline.
 *
 * Splits model layers between CPU and GPU. Layers [0, gpu_layers) run on GPU,
 * layers [gpu_layers, n_layer) run on CPU. The embedding lookup and final
 * lm_head always run on CPU (they're cheap). The hidden state is transferred
 * between devices at the boundary.
 *
 * This enables partial-GPU inference for models that exceed VRAM.
 */
#ifndef OXIDIZE_OFFLOAD_H
#define OXIDIZE_OFFLOAD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"
#include "oxidize/cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcOffloadConfig {
    uint32_t gpu_layers;     /* number of layers to run on GPU (0=all CPU) */
    uint32_t n_embd;         /* hidden size                            */
    uint32_t n_layer;        /* total layers                           */
    uint32_t n_head;        /* attention heads                         */
    uint32_t head_dim;       /* head dimension                          */
    uint32_t vocab_size;     /* vocabulary size                         */
    uint32_t n_ctx;          /* context length                          */
} OcOffloadConfig;

typedef struct OcOffloadPipeline {
    OcOffloadConfig config;
    OcLlamaModel   *model;          /* base model (CPU weights)        */
    OcLlamaSession *cpu_sess;      /* CPU session for CPU layers       */
    OcCudaContext  cuda_ctx;       /* CUDA context for GPU layers       */
    OcCudaContext  *cuda_ptr;      /* pointer to cuda_ctx if active     */
    float          *transfer_buf;  /* hidden state transfer buffer     */
    bool            cuda_active;   /* is CUDA available?                */
    bool            initialized;
} OcOffloadPipeline;

/* Initialize the offload pipeline. */
OcError oc_offload_init(OcOffloadPipeline *pipe, OcLlamaModel *model,
                        const OcOffloadConfig *cfg);

/* Run one forward step through the offload pipeline.
 * `token` is the input token, `logits_out` receives the output logits. */
OcError oc_offload_forward(OcOffloadPipeline *pipe, uint32_t token,
                           float *logits_out);

/* Reset the pipeline position. */
void oc_offload_reset(OcOffloadPipeline *pipe);

/* Free the offload pipeline. */
void oc_offload_free(OcOffloadPipeline *pipe);

/* Check if CUDA is available for offloading. */
bool oc_offload_cuda_available(void);

/* Suggest the optimal number of GPU layers based on available VRAM.
 * Returns 0 if CUDA is not available. */
uint32_t oc_offload_suggest_gpu_layers(uint64_t model_size_bytes,
                                       uint64_t available_vram_bytes);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OFFLOAD_H */
