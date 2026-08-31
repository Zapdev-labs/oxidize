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
    /* Partial GPU offload is not implemented: per-layer forward is not
     * exposed, so gpu_layers > 0 would run the whole model on GPU and
     * then again on CPU. Reject it until a real split path exists. */
    if (cfg->gpu_layers > 0) return OC_ERR_INVALID_ARG;
    memset(pipe, 0, sizeof(*pipe));
    pipe->config = *cfg;
    pipe->model = model;

    /* Allocate transfer buffer (n_embd floats). */
    pipe->transfer_buf = calloc(cfg->n_embd, sizeof(float));
    if (!pipe->transfer_buf) return OC_ERR_OOM;

    /* Check if CUDA is available (informational; gpu_layers must be 0). */
    pipe->cuda_active = oc_cuda_available();

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

    /* CPU-only path: oc_llama_forward advances sess->pos itself. */
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


/* Extract layer index from a tensor name.
 * Handles "blk.N." and "model.layers.N." prefixes.
 * Returns SIZE_MAX if not a layer tensor. */
static size_t layer_index_from_name(const char *name)
{
    if (!name) return SIZE_MAX;

    if (strncmp(name, "blk.", 4) == 0) {
        return strtoul(name + 4, NULL, 10);
    }
    if (strncmp(name, "model.layers.", 13) == 0) {
        return strtoul(name + 13, NULL, 10);
    }
    return SIZE_MAX;
}

/* Collect unique layer indices from tensor list. */
static size_t *collect_layer_indices(const OcPlanTensorInfo *tensors,
                                      size_t n_tensors, size_t *out_n)
{
    *out_n = 0;
    if (!tensors || n_tensors == 0) return NULL;

    size_t *indices = malloc(n_tensors * sizeof(size_t));
    if (!indices) return NULL;

    for (size_t i = 0; i < n_tensors; i++) {
        size_t idx = tensors[i].layer_index;
        if (idx == SIZE_MAX) {
            idx = layer_index_from_name(tensors[i].name);
        }
        if (idx == SIZE_MAX) continue;

        /* Check if already in list. */
        bool found = false;
        for (size_t j = 0; j < *out_n; j++) {
            if (indices[j] == idx) { found = true; break; }
        }
        if (!found)
            indices[(*out_n)++] = idx;
    }

    /* Sort. */
    for (size_t i = 0; i < *out_n; i++) {
        for (size_t j = i + 1; j < *out_n; j++) {
            if (indices[j] < indices[i]) {
                size_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    return indices;
}

static size_t tensor_parallel_gpu_index(const char *name, size_t gpu_count)
{
    uint64_t hash = 0;
    if (name) {
        for (const char *p = name; *p; p++) {
            hash = hash * 16777619ULL + (uint64_t)(unsigned char)*p;
        }
    }
    return (size_t)(hash % gpu_count);
}

OcLayerOffloadPlan oc_plan_layer_offload(const OcPlanTensorInfo *tensors,
                                          size_t n_tensors,
                                          size_t n_gpu_layers)
{
    OcLayerOffloadPlan plan = {0, 0, 0, 0};
    if (!tensors || n_tensors == 0) return plan;

    size_t n_layers = 0;
    size_t *layers = collect_layer_indices(tensors, n_tensors, &n_layers);
    plan.total_layers = n_layers;

    size_t selected = n_gpu_layers < n_layers ? n_gpu_layers : n_layers;
    plan.n_gpu_layers = selected;

    /* Count GPU tensors: those in the first `selected` layers. */
    for (size_t i = 0; i < n_tensors; i++) {
        size_t idx = tensors[i].layer_index;
        if (idx == SIZE_MAX)
            idx = layer_index_from_name(tensors[i].name);
        if (idx == SIZE_MAX) continue;

        /* Check if idx is in the first `selected` layers. */
        for (size_t j = 0; j < selected; j++) {
            if (layers && layers[j] == idx) {
                plan.gpu_tensor_count++;
                break;
            }
        }
    }

    plan.cpu_tensor_count = n_tensors - plan.gpu_tensor_count;
    free(layers);
    return plan;
}

OcError oc_plan_multi_gpu_offload(const OcPlanTensorInfo *tensors,
                                   size_t n_tensors,
                                   const OcMultiGpuConfig *config,
                                   OcMultiGpuOffloadPlan *out)
{
    if (!config || !out) return OC_ERR_INVALID_ARG;
    if (config->gpu_count == 0) return OC_ERR_INVALID_ARG;

    memset(out, 0, sizeof(*out));
    out->strategy = config->strategy;

    if (!tensors || n_tensors == 0) return OC_OK;

    size_t n_layers = 0;
    size_t *layers = collect_layer_indices(tensors, n_tensors, &n_layers);
    out->total_layers = n_layers;

    size_t selected = config->n_gpu_layers < n_layers ?
                      config->n_gpu_layers : n_layers;
    out->n_gpu_layers = selected;

    /* Compute pipeline stage ranges. */
    size_t base = selected / config->gpu_count;
    size_t remainder = selected % config->gpu_count;

    /* Per-GPU counts. */
    size_t *layer_counts = calloc(config->gpu_count, sizeof(size_t));
    size_t *tensor_counts = calloc(config->gpu_count, sizeof(size_t));
    if (!layer_counts || !tensor_counts) {
        free(layers); free(layer_counts); free(tensor_counts);
        return OC_ERR_OOM;
    }

    /* Pipeline stage mapping: layer -> gpu_index. */
    size_t *stage_map = NULL;
    if (selected > 0) {
        stage_map = malloc(n_layers * sizeof(size_t));
        if (stage_map) {
            size_t start = 0;
            for (size_t g = 0; g < config->gpu_count; g++) {
                size_t width = base + (g < remainder ? 1 : 0);
                for (size_t i = start; i < start + width && i < selected; i++)
                    stage_map[i] = g;
                start += width;
            }
        }
    }

    /* Count tensors per GPU. */
    size_t total_gpu = 0;
    for (size_t i = 0; i < n_tensors; i++) {
        size_t idx = tensors[i].layer_index;
        if (idx == SIZE_MAX)
            idx = layer_index_from_name(tensors[i].name);
        if (idx == SIZE_MAX) continue;

        /* Check if in selected layers. */
        size_t sel_idx = SIZE_MAX;
        for (size_t j = 0; j < selected; j++) {
            if (layers && layers[j] == idx) {
                sel_idx = j;
                break;
            }
        }
        if (sel_idx == SIZE_MAX) continue;

        size_t gpu_idx;
        if (config->strategy == OC_PARALLELISM_TENSOR) {
            gpu_idx = tensor_parallel_gpu_index(tensors[i].name, config->gpu_count);
        } else {
            gpu_idx = stage_map ? stage_map[sel_idx] : 0;
        }
        tensor_counts[gpu_idx]++;
        total_gpu++;
    }

    /* Count layers per GPU. */
    for (size_t i = 0; i < selected; i++) {
        size_t gpu_idx;
        if (config->strategy == OC_PARALLELISM_TENSOR) {
            gpu_idx = layers ? layers[i] % config->gpu_count : 0;
        } else {
            gpu_idx = stage_map ? stage_map[i] : 0;
        }
        layer_counts[gpu_idx]++;
    }

    out->total_gpu_tensor_count = total_gpu;
    out->cpu_tensor_count = n_tensors - total_gpu;

    /* Build GPU assignments. */
    out->gpu_assignments = malloc(config->gpu_count * sizeof(OcGpuAssignment));
    if (out->gpu_assignments) {
        for (size_t g = 0; g < config->gpu_count; g++) {
            out->gpu_assignments[g].gpu_index = g;
            out->gpu_assignments[g].layer_count = layer_counts[g];
            out->gpu_assignments[g].tensor_count = tensor_counts[g];
        }
        out->n_gpu_assignments = config->gpu_count;
    }

    /* Build pipeline stages. */
    if (config->strategy == OC_PARALLELISM_PIPELINE && selected > 0) {
        out->pipeline_stages = calloc(config->gpu_count, sizeof(OcPipelineStage));
        if (out->pipeline_stages) {
            size_t start = 0;
            for (size_t g = 0; g < config->gpu_count; g++) {
                size_t width = base + (g < remainder ? 1 : 0);
                out->pipeline_stages[g].gpu_index = g;
                if (width > 0 && start < selected) {
                    out->pipeline_stages[g].start_layer = layers ? layers[start] : 0;
                    out->pipeline_stages[g].end_layer = layers ? layers[start + width - 1] : 0;
                } else {
                    out->pipeline_stages[g].start_layer = SIZE_MAX;
                    out->pipeline_stages[g].end_layer = SIZE_MAX;
                }
                out->pipeline_stages[g].layer_count = width;
                out->pipeline_stages[g].tensor_count = tensor_counts[g];
                start += width;
            }
            out->n_pipeline_stages = config->gpu_count;
        }
    }

    free(layers);
    free(layer_counts);
    free(tensor_counts);
    free(stage_map);
    return OC_OK;
}

void oc_multi_gpu_plan_free(OcMultiGpuOffloadPlan *plan)
{
    if (!plan) return;
    free(plan->gpu_assignments);
    free(plan->pipeline_stages);
    memset(plan, 0, sizeof(*plan));
}

bool oc_layer_offload_has_gpu_tensors(const OcLayerOffloadPlan *plan)
{
    return plan && plan->gpu_tensor_count > 0;
}
