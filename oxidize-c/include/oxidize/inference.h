/*
 * inference.h — High-level inference engine.
 *
 * Wraps model loading, tokenization, and generation into a clean API.
 * Port from oxidize-core/src/model/inference.rs.
 */
#ifndef OXIDIZE_INFERENCE_H
#define OXIDIZE_INFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/generation.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_INF_MAX_CONTEXT 32768

typedef enum {
    OC_INF_MODEL_LLAMA = 0,
    OC_INF_MODEL_MISTRAL = 1,
    OC_INF_MODEL_GEMMA = 2,
    OC_INF_MODEL_PHI = 3,
    OC_INF_MODEL_GLM = 4,
    OC_INF_MODEL_QWEN = 5,
} OcInfModelType;

typedef struct {
    OcInfModelType model_type;
    const char *model_path;
    uint32_t n_threads;
    uint32_t n_ctx;
    uint32_t n_batch;
    bool use_gpu;
    bool use_numa;
    bool verbose;
} OcInfConfig;

typedef struct {
    void *model;
    void *tokenizer;
    OcInfConfig config;
    bool loaded;
    uint32_t n_loaded_layers;
    size_t model_size_bytes;
} OcInfEngine;

OcError oc_inf_config_init(OcInfConfig *cfg);
OcError oc_inf_engine_init(OcInfEngine *engine, const OcInfConfig *cfg);
OcError oc_inf_engine_load(OcInfEngine *engine, const char *model_path);
OcError oc_inf_engine_generate(OcInfEngine *engine, const char *prompt,
                              const OcGenConfig *gen_cfg,
                              char *out_text, size_t out_size,
                              OcGenResult *result);
OcError oc_inf_engine_encode(OcInfEngine *engine, const char *text,
                            uint32_t **out_tokens, size_t *out_n);
OcError oc_inf_engine_decode(OcInfEngine *engine, const uint32_t *tokens,
                           size_t n, char *out, size_t out_size);
OcError oc_inf_engine_stats(const OcInfEngine *engine,
                           char *out, size_t out_size);
bool oc_inf_engine_is_loaded(const OcInfEngine *engine);
OcInfModelType oc_inf_model_type_from_arch(const char *arch_name);
const char *oc_inf_model_type_name(OcInfModelType type);
const char *oc_inf_model_type_arch(OcInfModelType type);
void oc_inf_engine_free(OcInfEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_INFERENCE_H */
