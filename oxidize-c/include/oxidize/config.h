#ifndef OXIDIZE_CONFIG_H
#define OXIDIZE_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum architecture string length (NUL-terminated). */
#define OC_CONFIG_ARCH_LEN 64

/* Maximum RoPE scaling type string length (NUL-terminated). */
#define OC_CONFIG_ROPE_SCALING_LEN 32

/* Maximum FFN type string length (NUL-terminated). */
#define OC_CONFIG_FFN_TYPE_LEN 16

/* Normalized model configuration. */
typedef struct OcModelConfig {
    char     arch[OC_CONFIG_ARCH_LEN];            /* "llama", "qwen2", ... */
    uint32_t n_layers;                             /* transformer block count */
    uint32_t n_heads;                              /* attention head count */
    uint32_t n_kv_heads;                          /* KV head count (GQA: < n_heads) */
    uint32_t head_dim;                            /* per-head dimension */
    uint32_t hidden_dim;                          /* embedding / model dim */
    uint32_t intermediate_dim;                    /* feed-forward hidden dim */
    uint32_t vocab_size;                          /* tokenizer vocabulary size */
    uint32_t n_ctx;                               /* max context length */
    double   rope_theta;                          /* RoPE base frequency */
    char     rope_scaling_type[OC_CONFIG_ROPE_SCALING_LEN]; /* "" if none */
    double   rope_scaling_factor;                /* RoPE scaling factor */
    double   norm_eps;                            /* RMSNorm epsilon */
    char     ffn_type[OC_CONFIG_FFN_TYPE_LEN];    /* "swiglu", "geglu", ... */
    uint32_t n_expert;                            /* 0 = dense, >0 = MoE */
    uint32_t n_expert_used;                       /* active experts per token */
    int32_t  sliding_window;                      /* 0 = full attention, <0 = N/A */
} OcModelConfig;

/* Zero-initialize a config and apply sensible defaults (rope_theta = 10000,
 * norm_eps = 1e-5, ffn_type = "swiglu", sliding_window = 0). Returns OC_OK
 * or OC_ERR_INVALID_ARG. */
OcError oc_model_config_init(OcModelConfig *cfg);

/* Populate `cfg` from a parsed GGUF file's metadata. Reads */
OcError oc_model_config_from_gguf(const OcGgufFile *gguf, OcModelConfig *cfg);

/* Validate the config: required fields present (n_layers, n_heads, */
OcError oc_model_config_validate(const OcModelConfig *cfg);

/* Write a human-readable summary of the config into `out` (up to `out_size-1` */
size_t oc_model_config_print(const OcModelConfig *cfg, char *out, size_t out_size);

/* Return a pointer to the config's arch string (convenience accessor; never
 * NULL — returns "" for a zeroed config). */
const char *oc_model_config_arch_name(const OcModelConfig *cfg);

/* Estimate the total parameter count of the model (embedding + per-layer
 * attention/FFN + output norm + lm_head, accounting for GQA and MoE).
 * Returns 0 if `cfg` is NULL or invalid. */
uint64_t oc_model_config_n_params(const OcModelConfig *cfg);

/* True if the config describes a Mixture-of-Experts model (n_expert > 0). */
bool oc_model_config_is_moe(const OcModelConfig *cfg);

/* True if the config uses Grouped-Query Attention (n_kv_heads < n_heads). */
bool oc_model_config_has_gqa(const OcModelConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CONFIG_H */
