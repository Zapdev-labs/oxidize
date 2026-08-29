/* mlx_inference.h — MLX (Apple Metal) inference engine, macOS stub. */
#ifndef OXIDIZE_MLX_INFERENCE_H
#define OXIDIZE_MLX_INFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t n_layers;
    bool     use_metal;
    char     model_path[256];
} OcMlxConfig;


typedef struct {
    OcMlxConfig config;
    bool        loaded;
    bool        available;   /* false on non-macOS */
} OcMlxEngine;

/* Initialize a config with defaults:
 *   hidden_size=4096, vocab_size=32000, n_layers=32,
 *   use_metal=true, model_path="". */
void oc_mlx_config_init(OcMlxConfig *cfg);

/* Initialize an engine from a config. Sets `available` from the host; non-macOS still returns OC_OK (engine valid but unavailable). Returns OC_ERR_INVALID_ARG if any pointer is NULL. */
OcError oc_mlx_engine_init(OcMlxEngine *engine, const OcMlxConfig *cfg);

/* Load a model from `model_path`. NULL or empty path is OC_ERR_INVALID_ARG on every platform; non-macOS (or no Metal) returns OC_ERR_BACKEND without mutating `loaded`. */
OcError oc_mlx_engine_load(OcMlxEngine *engine, const char *model_path);

/* Generate tokens. NULL pointers, n_tokens==0, or max_new==0 return OC_ERR_INVALID_ARG; an unloaded engine returns OC_ERR_BACKEND. */
OcError oc_mlx_engine_generate(OcMlxEngine *engine,
                                const uint32_t *tokens, size_t n_tokens,
                                size_t max_new,
                                uint32_t *out_tokens, size_t *out_n);

/* True iff the MLX backend is usable on this host (macOS only).
 * Always false on non-macOS. */
bool oc_mlx_is_available(void);

/* Static backend name string. Always "mlx". */
const char *oc_mlx_backend_name(void);

/* Free an engine. Safe on NULL. Resets loaded/available flags. */
void oc_mlx_engine_free(OcMlxEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MLX_INFERENCE_H */
