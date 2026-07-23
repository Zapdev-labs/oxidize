/*
 * mlx_inference.h — MLX (Apple Metal) inference engine, macOS stub.
 *
 * Port of oxidize-core/src/model/mlx_inference.rs. On macOS this would
 * bind the MLX framework; on all other platforms every function is a
 * stub that returns OC_ERR_BACKEND and oc_mlx_is_available() returns
 * false. The structs and ABI are identical regardless of platform so
 * that callers can compile against this header unconditionally.
 */
#ifndef OXIDIZE_MLX_INFERENCE_H
#define OXIDIZE_MLX_INFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Config ──────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t n_layers;
    bool     use_metal;
    char     model_path[256];
} OcMlxConfig;

/* ─── Engine ──────────────────────────────────────────────────────────── */

typedef struct {
    OcMlxConfig config;
    bool        loaded;
    bool        available;   /* false on non-macOS */
} OcMlxEngine;

/* Initialize a config with defaults:
 *   hidden_size=4096, vocab_size=32000, n_layers=32,
 *   use_metal=true, model_path="". */
void oc_mlx_config_init(OcMlxConfig *cfg);

/* Initialize an engine from a config. Sets `available` based on the
 * host platform. On non-macOS this still returns OC_OK (the engine
 * object is valid but unavailable); load/generate will fail later.
 * Returns OC_ERR_INVALID_ARG if any pointer is NULL. */
OcError oc_mlx_engine_init(OcMlxEngine *engine, const OcMlxConfig *cfg);

/* Load a model from `model_path`. On non-macOS (or when use_metal is
 * false on a non-Metal machine) this returns OC_ERR_BACKEND and does
 * not mutate `loaded`. On macOS, a NULL or empty path returns
 * OC_ERR_INVALID_ARG. */
OcError oc_mlx_engine_load(OcMlxEngine *engine, const char *model_path);

/* Generate tokens. `tokens` is the prompt of length n_tokens; up to
 * `max_new` new tokens are written to `out_tokens` and the actual
 * count to `out_n`. On non-macOS this returns OC_ERR_BACKEND.
 * Returns OC_ERR_INVALID_ARG on NULL pointers, max_new==0, or when
 * the engine is not loaded. */
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
