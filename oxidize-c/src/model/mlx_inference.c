#define _POSIX_C_SOURCE 200809L

#include "oxidize/mlx_inference.h"

#include <stddef.h>
#include <string.h>


#if defined(__APPLE__) && defined(__MACH__)
#  define OC_MLX_HOST_MACOS 1
#else
#  define OC_MLX_HOST_MACOS 0
#endif


#define OC_MLX_DEFAULT_HIDDEN_SIZE 4096u
#define OC_MLX_DEFAULT_VOCAB_SIZE  32000u
#define OC_MLX_DEFAULT_N_LAYERS    32u


void oc_mlx_config_init(OcMlxConfig *cfg)
{
    if (cfg == NULL) {
        return;
    }
    cfg->hidden_size = OC_MLX_DEFAULT_HIDDEN_SIZE;
    cfg->vocab_size  = OC_MLX_DEFAULT_VOCAB_SIZE;
    cfg->n_layers    = OC_MLX_DEFAULT_N_LAYERS;
    cfg->use_metal   = true;
    cfg->model_path[0] = '\0';
}


OcError oc_mlx_engine_init(OcMlxEngine *engine, const OcMlxConfig *cfg)
{
    if (engine == NULL || cfg == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    engine->config    = *cfg;
    engine->loaded    = false;
    engine->available = oc_mlx_is_available();
    return OC_OK;
}

OcError oc_mlx_engine_load(OcMlxEngine *engine, const char *model_path)
{
    if (engine == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (model_path == NULL || model_path[0] == '\0') {
        return OC_ERR_INVALID_ARG;
    }
#if OC_MLX_HOST_MACOS
    /* A real MLX load would happen here. For the C port we still treat
     * macOS as unavailable to keep behaviour deterministic in tests
     * without a real MLX runtime. */
    (void)model_path;
    engine->loaded = false;
    return OC_ERR_BACKEND;
#else
    (void)model_path;
    engine->loaded = false;
    return OC_ERR_BACKEND;
#endif
}

OcError oc_mlx_engine_generate(OcMlxEngine *engine,
                                const uint32_t *tokens, size_t n_tokens,
                                size_t max_new,
                                uint32_t *out_tokens, size_t *out_n)
{
    if (engine == NULL || tokens == NULL || out_tokens == NULL ||
        out_n == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (n_tokens == 0 || max_new == 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (!engine->loaded) {
        return OC_ERR_BACKEND;
    }
    /* Unreachable on non-macOS because load() always fails. On macOS the
     * stub load also fails, so this branch is dead for now. */
    *out_n = 0;
    return OC_ERR_BACKEND;
}


bool oc_mlx_is_available(void)
{
#if OC_MLX_HOST_MACOS
    /* Even on macOS the C port ships stubs; report unavailable until a
     * real MLX binding is wired in. */
    return false;
#else
    return false;
#endif
}

const char *oc_mlx_backend_name(void)
{
    return "mlx";
}

void oc_mlx_engine_free(OcMlxEngine *engine)
{
    if (engine == NULL) {
        return;
    }
    engine->loaded    = false;
    engine->available = false;
    /* No owned heap resources in the current stub. */
}
