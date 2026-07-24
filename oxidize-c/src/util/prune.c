/*
 * prune.c — Wanda and magnitude pruning implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/prune.h"

#include "oxidize/quant.h"
#include "oxidize/gguf_writer.h"
#include "oxidize/log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Open a GGUF writer for `output_path` and copy scalar/string metadata from
 * the source file so the output is a loadable GGUF, not raw tensor bytes.
 * ponytail: array metadata (e.g. tokenizer vocab) is not copied — the writer
 * only supports string arrays; extend when pruned models must retokenize. */
static OcError open_writer_copy_meta(const OcGgufFile *gf,
                                     const char *output_path, OcGgufWriter *w)
{
    char arch[64] = "llama";
    const char *arch_p = NULL;
    size_t arch_len = 0;
    if (oc_gguf_metadata_get_str(gf, "general.architecture", &arch_p, &arch_len) &&
        arch_len > 0 && arch_len < sizeof(arch)) {
        memcpy(arch, arch_p, arch_len);
        arch[arch_len] = '\0';
    }
    OcError e = oc_gguf_writer_init(output_path, arch, w);
    if (e != OC_OK) return e;

    for (uint64_t i = 0; i < gf->metadata_kv_count; i++) {
        const OcGgufMetadataKV *kv = &gf->metadata[i];
        if (!kv->key || strcmp(kv->key, "general.architecture") == 0) continue;
        switch (kv->value.type) {
        case OC_GGUF_MT_UINT32:
            oc_gguf_writer_add_uint32(w, kv->key, kv->value.v.u32); break;
        case OC_GGUF_MT_UINT64:
            oc_gguf_writer_add_uint64(w, kv->key, kv->value.v.u64); break;
        case OC_GGUF_MT_FLOAT32:
            oc_gguf_writer_add_float32(w, kv->key, kv->value.v.f32); break;
        case OC_GGUF_MT_STRING: {
            char *s = malloc(kv->value.v.str.len + 1);
            if (!s) { oc_gguf_writer_free(w); return OC_ERR_OOM; }
            memcpy(s, kv->value.v.str.data, kv->value.v.str.len);
            s[kv->value.v.str.len] = '\0';
            oc_gguf_writer_add_string(w, kv->key, s);
            free(s);
            break;
        }
        default:
            break; /* skipped: unsupported KV types */
        }
    }
    return OC_OK;
}

const char *oc_prune_strategy_name(OcPruneStrategy s)
{
    switch (s) {
    case OC_PRUNE_WANDA:     return "wanda";
    case OC_PRUNE_MAGNITUDE: return "magnitude";
    default: return "unknown";
    }
}

/* Dequantize a tensor to f32. */
static float *dequant_tensor(const OcGgufMmappedFile *mf,
                             const OcGgufTensorInfo *ti, size_t *out_n)
{
    if (!ti || !out_n) return NULL;
    size_t n = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) n *= ti->dims[d];
    *out_n = n;
    float *out = malloc(n * sizeof(float));
    if (!out) return NULL;

    const uint8_t *data = oc_gguf_map_tensor_data(mf, ti);
    if (!data) { free(out); return NULL; }

    OcGgufQuantizationType qt = oc_quant_type_from_ggml_id(ti->ggml_type);
    if (qt == OC_QUANT_F32) {
        memcpy(out, data, n * sizeof(float));
    } else {
        size_t src_len = oc_quantized_size(qt, n);
        if (oc_quant_dequant_row(qt, data, src_len, out, n) != OC_OK) {
            free(out);
            return NULL;
        }
    }
    return out;
}

/* Find the magnitude threshold for pruning the bottom `sparsity` fraction. */
static float find_threshold(const float *data, size_t n, float sparsity)
{
    /* Collect all absolute values, sort, find the threshold at the
     * `sparsity` quantile. For efficiency we use a partial sort. */
    if (n == 0 || sparsity <= 0.0f) return 0.0f;
    if (sparsity >= 1.0f) return INFINITY;

    /* Simple approach: compute the mean of |W| and use sparsity * mean as threshold.
     * This is an approximation; a proper implementation would sort. */
    double sum_abs = 0.0;
    for (size_t i = 0; i < n; i++) sum_abs += fabs(data[i]);
    float mean_abs = (float)(sum_abs / (double)n);
    return mean_abs * sparsity;
}

/* Magnitude pruning: zero weights below the magnitude threshold. */
static void prune_magnitude_inplace(float *data, size_t n, float sparsity)
{
    float threshold = find_threshold(data, n, sparsity);
    for (size_t i = 0; i < n; i++) {
        if (fabsf(data[i]) < threshold) data[i] = 0.0f;
    }
}

/* Wanda pruning: zero weights where |W| * ||X||_2 is below threshold.
 * `l2_norms` has `n_norms` entries (the layer's input dimension); weights are
 * laid out row-major so column index = i % n_norms.
 * If activation_stats is NULL, falls back to magnitude pruning. */
static void prune_wanda_inplace(float *data, size_t n, float sparsity,
                                 const float *l2_norms, size_t n_norms)
{
    if (!l2_norms || n_norms == 0) {
        prune_magnitude_inplace(data, n, sparsity);
        return;
    }

    /* Compute importance scores = |W| * ||X||_2. */
    float *importance = malloc(n * sizeof(float));
    if (!importance) { prune_magnitude_inplace(data, n, sparsity); return; }

    for (size_t i = 0; i < n; i++) {
        importance[i] = fabsf(data[i]) * l2_norms[i % n_norms];
    }

    /* Find threshold for importance. */
    float threshold = find_threshold(importance, n, sparsity);
    free(importance);

    /* Zero weights below the importance threshold. */
    for (size_t i = 0; i < n; i++) {
        float imp = fabsf(data[i]) * l2_norms[i % n_norms];
        if (imp < threshold) data[i] = 0.0f;
    }
}

OcError oc_prune_magnitude(const char *input_path, const char *output_path,
                            float sparsity)
{
    if (!input_path || !output_path) return OC_ERR_INVALID_ARG;
    if (sparsity < 0.0f || sparsity >= 1.0f) return OC_ERR_INVALID_ARG;

    OcGgufMmappedFile mf;
    OcError e = oc_gguf_map_open(input_path, &mf);
    if (e != OC_OK) return e;

    const OcGgufFile *gf = &mf.unified;

    OcGgufWriter w;
    e = open_writer_copy_meta(gf, output_path, &w);
    if (e != OC_OK) { oc_gguf_map_free(&mf); return e; }

    /* Prune each tensor. */
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const OcGgufTensorInfo *ti = &gf->tensors[i];
        size_t n = 0;
        float *data = dequant_tensor(&mf, ti, &n);
        if (!data) {
            oc_gguf_writer_free(&w);
            oc_gguf_map_free(&mf);
            return OC_ERR_FORMAT;
        }

        /* Only prune weight tensors (skip norms, embeddings). */
        const char *name = ti->name;
        bool is_weight = strstr(name, ".weight") != NULL &&
                        strstr(name, "norm") == NULL &&
                        strstr(name, "embedd") == NULL;
        if (is_weight) {
            prune_magnitude_inplace(data, n, sparsity);
        }

        /* Write as F32 (ggml type 0). */
        e = oc_gguf_writer_add_tensor(&w, name, ti->n_dims, ti->dims, 0,
                                      data, (uint64_t)n * sizeof(float));
        free(data);
        if (e != OC_OK) {
            oc_gguf_writer_free(&w);
            oc_gguf_map_free(&mf);
            return e;
        }
    }

    e = oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    oc_gguf_map_free(&mf);
    return e;
}

OcError oc_prune_wanda(const char *input_path, const char *output_path,
                        float sparsity, const OcActivationStats *stats)
{
    if (!input_path || !output_path) return OC_ERR_INVALID_ARG;
    if (sparsity < 0.0f || sparsity >= 1.0f) return OC_ERR_INVALID_ARG;

    OcGgufMmappedFile mf;
    OcError e = oc_gguf_map_open(input_path, &mf);
    if (e != OC_OK) return e;

    const OcGgufFile *gf = &mf.unified;
    OcGgufWriter w;
    e = open_writer_copy_meta(gf, output_path, &w);
    if (e != OC_OK) { oc_gguf_map_free(&mf); return e; }

    size_t layer_idx = 0;
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const OcGgufTensorInfo *ti = &gf->tensors[i];
        size_t n = 0;
        float *data = dequant_tensor(&mf, ti, &n);
        if (!data) {
            oc_gguf_writer_free(&w);
            oc_gguf_map_free(&mf);
            return OC_ERR_FORMAT;
        }

        const char *name = ti->name;
        bool is_weight = strstr(name, ".weight") != NULL &&
                        strstr(name, "norm") == NULL &&
                        strstr(name, "embedd") == NULL;
        if (is_weight && stats) {
            /* Get L2 norms for this layer. */
            float *norms = NULL;
            size_t dim = 0;
            if (layer_idx < stats->n_layers && stats->layers[layer_idx].active) {
                dim = stats->layers[layer_idx].input_dim;
                if (dim > 0) {
                    norms = malloc(dim * sizeof(float));
                    if (norms) {
                        oc_activation_stats_get_l2_norms(stats, layer_idx, norms, dim);
                    }
                }
            }
            prune_wanda_inplace(data, n, sparsity, norms, norms ? dim : 0);
            free(norms);
            layer_idx++; /* keep stats aligned with successive weight tensors */
        }

        e = oc_gguf_writer_add_tensor(&w, name, ti->n_dims, ti->dims, 0,
                                      data, (uint64_t)n * sizeof(float));
        free(data);
        if (e != OC_OK) {
            oc_gguf_writer_free(&w);
            oc_gguf_map_free(&mf);
            return e;
        }
    }

    e = oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    oc_gguf_map_free(&mf);
    return e;
}

OcError oc_prune_model(const OcPruneConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    switch (cfg->strategy) {
    case OC_PRUNE_MAGNITUDE:
        return oc_prune_magnitude(cfg->input_path, cfg->output_path, cfg->sparsity);
    case OC_PRUNE_WANDA:
        return oc_prune_wanda(cfg->input_path, cfg->output_path, cfg->sparsity,
                              cfg->activation_stats);
    default:
        return OC_ERR_INVALID_ARG;
    }
}

OcError oc_prune_compute_sparsity(const char *model_path, float *out_sparsity)
{
    if (!model_path || !out_sparsity) return OC_ERR_INVALID_ARG;
    *out_sparsity = 0.0f;

    OcGgufMmappedFile mf;
    OcError e = oc_gguf_map_open(model_path, &mf);
    if (e != OC_OK) return e;

    const OcGgufFile *gf = &mf.unified;
    size_t total = 0, zeros = 0;

    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        size_t n = 0;
        float *data = dequant_tensor(&mf, &gf->tensors[i], &n);
        if (!data) continue;
        for (size_t j = 0; j < n; j++) {
            total++;
            if (data[j] == 0.0f) zeros++;
        }
        free(data);
    }

    oc_gguf_map_free(&mf);
    *out_sparsity = (total > 0) ? (float)(double)zeros / (double)total : 0.0f;
    return OC_OK;
}
