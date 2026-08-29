/* merge.c — checkpoint merging implementation. */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/merge.h"

#include "oxidize/quant.h"
#include "oxidize/gguf_writer.h"
#include "oxidize/log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static float *dequant_tensor(const OcGgufMmappedFile *mf,
                             const OcGgufTensorInfo *ti)
{
    if (!ti) return NULL;
    size_t n = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) n *= ti->dims[d];
    float *out = malloc(n * sizeof(float));
    if (!out) return NULL;

    const uint8_t *data = oc_gguf_map_tensor_data(mf, ti);
    if (!data) { free(out); return NULL; }

    OcGgufQuantizationType qt = oc_quant_type_from_ggml_id(ti->ggml_type);
    if (qt == OC_QUANT_F32) {
        memcpy(out, data, n * sizeof(float));
    } else {
        size_t src_len = oc_quantized_size(qt, n);
        OcError e = oc_quant_dequant_row(qt, data, src_len, out, n);
        if (e != OC_OK) { free(out); return NULL; }
    }
    return out;
}

const char *oc_merge_strategy_name(OcMergeStrategy s)
{
    switch (s) {
    case OC_MERGE_LINEAR: return "linear";
    case OC_MERGE_SLERP:  return "slerp";
    case OC_MERGE_TIES:   return "ties";
    case OC_MERGE_DARE:   return "dare";
    default: return "unknown";
    }
}

/* Write a merged F32 buffer to the GGUF writer, re-quantizing to the
 * original tensor's quantization type when possible. Falls back to F32
 * for unsupported types or block-size mismatches. */
static OcError write_merged_tensor(OcGgufWriter *w,
                                   const OcGgufTensorInfo *ti,
                                   const float *merged, size_t n)
{
    OcGgufQuantizationType orig_qtype =
        oc_quant_type_from_ggml_id(ti->ggml_type);

    if (orig_qtype == OC_QUANT_F16) {
        uint8_t *packed = malloc(n * 2);
        if (!packed) return OC_ERR_OOM;
        oc_quant_pack_row(OC_QUANT_F16, merged, n, packed, n * 2);
        OcError e = oc_gguf_writer_add_tensor(w, ti->name, ti->n_dims,
            ti->dims, 1, packed, (uint64_t)n * 2);
        free(packed);
        return e;
    }
    if (orig_qtype == OC_QUANT_BF16) {
        uint8_t *packed = malloc(n * 2);
        if (!packed) return OC_ERR_OOM;
        oc_quant_pack_row(OC_QUANT_BF16, merged, n, packed, n * 2);
        OcError e = oc_gguf_writer_add_tensor(w, ti->name, ti->n_dims,
            ti->dims, 30, packed, (uint64_t)n * 2);
        free(packed);
        return e;
    }
    if (orig_qtype != OC_QUANT_F32 && orig_qtype != OC_QUANT_UNKNOWN) {
        /* Quantized type: compute packed size and re-quantize. */
        OcQuantBlockLayout bs = oc_quant_block_size(orig_qtype);
        if (bs.elements_per_block > 0 && n % bs.elements_per_block == 0) {
            size_t n_blocks = n / bs.elements_per_block;
            size_t packed_size = n_blocks * bs.bytes_per_block;
            uint8_t *packed = malloc(packed_size);
            if (!packed) return OC_ERR_OOM;
            OcError qe = oc_quant_pack_row(orig_qtype, merged, n,
                                          packed, packed_size);
            if (qe == OC_OK) {
                OcError e = oc_gguf_writer_add_tensor(w, ti->name,
                    ti->n_dims, ti->dims, ti->ggml_type,
                    packed, (uint64_t)packed_size);
                free(packed);
                return e;
            }
            free(packed);
            /* Fall back to F32. */
        }
    }
    /* F32 or fallback. */
    return oc_gguf_writer_add_tensor(w, ti->name, ti->n_dims,
        ti->dims, 0, merged, (uint64_t)n * sizeof(float));
}


OcError oc_merge_linear(const OcMergeInput *inputs, size_t n_inputs,
                         const char *output_path)
{
    if (!inputs || n_inputs < 2 || !output_path)
        return OC_ERR_INVALID_ARG;

    /* Open all input files. */
    OcGgufMmappedFile *mfs = calloc(n_inputs, sizeof(OcGgufMmappedFile));
    if (!mfs) return OC_ERR_OOM;

    for (size_t i = 0; i < n_inputs; i++) {
        OcError e = oc_gguf_map_open(inputs[i].path, &mfs[i]);
        if (e != OC_OK) {
            for (size_t j = 0; j < i; j++) oc_gguf_map_free(&mfs[j]);
            free(mfs);
            return e;
        }
    }

    /* Use first file as template for metadata + tensor table. */
    const OcGgufFile *tmpl = &mfs[0].unified;

    /* Compute total weight. */
    double total_weight = 0.0;
    for (size_t i = 0; i < n_inputs; i++)
        total_weight += (double)inputs[i].weight;
    if (total_weight <= 0.0) total_weight = 1.0;

    /* Open output. */
    OcGgufWriter w;
    OcError we = oc_gguf_writer_init_from_file(output_path, tmpl, &w);
    if (we != OC_OK) {
        for (size_t i = 0; i < n_inputs; i++) oc_gguf_map_free(&mfs[i]);
        free(mfs);
        return we;
    }

    /* Merge each tensor. */
    for (uint64_t t = 0; t < tmpl->tensor_count && we == OC_OK; t++) {
        const OcGgufTensorInfo *ti = &tmpl->tensors[t];
        size_t n = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) n *= ti->dims[d];

        /* Dequantize all inputs for this tensor. */
        float **bufs = calloc(n_inputs, sizeof(float *));
        bool ok = bufs != NULL;
        for (size_t i = 0; ok && i < n_inputs; i++) {
            const OcGgufTensorInfo *src_ti = oc_gguf_map_tensor_get(&mfs[i], ti->name);
            if (src_ti) {
                bufs[i] = dequant_tensor(&mfs[i], src_ti);
                if (!bufs[i]) ok = false;
            } else {
                bufs[i] = calloc(n, sizeof(float));
                if (!bufs[i]) ok = false;
            }
        }

        if (ok) {
            /* Weighted average. */
            float *merged = calloc(n, sizeof(float));
            if (merged) {
                for (size_t i = 0; i < n_inputs; i++) {
                    float wgt = (float)(inputs[i].weight / total_weight);
                    for (size_t j = 0; j < n; j++)
                        merged[j] += wgt * bufs[i][j];
                }
                /* Re-quantize to original quant type, fall back to F32. */
                we = write_merged_tensor(&w, ti, merged, n);
                free(merged);
            } else {
                we = OC_ERR_OOM;
            }
        } else {
            we = OC_ERR_FORMAT;
        }

        if (bufs) {
            for (size_t i = 0; i < n_inputs; i++) free(bufs[i]);
            free(bufs);
        }
    }

    if (we == OC_OK) we = oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    for (size_t i = 0; i < n_inputs; i++) oc_gguf_map_free(&mfs[i]);
    free(mfs);
    return we;
}


OcError oc_merge_slerp(const char *path_a, const char *path_b,
                        float t, const char *output_path)
{
    if (!path_a || !path_b || !output_path) return OC_ERR_INVALID_ARG;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    OcGgufMmappedFile mfa, mfb;
    OcError e = oc_gguf_map_open(path_a, &mfa);
    if (e != OC_OK) return e;
    e = oc_gguf_map_open(path_b, &mfb);
    if (e != OC_OK) { oc_gguf_map_free(&mfa); return e; }

    const OcGgufFile *fa = &mfa.unified;

    OcGgufWriter w;
    e = oc_gguf_writer_init_from_file(output_path, fa, &w);
    if (e != OC_OK) { oc_gguf_map_free(&mfa); oc_gguf_map_free(&mfb); return e; }

    /* Merge each tensor via SLERP. */
    for (uint64_t ti_idx = 0; ti_idx < fa->tensor_count && e == OC_OK; ti_idx++) {
        const OcGgufTensorInfo *ta = &fa->tensors[ti_idx];
        const OcGgufTensorInfo *tb = oc_gguf_map_tensor_get(&mfb, ta->name);
        if (!tb) { e = OC_ERR_FORMAT; break; }

        size_t n = 1;
        for (uint32_t d = 0; d < ta->n_dims; d++) n *= ta->dims[d];

        float *va = dequant_tensor(&mfa, ta);
        float *vb = dequant_tensor(&mfb, tb);
        if (!va || !vb) { free(va); free(vb); e = OC_ERR_FORMAT; break; }

        /* SLERP: result = va + vb * sin(t * theta) / sin(theta) - va * sin((1-t) * theta) / sin(theta)
         * Simplified: for small angles, use linear interpolation as fallback. */
        float *merged = malloc(n * sizeof(float));
        if (merged) {
            for (size_t j = 0; j < n; j++) {
                float a = va[j], b = vb[j];
                float dot = a * b;
                float norm_a = fabsf(a);
                float norm_b = fabsf(b);
                if (norm_a < 1e-10f || norm_b < 1e-10f) {
                    merged[j] = (1.0f - t) * a + t * b;
                } else {
                    float cos_theta = dot / (norm_a * norm_b);
                    if (cos_theta > 1.0f) cos_theta = 1.0f;
                    if (cos_theta < -1.0f) cos_theta = -1.0f;
                    float theta = acosf(cos_theta);
                    if (theta < 1e-6f) {
                        merged[j] = (1.0f - t) * a + t * b;
                    } else {
                        float sin_theta = sinf(theta);
                        float w_a = sinf((1.0f - t) * theta) / sin_theta;
                        float w_b = sinf(t * theta) / sin_theta;
                        merged[j] = w_a * a + w_b * b;
                    }
                }
            }
            e = write_merged_tensor(&w, ta, merged, n);
            free(merged);
        } else {
            e = OC_ERR_OOM;
        }
        free(va);
        free(vb);
    }

    if (e == OC_OK) e = oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    oc_gguf_map_free(&mfa);
    oc_gguf_map_free(&mfb);
    return e;
}


OcError oc_merge_ties(const OcMergeInput *inputs, size_t n_inputs,
                       float density, const char *output_path)
{
    if (!inputs || n_inputs < 2 || !output_path) return OC_ERR_INVALID_ARG;
    if (density < 0.0f) density = 0.0f;
    if (density > 1.0f) density = 1.0f;

    /* TIES: Trim, Elect, Combine. */

    OcGgufMmappedFile *mfs = calloc(n_inputs, sizeof(OcGgufMmappedFile));
    if (!mfs) return OC_ERR_OOM;
    for (size_t i = 0; i < n_inputs; i++) {
        OcError e = oc_gguf_map_open(inputs[i].path, &mfs[i]);
        if (e != OC_OK) {
            for (size_t j = 0; j < i; j++) oc_gguf_map_free(&mfs[j]);
            free(mfs);
            return e;
        }
    }

    const OcGgufFile *tmpl = &mfs[0].unified;
    OcGgufWriter w;
    OcError we = oc_gguf_writer_init_from_file(output_path, tmpl, &w);
    if (we != OC_OK) {
        for (size_t i = 0; i < n_inputs; i++) oc_gguf_map_free(&mfs[i]);
        free(mfs);
        return we;
    }

    for (uint64_t ti_idx = 0; ti_idx < tmpl->tensor_count && we == OC_OK; ti_idx++) {
        const OcGgufTensorInfo *ti = &tmpl->tensors[ti_idx];
        size_t n = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) n *= ti->dims[d];

        float **bufs = calloc(n_inputs, sizeof(float *));
        if (!bufs) { we = OC_ERR_OOM; break; }
        for (size_t i = 0; i < n_inputs; i++) {
            const OcGgufTensorInfo *src = oc_gguf_map_tensor_get(&mfs[i], ti->name);
            bufs[i] = src ? dequant_tensor(&mfs[i], src) : calloc(n, sizeof(float));
        }

        /* Step 1: Trim - compute magnitude threshold and zero out small values. */
        if (density < 1.0f) {
            for (size_t i = 0; i < n_inputs; i++) {
                if (!bufs[i]) continue;
                /* Simple: zero the bottom (1-density) fraction by magnitude. */
                /* For efficiency, use mean magnitude as threshold. */
                float sum_abs = 0.0f;
                for (size_t j = 0; j < n; j++) sum_abs += fabsf(bufs[i][j]);
                float threshold = sum_abs / (float)n * (1.0f - density);
                for (size_t j = 0; j < n; j++) {
                    if (fabsf(bufs[i][j]) < threshold) bufs[i][j] = 0.0f;
                }
            }
        }

        /* Step 2+3: Elect sign and combine. */
        float *merged = calloc(n, sizeof(float));
        if (merged) {
            for (size_t j = 0; j < n; j++) {
                float pos_sum = 0.0f, neg_sum = 0.0f;
                int pos_count = 0, neg_count = 0;
                for (size_t i = 0; i < n_inputs; i++) {
                    if (!bufs[i]) continue;
                    float v = bufs[i][j];
                    if (v > 0) { pos_sum += v; pos_count++; }
                    else if (v < 0) { neg_sum += v; neg_count++; }
                }
                /* Elect the sign with larger cumulative magnitude. */
                if (pos_count > 0 && fabsf(pos_sum) >= fabsf(neg_sum)) {
                    merged[j] = pos_sum / (float)pos_count;
                } else if (neg_count > 0) {
                    merged[j] = neg_sum / (float)neg_count;
                }
            }
            we = write_merged_tensor(&w, ti, merged, n);
            free(merged);
        } else {
            we = OC_ERR_OOM;
        }

        for (size_t i = 0; i < n_inputs; i++) free(bufs[i]);
        free(bufs);
    }

    if (we == OC_OK) we = oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    for (size_t i = 0; i < n_inputs; i++) oc_gguf_map_free(&mfs[i]);
    free(mfs);
    return we;
}


OcError oc_merge_dare(const OcMergeInput *inputs, size_t n_inputs,
                      float drop_rate, const char *output_path)
{
    if (!inputs || n_inputs < 2 || !output_path) return OC_ERR_INVALID_ARG;
    if (drop_rate < 0.0f) drop_rate = 0.0f;
    if (drop_rate > 1.0f) drop_rate = 1.0f;

    /* DARE: Drop And Rescale. */

    OcGgufMmappedFile *mfs = calloc(n_inputs, sizeof(OcGgufMmappedFile));
    if (!mfs) return OC_ERR_OOM;
    for (size_t i = 0; i < n_inputs; i++) {
        OcError e = oc_gguf_map_open(inputs[i].path, &mfs[i]);
        if (e != OC_OK) {
            for (size_t j = 0; j < i; j++) oc_gguf_map_free(&mfs[j]);
            free(mfs);
            return e;
        }
    }

    const OcGgufFile *tmpl = &mfs[0].unified;
    OcGgufWriter w;
    OcError we = oc_gguf_writer_init_from_file(output_path, tmpl, &w);
    if (we != OC_OK) {
        for (size_t i = 0; i < n_inputs; i++) oc_gguf_map_free(&mfs[i]);
        free(mfs);
        return we;
    }

    float rescale = (drop_rate < 1.0f) ? 1.0f / (1.0f - drop_rate) : 1.0f;
    uint64_t rng = 0x1234567890abcdefULL;

    for (uint64_t ti_idx = 0; ti_idx < tmpl->tensor_count && we == OC_OK; ti_idx++) {
        const OcGgufTensorInfo *ti = &tmpl->tensors[ti_idx];
        size_t n = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) n *= ti->dims[d];

        float **bufs = calloc(n_inputs, sizeof(float *));
        if (!bufs) { we = OC_ERR_OOM; break; }
        for (size_t i = 0; i < n_inputs; i++) {
            const OcGgufTensorInfo *src = oc_gguf_map_tensor_get(&mfs[i], ti->name);
            bufs[i] = src ? dequant_tensor(&mfs[i], src) : calloc(n, sizeof(float));
            /* Drop and rescale. */
            if (bufs[i]) {
                for (size_t j = 0; j < n; j++) {
                    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                    if ((double)(rng >> 33) / (double)(1ULL << 31) < drop_rate) {
                        bufs[i][j] = 0.0f;
                    } else {
                        bufs[i][j] *= rescale;
                    }
                }
            }
        }

        /* Average. */
        float *merged = calloc(n, sizeof(float));
        if (merged) {
            for (size_t i = 0; i < n_inputs; i++) {
                if (!bufs[i]) continue;
                for (size_t j = 0; j < n; j++)
                    merged[j] += bufs[i][j];
            }
            float inv = 1.0f / (float)n_inputs;
            for (size_t j = 0; j < n; j++)
                merged[j] *= inv;
            we = write_merged_tensor(&w, ti, merged, n);
            free(merged);
        } else {
            we = OC_ERR_OOM;
        }

        for (size_t i = 0; i < n_inputs; i++) free(bufs[i]);
        free(bufs);
    }

    if (we == OC_OK) we = oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    for (size_t i = 0; i < n_inputs; i++) oc_gguf_map_free(&mfs[i]);
    free(mfs);
    return we;
}


OcError oc_merge_models(const OcMergeConfig *cfg)
{
    if (!cfg || !cfg->inputs || cfg->n_inputs < 2) return OC_ERR_INVALID_ARG;

    if (cfg->verbose) {
        oc_log(OC_LOG_INFO, "merge: strategy=%s inputs=%zu output=%s",
               oc_merge_strategy_name(cfg->strategy), cfg->n_inputs,
               cfg->output_path ? cfg->output_path : "(null)");
        for (size_t i = 0; i < cfg->n_inputs; i++)
            oc_log(OC_LOG_INFO, "merge: input[%zu]=%s weight=%.3f",
                   i, cfg->inputs[i].path, (double)cfg->inputs[i].weight);
    }

    switch (cfg->strategy) {
    case OC_MERGE_LINEAR:
        return oc_merge_linear(cfg->inputs, cfg->n_inputs, cfg->output_path);
    case OC_MERGE_SLERP:
        if (cfg->n_inputs != 2) return OC_ERR_INVALID_ARG;
        return oc_merge_slerp(cfg->inputs[0].path, cfg->inputs[1].path,
                              cfg->slerp_t, cfg->output_path);
    case OC_MERGE_TIES:
        return oc_merge_ties(cfg->inputs, cfg->n_inputs, cfg->ties_density,
                             cfg->output_path);
    case OC_MERGE_DARE:
        return oc_merge_dare(cfg->inputs, cfg->n_inputs,
                             cfg->dare_drop_rate, cfg->output_path);
    default:
        return OC_ERR_INVALID_ARG;
    }
}
