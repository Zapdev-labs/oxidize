/* turboquant.c — Fast online quantization with calibration. */
#include "oxidize/turboquant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


OcTurboQuantConfig oc_turboquant_config_default(OcGgufQuantizationType target_type)
{
    OcTurboQuantConfig cfg;
    cfg.target_type          = target_type;
    cfg.calibration_samples = OC_TQ_DEFAULT_CALIBRATION_SAMPLES;
    cfg.block_size           = OC_TQ_DEFAULT_BLOCK_SIZE;
    cfg.importance_weighting = true;
    return cfg;
}

OcError oc_turboquant_init(OcTurboQuant *tq, OcTurboQuantConfig config)
{
    if (!tq) return OC_ERR_INVALID_ARG;
    memset(tq, 0, sizeof(*tq));
    tq->config          = config;
    tq->calibration_cap = config.calibration_samples;
    tq->scale_adjust    = 1.0f;

    if (tq->calibration_cap > 0) {
        tq->calibration_data = malloc(tq->calibration_cap * sizeof(float));
        if (!tq->calibration_data) return OC_ERR_OOM;
    }
    return OC_OK;
}

OcError oc_turboquant_calibrate(OcTurboQuant *tq, const float *data, size_t n)
{
    if (!tq || !data) return OC_ERR_INVALID_ARG;

    /* Accumulate calibration samples up to the cap. If we exceed the cap,
     * we keep the first cap samples (simple ring-free truncation). */
    for (size_t i = 0; i < n; i++) {
        if (tq->calibration_count >= tq->calibration_cap) break;
        tq->calibration_data[tq->calibration_count++] = data[i];
    }

    /* Compute scale adjustment: the ratio of the calibration max-abs to
     * the expected dynamic range. This biases per-block scales to better
     * fit the data distribution. */
    if (tq->calibration_count > 0 && tq->config.importance_weighting) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < tq->calibration_count; i++) {
            float a = fabsf(tq->calibration_data[i]);
            if (a > max_abs) max_abs = a;
        }
        /* Compute the mean absolute value to detect skew. */
        double sum_abs = 0.0;
        for (size_t i = 0; i < tq->calibration_count; i++) {
            sum_abs += fabs((double)tq->calibration_data[i]);
        }
        double mean_abs = sum_abs / (double)tq->calibration_count;
        /* If the data is highly skewed (mean << max), boost scales to
         * preserve resolution for the common range. */
        if (max_abs > 0.0f && mean_abs > 0.0) {
            double skew = mean_abs / (double)max_abs;
            /* skew ~1.0 means uniform; skew <<1 means sparse. Scale adjust
             * compresses the range slightly for sparse data. */
            tq->scale_adjust = (float)(0.5 + 0.5 * skew);
        }
    }
    return OC_OK;
}

OcError oc_turboquant_quantize(OcTurboQuant *tq, const float *input,
                               size_t n, uint8_t *output, size_t *out_size)
{
    if (!tq || !input || !output || !out_size) return OC_ERR_INVALID_ARG;

    /* Validate the target type is a known quant type with a pack path. */
    OcQuantBlockLayout layout = oc_quant_block_size(tq->config.target_type);
    if (layout.elements_per_block == 0) return OC_ERR_QUANT;
    if (n % layout.elements_per_block != 0) return OC_ERR_INVALID_ARG;

    size_t expected = oc_quantized_size(tq->config.target_type, n);
    if (expected == 0) return OC_ERR_INVALID_ARG;

    /* Apply scale adjustment: pre-scale the input, then pack. This is a
     * lightweight form of calibration-based rescaling. */
    float *scaled = NULL;
    const float *src = input;
    if (tq->scale_adjust != 1.0f) {
        scaled = malloc(n * sizeof(float));
        if (!scaled) return OC_ERR_OOM;
        for (size_t i = 0; i < n; i++) {
            scaled[i] = input[i] * tq->scale_adjust;
        }
        src = scaled;
    }

    OcError err = oc_quant_pack_row(tq->config.target_type, src, n,
                                    output, expected);

    if (scaled) free(scaled);
    if (err != OC_OK) return err;

    *out_size = expected;

    /* Update statistics by dequantizing and comparing. */
    float *dequant = malloc(n * sizeof(float));
    if (dequant) {
        OcError derr = oc_quant_dequant_row_scalar(tq->config.target_type,
                                                   output, expected,
                                                   dequant, n);
        if (derr == OC_OK) {
            double sum_sq_err = 0.0;
            double sum_abs_err = 0.0;
            double max_err = 0.0;
            for (size_t i = 0; i < n; i++) {
                float orig = input[i] * tq->scale_adjust;
                float dq   = dequant[i];
                double diff = (double)orig - (double)dq;
                double abs_diff = fabs(diff);
                sum_sq_err += diff * diff;
                sum_abs_err += abs_diff;
                if (abs_diff > max_err) max_err = abs_diff;
            }
            /* Accumulate into the running stats. */
            size_t prev_blocks = tq->stats.n_blocks;
            size_t new_blocks  = n / layout.elements_per_block;
            size_t total_blocks = prev_blocks + new_blocks;

            /* Running MSE: weighted average of old and new. */
            if (total_blocks > 0) {
                double old_mse = tq->stats.mse * (double)prev_blocks;
                double new_mse = (sum_sq_err / (double)n) * (double)new_blocks;
                tq->stats.mse = (old_mse + new_mse) / (double)total_blocks;

                double old_avg = tq->stats.avg_error * (double)prev_blocks;
                double new_avg = (sum_abs_err / (double)n) * (double)new_blocks;
                tq->stats.avg_error = (old_avg + new_avg) / (double)total_blocks;
            }
            if (max_err > tq->stats.max_error) {
                tq->stats.max_error = max_err;
            }
            tq->stats.n_blocks = total_blocks;
        }
        free(dequant);
    }

    return OC_OK;
}

OcError oc_turboquant_stats(const OcTurboQuant *tq,
                            OcTurboQuantStats *out_stats)
{
    if (!tq || !out_stats) return OC_ERR_INVALID_ARG;
    *out_stats = tq->stats;
    return OC_OK;
}

void oc_turboquant_free(OcTurboQuant *tq)
{
    if (!tq) return;
    free(tq->calibration_data);
    memset(tq, 0, sizeof(*tq));
}
