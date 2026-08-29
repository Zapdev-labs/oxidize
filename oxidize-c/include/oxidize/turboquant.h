/* turboquant.h — Fast online quantization with calibration. */
#ifndef OXIDIZE_TURBOQUANT_H
#define OXIDIZE_TURBOQUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/quant.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_TQ_DEFAULT_CALIBRATION_SAMPLES 128u
#define OC_TQ_DEFAULT_BLOCK_SIZE          32u


/* Quality statistics collected across all quantize calls. */
typedef struct OcTurboQuantStats {
    double mse;          /* mean squared error                       */
    double max_error;    /* maximum absolute error                   */
    double avg_error;     /* average absolute error                   */
    size_t n_blocks;      /* number of blocks quantized              */
} OcTurboQuantStats;

/* Configuration. Use oc_turboquant_config_default() for sensible defaults. */
typedef struct OcTurboQuantConfig {
    OcGgufQuantizationType target_type;     /* target quant type          */
    size_t calibration_samples;            /* default 128                 */
    size_t block_size;                     /* default 32                  */
    bool   importance_weighting;           /* use calibration for weights */
} OcTurboQuantConfig;

/* Quantizer state. Calibration data is accumulated until
 * calibration_samples is reached. */
typedef struct OcTurboQuant {
    OcTurboQuantConfig  config;
    float              *calibration_data;   /* accumulated calibration     */
    size_t              calibration_count;   /* values stored so far        */
    size_t              calibration_cap;     /* capacity of calibration_data */
    float               scale_adjust;       /* derived from calibration     */
    OcTurboQuantStats   stats;
} OcTurboQuant;


/* Produce a default config for the given target type. */
OcTurboQuantConfig oc_turboquant_config_default(OcGgufQuantizationType target_type);

/* Allocate a new quantizer. Returns OC_OK on success. The caller owns the
 * struct and must free it with oc_turboquant_free(). */
OcError oc_turboquant_init(OcTurboQuant *tq, OcTurboQuantConfig config);

/* Feed calibration data. Accumulates values until calibration_samples is
 * reached; further calls extend the statistics. Returns OC_OK or
 * OC_ERR_INVALID_ARG. */
OcError oc_turboquant_calibrate(OcTurboQuant *tq, const float *data, size_t n);

/* Quantize `input` (length `n` f32 values) into the output buffer. */
OcError oc_turboquant_quantize(OcTurboQuant *tq, const float *input,
                               size_t n, uint8_t *output, size_t *out_size);

/* Get quality statistics. Returns OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_turboquant_stats(const OcTurboQuant *tq,
                            OcTurboQuantStats *out_stats);

/* Free quantizer buffers and zero the struct. Safe on NULL. */
void oc_turboquant_free(OcTurboQuant *tq);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_TURBOQUANT_H */
