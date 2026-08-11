#ifndef OXIDIZE_QWEN35_DELTA_H
#define OXIDIZE_QWEN35_DELTA_H

#include <stddef.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t n_key_heads;
    size_t n_value_heads;
    size_t key_head_dim;
    size_t value_head_dim;
    size_t conv_kernel;
} OcQwen35DeltaGeometry;

typedef struct {
    OcQwen35DeltaGeometry geometry;
    float *conv_state;
    size_t conv_state_len;
    float *recurrent_state;
    size_t recurrent_state_len;
} OcQwen35DeltaState;

typedef struct {
    const float *conv_weight;
    size_t conv_weight_len;
    const float *ssm_a;
    const float *dt_bias;
    const float *norm_weight;
    float norm_eps;
} OcQwen35DeltaParams;

typedef struct {
    const float *qkv;
    size_t qkv_len;
    const float *gate;
    size_t gate_len;
    const float *beta;
    size_t beta_len;
    const float *alpha;
    size_t alpha_len;
} OcQwen35DeltaInput;

/* Bind and clear caller-provided persistent storage. No storage is allocated. */
OcError oc_qwen35_delta_state_init(OcQwen35DeltaState *state,
                                   const OcQwen35DeltaGeometry *geometry,
                                   float *conv_state, size_t conv_state_len,
                                   float *recurrent_state,
                                   size_t recurrent_state_len);

void oc_qwen35_delta_state_reset(OcQwen35DeltaState *state);

/* Unbind storage. The caller remains responsible for releasing it. */
void oc_qwen35_delta_state_free(OcQwen35DeltaState *state);

/* Execute one recurrent token step using caller-provided scratch and output. */
OcError oc_qwen35_delta_step(OcQwen35DeltaState *state,
                             const OcQwen35DeltaParams *params,
                             const OcQwen35DeltaInput *input,
                             float *conv_output, size_t conv_output_len,
                             float *output, size_t output_len);

#ifdef __cplusplus
}
#endif

#endif
