/*
 * eagle3.h — Eagle-3 speculative decoding support.
 *
 * Eagle uses a lightweight draft model to generate candidate tokens
 * for speculative decoding. Port from oxidize-core/src/model/eagle3.rs.
 */
#ifndef OXIDIZE_EAGLE3_H
#define OXIDIZE_EAGLE3_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_EAGLE_MAX_DRAFT 8
#define OC_EAGLE_MAX_LAYERS 4
#define OC_EAGLE_VOCAB_SIZE 128256

typedef struct {
    uint32_t max_draft_tokens;
    uint32_t n_layers;
    uint32_t hidden_dim;
    float acceptance_threshold;
    bool dynamic_draft;
} OcEagleConfig;

typedef struct {
    OcEagleConfig config;
    float *hidden_states;
    uint32_t *draft_tokens;
    float *draft_probs;
    uint32_t n_draft;
    bool initialized;
} OcEagleState;

OcError oc_eagle_config_init(OcEagleConfig *cfg);
OcError oc_eagle_state_init(OcEagleState *state, const OcEagleConfig *cfg);
OcError oc_eagle_generate_draft(OcEagleState *state, const uint32_t *context,
                               size_t n_context, uint32_t max_tokens);
OcError oc_eagle_get_draft_tokens(const OcEagleState *state,
                                  const uint32_t **out_tokens, uint32_t *out_count);
OcError oc_eagle_get_draft_probs(const OcEagleState *state,
                                 const float **out_probs, uint32_t *out_count);
OcError oc_eagle_update_acceptance(OcEagleState *state, uint32_t n_accepted);
uint32_t oc_eagle_n_draft(const OcEagleState *state);
float oc_eagle_acceptance_rate(const OcEagleState *state);
void oc_eagle_state_free(OcEagleState *state);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_EAGLE3_H */
