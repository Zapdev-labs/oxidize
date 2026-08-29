/* mtp_weights.h — Multi-Token Prediction (MTP/nextn) weight bundle. */
#ifndef OXIDIZE_MTP_WEIGHTS_H
#define OXIDIZE_MTP_WEIGHTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/layer_weights.h"
#include "oxidize/inference.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    OcLayerWeights  layer;            /* attention + FFN for the MTP block */
    OcWeightStorage eh_proj;          /* embedding-hidden projection: [2*h, h] */
    float          *enorm;            /* embedding norm [h] */
    float          *hnorm;            /* hidden norm [h] */
    OcWeightStorage embed_tokens;     /* optional separate embedding (may be empty) */
    float          *shared_head_norm; /* optional shared head norm [h] */
    OcWeightStorage shared_head_head; /* optional shared output head */
} OcMtpWeights;

/* Initialize to empty. */
void oc_mtp_weights_init(OcMtpWeights *mw);

/* Free all owned memory. Safe on NULL. */
void oc_mtp_weights_free(OcMtpWeights *mw);

/* Check if the MTP weights are usable for the given config.
 * Mirrors Rust MtpWeights::is_usable(). */
bool oc_mtp_weights_is_usable(const OcMtpWeights *mw,
                               const OcInferenceConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MTP_WEIGHTS_H */
