#define _POSIX_C_SOURCE 200809L
#include "oxidize/mtp_weights.h"

#include <stdlib.h>
#include <string.h>

void oc_mtp_weights_init(OcMtpWeights *mw)
{
    if (!mw) return;
    memset(mw, 0, sizeof(*mw));
    oc_layer_weights_init(&mw->layer);
    oc_weight_storage_init(&mw->eh_proj);
    oc_weight_storage_init(&mw->embed_tokens);
    oc_weight_storage_init(&mw->shared_head_head);
}

void oc_mtp_weights_free(OcMtpWeights *mw)
{
    if (!mw) return;
    oc_layer_weights_free(&mw->layer);
    oc_weight_storage_free(&mw->eh_proj);
    oc_weight_storage_free(&mw->embed_tokens);
    oc_weight_storage_free(&mw->shared_head_head);
    free(mw->enorm);
    free(mw->hnorm);
    free(mw->shared_head_norm);
    memset(mw, 0, sizeof(*mw));
}

bool oc_mtp_weights_is_usable(const OcMtpWeights *mw,
                               const OcInferenceConfig *cfg)
{
    if (!mw || !cfg) return false;
    size_t h = cfg->hidden_size;
    if (h == 0) return false;

    /* eh_proj must be non-empty and output_dim(2*h) == h. */
    if (oc_weight_storage_is_empty(&mw->eh_proj)) return false;
    if (oc_weight_storage_output_dim(&mw->eh_proj, h * 2) != h) return false;

    /* enorm and hnorm must have h elements. */
    if (!mw->enorm || !mw->hnorm) return false;

    /* Layer must have attention + FFN weights. */
    if (!oc_layer_weights_has_attention(&mw->layer)) return false;
    if (!oc_layer_weights_has_dense_ffn(&mw->layer)) return false;

    return true;
}
