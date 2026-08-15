#define _POSIX_C_SOURCE 200809L
#include "oxidize/dspark.h"

#include "oxidize/sampling.h"

#include <math.h>
#include <string.h>

void oc_dspark_config_init(OcDsparkConfig *cfg)
{
    if (!cfg) return;
    cfg->block_size = 4;
    cfg->p_min = 0.10f;
}

float oc_dspark_pairwise_conf(const float *logits, size_t vocab)
{
    if (logits == NULL || vocab == 0) return 0.0f;
    float b1 = logits[0], b2 = logits[0];
    int have2 = 0;
    for (size_t i = 1; i < vocab; i++) {
        float v = logits[i];
        if (v > b1) {
            b2 = b1;
            b1 = v;
            have2 = 1;
        } else if (!have2 || v > b2) {
            b2 = v;
            have2 = 1;
        }
    }
    if (!have2) return 1.0f;
    return 1.0f / (1.0f + expf(b2 - b1));
}

OcError oc_dspark_advance(OcLlamaSession *sess, float *logits,
                          const OcDsparkConfig *cfg,
                          uint32_t *out_tokens, size_t max_out, size_t *n_out,
                          OcDsparkStats *stats)
{
    if (sess == NULL || logits == NULL || out_tokens == NULL || n_out == NULL)
        return OC_ERR_INVALID_ARG;
    *n_out = 0;
    if (stats) memset(stats, 0, sizeof(*stats));
    if (max_out == 0) return OC_OK;

    OcDsparkConfig local;
    oc_dspark_config_init(&local);
    if (cfg) local = *cfg;
    if (local.block_size == 0) local.block_size = 4;
    if (local.block_size > OC_DSPARK_MAX_BLOCK)
        local.block_size = OC_DSPARK_MAX_BLOCK;

    OcLlamaModel *m = sess->model;
    if (m == NULL) return OC_ERR_INVALID_ARG;
    const uint32_t vocab = m->cfg.vocab_size;

    if (!oc_llama_mtp_present(m) || max_out == 1) {
        uint32_t t = oc_argmax(logits, vocab);
        out_tokens[(*n_out)++] = t;
        if (stats) stats->emitted = 1;
        return oc_llama_forward(sess, t, logits);
    }

    const uint32_t target0 = oc_argmax(logits, vocab);
    uint32_t drafts[OC_DSPARK_MAX_BLOCK];
    float conf[OC_DSPARK_MAX_BLOCK];
    uint32_t n_draft = 0;
    uint32_t want = local.block_size;
    if (want > max_out) want = (uint32_t)max_out;
    OcError e = oc_llama_mtp_draft_tokens(sess, want, drafts, conf, &n_draft);
    if (e != OC_OK) return e;

    uint32_t keep = n_draft;
    if (local.p_min > 0.0f) {
        keep = 0;
        for (uint32_t i = 0; i < n_draft; i++) {
            if (conf[i] < local.p_min) break;
            keep++;
        }
    }
    if (stats) stats->drafted = keep;

    out_tokens[(*n_out)++] = target0;
    e = oc_llama_forward(sess, target0, logits);
    if (e != OC_OK) return e;
    size_t accepted = (keep > 0 && drafts[0] == target0) ? 1 : 0;
    if (accepted == 1) {
        while (accepted < keep && *n_out < max_out) {
            uint32_t t = oc_argmax(logits, vocab);
            if (t != drafts[accepted]) break;
            out_tokens[(*n_out)++] = t;
            e = oc_llama_forward(sess, t, logits);
            if (e != OC_OK) return e;
            accepted++;
        }
    }
    if (stats) {
        stats->accepted = (uint32_t)accepted;
        stats->emitted = (uint32_t)*n_out;
    }
    return OC_OK;
}
