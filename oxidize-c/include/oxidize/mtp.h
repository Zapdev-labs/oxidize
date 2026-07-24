/*
 * mtp.h — Multi-Token Prediction (MTP/nextn) draft generation.
 *
 * Port of oxidize-core/src/model/inference/mtp.rs.
 * Generates draft tokens using the model's native MTP/nextn block,
 * feeding accepted tokens back for multi-step speculative decoding.
 */
#ifndef OXIDIZE_MTP_H
#define OXIDIZE_MTP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_MTP_MAX_TOKENS   256
#define OC_MTP_MAX_HIDDEN   8192

typedef struct {
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t n_layers;
    uint32_t max_tokens;
    bool     quantspec_draft_kv;
    float    rms_norm_eps;
} OcMtpConfig;

typedef struct {
    OcMtpConfig config;
    uint32_t *draft_tokens;
    float    *draft_logits;
    size_t    n_draft;
    float    *hidden_buf;   /* working hidden state */
    /* Optional weight pointers (NULL = use heuristic logits). */
    float    *eh_proj;      /* [hidden, 2*hidden] embedding-hidden fusion */
    float    *attn_norm;    /* [hidden] */
    float    *ffn_norm;     /* [hidden] */
    float    *lm_head;      /* [vocab, hidden] */
    float    *tok_emb;      /* [vocab, hidden] */
    bool      has_mtp;
} OcMtpEngine;

typedef struct {
    uint32_t token;
    float    logit;
} OcMtpStepResult;

void oc_mtp_config_init(OcMtpConfig *cfg);
OcError oc_mtp_engine_init(OcMtpEngine *engine, const OcMtpConfig *cfg);
void oc_mtp_engine_free(OcMtpEngine *engine);

/* Generate draft tokens from start position.
 * start_token: the committed token at the current position.
 * start_hidden: hidden state vector (hidden_size floats).
 * Returns n_draft draft tokens in engine->draft_tokens. */
OcError oc_mtp_draft(OcMtpEngine *engine,
                     uint32_t start_token,
                     const float *start_hidden,
                     size_t hidden_len,
                     size_t max_tokens,
                     float (*rng)(void));

/* Get the number of draft tokens from the last call. */
size_t oc_mtp_n_draft(const OcMtpEngine *engine);

/* Get a draft token by index. Returns OC_ERR_INVALID_ARG if out of range. */
OcError oc_mtp_get_draft_token(const OcMtpEngine *engine, size_t idx, uint32_t *out_token);

/* Get draft logits by index. Returns OC_ERR_INVALID_ARG if out of range. */
OcError oc_mtp_get_draft_logits(const OcMtpEngine *engine, size_t idx, float **out_logits);

/* Check if the engine has a usable MTP block. */
bool oc_mtp_has_block(const OcMtpEngine *engine);

/* Reset the engine state for a new generation. */
void oc_mtp_reset(OcMtpEngine *engine);

const char *oc_mtp_config_name(const OcMtpConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MTP_H */
