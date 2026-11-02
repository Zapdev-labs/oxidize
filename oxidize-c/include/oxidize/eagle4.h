/*
 * eagle4.h — EAGLE-4 speculative drafter.
 *
 * A single sliding-window decoder reads three normalized target hidden
 * states plus a token embedding. An eh projection maps that pair down to
 * the draft width, and a rank-r head picks a few vocabulary candidates
 * before the full LM head scores them. This is an oxidize implementation
 * of the published architecture, not a copy of the eagle-harness sources.
 */
#ifndef OXIDIZE_EAGLE4_H
#define OXIDIZE_EAGLE4_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_EAGLE4_MAX_AUX 8
#define OC_EAGLE4_MAX_DRAFT 16

typedef struct OcEagle4Config {
    size_t hidden_size;
    size_t target_hidden_size;
    size_t n_aux;
    uint32_t extract_layers[OC_EAGLE4_MAX_AUX];
    size_t vocab_size;
    size_t n_heads;
    size_t n_kv_heads;
    size_t head_dim;
    size_t intermediate_size;
    size_t sliding_window; /* 0 = attend the whole cache */
    size_t lm_rank;
    size_t n_candidates;
    float rms_eps;
    float rope_theta;
    int use_rope;
} OcEagle4Config;

typedef struct OcEagle4Weight {
    float *data;
    size_t rows;
    size_t cols;
} OcEagle4Weight;

typedef struct OcEagle4Model {
    OcEagle4Config cfg;
    OcEagle4Weight fc_norm;     /* [target] */
    OcEagle4Weight fc;          /* [hidden, n_aux * target] */
    OcEagle4Weight embed;       /* [vocab, hidden] */
    OcEagle4Weight eh_proj;     /* [hidden, 2 * hidden] */
    OcEagle4Weight pre_norm;    /* [hidden] */
    OcEagle4Weight q_proj;      /* [n_heads * head_dim, hidden] */
    OcEagle4Weight k_proj;
    OcEagle4Weight v_proj;
    OcEagle4Weight o_proj;      /* [hidden, n_heads * head_dim] */
    OcEagle4Weight ffn_norm;    /* [hidden] */
    OcEagle4Weight gate;        /* [intermediate, hidden] */
    OcEagle4Weight up;
    OcEagle4Weight down;        /* [hidden, intermediate] */
    OcEagle4Weight post_norm;   /* [hidden] */
    OcEagle4Weight lm_head;     /* [vocab, hidden] */
    OcEagle4Weight lr_down;     /* [rank, hidden] */
    OcEagle4Weight lr_up;       /* [vocab, rank] */
    float *feedback;            /* [hidden], post-norm state */
    float *keys;                /* [cap, n_kv * head_dim] */
    float *values;
    size_t cache_cap;
    size_t cache_len;
    size_t position;
    int has_feedback;
    int loaded;
} OcEagle4Model;

void oc_eagle4_config_init(OcEagle4Config *cfg);

/* Allocate zeroed weights for `cfg`. */
OcError oc_eagle4_init(OcEagle4Model *m, const OcEagle4Config *cfg);

/* `dir` holds config.json and model.safetensors (F32). */
OcError oc_eagle4_load(OcEagle4Model *m, const char *dir);

void oc_eagle4_reset(OcEagle4Model *m);
void oc_eagle4_free(OcEagle4Model *m);

/* One draft step. `aux` is n_aux * target_hidden, or NULL to continue from
 * the previous post-norm state. `out_token` is the exact-scored candidate. */
OcError oc_eagle4_step(OcEagle4Model *m, uint32_t token, const float *aux,
                       float *out_hidden, uint32_t *out_token, float *out_logit);

OcError oc_eagle4_draft(OcEagle4Model *m, uint32_t token, const float *aux,
                        uint32_t n_draft, uint32_t *out_tokens);

/* Target-side feature dump. Hiddens are the wide residual after each
 * listed layer, stored [token][layer][hidden]. */
typedef struct OcEagleFeatureCapture {
    uint32_t layers[OC_EAGLE4_MAX_AUX];
    uint32_t n_layers;
    uint32_t hidden;
    uint32_t n_tokens;
    uint32_t cap;
    uint32_t *tokens;
    float *hiddens;
} OcEagleFeatureCapture;

OcEagleFeatureCapture *oc_eagle4_capture_new(const uint32_t *layers,
                                            uint32_t n_layers,
                                            uint32_t hidden,
                                            uint32_t cap);
void oc_eagle4_capture_free(OcEagleFeatureCapture *cap);
void oc_eagle4_capture_at(OcEagleFeatureCapture *cap, uint32_t layer,
                          uint32_t index, uint32_t token,
                          const float *hidden, size_t n);
OcError oc_eagle4_capture_concat(const OcEagleFeatureCapture *cap,
                                 uint32_t index, float *dst, size_t dst_n);
OcError oc_eagle4_capture_save(const OcEagleFeatureCapture *cap,
                               const char *path);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_EAGLE4_H */
