/*
 * tiny_model.h — shared OcInferenceModel fixture for the inf-model test
 * family (test_inf_forward / test_inf_model / test_gen_loop /
 * test_layer_range / test_layer_wise).
 *
 * Geometry: hidden=4, heads=2 (kv=2), intermediate=8, vocab=16, layers=2.
 * Weights are deterministic: token i embeds to [i,i,i,i]; attention and
 * FFN projections are identity-first; norms are ones.
 */
#ifndef OXIDIZE_C_TESTS_TINY_MODEL_H
#define OXIDIZE_C_TESTS_TINY_MODEL_H

#include "oxidize/inf_model.h"
#include "oxidize/inference.h"
#include "oxidize/layer_weights.h"
#include "oxidize/weight_storage.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Build the tiny model. context_size is the only knob the suites differ on
 * (32 in inf_forward / inf_model / gen_loop, 64 in layer_range). */
static void oc_test_setup_tiny_model(OcInferenceModel *m, uint32_t context_size)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.key_value_head_dim = 0;  /* = hidden/heads = 2 */
    cfg.intermediate_size = 8;
    cfg.vocab_size = 16;
    cfg.context_size = context_size;
    cfg.layer_count = 2;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.embedding_scale = 1.0f;
    cfg.gelu_ffn = false;
    cfg.sandwich_norm = false;
    cfg.num_experts = 0;
    cfg.num_experts_per_tok = 0;

    oc_inf_model_init(m, &cfg);

    /* Token embeddings: F32, vocab*hidden = 64 floats; token i -> [i,i,i,i]. */
    float *embed = malloc(64 * sizeof(float));
    for (size_t i = 0; i < 16; i++)
        for (size_t j = 0; j < 4; j++)
            embed[i * 4 + j] = (float)i;
    oc_weight_storage_f32(&m->tok_embeddings, embed, 64);

    /* Final norm: ones. */
    m->norm_weight = malloc(4 * sizeof(float));
    for (size_t i = 0; i < 4; i++) m->norm_weight[i] = 1.0f;

    /* Output head [16,4]: row i has 1.0 at column (i % 4). */
    float *out_w = malloc(64 * sizeof(float));
    memset(out_w, 0, 64 * sizeof(float));
    for (size_t i = 0; i < 16; i++)
        out_w[i * 4 + (i % 4)] = 1.0f;
    oc_weight_storage_f32(&m->output_weight, out_w, 64);

    for (uint32_t l = 0; l < 2; l++) {
        OcLayerWeights layer;
        oc_layer_weights_init(&layer);

        /* Attention / FFN norms: ones. */
        layer.attn_norm = malloc(4 * sizeof(float));
        for (size_t i = 0; i < 4; i++) layer.attn_norm[i] = 1.0f;
        layer.n_attn_norm = 4;
        layer.ffn_norm = malloc(4 * sizeof(float));
        for (size_t i = 0; i < 4; i++) layer.ffn_norm[i] = 1.0f;
        layer.n_ffn_norm = 4;

        /* Q [4,4], K [4,4], V [4,4], O [4,4]: identity-first. */
        float *wq = malloc(16 * sizeof(float));
        float *wk = malloc(16 * sizeof(float));
        float *wv = malloc(16 * sizeof(float));
        float *wo = malloc(16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) {
            wq[i * 4 + i] = 1.0f;
            wk[i * 4 + i] = 1.0f;
            wv[i * 4 + i] = 1.0f;
            wo[i * 4 + i] = 1.0f;
        }
        oc_weight_storage_f32(&layer.attn_q, wq, 16);
        oc_weight_storage_f32(&layer.attn_k, wk, 16);
        oc_weight_storage_f32(&layer.attn_v, wv, 16);
        oc_weight_storage_f32(&layer.attn_output, wo, 16);

        /* FFN gate [8,4] / up [8,4] / down [4,8]: identity-first. */
        float *fg_w = malloc(32 * sizeof(float));
        float *fu_w = malloc(32 * sizeof(float));
        float *fd_w = malloc(32 * sizeof(float));
        memset(fg_w, 0, 32 * sizeof(float));
        memset(fu_w, 0, 32 * sizeof(float));
        memset(fd_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) {
            fg_w[i * 4 + i] = 1.0f;
            fu_w[i * 4 + i] = 1.0f;
            fd_w[i * 8 + i] = 1.0f;
        }
        oc_weight_storage_f32(&layer.ffn_gate, fg_w, 32);
        oc_weight_storage_f32(&layer.ffn_up, fu_w, 32);
        oc_weight_storage_f32(&layer.ffn_down, fd_w, 32);

        oc_inf_model_add_layer(m, &layer);
    }

    m->loaded = true;
}

#endif /* OXIDIZE_C_TESTS_TINY_MODEL_H */
