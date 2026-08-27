#define _POSIX_C_SOURCE 200809L
#include "oxidize/layer_weights.h"

#include <stdlib.h>
#include <string.h>

void oc_layer_weights_init(OcLayerWeights *lw)
{
    if (!lw) return;
    memset(lw, 0, sizeof(*lw));
    /* Initialize WeightStorage fields. */
    oc_weight_storage_init(&lw->attn_q);
    oc_weight_storage_init(&lw->attn_k);
    oc_weight_storage_init(&lw->attn_v);
    oc_weight_storage_init(&lw->attn_output);
    oc_weight_storage_init(&lw->ffn_gate);
    oc_weight_storage_init(&lw->ffn_up);
    oc_weight_storage_init(&lw->ffn_down);
    oc_weight_storage_init(&lw->ffn_gate_exps);
    oc_weight_storage_init(&lw->ffn_up_exps);
    oc_weight_storage_init(&lw->ffn_down_exps);
    oc_weight_storage_init(&lw->ffn_gate_inp);
    oc_weight_storage_init(&lw->attn_qkv);
    oc_weight_storage_init(&lw->attn_gate);
    oc_weight_storage_init(&lw->ssm_out);
    oc_weight_storage_init(&lw->shortconv_in_proj);
    oc_weight_storage_init(&lw->shortconv_out_proj);
    oc_weight_storage_init(&lw->mla_q_a);
    oc_weight_storage_init(&lw->mla_q_b);
    oc_weight_storage_init(&lw->mla_kv_a_mqa);
    oc_weight_storage_init(&lw->mla_k_b);
    oc_weight_storage_init(&lw->mla_v_b);
}

void oc_layer_weights_free(OcLayerWeights *lw)
{
    if (!lw) return;
    /* Free owned float arrays. */
    free(lw->attn_norm);
    free(lw->ffn_norm);
    free(lw->post_attention_norm);
    free(lw->post_ffn_norm);
    free(lw->attn_q_bias);
    free(lw->attn_k_bias);
    free(lw->attn_v_bias);
    free(lw->attn_output_bias);
    free(lw->ffn_down_bias);
    free(lw->ssm_a);
    free(lw->ssm_alpha);
    free(lw->ssm_beta);
    free(lw->ssm_conv1d);
    free(lw->ssm_dt_bias);
    free(lw->ssm_norm);
    free(lw->attn_q_norm);
    free(lw->attn_k_norm);
    free(lw->shortconv_conv);
    free(lw->ffn_exp_probs_b);
    free(lw->mla_q_a_norm);
    free(lw->mla_kv_a_norm);

    /* Free WeightStorage (frees owned data). */
    oc_weight_storage_free(&lw->attn_q);
    oc_weight_storage_free(&lw->attn_k);
    oc_weight_storage_free(&lw->attn_v);
    oc_weight_storage_free(&lw->attn_output);
    oc_weight_storage_free(&lw->ffn_gate);
    oc_weight_storage_free(&lw->ffn_up);
    oc_weight_storage_free(&lw->ffn_down);
    oc_weight_storage_free(&lw->ffn_gate_exps);
    oc_weight_storage_free(&lw->ffn_up_exps);
    oc_weight_storage_free(&lw->ffn_down_exps);
    oc_weight_storage_free(&lw->ffn_gate_inp);
    oc_weight_storage_free(&lw->attn_qkv);
    oc_weight_storage_free(&lw->attn_gate);
    oc_weight_storage_free(&lw->ssm_out);
    oc_weight_storage_free(&lw->shortconv_in_proj);
    oc_weight_storage_free(&lw->shortconv_out_proj);
    oc_weight_storage_free(&lw->mla_q_a);
    oc_weight_storage_free(&lw->mla_q_b);
    oc_weight_storage_free(&lw->mla_kv_a_mqa);
    oc_weight_storage_free(&lw->mla_k_b);
    oc_weight_storage_free(&lw->mla_v_b);

    memset(lw, 0, sizeof(*lw));
}

bool oc_layer_weights_has_attention(const OcLayerWeights *lw)
{
    if (!lw) return false;
    return !oc_weight_storage_is_empty(&lw->attn_q) ||
           !oc_weight_storage_is_empty(&lw->attn_qkv);
}

bool oc_layer_weights_has_dense_ffn(const OcLayerWeights *lw)
{
    if (!lw) return false;
    return !oc_weight_storage_is_empty(&lw->ffn_gate);
}

bool oc_layer_weights_has_moe(const OcLayerWeights *lw)
{
    if (!lw) return false;
    return !oc_weight_storage_is_empty(&lw->ffn_gate_exps);
}

bool oc_layer_weights_has_ssm(const OcLayerWeights *lw)
{
    if (!lw) return false;
    return !oc_weight_storage_is_empty(&lw->ssm_out) ||
           (lw->ssm_a != NULL);
}

bool oc_layer_weights_has_mla(const OcLayerWeights *lw)
{
    if (!lw) return false;
    return !oc_weight_storage_is_empty(&lw->mla_q_a);
}
