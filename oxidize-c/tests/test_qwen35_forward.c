#include <criterion/criterion.h>

#include "oxidize/llama.h"

#include <math.h>
#include <string.h>

#define EMBED 4u
#define VOCAB 3u
#define CTX 4u
#define HEAD_DIM 2u
#define INNER 2u
#define EXPERTS 2u

typedef struct {
    OcLlamaModel model;
    OcLlamaLayer layers[2];
    float token_embedding[VOCAB * EMBED];
    float output[VOCAB * EMBED];
    float final_norm[EMBED];
    float attn_norm[2][EMBED];
    float post_attn_norm[2][EMBED];
    float qkv[6 * EMBED];
    float recurrent_gate[INNER * EMBED];
    float beta[EMBED];
    float alpha[EMBED];
    float conv[6 * 2];
    float ssm_a[1];
    float dt_bias[1];
    float ssm_norm[HEAD_DIM];
    float ssm_out[EMBED * INNER];
    float full_qgate[2 * HEAD_DIM * EMBED];
    float full_k[HEAD_DIM * EMBED];
    float full_v[HEAD_DIM * EMBED];
    float full_out[EMBED * HEAD_DIM];
    float q_norm[HEAD_DIM];
    float k_norm[HEAD_DIM];
    float router[2][EXPERTS * EMBED];
    float expert_gate[2][EXPERTS * INNER * EMBED];
    float expert_up[2][EXPERTS * INNER * EMBED];
    float expert_down[2][EXPERTS * EMBED * INNER];
    float shared_gate[2][INNER * EMBED];
    float shared_up[2][INNER * EMBED];
    float shared_down[2][EMBED * INNER];
    float shared_scale[2][EMBED];
} TinyQwen35;

static OcWeightView f32_view(float *data, size_t rows, size_t cols)
{
    OcWeightView view = {
        .data = (const uint8_t *)data,
        .qtype = OC_QUANT_F32,
        .rows = rows,
        .cols = cols,
        .row_bytes = cols * sizeof(float),
    };
    return view;
}

static void fill_pattern(float *dst, size_t n, int seed, float scale)
{
    for (size_t i = 0; i < n; i++) {
        int value = (int)((i * 7u + (size_t)seed * 3u) % 11u) - 5;
        dst[i] = (float)value * scale;
    }
}

static void tiny_init(TinyQwen35 *tiny)
{
    memset(tiny, 0, sizeof(*tiny));
    OcLlamaConfig *cfg = &tiny->model.cfg;
    cfg->vocab_size = VOCAB;
    cfg->n_embd = EMBED;
    cfg->n_layer = 2;
    cfg->n_head = 1;
    cfg->n_head_kv = 1;
    cfg->n_ff = INNER;
    cfg->n_ctx = CTX;
    cfg->head_dim = HEAD_DIM;
    cfg->kv_head_dim = HEAD_DIM;
    cfg->rope_dim = 2;
    cfg->rope_theta = 10000.0f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->num_experts = EXPERTS;
    cfg->num_experts_per_tok = 1;
    cfg->expert_intermediate_size = INNER;
    cfg->shared_expert_intermediate_size = INNER;
    cfg->expert_weights_scale = 1.0f;
    cfg->norm_scale = 1.0f;
    cfg->is_qwen35 = true;
    cfg->n_recurrent_layers = 1;
    cfg->n_full_attention_layers = 1;
    cfg->ssm_conv_kernel = 2;
    cfg->ssm_state_size = HEAD_DIM;
    cfg->ssm_group_count = 1;
    cfg->ssm_value_heads = 1;
    cfg->ssm_inner_size = INNER;
    tiny->model.layers = tiny->layers;

    const float embedding[] = {
        0.20f, -0.10f, 0.30f, 0.05f,
        -0.15f, 0.25f, 0.10f, -0.20f,
        0.05f, 0.12f, -0.18f, 0.22f,
    };
    memcpy(tiny->token_embedding, embedding, sizeof(embedding));
    fill_pattern(tiny->output, VOCAB * EMBED, 2, 0.07f);
    for (size_t i = 0; i < EMBED; i++) tiny->final_norm[i] = 1.0f;
    tiny->model.tok_embeddings = f32_view(tiny->token_embedding, VOCAB, EMBED);
    tiny->model.output = f32_view(tiny->output, VOCAB, EMBED);
    tiny->model.final_norm = tiny->final_norm;

    for (size_t layer = 0; layer < 2; layer++) {
        OcLlamaLayer *l = &tiny->layers[layer];
        for (size_t i = 0; i < EMBED; i++) {
            tiny->attn_norm[layer][i] = 1.0f;
            tiny->post_attn_norm[layer][i] = 1.0f;
        }
        l->attn_norm = tiny->attn_norm[layer];
        l->post_attention_norm = tiny->post_attn_norm[layer];
        fill_pattern(tiny->router[layer], EXPERTS * EMBED, 11 + (int)layer,
                     0.08f);
        fill_pattern(tiny->expert_gate[layer], EXPERTS * INNER * EMBED,
                     13 + (int)layer, 0.05f);
        fill_pattern(tiny->expert_up[layer], EXPERTS * INNER * EMBED,
                     17 + (int)layer, 0.06f);
        fill_pattern(tiny->expert_down[layer], EXPERTS * EMBED * INNER,
                     19 + (int)layer, 0.04f);
        fill_pattern(tiny->shared_gate[layer], INNER * EMBED, 23 + (int)layer,
                     0.05f);
        fill_pattern(tiny->shared_up[layer], INNER * EMBED, 29 + (int)layer,
                     0.04f);
        fill_pattern(tiny->shared_down[layer], EMBED * INNER, 31 + (int)layer,
                     0.03f);
        fill_pattern(tiny->shared_scale[layer], EMBED, 37 + (int)layer, 0.07f);
        l->ffn_gate_inp = f32_view(tiny->router[layer], EXPERTS, EMBED);
        l->ffn_gate_exps = f32_view(tiny->expert_gate[layer], INNER, EMBED);
        l->ffn_up_exps = f32_view(tiny->expert_up[layer], INNER, EMBED);
        l->ffn_down_exps = f32_view(tiny->expert_down[layer], EMBED, INNER);
        l->ffn_gate_shexp = f32_view(tiny->shared_gate[layer], INNER, EMBED);
        l->ffn_up_shexp = f32_view(tiny->shared_up[layer], INNER, EMBED);
        l->ffn_down_shexp = f32_view(tiny->shared_down[layer], EMBED, INNER);
        l->ffn_gate_inp_shexp = f32_view(tiny->shared_scale[layer], 1, EMBED);
        l->head_dim = HEAD_DIM;
        l->n_head_kv = 1;
        l->rope_dim = 2;
        l->rope_theta = 10000.0f;
    }

    OcLlamaLayer *recurrent = &tiny->layers[0];
    recurrent->kind = OC_LLAMA_LAYER_QWEN35_RECURRENT;
    recurrent->state_index = 0;
    fill_pattern(tiny->qkv, 6 * EMBED, 41, 0.05f);
    fill_pattern(tiny->recurrent_gate, INNER * EMBED, 43, 0.06f);
    fill_pattern(tiny->beta, EMBED, 47, 0.04f);
    fill_pattern(tiny->alpha, EMBED, 53, 0.03f);
    fill_pattern(tiny->conv, 12, 59, 0.05f);
    tiny->ssm_a[0] = -0.7f;
    tiny->dt_bias[0] = -0.1f;
    tiny->ssm_norm[0] = 1.1f;
    tiny->ssm_norm[1] = 0.9f;
    fill_pattern(tiny->ssm_out, EMBED * INNER, 61, 0.08f);
    recurrent->attn_qkv = f32_view(tiny->qkv, 6, EMBED);
    recurrent->attn_gate = f32_view(tiny->recurrent_gate, INNER, EMBED);
    recurrent->ssm_beta = f32_view(tiny->beta, 1, EMBED);
    recurrent->ssm_alpha = f32_view(tiny->alpha, 1, EMBED);
    recurrent->ssm_conv1d = f32_view(tiny->conv, 6, 2);
    recurrent->ssm_a = f32_view(tiny->ssm_a, 1, 1);
    recurrent->ssm_dt_bias = f32_view(tiny->dt_bias, 1, 1);
    recurrent->ssm_norm = f32_view(tiny->ssm_norm, 1, HEAD_DIM);
    recurrent->ssm_out = f32_view(tiny->ssm_out, EMBED, INNER);

    OcLlamaLayer *full = &tiny->layers[1];
    full->kind = OC_LLAMA_LAYER_FULL_ATTENTION;
    full->kv_cache_index = 0;
    fill_pattern(tiny->full_qgate, 2 * HEAD_DIM * EMBED, 67, 0.06f);
    fill_pattern(tiny->full_k, HEAD_DIM * EMBED, 71, 0.07f);
    fill_pattern(tiny->full_v, HEAD_DIM * EMBED, 73, 0.05f);
    fill_pattern(tiny->full_out, EMBED * HEAD_DIM, 79, 0.08f);
    tiny->q_norm[0] = 1.0f;
    tiny->q_norm[1] = 0.8f;
    tiny->k_norm[0] = 0.9f;
    tiny->k_norm[1] = 1.1f;
    full->attn_q = f32_view(tiny->full_qgate, 2 * HEAD_DIM, EMBED);
    full->attn_k = f32_view(tiny->full_k, HEAD_DIM, EMBED);
    full->attn_v = f32_view(tiny->full_v, HEAD_DIM, EMBED);
    full->attn_output = f32_view(tiny->full_out, EMBED, HEAD_DIM);
    full->attn_q_norm = tiny->q_norm;
    full->attn_k_norm = tiny->k_norm;
}

static void assert_finite(const float *values, size_t n)
{
    for (size_t i = 0; i < n; i++)
        cr_assert(isfinite(values[i]), "non-finite value at %zu", i);
}

static void assert_close(const float *actual, const float *expected, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const float scale = fmaxf(1.0f, fabsf(expected[i]));
        cr_assert_leq(fabsf(actual[i] - expected[i]) / scale, 1e-4f,
                      "value %zu: actual %.9g expected %.9g", i,
                      actual[i], expected[i]);
    }
}

Test(qwen35_forward, token_sequence_matches_golden)
{
    TinyQwen35 tiny;
    tiny_init(&tiny);
    OcLlamaSession session;
    cr_assert_eq(oc_llama_session_init(&tiny.model, &session), OC_OK);
    float logits[VOCAB];
    cr_assert_eq(oc_llama_forward(&session, 0, logits), OC_OK);
    assert_finite(logits, VOCAB);
    const float first_golden[] = {
        0.551780403f, -0.463613361f, 0.738137424f,
    };
    assert_close(logits, first_golden, VOCAB);
    cr_assert_eq(oc_llama_forward(&session, 1, logits), OC_OK);
    const float second_golden[] = {
        -0.478281081f, 0.876530290f, -0.519742250f,
    };
    assert_close(logits, second_golden, VOCAB);
    cr_assert_neq(session.qwen35_delta[0].recurrent_state[0], 0.0f,
                  "recurrent layer was skipped");
    oc_llama_session_free(&session);
}

Test(qwen35_forward, reset_replay_is_exact)
{
    TinyQwen35 tiny;
    tiny_init(&tiny);
    OcLlamaSession session;
    cr_assert_eq(oc_llama_session_init(&tiny.model, &session), OC_OK);
    float first[VOCAB], replay[VOCAB];
    cr_assert_eq(oc_llama_forward(&session, 1, first), OC_OK);
    oc_llama_session_reset(&session);
    cr_assert_eq(oc_llama_forward(&session, 1, replay), OC_OK);
    cr_assert_arr_eq(first, replay, sizeof(first));
    oc_llama_session_free(&session);
}

Test(qwen35_forward, prefill_matches_repeated_tokens)
{
    TinyQwen35 tiny;
    tiny_init(&tiny);
    OcLlamaSession token_session, prefill_session;
    cr_assert_eq(oc_llama_session_init(&tiny.model, &token_session), OC_OK);
    cr_assert_eq(oc_llama_session_init(&tiny.model, &prefill_session), OC_OK);
    const uint32_t tokens[] = {0, 1, 2};
    float token_logits[VOCAB], prefill_logits[VOCAB];
    for (size_t i = 0; i < 3; i++)
        cr_assert_eq(oc_llama_forward(&token_session, tokens[i], token_logits),
                     OC_OK);
    cr_assert_eq(oc_llama_prefill(&prefill_session, tokens, 3, 2,
                                  prefill_logits), OC_OK);
    cr_assert_arr_eq(token_logits, prefill_logits, sizeof(token_logits));
    cr_assert_arr_eq(token_session.qwen35_recurrent_state,
                     prefill_session.qwen35_recurrent_state,
                     INNER * HEAD_DIM * sizeof(float));
    oc_llama_session_free(&token_session);
    oc_llama_session_free(&prefill_session);
}

Test(qwen35_forward, copied_prefix_continues_exactly)
{
    TinyQwen35 tiny;
    tiny_init(&tiny);
    OcLlamaSession source, copy;
    cr_assert_eq(oc_llama_session_init(&tiny.model, &source), OC_OK);
    cr_assert_eq(oc_llama_session_init(&tiny.model, &copy), OC_OK);
    const uint32_t prefix[] = {0, 1};
    cr_assert_eq(oc_llama_prefill(&source, prefix, 2, 2, source.logits), OC_OK);
    cr_assert_eq(oc_llama_session_copy_prefix(&copy, &source), OC_OK);
    float source_logits[VOCAB], copy_logits[VOCAB];
    cr_assert_eq(oc_llama_forward(&source, 2, source_logits), OC_OK);
    cr_assert_eq(oc_llama_forward(&copy, 2, copy_logits), OC_OK);
    cr_assert_arr_eq(source_logits, copy_logits, sizeof(source_logits));
    cr_assert_arr_eq(source.qwen35_conv_state, copy.qwen35_conv_state,
                     source.qwen35_delta[0].conv_state_len * sizeof(float));
    cr_assert_arr_eq(source.qwen35_recurrent_state,
                     copy.qwen35_recurrent_state,
                     source.qwen35_delta[0].recurrent_state_len * sizeof(float));
    cr_assert_eq(source.pos, copy.pos);
    oc_llama_session_free(&source);
    oc_llama_session_free(&copy);
}

Test(qwen35_forward, rejects_position_at_context_limit)
{
    TinyQwen35 tiny;
    tiny_init(&tiny);
    OcLlamaSession session;
    cr_assert_eq(oc_llama_session_init(&tiny.model, &session), OC_OK);
    session.pos = CTX;
    session.qwen35_recurrent_state[0] = 0.25f;
    cr_assert_eq(oc_llama_forward(&session, 0, NULL), OC_ERR_INVALID_ARG);
    cr_assert_float_eq(session.qwen35_recurrent_state[0], 0.25f, 0.0f);
    cr_assert_eq(session.pos, CTX);
    oc_llama_session_free(&session);
}

Test(qwen35_forward, compact_q8_cache_is_finite)
{
    TinyQwen35 tiny;
    tiny_init(&tiny);
    OcLlamaSession session;
    cr_assert_eq(oc_llama_session_init_kv(&tiny.model, &session, OC_KV_Q8),
                 OC_OK);
    float logits[VOCAB];
    cr_assert_eq(oc_llama_forward(&session, 0, logits), OC_OK);
    cr_assert_eq(oc_llama_forward(&session, 1, logits), OC_OK);
    assert_finite(logits, VOCAB);
    cr_assert_not_null(session.kv_k_scale);
    cr_assert(isfinite(session.kv_k_scale[1]));
    oc_llama_session_free(&session);
}
