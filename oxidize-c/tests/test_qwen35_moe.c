#include <criterion/criterion.h>

#include "oxidize/llama.h"
#include "oxidize/matvec.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"

#include <stdlib.h>
#include <string.h>

#define MOE_DIM 256u
#define MOE_EXPERTS 8u

typedef struct {
    OcLlamaModel model;
    OcLlamaLayer layer;
    float *embedding;
    float *zero_attention;
    float *norm;
    float *router;
    uint8_t *gate;
    uint8_t *up;
    uint8_t *down;
    float *shared_gate;
    float *shared_up;
    float *shared_down;
} Qwen35MoeFixture;

static OcWeightView moe_f32_view(float *data, size_t rows, size_t cols)
{
    return (OcWeightView){
        .data = (const uint8_t *)data,
        .qtype = OC_QUANT_F32,
        .rows = rows,
        .cols = cols,
        .row_bytes = cols * sizeof(float),
    };
}

static uint8_t *moe_quant_matrix(OcGgufQuantizationType qtype, size_t rows,
                                 size_t cols, uint32_t seed)
{
    const size_t row_bytes = oc_quantized_size(qtype, cols);
    uint8_t *data = malloc(rows * row_bytes);
    float *row = malloc(cols * sizeof(*row));
    cr_assert_not_null(data);
    cr_assert_not_null(row);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            seed = seed * 1664525u + 1013904223u;
            row[c] = (float)((int32_t)(seed >> 24) - 128) / 256.0f;
        }
        cr_assert_eq(oc_quant_pack_row(qtype, row, cols,
                     data + r * row_bytes, row_bytes), OC_OK);
    }
    free(row);
    return data;
}

static OcWeightView moe_quant_view(uint8_t *data,
                                   OcGgufQuantizationType qtype,
                                   size_t rows, size_t cols)
{
    return (OcWeightView){
        .data = data,
        .qtype = qtype,
        .rows = rows,
        .cols = cols,
        .row_bytes = oc_quantized_size(qtype, cols),
    };
}

static void moe_fixture_init(Qwen35MoeFixture *f, uint32_t top_k,
                             uint32_t zero_experts, bool shared,
                             OcGgufQuantizationType gate_qtype,
                             OcGgufQuantizationType up_qtype)
{
    memset(f, 0, sizeof(*f));
    OcLlamaConfig *c = &f->model.cfg;
    c->vocab_size = 1;
    c->n_embd = MOE_DIM;
    c->n_layer = 1;
    c->n_head = 1;
    c->n_head_kv = 1;
    c->n_ff = MOE_DIM;
    c->n_ctx = 1;
    c->head_dim = MOE_DIM;
    c->kv_head_dim = MOE_DIM;
    c->rope_dim = MOE_DIM;
    c->rope_theta = 10000.0f;
    c->rms_norm_eps = 1e-6f;
    c->num_experts = MOE_EXPERTS;
    c->num_experts_per_tok = top_k;
    c->expert_intermediate_size = MOE_DIM;
    c->expert_weights_scale = 1.0f;
    c->zero_expert_count = zero_experts;
    c->norm_scale = 1.0f;
    c->is_qwen35 = true;
    c->n_full_attention_layers = 1;
    c->n_recurrent_layers = 1;
    c->ssm_conv_kernel = 2;
    c->ssm_state_size = 1;
    c->ssm_group_count = 1;
    c->ssm_value_heads = 1;
    c->ssm_inner_size = MOE_DIM;
    f->model.layers = &f->layer;

    f->embedding = malloc(MOE_DIM * sizeof(float));
    f->zero_attention = calloc(2u * MOE_DIM * MOE_DIM, sizeof(float));
    f->norm = malloc(MOE_DIM * sizeof(float));
    f->router = calloc((MOE_EXPERTS + zero_experts) * MOE_DIM, sizeof(float));
    cr_assert(f->embedding && f->zero_attention && f->norm && f->router);
    for (size_t i = 0; i < MOE_DIM; i++) {
        f->embedding[i] = 0.25f + (float)(i % 13u) / 64.0f;
        f->norm[i] = 1.0f;
    }
    for (uint32_t expert = 0; expert < MOE_EXPERTS + zero_experts; expert++) {
        const float rank = (float)(expert + 1u) / 2048.0f;
        for (size_t i = 0; i < MOE_DIM; i++)
            f->router[(size_t)expert * MOE_DIM + i] = rank;
    }

    f->gate = moe_quant_matrix(gate_qtype,
        MOE_EXPERTS * MOE_DIM, MOE_DIM, 0x12345678u);
    f->up = moe_quant_matrix(up_qtype,
        MOE_EXPERTS * MOE_DIM, MOE_DIM, 0x87654321u);
    f->down = moe_quant_matrix(OC_QUANT_Q4_0,
        MOE_EXPERTS * MOE_DIM, MOE_DIM, 0xA5A5A5A5u);

    f->model.tok_embeddings = moe_f32_view(f->embedding, 1, MOE_DIM);
    f->model.final_norm = f->norm;
    f->layer.kind = OC_LLAMA_LAYER_FULL_ATTENTION;
    f->layer.head_dim = MOE_DIM;
    f->layer.n_head_kv = 1;
    f->layer.rope_dim = MOE_DIM;
    f->layer.rope_theta = 10000.0f;
    f->layer.attn_norm = f->norm;
    f->layer.post_attention_norm = f->norm;
    f->layer.attn_q = moe_f32_view(f->zero_attention, 2u * MOE_DIM, MOE_DIM);
    f->layer.attn_k = moe_f32_view(f->zero_attention, MOE_DIM, MOE_DIM);
    f->layer.attn_v = moe_f32_view(f->zero_attention, MOE_DIM, MOE_DIM);
    f->layer.attn_output = moe_f32_view(f->zero_attention, MOE_DIM, MOE_DIM);
    f->layer.attn_q_norm = f->norm;
    f->layer.attn_k_norm = f->norm;
    f->layer.ffn_gate_inp = moe_f32_view(
        f->router, MOE_EXPERTS + zero_experts, MOE_DIM);
    f->layer.ffn_gate_exps = moe_quant_view(
        f->gate, gate_qtype, MOE_DIM, MOE_DIM);
    f->layer.ffn_up_exps = moe_quant_view(
        f->up, up_qtype, MOE_DIM, MOE_DIM);
    f->layer.ffn_down_exps = moe_quant_view(
        f->down, OC_QUANT_Q4_0, MOE_DIM, MOE_DIM);

    if (shared) {
        const size_t matrix_floats = MOE_DIM * MOE_DIM;
        f->shared_gate = calloc(matrix_floats, sizeof(float));
        f->shared_up = calloc(matrix_floats, sizeof(float));
        f->shared_down = calloc(matrix_floats, sizeof(float));
        cr_assert(f->shared_gate && f->shared_up && f->shared_down);
        for (size_t i = 0; i < MOE_DIM; i++) {
            f->shared_gate[i * MOE_DIM + i] = 0.25f;
            f->shared_up[i * MOE_DIM + i] = 0.5f;
            f->shared_down[i * MOE_DIM + i] = 0.75f;
        }
        f->layer.ffn_gate_shexp = moe_f32_view(
            f->shared_gate, MOE_DIM, MOE_DIM);
        f->layer.ffn_up_shexp = moe_f32_view(
            f->shared_up, MOE_DIM, MOE_DIM);
        f->layer.ffn_down_shexp = moe_f32_view(
            f->shared_down, MOE_DIM, MOE_DIM);
    }
}

static void moe_fixture_free(Qwen35MoeFixture *f)
{
    free(f->embedding);
    free(f->zero_attention);
    free(f->norm);
    free(f->router);
    free(f->gate);
    free(f->up);
    free(f->down);
    free(f->shared_gate);
    free(f->shared_up);
    free(f->shared_down);
}

static void run_moe_case(uint32_t top_k, uint32_t zero_experts, bool shared,
                         OcGgufQuantizationType up_qtype, size_t threads,
                         size_t expected_dispatches,
                         size_t expected_fallbacks, float *output)
{
    Qwen35MoeFixture fixture;
    moe_fixture_init(&fixture, top_k, zero_experts, shared,
                     OC_QUANT_Q4_0, up_qtype);
    OcLlamaSession grouped;
    OcLlamaSession serial;
    cr_assert_eq(oc_llama_session_init(&fixture.model, &grouped), OC_OK);
    cr_assert_eq(oc_llama_session_init(&fixture.model, &serial), OC_OK);

    float *serial_gate_all = serial.expert_gate_all;
    float *serial_up_all = serial.expert_up_all;
    float *serial_down_all = serial.expert_down_all;
    serial.expert_gate_all = NULL;
    serial.expert_up_all = NULL;
    serial.expert_down_all = NULL;

    cr_assert_eq(oc_parallel_set_threads(threads), OC_OK);
    cr_assert_eq(oc_llama_forward(&serial, 0, NULL), OC_OK);
    oc_matvec_fused_test_reset();
    cr_assert_eq(oc_llama_forward(&grouped, 0, NULL), OC_OK);
    const OcMatvecFusedTestStats stats = oc_matvec_fused_test_stats();
    cr_assert_eq(stats.parallel_dispatches, expected_dispatches);
    cr_assert_eq(stats.fallback_calls, expected_fallbacks);
    cr_assert_arr_eq(grouped.x, serial.x, MOE_DIM * sizeof(float));
    if (output) memcpy(output, grouped.x, MOE_DIM * sizeof(float));
    cr_assert_eq(oc_parallel_set_threads(1), OC_OK);

    serial.expert_gate_all = serial_gate_all;
    serial.expert_up_all = serial_up_all;
    serial.expert_down_all = serial_down_all;
    oc_llama_session_free(&grouped);
    oc_llama_session_free(&serial);
    moe_fixture_free(&fixture);
}

Test(qwen35_moe, homogeneous_top_k_one_and_eight_use_two_dispatches)
{
    float one_thread[MOE_DIM];
    float sixteen_threads[MOE_DIM];
    run_moe_case(1, 0, false, OC_QUANT_Q4_0, 1, 2, 0, NULL);
    run_moe_case(8, 0, false, OC_QUANT_Q4_0, 1, 2, 0, one_thread);
    run_moe_case(8, 0, false, OC_QUANT_Q4_0, 16, 2, 0,
                 sixteen_threads);
    cr_assert_arr_eq(one_thread, sixteen_threads, sizeof(one_thread));
}

Test(qwen35_moe, shared_and_zero_experts_preserve_selected_order)
{
    run_moe_case(8, 1, true, OC_QUANT_Q4_0, 1, 2, 0, NULL);
    run_moe_case(8, 1, true, OC_QUANT_Q4_0, 16, 2, 0, NULL);
}

Test(qwen35_moe, mixed_qtypes_use_exact_fallback)
{
    run_moe_case(8, 0, false, OC_QUANT_Q4_1, 1, 17, 16, NULL);
    run_moe_case(8, 0, false, OC_QUANT_Q4_1, 16, 17, 16, NULL);
}
