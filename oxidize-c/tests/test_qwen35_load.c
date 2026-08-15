#include <criterion/criterion.h>

#include "oxidize/gguf_writer.h"
#include "oxidize/llama.h"

#include <stdio.h>
#include <stdlib.h>

#define FIXTURE(name) "/tmp/oxidize-c-qwen35-" name ".gguf"

enum {
    N_EMBD = 8,
    N_HEAD = 2,
    N_HEAD_KV = 1,
    HEAD_DIM = 4,
    N_EXPERT = 3,
    EXPERT_FF = 4,
    N_VALUE_HEAD = 2,
    N_GROUP = 1,
    STATE_DIM = 2,
    INNER_DIM = 4,
    CONV_KERNEL = 2,
    N_VOCAB = 11,
    N_FF = 16,
};

static void add_tensor(OcGgufWriter *w, const char *name, uint32_t n_dims,
                       const uint64_t *dims)
{
    size_t count = 1;
    for (uint32_t i = 0; i < n_dims; i++) count *= (size_t)dims[i];
    float *data = calloc(count, sizeof(*data));
    cr_assert_not_null(data);
    cr_assert_eq(oc_gguf_writer_add_tensor(w, name, n_dims, dims, 0, data,
                                           count * sizeof(*data)), OC_OK);
    free(data);
}

static void add_1d(OcGgufWriter *w, const char *name, uint64_t d0)
{
    uint64_t dims[] = {d0};
    add_tensor(w, name, 1, dims);
}

static void add_2d(OcGgufWriter *w, const char *name, uint64_t d0, uint64_t d1)
{
    uint64_t dims[] = {d0, d1};
    add_tensor(w, name, 2, dims);
}

static void add_3d(OcGgufWriter *w, const char *name, uint64_t d0, uint64_t d1,
                   uint64_t d2)
{
    uint64_t dims[] = {d0, d1, d2};
    add_tensor(w, name, 3, dims);
}

static void layer_name(char *out, size_t cap, unsigned layer,
                       const char *suffix)
{
    snprintf(out, cap, "blk.%u.%s", layer, suffix);
}

static void add_layer_1d(OcGgufWriter *w, unsigned layer, const char *suffix,
                         uint64_t d0)
{
    char name[96];
    layer_name(name, sizeof(name), layer, suffix);
    add_1d(w, name, d0);
}

static void add_layer_2d(OcGgufWriter *w, unsigned layer, const char *suffix,
                         uint64_t d0, uint64_t d1)
{
    char name[96];
    layer_name(name, sizeof(name), layer, suffix);
    add_2d(w, name, d0, d1);
}

static void add_layer_3d(OcGgufWriter *w, unsigned layer, const char *suffix,
                         uint64_t d0, uint64_t d1, uint64_t d2)
{
    char name[96];
    layer_name(name, sizeof(name), layer, suffix);
    add_3d(w, name, d0, d1, d2);
}

static void add_moe(OcGgufWriter *w, unsigned layer)
{
    add_layer_2d(w, layer, "ffn_gate_inp.weight", N_EMBD, N_EXPERT);
    add_layer_3d(w, layer, "ffn_gate_exps.weight", N_EMBD, EXPERT_FF,
                 N_EXPERT);
    add_layer_3d(w, layer, "ffn_up_exps.weight", N_EMBD, EXPERT_FF,
                 N_EXPERT);
    add_layer_3d(w, layer, "ffn_down_exps.weight", EXPERT_FF, N_EMBD,
                 N_EXPERT);
    add_layer_2d(w, layer, "ffn_gate_shexp.weight", N_EMBD, EXPERT_FF);
    add_layer_2d(w, layer, "ffn_up_shexp.weight", N_EMBD, EXPERT_FF);
    add_layer_2d(w, layer, "ffn_down_shexp.weight", EXPERT_FF, N_EMBD);
    add_layer_1d(w, layer, "ffn_gate_inp_shexp.weight", N_EMBD);
}

static void add_recurrent(OcGgufWriter *w, unsigned layer, bool alpha,
                          bool wrong_shape)
{
    const uint64_t key_dim = N_GROUP * STATE_DIM;
    const uint64_t conv_dim = 2 * key_dim + INNER_DIM;
    add_layer_2d(w, layer, "attn_gate.weight", N_EMBD, INNER_DIM);
    add_layer_2d(w, layer, "attn_qkv.weight", N_EMBD, conv_dim);
    add_layer_1d(w, layer, "ssm_a", N_VALUE_HEAD);
    if (alpha)
        add_layer_2d(w, layer, "ssm_alpha.weight",
                     wrong_shape ? N_EMBD - 1 : N_EMBD, N_VALUE_HEAD);
    add_layer_2d(w, layer, "ssm_beta.weight", N_EMBD, N_VALUE_HEAD);
    add_layer_1d(w, layer, "ssm_conv1d.weight", CONV_KERNEL * conv_dim);
    add_layer_1d(w, layer, "ssm_dt.bias", N_VALUE_HEAD);
    add_layer_1d(w, layer, "ssm_norm.weight", INNER_DIM / N_VALUE_HEAD);
    add_layer_2d(w, layer, "ssm_out.weight", INNER_DIM, N_EMBD);
}

static void add_full(OcGgufWriter *w, unsigned layer)
{
    add_layer_2d(w, layer, "attn_q.weight", N_EMBD,
                 2 * N_HEAD * HEAD_DIM);
    add_layer_2d(w, layer, "attn_k.weight", N_EMBD, N_HEAD_KV * HEAD_DIM);
    add_layer_2d(w, layer, "attn_v.weight", N_EMBD, N_HEAD_KV * HEAD_DIM);
    add_layer_2d(w, layer, "attn_output.weight", N_HEAD * HEAD_DIM, N_EMBD);
    add_layer_1d(w, layer, "attn_q_norm.weight", HEAD_DIM);
    add_layer_1d(w, layer, "attn_k_norm.weight", HEAD_DIM);
}

static void build_fixture(const char *path, bool alpha, bool wrong_shape,
                          uint32_t nextn, bool reverse_kinds)
{
    OcGgufWriter w;
    cr_assert_eq(oc_gguf_writer_init(path, "qwen35moe", &w), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.block_count",
                                           2 + nextn), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.nextn_predict_layers",
                                           nextn), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.context_length", 8),
                 OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.embedding_length",
                                           N_EMBD), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.attention.head_count", N_HEAD), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.attention.head_count_kv", N_HEAD_KV), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.attention.key_length", HEAD_DIM), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.rope.dimension_count", 2), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.expert_count",
                                           N_EXPERT), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.expert_used_count", 2), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.expert_feed_forward_length", EXPERT_FF), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.expert_shared_feed_forward_length", EXPERT_FF),
                 OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.full_attention_interval", 2), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.ssm.conv_kernel",
                                           CONV_KERNEL), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.ssm.state_size",
                                           STATE_DIM), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.ssm.group_count",
                                           N_GROUP), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                 "qwen35moe.ssm.time_step_rank", N_VALUE_HEAD), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35moe.ssm.inner_size",
                                           INNER_DIM), OC_OK);
    add_2d(&w, "token_embd.weight", N_EMBD, N_VOCAB);
    add_2d(&w, "output.weight", N_EMBD, N_VOCAB);
    add_1d(&w, "output_norm.weight", N_EMBD);
    for (unsigned layer = 0; layer < 2; layer++) {
        add_layer_1d(&w, layer, "attn_norm.weight", N_EMBD);
        add_layer_1d(&w, layer, "post_attention_norm.weight", N_EMBD);
        add_moe(&w, layer);
    }
    if (reverse_kinds) {
        add_full(&w, 0);
        add_recurrent(&w, 1, alpha, wrong_shape);
    } else {
        add_recurrent(&w, 0, alpha, wrong_shape);
        add_full(&w, 1);
    }
    cr_assert_eq(oc_gguf_writer_finalize(&w), OC_OK);
    oc_gguf_writer_free(&w);
}

static void build_dense_fixture(const char *path)
{
    OcGgufWriter w;
    cr_assert_eq(oc_gguf_writer_init(path, "qwen35", &w), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35.block_count", 3), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                                           "qwen35.nextn_predict_layers", 1),
                 OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35.context_length", 8),
                 OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35.embedding_length",
                                           N_EMBD), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35.feed_forward_length",
                                           N_FF), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                                           "qwen35.attention.head_count",
                                           N_HEAD), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                                           "qwen35.attention.head_count_kv",
                                           N_HEAD_KV), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w,
                                           "qwen35.attention.key_length",
                                           HEAD_DIM), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&w, "qwen35.rope.dimension_count",
                                           2), OC_OK);
    add_2d(&w, "token_embd.weight", N_EMBD, N_VOCAB);
    add_2d(&w, "output.weight", N_EMBD, N_VOCAB);
    add_1d(&w, "output_norm.weight", N_EMBD);
    for (unsigned layer = 0; layer < 2; layer++) {
        add_layer_1d(&w, layer, "attn_norm.weight", N_EMBD);
        add_layer_1d(&w, layer, "post_attention_norm.weight", N_EMBD);
        add_full(&w, layer);
        add_layer_2d(&w, layer, "ffn_gate.weight", N_EMBD, N_FF);
        add_layer_2d(&w, layer, "ffn_up.weight", N_EMBD, N_FF);
        add_layer_2d(&w, layer, "ffn_down.weight", N_FF, N_EMBD);
    }
    add_layer_1d(&w, 2, "attn_norm.weight", N_EMBD);
    add_layer_1d(&w, 2, "post_attention_norm.weight", N_EMBD);
    add_full(&w, 2);
    add_layer_2d(&w, 2, "ffn_gate.weight", N_EMBD, N_FF);
    add_layer_2d(&w, 2, "ffn_up.weight", N_EMBD, N_FF);
    add_layer_2d(&w, 2, "ffn_down.weight", N_FF, N_EMBD);
    add_layer_2d(&w, 2, "nextn.eh_proj.weight", 2 * N_EMBD, N_EMBD);
    add_layer_1d(&w, 2, "nextn.enorm.weight", N_EMBD);
    add_layer_1d(&w, 2, "nextn.hnorm.weight", N_EMBD);
    add_layer_1d(&w, 2, "nextn.shared_head_norm.weight", N_EMBD);
    cr_assert_eq(oc_gguf_writer_finalize(&w), OC_OK);
    oc_gguf_writer_free(&w);
}

Test(qwen35_load, loads_valid_hybrid_fixture)
{
    const char *path = FIXTURE("valid");
    build_fixture(path, true, false, 0, false);
    OcLlamaModel model;
    cr_assert_eq(oc_llama_load(path, &model), OC_OK,
                 "unsupported architecture 'qwen35moe'");
    cr_assert(model.cfg.is_qwen35);
    cr_assert_eq(model.cfg.n_layer, 2);
    cr_assert_eq(model.cfg.n_recurrent_layers, 1);
    cr_assert_eq(model.cfg.n_full_attention_layers, 1);
    cr_assert_eq(model.cfg.full_attention_interval, 2);
    cr_assert_eq(model.layers[0].kind, OC_LLAMA_LAYER_QWEN35_RECURRENT);
    cr_assert_eq(model.layers[1].kind, OC_LLAMA_LAYER_FULL_ATTENTION);
    cr_assert_not_null(model.layers[0].ssm_alpha.data);
    cr_assert_not_null(model.layers[0].ffn_gate_exps.data);
    cr_assert_not_null(model.layers[0].ffn_gate_shexp.data);
    cr_assert_not_null(model.layers[1].attn_q.data);
    oc_llama_free(&model);
    remove(path);
}

Test(qwen35_load, rejects_missing_ssm_alpha)
{
    const char *path = FIXTURE("missing-alpha");
    build_fixture(path, false, false, 0, false);
    OcLlamaModel model;
    cr_assert_eq(oc_llama_load(path, &model), OC_ERR_TENSOR);
    oc_llama_free(&model);
    remove(path);
}

Test(qwen35_load, rejects_wrong_recurrent_tensor_shape)
{
    const char *path = FIXTURE("wrong-shape");
    build_fixture(path, true, true, 0, false);
    OcLlamaModel model;
    cr_assert_eq(oc_llama_load(path, &model), OC_ERR_TENSOR);
    oc_llama_free(&model);
    remove(path);
}

Test(qwen35_load, skips_trailing_mtp_blocks)
{
    const char *path = FIXTURE("mtp");
    build_fixture(path, true, false, 1, false);
    OcLlamaModel model;
    cr_assert_eq(oc_llama_load(path, &model), OC_OK);
    cr_assert_eq(model.cfg.nextn_predict_layers, 1);
    cr_assert_eq(model.cfg.n_layer, 2);
    oc_llama_free(&model);
    remove(path);
}

Test(qwen35_load, loads_dense_attention_model_with_trailing_mtp)
{
    const char *path = FIXTURE("dense-mtp");
    build_dense_fixture(path);
    OcLlamaModel model;
    cr_assert_eq(oc_llama_load(path, &model), OC_OK);
    cr_assert(model.cfg.is_qwen35);
    cr_assert_eq(model.cfg.nextn_predict_layers, 1);
    cr_assert_eq(model.cfg.n_layer, 2);
    cr_assert_eq(model.cfg.n_recurrent_layers, 0);
    cr_assert_eq(model.cfg.n_full_attention_layers, 2);
    cr_assert(model.mtp.present);
    cr_assert_not_null(model.mtp.eh_proj.data);
    cr_assert_not_null(model.mtp.enorm);
    cr_assert_not_null(model.mtp.layer.attn_q.data);
    oc_llama_free(&model);
    remove(path);
}

Test(qwen35_load, tensor_presence_overrides_interval_schedule)
{
    const char *path = FIXTURE("presence");
    build_fixture(path, true, false, 0, true);
    OcLlamaModel model;
    cr_assert_eq(oc_llama_load(path, &model), OC_OK);
    cr_assert_eq(model.layers[0].kind, OC_LLAMA_LAYER_FULL_ATTENTION);
    cr_assert_eq(model.layers[1].kind, OC_LLAMA_LAYER_QWEN35_RECURRENT);
    oc_llama_free(&model);
    remove(path);
}

Test(qwen35_load, session_owns_and_resets_hybrid_state)
{
    const char *path = FIXTURE("session");
    build_fixture(path, true, false, 0, false);
    OcLlamaModel model;
    cr_assert_eq(oc_llama_load(path, &model), OC_OK);
    cr_assert_eq(oc_llama_kv_cache_bytes(&model, OC_KV_F32),
                 2u * 1u * 8u * (N_HEAD_KV * HEAD_DIM) * sizeof(float));
    OcLlamaSession session;
    cr_assert_eq(oc_llama_session_init(&model, &session), OC_OK);
    cr_assert_not_null(session.qwen35_delta);
    cr_assert_not_null(session.qwen35_delta[0].conv_state);
    cr_assert_null(session.qwen35_delta[1].conv_state);
    session.pos = 3;
    session.qwen35_delta[0].conv_state[0] = 4.0f;
    session.qwen35_delta[0].recurrent_state[0] = 5.0f;
    oc_llama_session_reset(&session);
    cr_assert_eq(session.pos, 0);
    cr_assert_float_eq(session.qwen35_delta[0].conv_state[0], 0.0f, 0.0f);
    cr_assert_float_eq(session.qwen35_delta[0].recurrent_state[0], 0.0f, 0.0f);
    oc_llama_session_free(&session);
    cr_assert_null(session.model);
    cr_assert_null(session.qwen35_delta);
    oc_llama_free(&model);
    remove(path);
}

Test(qwen35_load, frees_partial_session_initialization)
{
    OcLlamaSession session = {0};
    session.kv_k = calloc(4, sizeof(*session.kv_k));
    session.qwen35_delta = calloc(2, sizeof(*session.qwen35_delta));
    session.qwen35_conv_state = calloc(3, sizeof(*session.qwen35_conv_state));
    cr_assert_not_null(session.kv_k);
    cr_assert_not_null(session.qwen35_delta);
    cr_assert_not_null(session.qwen35_conv_state);
    oc_llama_session_free(&session);
    cr_assert_null(session.kv_k);
    cr_assert_null(session.qwen35_delta);
    cr_assert_null(session.qwen35_conv_state);
}
