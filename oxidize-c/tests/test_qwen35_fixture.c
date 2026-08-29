#include "framework.h"

#include "oxidize/gguf.h"
#include "oxidize/gguf_writer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q35_BLOCKS 4u
#define Q35_EMBD 8u
#define Q35_HEADS 2u
#define Q35_KV_HEADS 1u
#define Q35_HEAD_DIM 4u
#define Q35_ROPE_DIM 2u
#define Q35_EXPERTS 3u
#define Q35_TOP_K 2u
#define Q35_EXPERT_FF 4u
#define Q35_SHARED_FF 4u
#define Q35_STATE 2u
#define Q35_GROUPS 1u
#define Q35_V_HEADS 2u
#define Q35_INNER 4u
#define Q35_CONV 2u
#define Q35_VOCAB 11u

#define FIXTURE(name) "/tmp/oxidize-c-qwen35-" name ".gguf"

static void assert_close(float actual, float expected)
{
    float scale = fmaxf(1.0f, fabsf(expected));
    cr_assert_leq(fabsf(actual - expected) / scale, 1e-4f,
                  "actual %.9g expected %.9g", actual, expected);
}

static void add_f32_tensor(OcGgufWriter *writer, const char *name,
                           uint32_t n_dims, const uint64_t *dims)
{
    size_t count = 1;
    for (uint32_t i = 0; i < n_dims; i++) count *= (size_t)dims[i];
    float *data = malloc(count * sizeof(*data));
    cr_assert_not_null(data, "allocate %s", name);
    for (size_t i = 0; i < count; i++)
        data[i] = (float)((i % 13u) + 1u) / 64.0f;
    cr_assert_eq(oc_gguf_writer_add_tensor(writer, name, n_dims, dims, 0,
                                           data, count * sizeof(*data)),
                 OC_OK, "write %s", name);
    free(data);
}

static void add_1d(OcGgufWriter *writer, const char *name, uint64_t d0)
{
    uint64_t dims[] = {d0};
    add_f32_tensor(writer, name, 1, dims);
}

static void add_2d(OcGgufWriter *writer, const char *name,
                   uint64_t d0, uint64_t d1)
{
    uint64_t dims[] = {d0, d1};
    add_f32_tensor(writer, name, 2, dims);
}

static void add_3d(OcGgufWriter *writer, const char *name,
                   uint64_t d0, uint64_t d1, uint64_t d2)
{
    uint64_t dims[] = {d0, d1, d2};
    add_f32_tensor(writer, name, 3, dims);
}

static void add_layer_tensor_1d(OcGgufWriter *writer, unsigned layer,
                                const char *suffix, uint64_t d0)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    add_1d(writer, name, d0);
}

static void add_layer_tensor_2d(OcGgufWriter *writer, unsigned layer,
                                const char *suffix, uint64_t d0, uint64_t d1)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    add_2d(writer, name, d0, d1);
}

static void add_layer_tensor_3d(OcGgufWriter *writer, unsigned layer,
                                const char *suffix, uint64_t d0, uint64_t d1,
                                uint64_t d2)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    add_3d(writer, name, d0, d1, d2);
}

static void add_moe_tensors(OcGgufWriter *writer, unsigned layer)
{
    add_layer_tensor_3d(writer, layer, "ffn_down_exps.weight", Q35_EXPERT_FF,
                        Q35_EMBD, Q35_EXPERTS);
    add_layer_tensor_2d(writer, layer, "ffn_down_shexp.weight", Q35_SHARED_FF,
                        Q35_EMBD);
    add_layer_tensor_3d(writer, layer, "ffn_gate_exps.weight", Q35_EMBD,
                        Q35_EXPERT_FF, Q35_EXPERTS);
    add_layer_tensor_2d(writer, layer, "ffn_gate_inp.weight", Q35_EMBD,
                        Q35_EXPERTS);
    add_layer_tensor_1d(writer, layer, "ffn_gate_inp_shexp.weight", Q35_EMBD);
    add_layer_tensor_2d(writer, layer, "ffn_gate_shexp.weight", Q35_EMBD,
                        Q35_SHARED_FF);
    add_layer_tensor_3d(writer, layer, "ffn_up_exps.weight", Q35_EMBD,
                        Q35_EXPERT_FF, Q35_EXPERTS);
    add_layer_tensor_2d(writer, layer, "ffn_up_shexp.weight", Q35_EMBD,
                        Q35_SHARED_FF);
}

static void add_linear_attention(OcGgufWriter *writer, unsigned layer,
                                 bool include_ssm_alpha)
{
    uint64_t key_dim = Q35_GROUPS * Q35_STATE;
    uint64_t conv_dim = 2u * key_dim + Q35_INNER;
    add_layer_tensor_2d(writer, layer, "attn_gate.weight", Q35_EMBD, Q35_INNER);
    add_layer_tensor_2d(writer, layer, "attn_qkv.weight", Q35_EMBD, conv_dim);
    add_layer_tensor_1d(writer, layer, "ssm_a", Q35_V_HEADS);
    if (include_ssm_alpha)
        add_layer_tensor_2d(writer, layer, "ssm_alpha.weight", Q35_EMBD,
                            Q35_V_HEADS);
    add_layer_tensor_2d(writer, layer, "ssm_beta.weight", Q35_EMBD,
                        Q35_V_HEADS);
    add_layer_tensor_2d(writer, layer, "ssm_conv1d.weight", Q35_CONV, conv_dim);
    add_layer_tensor_1d(writer, layer, "ssm_dt.bias", Q35_V_HEADS);
    add_layer_tensor_1d(writer, layer, "ssm_norm.weight",
                        Q35_INNER / Q35_V_HEADS);
    add_layer_tensor_2d(writer, layer, "ssm_out.weight", Q35_INNER, Q35_EMBD);
}

static void add_full_attention(OcGgufWriter *writer, unsigned layer)
{
    add_layer_tensor_2d(writer, layer, "attn_k.weight", Q35_EMBD,
                        Q35_KV_HEADS * Q35_HEAD_DIM);
    add_layer_tensor_1d(writer, layer, "attn_k_norm.weight", Q35_HEAD_DIM);
    add_layer_tensor_2d(writer, layer, "attn_output.weight",
                        Q35_HEADS * Q35_HEAD_DIM, Q35_EMBD);
    add_layer_tensor_2d(writer, layer, "attn_q.weight", Q35_EMBD,
                        2u * Q35_HEADS * Q35_HEAD_DIM);
    add_layer_tensor_1d(writer, layer, "attn_q_norm.weight", Q35_HEAD_DIM);
    add_layer_tensor_2d(writer, layer, "attn_v.weight", Q35_EMBD,
                        Q35_KV_HEADS * Q35_HEAD_DIM);
}

static void build_qwen35_fixture(const char *path, bool include_ssm_alpha)
{
    OcGgufWriter writer;
    cr_assert_eq(oc_gguf_writer_init(path, "qwen35moe", &writer), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer, "qwen35moe.block_count",
                                           Q35_BLOCKS), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer, "qwen35moe.context_length",
                                           32), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.embedding_length", Q35_EMBD), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.attention.head_count", Q35_HEADS), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.attention.head_count_kv", Q35_KV_HEADS), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_float32(&writer,
                 "qwen35moe.rope.freq_base", 10000000.0f), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_float32(&writer,
                 "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer, "qwen35moe.expert_count",
                                           Q35_EXPERTS), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.expert_used_count", Q35_TOP_K), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.attention.key_length", Q35_HEAD_DIM), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.attention.value_length", Q35_HEAD_DIM), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.expert_feed_forward_length", Q35_EXPERT_FF), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.expert_shared_feed_forward_length", Q35_SHARED_FF),
                 OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.ssm.conv_kernel", Q35_CONV), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.ssm.state_size", Q35_STATE), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.ssm.group_count", Q35_GROUPS), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.ssm.time_step_rank", Q35_V_HEADS), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer, "qwen35moe.ssm.inner_size",
                                           Q35_INNER), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.full_attention_interval", 4), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.rope.dimension_count", Q35_ROPE_DIM), OC_OK);
    cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                 "qwen35moe.nextn_predict_layers", 0), OC_OK);

    add_2d(&writer, "output.weight", Q35_EMBD, Q35_VOCAB);
    add_1d(&writer, "output_norm.weight", Q35_EMBD);
    add_2d(&writer, "token_embd.weight", Q35_EMBD, Q35_VOCAB);
    for (unsigned layer = 0; layer < Q35_BLOCKS; layer++) {
        add_layer_tensor_1d(&writer, layer, "attn_norm.weight", Q35_EMBD);
        add_layer_tensor_1d(&writer, layer, "post_attention_norm.weight",
                            Q35_EMBD);
        add_moe_tensors(&writer, layer);
        if ((layer + 1u) % 4u == 0)
            add_full_attention(&writer, layer);
        else
            add_linear_attention(&writer, layer,
                                 include_ssm_alpha || layer != 0u);
    }
    cr_assert_eq(oc_gguf_writer_finalize(&writer), OC_OK);
    oc_gguf_writer_free(&writer);
}

static OcError validate_hybrid_contract(const OcGgufFile *gguf,
                                        const char **missing)
{
    for (unsigned layer = 0; layer < Q35_BLOCKS; layer++) {
        char name[96];
        const char *required = (layer + 1u) % 4u == 0
                                 ? "attn_q.weight" : "ssm_alpha.weight";
        snprintf(name, sizeof(name), "blk.%u.%s", layer, required);
        if (!oc_gguf_tensor_get(gguf, name)) {
            if (missing) *missing = required;
            return OC_ERR_TENSOR;
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", layer);
        if (!oc_gguf_tensor_get(gguf, name)) {
            if (missing) *missing = "ffn_gate_exps.weight";
            return OC_ERR_TENSOR;
        }
    }
    return OC_OK;
}

Test(qwen35_fixture, writes_hybrid_tensor_contract)
{
    const char *path = FIXTURE("contract");
    build_qwen35_fixture(path, true);
    OcGgufFile gguf;
    cr_assert_eq(oc_gguf_open(path, &gguf), OC_OK);
    const char *arch = NULL;
    size_t arch_len = 0;
    cr_assert(oc_gguf_metadata_get_str(&gguf, "general.architecture", &arch,
                                       &arch_len));
    cr_assert_eq(arch_len, strlen("qwen35moe"));
    cr_assert_arr_eq(arch, "qwen35moe", arch_len);
    cr_assert_eq(gguf.tensor_count, 76u);
    cr_assert_eq(validate_hybrid_contract(&gguf, NULL), OC_OK);
    cr_assert_not_null(oc_gguf_tensor_get(&gguf, "blk.0.ssm_alpha.weight"));
    cr_assert_null(oc_gguf_tensor_get(&gguf, "blk.0.attn_q.weight"));
    cr_assert_not_null(oc_gguf_tensor_get(&gguf, "blk.3.attn_q.weight"));
    cr_assert_null(oc_gguf_tensor_get(&gguf, "blk.3.ssm_alpha.weight"));
    cr_assert_not_null(gguf.backing_buf);
    cr_assert_lt((uint64_t)gguf.backing_len, 2u * 1024u * 1024u);
    oc_gguf_free(&gguf);
    remove(path);
}

Test(qwen35_fixture, rejects_missing_ssm_alpha)
{
    const char *path = FIXTURE("missing-alpha");
    build_qwen35_fixture(path, false);
    OcGgufFile gguf;
    cr_assert_eq(oc_gguf_open(path, &gguf), OC_OK);
    const char *missing = NULL;
    cr_assert_eq(validate_hybrid_contract(&gguf, &missing), OC_ERR_TENSOR);
    cr_assert_str_eq(missing, "ssm_alpha.weight");
    oc_gguf_free(&gguf);
    remove(path);
}

Test(qwen35_fixture, gated_delta_scalar_golden)
{
    float state[] = {0.1f, -0.2f, 0.3f, 0.4f};
    float q[] = {0.3f, -0.4f};
    float k[] = {0.6f, 0.8f};
    float value[] = {0.5f, -0.25f};
    float q_norm = sqrtf(q[0] * q[0] + q[1] * q[1] + 1e-6f);
    float k_norm = sqrtf(k[0] * k[0] + k[1] * k[1] + 1e-6f);
    for (size_t i = 0; i < 2; i++) {
        q[i] /= q_norm * sqrtf(2.0f);
        k[i] /= k_norm;
    }
    float decay = expf(-0.7f * log1pf(expf(0.3f - 0.1f)));
    float beta = 1.0f / (1.0f + expf(-0.4f));
    float output[2];
    for (size_t row = 0; row < 2; row++) {
        float sk = 0.0f;
        for (size_t col = 0; col < 2; col++) {
            state[row * 2 + col] *= decay;
            sk += state[row * 2 + col] * k[col];
        }
        float delta = (value[row] - sk) * beta;
        output[row] = 0.0f;
        for (size_t col = 0; col < 2; col++) {
            state[row * 2 + col] += delta * k[col];
            output[row] += state[row * 2 + col] * q[col];
        }
    }
    const float state_golden[] = {0.25734687f, 0.15247790f,
                                  -0.02094338f, -0.02792450f};
    const float output_golden[] = {0.02292845f, 0.00691095f};
    for (size_t i = 0; i < 4; i++) assert_close(state[i], state_golden[i]);
    for (size_t i = 0; i < 2; i++) assert_close(output[i], output_golden[i]);
}

Test(qwen35_fixture, full_attention_scalar_golden)
{
    const float query[] = {0.2f, -0.3f};
    const float keys[][2] = {{0.1f, 0.4f}, {-0.5f, 0.2f}, {0.3f, -0.1f}};
    const float values[][2] = {{0.7f, -0.2f}, {0.1f, 0.5f}, {-0.4f, 0.8f}};
    float score[3];
    float max_score = -INFINITY;
    for (size_t pos = 0; pos < 3; pos++) {
        score[pos] = (query[0] * keys[pos][0] + query[1] * keys[pos][1]) /
                     sqrtf(2.0f);
        max_score = fmaxf(max_score, score[pos]);
    }
    float denom = 0.0f;
    for (size_t pos = 0; pos < 3; pos++) denom += expf(score[pos] - max_score);
    float output[] = {0.0f, 0.0f};
    for (size_t pos = 0; pos < 3; pos++) {
        float weight = expf(score[pos] - max_score) / denom;
        for (size_t i = 0; i < 2; i++) output[i] += weight * values[pos][i];
    }
    assert_close(output[0], 0.10905899f);
    assert_close(output[1], 0.38496689f);
}

Test(qwen35_fixture, router_top_k_scalar_golden)
{
    const float score[] = {0.2f, -0.4f, 1.1f, 0.7f};
    size_t selected[] = {0, 1};
    for (size_t expert = 0; expert < 4; expert++) {
        if (score[expert] > score[selected[0]]) {
            selected[1] = selected[0];
            selected[0] = expert;
        } else if (score[expert] > score[selected[1]]) {
            selected[1] = expert;
        }
    }
    float denom = expf(score[selected[0]]) + expf(score[selected[1]]);
    cr_assert_eq(selected[0], 2u);
    cr_assert_eq(selected[1], 3u);
    assert_close(expf(score[selected[0]]) / denom, 0.59868766f);
    assert_close(expf(score[selected[1]]) / denom, 0.40131234f);
}

Test(qwen35_fixture, one_token_logits_scalar_golden)
{
    const float embedding[] = {0.25f, -0.5f, 0.75f, -0.125f};
    const float norm_weight[] = {1.1f, 0.9f, 1.2f, 0.8f};
    const float output_weight[][4] = {
        {0.1f, -0.2f, 0.3f, 0.4f},
        {-0.5f, 0.6f, -0.7f, 0.8f},
        {0.9f, 0.2f, -0.1f, -0.3f},
    };
    float mean_square = 0.0f;
    for (size_t i = 0; i < 4; i++) mean_square += embedding[i] * embedding[i];
    float inverse_rms = 1.0f / sqrtf(mean_square / 4.0f + 1e-6f);
    float logits[3] = {0};
    for (size_t vocab = 0; vocab < 3; vocab++)
        for (size_t i = 0; i < 4; i++)
            logits[vocab] += output_weight[vocab][i] * embedding[i] *
                             inverse_rms * norm_weight[i];
    const float golden[] = {0.73643834f, -2.36825848f, 0.20662658f};
    for (size_t i = 0; i < 3; i++) assert_close(logits[i], golden[i]);
}
