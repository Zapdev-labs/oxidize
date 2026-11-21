#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include "oxidize/llama.h"
#include "oxidize/gguf_writer.h"
#include "oxidize/activation.h"
#include "oxidize/matvec.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define D 4u
#define Z 2u
#define I 3u
#define E 4u
#define G 3u

Test(latent_moe, canonical_and_legacy_metadata_do_not_enable_router_groups)
{
    for (unsigned variant = 0; variant < 3; variant++) {
        char path[] = "/tmp/oxidize-latent-metadata-XXXXXX";
        const int fd = mkstemp(path);
        cr_assert_geq(fd, 0);
        close(fd);
        OcGgufWriter writer;
        cr_assert_eq(oc_gguf_writer_init(path, "qwen4exp", &writer), OC_OK);
        cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
            "qwen4exp.embedding_length", D), OC_OK);
        cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
            "qwen4exp.expert_count", E), OC_OK);
        if (variant < 2)
            cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                "qwen4exp.expert_latent_length", Z), OC_OK);
        if (variant == 0)
            cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
                "qwen4exp.expert_latent_group_count", G), OC_OK);
        cr_assert_eq(oc_gguf_writer_add_uint32(&writer,
            "qwen4exp.expert_group_count", variant == 1 ? G : 2), OC_OK);
        cr_assert_eq(oc_gguf_writer_finalize(&writer), OC_OK);
        oc_gguf_writer_free(&writer);
        OcLlamaModel model = {0};
        cr_assert_eq(oc_llama_load_flags(path, OC_LLAMA_LOAD_STREAM, &model),
                     OC_ERR_MODEL);
        cr_assert(model.cfg.is_qwen4exp);
        cr_assert_eq(model.cfg.expert_latent_dim, variant < 2 ? Z : 0);
        cr_assert_eq(model.cfg.expert_latent_group_count, variant < 2 ? G : 0);
        cr_assert_eq(unlink(path), 0);
    }
}

typedef struct {
    OcLlamaConfig cfg;
    OcLlamaLayer layer;
    float labels[E];
    float d[G * Z * D];
    float u[G * D * Z];
    float gate[E * I * Z];
    float up[E * I * Z];
    float down[E * Z * I];
} Fixture;

static OcWeightView f32_view(float *p, size_t rows, size_t cols)
{
    return (OcWeightView){ .data = (const uint8_t *)p, .qtype = OC_QUANT_F32,
        .rows = rows, .cols = cols, .row_bytes = cols * sizeof(float) };
}

Test(latent_moe, quantized_gate_up_share_one_dispatch_without_changing_output)
{
    enum { N = 128, ROW_BYTES = N / 32 * 18 };
    float identity[N*N] = {0};
    uint8_t packed[3][N*ROW_BYTES];
    float row[N], input[N], gate[N], up[N], expected[N], actual[N], temp[N];
    for (size_t i = 0; i < N; i++) {
        identity[i*N+i] = 1.0f;
        input[i] = 1.0f + sinf((float)(i+1) * 0.13f);
    }
    for (size_t matrix = 0; matrix < 3; matrix++) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++)
                row[j] = 0.001f * cosf((float)(i*7+j*11+matrix*13));
            cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_0, row, N,
                packed[matrix]+i*ROW_BYTES, ROW_BYTES), OC_OK);
        }
    }
    oc_matvec_set_fused(true);
    oc_matvec_quantized(OC_QUANT_Q4_0, packed[0], N, N, ROW_BYTES,
                        input, gate, temp);
    oc_matvec_quantized(OC_QUANT_Q4_0, packed[1], N, N, ROW_BYTES,
                        input, up, temp);
    oc_swiglu_inplace_f32(gate, up, N);
    oc_matvec_quantized(OC_QUANT_Q4_0, packed[2], N, N, ROW_BYTES,
                        gate, expected, temp);
    float label = 0;
    OcLlamaConfig cfg = {
        .n_embd = N, .expert_latent_dim = N, .expert_latent_group_count = 1,
        .expert_intermediate_size = N, .num_experts = 1,
    };
    OcLlamaLayer layer = {
        .ffn_latent_d = f32_view(identity, N, N),
        .ffn_latent_u = f32_view(identity, N, N),
        .ffn_latent_labels = f32_view(&label, 1, 1),
        .expert_latent_dim = N,
    };
    OcWeightView *views[] = {&layer.ffn_gate_exps, &layer.ffn_up_exps,
                             &layer.ffn_down_exps};
    for (size_t matrix = 0; matrix < 3; matrix++)
        *views[matrix] = (OcWeightView){
            .data = packed[matrix], .qtype = OC_QUANT_Q4_0,
            .rows = N, .cols = N, .row_bytes = ROW_BYTES,
        };
    const uint32_t selected = 0;
    const float weight = 1;
    oc_matvec_fused_test_reset();
    cr_assert_eq(oc_llama_test_latent_moe_decode(&cfg, &layer, input,
        &selected, &weight, 1, actual), OC_OK);
    const OcMatvecFusedTestStats stats = oc_matvec_fused_test_stats();
    cr_assert_eq(stats.activation_quantizations, 1);
    cr_assert_eq(stats.parallel_dispatches, 1);
    for (size_t i = 0; i < N; i++)
        cr_assert_eq(memcmp(actual+i, expected+i, sizeof(float)), 0,
            "index=%zu actual=%.9g expected=%.9g", i, actual[i], expected[i]);
}

static void fixture_init(Fixture *f)
{
    memset(f, 0, sizeof(*f));
    f->cfg.n_embd = D;
    f->cfg.expert_latent_dim = Z;
    f->cfg.expert_latent_group_count = G;
    f->cfg.expert_intermediate_size = I;
    f->cfg.num_experts = E;
    f->labels[0] = 0.0f;
    f->labels[1] = 0.0f;
    f->labels[2] = 1.0f;
    f->labels[3] = 2.0f;
    for (size_t g = 0; g < G; g++) {
        for (size_t z = 0; z < Z; z++)
            for (size_t d = 0; d < D; d++)
                f->d[(g * Z + z) * D + d] =
                    0.13f * (float)(1 + g * 3 + z * 2 + d);
        for (size_t d = 0; d < D; d++)
            for (size_t z = 0; z < Z; z++)
                f->u[(g * D + d) * Z + z] =
                    0.09f * (float)(1 + g * 5 + d * 2 + z);
    }
    for (size_t e = 0; e < E; e++) {
        for (size_t i = 0; i < I; i++) {
            for (size_t z = 0; z < Z; z++) {
                f->gate[(e * I + i) * Z + z] =
                    0.07f * (float)(1 + e + i * 2 + z);
                f->up[(e * I + i) * Z + z] =
                    0.11f * (float)(1 + e * 2 + i + z);
            }
        }
        for (size_t z = 0; z < Z; z++) {
            for (size_t i = 0; i < I; i++)
                f->down[(e * Z + z) * I + i] =
                    0.05f * (float)(1 + e * 3 + z + i * 2);
        }
    }
    f->layer.ffn_latent_d = f32_view(f->d, Z, D);
    f->layer.ffn_latent_u = f32_view(f->u, D, Z);
    f->layer.ffn_latent_labels = f32_view(f->labels, 1, E);
    f->layer.ffn_gate_exps = f32_view(f->gate, I, Z);
    f->layer.ffn_up_exps = f32_view(f->up, I, Z);
    f->layer.ffn_down_exps = f32_view(f->down, Z, I);
}

static void reference(const Fixture *f, const float *x,
                      const uint32_t *selected, const float *weights,
                      uint32_t k, float *out)
{
    memset(out, 0, D * sizeof(float));
    for (uint32_t i = 0; i < k; i++) {
        const uint32_t group = (uint32_t)f->labels[selected[i]];
        float latent[Z] = {0};
        for (size_t z = 0; z < Z; z++)
            for (size_t d = 0; d < D; d++)
                latent[z] += f->d[(group * Z + z) * D + d] * x[d];
        const uint32_t e = selected[i];
        float inter[I] = {0};
        for (size_t r = 0; r < I; r++) {
            float a = 0.0f, b = 0.0f;
            for (size_t z = 0; z < Z; z++) {
                a += f->gate[(e * I + r) * Z + z] * latent[z];
                b += f->up[(e * I + r) * Z + z] * latent[z];
            }
            inter[r] = a / (1.0f + expf(-a)) * b;
        }
        float down[Z] = {0};
        for (size_t z = 0; z < Z; z++)
            for (size_t r = 0; r < I; r++)
                down[z] += f->down[(e * Z + z) * I + r] * inter[r];
        for (size_t d = 0; d < D; d++) {
            float lifted = 0.0f;
            for (size_t z = 0; z < Z; z++)
                lifted += f->u[(group * D + d) * Z + z] * down[z];
            out[d] += weights[i] * lifted;
        }
    }
}

static void check_close(const float *got, const float *want, size_t n)
{
    for (size_t i = 0; i < n; i++)
        cr_assert_float_eq(got[i], want[i], 1e-6f,
                           "element %zu: got %.9g want %.9g", i, got[i], want[i]);
}

Test(latent_moe, decode_zero_single_and_grouped_routing)
{
    Fixture f;
    fixture_init(&f);
    const float x[D] = {0.2f, -0.3f, 0.7f, 0.4f};
    const uint32_t none[] = {0};
    const float no_weight[] = {0};
    float got[D], want[D];
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x, none,
        no_weight, 0, got), OC_OK);
    memset(want, 0, sizeof(want));
    check_close(got, want, D);

    const uint32_t one[] = {2};
    const float one_w[] = {0.75f};
    reference(&f, x, one, one_w, 1, want);
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x, one,
        one_w, 1, got), OC_OK);
    check_close(got, want, D);

    const uint32_t repeated[] = {0, 1, 2, 3};
    const float repeated_w[] = {0.1f, 0.2f, 0.3f, 0.4f};
    reference(&f, x, repeated, repeated_w, 4, want);
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x,
        repeated, repeated_w, 4, got), OC_OK);
    check_close(got, want, D);
}

Test(latent_moe, prefill_groups_each_token_without_cross_group_mixing)
{
    Fixture f;
    fixture_init(&f);
    const float inputs[4 * D] = {
        0.2f, -0.3f, 0.7f, 0.4f,
        -0.1f, 0.8f, 0.2f, -0.5f,
        0.9f, 0.1f, -0.4f, 0.3f,
        0.6f, -0.2f, 0.5f, 0.1f,
    };
    const uint32_t selected[4 * 2] = {0, 2, 0, 1, 1, 3, 2, 3};
    const float weights[4 * 2] = {0.25f, 0.75f, 0.6f, 0.4f,
                                  0.3f, 0.7f, 0.45f, 0.55f};
    float got[4 * D], want[4 * D];
    cr_assert_eq(oc_llama_test_latent_moe_prefill(&f.cfg, &f.layer, inputs,
        4, selected, weights, 2, got), OC_OK);
    for (size_t row = 0; row < 4; row++)
        reference(&f, inputs + row * D, selected + row * 2,
                  weights + row * 2, 2, want + row * D);
    check_close(got, want, 4 * D);
}

Test(latent_moe, rejects_bad_labels_and_geometry)
{
    Fixture f;
    fixture_init(&f);
    const float x[D] = {1, 2, 3, 4};
    const uint32_t chosen[] = {2};
    const float weight[] = {1};
    float out[D];
    f.labels[2] = 1.5f;
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x,
        chosen, weight, 1, out), OC_ERR_MODEL);
    f.labels[2] = 3.0f;
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x,
        chosen, weight, 1, out), OC_ERR_MODEL);
    f.labels[2] = -1.0f;
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x,
        chosen, weight, 1, out), OC_ERR_MODEL);
    f.labels[2] = NAN;
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x,
        chosen, weight, 1, out), OC_ERR_MODEL);
    f.labels[2] = 1.0f;
    f.layer.ffn_latent_labels.data = NULL;
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x,
        chosen, weight, 1, out), OC_ERR_TENSOR);
    f.layer.ffn_latent_labels.data = (const uint8_t *)f.labels;
    f.layer.ffn_latent_u.cols = Z + 1;
    cr_assert_eq(oc_llama_test_latent_moe_decode(&f.cfg, &f.layer, x,
        chosen, weight, 1, out), OC_ERR_TENSOR);
}

#define QD 8u
#define QNL 2u
#define QE 16u
#define QK 2u
#define QI 4u
#define QSI 12u
#define QZ 4u
#define QG 2u
#define QHC 2u
#define QR 2u
#define QNH 2u
#define QNKV 1u
#define QHD 4u
#define QV 16u

typedef struct {
    const char *labels_name;
    uint32_t labels_n_dims;
    uint64_t labels_dims[2];
    uint32_t labels_type;
    bool mtp;
    bool mtp_latent;
} Q4eFile;

static Q4eFile q4e_file(void)
{
    return (Q4eFile){ .labels_name = "ffn_latent_labels", .labels_n_dims = 1,
                      .labels_dims = {QE, 0}, .labels_type = 0 };
}

static void q4e_tensor(OcGgufWriter *w, const char *name, uint32_t n_dims,
                       const uint64_t *dims, float base)
{
    size_t n = 1;
    for (uint32_t i = 0; i < n_dims; i++) n *= (size_t)dims[i];
    float *data = malloc(n * sizeof(float));
    cr_assert_not_null(data);
    const size_t seed = strlen(name);
    for (size_t i = 0; i < n; i++)
        data[i] = base + 0.04f * (float)((int)((i * 7u + seed) % 11u) - 5);
    cr_assert_eq(oc_gguf_writer_add_tensor(w, name, n_dims, dims, 0, data,
                                           n * sizeof(float)), OC_OK,
                 "%s", name);
    free(data);
}

static void q4e_block_tensor(OcGgufWriter *w, unsigned layer,
                             const char *suffix, uint32_t n_dims,
                             const uint64_t *dims, float base)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    q4e_tensor(w, name, n_dims, dims, base);
}

static void q4e_block(OcGgufWriter *w, unsigned l, bool latent)
{
    const uint64_t hc = QHC * QD;
    const uint64_t width = latent ? QZ : QD;
    q4e_block_tensor(w, l, "attn_q.weight", 2,
                     (const uint64_t[]){QD, 2 * QNH * QHD}, 0.0f);
    q4e_block_tensor(w, l, "attn_k.weight", 2,
                     (const uint64_t[]){QD, QNKV * QHD}, 0.0f);
    q4e_block_tensor(w, l, "attn_v.weight", 2,
                     (const uint64_t[]){QD, QNKV * QHD}, 0.0f);
    q4e_block_tensor(w, l, "attn_output.weight", 2,
                     (const uint64_t[]){QNH * QHD, QD}, 0.0f);
    q4e_block_tensor(w, l, "attn_q_norm.weight", 1, (const uint64_t[]){QHD},
                     1.0f);
    q4e_block_tensor(w, l, "attn_k_norm.weight", 1, (const uint64_t[]){QHD},
                     1.0f);
    q4e_block_tensor(w, l, "hc_attn_norm.weight", 1, (const uint64_t[]){hc},
                     1.0f);
    q4e_block_tensor(w, l, "hc_attn_down.weight", 2,
                     (const uint64_t[]){hc, QR}, 0.0f);
    q4e_block_tensor(w, l, "hc_attn_up.weight", 2,
                     (const uint64_t[]){QR, hc}, 0.0f);
    q4e_block_tensor(w, l, "hc_ffn_norm.weight", 1, (const uint64_t[]){hc},
                     1.0f);
    q4e_block_tensor(w, l, "hc_ffn_down.weight", 2,
                     (const uint64_t[]){hc, QR}, 0.0f);
    q4e_block_tensor(w, l, "hc_ffn_up.weight", 2,
                     (const uint64_t[]){QR, hc}, 0.0f);
    q4e_block_tensor(w, l, "ffn_gate_inp.weight", 2,
                     (const uint64_t[]){QD, QE}, 0.0f);
    q4e_block_tensor(w, l, "ffn_gate_exps.weight", 3,
                     (const uint64_t[]){width, QI, QE}, 0.0f);
    q4e_block_tensor(w, l, "ffn_up_exps.weight", 3,
                     (const uint64_t[]){width, QI, QE}, 0.0f);
    q4e_block_tensor(w, l, "ffn_down_exps.weight", 3,
                     (const uint64_t[]){QI, width, QE}, 0.0f);
    q4e_block_tensor(w, l, "ffn_gate_shexp.weight", 2,
                     (const uint64_t[]){QD, QSI}, 0.0f);
    q4e_block_tensor(w, l, "ffn_up_shexp.weight", 2,
                     (const uint64_t[]){QD, QSI}, 0.0f);
    q4e_block_tensor(w, l, "ffn_down_shexp.weight", 2,
                     (const uint64_t[]){QSI, QD}, 0.0f);
    q4e_block_tensor(w, l, "ffn_gate_inp_shexp.weight", 1,
                     (const uint64_t[]){QD}, 0.0f);
    if (latent) {
        q4e_block_tensor(w, l, "ffn_latent_d.weight", 3,
                         (const uint64_t[]){QD, QZ, QG}, 0.0f);
        q4e_block_tensor(w, l, "ffn_latent_u.weight", 3,
                         (const uint64_t[]){QZ, QD, QG}, 0.0f);
    }
}

static void q4e_labels(OcGgufWriter *w, unsigned layer, const Q4eFile *f)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, f->labels_name);
    size_t n = 1;
    for (uint32_t i = 0; i < f->labels_n_dims; i++)
        n *= (size_t)f->labels_dims[i];
    cr_assert_leq(n, QE);
    float values[QE];
    uint16_t halves[QE];
    for (size_t e = 0; e < n; e++) {
        values[e] = (float)(e % QG);
        halves[e] = (uint16_t)(e % QG ? 0x3c00u : 0u);
    }
    const bool f16 = f->labels_type == 1;
    cr_assert_eq(oc_gguf_writer_add_tensor(w, name, f->labels_n_dims,
        f->labels_dims, f->labels_type,
        f16 ? (const void *)halves : (const void *)values,
        n * (f16 ? sizeof(uint16_t) : sizeof(float))), OC_OK, "%s", name);
}

static void q4e_write(char *path, const Q4eFile *f)
{
    const int fd = mkstemp(path);
    cr_assert_geq(fd, 0);
    close(fd);
    OcGgufWriter w;
    cr_assert_eq(oc_gguf_writer_init(path, "qwen4exp", &w), OC_OK);
    const struct { const char *key; uint32_t value; } meta[] = {
        {"qwen4exp.block_count", QNL + (f->mtp ? 1u : 0u)},
        {"qwen4exp.context_length", 64},
        {"qwen4exp.embedding_length", QD},
        {"qwen4exp.feed_forward_length", QI},
        {"qwen4exp.attention.head_count", QNH},
        {"qwen4exp.attention.head_count_kv", QNKV},
        {"qwen4exp.attention.key_length", QHD},
        {"qwen4exp.expert_count", QE},
        {"qwen4exp.expert_used_count", QK},
        {"qwen4exp.expert_feed_forward_length", QI},
        {"qwen4exp.expert_shared_feed_forward_length", QSI},
        {"qwen4exp.expert_latent_length", QZ},
        {"qwen4exp.expert_latent_group_count", QG},
        {"qwen4exp.nextn_predict_layers", f->mtp ? 1u : 0u},
        {"qwen4exp.full_attention_interval", 1},
        {"qwen4exp.ssm.conv_kernel", 2},
        {"qwen4exp.ssm.state_size", 2},
        {"qwen4exp.ssm.group_count", 1},
        {"qwen4exp.ssm.time_step_rank", 1},
        {"qwen4exp.ssm.inner_size", 2},
        {"qwen4exp.hyper_connection.count", QHC},
        {"qwen4exp.hyper_connection.low_rank", QR},
    };
    for (size_t i = 0; i < sizeof(meta) / sizeof(meta[0]); i++)
        cr_assert_eq(oc_gguf_writer_add_uint32(&w, meta[i].key,
                                               meta[i].value), OC_OK,
                     "%s", meta[i].key);
    const uint64_t hc = QHC * QD;
    q4e_tensor(&w, "token_embd.weight", 2, (const uint64_t[]){QD, QV}, 0.0f);
    q4e_tensor(&w, "output.weight", 2, (const uint64_t[]){QD, QV}, 0.0f);
    q4e_tensor(&w, "output_hc_norm.weight", 1, (const uint64_t[]){hc}, 1.0f);
    q4e_tensor(&w, "output_hc_down.weight", 2, (const uint64_t[]){hc, QR},
               0.0f);
    q4e_tensor(&w, "output_hc_up.weight", 2, (const uint64_t[]){QR, hc},
               0.0f);
    for (unsigned l = 0; l < QNL; l++) q4e_block(&w, l, true);
    if (f->mtp) {
        q4e_block(&w, QNL, f->mtp_latent);
        if (f->mtp_latent) q4e_labels(&w, QNL, f);
        q4e_block_tensor(&w, QNL, "nextn.eh_proj.weight", 2,
                         (const uint64_t[]){2 * QD, QD}, 0.0f);
        q4e_block_tensor(&w, QNL, "nextn.enorm.weight", 1,
                         (const uint64_t[]){QD}, 1.0f);
        q4e_block_tensor(&w, QNL, "nextn.hnorm.weight", 1,
                         (const uint64_t[]){hc}, 1.0f);
        q4e_block_tensor(&w, QNL, "nextn.hc_head_norm.weight", 1,
                         (const uint64_t[]){hc}, 1.0f);
        q4e_block_tensor(&w, QNL, "nextn.hc_head_down.weight", 2,
                         (const uint64_t[]){hc, QR}, 0.0f);
        q4e_block_tensor(&w, QNL, "nextn.hc_head_up.weight", 2,
                         (const uint64_t[]){QR, hc}, 0.0f);
    }
    for (unsigned l = QNL; l-- > 0;) q4e_labels(&w, l, f);
    cr_assert_eq(oc_gguf_writer_finalize(&w), OC_OK);
    oc_gguf_writer_free(&w);
}

Test(latent_moe, loader_accepts_flat_and_row_labels_under_both_names)
{
    const char *names[] = {"ffn_latent_labels", "ffn_latent_labels.weight"};
    for (size_t name = 0; name < 2; name++) {
        for (uint32_t n_dims = 1; n_dims <= 2; n_dims++) {
            Q4eFile f = q4e_file();
            f.labels_name = names[name];
            f.labels_n_dims = n_dims;
            if (n_dims == 2) {
                f.labels_dims[0] = 1;
                f.labels_dims[1] = QE;
            }
            char path[] = "/tmp/oxidize-latent-labels-XXXXXX";
            q4e_write(path, &f);
            OcLlamaModel model;
            cr_assert_eq(oc_llama_load(path, &model), OC_OK, "%s %u-D",
                         names[name], n_dims);
            for (uint32_t l = 0; l < QNL; l++) {
                const OcWeightView *v = &model.layers[l].ffn_latent_labels;
                cr_assert_eq(v->rows, 1);
                cr_assert_eq(v->cols, QE);
                cr_assert_eq(v->row_bytes, QE * sizeof(float));
                const float *labels = (const float *)v->data;
                for (size_t e = 0; e < QE; e++)
                    cr_assert_eq(labels[e], (float)(e % QG));
            }
            oc_llama_free(&model);
            cr_assert_eq(unlink(path), 0);
        }
    }
}

Test(latent_moe, loader_rejects_mis_shaped_and_non_f32_labels)
{
    const struct {
        uint32_t n_dims;
        uint64_t dims[2];
        uint32_t type;
    } bad[] = {
        {2, {QE, 1}, 0},
        {1, {1, 0}, 0},
        {2, {1, 1}, 0},
        {1, {QE, 0}, 1},
        {2, {1, QE}, 1},
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        Q4eFile f = q4e_file();
        f.labels_n_dims = bad[i].n_dims;
        f.labels_dims[0] = bad[i].dims[0];
        f.labels_dims[1] = bad[i].dims[1];
        f.labels_type = bad[i].type;
        char path[] = "/tmp/oxidize-latent-labels-XXXXXX";
        q4e_write(path, &f);
        OcLlamaModel model;
        cr_assert_eq(oc_llama_load(path, &model), OC_ERR_TENSOR, "case %zu",
                     i);
        cr_assert_eq(unlink(path), 0);
    }
}

Test(latent_moe, full_width_mtp_head_runs_and_latent_head_is_refused)
{
    for (int latent_head = 0; latent_head < 2; latent_head++) {
        Q4eFile f = q4e_file();
        f.mtp = true;
        f.mtp_latent = latent_head != 0;
        char path[] = "/tmp/oxidize-latent-mtp-XXXXXX";
        q4e_write(path, &f);
        OcLlamaModel model;
        cr_assert_eq(oc_llama_load(path, &model), OC_OK);
        cr_assert_eq(model.cfg.n_layer, QNL);
        cr_assert_eq(oc_llama_mtp_present(&model), !latent_head);
        OcLlamaSession s;
        cr_assert_eq(oc_llama_session_init_kv(&model, &s, OC_KV_F32), OC_OK);
        cr_assert_eq(oc_llama_mtp_enable(&s, true),
                     latent_head ? OC_ERR_MODEL : OC_OK);
        if (!latent_head) {
            const uint32_t prompt[] = {1, 3, 5, 7};
            float logits[QV];
            cr_assert_eq(oc_llama_prefill(&s, prompt, 4, 4, logits), OC_OK);
            uint32_t out[OC_K2_MTP_MAX_K + 1];
            size_t n_out = 0;
            OcMtpStats st = {0};
            cr_assert_eq(oc_llama_mtp_step(&s, 2, logits, out,
                                           OC_K2_MTP_MAX_K + 1, &n_out, &st),
                         OC_OK);
            cr_assert_eq(st.drafted, 2);
            cr_assert_geq(n_out, 1);
            for (size_t i = 0; i < QV; i++) cr_assert(isfinite(logits[i]));
        }
        oc_llama_session_free(&s);
        oc_llama_free(&model);
        cr_assert_eq(unlink(path), 0);
    }
}

Test(latent_moe, decode_and_prefill_agree_when_shared_expert_is_widest)
{
    Q4eFile f = q4e_file();
    char path[] = "/tmp/oxidize-latent-forward-XXXXXX";
    q4e_write(path, &f);
    OcLlamaModel model;
    cr_assert_eq(oc_llama_load(path, &model), OC_OK);
    cr_assert_gt(model.cfg.shared_expert_intermediate_size, model.cfg.n_ff);
    cr_assert_gt(model.cfg.shared_expert_intermediate_size,
                 model.cfg.expert_intermediate_size);
    OcLlamaSession a, b;
    cr_assert_eq(oc_llama_session_init_kv(&model, &a, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_session_init_kv(&model, &b, OC_KV_F32), OC_OK);
    const uint32_t tokens[] = {1, 3, 5};
    float decoded[QV], prefilled[QV];
    for (size_t i = 0; i < 3; i++)
        cr_assert_eq(oc_llama_forward(&a, tokens[i], decoded), OC_OK);
    cr_assert_eq(oc_llama_prefill(&b, tokens, 3, 3, prefilled), OC_OK);
    for (size_t i = 0; i < QV; i++) {
        cr_assert(isfinite(decoded[i]) && isfinite(prefilled[i]));
        cr_assert_float_eq(decoded[i], prefilled[i],
                           1e-5f + fabsf(prefilled[i]) * 1e-4f,
                           "logit %zu: decode %.9g prefill %.9g", i,
                           decoded[i], prefilled[i]);
    }
    oc_llama_session_free(&a);
    oc_llama_session_free(&b);
    oc_llama_free(&model);
    cr_assert_eq(unlink(path), 0);
}
