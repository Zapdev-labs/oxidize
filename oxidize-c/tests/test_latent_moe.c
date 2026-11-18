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
