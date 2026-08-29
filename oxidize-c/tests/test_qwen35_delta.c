#include "framework.h"

#include "oxidize/parallel.h"
#include "oxidize/qwen35_delta.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define NK 2u
#define NV 16u
#define DK 3u
#define DV 2u
#define KW 3u
#define KEY_DIM (NK * DK)
#define VALUE_DIM (NV * DV)
#define CONV_DIM (2u * KEY_DIM + VALUE_DIM)
#define CONV_STATE_LEN (CONV_DIM * (KW - 1u))
#define RECURRENT_LEN (VALUE_DIM * DK)

typedef struct {
    float conv_state[CONV_STATE_LEN];
    float recurrent[RECURRENT_LEN];
    float conv_weight[CONV_DIM * KW];
    float ssm_a[NV];
    float dt_bias[NV];
    float norm_weight[DV];
    float qkv[CONV_DIM];
    float gate[VALUE_DIM];
    float beta[NV];
    float alpha[NV];
    float conv_output[CONV_DIM];
    float output[VALUE_DIM];
    OcQwen35DeltaState state;
    OcQwen35DeltaParams params;
    OcQwen35DeltaInput input;
} Fixture;

static float sample(size_t i, size_t salt)
{
    uint32_t x = (uint32_t)(i + 1u) * 747796405u +
                 (uint32_t)(salt + 3u) * 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (float)((int32_t)(x >> 12u) % 2001 - 1000) / 1000.0f;
}

static void fixture_init(Fixture *f)
{
    memset(f, 0, sizeof(*f));
    for (size_t i = 0; i < CONV_DIM * KW; i++)
        f->conv_weight[i] = sample(i, 1) * 0.3f;
    for (size_t i = 0; i < NV; i++) {
        f->ssm_a[i] = -0.15f - 0.01f * (float)i;
        f->dt_bias[i] = sample(i, 2) * 0.2f;
        f->beta[i] = sample(i, 3) * 1.5f;
        f->alpha[i] = sample(i, 4) * 2.0f;
    }
    for (size_t i = 0; i < DV; i++) f->norm_weight[i] = 0.8f + 0.1f * (float)i;
    for (size_t i = 0; i < CONV_DIM; i++) f->qkv[i] = sample(i, 5);
    for (size_t i = 0; i < VALUE_DIM; i++) f->gate[i] = sample(i, 6) * 1.3f;

    OcQwen35DeltaGeometry geometry = {NK, NV, DK, DV, KW};
    cr_assert_eq(oc_qwen35_delta_state_init(&f->state, &geometry,
                                            f->conv_state, CONV_STATE_LEN,
                                            f->recurrent, RECURRENT_LEN), OC_OK);
    f->params = (OcQwen35DeltaParams){
        .conv_weight = f->conv_weight,
        .conv_weight_len = CONV_DIM * KW,
        .ssm_a = f->ssm_a,
        .ssm_a_len = NV,
        .dt_bias = f->dt_bias,
        .dt_bias_len = NV,
        .norm_weight = f->norm_weight,
        .norm_weight_len = DV,
        .norm_eps = 1e-6f,
    };
    f->input = (OcQwen35DeltaInput){
        f->qkv, CONV_DIM, f->gate, VALUE_DIM,
        f->beta, NV, f->alpha, NV
    };
}

static float ref_sigmoid(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

static float ref_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static float ref_silu(float x)
{
    if (x >= 0.0f) return x / (1.0f + expf(-x));
    float e = expf(x);
    return x * e / (1.0f + e);
}

static void scalar_step(float *conv_state, float *recurrent,
                        const OcQwen35DeltaParams *p,
                        const OcQwen35DeltaInput *in,
                        float *conv, float *out)
{
    for (size_t c = 0; c < CONV_DIM; c++) {
        float *past = conv_state + c * (KW - 1u);
        const float *w = p->conv_weight + c * KW;
        float sum = w[KW - 1u] * in->qkv[c];
        for (size_t i = 0; i < KW - 1u; i++) sum += w[i] * past[i];
        for (size_t i = 0; i + 1u < KW - 1u; i++) past[i] = past[i + 1u];
        past[KW - 2u] = in->qkv[c];
        conv[c] = ref_silu(sum);
    }
    for (size_t h = 0; h < NK; h++) {
        float *q = conv + h * DK;
        float *k = conv + KEY_DIM + h * DK;
        float qs = 0.0f, ks = 0.0f;
        for (size_t i = 0; i < DK; i++) { qs += q[i] * q[i]; ks += k[i] * k[i]; }
        float qi = 1.0f / sqrtf(qs > p->norm_eps ? qs : p->norm_eps);
        float ki = 1.0f / sqrtf(ks > p->norm_eps ? ks : p->norm_eps);
        for (size_t i = 0; i < DK; i++) {
            q[i] *= qi / sqrtf((float)DK);
            k[i] *= ki;
        }
    }
    for (size_t h = 0; h < NV; h++) {
        const size_t key_head = h / (NV / NK);
        const float *q = conv + key_head * DK;
        const float *k = conv + KEY_DIM + key_head * DK;
        const float *v = conv + 2u * KEY_DIM + h * DV;
        float decay = expf(p->ssm_a[h] * ref_softplus(in->alpha[h] + p->dt_bias[h]));
        float beta = ref_sigmoid(in->beta[h]);
        for (size_t i = 0; i < DV; i++) {
            float *row = recurrent + (h * DV + i) * DK;
            float sk = 0.0f;
            for (size_t j = 0; j < DK; j++) { row[j] *= decay; sk += row[j] * k[j]; }
            float delta = (v[i] - sk) * beta;
            out[h * DV + i] = 0.0f;
            for (size_t j = 0; j < DK; j++) {
                row[j] += delta * k[j];
                out[h * DV + i] += row[j] * q[j];
            }
        }
        float ss = 0.0f;
        for (size_t i = 0; i < DV; i++) ss += out[h * DV + i] * out[h * DV + i];
        float inv = 1.0f / sqrtf(ss / (float)DV + p->norm_eps);
        for (size_t i = 0; i < DV; i++)
            out[h * DV + i] *= inv * p->norm_weight[i] * ref_silu(in->gate[h * DV + i]);
    }
}

static void assert_close(const float *got, const float *want, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float tolerance = 1e-4f * fmaxf(1.0f, fabsf(want[i]));
        cr_assert_float_eq(got[i], want[i], tolerance,
                           "index %zu: got %.9g want %.9g", i, got[i], want[i]);
    }
}

static void run_step(Fixture *f)
{
    cr_assert_eq(oc_qwen35_delta_step(&f->state, &f->params, &f->input,
                                      f->conv_output, CONV_DIM,
                                      f->output, VALUE_DIM), OC_OK);
}

Test(qwen35_delta, one_step_matches_scalar)
{
    Fixture got, ref;
    fixture_init(&got);
    fixture_init(&ref);
    run_step(&got);
    scalar_step(ref.conv_state, ref.recurrent, &ref.params, &ref.input,
                ref.conv_output, ref.output);
    assert_close(got.conv_state, ref.conv_state, CONV_STATE_LEN);
    assert_close(got.recurrent, ref.recurrent, RECURRENT_LEN);
    assert_close(got.output, ref.output, VALUE_DIM);
}

Test(qwen35_delta, two_steps_match_scalar)
{
    Fixture got, ref;
    fixture_init(&got);
    fixture_init(&ref);
    for (size_t step = 0; step < 2; step++) {
        for (size_t i = 0; i < CONV_DIM; i++) {
            got.qkv[i] += 0.03f * (float)step;
            ref.qkv[i] += 0.03f * (float)step;
        }
        run_step(&got);
        scalar_step(ref.conv_state, ref.recurrent, &ref.params, &ref.input,
                    ref.conv_output, ref.output);
    }
    assert_close(got.conv_state, ref.conv_state, CONV_STATE_LEN);
    assert_close(got.recurrent, ref.recurrent, RECURRENT_LEN);
    assert_close(got.output, ref.output, VALUE_DIM);
}

Test(qwen35_delta, reset_replays)
{
    Fixture f;
    float first[VALUE_DIM];
    fixture_init(&f);
    run_step(&f);
    memcpy(first, f.output, sizeof(first));
    oc_qwen35_delta_state_reset(&f.state);
    run_step(&f);
    assert_close(f.output, first, VALUE_DIM);
    oc_qwen35_delta_state_free(&f.state);
    cr_assert_null(f.state.conv_state);
    cr_assert_null(f.state.recurrent_state);
}

Test(qwen35_delta, extreme_alpha_is_finite)
{
    Fixture f;
    fixture_init(&f);
    for (size_t i = 0; i < NV; i++) f.alpha[i] = (i & 1u) ? 100.0f : -100.0f;
    run_step(&f);
    for (size_t i = 0; i < RECURRENT_LEN; i++) cr_assert(isfinite(f.recurrent[i]));
    for (size_t i = 0; i < VALUE_DIM; i++) cr_assert(isfinite(f.output[i]));
}

Test(qwen35_delta, rejects_invalid_geometry)
{
    float conv_state[8] = {1.0f};
    float recurrent[8] = {2.0f};
    float conv_before[8], recurrent_before[8];
    memcpy(conv_before, conv_state, sizeof(conv_state));
    memcpy(recurrent_before, recurrent, sizeof(recurrent));
    OcQwen35DeltaState state = {0};
    OcQwen35DeltaGeometry bad = {2, 3, 2, 2, 3};
    cr_assert_eq(oc_qwen35_delta_state_init(&state, &bad, conv_state, 8,
                                            recurrent, 8), OC_ERR_INVALID_ARG);
    cr_assert_arr_eq(conv_state, conv_before, sizeof(conv_state));
    cr_assert_arr_eq(recurrent, recurrent_before, sizeof(recurrent));

    Fixture f;
    fixture_init(&f);
    float state_before[CONV_STATE_LEN];
    memcpy(state_before, f.conv_state, sizeof(state_before));
    f.input.qkv_len--;
    cr_assert_eq(oc_qwen35_delta_step(&f.state, &f.params, &f.input,
                                      f.conv_output, CONV_DIM,
                                      f.output, VALUE_DIM), OC_ERR_INVALID_ARG);
    cr_assert_arr_eq(f.conv_state, state_before, sizeof(state_before));
}

Test(qwen35_delta, rejects_short_parameter_buffers)
{
    Fixture f;
    fixture_init(&f);
    f.params.ssm_a_len--;
    cr_assert_eq(oc_qwen35_delta_step(&f.state, &f.params, &f.input,
                                      f.conv_output, CONV_DIM,
                                      f.output, VALUE_DIM), OC_ERR_INVALID_ARG);

    fixture_init(&f);
    f.params.dt_bias_len--;
    cr_assert_eq(oc_qwen35_delta_step(&f.state, &f.params, &f.input,
                                      f.conv_output, CONV_DIM,
                                      f.output, VALUE_DIM), OC_ERR_INVALID_ARG);

    fixture_init(&f);
    f.params.norm_weight_len--;
    cr_assert_eq(oc_qwen35_delta_step(&f.state, &f.params, &f.input,
                                      f.conv_output, CONV_DIM,
                                      f.output, VALUE_DIM), OC_ERR_INVALID_ARG);
}

Test(qwen35_delta, one_and_sixteen_threads_match)
{
    Fixture serial, threaded;
    fixture_init(&serial);
    fixture_init(&threaded);
    cr_assert_eq(oc_parallel_set_threads(1), OC_OK);
    run_step(&serial);
    cr_assert_eq(oc_parallel_set_threads(16), OC_OK);
    run_step(&threaded);
    cr_assert_eq(oc_parallel_set_threads(1), OC_OK);
    assert_close(threaded.conv_state, serial.conv_state, CONV_STATE_LEN);
    assert_close(threaded.recurrent, serial.recurrent, RECURRENT_LEN);
    assert_close(threaded.output, serial.output, VALUE_DIM);
}
