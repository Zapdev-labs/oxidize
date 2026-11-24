#define _POSIX_C_SOURCE 200809L
/* test_inkling_convert.c — Criterion tests for src/model/inkling_convert.c:
 * NVFP4 experts are copied bit for bit (with the w13 gate/up rows split),
 * MXFP8 experts and dense weights requantize within NVFP4 error. */
#include <criterion/criterion.h>

#include "inkling_fixture.h"
#include "oxidize/gguf.h"
#include "oxidize/inkling.h"
#include "oxidize/nvfp4.h"
#include "oxidize/safetensors.h"

static const char *g_dir;
static char g_gguf[512], g_draft[512];

static void setup(void)
{
    g_dir = fx_build();
    cr_assert_not_null(g_dir);
    snprintf(g_gguf, sizeof g_gguf, "%s/model.gguf", g_dir);
    snprintf(g_draft, sizeof g_draft, "%s/dspark.gguf", g_dir);
    OcInklingConvertOptions o = { .hf_dir = g_dir, .out_path = g_gguf, .draft_out_path = g_draft };
    cr_assert_eq(oc_inkling_convert(&o), OC_OK);
}

static void teardown(void) { fx_cleanup(g_dir); }

static float e4m3(uint8_t b)
{
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}

static const float E2M1[16] = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, -0.5f, -1, -1.5f, -2, -3, -4, -6 };

static const uint8_t *st_data(const OcSafetensorsFile *f, const char *name)
{
    const OcSafetensorsTensor *t = NULL;
    const void *p = NULL;
    cr_assert_eq(oc_safetensors_get_tensor(f, name, &t), OC_OK, "%s", name);
    cr_assert_eq(oc_safetensors_get_tensor_data(f, t, &p), OC_OK);
    return p;
}

Test(inkling_convert, nvfp4_experts_bit_exact, .init = setup, .fini = teardown)
{
    char p[600];
    snprintf(p, sizeof p, "%s/model-00001-of-00001.safetensors", g_dir);
    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(p, &st), OC_OK);
    const uint8_t *w = st_data(&st, "model.llm.layers.1.mlp.experts.w13_weight");
    const uint8_t *sc = st_data(&st, "model.llm.layers.1.mlp.experts.w13_weight_scale");
    const float *amax = (const float *)st_data(&st, "model.llm.layers.1.mlp.experts.w13_weight_amax");
    const float denom = *(const float *)st_data(&st, "model.llm.layers.1.mlp.experts.w13_weight_outer_scale_denominator");

    OcGgufMmappedFile map;
    cr_assert_eq(oc_gguf_map_open(g_gguf, &map), OC_OK);
    const OcGgufTensorInfo *ti = oc_gguf_map_tensor_get(&map, "blk.1.ffn_gate_up_exps.weight");
    const OcGgufTensorInfo *si = oc_gguf_map_tensor_get(&map, "blk.1.ffn_gate_up_exps.scale");
    cr_assert_not_null(ti);
    cr_assert_not_null(si);
    cr_assert_eq(ti->ggml_type, 40u);
    const uint8_t *g = oc_gguf_map_tensor_data(&map, ti);
    const float *gs = (const float *)oc_gguf_map_tensor_data(&map, si);
    const size_t rows = 2 * FX_FF, cols = FX_E, rb = oc_nvfp4_row_bytes(cols);
    float out[FX_E];
    for (size_t e = 0; e < FX_NE; e++) {
        cr_assert_eq(gs[e], amax[e] / denom);
        for (size_t r = 0; r < rows; r++) {
            const size_t sr = r < rows / 2 ? 2 * r : 2 * (r - rows / 2) + 1;
            oc_nvfp4_dequant_row(g + (e * rows + r) * rb, cols, out);
            for (size_t c = 0; c < cols; c++) {
                const uint8_t byte = w[(e * rows + sr) * (cols / 2) + c / 2];
                const float want = E2M1[c % 2 ? byte >> 4 : byte & 15] * e4m3(sc[(e * rows + sr) * (cols / 16) + c / 16] & 0x7F);
                cr_assert_eq(out[c] * gs[e], want * gs[e], "expert %zu row %zu col %zu", e, r, c);
            }
        }
    }
    oc_gguf_map_free(&map);
    oc_safetensors_close(&st);
}

Test(inkling_convert, mxfp8_experts_requantized, .init = setup, .fini = teardown)
{
    char p[600];
    snprintf(p, sizeof p, "%s/model-00001-of-00001.safetensors", g_dir);
    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(p, &st), OC_OK);
    const uint8_t *w = st_data(&st, "model.llm.layers.5.mlp.experts.w2_weight");
    const uint8_t *sc = st_data(&st, "model.llm.layers.5.mlp.experts.w2_weight_scale");
    OcGgufMmappedFile map;
    cr_assert_eq(oc_gguf_map_open(g_gguf, &map), OC_OK);
    const OcGgufTensorInfo *ti = oc_gguf_map_tensor_get(&map, "blk.5.ffn_down_exps.weight");
    const float *gs = (const float *)oc_gguf_map_tensor_data(&map, oc_gguf_map_tensor_get(&map, "blk.5.ffn_down_exps.scale"));
    const uint8_t *g = oc_gguf_map_tensor_data(&map, ti);
    const size_t rows = FX_E, cols = FX_FF, rb = oc_nvfp4_row_bytes(cols);
    double err = 0.0, ref = 0.0;
    float out[FX_FF];
    for (size_t e = 0; e < FX_NE; e++)
        for (size_t r = 0; r < rows; r++) {
            oc_nvfp4_dequant_row(g + (e * rows + r) * rb, cols, out);
            for (size_t c = 0; c < cols; c++) {
                const float want = e4m3(w[(e * rows + r) * cols + c]) * ldexpf(1.0f, (int)sc[(e * rows + r) * (cols / 32) + c / 32] - 127);
                const float d = out[c] * gs[e] - want;
                err += (double)d * d;
                ref += (double)want * want;
            }
        }
    cr_assert_lt(sqrt(err / ref), 0.15, "relative rms %f", sqrt(err / ref));
    oc_gguf_map_free(&map);
    oc_safetensors_close(&st);
}

Test(inkling_convert, draft_metadata, .init = setup, .fini = teardown)
{
    OcInklingModel m;
    cr_assert_eq(oc_inkling_load(g_gguf, &m), OC_OK);
    OcInklingDraft d;
    cr_assert_eq(oc_inkling_draft_load(g_draft, &m, &d), OC_OK);
    cr_assert_eq(d.n_layer, (uint32_t)FX_D_L);
    cr_assert_eq(d.n_targets, 4u);
    cr_assert_eq(d.block_size, 4u);
    cr_assert(d.layers[0].swa && !d.layers[1].swa);
    float s = 0.0f;
    for (int i = 0; i < 4; i++) s += d.layers[0].ctx_mix[i];
    cr_assert_float_eq(s, 1.0f, 1e-5f);
    oc_inkling_draft_free(&d);
    oc_inkling_free(&m);
}

Test(inkling_convert, missing_dir_fails)
{
    OcInklingConvertOptions o = { .hf_dir = "/nonexistent/inkling", .out_path = "/tmp/never.gguf" };
    cr_assert_neq(oc_inkling_convert(&o), OC_OK);
}
