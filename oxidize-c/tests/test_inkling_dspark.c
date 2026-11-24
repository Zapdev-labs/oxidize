#define _POSIX_C_SOURCE 200809L
/* test_inkling_dspark.c — Criterion tests for src/model/inkling_dspark.c:
 * drafting is lossless (speculative output equals plain greedy) and the
 * drafter produces in-vocabulary blocks. */
#include <criterion/criterion.h>

#include "inkling_fixture.h"
#include "oxidize/inkling.h"

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

typedef struct { uint32_t ids[64]; size_t n; } Rec;

static bool rec(uint32_t t, void *u)
{
    Rec *r = u;
    if (r->n < 64) r->ids[r->n++] = t;
    return true;
}

Test(inkling_dspark, speculative_equals_greedy, .init = setup, .fini = teardown)
{
    OcInklingModel m;
    cr_assert_eq(oc_inkling_load(g_gguf, &m), OC_OK);
    OcInklingDraft d;
    cr_assert_eq(oc_inkling_draft_load(g_draft, &m, &d), OC_OK);
    const uint32_t prompt[6] = { 3, 17, 99, 4, 200, 7 };

    Rec plain = { { 0 }, 0 }, spec = { { 0 }, 0 };
    OcInklingSession *s = NULL;
    cr_assert_eq(oc_inkling_session_new(&m, 128, NULL, 0, &s), OC_OK);
    cr_assert_eq(oc_inkling_generate(s, NULL, prompt, 6, 40, 0, 0.0f, rec, &plain, NULL), OC_OK);
    oc_inkling_session_free(s);

    OcInklingSpec *sp = NULL;
    OcInklingSpecStats st;
    cr_assert_eq(oc_inkling_session_new(&m, 128, d.target_layers, d.n_targets, &s), OC_OK);
    cr_assert_eq(oc_inkling_spec_new(&d, s, 128, &sp), OC_OK);
    cr_assert_eq(oc_inkling_generate(s, sp, prompt, 6, 40, 3, 0.0f, rec, &spec, &st), OC_OK);
    cr_assert_eq(spec.n, plain.n);
    for (size_t i = 0; i < plain.n; i++) cr_assert_eq(spec.ids[i], plain.ids[i], "token %zu", i);
    cr_assert_gt(st.rounds, 0u);
    cr_assert_gt(st.drafted, 0u);
    cr_assert_eq(st.emitted, plain.n);
    oc_inkling_spec_free(sp);
    oc_inkling_session_free(s);
    oc_inkling_draft_free(&d);
    oc_inkling_free(&m);
}

Test(inkling_dspark, draft_block_in_vocab, .init = setup, .fini = teardown)
{
    OcInklingModel m;
    cr_assert_eq(oc_inkling_load(g_gguf, &m), OC_OK);
    OcInklingDraft d;
    cr_assert_eq(oc_inkling_draft_load(g_draft, &m, &d), OC_OK);
    OcInklingSession *s = NULL;
    OcInklingSpec *sp = NULL;
    cr_assert_eq(oc_inkling_session_new(&m, 64, d.target_layers, d.n_targets, &s), OC_OK);
    cr_assert_eq(oc_inkling_spec_new(&d, s, 64, &sp), OC_OK);
    uint32_t out[8], n = 99;
    /* nothing injected yet: no draft */
    cr_assert_eq(oc_inkling_spec_draft(sp, 5, 3, out, &n), OC_OK);
    cr_assert_eq(n, 0u);
    const uint32_t prompt[4] = { 1, 2, 3, 4 };
    cr_assert_eq(oc_inkling_forward(s, prompt, 4, NULL, false), OC_OK);
    cr_assert_eq(oc_inkling_spec_inject(sp, 4), OC_OK);
    cr_assert_eq(oc_inkling_spec_draft(sp, 5, 7, out, &n), OC_OK);
    cr_assert_eq(n, 3u, "clamped to block_size - 1");
    for (uint32_t i = 0; i < n; i++) cr_assert_lt(out[i], (uint32_t)FX_V);
    oc_inkling_spec_free(sp);
    oc_inkling_session_free(s);
    oc_inkling_draft_free(&d);
    oc_inkling_free(&m);
}
