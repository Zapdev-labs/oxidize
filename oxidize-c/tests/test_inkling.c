#define _POSIX_C_SOURCE 200809L
/* test_inkling.c — Criterion tests for src/model/inkling.c on a tiny
 * synthetic checkpoint (tests/inkling_fixture.h): the layer-visit schedule,
 * batched vs token-by-token forward, rollback, and the SWA ring. */
#include <criterion/criterion.h>

#include "inkling_fixture.h"
#include "oxidize/inkling.h"

static const char *g_dir;
static char g_gguf[512];

static void setup(void)
{
    g_dir = fx_build();
    cr_assert_not_null(g_dir);
    snprintf(g_gguf, sizeof g_gguf, "%s/model.gguf", g_dir);
    OcInklingConvertOptions o = { .hf_dir = g_dir, .out_path = g_gguf };
    cr_assert_eq(oc_inkling_convert(&o), OC_OK);
}

static void teardown(void) { fx_cleanup(g_dir); }

static float max_diff(const float *a, const float *b, size_t n)
{
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        cr_assert(isfinite(a[i]) && isfinite(b[i]));
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static const uint32_t TOKS[40] = { 3, 17, 99, 4, 200, 7, 7, 31, 150, 2, 64, 65, 66, 1, 0, 249,
                                   12, 13, 14, 15, 101, 102, 103, 104, 5, 6, 8, 9, 10, 11, 18, 19,
                                   20, 21, 22, 23, 24, 25, 26, 27 };

Test(inkling, visit_schedule, .init = setup, .fini = teardown)
{
    OcInklingModel m;
    cr_assert_eq(oc_inkling_load(g_gguf, &m), OC_OK);
    const uint8_t want[] = { 0, 1, 2, 3, 2, 3, 2, 3, 4, 5 };
    cr_assert_eq(m.n_visits, 10u);
    for (uint32_t i = 0; i < m.n_visits; i++) cr_assert_eq(m.visit_layer[i], want[i], "visit %u", i);
    cr_assert(m.layers[0].swa && !m.layers[1].swa);
    cr_assert_eq(m.n_vocab, (uint32_t)FX_V);
    oc_inkling_free(&m);
}

/* A batched forward must match feeding the same tokens one at a time: the
 * per-token activation quantization, causal masking inside the batch, the
 * convolution history and the expert grouping all have to line up. */
Test(inkling, batch_matches_sequential, .init = setup, .fini = teardown)
{
    OcInklingModel m;
    cr_assert_eq(oc_inkling_load(g_gguf, &m), OC_OK);
    OcInklingSession *a = NULL, *b = NULL;
    cr_assert_eq(oc_inkling_session_new(&m, 64, NULL, 0, &a), OC_OK);
    cr_assert_eq(oc_inkling_session_new(&m, 64, NULL, 0, &b), OC_OK);
    float la[OC_INK_MAX_BATCH * FX_V], lb[FX_V * 8];
    cr_assert_eq(oc_inkling_forward(a, TOKS, 8, la, true), OC_OK);
    for (size_t t = 0; t < 8; t++) cr_assert_eq(oc_inkling_forward(b, TOKS + t, 1, lb + t * FX_V, false), OC_OK);
    cr_assert_lt(max_diff(la, lb, 8 * FX_V), 1e-4f);
    oc_inkling_session_free(a);
    oc_inkling_session_free(b);
    oc_inkling_free(&m);
}

/* Rolling a verify batch back to a prefix and continuing must equal a
 * session that never saw the rejected tokens. */
Test(inkling, rollback_matches_fresh, .init = setup, .fini = teardown)
{
    OcInklingModel m;
    cr_assert_eq(oc_inkling_load(g_gguf, &m), OC_OK);
    OcInklingSession *a = NULL, *b = NULL;
    cr_assert_eq(oc_inkling_session_new(&m, 64, NULL, 0, &a), OC_OK);
    cr_assert_eq(oc_inkling_session_new(&m, 64, NULL, 0, &b), OC_OK);
    float la[OC_INK_MAX_BATCH * FX_V], lb[OC_INK_MAX_BATCH * FX_V];

    cr_assert_eq(oc_inkling_forward(a, TOKS, 4, NULL, false), OC_OK);
    const uint32_t verify[5] = { 9, 8, 7, 6, 5 };
    cr_assert_eq(oc_inkling_forward(a, verify, 5, NULL, true), OC_OK);
    cr_assert_eq(oc_inkling_rollback(a, 2), OC_OK);
    cr_assert_eq(oc_inkling_session_pos(a), 6u);
    const uint32_t next[3] = { 40, 41, 42 };
    cr_assert_eq(oc_inkling_forward(a, next, 3, la, true), OC_OK);

    const uint32_t seq[9] = { 3, 17, 99, 4, 9, 8, 40, 41, 42 };
    cr_assert_eq(oc_inkling_forward(b, seq, 6, NULL, false), OC_OK);
    cr_assert_eq(oc_inkling_forward(b, seq + 6, 3, lb, true), OC_OK);
    cr_assert_lt(max_diff(la, lb, 3 * FX_V), 1e-4f);

    /* a rollback can only keep tokens of the most recent call */
    cr_assert_eq(oc_inkling_forward(a, next, 2, NULL, false), OC_OK);
    cr_assert_neq(oc_inkling_rollback(a, 3), OC_OK);
    cr_assert_eq(oc_inkling_rollback(a, 1), OC_OK);
    cr_assert_eq(oc_inkling_session_pos(a), 10u);
    oc_inkling_session_free(a);
    oc_inkling_session_free(b);
    oc_inkling_free(&m);
}

/* Past n_swa + batch positions the SWA caches wrap; chunked prefill and
 * token-by-token decode must still agree. */
Test(inkling, swa_ring_wraps, .init = setup, .fini = teardown)
{
    OcInklingModel m;
    cr_assert_eq(oc_inkling_load(g_gguf, &m), OC_OK);
    OcInklingSession *a = NULL, *b = NULL;
    cr_assert_eq(oc_inkling_session_new(&m, 128, NULL, 0, &a), OC_OK);
    cr_assert_eq(oc_inkling_session_new(&m, 128, NULL, 0, &b), OC_OK);
    float la[FX_V], lb[FX_V];
    uint32_t seq[80];
    for (size_t i = 0; i < 80; i++) seq[i] = TOKS[i % 40];
    cr_assert_eq(oc_inkling_prefill(a, seq, 80, la), OC_OK);
    for (size_t i = 0; i < 80; i++) cr_assert_eq(oc_inkling_forward(b, seq + i, 1, lb, false), OC_OK);
    cr_assert_lt(max_diff(la, lb, FX_V), 1e-4f);
    cr_assert_eq(oc_inkling_forward(a, seq, 1, NULL, false), OC_OK);
    oc_inkling_session_free(a);
    oc_inkling_session_free(b);
    oc_inkling_free(&m);
}

Test(inkling, rejects_bad_arguments, .init = setup, .fini = teardown)
{
    OcInklingModel m;
    cr_assert_eq(oc_inkling_load(g_gguf, &m), OC_OK);
    OcInklingSession *s = NULL;
    cr_assert_eq(oc_inkling_session_new(&m, 4, NULL, 0, &s), OC_OK);
    const uint32_t bad = FX_V;
    cr_assert_neq(oc_inkling_forward(s, &bad, 1, NULL, false), OC_OK);
    cr_assert_neq(oc_inkling_forward(s, TOKS, 5, NULL, false), OC_OK);
    cr_assert_eq(oc_inkling_forward(s, TOKS, 4, NULL, false), OC_OK);
    cr_assert_neq(oc_inkling_forward(s, TOKS, 1, NULL, false), OC_OK);
    oc_inkling_session_free(s);
    cr_assert(oc_inkling_gguf_is_inkling(g_gguf));
    oc_inkling_free(&m);
}
