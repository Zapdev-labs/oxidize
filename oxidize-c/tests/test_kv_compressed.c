/* test_kv_compressed.c — CompressedKvCache facade. */
#include <criterion/criterion.h>
#include "oxidize/kv_compressed.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float lcg(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(*state >> 40) / (float)(1u << 24) * 2.0f - 1.0f;
}

static void rope_ref_interleaved(const float *row, size_t pos, size_t head_dim,
                                 float theta, float *out)
{
    size_t p;
    for (p = 0; p < head_dim / 2; p++) {
        const float freq =
            powf(theta, -2.0f * (float)p / (float)head_dim);
        const float a = freq * (float)pos;
        const float c = cosf(a);
        const float s = sinf(a);
        out[2 * p]     = row[2 * p] * c - row[2 * p + 1] * s;
        out[2 * p + 1] = row[2 * p] * s + row[2 * p + 1] * c;
    }
}

Test(kv_compressed, default_scheme_is_rotor)
{
    OcCompressedKvCache cache;
    cr_assert_eq(oc_compressed_kv_init(&cache, 128, OC_KV_SCHEME_ROTOR, 64,
                                       10000.0f),
                 OC_OK);
    cr_assert_eq(oc_compressed_kv_scheme(&cache), OC_KV_SCHEME_ROTOR);
    cr_assert_not_null(oc_compressed_kv_rotor(&cache));
    cr_assert_null(oc_compressed_kv_helix(&cache));
    oc_compressed_kv_free(&cache);
}

Test(kv_compressed, both_schemes_track_f32_reference)
{
    const size_t head_dim = 128, tokens = 64;
    const float theta = 10000.0f;
    uint64_t rng = 31;
    float *keys, *values, *query, *ref, *rq, *row, *rk, *out, *scores;
    size_t *positions;
    size_t t, i;
    float max_s, z, scale;
    OcKvScheme schemes[2];
    int si;
    keys = malloc(tokens * head_dim * sizeof(float));
    values = malloc(tokens * head_dim * sizeof(float));
    query = malloc(head_dim * sizeof(float));
    ref = calloc(head_dim, sizeof(float));
    rq = malloc(head_dim * sizeof(float));
    row = malloc(head_dim * sizeof(float));
    rk = malloc(head_dim * sizeof(float));
    out = malloc(head_dim * sizeof(float));
    scores = malloc(tokens * sizeof(float));
    positions = malloc(tokens * sizeof(size_t));
    for (i = 0; i < tokens * head_dim; i++) keys[i] = lcg(&rng);
    for (i = 0; i < tokens * head_dim; i++) values[i] = lcg(&rng);
    for (i = 0; i < head_dim; i++) query[i] = lcg(&rng);
    for (t = 0; t < tokens; t++) positions[t] = t;
    rope_ref_interleaved(query, tokens, head_dim, theta, rq);
    scale = 1.0f / sqrtf((float)head_dim);
    max_s = -1.0e30f;
    for (t = 0; t < tokens; t++) {
        float dot = 0.0f;
        memcpy(row, keys + t * head_dim, head_dim * sizeof(float));
        rope_ref_interleaved(row, t, head_dim, theta, rk);
        for (i = 0; i < head_dim; i++) dot += rq[i] * rk[i];
        scores[t] = dot * scale;
        if (scores[t] > max_s) max_s = scores[t];
    }
    z = 0.0f;
    for (t = 0; t < tokens; t++) {
        scores[t] = expf(scores[t] - max_s);
        z += scores[t];
    }
    for (t = 0; t < tokens; t++) {
        for (i = 0; i < head_dim; i++)
            ref[i] += scores[t] / z * values[t * head_dim + i];
    }

    schemes[0] = OC_KV_SCHEME_ROTOR;
    schemes[1] = OC_KV_SCHEME_HELIX;
    for (si = 0; si < 2; si++) {
        OcCompressedKvCache cache;
        const float tol = schemes[si] == OC_KV_SCHEME_HELIX ? 0.25f : 0.1f;
        cr_assert_eq(oc_compressed_kv_init(&cache, head_dim, schemes[si],
                                           tokens, theta),
                     OC_OK);
        cr_assert_eq(oc_compressed_kv_store_page(&cache, 0, 0, keys, values,
                                                 positions, tokens),
                     OC_OK);
        cr_assert_eq(oc_compressed_kv_attention(&cache, 0, 0, query, head_dim,
                                                tokens, out),
                     OC_OK);
        for (i = 0; i < head_dim; i++) {
            cr_assert(fabsf(out[i] - ref[i]) <= tol,
                      "scheme %d dim %zu: %f vs ref %f", (int)schemes[si], i,
                      out[i], ref[i]);
        }
        cr_assert(oc_compressed_kv_compression_ratio(&cache) > 6.0f,
                  "scheme %d ratio %f", (int)schemes[si],
                  oc_compressed_kv_compression_ratio(&cache));
        oc_compressed_kv_free(&cache);
    }
    free(keys); free(values); free(query); free(ref); free(rq); free(row);
    free(rk); free(out); free(scores); free(positions);
}

Test(kv_compressed, query_oob_is_rejected)
{
    OcCompressedKvCache cache;
    float q[4] = {1, 0, 0, 0};
    float out[128];
    cr_assert_eq(oc_compressed_kv_init(&cache, 128, OC_KV_SCHEME_ROTOR, 8,
                                       10000.0f),
                 OC_OK);
    cr_assert_eq(oc_compressed_kv_attention(&cache, 0, 0, q, 4, 0, out),
                 OC_ERR_INVALID_ARG,
                 "P1: short query must not index past query_n during RoPE");
    oc_compressed_kv_free(&cache);
}
