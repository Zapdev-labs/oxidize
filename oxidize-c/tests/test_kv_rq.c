/* test_kv_rq.c — OC_KV_RQ codec, fused kernels and lazily-committed cache. */
#include <criterion/criterion.h>
#include "oxidize/kv_rq.h"
#include "oxidize/flash_attention.h"   /* oc_f16_to_f32_bits */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_s;
static void rs(uint64_t s) { g_s = s; }
static double ru(void)
{
    g_s = g_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((g_s >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
}
static float rg(void)
{
    double u1 = ru(), u2 = ru();
    if (u1 < 1e-300) u1 = 1e-300;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

/* Vectors with a few large outlier channels, like post-RoPE keys. */
static void fill_outlier(float *x, size_t d)
{
    for (size_t i = 0; i < d; i++) x[i] = rg();
    x[3] *= 12.0f;
    x[d / 2 + 1] *= 9.0f;
    x[d - 5] *= 6.0f;
}

Test(kv_rq, rotation_roundtrip_and_isometry)
{
    const OcKvRqRotKind kinds[3] = { OC_KVRQ_ROT_HADAMARD, OC_KVRQ_ROT_ISO,
                                     OC_KVRQ_ROT_NONE };
    const size_t ds[3] = { 64, 128, 256 };
    rs(7);
    for (int k = 0; k < 3; k++) {
        for (int di = 0; di < 3; di++) {
            const size_t d = ds[di];
            OcKvRqRot r;
            cr_assert_eq(oc_kvrq_rot_init(&r, d, kinds[k], 11), OC_OK);
            float x[256], y[256], z[256];
            for (size_t i = 0; i < d; i++) x[i] = rg();
            oc_kvrq_rotate(&r, x, y);
            oc_kvrq_unrotate(&r, y, z);
            double nx = 0, ny = 0, err = 0;
            for (size_t i = 0; i < d; i++) {
                nx += (double)x[i] * x[i];
                ny += (double)y[i] * y[i];
                err += fabs((double)z[i] - x[i]);
            }
            cr_expect(fabs(sqrt(nx) - sqrt(ny)) < 1e-4 * sqrt(nx),
                      "kind %d d %zu: norm not preserved", k, d);
            cr_expect(err / d < 1e-5, "kind %d d %zu: roundtrip err %g", k, d,
                      err / d);
            /* In place must match out of place. */
            memcpy(z, x, d * sizeof(float));
            oc_kvrq_rotate(&r, z, z);
            for (size_t i = 0; i < d; i++) cr_assert_float_eq(z[i], y[i], 1e-6);
            oc_kvrq_rot_free(&r);
        }
    }
}

Test(kv_rq, hadamard_whitens_outliers)
{
    OcKvRqRot r;
    cr_assert_eq(oc_kvrq_rot_init(&r, 128, OC_KVRQ_ROT_HADAMARD, 3), OC_OK);
    float x[128] = {0}, y[128];
    x[5] = 1.0f;   /* one-hot: the worst case for per-coordinate codes */
    oc_kvrq_rotate(&r, x, y);
    for (int i = 0; i < 128; i++)
        cr_assert_float_eq(fabsf(y[i]), 1.0f / sqrtf(128.0f), 1e-6);
    oc_kvrq_rot_free(&r);
}

static double codec_rel_err(unsigned bits, OcKvRqRotKind rk, int outliers)
{
    const size_t d = 128;
    OcKvRqCodec c;
    OcKvRqRot r;
    cr_assert_eq(oc_kvrq_codec_init(&c, d, bits), OC_OK);
    cr_assert_eq(oc_kvrq_rot_init(&r, d, rk, 5), OC_OK);
    cr_assert_eq(c.block_bytes, 2 + d * bits / 8);
    uint8_t blk[2 + 64];
    float x[128], xr[128], yr[128], y[128];
    double tot = 0;
    rs(99 + bits);
    for (int n = 0; n < 500; n++) {
        if (outliers) fill_outlier(x, d);
        else for (size_t i = 0; i < d; i++) x[i] = rg() * 3.0f;
        oc_kvrq_rotate(&r, x, xr);
        oc_kvrq_encode(&c, xr, blk);
        oc_kvrq_decode(&c, blk, yr);
        oc_kvrq_unrotate(&r, yr, y);
        double e = 0, nx = 0, ny = 0;
        for (size_t i = 0; i < d; i++) {
            e += ((double)y[i] - x[i]) * ((double)y[i] - x[i]);
            nx += (double)x[i] * x[i];
            ny += (double)y[i] * y[i];
        }
        /* Length-preserving scale (f16-rounded). */
        cr_assert(fabs(sqrt(ny) - sqrt(nx)) < 2e-3 * sqrt(nx));
        tot += sqrt(e / nx);
    }
    oc_kvrq_rot_free(&r);
    return tot / 500;
}

Test(kv_rq, codec_roundtrip_error_bounds)
{
    /* Lloyd-Max on N(0,1): relative RMS error 0.34 / 0.19 / 0.10 for 2/3/4
     * bits; the length-preserving scale adds a little on top. */
    const double bound[5] = { 0, 0, 0.40, 0.22, 0.12 };
    for (unsigned b = 2; b <= 4; b++) {
        double e = codec_rel_err(b, OC_KVRQ_ROT_HADAMARD, 0);
        cr_expect(e < bound[b], "bits %u gaussian: rel err %.4f", b, e);
        double eo = codec_rel_err(b, OC_KVRQ_ROT_HADAMARD, 1);
        cr_expect(eo < bound[b], "bits %u outliers (hadamard): rel err %.4f",
                  b, eo);
        double ei = codec_rel_err(b, OC_KVRQ_ROT_ISO, 1);
        /* Iso only mixes 4 coordinates, so outlier channels stay heavy. */
        cr_expect(eo < ei, "bits %u: hadamard %.4f not better than iso %.4f",
                  b, eo, ei);
    }
}

Test(kv_rq, codec_zero_vector)
{
    OcKvRqCodec c;
    cr_assert_eq(oc_kvrq_codec_init(&c, 128, 3), OC_OK);
    float x[128] = {0}, y[128];
    uint8_t blk[64];
    oc_kvrq_encode(&c, x, blk);
    oc_kvrq_decode(&c, blk, y);
    for (int i = 0; i < 128; i++) cr_assert_eq(y[i], 0.0f);
}

Test(kv_rq, codec_rejects_bad_args)
{
    OcKvRqCodec c;
    cr_assert_neq(oc_kvrq_codec_init(&c, 128, 1), OC_OK);
    cr_assert_neq(oc_kvrq_codec_init(&c, 128, 5), OC_OK);
    cr_assert_neq(oc_kvrq_codec_init(&c, 96, 3), OC_OK);
    OcKvRqRot r;
    cr_assert_neq(oc_kvrq_rot_init(&r, 96, OC_KVRQ_ROT_HADAMARD, 1), OC_OK);
}

/* Fused score/accum/decode_rows kernels vs decode + scalar reference, for
 * every bit width, head-dim, and G (incl. G not a multiple of 4). */
Test(kv_rq, fused_kernels_match_scalar_reference)
{
    const size_t ds[3] = { 64, 128, 256 };
    const size_t Gs[4] = { 1, 3, 4, 6 };
    rs(1234);
    for (int di = 0; di < 3; di++) {
        const size_t d = ds[di];
        for (unsigned bits = 2; bits <= 4; bits++) {
            OcKvRqCodec c;
            cr_assert_eq(oc_kvrq_codec_init(&c, d, bits), OC_OK);
            const size_t n = 77;
            uint8_t *blocks = malloc(n * c.block_bytes);
            float *rows = malloc(n * d * sizeof(float));
            float *ref_rows = malloc(n * d * sizeof(float));
            float x[256];
            for (size_t t = 0; t < n; t++) {
                for (size_t i = 0; i < d; i++) x[i] = rg() * 2.0f;
                oc_kvrq_encode(&c, x, blocks + t * c.block_bytes);
                oc_kvrq_decode(&c, blocks + t * c.block_bytes, ref_rows + t * d);
            }
            oc_kvrq_decode_rows(&c, blocks, n, rows);
            for (size_t i = 0; i < n * d; i++)
                cr_assert_float_eq(rows[i], ref_rows[i], 1e-6,
                                   "decode_rows d %zu bits %u", d, bits);
            for (int gi = 0; gi < 4; gi++) {
                const size_t G = Gs[gi];
                float *q = malloc(G * d * sizeof(float));
                float *sc = malloc(G * n * sizeof(float));
                float *w = malloc(G * n * sizeof(float));
                float *acc = calloc(G * d, sizeof(float));
                float *acc_ref = calloc(G * d, sizeof(float));
                for (size_t i = 0; i < G * d; i++) q[i] = rg();
                for (size_t i = 0; i < G * n; i++) w[i] = (float)ru();
                for (size_t i = 0; i < G * d; i++) acc[i] = acc_ref[i] = rg();
                oc_kvrq_score(&c, blocks, n, q, G, sc, n);
                oc_kvrq_accum(&c, blocks, n, w, n, G, acc);
                for (size_t g = 0; g < G; g++) {
                    for (size_t t = 0; t < n; t++) {
                        double r = 0;
                        for (size_t i = 0; i < d; i++)
                            r += (double)q[g * d + i] * ref_rows[t * d + i];
                        cr_assert(fabs(sc[g * n + t] - r) <= 1e-4 * (1 + fabs(r)),
                                  "score d %zu bits %u G %zu: %f vs %f", d, bits,
                                  G, sc[g * n + t], r);
                        for (size_t i = 0; i < d; i++)
                            acc_ref[g * d + i] += w[g * n + t] * ref_rows[t * d + i];
                    }
                }
                for (size_t i = 0; i < G * d; i++)
                    cr_assert(fabs(acc[i] - acc_ref[i]) <= 1e-4 * (1 + fabs(acc_ref[i])),
                              "accum d %zu bits %u G %zu", d, bits, G);
                free(q); free(sc); free(w); free(acc); free(acc_ref);
            }
            free(blocks); free(rows); free(ref_rows);
        }
    }
}

Test(kv_rq, centroids_sit_on_int8_grid)
{
    for (unsigned bits = 2; bits <= 4; bits++) {
        OcKvRqCodec c;
        cr_assert_eq(oc_kvrq_codec_init(&c, 128, bits), OC_OK);
        for (unsigned k = 0; k < c.n_levels; k++) {
            cr_assert_eq(c.centroids[k], (float)c.ci[k] * c.cscale);
            cr_assert(c.ci[k] >= -63 && c.ci[k] <= 63);
            cr_assert_eq((int)c.ciu[k], (int)c.ci[k] + 64);
            if (k > 0) cr_assert(c.ci[k] > c.ci[k - 1], "levels must stay distinct");
        }
    }
}

/* Integer score kernel: bit-exact against an integer reference built from
 * decoded codes, and close to the f32 score (only q is rounded). */
Test(kv_rq, int8_score_matches_integer_reference)
{
    const size_t ds[3] = { 64, 128, 256 };
    const size_t Gs[4] = { 1, 3, 4, 6 };
    rs(4321);
    for (int di = 0; di < 3; di++) {
        const size_t d = ds[di];
        for (unsigned bits = 2; bits <= 4; bits++) {
            OcKvRqCodec c;
            cr_assert_eq(oc_kvrq_codec_init(&c, d, bits), OC_OK);
            const size_t n = 70;
            uint8_t *blocks = malloc(n * c.block_bytes);
            int *ci = malloc(n * d * sizeof(int));
            float *rows = malloc(n * d * sizeof(float));
            float x[256];
            uint8_t one[2 + 128];
            for (size_t t = 0; t < n; t++) {
                for (size_t i = 0; i < d; i++) x[i] = rg() * (t % 5 + 1);
                if (t == 9) fill_outlier(x, d);
                uint8_t *b = blocks + t * c.block_bytes;
                oc_kvrq_encode(&c, x, b);
                oc_kvrq_decode(&c, b, rows + t * d);
                memcpy(one, b, c.block_bytes);
                one[0] = 0x00; one[1] = 0x3c;           /* f16 1.0 */
                oc_kvrq_decode(&c, one, x);
                for (size_t i = 0; i < d; i++)
                    ci[t * d + i] = (int)lrintf(x[i] / c.cscale);
            }
            for (int gi = 0; gi < 4; gi++) {
                const size_t G = Gs[gi];
                float q[6 * 256], qs[6], sc[6 * 70], ref[6 * 70];
                int8_t q8[6 * 256];
                int32_t qsum[6];
                for (size_t i = 0; i < G * d; i++) q[i] = rg() * 0.3f;
                q[1] = 4.0f;                                 /* outlier */
                oc_kvrq_prep_q(q, G, d, q8, qs, qsum);
                oc_kvrq_score_i8(&c, blocks, n, q8, qs, qsum, G, sc, n);
                oc_kvrq_score(&c, blocks, n, q, G, ref, n);
                for (size_t g = 0; g < G; g++) {
                    int32_t s8 = 0;
                    for (size_t i = 0; i < d; i++) s8 += q8[g * d + i];
                    cr_assert_eq(s8, qsum[g]);
                    double qn = 0;
                    for (size_t i = 0; i < d; i++) qn += (double)q[g * d + i] * q[g * d + i];
                    for (size_t t = 0; t < n; t++) {
                        int32_t I = 0;
                        for (size_t i = 0; i < d; i++)
                            I += (ci[t * d + i] + 64) * q8[g * d + i];
                        I -= 64 * qsum[g];
                        uint16_t h;
                        memcpy(&h, blocks + t * c.block_bytes, 2);
                        const float s = oc_f16_to_f32_bits(h);
                        const float want = ((float)I * (qs[g] * c.cscale)) * s;
                        cr_assert_eq(sc[g * n + t], want,
                                     "d %zu bits %u G %zu g %zu t %zu: %.9g vs %.9g",
                                     d, bits, G, g, t, sc[g * n + t], want);
                        double kn = 0;
                        for (size_t i = 0; i < d; i++)
                            kn += (double)rows[t * d + i] * rows[t * d + i];
                        cr_assert(fabs(sc[g * n + t] - ref[g * n + t]) <=
                                  0.01 * sqrt(qn * kn) + 1e-6,
                                  "int8 score drifts from f32: %f vs %f",
                                  sc[g * n + t], ref[g * n + t]);
                    }
                }
            }
            free(blocks); free(ci); free(rows);
        }
    }
}

Test(kv_rq, q8_and_f32_kernels_match_reference)
{
    const size_t d = 128, n = 45, cs = 3 * d, sst = 5;
    rs(77);
    int8_t *codes = malloc(n * cs);
    float *sc = malloc(n * sst * sizeof(float));
    float *f = malloc(n * cs * sizeof(float));
    for (size_t t = 0; t < n; t++) {
        for (size_t i = 0; i < d; i++) {
            codes[t * cs + i] = (int8_t)((int)(ru() * 254) - 127);
            f[t * cs + i] = rg();
        }
        sc[t * sst] = (float)ru() * 0.1f;
    }
    for (size_t G = 1; G <= 5; G++) {
        float q[5 * 128], s1[5 * 45], s2[5 * 45], w[5 * 45];
        float a1[5 * 128] = {0}, a2[5 * 128] = {0}, r1[5 * 128] = {0},
              r2[5 * 128] = {0};
        for (size_t i = 0; i < G * d; i++) q[i] = rg();
        for (size_t i = 0; i < G * n; i++) w[i] = (float)ru();
        oc_kvq8_score(codes, cs, sc, sst, d, n, q, G, s1, n);
        oc_kvf32_score(f, cs, d, n, q, G, s2, n);
        oc_kvq8_accum(codes, cs, sc, sst, d, n, w, n, G, a1);
        oc_kvf32_accum(f, cs, d, n, w, n, G, a2);
        for (size_t g = 0; g < G; g++)
            for (size_t t = 0; t < n; t++) {
                double e1 = 0, e2 = 0;
                for (size_t i = 0; i < d; i++) {
                    e1 += (double)q[g * d + i] * codes[t * cs + i];
                    e2 += (double)q[g * d + i] * f[t * cs + i];
                    r1[g * d + i] += w[g * n + t] * sc[t * sst] * codes[t * cs + i];
                    r2[g * d + i] += w[g * n + t] * f[t * cs + i];
                }
                e1 *= sc[t * sst];
                cr_assert(fabs(s1[g * n + t] - e1) < 1e-3 * (1 + fabs(e1)));
                cr_assert(fabs(s2[g * n + t] - e2) < 1e-4 * (1 + fabs(e2)));
            }
        for (size_t i = 0; i < G * d; i++) {
            cr_assert(fabs(a1[i] - r1[i]) < 1e-3 * (1 + fabs(r1[i])));
            cr_assert(fabs(a2[i] - r2[i]) < 1e-4 * (1 + fabs(r2[i])));
        }
    }
    free(codes); free(sc); free(f);
}

static long statm_resident_pages(void)
{
    FILE *fp = fopen("/proc/self/statm", "r");
    if (fp == NULL) return -1;
    long size = 0, res = 0;
    if (fscanf(fp, "%ld %ld", &size, &res) != 2) res = -1;
    fclose(fp);
    return res;
}

Test(kv_rq, cache_is_lazily_committed)
{
    /* K2 geometry at 262K: ~13 GB of virtual RQ blocks at K4/V4. Creating it
     * must not commit memory; filling a few positions commits only those. */
    OcKvRqParams p = { .k_bits = 4, .v_bits = 4, .n_sink = 4, .window = 64,
                       .rot = OC_KVRQ_ROT_HADAMARD, .seed = 1 };
    OcKvRqCache c;
    long before = statm_resident_pages();
    cr_assert_eq(oc_kvrq_cache_init(&c, 48, 8, 128, 262144, &p), OC_OK);
    long after = statm_resident_pages();
    cr_assert_eq(oc_kvrq_bytes_per_token(&c), 48u * 8u * (66u + 66u));
    if (before >= 0 && after >= 0)
        cr_expect(after - before < 4096, "init committed %ld pages",
                  after - before);
    float k[8 * 128], v[8 * 128], scr[256];
    for (int i = 0; i < 8 * 128; i++) { k[i] = (float)i * 0.01f; v[i] = -k[i]; }
    for (int64_t pos = 0; pos < 100; pos++)
        for (size_t l = 0; l < 48; l++) oc_kvrq_store(&c, l, pos, k, v, scr);
    long filled = statm_resident_pages();
    if (after >= 0 && filled >= 0)
        cr_expect(filled - after < 40000, "100 positions committed %ld pages",
                  filled - after);
    oc_kvrq_cache_free(&c);
}

Test(kv_rq, cache_slots_tags_and_rewind)
{
    OcKvRqParams p = { .k_bits = 3, .v_bits = 2, .n_sink = 4, .window = 8,
                       .rot = OC_KVRQ_ROT_HADAMARD, .seed = 1 };
    OcKvRqCache c;
    cr_assert_eq(oc_kvrq_cache_init(&c, 2, 2, 64, 1000, &p), OC_OK);
    float k[2 * 64], v[2 * 64], scr[128];
    rs(3);
    for (int64_t pos = 0; pos < 40; pos++) {
        for (int i = 0; i < 128; i++) { k[i] = rg(); v[i] = rg(); }
        oc_kvrq_store(&c, 1, pos, k, v, scr);
        if (pos == 39) {
            /* The exact slot holds the rotated vector at int8 precision. */
            float kr[64];
            oc_kvrq_rotate(&c.rot, k + 64, kr);
            const int64_t s = oc_kvrq_slot(&c, 1, 39);
            cr_assert(s >= 0);
            const int8_t *xq = oc_kvrq_xq(&c, 1, 0, 1) + s * 64;
            const float xs = oc_kvrq_xs(&c, 1, 0, 1)[s];
            for (int i = 0; i < 64; i++)
                cr_assert(fabsf(xq[i] * xs - kr[i]) <= xs * 0.51f);
        }
    }
    for (int64_t t = 0; t < 4; t++) cr_assert_eq(oc_kvrq_slot(&c, 1, t), t);
    for (int64_t t = 4; t < 32; t++) cr_assert_eq(oc_kvrq_slot(&c, 1, t), -1);
    for (int64_t t = 32; t < 40; t++)
        cr_assert_eq(oc_kvrq_slot(&c, 1, t), 4 + t % 8);
    cr_assert_eq(oc_kvrq_slot(&c, 0, 35), -1, "layer 0 never written");
    oc_kvrq_cache_rewind(&c, 36);
    for (int64_t t = 36; t < 40; t++) cr_assert_eq(oc_kvrq_slot(&c, 1, t), -1);
    cr_assert_eq(oc_kvrq_slot(&c, 1, 35), 4 + 35 % 8);
    cr_assert_eq(oc_kvrq_slot(&c, 1, 2), 2);
    oc_kvrq_cache_clear(&c);
    cr_assert_eq(oc_kvrq_slot(&c, 1, 2), -1);
    oc_kvrq_cache_free(&c);
}

/* Keys with a large shared (per-head) component: the centered cache codes
 * only the residual, so the error scales with the residual, not |k|. */
Test(kv_rq, centering_removes_shared_component)
{
    const size_t d = 128, n = 300;
    OcKvRqParams p = { .k_bits = 3, .v_bits = 2, .n_sink = 4, .window = 64,
                       .rot = OC_KVRQ_ROT_HADAMARD, .seed = 1 };
    OcKvRqCache c;
    cr_assert_eq(oc_kvrq_cache_init(&c, 1, 1, d, n, &p), OC_OK);
    float shared[128], k[300][128], v[128], scr[256];
    rs(55);
    for (size_t i = 0; i < d; i++) shared[i] = rg() * 8.0f;
    for (size_t t = 0; t < n; t++) {
        for (size_t i = 0; i < d; i++) { k[t][i] = shared[i] + rg(); v[i] = rg(); }
        oc_kvrq_store(&c, 0, (int64_t)t, k[t], v, scr);
        if (t + 1 < 4 + 64) cr_assert_eq(c.mu_valid[0], 0);
    }
    cr_assert_eq(c.mu_valid[0], 1);
    double err = 0, res = 0;
    float kr[128], dec[128];
    for (size_t t = 4; t < n - 64; t++) {
        cr_assert_lt(oc_kvrq_slot(&c, 0, (int64_t)t), 0);
        oc_kvrq_rotate(&c.rot, k[t], kr);
        oc_kvrq_decode_pos(&c, 0, 0, 0, (int64_t)t, dec);
        for (size_t i = 0; i < d; i++) {
            err += ((double)dec[i] - kr[i]) * ((double)dec[i] - kr[i]);
            res += 1.0;   /* residual variance per coordinate */
        }
    }
    /* 3-bit Lloyd-Max on the unit-variance residual: rel err ~0.19;
     * without centering it would be ~0.19 * |k| / |residual| ~ 1.5. */
    cr_expect(sqrt(err / res) < 0.3, "centered rel err %.3f", sqrt(err / res));
    oc_kvrq_cache_rewind(&c, 10);
    cr_assert_eq(c.mu_valid[0], 0);
    oc_kvrq_cache_free(&c);
}
