/* test_rotorquant.c — RotorQuant (PlanarQuant / IsoQuant) KV compression. */
#include "framework.h"
#include "oxidize/rotorquant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TEST_D 128

/* Deterministic Gaussian source so error bounds are reproducible. */
static uint64_t g_state;

static void rng_seed(uint64_t s) { g_state = s; }

static double rng_u(void)
{
    g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((g_state >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
}

static float rng_gauss(void)
{
    double u1 = rng_u();
    double u2 = rng_u();
    if (u1 < 1e-300) u1 = 1e-300;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

/* Mean relative L2 error over `n` random vectors. */
static double roundtrip_rel_error(OcRotorQuantVariant variant, unsigned bits,
                                   size_t d, size_t n)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, variant, d, bits, OC_RQ_DEFAULT_SEED),
                 OC_OK);

    uint8_t *buf = malloc(rq.bytes_per_vector);
    float   *x   = malloc(d * sizeof(float));
    float   *y   = malloc(d * sizeof(float));
    cr_assert_not_null(buf);
    cr_assert_not_null(x);
    cr_assert_not_null(y);

    rng_seed(1234);
    double total = 0.0;
    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < d; i++) x[i] = rng_gauss();

        cr_assert_eq(oc_rotorquant_encode(&rq, x, buf), OC_OK);
        cr_assert_eq(oc_rotorquant_decode(&rq, buf, y), OC_OK);

        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < d; i++) {
            double e = (double)x[i] - (double)y[i];
            num += e * e;
            den += (double)x[i] * (double)x[i];
        }
        total += sqrt(num / den);
    }

    free(buf);
    free(x);
    free(y);
    oc_rotorquant_free(&rq);
    return total / (double)n;
}

/* ─── Init / config ──────────────────────────────────────────────────── */

Test(rotorquant, init_planar_geometry)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_PLANAR, TEST_D, 3, 42), OC_OK);
    cr_assert_eq(rq.block, 2);
    cr_assert_eq(rq.n_groups, TEST_D / 2);
    cr_assert_eq(rq.d_padded, TEST_D);
    cr_assert_eq(rq.n_levels, 8);
    /* 4-byte norm + 128 * 3 bits = 48 bytes. */
    cr_assert_eq(rq.bytes_per_vector, 52);
    oc_rotorquant_free(&rq);
    cr_assert_null(rq.rot);
}

Test(rotorquant, init_iso_geometry)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_ISO, TEST_D, 4, 42), OC_OK);
    cr_assert_eq(rq.block, 4);
    cr_assert_eq(rq.n_groups, TEST_D / 4);
    cr_assert_eq(rq.d_padded, TEST_D);
    cr_assert_eq(rq.bytes_per_vector, 68);
    oc_rotorquant_free(&rq);
}

Test(rotorquant, init_rejects_bad_args)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_PLANAR, 0, 3, 42),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_PLANAR, TEST_D, 1, 42),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_PLANAR, TEST_D, 5, 42),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_rotorquant_init(NULL, OC_RQ_PLANAR, TEST_D, 3, 42),
                 OC_ERR_INVALID_ARG);
}

Test(rotorquant, odd_dimension_pads)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_ISO, 30, 3, 42), OC_OK);
    cr_assert_eq(rq.n_groups, 8);
    cr_assert_eq(rq.d_padded, 32);
    /* 32 * 3 = 96 bits = 12 bytes, plus the norm. */
    cr_assert_eq(rq.bytes_per_vector, 16);

    float x[30], y[30];
    uint8_t buf[16];
    rng_seed(7);
    for (size_t i = 0; i < 30; i++) x[i] = rng_gauss();
    cr_assert_eq(oc_rotorquant_encode(&rq, x, buf), OC_OK);
    cr_assert_eq(oc_rotorquant_decode(&rq, buf, y), OC_OK);
    for (size_t i = 0; i < 30; i++) cr_assert(isfinite(y[i]));
    oc_rotorquant_free(&rq);
}

/* ─── Lloyd-Max solver ───────────────────────────────────────────────── */

Test(rotorquant, lloyd_max_is_sorted_and_symmetric)
{
    for (unsigned bits = OC_RQ_MIN_BITS; bits <= OC_RQ_MAX_BITS; bits++) {
        float c[OC_RQ_MAX_LEVELS];
        cr_assert_eq(oc_rotorquant_lloyd_max(TEST_D, bits, c), OC_OK);
        unsigned n = 1u << bits;
        for (unsigned i = 0; i + 1 < n; i++) {
            cr_assert(c[i] < c[i + 1], "centroids must be ascending");
        }
        /* The coordinate density is symmetric, so the codebook must be too. */
        for (unsigned i = 0; i < n; i++) {
            cr_assert_float_eq(c[i], -c[n - 1 - i], 1e-5f);
        }
    }
}

/* The known Lloyd-Max solution for a unit Gaussian, scaled by sigma = 1/sqrt(d).
 * If the solver's own integration drifts, this catches it. */
Test(rotorquant, lloyd_max_matches_reference_gaussian)
{
    const float unit_2bit[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
    const float unit_3bit[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f,
                                 0.2451f,  0.7560f,  1.3439f,  2.1519f};
    float sigma = 1.0f / sqrtf((float)TEST_D);
    float c[OC_RQ_MAX_LEVELS];

    cr_assert_eq(oc_rotorquant_lloyd_max(TEST_D, 2, c), OC_OK);
    for (unsigned i = 0; i < 4; i++)
        cr_assert_float_eq(c[i], unit_2bit[i] * sigma, 1e-4f);

    cr_assert_eq(oc_rotorquant_lloyd_max(TEST_D, 3, c), OC_OK);
    for (unsigned i = 0; i < 8; i++)
        cr_assert_float_eq(c[i], unit_3bit[i] * sigma, 1e-4f);
}

/* ─── Rotation correctness ───────────────────────────────────────────── */

Test(rotorquant, unrotate_inverts_rotate)
{
    OcRotorQuantVariant variants[2] = {OC_RQ_PLANAR, OC_RQ_ISO};
    for (int vi = 0; vi < 2; vi++) {
        OcRotorQuant rq;
        cr_assert_eq(oc_rotorquant_init(&rq, variants[vi], TEST_D, 3, 42), OC_OK);

        float v[TEST_D], orig[TEST_D];
        rng_seed(99);
        for (size_t i = 0; i < TEST_D; i++) { v[i] = rng_gauss(); orig[i] = v[i]; }

        oc_rotorquant_rotate(&rq, v);
        /* A rotation must not be the identity, else the test is vacuous. */
        double diff = 0.0;
        for (size_t i = 0; i < TEST_D; i++) diff += fabs(v[i] - orig[i]);
        cr_assert(diff > 1.0, "rotation must actually move the vector");

        oc_rotorquant_unrotate(&rq, v);
        for (size_t i = 0; i < TEST_D; i++)
            cr_assert_float_eq(v[i], orig[i], 1e-4f);

        oc_rotorquant_free(&rq);
    }
}

Test(rotorquant, rotation_preserves_norm)
{
    OcRotorQuantVariant variants[2] = {OC_RQ_PLANAR, OC_RQ_ISO};
    for (int vi = 0; vi < 2; vi++) {
        OcRotorQuant rq;
        cr_assert_eq(oc_rotorquant_init(&rq, variants[vi], TEST_D, 3, 42), OC_OK);

        float v[TEST_D];
        rng_seed(5);
        double before = 0.0;
        for (size_t i = 0; i < TEST_D; i++) {
            v[i] = rng_gauss();
            before += (double)v[i] * v[i];
        }
        oc_rotorquant_rotate(&rq, v);
        double after = 0.0;
        for (size_t i = 0; i < TEST_D; i++) after += (double)v[i] * v[i];
        cr_assert_float_eq(after, before, 1e-3);

        oc_rotorquant_free(&rq);
    }
}

/* Regression guard for the V-cache bug in the reference: decoding with the
 * FORWARD rotation instead of its inverse still yields a finite, unit-ish
 * vector, so only an explicit error comparison catches it. Here we simulate
 * the wrong decode and assert it is dramatically worse than the real one. */
Test(rotorquant, inverted_rotation_regression)
{
    OcRotorQuantVariant variants[2] = {OC_RQ_PLANAR, OC_RQ_ISO};
    for (int vi = 0; vi < 2; vi++) {
        OcRotorQuant rq;
        cr_assert_eq(oc_rotorquant_init(&rq, variants[vi], TEST_D, 4, 42), OC_OK);

        float x[TEST_D], good[TEST_D], bad[TEST_D];
        uint8_t buf[4 + TEST_D];
        cr_assert(rq.bytes_per_vector <= sizeof(buf));

        rng_seed(2024);
        for (size_t i = 0; i < TEST_D; i++) x[i] = rng_gauss();

        cr_assert_eq(oc_rotorquant_encode(&rq, x, buf), OC_OK);
        cr_assert_eq(oc_rotorquant_decode(&rq, buf, good), OC_OK);

        /* Wrong decode: centroids + FORWARD rotation (the 15,369-PPL bug). */
        float norm;
        memcpy(&norm, buf, sizeof(float));
        float v[TEST_D];
        for (size_t i = 0; i < TEST_D; i++) {
            unsigned idx = 0;
            for (unsigned b = 0; b < rq.bits; b++) {
                size_t p = i * rq.bits + b;
                if (buf[4 + (p >> 3)] & (1u << (p & 7u))) idx |= (1u << b);
            }
            v[i] = rq.centroids[idx];
        }
        oc_rotorquant_rotate(&rq, v);
        for (size_t i = 0; i < TEST_D; i++) bad[i] = v[i] * norm;

        double eg = 0.0, eb = 0.0, den = 0.0;
        for (size_t i = 0; i < TEST_D; i++) {
            eg  += ((double)x[i] - good[i]) * ((double)x[i] - good[i]);
            eb  += ((double)x[i] - bad[i])  * ((double)x[i] - bad[i]);
            den += (double)x[i] * (double)x[i];
        }
        eg = sqrt(eg / den);
        eb = sqrt(eb / den);

        cr_assert(eg < 0.15, "correct decode rel err %.4f too high", eg);
        /* Applying the rotation twice destroys the direction entirely: the
         * result is essentially uncorrelated with x (rel err ~sqrt(2)). */
        cr_assert(eb > 5.0 * eg,
                  "inverted rotation not detected: good %.4f bad %.4f", eg, eb);
        cr_assert(eb > 1.0, "inverted decode rel err %.4f suspiciously low", eb);

        oc_rotorquant_free(&rq);
    }
}

/* ─── Round-trip quality ─────────────────────────────────────────────── */

/* Bounds come from the Lloyd-Max Gaussian distortion: rel L2 error ~
 * sqrt(D_b) with D_b = 0.1175 / 0.03454 / 0.009497 for 2/3/4 bits, i.e.
 * ~0.343 / 0.186 / 0.097. Bounds sit just above the measured values. */
Test(rotorquant, planar_roundtrip_error_bounds)
{
    double e2 = roundtrip_rel_error(OC_RQ_PLANAR, 2, TEST_D, 64);
    double e3 = roundtrip_rel_error(OC_RQ_PLANAR, 3, TEST_D, 64);
    double e4 = roundtrip_rel_error(OC_RQ_PLANAR, 4, TEST_D, 64);
    cr_assert(e2 < 0.40, "planar 2-bit rel err %.4f", e2);
    cr_assert(e3 < 0.22, "planar 3-bit rel err %.4f", e3);
    cr_assert(e4 < 0.12, "planar 4-bit rel err %.4f", e4);
    cr_assert(e3 < e2 && e4 < e3, "more bits must reduce error");
}

Test(rotorquant, iso_roundtrip_error_bounds)
{
    double e2 = roundtrip_rel_error(OC_RQ_ISO, 2, TEST_D, 64);
    double e3 = roundtrip_rel_error(OC_RQ_ISO, 3, TEST_D, 64);
    double e4 = roundtrip_rel_error(OC_RQ_ISO, 4, TEST_D, 64);
    cr_assert(e2 < 0.40, "iso 2-bit rel err %.4f", e2);
    cr_assert(e3 < 0.22, "iso 3-bit rel err %.4f", e3);
    cr_assert(e4 < 0.12, "iso 4-bit rel err %.4f", e4);
    cr_assert(e3 < e2 && e4 < e3, "more bits must reduce error");
}

Test(rotorquant, norm_is_preserved_closely)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_ISO, TEST_D, 4, 42), OC_OK);

    float x[TEST_D], y[TEST_D];
    uint8_t buf[68];
    rng_seed(31337);
    for (size_t i = 0; i < TEST_D; i++) x[i] = rng_gauss() * 3.7f;

    cr_assert_eq(oc_rotorquant_encode(&rq, x, buf), OC_OK);
    cr_assert_eq(oc_rotorquant_decode(&rq, buf, y), OC_OK);

    double nx = 0.0, ny = 0.0;
    for (size_t i = 0; i < TEST_D; i++) {
        nx += (double)x[i] * x[i];
        ny += (double)y[i] * y[i];
    }
    /* The stored norm is exact; only the quantized direction shrinks slightly. */
    cr_assert(fabs(sqrt(ny) / sqrt(nx) - 1.0) < 0.05);
    oc_rotorquant_free(&rq);
}

Test(rotorquant, zero_vector_roundtrips)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_PLANAR, TEST_D, 3, 42), OC_OK);
    float x[TEST_D] = {0}, y[TEST_D];
    uint8_t buf[52];
    cr_assert_eq(oc_rotorquant_encode(&rq, x, buf), OC_OK);
    cr_assert_eq(oc_rotorquant_decode(&rq, buf, y), OC_OK);
    for (size_t i = 0; i < TEST_D; i++) cr_assert_float_eq(y[i], 0.0f, 1e-6f);
    oc_rotorquant_free(&rq);
}

/* ─── Determinism, batching, packing ─────────────────────────────────── */

Test(rotorquant, same_seed_decodes_other_instance)
{
    OcRotorQuant enc, dec;
    cr_assert_eq(oc_rotorquant_init(&enc, OC_RQ_ISO, TEST_D, 3, 7), OC_OK);
    cr_assert_eq(oc_rotorquant_init(&dec, OC_RQ_ISO, TEST_D, 3, 7), OC_OK);

    float x[TEST_D], y1[TEST_D], y2[TEST_D];
    uint8_t buf[52];
    rng_seed(11);
    for (size_t i = 0; i < TEST_D; i++) x[i] = rng_gauss();

    cr_assert_eq(oc_rotorquant_encode(&enc, x, buf), OC_OK);
    cr_assert_eq(oc_rotorquant_decode(&enc, buf, y1), OC_OK);
    cr_assert_eq(oc_rotorquant_decode(&dec, buf, y2), OC_OK);
    for (size_t i = 0; i < TEST_D; i++) cr_assert_float_eq(y1[i], y2[i], 0.0f);

    oc_rotorquant_free(&enc);
    oc_rotorquant_free(&dec);
}

Test(rotorquant, different_seed_gives_different_rotation)
{
    OcRotorQuant a, b;
    cr_assert_eq(oc_rotorquant_init(&a, OC_RQ_PLANAR, TEST_D, 3, 1), OC_OK);
    cr_assert_eq(oc_rotorquant_init(&b, OC_RQ_PLANAR, TEST_D, 3, 2), OC_OK);
    int differs = 0;
    for (size_t i = 0; i < 2 * a.n_groups; i++)
        if (a.rot[i] != b.rot[i]) differs = 1;
    cr_assert(differs);
    oc_rotorquant_free(&a);
    oc_rotorquant_free(&b);
}

Test(rotorquant, batch_matches_single)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_PLANAR, TEST_D, 3, 42), OC_OK);

    const size_t n = 5;
    float *x = malloc(n * TEST_D * sizeof(float));
    float *y = malloc(n * TEST_D * sizeof(float));
    uint8_t *buf = malloc(n * rq.bytes_per_vector);
    cr_assert_not_null(x);
    cr_assert_not_null(y);
    cr_assert_not_null(buf);

    rng_seed(64);
    for (size_t i = 0; i < n * TEST_D; i++) x[i] = rng_gauss();

    cr_assert_eq(oc_rotorquant_encode_many(&rq, x, n, buf), OC_OK);
    cr_assert_eq(oc_rotorquant_decode_many(&rq, buf, n, y), OC_OK);

    for (size_t k = 0; k < n; k++) {
        float single[TEST_D];
        uint8_t one[52];
        cr_assert_eq(oc_rotorquant_encode(&rq, x + k * TEST_D, one), OC_OK);
        cr_assert_arr_eq(one, buf + k * rq.bytes_per_vector, rq.bytes_per_vector);
        cr_assert_eq(oc_rotorquant_decode(&rq, one, single), OC_OK);
        for (size_t i = 0; i < TEST_D; i++)
            cr_assert_float_eq(single[i], y[k * TEST_D + i], 0.0f);
    }

    free(x);
    free(y);
    free(buf);
    oc_rotorquant_free(&rq);
}

/* Indices must cost exactly `bits` bits — the whole point of the format. */
Test(rotorquant, compression_ratio_is_measured)
{
    struct { unsigned bits; size_t expect_bytes; } cases[] = {
        {2, 36}, {3, 52}, {4, 68},
    };
    for (size_t k = 0; k < 3; k++) {
        OcRotorQuant rq;
        cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_PLANAR, TEST_D,
                                        cases[k].bits, 42), OC_OK);
        cr_assert_eq(oc_rotorquant_bytes_per_vector(&rq), cases[k].expect_bytes);

        /* Measured against an actual encoded batch, not just the field. */
        const size_t n = 16;
        float *x = malloc(n * TEST_D * sizeof(float));
        uint8_t *buf = malloc(n * rq.bytes_per_vector);
        cr_assert_not_null(x);
        cr_assert_not_null(buf);
        rng_seed(8);
        for (size_t i = 0; i < n * TEST_D; i++) x[i] = rng_gauss();
        cr_assert_eq(oc_rotorquant_encode_many(&rq, x, n, buf), OC_OK);

        double f32_bytes = (double)(n * TEST_D * 4);
        double got_bytes = (double)(n * rq.bytes_per_vector);
        double ratio_f32 = f32_bytes / got_bytes;
        double ratio_f16 = (f32_bytes / 2.0) / got_bytes;

        cr_assert_float_eq(oc_rotorquant_compression_ratio(&rq, 4),
                           (float)ratio_f32, 1e-3f);
        cr_assert_float_eq(oc_rotorquant_compression_ratio(&rq, 2),
                           (float)ratio_f16, 1e-3f);
        /* Documented header numbers: 14.22x / 9.85x / 7.53x vs f32. */
        cr_assert(ratio_f32 > 7.0, "ratio vs f32 %.2f too low", ratio_f32);
        cr_assert(ratio_f16 > 3.5, "ratio vs f16 %.2f too low", ratio_f16);

        free(x);
        free(buf);
        oc_rotorquant_free(&rq);
    }
}

Test(rotorquant, null_arguments_rejected)
{
    OcRotorQuant rq;
    cr_assert_eq(oc_rotorquant_init(&rq, OC_RQ_PLANAR, TEST_D, 3, 42), OC_OK);
    float x[TEST_D] = {0};
    uint8_t buf[52];
    cr_assert_eq(oc_rotorquant_encode(NULL, x, buf), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_rotorquant_encode(&rq, NULL, buf), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_rotorquant_encode(&rq, x, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_rotorquant_decode(&rq, NULL, x), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_rotorquant_decode(&rq, buf, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_rotorquant_bytes_per_vector(NULL), 0);
    cr_assert_float_eq(oc_rotorquant_compression_ratio(NULL, 4), 0.0f, 0.0f);
    cr_assert_float_eq(oc_rotorquant_compression_ratio(&rq, 0), 0.0f, 0.0f);
    oc_rotorquant_free(&rq);
    oc_rotorquant_free(NULL);
}
