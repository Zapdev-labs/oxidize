/* rotorquant.c — RotorQuant KV-cache compression (PlanarQuant / IsoQuant). */
#include "oxidize/rotorquant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


/* splitmix64. The exact stream need not match PyTorch's — the only
 * requirement is that the same seed yields the same rotations everywhere, so
 * an encoder and a decoder built independently agree. */
static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Uniform in [0, 1). 53 bits of mantissa, so no value repeats spuriously. */
static double rng_uniform(uint64_t *state)
{
    return (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

/* Box-Muller. Only the first variate is kept: generating quaternions four
 * independent draws at a time keeps the group->parameter mapping trivial. */
static double rng_normal(uint64_t *state)
{
    double u1 = rng_uniform(state);
    double u2 = rng_uniform(state);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}


/* Coordinate density after a random rotation of a unit vector in R^d. */
static double coord_pdf(double x, double sigma)
{
    double t = x / sigma;
    return exp(-0.5 * t * t) / (sigma * 2.50662827463100050242); /* sqrt(2pi) */
}

/* Composite Simpson over [a,b] of pdf(x) and x*pdf(x) in one sweep. scipy's */
#define LM_PANELS 512 /* must be even */

static void simpson_moments(double a, double b, double sigma,
                            double *out_mass, double *out_first)
{
    double h = (b - a) / (double)LM_PANELS;
    double mass = 0.0, first = 0.0;

    for (size_t i = 0; i <= LM_PANELS; i++) {
        double x = a + h * (double)i;
        double w = (i == 0 || i == LM_PANELS) ? 1.0 : ((i & 1u) ? 4.0 : 2.0);
        double f = coord_pdf(x, sigma);
        mass  += w * f;
        first += w * x * f;
    }
    *out_mass  = mass * h / 3.0;
    *out_first = first * h / 3.0;
}

OcError oc_rotorquant_lloyd_max(size_t d, unsigned bits, float *centroids_out)
{
    if (!centroids_out || d == 0) return OC_ERR_INVALID_ARG;
    if (bits < OC_RQ_MIN_BITS || bits > OC_RQ_MAX_BITS) return OC_ERR_INVALID_ARG;

    const unsigned n_levels = 1u << bits;
    const double   sigma    = 1.0 / sqrt((double)d);
    const double   lo       = -3.5 * sigma;
    const double   hi       =  3.5 * sigma;

    double centroids[OC_RQ_MAX_LEVELS];
    for (unsigned i = 0; i < n_levels; i++) {
        centroids[i] = lo + (hi - lo) * ((double)i + 0.5) / (double)n_levels;
    }

    /* Fixed-point iteration: boundaries are midpoints between neighbouring
     * centroids, centroids are the conditional means of their partitions. */
    for (unsigned iter = 0; iter < 200; iter++) {
        double edges[OC_RQ_MAX_LEVELS + 1];
        edges[0]        = lo * 3.0;  /* effectively -inf at 10.5 sigma */
        edges[n_levels] = hi * 3.0;
        for (unsigned i = 0; i + 1 < n_levels; i++) {
            edges[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
        }

        double max_shift = 0.0;
        for (unsigned i = 0; i < n_levels; i++) {
            double mass, first;
            simpson_moments(edges[i], edges[i + 1], sigma, &mass, &first);
            /* An empty partition keeps its centroid rather than collapsing. */
            if (mass > 1e-15) {
                double c = first / mass;
                double shift = fabs(c - centroids[i]);
                if (shift > max_shift) max_shift = shift;
                centroids[i] = c;
            }
        }
        if (max_shift < 1e-10) break;
    }

    for (unsigned i = 0; i < n_levels; i++) centroids_out[i] = (float)centroids[i];
    return OC_OK;
}


/* Hamilton product, [w, x, y, z]. */
static void quat_mul(const float *a, const float *b, float *out)
{
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

static void quat_conj(const float *q, float *out)
{
    out[0] =  q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

void oc_rotorquant_rotate(const OcRotorQuant *rq, float *v)
{
    if (!rq || !v) return;

    if (rq->variant == OC_RQ_PLANAR) {
        for (size_t g = 0; g < rq->n_groups; g++) {
            float c = rq->rot[2 * g];
            float s = rq->rot[2 * g + 1];
            float v0 = v[2 * g], v1 = v[2 * g + 1];
            v[2 * g]     = c * v0 - s * v1;
            v[2 * g + 1] = s * v0 + c * v1;
        }
        return;
    }

    /* T(v) = q_L * v * conj(q_R) — full SO(4), 6 DOF per block. */
    for (size_t g = 0; g < rq->n_groups; g++) {
        const float *qL = &rq->rot[8 * g];
        const float *qR = &rq->rot[8 * g + 4];
        float tmp[4], qRc[4];
        quat_mul(qL, &v[4 * g], tmp);
        quat_conj(qR, qRc);
        quat_mul(tmp, qRc, &v[4 * g]);
    }
}

void oc_rotorquant_unrotate(const OcRotorQuant *rq, float *v)
{
    if (!rq || !v) return;

    if (rq->variant == OC_RQ_PLANAR) {
        /* Transpose of the Givens block: negate sin. */
        for (size_t g = 0; g < rq->n_groups; g++) {
            float c = rq->rot[2 * g];
            float s = rq->rot[2 * g + 1];
            float v0 = v[2 * g], v1 = v[2 * g + 1];
            v[2 * g]     =  c * v0 + s * v1;
            v[2 * g + 1] = -s * v0 + c * v1;
        }
        return;
    }

    /* T^-1(v) = conj(q_L) * v * q_R. */
    for (size_t g = 0; g < rq->n_groups; g++) {
        const float *qL = &rq->rot[8 * g];
        const float *qR = &rq->rot[8 * g + 4];
        float tmp[4], qLc[4];
        quat_conj(qL, qLc);
        quat_mul(qLc, &v[4 * g], tmp);
        quat_mul(tmp, qR, &v[4 * g]);
    }
}


/* Little-endian bit stream: index i lives at bit offset i*bits, LSB first.
 * A 3-bit index costs exactly 3 bits; indices straddle byte boundaries. */
static void bits_put(uint8_t *buf, size_t bit_off, unsigned bits, unsigned value)
{
    for (unsigned b = 0; b < bits; b++) {
        size_t p = bit_off + b;
        if (value & (1u << b)) buf[p >> 3] |= (uint8_t)(1u << (p & 7u));
    }
}

static unsigned bits_get(const uint8_t *buf, size_t bit_off, unsigned bits)
{
    unsigned value = 0;
    for (unsigned b = 0; b < bits; b++) {
        size_t p = bit_off + b;
        if (buf[p >> 3] & (uint8_t)(1u << (p & 7u))) value |= (1u << b);
    }
    return value;
}


OcError oc_rotorquant_init(OcRotorQuant *rq, OcRotorQuantVariant variant,
                           size_t d, unsigned bits, uint64_t seed)
{
    if (!rq || d == 0) return OC_ERR_INVALID_ARG;
    if (bits < OC_RQ_MIN_BITS || bits > OC_RQ_MAX_BITS) return OC_ERR_INVALID_ARG;
    if (variant != OC_RQ_PLANAR && variant != OC_RQ_ISO) return OC_ERR_INVALID_ARG;

    memset(rq, 0, sizeof(*rq));
    rq->variant  = variant;
    rq->d        = d;
    rq->bits     = bits;
    rq->n_levels = 1u << bits;
    rq->block    = (variant == OC_RQ_PLANAR) ? 2u : 4u;
    rq->n_groups = (d + rq->block - 1) / rq->block;
    rq->d_padded = rq->n_groups * rq->block;
    rq->seed     = seed;
    rq->bytes_per_vector =
        sizeof(float) + (rq->d_padded * bits + 7u) / 8u;

    OcError err = oc_rotorquant_lloyd_max(d, bits, rq->centroids);
    if (err != OC_OK) return err;

    size_t per_group = (variant == OC_RQ_PLANAR) ? 2u : 8u;
    rq->rot = malloc(rq->n_groups * per_group * sizeof(float));
    if (!rq->rot) return OC_ERR_OOM;

    uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
    if (variant == OC_RQ_PLANAR) {
        for (size_t g = 0; g < rq->n_groups; g++) {
            double theta = rng_uniform(&state) * 2.0 * 3.14159265358979323846;
            rq->rot[2 * g]     = (float)cos(theta);
            rq->rot[2 * g + 1] = (float)sin(theta);
        }
    } else {
        for (size_t g = 0; g < rq->n_groups; g++) {
            for (unsigned q = 0; q < 2; q++) {
                float *dst = &rq->rot[8 * g + 4 * q];
                double n2 = 0.0;
                double tmp[4];
                /* Normalized Gaussian 4-vector = uniform on the unit
                 * quaternion sphere, i.e. a uniform SO(3) factor. */
                do {
                    n2 = 0.0;
                    for (unsigned k = 0; k < 4; k++) {
                        tmp[k] = rng_normal(&state);
                        n2 += tmp[k] * tmp[k];
                    }
                } while (n2 < 1e-12);
                double inv = 1.0 / sqrt(n2);
                for (unsigned k = 0; k < 4; k++) dst[k] = (float)(tmp[k] * inv);
            }
        }
    }

    return OC_OK;
}

void oc_rotorquant_free(OcRotorQuant *rq)
{
    if (!rq) return;
    free(rq->rot);
    memset(rq, 0, sizeof(*rq));
}

size_t oc_rotorquant_bytes_per_vector(const OcRotorQuant *rq)
{
    return rq ? rq->bytes_per_vector : 0;
}

float oc_rotorquant_compression_ratio(const OcRotorQuant *rq,
                                      size_t src_elem_bytes)
{
    if (!rq || src_elem_bytes == 0 || rq->bytes_per_vector == 0) return 0.0f;
    return (float)(rq->d * src_elem_bytes) / (float)rq->bytes_per_vector;
}


/* Padded scratch lives on the stack for the common head dims and falls back
 * to malloc only for unusually large d, keeping the hot path allocation-free. */
#define RQ_STACK_DIM 512u

OcError oc_rotorquant_encode(const OcRotorQuant *rq, const float *x, uint8_t *out)
{
    if (!rq || !rq->rot || !x || !out) return OC_ERR_INVALID_ARG;

    float  stack_buf[RQ_STACK_DIM];
    float *v = stack_buf;
    if (rq->d_padded > RQ_STACK_DIM) {
        v = malloc(rq->d_padded * sizeof(float));
        if (!v) return OC_ERR_OOM;
    }

    /* Norm and direction are stored separately: the rotation only whitens the
     * direction, and the codebook is calibrated for unit vectors. */
    double sum = 0.0;
    for (size_t i = 0; i < rq->d; i++) sum += (double)x[i] * (double)x[i];
    float norm = (float)sqrt(sum);
    float inv  = (norm > 1e-8f) ? 1.0f / norm : 0.0f;

    for (size_t i = 0; i < rq->d; i++) v[i] = x[i] * inv;
    for (size_t i = rq->d; i < rq->d_padded; i++) v[i] = 0.0f;

    oc_rotorquant_rotate(rq, v);

    memcpy(out, &norm, sizeof(float));
    memset(out + sizeof(float), 0, rq->bytes_per_vector - sizeof(float));

    for (size_t i = 0; i < rq->d_padded; i++) {
        unsigned best = 0;
        float best_d = fabsf(v[i] - rq->centroids[0]);
        for (unsigned c = 1; c < rq->n_levels; c++) {
            float dd = fabsf(v[i] - rq->centroids[c]);
            if (dd < best_d) { best_d = dd; best = c; }
        }
        bits_put(out + sizeof(float), i * rq->bits, rq->bits, best);
    }

    if (v != stack_buf) free(v);
    return OC_OK;
}

OcError oc_rotorquant_decode(const OcRotorQuant *rq, const uint8_t *in,
                             float *x_out)
{
    if (!rq || !rq->rot || !in || !x_out) return OC_ERR_INVALID_ARG;

    float  stack_buf[RQ_STACK_DIM];
    float *v = stack_buf;
    if (rq->d_padded > RQ_STACK_DIM) {
        v = malloc(rq->d_padded * sizeof(float));
        if (!v) return OC_ERR_OOM;
    }

    float norm;
    memcpy(&norm, in, sizeof(float));

    for (size_t i = 0; i < rq->d_padded; i++) {
        unsigned idx = bits_get(in + sizeof(float), i * rq->bits, rq->bits);
        v[i] = rq->centroids[idx];
    }

    /* INVERSE of the encode rotation — never the forward one. */
    oc_rotorquant_unrotate(rq, v);

    for (size_t i = 0; i < rq->d; i++) x_out[i] = v[i] * norm;

    if (v != stack_buf) free(v);
    return OC_OK;
}

OcError oc_rotorquant_encode_many(const OcRotorQuant *rq, const float *x,
                                  size_t n_vectors, uint8_t *out)
{
    if (!rq || !x || !out) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < n_vectors; i++) {
        OcError err = oc_rotorquant_encode(rq, x + i * rq->d,
                                           out + i * rq->bytes_per_vector);
        if (err != OC_OK) return err;
    }
    return OC_OK;
}

OcError oc_rotorquant_decode_many(const OcRotorQuant *rq, const uint8_t *in,
                                  size_t n_vectors, float *x_out)
{
    if (!rq || !in || !x_out) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < n_vectors; i++) {
        OcError err = oc_rotorquant_decode(rq, in + i * rq->bytes_per_vector,
                                            x_out + i * rq->d);
        if (err != OC_OK) return err;
    }
    return OC_OK;
}
