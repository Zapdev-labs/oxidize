/*
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Packed format/codebooks adapted from exllamav3 tests/testlib/trellis.py,
058b8630fd081f3ff46fce6246955b157c3a2e3b.
*/

#include "oxidize/exl3.h"
#include "oxidize/flash_attention.h"
#include "oxidize/parallel.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* The LUT inputs are exact binary32 values. Round once to binary16 using
 * round-to-nearest, ties-to-even, independently of the ambient FP mode. */
static uint16_t half_rne(float f)
{
    uint32_t u; memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000, mant = u & 0x7fffff;
    int e = (int)((u >> 23) & 255) - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    if (e < -10) return (uint16_t)sign;
    unsigned shift = 13;
    if (e <= 0) { mant |= 0x800000; shift = (unsigned)(14 - e); e = 0; }
    uint32_t q = mant >> shift, rem = mant & ((1u << shift) - 1);
    uint32_t tie = 1u << (shift - 1);
    q += rem > tie || (rem == tie && (q & 1));
    return (uint16_t)(sign | (((uint32_t)e << 10) + q));
}
static float lut[3][65536];
static pthread_once_t lut_once[3] = {PTHREAD_ONCE_INIT, PTHREAD_ONCE_INIT, PTHREAD_ONCE_INIT};
static void init_lut(unsigned cb)
{
    for (uint32_t i = 0; i < 65536; ++i) {
        uint32_t x; float f;
        if (cb < 2) {
            x = cb == 0 ? i * 89226354u + 64248484u : i * 0xcbac1fedu;
            x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
            f = oc_f16_to_f32_bits((uint16_t)x) + oc_f16_to_f32_bits((uint16_t)(x >> 16));
        } else {
            x = i * 0x83dcd12du;
            unsigned sum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
            /* Double keeps the multiply-add exact before binary16 rounding. */
            f = (float)((1024.0 + sum) * (double)oc_f16_to_f32_bits(0x1eee)
                        + (double)oc_f16_to_f32_bits(0xc931));
        }
        lut[cb][i] = oc_f16_to_f32_bits(half_rne(f));
    }
}
static void init0(void) { init_lut(0); }
static void init1(void) { init_lut(1); }
static void init2(void) { init_lut(2); }
static const float *get_lut(unsigned cb)
{
    void (*const init[3])(void) = {init0, init1, init2};
    pthread_once(&lut_once[cb], init[cb]);
    return lut[cb];
}
float oc_exl3_codebook_value(unsigned cb, uint16_t idx)
{
    return cb < 3 ? get_lut(cb)[idx] : NAN;
}
OcError oc_exl3_validate(const OcExl3Matrix *m)
{
    if (!m || !m->packed || !m->suh || !m->svh || !m->input_dim || !m->output_dim ||
        m->input_dim % 128 || m->output_dim % 128) return OC_ERR_INVALID_ARG;
    unsigned b = m->bits_twice;
    if (m->codebook > 2 || b < 2 || b > 16 ||
        ((b & 1) && (m->codebook != 2 || b > 7))) return OC_ERR_QUANT;
    size_t k = m->input_dim / 16, n = m->output_dim / 16, bytes = 16 * b;
    size_t kts = m->packed_kt_stride ? m->packed_kt_stride : n;
    if (k > SIZE_MAX / (kts ? kts : 1) || k < 1 ||
        (k - 1) * kts > SIZE_MAX - n - m->packed_nt_base ||
        (k - 1) * kts + n + m->packed_nt_base > SIZE_MAX / bytes ||
        m->packed_size != ((k - 1) * kts + n + m->packed_nt_base) * bytes ||
        m->input_dim > SIZE_MAX / sizeof(float) || m->output_dim > SIZE_MAX / sizeof(float))
        return OC_ERR_FORMAT;
    /* Block permutations must be bijections of their 128-block index space. */
    const size_t nb_in = m->input_dim / 128, nb_out = m->output_dim / 128;
    for (size_t which = 0; which < 2; ++which) {
        const uint32_t *perm = which ? m->out_block_perm : m->in_block_perm;
        const size_t nb = which ? nb_out : nb_in;
        if (!perm) continue;
        unsigned char *seen = malloc(nb ? nb : 1);
        if (!seen) return OC_ERR_OOM;
        memset(seen, 0, nb ? nb : 1);
        for (size_t i = 0; i < nb; ++i) {
            if (perm[i] >= nb || seen[perm[i]]) { free(seen); return OC_ERR_FORMAT; }
            seen[perm[i]] = 1;
        }
        free(seen);
    }
    for (size_t i = 0; i < m->input_dim; ++i) if (!isfinite(m->suh[i])) return OC_ERR_FORMAT;
    for (size_t i = 0; i < m->output_dim; ++i)
        if (!isfinite(m->svh[i]) || (m->bias && !isfinite(m->bias[i]))) return OC_ERR_FORMAT;
    return OC_OK;
}
static uint32_t word(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static void decode(const OcExl3Matrix *m, size_t kt, size_t nt, const float *table, float *out)
{
    unsigned b = m->bits_twice, ring = 128 * b, end = 0;
    size_t kts = m->packed_kt_stride ? m->packed_kt_stride : m->output_dim / 16;
    const uint8_t *p = m->packed + (kt * kts + nt + m->packed_nt_base) * (16 * b);
    for (unsigned i = 0; i < 256; ++i) {
        end += b / 2 + ((b & 1) && (i & 1));
        unsigned start = end >= 16 ? end - 16 : end + ring - 16;
        unsigned wi = start / 32, off = start % 32;
        uint32_t v = word(p + 4 * wi) << off;
        if (off > 16) {
            unsigned next = wi + 1 == ring / 32 ? 0 : wi + 1;
            v |= word(p + 4 * next) >> (32 - off);
        }
        unsigned t = i / 8, j = i % 8;
        unsigned r = (t % 4) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
        unsigned c = t / 4 + ((j & 4) ? 8 : 0);
        out[r * 16 + c] = table[v >> 16];
    }
}
OcError oc_exl3_decode_tile(const OcExl3Matrix *m, size_t kt, size_t nt, float out[256])
{
    OcError e = oc_exl3_validate(m);
    if (e != OC_OK) return e;
    if (!out || kt >= m->input_dim / 16 || nt >= m->output_dim / 16) return OC_ERR_INVALID_ARG;
    decode(m, kt, nt, get_lut(m->codebook), out);
    return OC_OK;
}

OcError oc_exl3_dequant_f32(const OcExl3Matrix *m, float *out)
{
    if (!m || !out) return OC_ERR_INVALID_ARG;
    OcError e = oc_exl3_validate(m);
    if (e != OC_OK) return e;
    size_t n_kt = m->input_dim / 16, n_nt = m->output_dim / 16;
    for (size_t nt = 0; nt < n_nt; nt++)
        for (size_t kt = 0; kt < n_kt; kt++) {
            float tile[256];
            e = oc_exl3_decode_tile(m, kt, nt, tile);
            if (e != OC_OK) return e;
            for (unsigned r = 0; r < 16; r++)
                for (unsigned c = 0; c < 16; c++)
                    out[(nt * 16 + r) * m->input_dim + kt * 16 + c] =
                        tile[r * 16 + c];
        }
    /* Fold the su/sv Hadamard scales so `out` is the effective matrix —
     * the same map oc_exl3_matmul applies to its input. */
    for (size_t r = 0; r < m->output_dim; r++) {
        float sv = m->svh ? m->svh[r] : 1.0f;
        float *row = out + r * m->input_dim;
        for (size_t c = 0; c < m->input_dim; c++)
            row[c] *= sv * (m->suh ? m->suh[c] : 1.0f);
    }
    return OC_OK;
}
static void hadamard(float *x)
{
    for (unsigned s = 1; s < 128; s *= 2)
        for (unsigned i = 0; i < 128; i += 2 * s)
            for (unsigned j = 0; j < s; ++j) {
                float a = x[i+j], b = x[i+j+s]; x[i+j] = a+b; x[i+j+s] = a-b;
            }
    for (unsigned i = 0; i < 128; ++i) x[i] *= 0.08838834764831844055f;
}
/* Caller TLS is distinct from pool scratch (including worker zero), retained
 * across tokens, and released when the caller thread exits. */
typedef struct { float *x; size_t capacity; } Scratch;
static pthread_key_t scratch_key;
static int scratch_key_error;
static pthread_once_t scratch_once = PTHREAD_ONCE_INIT;
static void free_scratch(void *p) { Scratch *s = p; if (s) { free(s->x); free(s); } }
static void init_scratch(void) { scratch_key_error = pthread_key_create(&scratch_key, free_scratch); }
static float *scratch_get(size_t count)
{
    pthread_once(&scratch_once, init_scratch);
    if (scratch_key_error) return NULL;
    Scratch *s = pthread_getspecific(scratch_key);
    if (!s) {
        s = calloc(1, sizeof(*s));
        if (!s) return NULL;
        if (pthread_setspecific(scratch_key, s)) { free(s); return NULL; }
    }
    if (count > s->capacity) {
        float *p = realloc(s->x, count * sizeof(float));
        if (!p) return NULL;
        s->x = p; s->capacity = count;
    }
    return s->x;
}
#define ROWS 4
/* Logical→physical relabeling for permuted views (see OcExl3Matrix). */
static inline size_t kt_phys(const OcExl3Matrix *m, size_t kt)
{
    return m->in_block_perm ? (size_t)m->in_block_perm[kt / 8] * 8 + kt % 8 : kt;
}
static inline size_t nt_phys(const OcExl3Matrix *m, size_t block, size_t nt)
{
    return m->out_block_perm ? (size_t)m->out_block_perm[block] * 8 + nt
                             : block * 8 + nt;
}
static inline size_t col_phys(const OcExl3Matrix *m, size_t col)
{
    return m->out_block_perm
        ? (size_t)m->out_block_perm[col / 128] * 128 + col % 128 : col;
}

/* Output blocks [begin, end) of a batch whose input rows are ALREADY in the
 * su-hadamard domain (x, logical order). Called both from oc_parallel_for
 * jobs and from the expert runner, so it never touches the pool itself.
 * Per-output accumulation order (kt ascending, then k, then n) matches the
 * serial reference exactly. */
static void blocks_prepped(const OcExl3Matrix *m, const float *x, size_t batch,
                           float *out, size_t out_stride, size_t out_row0,
                           size_t begin, size_t end, const float *table,
                           size_t tid)
{
    float *acc = oc_parallel_scratch(tid, batch * 128 * sizeof(float));
    if (acc == NULL) {
        /* Scratch failure: keep the old stack-buffered chunking. */
        for (size_t block = begin; block < end; ++block) {
            float acc4[ROWS][128] = {{0}}, tile[256];
            for (size_t base = 0; base < batch; base += ROWS) {
                const size_t rows = batch - base < ROWS ? batch - base : ROWS;
                memset(acc4, 0, rows * 128 * sizeof(float));
                for (size_t kt = 0; kt < m->input_dim / 16; ++kt)
                    for (unsigned nt = 0; nt < 8; ++nt) {
                        decode(m, kt_phys(m, kt),
                               nt_phys(m, block, nt), table, tile);
                        for (size_t row = 0; row < rows; ++row)
                            for (unsigned k = 0; k < 16; ++k) {
                                float xx = x[(base+row) * m->input_dim + kt * 16 + k];
                                for (unsigned n = 0; n < 16; ++n)
                                    acc4[row][nt*16+n] += xx * tile[k*16+n];
                            }
                    }
                for (size_t row = 0; row < rows; ++row) {
                    hadamard(acc4[row]);
                    float *orow = out + (out_row0 + base + row) * out_stride;
                    for (unsigned n = 0; n < 128; ++n) {
                        size_t col = col_phys(m, block * 128 + n);
                        orow[block * 128 + n] =
                            acc4[row][n] * m->svh[col] + (m->bias ? m->bias[col] : 0);
                    }
                }
            }
        }
        return;
    }
    for (size_t block = begin; block < end; ++block) {
        memset(acc, 0, batch * 128 * sizeof(float));
        float tile[256];
        for (size_t kt = 0; kt < m->input_dim / 16; ++kt)
            for (unsigned nt = 0; nt < 8; ++nt) {
                decode(m, kt_phys(m, kt), nt_phys(m, block, nt), table, tile);
                for (size_t row = 0; row < batch; ++row)
                    for (unsigned k = 0; k < 16; ++k) {
                        float xx = x[row * m->input_dim + kt * 16 + k];
                        for (unsigned n = 0; n < 16; ++n)
                            acc[row * 128 + nt*16+n] += xx * tile[k*16+n];
                    }
            }
        for (size_t row = 0; row < batch; ++row) {
            hadamard(acc + row * 128);
            float *orow = out + (out_row0 + row) * out_stride;
            for (unsigned n = 0; n < 128; ++n) {
                size_t col = col_phys(m, block * 128 + n);
                orow[block * 128 + n] =
                    acc[row * 128 + n] * m->svh[col] + (m->bias ? m->bias[col] : 0);
            }
        }
    }
}

/* Gather + su-hadamard for `batch` rows selected by rows (NULL = contiguous
 * r-th row). dst gets the logical-order prepped rows. */
static void prep_rows(const OcExl3Matrix *m, const float *in, size_t in_stride,
                      const uint32_t *rows, size_t batch, float *dst)
{
    for (size_t r = 0; r < batch; ++r) {
        const float *src = in + (rows ? (size_t)rows[r] : r) * in_stride;
        float *d = dst + r * m->input_dim;
        for (size_t b = 0; b < m->input_dim / 128; ++b) {
            size_t pb = m->in_block_perm ? m->in_block_perm[b] : b;
            for (size_t k = 0; k < 128; ++k)
                d[b * 128 + k] = src[pb * 128 + k] * m->suh[pb * 128 + k];
            hadamard(d + b * 128);
        }
    }
}

typedef struct { const OcExl3Matrix *m; const float *x, *table;
                 float *out; size_t stride, rows, row0; } Job;
static void product(size_t begin, size_t end, size_t tid, void *arg)
{
    Job *j = arg;
    /* Whole batch in one sweep: packed tiles are decoded exactly once each
     * and reused across every batch row, instead of once per ROWS-row chunk.
     * Per-output accumulation order (kt ascending, then k, then n) is
     * unchanged, so results stay bit-identical to the serial reference. */
    blocks_prepped(j->m, j->x, j->rows, j->out, j->stride, j->row0,
                   begin, end, j->table, tid);
}
OcError oc_exl3_matmul(const OcExl3Matrix *m, const float *in, size_t is,
                      float *out, size_t os, size_t batch)
{
    OcError e = oc_exl3_validate(m);
    if (e != OC_OK) return e;
    if (!in || !out || !batch || is < m->input_dim || os < m->output_dim ||
        (batch-1) > (SIZE_MAX/sizeof(float)-m->input_dim)/is ||
        (batch-1) > (SIZE_MAX/sizeof(float)-m->output_dim)/os ||
        batch > SIZE_MAX/(128 * sizeof(float)) ||
        m->input_dim > SIZE_MAX/sizeof(float)/ROWS) return OC_ERR_INVALID_ARG;
    /* Input prep (su-hadamard) once for the whole batch; workers read it
     * read-only inside the single parallel sweep below. */
    float *x = scratch_get(m->input_dim * batch);
    if (!x) return OC_ERR_OOM;
    prep_rows(m, in, is, NULL, batch, x);
    Job j = {m, x, get_lut(m->codebook), out, os, batch, 0};
    oc_parallel_for(m->output_dim/128, product, &j);
    return OC_OK;
}
OcError oc_exl3_matvec(const OcExl3Matrix *m, const float *in, float *out)
{
    if (!m) return OC_ERR_INVALID_ARG;
    return oc_exl3_matmul(m, in, m->input_dim, out, m->output_dim, 1);
}

/* ── Per-expert MoE runner ───────────────────────────────────────────────
 *
 * One job per (selected expert, output block): each job gathers and preps
 * its own input rows (the packed tiles it decodes are disjoint from every
 * other job's), so the whole routed MoE runs in a single parallel region
 * per phase instead of one region per expert. */

typedef struct {
    const OcExl3Expert *ex;         /* n experts, same dims throughout  */
    uint32_t n;
    /* Phase input: rows==NULL → every expert consumes the single row `in`
     * (decode); otherwise expert e consumes in-rows rows[off[e]..off[e+1]). */
    const float *in;
    size_t in_stride;
    const uint32_t *rows, *off;
    /* Phase outputs: pair p = off[e]+j writes out row p (rows==NULL: row e).
     * gate/up phase writes gate_out/up_out; down phase reads act and writes
     * out. */
    float *gate_out, *up_out;
    const float *act;
    float *out;
    size_t gu_stride, act_stride, out_stride;
} ExpertJob;

static void expert_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    ExpertJob *j = ud;
    const OcExl3Expert *ex = j->ex;
    const OcExl3Matrix *g = ex[0].gate, *u = ex[0].up;
    const size_t nb_g = g->output_dim / 128;
    const size_t nb_u = u->output_dim / 128;
    const size_t per_e = nb_g + nb_u;
    const float *table = get_lut(g->codebook);
    for (size_t it = begin; it < end; ++it) {
        size_t e = it / per_e, r = it % per_e;
        const OcExl3Matrix *m = r < nb_g ? g : u;
        size_t block = r < nb_g ? r : r - nb_g;
        float *dst = (r < nb_g ? j->gate_out : j->up_out) +
            (j->rows ? (size_t)j->off[e] : e) * j->gu_stride;
        size_t batch = 1;
        const uint32_t *ids = NULL;
        if (j->rows) {
            batch = j->off[e + 1] - j->off[e];
            if (!batch) continue;
            ids = j->rows + j->off[e];
        }
        /* Per-job scratch: prepped input rows followed by the accumulator
         * rows blocks_prepped needs. The scratch persists per worker, so
         * the allocation cost amortizes across tokens. */
        if (batch > SIZE_MAX / (m->input_dim + 128) / sizeof(float)) continue;
        float *sc = oc_parallel_scratch(tid,
            batch * (m->input_dim + 128) * sizeof(float));
        if (!sc) continue;
        prep_rows(m, j->in, j->in_stride, ids, batch, sc);
        blocks_prepped(m, sc, batch, dst, j->gu_stride, 0, block, block + 1,
                       table, tid);
    }
}

static OcError experts_validate(const OcExl3Expert *ex, uint32_t n,
                                int phase, size_t *nb_out)
{
    if (!ex || !n) return OC_ERR_INVALID_ARG;
    const OcExl3Matrix *g = ex[0].gate, *u = ex[0].up, *d = ex[0].down;
    for (uint32_t i = 0; i < 3; ++i) {
        OcError e = oc_exl3_validate(i == 0 ? g : i == 1 ? u : d);
        if (e != OC_OK) return e;
    }
    /* gate and up share the expert shape; down consumes the activation. */
    if (u->input_dim != g->input_dim || u->output_dim != g->output_dim ||
        u->bits_twice != g->bits_twice || u->codebook != g->codebook ||
        d->input_dim != g->output_dim ||
        (d->in_block_perm != NULL) != (g->in_block_perm != NULL) ||
        (d->out_block_perm != NULL) != (g->out_block_perm != NULL))
        return OC_ERR_MODEL;
    *nb_out = phase ? d->output_dim / 128
                    : (g->output_dim + u->output_dim) / 128;
    return OC_OK;
}

OcError oc_exl3_experts_gate_up(const OcExl3Expert *experts, uint32_t n,
                                const float *in, size_t in_stride,
                                const uint32_t *rows, const uint32_t *off,
                                float *gate_out, float *up_out,
                                size_t out_stride)
{
    size_t nb = 0;
    OcError e = experts_validate(experts, n, 0, &nb);
    if (e != OC_OK) return e;
    if (!in || !gate_out || !up_out || !out_stride || (rows != NULL) != (off != NULL))
        return OC_ERR_INVALID_ARG;
    ExpertJob j = { experts, n, in, in_stride, rows, off,
                    gate_out, up_out, NULL, NULL, out_stride, 0, out_stride };
    oc_parallel_for((size_t)n * nb, expert_slice, &j);
    return OC_OK;
}

static void down_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    ExpertJob *j = ud;
    const OcExl3Expert *ex = j->ex;
    const OcExl3Matrix *d = ex[0].down;
    const size_t nb_d = d->output_dim / 128;
    const float *table = get_lut(d->codebook);
    for (size_t it = begin; it < end; ++it) {
        size_t e = it / nb_d, block = it % nb_d;
        size_t batch = 1, row0 = e;
        if (j->rows) {
            batch = j->off[e + 1] - j->off[e];
            if (!batch) continue;
            row0 = j->off[e];
        }
        if (batch > SIZE_MAX / (d->input_dim + 128) / sizeof(float)) continue;
        float *sc = oc_parallel_scratch(tid,
            batch * (d->input_dim + 128) * sizeof(float));
        if (!sc) continue;
        /* The down input is the SwiGLU'd activation rows, already gathered
         * by pair — copy them through the su-hadamard into scratch. */
        const float *src = j->act + (j->rows ? (size_t)j->off[e] : e) * j->act_stride;
        prep_rows(d, src, d->input_dim, NULL, batch, sc);
        blocks_prepped(d, sc, batch, j->out, j->out_stride, row0,
                       block, block + 1, table, tid);
    }
}

OcError oc_exl3_experts_down(const OcExl3Expert *experts, uint32_t n,
                             const float *act, size_t act_stride,
                             const uint32_t *rows, const uint32_t *off,
                             float *out, size_t out_stride)
{
    size_t nb = 0;
    OcError e = experts_validate(experts, n, 1, &nb);
    if (e != OC_OK) return e;
    if (!act || !out || !out_stride || (rows != NULL) != (off != NULL))
        return OC_ERR_INVALID_ARG;
    ExpertJob j = { experts, n, act, act_stride, rows, off,
                    NULL, NULL, act, out, 0, act_stride, out_stride };
    oc_parallel_for((size_t)n * nb, down_slice, &j);
    return OC_OK;
}
