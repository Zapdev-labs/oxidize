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
    if (k > SIZE_MAX / n || k * n > SIZE_MAX / bytes ||
        m->packed_size != k * n * bytes ||
        m->input_dim > SIZE_MAX / sizeof(float) || m->output_dim > SIZE_MAX / sizeof(float))
        return OC_ERR_FORMAT;
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
    const uint8_t *p = m->packed + (kt * (m->output_dim / 16) + nt) * (16 * b);
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
/* Fallback accumulator rows when per-thread scratch cannot be reserved: the
 * stack-buffered chunk size used by the serial path. */
typedef struct { const OcExl3Matrix *m; const float *x, *table; float *out; size_t stride, rows; } Job;
static void product(size_t begin, size_t end, size_t tid, void *arg)
{
    Job *j = arg; const OcExl3Matrix *m = j->m;
    /* Whole batch in one sweep: packed tiles are decoded exactly once each
     * and reused across every batch row, instead of once per ROWS-row chunk.
     * Per-output accumulation order (kt ascending, then k, then n) is
     * unchanged, so results stay bit-identical to the serial reference. */
    float *acc = oc_parallel_scratch(tid, j->rows * 128 * sizeof(float));
    if (acc == NULL) {
        /* Scratch failure: keep the old stack-buffered chunking. */
        for (size_t block = begin; block < end; ++block) {
            float acc4[ROWS][128] = {{0}}, tile[256];
            for (size_t base = 0; base < j->rows; base += ROWS) {
                const size_t rows = j->rows - base < ROWS ? j->rows - base : ROWS;
                memset(acc4, 0, rows * 128 * sizeof(float));
                for (size_t kt = 0; kt < m->input_dim / 16; ++kt)
                    for (unsigned nt = 0; nt < 8; ++nt) {
                        decode(m, kt, block * 8 + nt, j->table, tile);
                        for (size_t row = 0; row < rows; ++row)
                            for (unsigned k = 0; k < 16; ++k) {
                                float x = j->x[(base+row) * m->input_dim + kt * 16 + k];
                                for (unsigned n = 0; n < 16; ++n) acc4[row][nt*16+n] += x * tile[k*16+n];
                            }
                    }
                for (size_t row = 0; row < rows; ++row) {
                    hadamard(acc4[row]);
                    for (unsigned n = 0; n < 128; ++n) {
                        size_t col = block * 128 + n;
                        j->out[(base+row)*j->stride+col] = acc4[row][n] * m->svh[col] + (m->bias ? m->bias[col] : 0);
                    }
                }
            }
        }
        return;
    }
    for (size_t block = begin; block < end; ++block) {
        memset(acc, 0, j->rows * 128 * sizeof(float));
        float tile[256];
        for (size_t kt = 0; kt < m->input_dim / 16; ++kt)
            for (unsigned nt = 0; nt < 8; ++nt) {
                decode(m, kt, block * 8 + nt, j->table, tile);
                for (size_t row = 0; row < j->rows; ++row)
                    for (unsigned k = 0; k < 16; ++k) {
                        float x = j->x[row * m->input_dim + kt * 16 + k];
                        for (unsigned n = 0; n < 16; ++n) acc[row * 128 + nt*16+n] += x * tile[k*16+n];
                    }
            }
        for (size_t row = 0; row < j->rows; ++row) {
            hadamard(acc + row * 128);
            for (unsigned n = 0; n < 128; ++n) {
                size_t col = block * 128 + n;
                j->out[row*j->stride+col] = acc[row * 128 + n] * m->svh[col] + (m->bias ? m->bias[col] : 0);
            }
        }
    }
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
    for (size_t r = 0; r < batch; ++r) {
        for (size_t k = 0; k < m->input_dim; ++k)
            x[r*m->input_dim+k] = in[r*is+k] * m->suh[k];
        for (size_t k = 0; k < m->input_dim; k += 128) hadamard(x+r*m->input_dim+k);
    }
    Job j = {m, x, get_lut(m->codebook), out, os, batch};
    oc_parallel_for(m->output_dim/128, product, &j);
    return OC_OK;
}
OcError oc_exl3_matvec(const OcExl3Matrix *m, const float *in, float *out)
{
    if (!m) return OC_ERR_INVALID_ARG;
    return oc_exl3_matmul(m, in, m->input_dim, out, m->output_dim, 1);
}
