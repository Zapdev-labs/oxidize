/* From-scratch GPT-2 trainer. C + OpenBLAS SGEMM. Tied embeddings. */
#include <alloca.h>
#include <cblas.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum { BOS = 256, EOS = 257, FIRST_MERGE = 258 };

typedef struct {
    uint16_t left, right, id;
} Merge;

typedef struct {
    int n_layer, n_head, n_embd, vocab, seq, batch;
} Shape;

typedef struct {
    Shape s;
    size_t nparams;
    float *params;
    float *grads;
    float *m;
    float *v;
    /* offsets */
    size_t wte, wpe;
    size_t *ln1w, *ln1b, *qkvw, *qkvb, *attpw, *attpb;
    size_t *ln2w, *ln2b, *fcw, *fcb, *fcpw, *fcpb;
    size_t lnfw, lnfb;
} GPT;

static double wall_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *m) {
    fprintf(stderr, "%s\n", m);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = NULL;
    if (n == 0) n = 1;
    if (posix_memalign(&p, 64, n) != 0 || !p) die("oom");
    memset(p, 0, n);
    return p;
}

static float *falloc(size_t n) { return (float *)xmalloc(n * sizeof(float)); }

static unsigned rng_state = 1u;
static unsigned rngu(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static float rngn(void) {
    /* Box-Muller */
    float u = ((rngu() >> 8) + 1) / 16777216.0f;
    float v = ((rngu() >> 8) + 1) / 16777216.0f;
    return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
}

static size_t gpt_layout(GPT *g) {
    Shape s = g->s;
    int C = s.n_embd, L = s.n_layer, V = s.vocab, T = s.seq;
    size_t p = 0;
    g->wte = p;
    p += (size_t)V * (size_t)C;
    g->wpe = p;
    p += (size_t)T * (size_t)C;
    g->ln1w = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->ln1b = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->qkvw = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->qkvb = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->attpw = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->attpb = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->ln2w = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->ln2b = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->fcw = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->fcb = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->fcpw = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    g->fcpb = (size_t *)xmalloc((size_t)L * sizeof(size_t));
    for (int l = 0; l < L; l++) {
        g->ln1w[l] = p;
        p += (size_t)C;
        g->ln1b[l] = p;
        p += (size_t)C;
        g->qkvw[l] = p;
        p += (size_t)(3 * C) * (size_t)C;
        g->qkvb[l] = p;
        p += (size_t)(3 * C);
        g->attpw[l] = p;
        p += (size_t)C * (size_t)C;
        g->attpb[l] = p;
        p += (size_t)C;
        g->ln2w[l] = p;
        p += (size_t)C;
        g->ln2b[l] = p;
        p += (size_t)C;
        g->fcw[l] = p;
        p += (size_t)(4 * C) * (size_t)C;
        g->fcb[l] = p;
        p += (size_t)(4 * C);
        g->fcpw[l] = p;
        p += (size_t)C * (size_t)(4 * C);
        g->fcpb[l] = p;
        p += (size_t)C;
    }
    g->lnfw = p;
    p += (size_t)C;
    g->lnfb = p;
    p += (size_t)C;
    return p;
}

static void gpt_init(GPT *g, Shape s, unsigned seed) {
    memset(g, 0, sizeof(*g));
    g->s = s;
    g->nparams = gpt_layout(g);
    g->params = falloc(g->nparams);
    g->grads = falloc(g->nparams);
    g->m = falloc(g->nparams);
    g->v = falloc(g->nparams);
    rng_state = seed ? seed : 1u;
    float scale = 0.02f;
    for (size_t i = 0; i < g->nparams; i++) g->params[i] = scale * rngn();
}

static float *P(GPT *g, size_t off) { return g->params + off; }
static float *G(GPT *g, size_t off) { return g->grads + off; }

/* out[M,N] = inp[M,K] * weight[N,K]^T (+ bias[N]) */
static void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                           int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, inp, K, weight, K, 0.0f, out, N);
    if (bias) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < M; i++) {
            float *o = out + (size_t)i * (size_t)N;
            for (int n = 0; n < N; n++) o[n] += bias[n];
        }
    }
}

static void matmul_backward(const float *inp, const float *weight, const float *dout,
                            float *dinp, float *dweight, float *dbias, int M, int K, int N) {
    if (dweight) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, K, M, 1.0f, dout, N, inp, K, 1.0f,
                    dweight, K);
    }
    if (dinp) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1.0f, dout, N, weight, K, 1.0f,
                    dinp, K);
    }
    if (dbias) {
        for (int i = 0; i < M; i++) {
            const float *d = dout + (size_t)i * (size_t)N;
            for (int n = 0; n < N; n++) dbias[n] += d[n];
        }
    }
}

static void layernorm_forward(float *out, float *mean, float *rstd, const float *inp, const float *w,
                              const float *b, int BT, int C) {
    const float eps = 1e-5f;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < BT; i++) {
        const float *x = inp + (size_t)i * (size_t)C;
        float m = 0.0f;
        for (int c = 0; c < C; c++) m += x[c];
        m /= (float)C;
        float v = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = x[c] - m;
            v += d * d;
        }
        v /= (float)C;
        float rs = 1.0f / sqrtf(v + eps);
        mean[i] = m;
        rstd[i] = rs;
        float *o = out + (size_t)i * (size_t)C;
        for (int c = 0; c < C; c++) o[c] = (x[c] - m) * rs * w[c] + b[c];
    }
}

static void layernorm_backward(float *dinp, float *dw, float *db, const float *dout, const float *inp,
                               const float *w, const float *mean, const float *rstd, int BT, int C) {
    for (int i = 0; i < BT; i++) {
        const float *x = inp + (size_t)i * (size_t)C;
        const float *d = dout + (size_t)i * (size_t)C;
        float *dx = dinp + (size_t)i * (size_t)C;
        float m = mean[i], rs = rstd[i];
        float dnorm_mean = 0.0f, dnorm_norm = 0.0f;
        for (int c = 0; c < C; c++) {
            float nrm = (x[c] - m) * rs;
            db[c] += d[c];
            dw[c] += d[c] * nrm;
            float dnorm = d[c] * w[c];
            dnorm_mean += dnorm;
            dnorm_norm += dnorm * nrm;
        }
        float invC = 1.0f / (float)C;
        for (int c = 0; c < C; c++) {
            float nrm = (x[c] - m) * rs;
            float dnorm = d[c] * w[c];
            dx[c] += (dnorm - dnorm_mean * invC - nrm * dnorm_norm * invC) * rs;
        }
    }
}

static void gelu_forward(float *out, const float *inp, int N) {
    const float s = 0.7978845608f; /* sqrt(2/pi) */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float u = s * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(u));
    }
}

static void gelu_backward(float *dinp, const float *inp, const float *dout, int N) {
    const float s = 0.7978845608f;
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float x3 = x * x * x;
        float u = s * (x + 0.044715f * x3);
        float t = tanhf(u);
        float dt = 1.0f - t * t;
        float du = s * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += dout[i] * (0.5f * (1.0f + t) + 0.5f * x * dt * du);
    }
}

static void residual(float *out, const float *a, const float *b, int N) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++) out[i] = a[i] + b[i];
}

static void encoder_forward(float *out, const int *tok, const float *wte, const float *wpe, int B, int T,
                            int C, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int id = tok[b * T + t];
            if (id < 0 || id >= V) id = 0;
            const float *e = wte + (size_t)id * (size_t)C;
            const float *p = wpe + (size_t)t * (size_t)C;
            float *o = out + ((size_t)b * (size_t)T + (size_t)t) * (size_t)C;
            for (int c = 0; c < C; c++) o[c] = e[c] + p[c];
        }
    }
}

static void encoder_backward(float *dwte, float *dwpe, const float *dout, const int *tok, int B, int T,
                             int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int id = tok[b * T + t];
            const float *d = dout + ((size_t)b * (size_t)T + (size_t)t) * (size_t)C;
            float *e = dwte + (size_t)id * (size_t)C;
            float *p = dwpe + (size_t)t * (size_t)C;
            for (int c = 0; c < C; c++) {
                e[c] += d[c];
                p[c] += d[c];
            }
        }
    }
}

static void attention_forward(float *out, float *preatt, float *att, const float *inp, int B, int T, int C,
                              int NH) {
    int HS = C / NH;
    float scale = 1.0f / sqrtf((float)HS);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {
                const float *q = inp + ((size_t)b * T + t) * 3 * C + h * HS;
                float *patt = preatt + ((size_t)((b * NH + h) * T + t) * T);
                float maxv = -1e9f;
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 > t) {
                        patt[t2] = -1e10f;
                        continue;
                    }
                    const float *k = inp + ((size_t)b * T + t2) * 3 * C + C + h * HS;
                    float s = 0.0f;
                    for (int i = 0; i < HS; i++) s += q[i] * k[i];
                    s *= scale;
                    patt[t2] = s;
                    if (s > maxv) maxv = s;
                }
                float *at = att + ((size_t)((b * NH + h) * T + t) * T);
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float e = expf(patt[t2] - maxv);
                    at[t2] = e;
                    sum += e;
                }
                for (int t2 = t + 1; t2 < T; t2++) at[t2] = 0.0f;
                float inv = 1.0f / (sum + 1e-20f);
                for (int t2 = 0; t2 <= t; t2++) at[t2] *= inv;
                float *o = out + ((size_t)b * T + t) * C + h * HS;
                for (int i = 0; i < HS; i++) o[i] = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *v = inp + ((size_t)b * T + t2) * 3 * C + 2 * C + h * HS;
                    float a = at[t2];
                    for (int i = 0; i < HS; i++) o[i] += a * v[i];
                }
            }
        }
    }
}

static void attention_backward(float *dinp, const float *dout, const float *inp, const float *att, int B,
                               int T, int C, int NH) {
    int HS = C / NH;
    float scale = 1.0f / sqrtf((float)HS);
    memset(dinp, 0, (size_t)B * T * 3 * C * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {
                const float *at = att + ((size_t)((b * NH + h) * T + t) * T);
                const float *d = dout + ((size_t)b * T + t) * C + h * HS;
                float *dq = dinp + ((size_t)b * T + t) * 3 * C + h * HS;
                /* dV */
                for (int t2 = 0; t2 <= t; t2++) {
                    float *dv = dinp + ((size_t)b * T + t2) * 3 * C + 2 * C + h * HS;
                    float a = at[t2];
                    for (int i = 0; i < HS; i++) dv[i] += a * d[i];
                }
                /* dAtt then softmax backward then dQ dK */
                float datt[1024];
                if (T > 1024) die("seq too long");
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *v = inp + ((size_t)b * T + t2) * 3 * C + 2 * C + h * HS;
                    float s = 0.0f;
                    for (int i = 0; i < HS; i++) s += v[i] * d[i];
                    datt[t2] = s;
                }
                float dpre[1024];
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) sum += at[t2] * datt[t2];
                for (int t2 = 0; t2 <= t; t2++) dpre[t2] = at[t2] * (datt[t2] - sum);
                const float *q = inp + ((size_t)b * T + t) * 3 * C + h * HS;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *k = inp + ((size_t)b * T + t2) * 3 * C + C + h * HS;
                    float *dk = dinp + ((size_t)b * T + t2) * 3 * C + C + h * HS;
                    float dp = dpre[t2] * scale;
                    for (int i = 0; i < HS; i++) {
                        dq[i] += dp * k[i];
                        dk[i] += dp * q[i];
                    }
                }
            }
        }
    }
}

static float softmax_ce_backward(float *dlogits, const float *logits, const int *targets, int B, int T,
                                 int V) {
    float loss = 0.0f;
    int n = 0;
    for (int i = 0; i < B * T; i++) {
        const float *lg = logits + (size_t)i * (size_t)V;
        float *dg = dlogits + (size_t)i * (size_t)V;
        float m = lg[0];
        for (int v = 1; v < V; v++)
            if (lg[v] > m) m = lg[v];
        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            float e = expf(lg[v] - m);
            dg[v] = e;
            sum += e;
        }
        float inv = 1.0f / (sum + 1e-20f);
        int y = targets[i];
        if (y < 0 || y >= V) y = 0;
        for (int v = 0; v < V; v++) dg[v] *= inv;
        loss += -logf(dg[y] + 1e-20f);
        dg[y] -= 1.0f;
        n++;
    }
    float scale = 1.0f / (float)n;
    for (int i = 0; i < B * T * V; i++) dlogits[i] *= scale;
    return loss * scale;
}

typedef struct {
    float *emb;
    float *ln1, *ln1_mean, *ln1_rstd;
    float *qkv;
    float *atty;
    float *preatt, *att;
    float *attproj;
    float *residual2;
    float *ln2, *ln2_mean, *ln2_rstd;
    float *fc, *fcg;
    float *fcproj;
    float *residual3;
} LayerAct;

typedef struct {
    float *encoded;
    LayerAct *L;
    float *lnf, *lnf_mean, *lnf_rstd;
    float *logits;
    float *dencoded;
    float *dlogits;
    LayerAct *dL;
} Acts;

static Acts acts_alloc(Shape s) {
    Acts a;
    memset(&a, 0, sizeof(a));
    int B = s.batch, T = s.seq, C = s.n_embd, V = s.vocab, NH = s.n_head, L = s.n_layer;
    int BT = B * T;
    a.encoded = falloc((size_t)BT * C);
    a.dencoded = falloc((size_t)BT * C);
    a.lnf = falloc((size_t)BT * C);
    a.lnf_mean = falloc((size_t)BT);
    a.lnf_rstd = falloc((size_t)BT);
    a.logits = falloc((size_t)BT * V);
    a.dlogits = falloc((size_t)BT * V);
    a.L = (LayerAct *)xmalloc((size_t)L * sizeof(LayerAct));
    a.dL = (LayerAct *)xmalloc((size_t)L * sizeof(LayerAct));
    for (int l = 0; l < L; l++) {
        LayerAct *x = &a.L[l];
        LayerAct *d = &a.dL[l];
        x->ln1 = falloc((size_t)BT * C);
        x->ln1_mean = falloc((size_t)BT);
        x->ln1_rstd = falloc((size_t)BT);
        x->qkv = falloc((size_t)BT * 3 * C);
        x->atty = falloc((size_t)BT * C);
        x->preatt = falloc((size_t)B * NH * T * T);
        x->att = falloc((size_t)B * NH * T * T);
        x->attproj = falloc((size_t)BT * C);
        x->residual2 = falloc((size_t)BT * C);
        x->ln2 = falloc((size_t)BT * C);
        x->ln2_mean = falloc((size_t)BT);
        x->ln2_rstd = falloc((size_t)BT);
        x->fc = falloc((size_t)BT * 4 * C);
        x->fcg = falloc((size_t)BT * 4 * C);
        x->fcproj = falloc((size_t)BT * C);
        x->residual3 = falloc((size_t)BT * C);
        d->ln1 = falloc((size_t)BT * C);
        d->qkv = falloc((size_t)BT * 3 * C);
        d->atty = falloc((size_t)BT * C);
        d->attproj = falloc((size_t)BT * C);
        d->residual2 = falloc((size_t)BT * C);
        d->ln2 = falloc((size_t)BT * C);
        d->fc = falloc((size_t)BT * 4 * C);
        d->fcg = falloc((size_t)BT * 4 * C);
        d->fcproj = falloc((size_t)BT * C);
        d->residual3 = falloc((size_t)BT * C);
    }
    return a;
}

static float gpt_forward_backward(GPT *g, Acts *a, const int *inputs, const int *targets, int do_bwd) {
    Shape s = g->s;
    int B = s.batch, T = s.seq, C = s.n_embd, V = s.vocab, NH = s.n_head, L = s.n_layer;
    int BT = B * T;
    encoder_forward(a->encoded, inputs, P(g, g->wte), P(g, g->wpe), B, T, C, V);
    const float *x = a->encoded;
    for (int l = 0; l < L; l++) {
        LayerAct *la = &a->L[l];
        layernorm_forward(la->ln1, la->ln1_mean, la->ln1_rstd, x, P(g, g->ln1w[l]), P(g, g->ln1b[l]), BT, C);
        matmul_forward(la->qkv, la->ln1, P(g, g->qkvw[l]), P(g, g->qkvb[l]), BT, C, 3 * C);
        attention_forward(la->atty, la->preatt, la->att, la->qkv, B, T, C, NH);
        matmul_forward(la->attproj, la->atty, P(g, g->attpw[l]), P(g, g->attpb[l]), BT, C, C);
        residual(la->residual2, x, la->attproj, BT * C);
        layernorm_forward(la->ln2, la->ln2_mean, la->ln2_rstd, la->residual2, P(g, g->ln2w[l]),
                          P(g, g->ln2b[l]), BT, C);
        matmul_forward(la->fc, la->ln2, P(g, g->fcw[l]), P(g, g->fcb[l]), BT, C, 4 * C);
        gelu_forward(la->fcg, la->fc, BT * 4 * C);
        matmul_forward(la->fcproj, la->fcg, P(g, g->fcpw[l]), P(g, g->fcpb[l]), BT, 4 * C, C);
        residual(la->residual3, la->residual2, la->fcproj, BT * C);
        x = la->residual3;
    }
    layernorm_forward(a->lnf, a->lnf_mean, a->lnf_rstd, x, P(g, g->lnfw), P(g, g->lnfb), BT, C);
    matmul_forward(a->logits, a->lnf, P(g, g->wte), NULL, BT, C, V);
    float loss = softmax_ce_backward(a->dlogits, a->logits, targets, B, T, V);
    if (!do_bwd) return loss;

    memset(g->grads, 0, g->nparams * sizeof(float));
    memset(a->dencoded, 0, (size_t)BT * C * sizeof(float));
    float *dlnf = falloc((size_t)BT * C);
    memset(dlnf, 0, (size_t)BT * C * sizeof(float));
    matmul_backward(a->lnf, P(g, g->wte), a->dlogits, dlnf, G(g, g->wte), NULL, BT, C, V);
    float *dcur = falloc((size_t)BT * C);
    memset(dcur, 0, (size_t)BT * C * sizeof(float));
    layernorm_backward(dcur, G(g, g->lnfw), G(g, g->lnfb), dlnf, x, P(g, g->lnfw), a->lnf_mean, a->lnf_rstd,
                       BT, C);
    free(dlnf);
    for (int l = L - 1; l >= 0; l--) {
        LayerAct *la = &a->L[l];
        LayerAct *da = &a->dL[l];
        memset(da->fcproj, 0, (size_t)BT * C * sizeof(float));
        memset(da->residual2, 0, (size_t)BT * C * sizeof(float));
        memset(da->fcg, 0, (size_t)BT * 4 * C * sizeof(float));
        memset(da->fc, 0, (size_t)BT * 4 * C * sizeof(float));
        memset(da->ln2, 0, (size_t)BT * C * sizeof(float));
        memset(da->attproj, 0, (size_t)BT * C * sizeof(float));
        memset(da->atty, 0, (size_t)BT * C * sizeof(float));
        memset(da->qkv, 0, (size_t)BT * 3 * C * sizeof(float));
        memset(da->ln1, 0, (size_t)BT * C * sizeof(float));
        for (int i = 0; i < BT * C; i++) {
            da->fcproj[i] = dcur[i];
            da->residual2[i] = dcur[i];
        }
        matmul_backward(la->fcg, P(g, g->fcpw[l]), da->fcproj, da->fcg, G(g, g->fcpw[l]), G(g, g->fcpb[l]), BT,
                        4 * C, C);
        gelu_backward(da->fc, la->fc, da->fcg, BT * 4 * C);
        matmul_backward(la->ln2, P(g, g->fcw[l]), da->fc, da->ln2, G(g, g->fcw[l]), G(g, g->fcb[l]), BT, C,
                        4 * C);
        layernorm_backward(da->residual2, G(g, g->ln2w[l]), G(g, g->ln2b[l]), da->ln2, la->residual2,
                           P(g, g->ln2w[l]), la->ln2_mean, la->ln2_rstd, BT, C);
        for (int i = 0; i < BT * C; i++) da->attproj[i] = da->residual2[i];
        const float *xin = (l == 0) ? a->encoded : a->L[l - 1].residual3;
        float *dxin = (l == 0) ? a->dencoded : dcur;
        if (l == 0)
            memset(a->dencoded, 0, (size_t)BT * C * sizeof(float));
        else
            memset(dcur, 0, (size_t)BT * C * sizeof(float));
        for (int i = 0; i < BT * C; i++) dxin[i] += da->residual2[i];
        matmul_backward(la->atty, P(g, g->attpw[l]), da->attproj, da->atty, G(g, g->attpw[l]),
                        G(g, g->attpb[l]), BT, C, C);
        attention_backward(da->qkv, da->atty, la->qkv, la->att, B, T, C, NH);
        matmul_backward(la->ln1, P(g, g->qkvw[l]), da->qkv, da->ln1, G(g, g->qkvw[l]), G(g, g->qkvb[l]), BT, C,
                        3 * C);
        layernorm_backward(dxin, G(g, g->ln1w[l]), G(g, g->ln1b[l]), da->ln1, xin, P(g, g->ln1w[l]),
                           la->ln1_mean, la->ln1_rstd, BT, C);
        if (l == 0) encoder_backward(G(g, g->wte), G(g, g->wpe), a->dencoded, inputs, B, T, C);
    }
    free(dcur);
    return loss;
}

static void adamw(GPT *g, float lr, float wd, int t) {
    const float b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, clip = 1.0f;
    float n2 = 0.0f;
    for (size_t i = 0; i < g->nparams; i++) n2 += g->grads[i] * g->grads[i];
    float nrm = sqrtf(n2);
    float scale = nrm > clip ? clip / (nrm + 1e-20f) : 1.0f;
    float bc1 = 1.0f - powf(b1, (float)t);
    float bc2 = 1.0f - powf(b2, (float)t);
    for (size_t i = 0; i < g->nparams; i++) {
        float gti = g->grads[i] * scale;
        g->m[i] = b1 * g->m[i] + (1.0f - b1) * gti;
        g->v[i] = b2 * g->v[i] + (1.0f - b2) * gti * gti;
        float mhat = g->m[i] / bc1;
        float vhat = g->v[i] / bc2;
        g->params[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * g->params[i]);
    }
}

static uint16_t *load_tokens(const char *path, uint32_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open tokens");
    if (fread(n, 4, 1, f) != 1) die("tokens header");
    uint16_t *t = (uint16_t *)xmalloc((size_t)*n * 2);
    if (fread(t, 2, *n, f) != *n) die("tokens body");
    fclose(f);
    return t;
}

static Merge *load_merges(const char *path, int *nmerge, int *vocab) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open vocab");
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) die("vocab header");
    Merge *m = (Merge *)xmalloc((n ? n : 1) * sizeof(Merge));
    if (n && fread(m, sizeof(Merge), n, f) != n) die("vocab body");
    fclose(f);
    *nmerge = (int)n;
    *vocab = FIRST_MERGE + (int)n;
    return m;
}

static void decode_id(int id, const Merge *merges, int nmerge, FILE *out) {
    if (id < 256) {
        fputc(id, out);
        return;
    }
    if (id == BOS || id == EOS) return;
    for (int i = 0; i < nmerge; i++) {
        if (merges[i].id == (uint16_t)id) {
            decode_id(merges[i].left, merges, nmerge, out);
            decode_id(merges[i].right, merges, nmerge, out);
            return;
        }
    }
}

static int encode_prompt(const char *p, const Merge *merges, int nmerge, int *ids, int cap) {
    int n = 0;
    ids[n++] = BOS;
    int tmp[4096];
    int tn = 0;
    for (const unsigned char *s = (const unsigned char *)p; *s && tn < 4000; s++) tmp[tn++] = *s;
    /* apply merges like bpe.c */
    uint16_t buf[4096];
    for (int i = 0; i < tn; i++) buf[i] = (uint16_t)tmp[i];
    int m = tn;
    for (int mi = 0; mi < nmerge; mi++) {
        uint16_t a = merges[mi].left, b = merges[mi].right, id = merges[mi].id;
        int w = 0;
        for (int i = 0; i < m; i++) {
            if (i + 1 < m && buf[i] == a && buf[i + 1] == b) {
                buf[w++] = id;
                i++;
            } else
                buf[w++] = buf[i];
        }
        m = w;
    }
    for (int i = 0; i < m && n < cap - 1; i++) ids[n++] = buf[i];
    return n;
}

static int argi(int argc, char **argv, const char *k, int def) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], k)) return atoi(argv[i + 1]);
    return def;
}
static const char *args(int argc, char **argv, const char *k, const char *def) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}
static int has(int argc, char **argv, const char *k) {
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], k)) return 1;
    return 0;
}

static void save_ckpt(const char *path, GPT *g) {
    FILE *f = fopen(path, "wb");
    if (!f) die("ckpt write");
    char mag[4] = {'O', 'X', 'T', 'R'};
    fwrite(mag, 1, 4, f);
    uint32_t v = 1;
    fwrite(&v, 4, 1, f);
    int32_t sh[6] = {g->s.n_layer, g->s.n_head, g->s.n_embd, g->s.vocab, g->s.seq, g->s.batch};
    fwrite(sh, 4, 6, f);
    uint64_t np = g->nparams;
    fwrite(&np, 8, 1, f);
    fwrite(g->params, 4, g->nparams, f);
    fclose(f);
}

static void load_ckpt(const char *path, GPT *g) {
    FILE *f = fopen(path, "rb");
    if (!f) die("ckpt open");
    char mag[4];
    if (fread(mag, 1, 4, f) != 4 || memcmp(mag, "OXTR", 4) != 0) die("bad magic");
    uint32_t v = 0;
    fread(&v, 4, 1, f);
    int32_t sh[6];
    fread(sh, 4, 6, f);
    Shape s = {sh[0], sh[1], sh[2], sh[3], sh[4], sh[5]};
    gpt_init(g, s, 1);
    uint64_t np = 0;
    fread(&np, 8, 1, f);
    if (np != g->nparams) die("ckpt param mismatch");
    if (fread(g->params, 4, g->nparams, f) != g->nparams) die("ckpt body");
    fclose(f);
}

static int sample_token(const float *logits, int V, float temp) {
    float m = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > m) m = logits[i];
    float sum = 0.0f;
    float *p = (float *)alloca((size_t)V * 4);
    float invt = 1.0f / (temp < 1e-4f ? 1e-4f : temp);
    for (int i = 0; i < V; i++) {
        p[i] = expf((logits[i] - m) * invt);
        sum += p[i];
    }
    float r = (rngu() / 4294967296.0f) * sum;
    float acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += p[i];
        if (r <= acc) return i;
    }
    return V - 1;
}

static int encode_str_last(const char *p, const Merge *merges, int nmerge) {
    int ids[2048];
    int n = encode_prompt(p, merges, nmerge, ids, 2048);
    return n > 0 ? ids[n - 1] : 0;
}

typedef struct {
    int b, s, h;
} ActTok;

static ActTok make_act_tok(const Merge *merges, int nmerge) {
    ActTok a;
    a.b = encode_str_last("ACTION=BUY", merges, nmerge);
    a.s = encode_str_last("ACTION=SELL", merges, nmerge);
    a.h = encode_str_last("ACTION=HOLD", merges, nmerge);
    return a;
}

/* ACTION= next piece may be a merged BUY/SELL/HOLD token, not the bytes B/S/H. */
static int sample_action_token(const float *logits, int V, ActTok t) {
    int cands[3] = {t.b, t.s, t.h};
    int best = cands[0];
    float bv = -1e30f;
    for (int i = 0; i < 3; i++) {
        int id = cands[i];
        if (id < 0 || id >= V) continue;
        if (logits[id] > bv) {
            bv = logits[id];
            best = id;
        }
    }
    return best;
}

static const char *WINTER_CLS[9] = {
    "ACTION=1", "ACTION=2", "ACTION=3",
    "ACTION=4", "ACTION=5", "ACTION=6",
    "ACTION=7", "ACTION=8", "ACTION=9",
};

typedef struct {
    int id[9];
} WinterTok;

static WinterTok make_winter_tok(const Merge *merges, int nmerge) {
    WinterTok w;
    for (int i = 0; i < 9; i++)
        w.id[i] = encode_str_last(WINTER_CLS[i], merges, nmerge);
    return w;
}

static int sample_winter_token(const float *logits, int V, WinterTok t) {
    int best = t.id[0];
    float bv = -1e30f;
    for (int i = 0; i < 9; i++) {
        int id = t.id[i];
        if (id < 0 || id >= V)
            continue;
        if (logits[id] > bv) {
            bv = logits[id];
            best = id;
        }
    }
    return best;
}

static void cmd_sample(int argc, char **argv) {
    const char *ckpt = args(argc, argv, "--ckpt", "out/model.bin");
    const char *vocab = args(argc, argv, "--vocab", "data/vocab.bin");
    const char *prompt = args(argc, argv, "--prompt", "AAPL 1h close 228 RSI 32 ATR 9. Setup: opening range breakout. ACTION=");
    int ntok = argi(argc, argv, "--tokens", 80);
    float temp = (float)atof(args(argc, argv, "--temp", "0.8"));
    unsigned seed = (unsigned)argi(argc, argv, "--seed", 1);
    rng_state = seed ? seed : 1u;
    int force_action = !has(argc, argv, "--no-force-action");
    int nmerge = 0, vocab_sz = 0;
    Merge *merges = load_merges(vocab, &nmerge, &vocab_sz);
    ActTok atok = make_act_tok(merges, nmerge);
    GPT g;
    load_ckpt(ckpt, &g);
    g.s.batch = 1;
    int T = g.s.seq, C = g.s.n_embd, V = g.s.vocab;
    Acts a = acts_alloc(g.s);
    int ids[4096];
    int n = encode_prompt(prompt, merges, nmerge, ids, T);
    if (n >= T) n = T - 1;
    printf("prompt: %s\n---\n", prompt);
    for (int t = 0; t < n; t++) decode_id(ids[t], merges, nmerge, stdout);
    fflush(stdout);
    int *inp = (int *)xmalloc((size_t)T * sizeof(int));
    int *tgt = (int *)xmalloc((size_t)T * sizeof(int));
    for (int k = 0; k < ntok; k++) {
        memset(inp, 0, (size_t)T * sizeof(int));
        int start = n > T ? n - T : 0;
        int used = n - start;
        for (int i = 0; i < used; i++) inp[i] = ids[start + i];
        for (int i = 0; i < T; i++) tgt[i] = 0;
        gpt_forward_backward(&g, &a, inp, tgt, 0);
        int pos = used - 1;
        if (pos < 0) pos = 0;
        const float *lg = a.logits + (size_t)pos * (size_t)V;
        if (k == 0 && has(argc, argv, "--show-top")) {
            printf("\n[top next tokens]\n");
            int used_i[16];
            float used_v[16];
            int nt = 0;
            for (int r = 0; r < 8; r++) {
                int bi = 0;
                float bv = -1e30f;
                for (int i = 0; i < V; i++) {
                    int skip = 0;
                    for (int j = 0; j < nt; j++)
                        if (used_i[j] == i) skip = 1;
                    if (skip) continue;
                    if (lg[i] > bv) {
                        bv = lg[i];
                        bi = i;
                    }
                }
                used_i[nt] = bi;
                used_v[nt] = bv;
                nt++;
                printf("  %d %.3f '", bi, used_v[nt - 1]);
                decode_id(bi, merges, nmerge, stdout);
                printf("'\n");
            }
            fflush(stdout);
        }
        int tok;
        if (k == 0 && force_action && strstr(prompt, "ACTION="))
            tok = sample_action_token(lg, V, atok);
        else
            tok = sample_token(lg, V, temp);
        if (tok == EOS) break;
        decode_id(tok, merges, nmerge, stdout);
        fflush(stdout);
        if (n < 4000) ids[n++] = tok;
        (void)C;
    }
    printf("\n");
}

static void cmd_complete(int argc, char **argv) {
    const char *ckpt = args(argc, argv, "--ckpt", "out/model.bin");
    const char *vocab = args(argc, argv, "--vocab", "data/vocab.bin");
    int ntok = argi(argc, argv, "--tokens", 16);
    float temp = (float)atof(args(argc, argv, "--temp", "0.4"));
    unsigned seed = (unsigned)argi(argc, argv, "--seed", 1);
    rng_state = seed ? seed : 1u;
    int nmerge = 0, vocab_sz = 0;
    Merge *merges = load_merges(vocab, &nmerge, &vocab_sz);
    WinterTok wtok = make_winter_tok(merges, nmerge);
    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (wtok.id[i] == wtok.id[j])
                die("winter class token collision");
        }
    }
    GPT g;
    load_ckpt(ckpt, &g);
    g.s.batch = 1;
    int T = g.s.seq, V = g.s.vocab;
    Acts a = acts_alloc(g.s);
    int *inp = (int *)xmalloc((size_t)T * sizeof(int));
    int *tgt = (int *)xmalloc((size_t)T * sizeof(int));
    char line[4096];
    while (fgets(line, (int)sizeof line, stdin)) {
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r'))
            line[--L] = 0;
        if (!L)
            continue;
        int ids[4096];
        int n = encode_prompt(line, merges, nmerge, ids, T);
        if (n >= T)
            n = T - 1;
        int n0 = n;
        for (int k = 0; k < ntok; k++) {
            memset(inp, 0, (size_t)T * sizeof(int));
            memset(tgt, 0, (size_t)T * sizeof(int));
            int start = n > T ? n - T : 0;
            int used = n - start;
            for (int i = 0; i < used; i++)
                inp[i] = ids[start + i];
            gpt_forward_backward(&g, &a, inp, tgt, 0);
            int pos = used - 1;
            if (pos < 0)
                pos = 0;
            const float *lg = a.logits + (size_t)pos * (size_t)V;
            int tok;
            if (k == 0)
                tok = sample_winter_token(lg, V, wtok);
            else
                tok = sample_token(lg, V, temp);
            if (tok == EOS)
                break;
            if (n < 4000)
                ids[n++] = tok;
        }
        for (int t = n0; t < n; t++)
            decode_id(ids[t], merges, nmerge, stdout);
        printf("\n");
        fflush(stdout);
        (void)vocab_sz;
    }
}

static int action_from_logits(const float *lg, int V, ActTok t, float *lb, float *ls, float *lh) {
    *lb = (t.b >= 0 && t.b < V) ? lg[t.b] : -1e30f;
    *ls = (t.s >= 0 && t.s < V) ? lg[t.s] : -1e30f;
    *lh = (t.h >= 0 && t.h < V) ? lg[t.h] : -1e30f;
    if (*ls >= *lb && *ls >= *lh) return 'S';
    if (*lh >= *lb && *lh >= *ls) return 'H';
    return 'B';
}

static int score_prompt(GPT *g, Acts *a, const Merge *merges, int nmerge, const char *prompt, int *inp,
                        int *tgt, ActTok t) {
    int T = g->s.seq, V = g->s.vocab;
    int ids[4096];
    int n = encode_prompt(prompt, merges, nmerge, ids, T);
    if (n >= T) n = T - 1;
    if (n < 1) n = 1;
    memset(inp, 0, (size_t)T * sizeof(int));
    memset(tgt, 0, (size_t)T * sizeof(int));
    for (int i = 0; i < n; i++) inp[i] = ids[i];
    gpt_forward_backward(g, a, inp, tgt, 0);
    int pos = n - 1;
    const float *lg = a->logits + (size_t)pos * (size_t)V;
    float lb, ls, lh;
    return action_from_logits(lg, V, t, &lb, &ls, &lh);
}

static void cmd_score(int argc, char **argv) {
    const char *ckpt = args(argc, argv, "--ckpt", "out/model.bin");
    const char *vocab = args(argc, argv, "--vocab", "data/vocab.bin");
    int nmerge = 0, vocab_sz = 0;
    Merge *merges = load_merges(vocab, &nmerge, &vocab_sz);
    ActTok atok = make_act_tok(merges, nmerge);
    GPT g;
    load_ckpt(ckpt, &g);
    g.s.batch = 1;
    int T = g.s.seq, V = g.s.vocab;
    Acts a = acts_alloc(g.s);
    int *inp = (int *)xmalloc((size_t)T * sizeof(int));
    int *tgt = (int *)xmalloc((size_t)T * sizeof(int));
    char line[2048];
    while (fgets(line, (int)sizeof line, stdin)) {
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!L) continue;
        int ids[4096];
        int n = encode_prompt(line, merges, nmerge, ids, T);
        if (n >= T) n = T - 1;
        memset(inp, 0, (size_t)T * sizeof(int));
        memset(tgt, 0, (size_t)T * sizeof(int));
        for (int i = 0; i < n; i++) inp[i] = ids[i];
        gpt_forward_backward(&g, &a, inp, tgt, 0);
        int pos = n - 1;
        if (pos < 0) pos = 0;
        const float *lg = a.logits + (size_t)pos * (size_t)V;
        float lb, ls, lh;
        int act = action_from_logits(lg, V, atok, &lb, &ls, &lh);
        const char *name = act == 'B' ? "BUY" : act == 'S' ? "SELL" : "HOLD";
        printf("%s %.4f %.4f %.4f\n", name, lb, ls, lh);
        fflush(stdout);
        (void)vocab_sz;
    }
}

typedef struct {
    char d[12];
    double o, h, l, c;
} Bar;

static int load_ohlc(const char *path, Bar *b, int cap) {
    FILE *f = fopen(path, "r");
    if (!f) die("open csv");
    char line[512];
    if (!fgets(line, (int)sizeof line, f)) die("empty csv");
    int n = 0;
    while (fgets(line, (int)sizeof line, f) && n < cap) {
        char d[12];
        double o, h, l, c, v;
        if (sscanf(line, " %11[^,],%lf,%lf,%lf,%lf,%lf", d, &o, &h, &l, &c, &v) != 6) {
            if (sscanf(line, " %11[^,],%lf,%lf,%lf,%lf", d, &o, &h, &l, &c) != 5) continue;
        }
        if (!(c > 0.0 && h >= l && o > 0.0)) continue;
        snprintf(b[n].d, sizeof b[n].d, "%s", d);
        b[n].o = o;
        b[n].h = h;
        b[n].l = l;
        b[n].c = c;
        n++;
    }
    fclose(f);
    if (n < 50) die("csv too short");
    if (strcmp(b[0].d, b[n - 1].d) > 0) {
        for (int i = 0; i < n / 2; i++) {
            Bar t = b[i];
            b[i] = b[n - 1 - i];
            b[n - 1 - i] = t;
        }
    }
    return n;
}

static void load_news_day(const char *dir, const char *day, char *out, int cap) {
    if (!dir || !dir[0] || !day || !day[0]) {
        snprintf(out, cap, "Quiet political tape.");
        return;
    }
    char path[768];
    snprintf(path, sizeof path, "%s/%s.txt", dir, day);
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(out, cap, "Quiet political tape.");
        return;
    }
    if (!fgets(out, cap, f)) snprintf(out, cap, "Quiet political tape.");
    fclose(f);
    size_t n = strlen(out);
    while (n && (out[n - 1] == '\n' || out[n - 1] == '\r')) out[--n] = 0;
    if ((int)n > 180 && cap > 181) {
        out[177] = '.';
        out[178] = '.';
        out[179] = '.';
        out[180] = 0;
    }
}

static void wilder_rsi_atr(const Bar *b, int n, double *rsi, double *atr, double *ema) {
    const int p = 14;
    double ag = 0, al = 0, atr_s = 0, sma = 0;
    rsi[0] = 50;
    atr[0] = b[0].h - b[0].l;
    ema[0] = b[0].c;
    for (int i = 1; i < n; i++) {
        double ch = b[i].c - b[i - 1].c;
        double g = ch > 0 ? ch : 0;
        double lss = ch < 0 ? -ch : 0;
        double tr = b[i].h - b[i].l;
        double x = fabs(b[i].h - b[i - 1].c);
        double y = fabs(b[i].l - b[i - 1].c);
        if (x > tr) tr = x;
        if (y > tr) tr = y;
        if (i < p) {
            ag += g;
            al += lss;
            atr_s += tr;
            rsi[i] = 50;
            atr[i] = atr_s / i;
        } else if (i == p) {
            ag = (ag + g) / p;
            al = (al + lss) / p;
            atr_s = (atr_s + tr) / p;
            rsi[i] = al < 1e-12 ? 100 : 100.0 - 100.0 / (1.0 + ag / al);
            atr[i] = atr_s;
        } else {
            ag = (ag * (p - 1) + g) / p;
            al = (al * (p - 1) + lss) / p;
            atr_s = (atr_s * (p - 1) + tr) / p;
            rsi[i] = al < 1e-12 ? 100 : 100.0 - 100.0 / (1.0 + ag / al);
            atr[i] = atr_s;
        }
        sma += b[i].c;
        if (i == 20) {
            ema[i] = sma / 20.0;
        } else if (i > 20) {
            ema[i] = (2.0 / 21.0) * b[i].c + (19.0 / 21.0) * ema[i - 1];
        } else {
            ema[i] = sma / i;
        }
    }
}

static const char *pick_setup(const Bar *b, int i, double rsi, double ema, double atr) {
    int inside = i > 0 && b[i].h <= b[i - 1].h && b[i].l >= b[i - 1].l;
    if (rsi < 40.0) {
        if (atr > 0 && fabs(b[i].c - ema) <= 0.5 * atr) return "trend pullback to the 20 EMA";
        if (i > 0 && b[i].l < b[i - 1].l && b[i].c > b[i - 1].l) return "failed breakdown reclaim";
        if (i > 0 && b[i].c > b[i - 1].h) return "opening range breakout";
        return "liquidity sweep then reverse";
    }
    if (rsi > 60.0) {
        double hi20 = b[i].h;
        int j0 = i - 19;
        if (j0 < 0) j0 = 0;
        for (int j = j0; j <= i; j++)
            if (b[j].h > hi20) hi20 = b[j].h;
        if (atr > 0 && hi20 - b[i].c <= 0.5 * atr) return "exhaustion at session highs";
        if (i > 0 && b[i].h < b[i - 1].h) return "failed to take prior day high";
        return "range fade at value area high";
    }
    if (inside) return "inside day";
    return "balanced auction";
}

static int rsi_rule(double rsi) {
    if (rsi < 40.0) return 'B';
    if (rsi > 60.0) return 'S';
    return 'H';
}

typedef struct {
    int pos;
    int ntr;
    int nw;
    double cash;
    double sh;
    double entry;
    double stop;
    double tgt;
    double peak;
    double maxdd;
    double eq;
} Book;

static void book_init(Book *k, double cash) {
    memset(k, 0, sizeof *k);
    k->cash = cash;
    k->peak = cash;
    k->eq = cash;
}

static double mark(const Book *k, double px) { return k->cash + k->sh * px; }

static void flatten(Book *k, double px, double fee) {
    if (k->pos == 0 || k->sh == 0) return;
    double pnl = k->sh * (px - k->entry);
    k->cash += k->sh * px;
    k->cash -= fabs(k->sh * px) * fee;
    k->ntr++;
    if (pnl > 0) k->nw++;
    k->sh = 0;
    k->pos = 0;
    k->entry = k->stop = k->tgt = 0;
}

static void enter(Book *k, int side, double px, double atr, double fee, double equity) {
    if (side == 0 || atr < 1e-9 || px <= 0) return;
    double risk = 0.01 * equity;
    double n = risk / atr;
    double notional = n * px;
    if (notional > equity) n = equity / px;
    if (n < 1e-9) return;
    if (side < 0) n = -n;
    k->cash -= n * px;
    k->cash -= fabs(n * px) * fee;
    k->sh = n;
    k->pos = side;
    k->entry = px;
    if (side > 0) {
        k->stop = px - atr;
        k->tgt = px + 2.0 * atr;
    } else {
        k->stop = px + atr;
        k->tgt = px - 2.0 * atr;
    }
}

static int hit_stop_target(const Book *k, double h, double l) {
    if (k->pos > 0) {
        if (l <= k->stop) return -1;
        if (h >= k->tgt) return 1;
    } else if (k->pos < 0) {
        if (h >= k->stop) return -1;
        if (l <= k->tgt) return 1;
    }
    return 0;
}

static void apply_bar(Book *k, double o, double h, double l, double c, int want, double atr, double fee) {
    if (k->pos) {
        int ht = hit_stop_target(k, h, l);
        if (ht < 0)
            flatten(k, k->stop, fee);
        else if (ht > 0)
            flatten(k, k->tgt, fee);
    }
    int side = want == 'B' ? 1 : want == 'S' ? -1 : 0;
    if (!k->pos && side) {
        double eq = k->eq > 1.0 ? k->eq : k->cash;
        enter(k, side, o, atr, fee, eq);
        if (k->pos) {
            int ht = hit_stop_target(k, h, l);
            if (ht < 0)
                flatten(k, k->stop, fee);
            else if (ht > 0)
                flatten(k, k->tgt, fee);
        }
    }
    k->eq = mark(k, c);
    if (k->eq > k->peak) k->peak = k->eq;
    double dd = k->peak > 0 ? (k->peak - k->eq) / k->peak : 0;
    if (dd > k->maxdd) k->maxdd = dd;
}

static void cmd_backtest(int argc, char **argv) {
    const char *ckpt = args(argc, argv, "--ckpt", "out/trader.bin");
    const char *vocab = args(argc, argv, "--vocab", "data/vocab.bin");
    const char *csv = args(argc, argv, "--csv", "data/spy.csv");
    const char *sym = args(argc, argv, "--symbol", "SPY");
    const char *outp = args(argc, argv, "--out", "");
    const char *newsdir = args(argc, argv, "--news-dir", "data/news");
    int max_bars = argi(argc, argv, "--max-bars", 750);
    int legacy = has(argc, argv, "--legacy");
    int no_setup = has(argc, argv, "--no-setup");
    double cash0 = atof(args(argc, argv, "--cash", "100000"));
    const double fee = 0.00025;
    Bar *bars = (Bar *)xmalloc((size_t)16384 * sizeof(Bar));
    int nall = load_ohlc(csv, bars, 16384);
    int start = 40;
    if (max_bars > 0 && nall - start > max_bars) {
        int keep = max_bars + start;
        int off = nall - keep;
        memmove(bars, bars + off, (size_t)keep * sizeof(Bar));
        nall = keep;
    }
    double *rsi = (double *)xmalloc((size_t)nall * sizeof(double));
    double *atr = (double *)xmalloc((size_t)nall * sizeof(double));
    double *ema = (double *)xmalloc((size_t)nall * sizeof(double));
    wilder_rsi_atr(bars, nall, rsi, atr, ema);
    int nmerge = 0, vocab_sz = 0;
    Merge *merges = load_merges(vocab, &nmerge, &vocab_sz);
    ActTok atok = make_act_tok(merges, nmerge);
    GPT g;
    load_ckpt(ckpt, &g);
    g.s.batch = 1;
    int T = g.s.seq;
    Acts a = acts_alloc(g.s);
    int *inp = (int *)xmalloc((size_t)T * sizeof(int));
    int *tgt = (int *)xmalloc((size_t)T * sizeof(int));
    Book m, r, bh;
    book_init(&m, cash0);
    book_init(&r, cash0);
    book_init(&bh, cash0);
    enter(&bh, 1, bars[start].o, bars[start].c * 0.01, fee, cash0);
    Bar *spy = NULL;
    int nspy = 0;
    if (strcmp(sym, "SPY") != 0) {
        spy = (Bar *)xmalloc((size_t)16384 * sizeof(Bar));
        FILE *sf = fopen("data/spy.csv", "r");
        if (sf) {
            fclose(sf);
            nspy = load_ohlc("data/spy.csv", spy, 16384);
        }
    }
    FILE *tf = NULL;
    if (outp[0]) {
        tf = fopen(outp, "w");
        if (!tf) die("open trades");
        fprintf(tf, "date\tclose\trsi\tatr\tsetup\tmodel\trsi_rule\teq_model\teq_rsi\teq_bh\n");
    }
    int agree = 0, scored = 0;
    double t0 = wall_now();
    for (int i = start; i < nall; i++) {
        char prompt[768];
        int iclose = (int)(bars[i].c + 0.5);
        int irsi = (int)(rsi[i] + 0.5);
        int iatr = (int)(atr[i] + 0.5);
        if (iatr < 1) iatr = 1;
        if (irsi < 1) irsi = 1;
        if (irsi > 99) irsi = 99;
        double r5 = 0.0;
        if (i >= 5 && bars[i - 5].c > 0) r5 = 100.0 * (bars[i].c / bars[i - 5].c - 1.0);
        double spy5 = 0.0;
        if (nspy > 0) {
            for (int k = 5; k < nspy; k++) {
                if (!strcmp(spy[k].d, bars[i].d) && spy[k - 5].c > 0) {
                    spy5 = 100.0 * (spy[k].c / spy[k - 5].c - 1.0);
                    break;
                }
            }
        }
        const char *setup = pick_setup(bars, i, rsi[i], ema[i], atr[i]);
        if (legacy && no_setup)
            snprintf(prompt, sizeof prompt, "%s 1d close %d RSI %d ATR %d. ACTION=", sym, iclose, irsi, iatr);
        else if (legacy)
            snprintf(prompt, sizeof prompt, "%s 1d close %d RSI %d ATR %d. Setup: %s. ACTION=", sym, iclose, irsi,
                     iatr, setup);
        else {
            char news[192];
            load_news_day(newsdir, bars[i].d, news, (int)sizeof news);
            snprintf(prompt, sizeof prompt,
                     "%s %s close %d RSI %d ATR %d r5d %+.1f vsSPY %+.1f. NEWS: %s ACTION=",
                     sym, bars[i].d, iclose, irsi, iatr, r5, strcmp(sym, "SPY") ? r5 - spy5 : 0.0, news);
        }
        int act = score_prompt(&g, &a, merges, nmerge, prompt, inp, tgt, atok);
        int rr = rsi_rule(rsi[i]);
        scored++;
        if (act == rr) agree++;
        double pxo = i + 1 < nall ? bars[i + 1].o : bars[i].c;
        double pxh = i + 1 < nall ? bars[i + 1].h : bars[i].h;
        double pxl = i + 1 < nall ? bars[i + 1].l : bars[i].l;
        double pxc = i + 1 < nall ? bars[i + 1].c : bars[i].c;
        apply_bar(&m, pxo, pxh, pxl, pxc, act, atr[i], fee);
        apply_bar(&r, pxo, pxh, pxl, pxc, rr, atr[i], fee);
        bh.eq = mark(&bh, pxc);
        if (bh.eq > bh.peak) bh.peak = bh.eq;
        double dd = bh.peak > 0 ? (bh.peak - bh.eq) / bh.peak : 0;
        if (dd > bh.maxdd) bh.maxdd = dd;
        if (tf) {
            const char *an = act == 'B' ? "BUY" : act == 'S' ? "SELL" : "HOLD";
            const char *rn = rr == 'B' ? "BUY" : rr == 'S' ? "SELL" : "HOLD";
            fprintf(tf, "%s\t%.2f\t%.1f\t%.2f\t%s\t%s\t%s\t%.2f\t%.2f\t%.2f\n", bars[i].d, bars[i].c, rsi[i],
                    atr[i], setup, an, rn, m.eq, r.eq, bh.eq);
        }
        if (scored % 50 == 0) {
            fprintf(stderr, "%s %d/%d %s eq %.0f\n", bars[i].d, scored, nall - start,
                    act == 'B' ? "BUY" : act == 'S' ? "SELL" : "HOLD", m.eq);
            fflush(stderr);
        }
    }
    flatten(&m, bars[nall - 1].c, fee);
    flatten(&r, bars[nall - 1].c, fee);
    flatten(&bh, bars[nall - 1].c, fee);
    double dt = wall_now() - t0;
    double mp = (m.eq - cash0) / cash0 * 100.0;
    double rp = (r.eq - cash0) / cash0 * 100.0;
    double bp = (bh.eq - cash0) / cash0 * 100.0;
    double wr = m.ntr ? 100.0 * m.nw / m.ntr : 0;
    printf("symbol %s bars %d scored %d sec %.1f\n", sym, nall, scored, dt);
    printf("model  ret %+.2f%%  trades %d  win %.0f%%  maxdd %.1f%%  end %.0f\n", mp, m.ntr, wr, m.maxdd * 100.0,
           m.eq);
    printf("rsi    ret %+.2f%%  trades %d  win %.0f%%  maxdd %.1f%%  end %.0f\n", rp, r.ntr,
           r.ntr ? 100.0 * r.nw / r.ntr : 0, r.maxdd * 100.0, r.eq);
    printf("bh     ret %+.2f%%  maxdd %.1f%%  end %.0f\n", bp, bh.maxdd * 100.0, bh.eq);
    printf("agree_rsi %.1f%%\n", scored ? 100.0 * agree / scored : 0);
    if (tf) fclose(tf);
    (void)vocab_sz;
}

static const char *PICK_SYMS[] = {
    "SPY", "QQQ", "IWM", "TLT", "GLD", "SMH", "XLF", "AAPL", "MSFT", "NVDA",
    "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "JPM", "XOM", "NFLX", "COST", NULL};

static int find_date(const Bar *b, int n, const char *day) {
    for (int i = 0; i < n; i++)
        if (!strcmp(b[i].d, day)) return i;
    return -1;
}

static void cmd_picks(int argc, char **argv) {
    const char *ckpt = args(argc, argv, "--ckpt", "out/trader.bin");
    const char *vocab = args(argc, argv, "--vocab", "data/vocab.bin");
    const char *datadir = args(argc, argv, "--data-dir", "data");
    const char *newsdir = args(argc, argv, "--news-dir", "data/news");
    const char *day = args(argc, argv, "--date", "");
    char asof[12];
    Bar spy[4096];
    char spypath[512];
    snprintf(spypath, sizeof spypath, "%s/spy.csv", datadir);
    int nspy = load_ohlc(spypath, spy, 4096);
    if (!day[0]) {
        snprintf(asof, sizeof asof, "%s", spy[nspy - 1].d);
        day = asof;
    }
    char news[192];
    load_news_day(newsdir, day, news, (int)sizeof news);
    int nmerge = 0, vocab_sz = 0;
    Merge *merges = load_merges(vocab, &nmerge, &vocab_sz);
    ActTok atok = make_act_tok(merges, nmerge);
    GPT g;
    load_ckpt(ckpt, &g);
    g.s.batch = 1;
    int T = g.s.seq;
    Acts a = acts_alloc(g.s);
    int *inp = (int *)xmalloc((size_t)T * sizeof(int));
    int *tgt = (int *)xmalloc((size_t)T * sizeof(int));
    printf("asof %s\nNEWS %s\n", day, news);
    char longs[256] = {0};
    char shorts[256] = {0};
    for (int s = 0; PICK_SYMS[s]; s++) {
        const char *sym = PICK_SYMS[s];
        char low[16];
        snprintf(low, sizeof low, "%s", sym);
        for (int j = 0; low[j]; j++)
            if (low[j] >= 'A' && low[j] <= 'Z') low[j] = (char)(low[j] - 'A' + 'a');
        char path[512];
        snprintf(path, sizeof path, "%s/%s.csv", datadir, low);
        FILE *tf = fopen(path, "r");
        if (!tf) continue;
        fclose(tf);
        Bar *bars = (Bar *)xmalloc((size_t)16384 * sizeof(Bar));
        int nall = load_ohlc(path, bars, 16384);
        int i = find_date(bars, nall, day);
        if (i < 40) {
            free(bars);
            continue;
        }
        double *rsi = (double *)xmalloc((size_t)nall * sizeof(double));
        double *atr = (double *)xmalloc((size_t)nall * sizeof(double));
        double *ema = (double *)xmalloc((size_t)nall * sizeof(double));
        wilder_rsi_atr(bars, nall, rsi, atr, ema);
        int iclose = (int)(bars[i].c + 0.5);
        int irsi = (int)(rsi[i] + 0.5);
        int iatr = (int)(atr[i] + 0.5);
        if (iatr < 1) iatr = 1;
        double r5 = 0.0;
        if (i >= 5 && bars[i - 5].c > 0) r5 = 100.0 * (bars[i].c / bars[i - 5].c - 1.0);
        double spy5 = 0.0;
        int si = find_date(spy, nspy, day);
        if (si >= 5 && spy[si - 5].c > 0) spy5 = 100.0 * (spy[si].c / spy[si - 5].c - 1.0);
        double vs = strcmp(sym, "SPY") ? r5 - spy5 : 0.0;
        char prompt[768];
        snprintf(prompt, sizeof prompt,
                 "%s %s close %d RSI %d ATR %d r5d %+.1f vsSPY %+.1f. NEWS: %s ACTION=",
                 sym, day, iclose, irsi, iatr, r5, vs, news);
        int ids[4096];
        int ntok = encode_prompt(prompt, merges, nmerge, ids, T);
        if (ntok >= T) ntok = T - 1;
        memset(inp, 0, (size_t)T * sizeof(int));
        memset(tgt, 0, (size_t)T * sizeof(int));
        for (int t = 0; t < ntok; t++) inp[t] = ids[t];
        gpt_forward_backward(&g, &a, inp, tgt, 0);
        int pos = ntok - 1;
        const float *lg = a.logits + (size_t)pos * (size_t)g.s.vocab;
        float lb, ls, lh;
        int act = action_from_logits(lg, g.s.vocab, atok, &lb, &ls, &lh);
        const char *name = act == 'B' ? "BUY" : act == 'S' ? "SELL" : "HOLD";
        printf("%-5s %s  B %.2f S %.2f H %.2f  close %.2f RSI %.0f r5d %+.1f\n", sym, name, lb, ls, lh,
               bars[i].c, rsi[i], r5);
        if (act == 'B') {
            size_t n = strlen(longs);
            snprintf(longs + n, sizeof longs - n, "%s%s", n ? " " : "", sym);
        }
        if (act == 'S') {
            size_t n = strlen(shorts);
            snprintf(shorts + n, sizeof shorts - n, "%s%s", n ? " " : "", sym);
        }
        free(bars);
        free(rsi);
        free(atr);
        free(ema);
    }
    printf("PICKS long %s\nPICKS short %s\n", longs[0] ? longs : "(none)", shorts[0] ? shorts : "(none)");
    (void)vocab_sz;
}

static void cmd_train(int argc, char **argv) {
    const char *tokpath = args(argc, argv, "--tokens", "data/tokens.bin");
    const char *out = args(argc, argv, "--out", "out/model.bin");
    int steps = argi(argc, argv, "--steps", 2000);
    int batch = argi(argc, argv, "--batch", 8);
    int seq = argi(argc, argv, "--seq", 256);
    int n_layer = argi(argc, argv, "--n-layer", 8);
    int n_head = argi(argc, argv, "--n-head", 6);
    int n_embd = argi(argc, argv, "--n-embd", 384);
    float lr = (float)atof(args(argc, argv, "--lr", "3e-4"));
    float wd = (float)atof(args(argc, argv, "--wd", "0.01"));
    int overfit = has(argc, argv, "--overfit");
    uint32_t ntok = 0;
    uint16_t *toks = load_tokens(tokpath, &ntok);
    int vmax = 0;
    for (uint32_t i = 0; i < ntok; i++)
        if (toks[i] > vmax) vmax = toks[i];
    int vocab = vmax + 1;
    if (vocab < 258) vocab = 258;
    Shape sh = {n_layer, n_head, n_embd, vocab, seq, batch};
    if (n_embd % n_head) die("n_embd must be divisible by n_head");
    GPT g;
    gpt_init(&g, sh, 42);
    Acts acts = acts_alloc(sh);
    int *inp = (int *)xmalloc((size_t)batch * seq * sizeof(int));
    int *tgt = (int *)xmalloc((size_t)batch * seq * sizeof(int));
    fprintf(stderr, "params=%.2fM vocab=%d B=%d T=%d C=%d L=%d tokens=%u\n", g.nparams / 1e6, vocab, batch,
            seq, n_embd, n_layer, ntok);
    double t0 = wall_now();
    float ema = 0.0f;
    uint32_t cursor = 0;
    for (int step = 1; step <= steps; step++) {
        for (int i = 0; i < batch * seq; i++) {
            if (overfit) {
                uint32_t j = (uint32_t)i % (ntok - 1);
                inp[i] = toks[j];
                tgt[i] = toks[j + 1];
            } else {
                if (cursor + 1 >= ntok) cursor = 0;
                inp[i] = toks[cursor];
                tgt[i] = toks[cursor + 1];
                cursor++;
            }
        }
        float loss = gpt_forward_backward(&g, &acts, inp, tgt, 1);
        float lnow = lr;
        if (step < 50) lnow = lr * (float)step / 50.0f;
        adamw(&g, lnow, wd, step);
        ema = step == 1 ? loss : 0.95f * ema + 0.05f * loss;
        if (step == 1 || step % 20 == 0 || step == steps) {
            double dt = wall_now() - t0;
            double tps = (double)step * batch * seq / (dt > 1e-6 ? dt : 1e-6);
            fprintf(stderr, "step %d/%d loss %.4f ema %.4f tok/s %.0f lr %.2e\n", step, steps, loss, ema, tps,
                    lnow);
            fflush(stderr);
        }
        if (step % 200 == 0 || step == steps) save_ckpt(out, &g);
    }
    fprintf(stderr, "wrote %s\n", out);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage:\n"
                "  oxidize-training train --tokens data/tokens.bin --out out/model.bin\n"
                "  oxidize-training sample --ckpt out/model.bin --vocab data/vocab.bin --prompt '...'\n"
                "  oxidize-training score --ckpt out/trader.bin --vocab data/vocab.bin < prompts.txt\n"
                "  oxidize-training backtest --ckpt out/trader.bin --vocab data/vocab.bin --csv data/spy.csv --symbol SPY\n"
                "  oxidize-training picks --ckpt out/trader.bin --vocab data/vocab.bin --date 2026-09-11\n"
                "  oxidize-training complete --ckpt out/model.bin --vocab data/vocab.bin --tokens 16 < prompts.txt\n"
                "  If the prompt contains ACTION=, the first token is B, S, or H. Pass --no-force-action to disable.\n");
        return 1;
    }
    if (!strcmp(argv[1], "train")) {
        cmd_train(argc, argv);
        return 0;
    }
    if (!strcmp(argv[1], "sample")) {
        cmd_sample(argc, argv);
        return 0;
    }
    if (!strcmp(argv[1], "complete")) {
        cmd_complete(argc, argv);
        return 0;
    }
    if (!strcmp(argv[1], "score")) {
        cmd_score(argc, argv);
        return 0;
    }
    if (!strcmp(argv[1], "backtest")) {
        cmd_backtest(argc, argv);
        return 0;
    }
    if (!strcmp(argv[1], "picks")) {
        cmd_picks(argc, argv);
        return 0;
    }
    die("unknown command");
}
