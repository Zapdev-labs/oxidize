/*
 * oc_vkbench.c — Vulkan kernel benchmark and CPU parity check.
 *
 *   make oc-vkbench VULKAN_INCLUDE=~/vulkan-prefix/include
 *   ./oc-vkbench [bw] [disp] [mm] [moe]      (no argument: all sections)
 *
 * bw    read bandwidth per memory kind
 * disp  per-dispatch cost with barriers, and submit round-trip latency
 * mm    NVFP4 x int8 matmul vs the CPU kernel, GMAC/s by column count
 * moe   Inkling-sized grouped expert matmul (gate_up + situ, down)
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/nvfp4.h"
#include "oxidize/vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint32_t SPV_QUANT[] =
#include "../src/backends/vk_shaders/quant_act.spv.h"
    ;
static const uint32_t SPV_MM[] =
#include "../src/backends/vk_shaders/nvfp4_mm.spv.h"
    ;
static const uint32_t SPV_MV[] =
#include "../src/backends/vk_shaders/nvfp4_mv.spv.h"
    ;
static const uint32_t SPV_BW[] =
#include "../src/backends/vk_shaders/bw.spv.h"
    ;

typedef struct {
    uint64_t x, q, d, rows;
    uint32_t n, cols, x_stride, q_stride, d_stride, pad;
} QuantPush;

typedef struct {
    uint64_t w, gscale, aq, ad, items, cidx, y, mat_bytes;
    uint32_t rows, cols, n, y_stride, aq_stride, ad_stride, post, ff;
    float    beta, lbeta, cap, pad;
} MmPush;

typedef struct {
    uint64_t src, dst;
    uint32_t n, pad;
} BwPush;

#define CK(x) do {                                                                   \
        OcError e_ = (x);                                                            \
        if (e_ != OC_OK) {                                                           \
            fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #x, (int)e_);   \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static OcVk *vk;
static OcVkPipe *p_quant, *p_mm, *p_mv[9], *p_mv_situ[9], *p_bw;

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static float rndf(void) { return (float)(rnd() & 0xFFFFFF) / 8388608.0f - 1.0f; }

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_nvfp4(uint8_t *w, size_t bytes)
{
    for (size_t b = 0; b < bytes / 36; b++) {
        uint8_t *blk = w + b * 36;
        for (int s = 0; s < 4; s++) blk[s] = (uint8_t)(0x30 + rnd() % 16);
        for (int i = 4; i < 36; i += 4) { uint32_t r = rnd(); memcpy(blk + i, &r, 4); }
    }
}

static double gpu_us(void) { return (oc_vk_stamp_ns(vk, 1) - oc_vk_stamp_ns(vk, 0)) * 1e-3; }

static void quant(const OcVkBuf *x, size_t n, size_t cols, const OcVkBuf *q, const OcVkBuf *d, uint64_t rows)
{
    QuantPush qp = { x->addr, q->addr, d->addr, rows, (uint32_t)n, (uint32_t)cols, (uint32_t)cols, (uint32_t)cols,
                     (uint32_t)(cols / 64), 0 };
    oc_vk_dispatch(vk, p_quant, &qp, (uint32_t)((n * cols / 64 + 63) / 64), 1, 1);
}

static void sec_bw(void)
{
    static const char *names[3] = { "device", "host", "stream" };
    const size_t bytes = (size_t)1 << 30;
    for (int k = 0; k < 3; k++) {
        OcVkBuf b, dst;
        if (oc_vk_buf_new(vk, bytes, (OcVkMemKind)k, &b) != OC_OK) { printf("bw %-6s: alloc failed\n", names[k]); continue; }
        CK(oc_vk_buf_new(vk, 64, OC_VK_MEM_HOST, &dst));
        CK(oc_vk_begin(vk));
        oc_vk_fill(vk, &b, 0, bytes, 0x01020304u);
        oc_vk_barrier(vk);
        BwPush bp = { b.addr, dst.addr, (uint32_t)(bytes / 16), 0 };
        oc_vk_dispatch(vk, p_bw, &bp, 768, 1, 1);
        oc_vk_barrier(vk);
        oc_vk_stamp(vk, 0);
        for (int r = 0; r < 4; r++) { oc_vk_dispatch(vk, p_bw, &bp, 768, 1, 1); oc_vk_barrier(vk); }
        oc_vk_stamp(vk, 1);
        CK(oc_vk_submit(vk));
        printf("bw %-6s: %6.1f GB/s\n", names[k], 4.0 * (double)bytes / (gpu_us() * 1e3));
        oc_vk_buf_free(vk, &b);
        oc_vk_buf_free(vk, &dst);
    }
}

static void sec_disp(void)
{
    OcVkBuf b, dst;
    CK(oc_vk_buf_new(vk, 4096, OC_VK_MEM_HOST, &b));
    CK(oc_vk_buf_new(vk, 64, OC_VK_MEM_HOST, &dst));
    memset(b.map, 0, 4096);
    BwPush bp = { b.addr, dst.addr, 256, 0 };
    const int N = 2000;
    CK(oc_vk_begin(vk));
    oc_vk_stamp(vk, 0);
    for (int i = 0; i < N; i++) { oc_vk_dispatch(vk, p_bw, &bp, 1, 1, 1); oc_vk_barrier(vk); }
    oc_vk_stamp(vk, 1);
    const double t0 = now();
    CK(oc_vk_submit(vk));
    const double wall = now() - t0;
    printf("disp: %.2f us per dispatch+barrier (gpu), %.2f us (wall incl. submit)\n", gpu_us() / N, wall * 1e6 / N);
    const int S = 200;
    const double t1 = now();
    for (int i = 0; i < S; i++) {
        CK(oc_vk_begin(vk));
        oc_vk_dispatch(vk, p_bw, &bp, 1, 1, 1);
        CK(oc_vk_submit(vk));
    }
    printf("disp: %.1f us per submit+wait round trip\n", (now() - t1) * 1e6 / S);
    oc_vk_buf_free(vk, &b);
    oc_vk_buf_free(vk, &dst);
}

/* y[n][rows] on the CPU from f32 x[n][cols] */
static void cpu_ref(const uint8_t *w, size_t rows, size_t cols, const float *x, size_t n, float *y)
{
    int8_t *q = aligned_alloc(64, cols);
    float *d = malloc(cols / 64 * sizeof(float));
    for (size_t t = 0; t < n; t++) {
        OcNvfp4Act a = { q, d };
        oc_nvfp4_quantize_act(x + t * cols, cols, a);
        oc_nvfp4_gemm_rows(w, cols, 0, rows, 1.0f, &a, 1, y + t * rows, rows);
    }
    free(q);
    free(d);
}

static double rel_err(const float *a, const float *ref, size_t n)
{
    double md = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - ref[i]);
        if (!(d <= md)) md = d;
        if (fabs(ref[i]) > mr) mr = fabs(ref[i]);
    }
    return mr > 0 ? md / mr : md;
}

static void mm_push(MmPush *mp, const OcVkBuf *w, size_t off, size_t rows, size_t cols, size_t n, const OcVkBuf *q,
                    const OcVkBuf *d, const OcVkBuf *y)
{
    *mp = (MmPush){ .w = w->addr + off, .aq = q->addr, .ad = d->addr, .y = y->addr, .rows = (uint32_t)rows,
                    .cols = (uint32_t)cols, .n = (uint32_t)n, .y_stride = (uint32_t)rows, .aq_stride = (uint32_t)cols,
                    .ad_stride = (uint32_t)(cols / 64) };
}

static void run_mm(const MmPush *mp, size_t rows, size_t n)
{
    if (n <= 8) oc_vk_dispatch(vk, p_mv[n], mp, (uint32_t)((rows + 31) / 32), 1, 1);
    else oc_vk_dispatch(vk, p_mm, mp, (uint32_t)((rows + 63) / 64), (uint32_t)((n + 127) / 128), 1);
}

static void sec_mm(void)
{
    static const size_t shapes[][2] = { { 2304, 2304 }, { 2304, 1152 }, { 2304, 3072 }, { 8192, 2304 } };
    static const size_t ns[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    for (size_t si = 0; si < sizeof shapes / sizeof shapes[0]; si++) {
        const size_t rows = shapes[si][0], cols = shapes[si][1];
        const size_t mb = oc_nvfp4_row_bytes(cols) * rows;
        size_t n_mat = ((size_t)384 << 20) / mb;
        if (n_mat < 1) n_mat = 1;
        uint8_t *host = malloc(mb);
        fill_nvfp4(host, mb);
        OcVkBuf w, x, q, d, y;
        CK(oc_vk_buf_new(vk, mb * n_mat, OC_VK_MEM_DEVICE, &w));
        uint8_t *planar = malloc(mb);
        oc_nvfp4_planar(host, rows, cols, planar);
        for (size_t i = 0; i < n_mat; i++) memcpy((uint8_t *)w.map + i * mb, planar, mb);
        free(planar);
        CK(oc_vk_buf_new(vk, 512 * cols * 4, OC_VK_MEM_HOST, &x));
        CK(oc_vk_buf_new(vk, 512 * cols, OC_VK_MEM_DEVICE, &q));
        CK(oc_vk_buf_new(vk, 512 * cols / 64 * 4, OC_VK_MEM_DEVICE, &d));
        CK(oc_vk_buf_new(vk, 512 * rows * 4, OC_VK_MEM_DEVICE, &y));
        float *xs = x.map;
        for (size_t i = 0; i < 512 * cols; i++) xs[i] = rndf();
        float *ref = malloc(512 * rows * sizeof(float));
        printf("mm %zux%zu (%.1f MB/mat, %zu mats)\n", rows, cols, mb / 1e6, n_mat);
        for (size_t ni = 0; ni < sizeof ns / sizeof ns[0]; ni++) {
            const size_t n = ns[ni];
            MmPush mp;
            mm_push(&mp, &w, 0, rows, cols, n, &q, &d, &y);
            CK(oc_vk_begin(vk));
            quant(&x, n, cols, &q, &d, 0);
            oc_vk_barrier(vk);
            run_mm(&mp, rows, n);
            CK(oc_vk_submit(vk));
            const size_t nc = n < 16 ? n : 16;
            cpu_ref(host, rows, cols, xs, nc, ref);
            const double err = rel_err(y.map, ref, nc * rows);

            const int R = n <= 8 ? (int)n_mat * 2 : 10;
            CK(oc_vk_begin(vk));
            oc_vk_stamp(vk, 0);
            for (int r = 0; r < R; r++) {
                mm_push(&mp, &w, (size_t)r % n_mat * mb, rows, cols, n, &q, &d, &y);
                run_mm(&mp, rows, n);
            }
            oc_vk_stamp(vk, 1);
            CK(oc_vk_submit(vk));
            const double us = gpu_us() / R;
            printf("  n=%-4zu %s %9.1f us  %7.1f GMAC/s  %6.1f GB/s  err %.2e\n", n, n <= 8 ? "mv" : "mm", us,
                   (double)rows * cols * n / (us * 1e3), (double)mb / (us * 1e3), err);
        }
        free(host);
        free(ref);
        oc_vk_buf_free(vk, &w); oc_vk_buf_free(vk, &x); oc_vk_buf_free(vk, &q);
        oc_vk_buf_free(vk, &d); oc_vk_buf_free(vk, &y);
    }
}

static float situ_ref(float g, float u, float beta, float lb)
{
    const float ga = beta * tanhf(g / beta) / (1.0f + expf(-g));
    return ga * (lb > 0.0f ? lb * tanhf(u / lb) : u);
}

static void sec_moe(void)
{
    const size_t E = 2304, FF = 1152, NE = 128, K = 8;
    const size_t gub = oc_nvfp4_row_bytes(E) * 2 * FF, dnb = oc_nvfp4_row_bytes(FF) * E;
    const float beta = 4.0f, lbeta = 25.0f;
    uint8_t *hgu = malloc(gub * NE), *hdn = malloc(dnb * NE);
    fill_nvfp4(hgu, gub * NE);
    fill_nvfp4(hdn, dnb * NE);
    OcVkBuf wgu, wdn, x, q, d, act, q2, d2, y, items, cidx;
    CK(oc_vk_buf_new(vk, gub * NE, OC_VK_MEM_DEVICE, &wgu));
    CK(oc_vk_buf_new(vk, dnb * NE, OC_VK_MEM_DEVICE, &wdn));
    for (size_t e = 0; e < NE; e++) {
        oc_nvfp4_planar(hgu + e * gub, 2 * FF, E, (uint8_t *)wgu.map + e * gub);
        oc_nvfp4_planar(hdn + e * dnb, E, FF, (uint8_t *)wdn.map + e * dnb);
    }
    const size_t NMAX = 512, PMAX = NMAX * K;
    CK(oc_vk_buf_new(vk, NMAX * E * 4, OC_VK_MEM_HOST, &x));
    CK(oc_vk_buf_new(vk, NMAX * E, OC_VK_MEM_DEVICE, &q));
    CK(oc_vk_buf_new(vk, NMAX * E / 64 * 4, OC_VK_MEM_DEVICE, &d));
    CK(oc_vk_buf_new(vk, PMAX * FF * 4, OC_VK_MEM_DEVICE, &act));
    CK(oc_vk_buf_new(vk, PMAX * FF, OC_VK_MEM_DEVICE, &q2));
    CK(oc_vk_buf_new(vk, PMAX * FF / 64 * 4, OC_VK_MEM_DEVICE, &d2));
    CK(oc_vk_buf_new(vk, PMAX * E * 4, OC_VK_MEM_DEVICE, &y));
    CK(oc_vk_buf_new(vk, (PMAX + NE) * 16, OC_VK_MEM_HOST, &items));
    CK(oc_vk_buf_new(vk, PMAX * 4, OC_VK_MEM_HOST, &cidx));
    float *xs = x.map;
    for (size_t i = 0; i < NMAX * E; i++) xs[i] = rndf();
    static const size_t ns[] = { 1, 2, 4, 16, 64, 128, 256, 512 };
    int32_t *sel = malloc(NMAX * K * sizeof(int32_t));
    int32_t *pair_tok = malloc(PMAX * sizeof(int32_t)), *pair_ex = malloc(PMAX * sizeof(int32_t));
    for (size_t ni = 0; ni < sizeof ns / sizeof ns[0]; ni++) {
        const size_t n = ns[ni];
        for (size_t t = 0; t < n; t++)
            for (size_t k = 0; k < K; k++) {
                int32_t e;
                bool dup;
                do {
                    e = (int32_t)(rnd() % NE);
                    dup = false;
                    for (size_t j = 0; j < k; j++) dup |= sel[t * K + j] == e;
                } while (dup);
                sel[t * K + k] = e;
            }
        /* pairs grouped by expert, split into items of at most BN (128) or NV (8) columns */
        const bool small = n * K <= 8 * NE && n <= 2;
        const size_t cap = small ? 8 : 128;
        int32_t *it = items.map, *ci = cidx.map;
        size_t np = 0, ni_ = 0, maxc = 1;
        for (size_t e = 0; e < NE; e++) {
            const size_t p0 = np;
            for (size_t t = 0; t < n; t++)
                for (size_t k = 0; k < K; k++)
                    if (sel[t * K + k] == (int32_t)e) { ci[np] = (int32_t)t; pair_tok[np] = (int32_t)t; pair_ex[np++] = (int32_t)e; }
            for (size_t f = p0; f < np; f += cap) {
                it[ni_ * 4 + 0] = (int32_t)e;
                it[ni_ * 4 + 1] = (int32_t)f;
                it[ni_ * 4 + 2] = (int32_t)(np - f < cap ? np - f : cap);
                it[ni_ * 4 + 3] = 0;
                if ((size_t)it[ni_ * 4 + 2] > maxc) maxc = (size_t)it[ni_ * 4 + 2];
                ni_++;
            }
        }
        MmPush gu = { .w = wgu.addr, .aq = q.addr, .ad = d.addr, .items = items.addr, .cidx = cidx.addr, .y = act.addr,
                      .mat_bytes = gub, .rows = (uint32_t)(2 * FF), .cols = (uint32_t)E, .y_stride = (uint32_t)FF,
                      .aq_stride = (uint32_t)E, .ad_stride = (uint32_t)(E / 64), .post = 1, .ff = (uint32_t)FF,
                      .beta = beta, .lbeta = lbeta };
        MmPush dn = { .w = wdn.addr, .aq = q2.addr, .ad = d2.addr, .items = items.addr, .y = y.addr, .mat_bytes = dnb,
                      .rows = (uint32_t)E, .cols = (uint32_t)FF, .y_stride = (uint32_t)E, .aq_stride = (uint32_t)FF,
                      .ad_stride = (uint32_t)(FF / 64) };
        const int R = 3;
        CK(oc_vk_begin(vk));
        quant(&x, n, E, &q, &d, 0);
        oc_vk_barrier(vk);
        oc_vk_stamp(vk, 0);
        for (int r = 0; r < R; r++) {
            if (small) oc_vk_dispatch(vk, p_mv_situ[maxc], &gu, (uint32_t)((FF + 31) / 32), (uint32_t)ni_, 1);
            else oc_vk_dispatch(vk, p_mm, &gu, (uint32_t)((FF + 31) / 32), (uint32_t)ni_, 1);
            oc_vk_barrier(vk);
        }
        oc_vk_stamp(vk, 1);
        quant(&act, np, FF, &q2, &d2, 0);
        oc_vk_barrier(vk);
        oc_vk_stamp(vk, 2);
        for (int r = 0; r < R; r++) {
            if (small) oc_vk_dispatch(vk, p_mv[maxc], &dn, (uint32_t)((E + 31) / 32), (uint32_t)ni_, 1);
            else oc_vk_dispatch(vk, p_mm, &dn, (uint32_t)((E + 63) / 64), (uint32_t)ni_, 1);
            oc_vk_barrier(vk);
        }
        oc_vk_stamp(vk, 3);
        CK(oc_vk_submit(vk));
        const double t_gu = (oc_vk_stamp_ns(vk, 1) - oc_vk_stamp_ns(vk, 0)) * 1e-3 / R;
        const double t_dn = (oc_vk_stamp_ns(vk, 3) - oc_vk_stamp_ns(vk, 2)) * 1e-3 / R;

        /* parity on the first pairs */
        double eg = 0.0, ed = 0.0;
        const size_t chk = np < 6 ? np : 6;
        float *gref = malloc(2 * FF * sizeof(float)), *aref = malloc(FF * sizeof(float)), *dref = malloc(E * sizeof(float));
        for (size_t pi = 0; pi < chk; pi++) {
            const size_t pp = pi * (np / chk);
            cpu_ref(hgu + (size_t)pair_ex[pp] * gub, 2 * FF, E, xs + (size_t)pair_tok[pp] * E, 1, gref);
            for (size_t r = 0; r < FF; r++) aref[r] = situ_ref(gref[r], gref[FF + r], beta, lbeta);
            const double e1 = rel_err((const float *)act.map + pp * FF, aref, FF);
            cpu_ref(hdn + (size_t)pair_ex[pp] * dnb, E, FF, (const float *)act.map + pp * FF, 1, dref);
            const double e2 = rel_err((const float *)y.map + pp * E, dref, E);
            if (e1 > eg) eg = e1;
            if (e2 > ed) ed = e2;
        }
        free(gref); free(aref); free(dref);
        const double mac = (double)np * (2.0 * FF * E + (double)E * FF);
        printf("moe n=%-4zu pairs %-5zu items %-4zu gate_up %8.1f us  down %8.1f us  %7.1f GMAC/s  %7.1f tok/s(moe only)  err %.1e/%.1e\n",
               n, np, ni_, t_gu, t_dn, mac / ((t_gu + t_dn) * 1e3), (double)n / ((t_gu + t_dn) * 48e-6), eg, ed);
    }
    free(sel); free(pair_tok); free(pair_ex); free(hgu); free(hdn);
    OcVkBuf *all[] = { &wgu, &wdn, &x, &q, &d, &act, &q2, &d2, &y, &items, &cidx };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) oc_vk_buf_free(vk, all[i]);
}

static bool want(int argc, char **argv, const char *s)
{
    if (argc < 2) return true;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], s)) return true;
    return false;
}

int main(int argc, char **argv)
{
    CK(oc_vk_init(&vk));
    const OcVkInfo *in = oc_vk_info(vk);
    printf("device: %s  subgroup %u  heaps: device %.2f GB, host %.2f GB  max alloc %.2f GB  ts %.2f ns\n", in->name,
           in->subgroup_size, in->heap_device / 1e9, in->heap_host / 1e9, in->max_alloc / 1e9, in->timestamp_ns);
    CK(oc_vk_pipe_new(vk, SPV_QUANT, sizeof SPV_QUANT, sizeof(QuantPush), NULL, 0, &p_quant));
    CK(oc_vk_pipe_new(vk, SPV_MM, sizeof SPV_MM, sizeof(MmPush), NULL, 0, &p_mm));
    for (uint32_t nv = 1; nv <= 8; nv++) {
        const uint32_t s0[2] = { nv, 0 }, s1[2] = { nv, 1 };
        CK(oc_vk_pipe_new(vk, SPV_MV, sizeof SPV_MV, sizeof(MmPush), s0, 2, &p_mv[nv]));
        CK(oc_vk_pipe_new(vk, SPV_MV, sizeof SPV_MV, sizeof(MmPush), s1, 2, &p_mv_situ[nv]));
    }
    CK(oc_vk_pipe_new(vk, SPV_BW, sizeof SPV_BW, sizeof(BwPush), NULL, 0, &p_bw));
    if (want(argc, argv, "bw")) sec_bw();
    if (want(argc, argv, "disp")) sec_disp();
    if (want(argc, argv, "mm")) sec_mm();
    if (want(argc, argv, "moe")) sec_moe();
    oc_vk_pipe_free(vk, p_quant); oc_vk_pipe_free(vk, p_mm); oc_vk_pipe_free(vk, p_bw);
    for (int nv = 1; nv <= 8; nv++) { oc_vk_pipe_free(vk, p_mv[nv]); oc_vk_pipe_free(vk, p_mv_situ[nv]); }
    oc_vk_free(vk);
    return 0;
}
