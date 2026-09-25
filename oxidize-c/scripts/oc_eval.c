/* oc_eval.c — token-id driven eval/bench harness (`make oc-eval`).
 *
 * Parity mode (default): reads one case per stdin line,
 *     NAME N_GEN ID ID ID ...
 * prefills the prompt ids, then greedily generates N_GEN tokens and prints
 * one JSON object per case with each step's argmax id and top-5 logprobs.
 *
 *   oc-eval MODEL.gguf [--threads N] [--ctx N] [--kv f32|q8]
 *                      [--no-prefill]   (feed the prompt token by token)
 *                      [--bench PP TG --reps R]  (llama-bench style pp/tg)
 *
 * Bench mode prefills PP tokens (one timed oc_llama_prefill call) and then
 * decodes TG tokens (timed), R times, and prints tokens/second for each.
 *
 * K2 MTP (speculative decoding with the nextn head):
 *   --mtp-model SIDECAR.gguf  overlay an mtp-*.gguf head onto the base model
 *   --mtp K                   decode with oc_llama_mtp_step, K drafts/step
 *                             (0 = batched-verify path with no drafts; the
 *                             exact-equality reference for K >= 1)
 *   --mtp-top5                with --mtp 0: top-5 of every step's logits
 *   --mtp-golden FILE.bin     head parity vs golden_mtp.npz (converted by
 *                             scripts/mtp_golden_to_bin.py), then exit
 * With --mtp, parity mode also prints per-case acceptance statistics.
 */
#define _POSIX_C_SOURCE 200809L  /* clock_gettime */

#include "oxidize/llama.h"
#include "oxidize/parallel.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void top5(const float *logits, uint32_t n, uint32_t *ids, double *lp)
{
    double mx = -INFINITY;
    for (uint32_t i = 0; i < n; ++i)
        if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) sum += exp((double)logits[i] - mx);
    double lse = mx + log(sum);
    for (int k = 0; k < 5; ++k) { ids[k] = 0; lp[k] = -INFINITY; }
    for (uint32_t i = 0; i < n; ++i) {
        double v = (double)logits[i] - lse;
        if (v <= lp[4]) continue;
        int k = 4;
        while (k > 0 && lp[k - 1] < v) {
            lp[k] = lp[k - 1];
            ids[k] = ids[k - 1];
            --k;
        }
        lp[k] = v;
        ids[k] = i;
    }
}

static int run_bench(OcLlamaModel *model, OcKvCacheType kv, uint32_t pp,
                     uint32_t tg, int reps)
{
    uint32_t vocab = model->cfg.vocab_size;
    float *logits = malloc((size_t)vocab * sizeof(float));
    uint32_t *toks = malloc((size_t)(pp ? pp : 1) * sizeof(uint32_t));
    if (!logits || !toks) return 1;
    srand(1234);
    for (uint32_t i = 0; i < pp; ++i) toks[i] = 1000u + (uint32_t)(rand() % 20000);
    OcLlamaSession sess;
    if (oc_llama_session_init_kv(model, &sess, kv) != OC_OK) return 1;
    for (int r = 0; r < reps; ++r) {
        oc_llama_session_reset(&sess);
        double t0 = now_s();
        if (pp > 0 && oc_llama_prefill(&sess, toks, pp, 0, logits) != OC_OK) {
            fprintf(stderr, "prefill failed\n");
            return 1;
        }
        double t1 = now_s();
        uint32_t tok = 1000;
        if (pp == 0) oc_llama_forward(&sess, tok, logits);
        double t2 = now_s();
        for (uint32_t i = 0; i < tg; ++i) {
            uint32_t best = 0;
            for (uint32_t v = 1; v < vocab; ++v)
                if (logits[v] > logits[best]) best = v;
            if (oc_llama_forward(&sess, best, logits) != OC_OK) {
                fprintf(stderr, "forward failed\n");
                return 1;
            }
        }
        double t3 = now_s();
        printf("{\"rep\":%d,\"pp\":%u,\"pp_tps\":%.3f,\"tg\":%u,\"tg_tps\":%.3f}\n",
               r, pp, pp ? pp / (t1 - t0) : 0.0, tg, tg ? tg / (t3 - t2) : 0.0);
        fflush(stdout);
    }
    oc_llama_session_free(&sess);
    free(logits);
    free(toks);
    return 0;
}

static double vec_rel_err(const float *a, const float *b, size_t n,
                          double *max_abs)
{
    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    if (max_abs) *max_abs = mx;
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

/* Golden file (little endian): u32 magic 'MTPG', u32 T, u32 D, u32 n_kv,
 * u32 hd, then i32 tokens[T+2], f32 h[T+1][D], f32 s1[T][D],
 * f32 s2[T-1][D], f32 k1[n_kv][T][hd], f32 v1[n_kv][T][hd]. */
static int run_mtp_golden(OcLlamaModel *model, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    uint32_t hdr[5];
    if (fread(hdr, sizeof hdr, 1, fp) != 1 || hdr[0] != 0x4750544du) {
        fprintf(stderr, "bad golden header\n");
        fclose(fp);
        return 1;
    }
    const size_t T = hdr[1], D = hdr[2], NKV = hdr[3], HD = hdr[4];
    if (D != model->cfg.n_embd || NKV != model->cfg.n_head_kv ||
        HD != model->cfg.head_dim) {
        fprintf(stderr, "golden geometry mismatch\n");
        fclose(fp);
        return 1;
    }
    int32_t *tok32 = malloc((T + 2) * sizeof(int32_t));
    uint32_t *toks = malloc((T + 2) * sizeof(uint32_t));
    float *h = malloc((T + 1) * D * sizeof(float));
    float *s1 = malloc(T * D * sizeof(float));
    float *s2 = malloc(T * D * sizeof(float));
    float *k1 = malloc(NKV * T * HD * sizeof(float));
    float *v1 = malloc(NKV * T * HD * sizeof(float));
    float *o1 = malloc(T * D * sizeof(float));
    float *o2 = malloc(T * D * sizeof(float));
    float *kr = malloc(NKV * HD * sizeof(float));
    float *vr = malloc(NKV * HD * sizeof(float));
    if (!tok32 || !toks || !h || !s1 || !s2 || !k1 || !v1 || !o1 || !o2 ||
        !kr || !vr) return 1;
    bool ok = fread(tok32, sizeof(int32_t), T + 2, fp) == T + 2 &&
              fread(h, sizeof(float), (T + 1) * D, fp) == (T + 1) * D &&
              fread(s1, sizeof(float), T * D, fp) == T * D &&
              fread(s2, sizeof(float), (T - 1) * D, fp) == (T - 1) * D &&
              fread(k1, sizeof(float), NKV * T * HD, fp) == NKV * T * HD &&
              fread(v1, sizeof(float), NKV * T * HD, fp) == NKV * T * HD;
    fclose(fp);
    if (!ok) { fprintf(stderr, "short golden file\n"); return 1; }
    for (size_t i = 0; i < T + 2; ++i) toks[i] = (uint32_t)tok32[i];

    OcLlamaSession sess;
    if (oc_llama_session_init_kv(model, &sess, OC_KV_F32) != OC_OK ||
        oc_llama_mtp_enable(&sess, true) != OC_OK) {
        fprintf(stderr, "mtp session init failed\n");
        return 1;
    }
    /* Step 1: rows j = 0..T-1 are (E[t_{j+1}], h_j) at position j+1. */
    if (oc_llama_mtp_forward_rows(&sess, toks + 1, h, T, 1, o1) != OC_OK) return 1;
    double worst = 0.0, mx;
    printf("{\"mtp_golden\":{\"step1\":[");
    for (size_t j = 0; j < T; ++j) {
        double r = vec_rel_err(o1 + j * D, s1 + j * D, D, &mx);
        if (r > worst) worst = r;
        printf("%s{\"row\":%zu,\"rel\":%.3e,\"max_abs\":%.3e}", j ? "," : "", j, r, mx);
    }
    printf("],\"kv\":[");
    for (size_t j = 0; j < T; ++j) {
        oc_llama_mtp_kv_row(&sess, (int64_t)j + 1, kr, vr);
        float kg[4096], vg[4096];
        for (size_t hh = 0; hh < NKV; ++hh)
            for (size_t d = 0; d < HD; ++d) {
                kg[hh * HD + d] = k1[(hh * T + j) * HD + d];
                vg[hh * HD + d] = v1[(hh * T + j) * HD + d];
            }
        double rk = vec_rel_err(kr, kg, NKV * HD, NULL);
        double rv = vec_rel_err(vr, vg, NKV * HD, NULL);
        if (rk > worst) worst = rk;
        if (rv > worst) worst = rv;
        printf("%s{\"pos\":%zu,\"k_rel\":%.3e,\"v_rel\":%.3e}", j ? "," : "", j + 1, rk, rv);
    }
    /* Step 2 (chained): row j is (E[t_{j+2}], s1_j) at position j+2 and
     * attends step-1 entries 1..j+1 plus its own. Re-run step 1 before each
     * row so slot j+2 holds the step-1 entry the later rows expect. */
    printf("],\"step2\":[");
    for (size_t j = 0; j + 1 < T; ++j) {
        if (oc_llama_mtp_forward_rows(&sess, toks + 1, h, T, 1, o1) != OC_OK) return 1;
        if (oc_llama_mtp_forward_rows(&sess, toks + 2 + j, s1 + j * D, 1,
                                      (int64_t)j + 2, o2 + j * D) != OC_OK)
            return 1;
        double r = vec_rel_err(o2 + j * D, s2 + j * D, D, &mx);
        if (r > worst) worst = r;
        printf("%s{\"row\":%zu,\"rel\":%.3e,\"max_abs\":%.3e}", j ? "," : "", j, r, mx);
    }
    printf("],\"worst_rel\":%.3e}}\n", worst);
    oc_llama_session_free(&sess);
    free(tok32); free(toks); free(h); free(s1); free(s2); free(k1); free(v1);
    free(o1); free(o2); free(kr); free(vr);
    return worst < 2e-2 ? 0 : 3;
}

static void print_mtp_stats(const OcMtpStats *st)
{
    printf(",\"mtp\":{\"steps\":%llu,\"drafted\":%llu,\"accepted\":%llu,"
           "\"emitted\":%llu,\"accept_rate\":%.4f,\"tok_per_step\":%.4f,"
           "\"t_verify\":%.3f,\"t_draft\":%.3f,\"t_mtp\":%.3f,\"hist\":[",
           (unsigned long long)st->steps, (unsigned long long)st->drafted,
           (unsigned long long)st->accepted, (unsigned long long)st->emitted,
           st->drafted ? (double)st->accepted / (double)st->drafted : 0.0,
           st->steps ? (double)st->emitted / (double)st->steps : 0.0,
           st->t_verify, st->t_draft, st->t_mtp);
    for (int i = 0; i < 5; ++i)
        printf("%s%llu", i ? "," : "", (unsigned long long)st->accept_hist[i]);
    printf("],\"acc_at\":[");
    for (int i = 0; i < 4; ++i)
        printf("%s%.4f", i ? "," : "",
               st->draft_at[i] ? (double)st->accept_at[i] / (double)st->draft_at[i] : 0.0);
    printf("]}");
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--threads N] [--ctx N] [--kv f32|q8] "
                        "[--no-prefill] [--bench PP TG] [--reps R]\n", argv[0]);
        return 2;
    }
    size_t threads = 0;
    uint32_t ctx = 4096, pp = 0, tg = 0;
    int reps = 3;
    bool bench = false, no_prefill = false;
    const char *kv_name = "f32";
    const char *mtp_model = NULL, *mtp_golden = NULL;
    int mtp_k = -1;
    bool mtp_top5 = false;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = (size_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv_name = argv[++i];
        else if (strcmp(argv[i], "--no-prefill") == 0) no_prefill = true;
        else if (strcmp(argv[i], "--mtp-model") == 0 && i + 1 < argc) mtp_model = argv[++i];
        else if (strcmp(argv[i], "--mtp-top5") == 0) mtp_top5 = true;
        else if (strcmp(argv[i], "--mtp-golden") == 0 && i + 1 < argc) mtp_golden = argv[++i];
        else if (strcmp(argv[i], "--mtp") == 0 && i + 1 < argc) mtp_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bench") == 0 && i + 2 < argc) {
            bench = true;
            pp = (uint32_t)atoi(argv[++i]);
            tg = (uint32_t)atoi(argv[++i]);
        } else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
    }
    if (oc_parallel_set_threads(threads) != OC_OK) {
        fprintf(stderr, "thread pool init failed\n");
        return 1;
    }
    double t0 = now_s();
    OcLlamaModel model;
    OcError e = oc_llama_load(argv[1], &model);
    if (e != OC_OK) {
        fprintf(stderr, "load failed: %s\n", oc_error_msg(e));
        return 1;
    }
    if (ctx < model.cfg.n_ctx) model.cfg.n_ctx = ctx;
    if (mtp_model != NULL) {
        e = oc_llama_load_mtp_sidecar(&model, mtp_model);
        if (e != OC_OK) {
            fprintf(stderr, "mtp sidecar load failed: %s\n", oc_error_msg(e));
            return 1;
        }
    }
    if (mtp_golden != NULL) {
        int rc = run_mtp_golden(&model, mtp_golden);
        oc_llama_free(&model);
        return rc;
    }
    OcKvCacheType kv = oc_llama_select_kv_type(model.cfg.n_ctx, kv_name);
    fprintf(stderr, "loaded in %.2fs, threads=%zu\n", now_s() - t0,
            oc_parallel_n_threads());
    if (bench) {
        int rc = run_bench(&model, kv, pp, tg, reps);
        oc_llama_free(&model);
        return rc;
    }

    uint32_t vocab = model.cfg.vocab_size;
    float *logits = malloc((size_t)vocab * sizeof(float));
    uint32_t *ids = malloc(65536 * sizeof(uint32_t));
    char *line = malloc(1 << 20);
    if (!logits || !ids || !line) return 1;
    OcLlamaSession sess;
    if (oc_llama_session_init_kv(&model, &sess, kv) != OC_OK) return 1;
    if (mtp_k >= 0 && oc_llama_mtp_enable(&sess, true) != OC_OK) {
        fprintf(stderr, "mtp enable failed (no head?)\n");
        return 1;
    }
    while (fgets(line, 1 << 20, stdin)) {
        char name[128];
        int ngen = 0, off = 0;
        if (sscanf(line, "%127s %d%n", name, &ngen, &off) < 2) continue;
        size_t n = 0;
        char *p = line + off;
        for (;;) {
            char *end;
            unsigned long v = strtoul(p, &end, 10);
            if (end == p) break;
            ids[n++] = (uint32_t)v;
            p = end;
        }
        oc_llama_session_reset(&sess);
        double ts = now_s();
        if (no_prefill) {
            for (size_t i = 0; i < n; ++i)
                oc_llama_forward(&sess, ids[i], i + 1 == n ? logits : NULL);
        } else {
            oc_llama_prefill(&sess, ids, n, 0, logits);
        }
        double tp = now_s() - ts;
        if (mtp_k >= 0) {
            /* Speculative: ids only (no top-5), plus acceptance stats. */
            OcMtpStats st;
            memset(&st, 0, sizeof st);
            uint32_t *gen = malloc(((size_t)ngen + 8) * sizeof(uint32_t));
            size_t ng = 0;
            /* --mtp-top5 (k = 0 only: one token per step): the top-5 of the
             * logits each emitted token was chosen from, to measure how far
             * the batched verify path's logits sit from plain decode's. */
            double *t5 = (mtp_top5 && mtp_k == 0)
                       ? malloc(((size_t)ngen + 1) * 10 * sizeof(double)) : NULL;
            ts = now_s();
            while ((int)ng < ngen) {
                size_t got = 0;
                if (t5 != NULL) {
                    uint32_t tid[5];
                    double lp[5];
                    top5(logits, vocab, tid, lp);
                    for (int q = 0; q < 5; ++q) {
                        t5[ng * 10 + 2 * q] = tid[q];
                        t5[ng * 10 + 2 * q + 1] = lp[q];
                    }
                }
                e = oc_llama_mtp_step(&sess, (uint32_t)mtp_k, logits, gen + ng,
                                      (size_t)ngen - ng, &got, &st);
                if (e != OC_OK || got == 0) break;
                ng += got;
            }
            double td = now_s() - ts;
            printf("{\"name\":\"%s\",\"n_prompt\":%zu,\"prefill_s\":%.4f,"
                   "\"decode_s\":%.4f,\"n_gen\":%zu,\"tg_tps\":%.3f,\"ids\":[",
                   name, n, tp, td, ng, td > 0 ? (double)ng / td : 0.0);
            for (size_t i = 0; i < ng; ++i) printf("%s%u", i ? "," : "", gen[i]);
            printf("]");
            if (t5 != NULL) {
                printf(",\"top5\":[");
                for (size_t i = 0; i < ng; ++i) {
                    printf("%s[", i ? "," : "");
                    for (int q = 0; q < 5; ++q)
                        printf("%s[%u,%.5f]", q ? "," : "",
                               (unsigned)t5[i * 10 + 2 * q], t5[i * 10 + 2 * q + 1]);
                    printf("]");
                }
                printf("]");
                free(t5);
            }
            print_mtp_stats(&st);
            printf("}\n");
            fflush(stdout);
            free(gen);
            continue;
        }
        printf("{\"name\":\"%s\",\"n_prompt\":%zu,\"prefill_s\":%.4f,\"steps\":[", name, n, tp);
        ts = now_s();
        for (int s = 0; s < ngen; ++s) {
            uint32_t tid[5];
            double lp[5];
            top5(logits, vocab, tid, lp);
            printf("%s{\"id\":%u,\"top5\":[", s ? "," : "", tid[0]);
            for (int k = 0; k < 5; ++k)
                printf("%s[%u,%.5f]", k ? "," : "", tid[k], lp[k]);
            printf("]}");
            if (s + 1 < ngen) oc_llama_forward(&sess, tid[0], logits);
        }
        printf("],\"decode_s\":%.4f}\n", now_s() - ts);
        fflush(stdout);
    }
    oc_llama_session_free(&sess);
    oc_llama_free(&model);
    free(logits);
    free(ids);
    free(line);
    return 0;
}
