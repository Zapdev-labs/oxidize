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
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = (size_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv_name = argv[++i];
        else if (strcmp(argv[i], "--no-prefill") == 0) no_prefill = true;
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
        printf("{\"name\":\"%s\",\"prefill_s\":%.4f,\"steps\":[", name, tp);
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
