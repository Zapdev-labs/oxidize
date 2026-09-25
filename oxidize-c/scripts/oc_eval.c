/* oc_eval.c — token-id driven eval/bench harness (`make oc-eval`).
 *
 * Parity mode (default): reads one case per stdin line,
 *     NAME N_GEN ID ID ID ...
 * prefills the prompt ids, then greedily generates N_GEN tokens and prints
 * one JSON object per case with each step's argmax id and top-5 logprobs.
 *
 *   oc-eval MODEL.gguf [--threads N] [--ctx N] [--kv f32|q8|rq|rq:K,V]
 *                      [--rq-window N] [--rq-sinks N] [--rq-rot iso]
 *                      [--no-prefill]   (feed the prompt token by token)
 *                      [--bench PP TG --reps R [--depth D]]
 *                      [--ppl IDS_FILE [--chunks N] [--bos ID]]
 *
 * Bench mode prefills PP tokens (one timed oc_llama_prefill call) and then
 * decodes TG tokens (timed), R times, and prints tokens/second for each.
 * --depth D tiles the prefilled KV out to D positions (synthetic depth; the
 * attention cost does not depend on cache contents) before decoding.
 *
 * Perplexity mode matches llama-perplexity: IDS_FILE holds whitespace
 * separated token ids (BOS first); chunk i is ids[i*ctx, (i+1)*ctx) with its
 * first id replaced by BOS, and the second half of each chunk is scored.
 * --rq-window/--rq-sinks take -1 to disable the exact window / sinks.
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

static long rss_kb(const char *key)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char ln[256];
    long v = -1;
    size_t kl = strlen(key);
    while (fgets(ln, sizeof ln, f))
        if (strncmp(ln, key, kl) == 0) { v = atol(ln + kl); break; }
    fclose(f);
    return v;
}

typedef struct {
    const uint32_t *toks;
    size_t n;
    double nll;
    size_t count;
} PplAcc;

static uint32_t g_vocab;

static void ppl_cb(void *ud, size_t j, const float *lg)
{
    PplAcc *a = (PplAcc *)ud;
    if (j + 1 >= a->n) return;
    double mx = -INFINITY;
    for (uint32_t i = 0; i < g_vocab; ++i) if (lg[i] > mx) mx = lg[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < g_vocab; ++i) sum += exp((double)lg[i] - mx);
    a->nll += (mx + log(sum)) - (double)lg[a->toks[j + 1]];
    a->count++;
}

static int run_ppl(OcLlamaModel *model, const OcKvOptions *kvo,
                   const char *path, uint32_t ctx, int chunks, uint32_t bos)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    size_t cap = 1 << 16, n = 0;
    uint32_t *ids = malloc(cap * sizeof(uint32_t));
    unsigned long v;
    while (fscanf(f, "%lu", &v) == 1) {
        if (n == cap) { cap *= 2; ids = realloc(ids, cap * sizeof(uint32_t)); }
        ids[n++] = (uint32_t)v;
    }
    fclose(f);
    int max_chunks = (int)(n / ctx);
    if (chunks <= 0 || chunks > max_chunks) chunks = max_chunks;
    g_vocab = model->cfg.vocab_size;
    OcLlamaSession sess;
    if (oc_llama_session_init_kv_opts(model, &sess, kvo) != OC_OK) return 1;
    uint32_t *chunk = malloc(ctx * sizeof(uint32_t));
    double nll = 0.0;
    size_t count = 0;
    fprintf(stderr, "ppl: %zu ids, %d chunks of %u\n", n, chunks, ctx);
    for (int c = 0; c < chunks; ++c) {
        memcpy(chunk, ids + (size_t)c * ctx, ctx * sizeof(uint32_t));
        chunk[0] = bos;
        oc_llama_session_reset(&sess);
        PplAcc a = { chunk, ctx, 0.0, 0 };
        double t0 = now_s();
        if (oc_llama_prefill_all_logits(&sess, chunk, ctx, ctx / 2, ppl_cb, &a)
            != OC_OK) {
            fprintf(stderr, "prefill failed\n");
            return 1;
        }
        nll += a.nll;
        count += a.count;
        printf("{\"chunk\":%d,\"chunk_ppl\":%.4f,\"ppl\":%.4f,\"n\":%zu,"
               "\"s\":%.1f,\"rss_mb\":%ld}\n", c + 1, exp(a.nll / a.count),
               exp(nll / count), count, now_s() - t0, rss_kb("VmRSS:") / 1024);
        fflush(stdout);
    }
    printf("{\"final_ppl\":%.4f,\"tokens\":%zu}\n", exp(nll / count), count);
    oc_llama_session_free(&sess);
    free(chunk);
    free(ids);
    return 0;
}

static int run_bench(OcLlamaModel *model, const OcKvOptions *kvo, uint32_t pp,
                     uint32_t tg, int reps, int64_t depth)
{
    uint32_t vocab = model->cfg.vocab_size;
    float *logits = malloc((size_t)vocab * sizeof(float));
    uint32_t *toks = malloc((size_t)(pp ? pp : 1) * sizeof(uint32_t));
    if (!logits || !toks) return 1;
    srand(1234);
    for (uint32_t i = 0; i < pp; ++i) toks[i] = 1000u + (uint32_t)(rand() % 20000);
    OcLlamaSession sess;
    if (oc_llama_session_init_kv_opts(model, &sess, kvo) != OC_OK) return 1;
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
        if (depth > sess.pos) {
            if (oc_llama_session_fake_fill(&sess, depth) != OC_OK) {
                fprintf(stderr, "fake fill failed\n");
                return 1;
            }
        }
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
        printf("{\"rep\":%d,\"pp\":%u,\"pp_tps\":%.3f,\"depth\":%lld,\"tg\":%u,"
               "\"tg_tps\":%.3f,\"kv_bytes_per_tok\":%zu,\"rss_mb\":%ld,"
               "\"swap_mb\":%ld}\n",
               r, pp, pp ? pp / (t1 - t0) : 0.0, (long long)(depth > 0 ? depth : pp),
               tg, tg ? tg / (t3 - t2) : 0.0, oc_llama_kv_bytes_per_token(&sess),
               rss_kb("VmRSS:") / 1024, rss_kb("VmSwap:") / 1024);
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
    uint32_t ctx = 4096, pp = 0, tg = 0, bos = 0;
    int reps = 3, chunks = 0;
    long long depth = 0;
    bool bench = false, no_prefill = false;
    const char *kv_name = "f32", *ppl_path = NULL;
    OcKvOptions kvo = {0};
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = (size_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv_name = argv[++i];
        else if (strcmp(argv[i], "--no-prefill") == 0) no_prefill = true;
        else if (strcmp(argv[i], "--rq-window") == 0 && i + 1 < argc) kvo.rq_window = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rq-sinks") == 0 && i + 1 < argc) kvo.rq_sinks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rq-rot") == 0 && i + 1 < argc) kvo.rq_rot = strcmp(argv[++i], "iso") == 0;
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) depth = atoll(argv[++i]);
        else if (strcmp(argv[i], "--ppl") == 0 && i + 1 < argc) ppl_path = argv[++i];
        else if (strcmp(argv[i], "--chunks") == 0 && i + 1 < argc) chunks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bos") == 0 && i + 1 < argc) bos = (uint32_t)atoi(argv[++i]);
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
    if (ctx != 0 && ctx < model.cfg.n_ctx) model.cfg.n_ctx = ctx;
    if (!oc_llama_parse_kv_type(kv_name, &kvo)) {
        fprintf(stderr, "bad --kv %s\n", kv_name);
        return 2;
    }
    fprintf(stderr, "loaded in %.2fs, threads=%zu\n", now_s() - t0,
            oc_parallel_n_threads());
    if (ppl_path != NULL) {
        int rc = run_ppl(&model, &kvo, ppl_path, ctx, chunks, bos);
        oc_llama_free(&model);
        return rc;
    }
    if (bench) {
        int rc = run_bench(&model, &kvo, pp, tg, reps, depth);
        oc_llama_free(&model);
        return rc;
    }

    uint32_t vocab = model.cfg.vocab_size;
    float *logits = malloc((size_t)vocab * sizeof(float));
    const size_t max_ids = 1u << 20, line_cap = 16u << 20;
    uint32_t *ids = malloc(max_ids * sizeof(uint32_t));
    char *line = malloc(line_cap);
    if (!logits || !ids || !line) return 1;
    OcLlamaSession sess;
    if (oc_llama_session_init_kv_opts(&model, &sess, &kvo) != OC_OK) return 1;
    while (fgets(line, (int)line_cap, stdin)) {
        char name[128];
        int ngen = 0, off = 0;
        if (sscanf(line, "%127s %d%n", name, &ngen, &off) < 2) continue;
        size_t n = 0;
        char *p = line + off;
        for (;;) {
            char *end;
            unsigned long v = strtoul(p, &end, 10);
            if (end == p || n == max_ids) break;
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
        const double dt = now_s() - ts;
        printf("],\"decode_s\":%.4f,\"n_prompt\":%zu,\"pp_tps\":%.3f,"
               "\"tg_tps\":%.3f,\"rss_mb\":%ld,\"swap_mb\":%ld}\n", dt, n,
               tp > 0 ? (double)n / tp : 0.0,
               ngen > 1 ? (double)(ngen - 1) / dt : 0.0,
               rss_kb("VmRSS:") / 1024, rss_kb("VmSwap:") / 1024);
        fflush(stdout);
    }
    oc_llama_session_free(&sess);
    oc_llama_free(&model);
    free(logits);
    free(ids);
    free(line);
    return 0;
}
