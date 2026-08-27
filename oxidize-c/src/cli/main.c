/* strdup is POSIX.1-2008; needs _POSIX_C_SOURCE to be declared. Must be
 * the first non-comment thing in the file, before any system header. */
#define _POSIX_C_SOURCE 200809L

/*
 * main.c — oxidize-c CLI entry point.
 *
 * Implements the `cli-flags-modes` feature: flag parsing, prompt/prefill
 * generation, and scaffolds for --serve-api / --print-plan / --auto.
 *
 * Wires oc_llama_load → oc_tokenizer_load_from_gguf → oc_llama_session_init
 * → encode(prompt) → prefill forward loop → sample → decode → print, looping
 * until --n-predict tokens are emitted or EOS is reached.
 *
 * Flag set mirrors the Rust `oxidize-cli` conventions and the learned user
 * preferences (--numa, --auto/--no-auto, --print-plan, --serve-api,
 * --threads). --threads/--numa override the autotune plan and are applied
 * by apply_thread_numa_policy() below.
 */
#include "oxidize/activation.h"   /* ensure link for forward deps */
#include "oxidize/autotune.h"
#include "oxidize/numa.h"
#include "oxidize/parallel.h"
#include "oxidize/version.h"
#include "oxidize/cli_commands.h"
#include "oxidize/cuda.h"
#include "oxidize/dspark.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/http.h"
#include "oxidize/llama.h"
#include "oxidize/log.h"
#include "oxidize/oc.h"
#include "oxidize/openai.h"
#include "oxidize/perplexity.h"
#include "oxidize/quantize_tool.h"
#include "oxidize/sampling.h"
#include "oxidize/tokenizer.h"

#include "args.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Kept as an alias so the existing call sites read unchanged; the value
 * itself lives in version.h, shared with the server's /openapi.json. */
#define OC_CLI_VERSION OC_VERSION

/* Apply the context cap to an already-loaded model. `requested` is the --ctx
 * value (0 = use the default cap). Shared by every path in this file that
 * loads a model; the server especially needs it, because a session — and so a
 * full KV cache — is created per request. */
static void cap_model_ctx(OcLlamaModel *model, uint32_t requested)
{
    if (model == NULL || model->cfg.n_ctx == 0) return;
    uint32_t want = requested > 0 ? requested : OC_CLI_DEFAULT_MAX_CTX;
    if (want >= model->cfg.n_ctx) return;
    oc_log(OC_LOG_INFO, "ctx: capping context %u -> %u%s",
           model->cfg.n_ctx, want,
           requested > 0 ? "" : " (default; --ctx N to change)");
    model->cfg.n_ctx = want;
}

/* Size and start the compute pool. Called once, early, so that every mode —
 * generation, bench, serve, and the subcommands — gets a threaded forward
 * pass; previously only the autotune path touched threading and the forward
 * pass ran on a single core regardless.
 *
 * Default is physical cores, not logical: these kernels are memory-bandwidth
 * bound, so the second SMT thread on a core mostly adds contention. --threads
 * overrides, and --auto may refine it later once the model is fingerprinted. */
static void init_compute_threads(int requested)
{
    size_t n;
    if (requested > 0) {
        n = (size_t)requested;
    } else {
        OcCpuInfo cpu;
        if (oc_autotune_detect_cpu(&cpu) == OC_OK && cpu.physical_cores > 0) {
            n = cpu.physical_cores;
        } else {
            n = 0;   /* let the pool ask the OS */
        }
    }
    if (oc_parallel_set_threads(n) != OC_OK) {
        oc_log(OC_LOG_WARN, "parallel: pool init failed; running inline");
        return;
    }
    oc_log(OC_LOG_INFO, "parallel: %zu compute threads",
           oc_parallel_n_threads());
}

static double wall_now(void)
{
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Flag parsing + OcCliArgs live in args.c (extracted for testability). */

static void print_help(void)
{
    printf(
"oxidize-c v%s — dependency-free C11 LLM inference\n"
"\n"
"USAGE:\n"
"  oxidize-c --model <path.gguf> --prompt \"text\" [OPTIONS]\n"
"\n"
"OPTIONS:\n"
"  --model PATH           GGUF model file (required for generation)\n"
"  --prompt TEXT          Prompt text (also accepted as positional arg)\n"
"  --prompt-file PATH     Read prompt from a file\n"
"  --n-predict N          Max tokens to generate (default 128)\n"
"  --ctx N                Cap KV context below the model's advertised length\n"
"  --kv f32|q8            KV cache dtype (default: q8 when ctx>=8192)\n"
"  --threads N            CPU thread hint (0 = auto)\n"
"  --numa MODE            single | interleave | none (default none)\n"
"  --auto                 Enable autotune (detect + plan)\n"
"  --no-auto              Disable autotune\n"
"  --print-plan           Print the autotune plan and exit\n"
"  --serve-api            Start the OpenAI-compatible HTTP server\n"
"  --host HOST            Server bind host (default 127.0.0.1)\n"
"  --port PORT            Server bind port (default 8080)\n"
"  --api-key KEY          Require `Authorization: Bearer KEY` (default off)\n"
"  --rate-limit N         Per-IP request/minute cap (default off)\n"
"  --cors-origin ORIGIN   Send CORS headers for ORIGIN (default off)\n"
"  --temperature T        Sampling temperature (0 = greedy, default 0)\n"
"  --top-k K              Top-K sampling (default 40, 0 = disabled)\n"
"  --top-p P              Top-P / nucleus (default 0.95)\n"
"  --repeat-penalty P     Repeat penalty (default 1.1)\n"
"  --seed N               RNG seed (0 = deterministic default)\n"
"  --spec-type TYPE       none | mtp | dspark (default: dspark if GGUF has MTP)\n"
"  --draft-tokens N       MTP/DSpark draft block size (default 4)\n"
"  --backend cpu|cuda     Compute backend (default cpu)\n"
"  --cuda-selftest        Run CUDA kernel self-test (no GGUF) and exit\n"
"  -v, --verbose          Verbose logging\n"
"  -h, --help             Show this help\n"
"  --version              Print version and exit\n",
    OC_CLI_VERSION);
}

/* ─── Generation ──────────────────────────────────────────────────────── */

/* Apply the thread + NUMA half of a tuning plan, honoring explicit CLI
 * overrides. NUMA policy binds this process to a socket; the thread count
 * resizes the compute pool (already started by init_compute_threads) and
 * drives the parallel weight prefault. */
static void apply_thread_numa_policy(const OcCliArgs *args,
                                     const OcTuningPlan *plan,
                                     const OcCpuInfo *cpu,
                                     const OcGgufMmappedFile *weights)
{
    /* --threads N overrides the plan; 0 means "use the plan". */
    uint32_t threads = args->threads > 0 ? (uint32_t)args->threads
                                         : plan->threads;
    if (threads == 0) threads = 1;

    /* --numa MODE overrides the plan. "none" is also the default value of
     * the flag, so it cannot be told apart from "unset" and leaves the plan
     * (or, without --auto, OC_NUMA_NONE) in place. */
    OcNumaPolicy numa = args->auto_tune ? plan->numa : OC_NUMA_NONE;
    if (args->numa) {
        if (strcmp(args->numa, "single") == 0)          numa = OC_NUMA_SINGLE;
        else if (strcmp(args->numa, "interleave") == 0) numa = OC_NUMA_INTERLEAVE;
    }

    /* Memory policy must be set BEFORE the weights are faulted in below:
     * set_mempolicy applies to future faults and does not migrate pages that
     * already exist. */
    if (numa == OC_NUMA_SINGLE) {
        /* Bind to node 0: the plan picks SINGLE only when the model fits in
         * one socket's memory, so any single node works and 0 always exists.
         * Bind both the threads (affinity) and the pages (mempolicy) — CPU
         * affinity alone still lets pages land on the far node. */
        if (oc_autotune_bind_to_numa_node(0) == OC_OK) {
            oc_log(OC_LOG_INFO, "autotune: bound to NUMA node 0");
        }
        if (oc_numa_set_policy(OC_NUMA_POLICY_BIND, 0) == OC_OK) {
            oc_log(OC_LOG_INFO, "autotune: memory bound to NUMA node 0");
        }
    } else if (numa == OC_NUMA_INTERLEAVE && cpu->numa_nodes > 1) {
        /* Interleave is NOT the kernel default — MPOL_DEFAULT allocates on
         * the first-touching thread's local node, so a model faulted in by
         * threads sitting on one socket lands entirely on that socket and
         * every read from the other socket crosses the interconnect. Request
         * it explicitly. */
        if (oc_numa_set_policy(OC_NUMA_POLICY_INTERLEAVE, 0) == OC_OK) {
            oc_log(OC_LOG_INFO, "autotune: memory interleaved across %u "
                   "NUMA nodes", cpu->numa_nodes);
        } else {
            oc_log(OC_LOG_WARN, "autotune: could not set interleave policy; "
                   "falling back to first-touch placement");
        }
    }

    oc_log(OC_LOG_INFO, "autotune: applying %u threads, numa=%s, simd=%s",
           threads, oc_autotune_numa_name(numa), cpu->simd.name);

    /* Start the compute pool. Until this call the forward pass runs inline on
     * one core no matter what --threads said, which is what made an 8B model
     * crawl on a 96-core box. */
    if (oc_parallel_set_threads(threads) != OC_OK) {
        oc_log(OC_LOG_WARN, "parallel: pool init failed; running inline");
    }

    /* Fault the weights in with the resolved thread count so the first
     * tokens don't pay page-fault cost serially. */
    if (weights != NULL && threads > 1) {
        (void)oc_gguf_map_prefault_parallel(weights, (size_t)threads);
    }
}

static OcError run_generation(const OcCliArgs *args)
{
    if (args->model_path == NULL) {
        fprintf(stderr, "error: --model is required for generation\n");
        fprintf(stderr, "  (run `oxidize-c --help` for usage)\n");
        return OC_ERR_INVALID_ARG;
    }
    const char *prompt = args->prompt;
    char *file_prompt = NULL;
    if (prompt == NULL && args->prompt_file != NULL) {
        FILE *f = fopen(args->prompt_file, "rb");
        if (f == NULL) {
            fprintf(stderr, "error: cannot open prompt-file: %s\n", args->prompt_file);
            return OC_ERR_IO;
        }
        if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return OC_ERR_IO; }
        long sz = ftell(f);
        if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return OC_ERR_IO; }
        file_prompt = malloc((size_t)sz + 1);
        if (file_prompt == NULL) { fclose(f); return OC_ERR_OOM; }
        size_t rd = fread(file_prompt, 1, (size_t)sz, f);
        file_prompt[rd] = '\0';
        fclose(f);
        prompt = file_prompt;
    }
    if (prompt == NULL) prompt = "";

    oc_log(OC_LOG_INFO, "loading model: %s", args->model_path);
    OcLlamaModel model;
    OcError e = oc_llama_load(args->model_path, &model);
    if (e != OC_OK) {
        fprintf(stderr, "error: failed to load model (%s)\n", oc_error_msg(e));
        free(file_prompt);
        return e;
    }

    /* Clamp the context before anything sizes a KV cache off it. Models now
     * advertise context lengths far beyond what their cache can occupy:
     * Gemma 4 reports 262144, which at its 4096-element KV row needs 515 GB
     * of f32 cache. Never raise it above the model's own value — the RoPE
     * tables and the cache indexing both assume positions stay in range. */
    /* Without --ctx, fall back to OC_CLI_DEFAULT_MAX_CTX rather than the
     * advertised value. Defaulting to the model's own number means an 8B
     * Llama-3.1 tries to allocate 34 GB of KV cache before emitting a token,
     * which is the difference between "slow" and "unusable". */
    {
        uint32_t want = args->n_ctx > 0 ? args->n_ctx : OC_CLI_DEFAULT_MAX_CTX;
        if (want < model.cfg.n_ctx) {
            oc_log(OC_LOG_INFO, "ctx: capping context %u -> %u%s",
                   model.cfg.n_ctx, want,
                   args->n_ctx > 0 ? "" : " (default; --ctx N to change)");
            model.cfg.n_ctx = want;
        }
    }

    /* CUDA backend: upload weights to GPU and use GPU forward path. */
    bool use_cuda = (args->backend && strcmp(args->backend, "cuda") == 0);
    OcCudaContext cuda_ctx;
    if (use_cuda) {
        if (!oc_cuda_available()) {
            fprintf(stderr, "error: CUDA not available (compiled without OC_CUDA or no GPU)\n");
            oc_llama_free(&model);
            free(file_prompt);
            return OC_ERR_BACKEND;
        }
        e = oc_cuda_init(&cuda_ctx, &model);
        if (e != OC_OK) {
            fprintf(stderr, "error: CUDA init failed (%s), falling back to CPU\n",
                    oc_error_msg(e));
            use_cuda = false;
        } else {
            oc_log(OC_LOG_INFO, "cuda: model uploaded to GPU, using CUDA forward");
        }
    }

    /* Autotune: detect CPU, fingerprint the model, plan, and apply. The
     * memory-side policy (hugepages/mlock) goes to the mmap'd weights;
     * thread and NUMA policy are applied here (see apply_thread_numa_policy).
     * Explicit --threads/--numa always win over the plan, so the policy runs
     * even without --auto when the user asked for something specific. */
    if (args->auto_tune || args->threads > 0 ||
        (args->numa && strcmp(args->numa, "none") != 0)) {
        OcCpuInfo cpu;
        OcModelFingerprint fp;
        if (oc_autotune_detect_cpu(&cpu) == OC_OK &&
            oc_autotune_fingerprint_gguf(&model.gguf, &fp) == OC_OK) {
            OcTuningPlan plan = oc_autotune_plan(&cpu, &fp);
            if (args->auto_tune) oc_autotune_apply(&plan, &model.gguf);
            apply_thread_numa_policy(args, &plan, &cpu, &model.gguf);
        }
    }

    /* Tokenizer (loaded from the same GGUF metadata). */
    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&model.gguf.unified, &tok);
    if (e != OC_OK) {
        fprintf(stderr, "error: tokenizer load failed (%s)\n", oc_error_msg(e));
        oc_llama_free(&model);
        free(file_prompt);
        return e;
    }

    OcLlamaSession sess;
    OcKvCacheType kv = oc_llama_select_kv_type(model.cfg.n_ctx, args->kv_type);
    e = oc_llama_session_init_kv(&model, &sess, kv);
    if (e != OC_OK) {
        fprintf(stderr, "error: session init failed (%s)\n", oc_error_msg(e));
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        free(file_prompt);
        return e;
    }

    /* Encode prompt. */
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    OcSpecialTokenPolicy policy = tok.has_add_bos_token && tok.add_bos_token
        ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
    e = oc_tokenizer_encode(&tok, prompt, policy, &ids, &n_ids);
    if (e != OC_OK || n_ids == 0) {
        fprintf(stderr, "error: prompt encode failed (%s)\n", oc_error_msg(e));
        oc_llama_session_free(&sess);
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        free(file_prompt);
        return e != OC_OK ? e : OC_ERR_TOKENIZER;
    }

    oc_log(OC_LOG_INFO, "prompt: %zu tokens, generating up to %d",
           n_ids, args->n_predict);

    /* Build sampler config. */
    OcSamplerConfig scfg = OC_SAMPLER_DEFAULT;
    scfg.temperature    = args->temperature;
    scfg.repeat_penalty = args->repeat_penalty;
    scfg.seed           = args->seed;
    scfg.min_p = args->min_p;
    scfg.tau = args->mirostat_tau;
    scfg.eta = args->mirostat_eta;
    scfg.mu = 2.0f * args->mirostat_tau;
    if (args->temperature <= 0.0f) {
        scfg.type = OC_SAMPLER_GREEDY;
    } else if (args->mirostat_tau > 0.0f) {
        scfg.type = OC_SAMPLER_MIROSTAT_V2;
    } else if (args->min_p > 0.0f) {
        scfg.type = OC_SAMPLER_MIN_P;
    } else if (args->top_k > 0) {
        scfg.type   = OC_SAMPLER_TOP_K;
        scfg.top_k  = args->top_k;
        scfg.top_p  = args->top_p;
    } else {
        scfg.type   = OC_SAMPLER_TOP_P;
        scfg.top_k  = 0;
        scfg.top_p  = args->top_p;
    }

    /* Recent-token ring buffer for repeat penalty (last 64 tokens). */
    #define RECENT_CAP 64
    uint32_t recent[RECENT_CAP];
    size_t recent_len = 0;

    /* Prefill. Every prompt token is known up front, so on CPU they go
     * through oc_llama_prefill(), which pushes a chunk of them through each
     * weight matrix together instead of sweeping the whole model once per
     * token. The last token's logits seed the generation loop.
     *
     * The CUDA path keeps the per-token loop: its forward already uploads the
     * weights once and the batched CPU scratch would not help it. */
    float *logits = sess.logits;
    uint32_t next_tok = ids[n_ids - 1];
    const double prefill_start = wall_now();
    if (use_cuda) {
        for (size_t i = 0; i + 1 < n_ids; i++) {
            e = oc_cuda_forward(&cuda_ctx, ids[i], sess.pos, NULL);
            if (e != OC_OK) {
                fprintf(stderr,
                        "error: prefill forward failed at token %zu (%s)\n",
                        i, oc_error_msg(e));
                break;
            }
            sess.pos++;
        }
        if (e == OC_OK) {
            e = oc_cuda_forward(&cuda_ctx, next_tok, sess.pos, logits);
            sess.pos++;
        }
    } else {
        e = oc_llama_prefill(&sess, ids, n_ids, args->batch_size, logits);
        if (e != OC_OK) {
            fprintf(stderr, "error: prefill failed (%s)\n", oc_error_msg(e));
        }
    }
    if (e == OC_OK) {
        const double pf = wall_now() - prefill_start;
        if (pf > 0.0) {
            oc_log(OC_LOG_INFO, "prefill: %zu tokens in %.3fs = %.2f tok/s",
                   n_ids, pf, (double)n_ids / pf);
        }
    }
    /* Seed the repeat-penalty ring with the prompt. */
    for (size_t i = 0; i + 1 < n_ids && e == OC_OK; i++) {
        if (recent_len < RECENT_CAP) recent[recent_len++] = ids[i];
        else recent[i % RECENT_CAP] = ids[i];
    }
    if (e != OC_OK) {
        fprintf(stderr, "error: seed forward failed (%s)\n", oc_error_msg(e));
    } else {
        if (recent_len < RECENT_CAP) recent[recent_len++] = next_tok;
        else recent[sess.pos % RECENT_CAP] = next_tok;
        /* Decode + print the prompt's last token context implicitly; generate. */
        size_t emitted = 0;
        bool eos_reached = false;
        double decode_start = wall_now();
        while (emitted < (size_t)args->n_predict && !eos_reached && e == OC_OK) {
            uint32_t sampled;
            if (!use_cuda && scfg.type == OC_SAMPLER_GREEDY &&
                oc_llama_mtp_present(&model) &&
                !(args->spec_type && strcmp(args->spec_type, "none") == 0)) {
                uint32_t toks[8];
                size_t n = 0;
                size_t want = (size_t)args->n_predict - emitted;
                if (want > 8) want = 8;
                OcDsparkConfig dcfg;
                oc_dspark_config_init(&dcfg);
                if (args->draft_tokens > 0) dcfg.block_size = (uint32_t)args->draft_tokens;
                if (args->spec_type && strcmp(args->spec_type, "mtp") == 0)
                    dcfg.p_min = 0.0f;
                OcDsparkStats st;
                e = oc_dspark_advance(&sess, logits, &dcfg, toks, want, &n, &st);
                if (e != OC_OK || n == 0) break;
                for (size_t i = 0; i < n; i++) {
                    sampled = toks[i];
                    if (tok.has_eos && sampled == tok.eos_id) {
                        eos_reached = true;
                        break;
                    }
                    char *piece = NULL;
                    if (oc_tokenizer_decode(&tok, &sampled, 1, &piece) == OC_OK && piece) {
                        fputs(piece, stdout);
                        fflush(stdout);
                        free(piece);
                    }
                    emitted++;
                    if (recent_len < RECENT_CAP) recent[recent_len++] = sampled;
                    else recent[emitted % RECENT_CAP] = sampled;
                    if (emitted >= (size_t)args->n_predict) break;
                }
                continue;
            }
            sampled = oc_sample(logits, model.cfg.vocab_size, &scfg,
                                        recent, recent_len);
            scfg.seed++;
            if (tok.has_eos && sampled == tok.eos_id) { eos_reached = true; break; }

            /* Decode + print the sampled token. */
            char *piece = NULL;
            if (oc_tokenizer_decode(&tok, &sampled, 1, &piece) == OC_OK && piece) {
                fputs(piece, stdout);
                fflush(stdout);
                free(piece);
            }
            emitted++;

            /* Update recent ring. */
            if (recent_len < RECENT_CAP) recent[recent_len++] = sampled;
            else recent[emitted % RECENT_CAP] = sampled;

            /* Forward the sampled token to get next logits. */
            if (use_cuda)
                e = oc_cuda_forward(&cuda_ctx, sampled, sess.pos, logits);
            else
                e = oc_llama_forward(&sess, sampled, logits);
            if (use_cuda) sess.pos++;
            if (e == OC_ERR_INVALID_ARG) {
                /* Context window exhausted. */
                oc_log(OC_LOG_WARN, "context window full at position %lld",
                       (long long)sess.pos);
                e = OC_OK;
                break;
            }
        }
        if (emitted > 0) fputs("\n", stdout);
        oc_log(OC_LOG_INFO, "generated %zu tokens", emitted);
        if (decode_start > 0 && emitted > 0) {
            double elapsed = wall_now() - decode_start;
            if (elapsed > 0) {
                double tps = (double)emitted / elapsed;
                fprintf(stderr, "\n%.2f tok/s (%zu tokens in %.3fs)\n",
                        tps, emitted, elapsed);
                oc_log(OC_LOG_INFO, "speed: %.2f tok/s", tps);
            }
        }
    }

    free(ids);
    if (use_cuda) oc_cuda_free(&cuda_ctx);
    oc_llama_session_free(&sess);
    oc_tokenizer_free(&tok);
    oc_llama_free(&model);
    free(file_prompt);
    return e;
}

static void print_metadata_value(const OcGgufMetadataValue *value)
{
    switch (value->type) {
    case OC_GGUF_MT_UINT8: printf("%u", value->v.u8); break;
    case OC_GGUF_MT_INT8: printf("%d", value->v.i8); break;
    case OC_GGUF_MT_UINT16: printf("%u", value->v.u16); break;
    case OC_GGUF_MT_INT16: printf("%d", value->v.i16); break;
    case OC_GGUF_MT_UINT32: printf("%u", value->v.u32); break;
    case OC_GGUF_MT_INT32: printf("%d", value->v.i32); break;
    case OC_GGUF_MT_FLOAT32: printf("%g", value->v.f32); break;
    case OC_GGUF_MT_BOOL: printf("%s", value->v.b ? "true" : "false"); break;
    case OC_GGUF_MT_STRING:
        printf("\"%.*s\"", (int)(value->v.str.len > 80 ? 80 : value->v.str.len),
               value->v.str.data);
        break;
    case OC_GGUF_MT_ARRAY:
        putchar('[');
        for (size_t i = 0; i < value->v.arr.len; i++) {
            if (i > 0) printf(", ");
            print_metadata_value(&value->v.arr.values[i]);
        }
        putchar(']');
        break;
    case OC_GGUF_MT_UINT64: printf("%llu", (unsigned long long)value->v.u64); break;
    case OC_GGUF_MT_INT64: printf("%lld", (long long)value->v.i64); break;
    case OC_GGUF_MT_FLOAT64: printf("%.17g", value->v.f64); break;
    default: printf("(%u)", value->type); break;
    }
}

/* ─── Entrypoint ──────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    oc_log_init_from_env();

    /* Subcommand form (`oxidize-c quantize --input ... --output ...`) takes
     * precedence. Falls through to the flag-only form when argv[1] is not a
     * known subcommand, so the original `--model/--prompt` invocation and
     * every existing script keep working. */
    OcCliContext ctx;
    if (oc_cli_context_parse(argc, argv, &ctx)) {
        init_compute_threads(ctx.threads);
        OcError ce = oc_cli_command_run(&ctx);
        oc_parallel_shutdown();
        return ce == OC_OK ? 0 : 1;
    }

    OcCliArgs args;
    oc_cli_parse_args(argc, argv, &args);

    /* After the early exits: neither help nor --version does any compute, and
     * starting a pool only to tear it down would just add startup latency. */
    if (args.show_help)    { print_help(); return 0; }
    if (args.show_version) { printf("oxidize-c v%s\n", OC_CLI_VERSION); return 0; }
    if (args.cuda_selftest) {
        OcError se = oc_cuda_selftest();
        if (se != OC_OK) {
            fprintf(stderr, "error: CUDA self-test failed (%s)\n",
                    oc_error_msg(se));
            return 1;
        }
        return 0;
    }
    init_compute_threads(args.threads);

    if (args.print_plan) {
        OcCpuInfo cpu;
        oc_autotune_detect_cpu(&cpu);
        OcModelFingerprint model;
        memset(&model, 0, sizeof(model));
        if (args.model_path) {
            OcGgufMmappedFile m;
            if (oc_gguf_map_open(args.model_path, &m) == OC_OK) {
                oc_autotune_fingerprint_gguf(&m, &model);
                OcTuningPlan plan = oc_autotune_plan(&cpu, &model);
                oc_autotune_plan_dump(&plan, &cpu, &model);
                oc_gguf_map_free(&m);
                return 0;
            } else {
                fprintf(stderr, "error: could not open model for fingerprint: %s\n",
                        args.model_path);
                return 1;
            }
        }
        /* No model: dump CPU-only plan. */
        OcTuningPlan plan = oc_autotune_plan(&cpu, &model);
        oc_autotune_plan_dump(&plan, &cpu, NULL);
        return 0;
    }

    if (args.serve_api) {
        /* Start the OpenAI-compatible HTTP server. If --model is given,
         * load it; otherwise serve placeholder responses. */
        OcOpenaiState st;
        memset(&st, 0, sizeof(st));
        st.model_loaded = false;
        if (args.model_path) {
            OcLlamaModel *model = calloc(1, sizeof(OcLlamaModel));
            OcTokenizer *tok = calloc(1, sizeof(OcTokenizer));
            if (model && tok &&
                oc_llama_load(args.model_path, model) == OC_OK &&
                (cap_model_ctx(model, args.n_ctx), true) &&
                oc_tokenizer_load_from_gguf(&model->gguf.unified, tok) == OC_OK) {
                st.model = model;
                st.tokenizer = tok;
                st.model_loaded = true;
                /* Derive a model id from the filename. */
                const char *slash = strrchr(args.model_path, '/');
                st.model_id = strdup(slash ? slash + 1 : args.model_path);
                oc_log(OC_LOG_INFO, "serve: model loaded, starting server on %s:%d",
                       args.host, args.port);
            } else {
                fprintf(stderr, "error: failed to load model for serve mode\n");
                free(model); free(tok);
                return 1;
            }
        } else {
            oc_log(OC_LOG_INFO, "serve: no model — starting placeholder server on %s:%d",
                   args.host, args.port);
        }
        /* Middleware stack. Metrics + audit are always on so /metrics has
         * something to serve; auth and rate limiting engage only when their
         * flag is given, leaving the default local invocation unchanged. */
        OcMiddleware mw;
        uint32_t mw_enabled = OC_MW_METRICS | OC_MW_AUDIT;
        if (args.api_key != NULL)       mw_enabled |= OC_MW_AUTH;
        if (args.rate_limit_rpm > 0)    mw_enabled |= OC_MW_RATE_LIMIT;
        if (args.cors_origin != NULL)   mw_enabled |= OC_MW_CORS;
        OcRateLimitConfig rl = {
            .requests_per_minute = (uint32_t)(args.rate_limit_rpm > 0
                                              ? args.rate_limit_rpm : 0),
            /* One second's worth of burst (min 1), so a browser opening
             * several connections at once is not throttled. */
            .burst_size = (uint32_t)(args.rate_limit_rpm > 0
                                     ? args.rate_limit_rpm / 60 + 1 : 1),
            .per_ip = true,
        };
        OcError me = oc_middleware_init(&mw, mw_enabled, args.api_key, &rl,
                                        args.cors_origin);
        if (me != OC_OK) {
            fprintf(stderr, "error: middleware init failed (%s)\n",
                    oc_error_msg(me));
            return 1;
        }
        st.mw = &mw;

        OcHttpServer srv;
        memset(&srv, 0, sizeof(srv));
        oc_openai_attach_http(&srv, &st);
        OcError e = oc_http_server_start_configured(
            args.host, (uint16_t)args.port, 4, oc_openai_handler, &st, &srv);
        if (e != OC_OK) {
            fprintf(stderr, "error: server start failed (%s)\n", oc_error_msg(e));
            oc_middleware_free(&mw);
            return 1;
        }
        printf("oxidize-c server listening on http://%s:%u\n"
               "  GET  /healthz  /livez  /readyz\n"
               "  GET  /metrics  /openapi.json\n"
               "  GET  /v1/models\n"
               "  POST /v1/completions\n"
               "  POST /v1/chat/completions\n"
               "  POST /v1/embeddings\n"
               "  POST /v1/responses\n"
               "%s%s"
               "(Ctrl+C to stop)\n", args.host, srv.port,
               args.api_key ? "  auth: enabled (Bearer token required)\n" : "",
               args.rate_limit_rpm > 0 ? "  rate limit: enabled (per-IP)\n" : "");
        fflush(stdout);
        oc_http_server_join(&srv);
        oc_middleware_free(&mw);
        /* Cleanup (only reached after stop). */
        if (st.model_loaded) {
            free(st.model_id);
            oc_tokenizer_free(st.tokenizer);
            oc_llama_free(st.model);
            free(st.tokenizer);
            free(st.model);
        }
        return 0;
    }

    if (args.inspect && args.model_path) {
        /* Inspect mode: print model info and exit. */
        OcGgufMmappedFile mf;
        OcError ie = oc_gguf_map_open(args.model_path, &mf);
        if (ie != OC_OK) {
            fprintf(stderr, "error: cannot open GGUF (%s)\n", oc_error_msg(ie));
            return 1;
        }
        const OcGgufFile *gf = &mf.unified;
        OcModelArchitecture arch = oc_gguf_arch_from_file(gf);
        printf("=== oxidize-c model inspection ===\n");
        printf("File: %s\n", args.model_path);
        printf("GGUF version: %u\n", gf->version);
        printf("Tensors: %llu\n", (unsigned long long)gf->tensor_count);
        printf("Metadata KV pairs: %llu\n", (unsigned long long)gf->metadata_kv_count);
        printf("Architecture: %s\n", oc_model_arch_name(arch));
        printf("Alignment: %llu\n", (unsigned long long)gf->alignment);
        printf("\n--- Metadata ---\n");
        for (uint64_t i = 0; i < gf->metadata_kv_count && i < 50; i++) {
            const OcGgufMetadataKV *kv = &gf->metadata[i];
            printf("  %s = ", kv->key);
            print_metadata_value(&kv->value);
            putchar('\n');
        }
        printf("\n--- Tensors (first 20) ---\n");
        for (uint64_t i = 0; i < gf->tensor_count && i < 20; i++) {
            const OcGgufTensorInfo *t = &gf->tensors[i];
            printf("  %-40s type=%-3u dims=[", t->name, t->ggml_type);
            for (uint32_t d = 0; d < t->n_dims; d++) {
                printf("%llu%s", (unsigned long long)t->dims[d],
                       d + 1 < t->n_dims ? "," : "");
            }
            printf("]\n");
        }
        if (gf->tensor_count > 20)
            printf("  ... (%llu more tensors)\n", (unsigned long long)(gf->tensor_count - 20));
        printf("\n=== Total file size: %llu bytes ===\n",
               (unsigned long long)oc_gguf_map_total_bytes(&mf));
        oc_gguf_map_free(&mf);
        return 0;
    }

    if (args.quantize_input) {
        const char *target = args.quantize_type ? args.quantize_type : "Q4_K_M";
        OcQuantizeConfig qcfg = {
            .input_path  = args.quantize_input,
            .output_path = args.quantize_output ? args.quantize_output : "quantized.gguf",
            .target_type = target,
            .verbose     = args.verbose,
        };
        OcError qe = oc_quantize_model(&qcfg);
        if (qe != OC_OK) {
            fprintf(stderr, "error: quantization failed (%s)\n", oc_error_msg(qe));
            return 1;
        }
        printf("quantization complete: %s → %s (%s)\n",
               args.quantize_input, qcfg.output_path, target);
        return 0;
    }

    if (args.model_path == NULL) {
        /* Placeholder behavior (mirrors Rust CLI when no model is given). */
        printf("oxidize-c v%s (no --model given; echoing prompt)\n", OC_CLI_VERSION);
        if (args.prompt) printf("%s\n", args.prompt);
        return 0;
    }

    if (args.perplexity && args.model_path) {
        /* Perplexity evaluation mode. */
        OcLlamaModel model;
        OcError e = oc_llama_load(args.model_path, &model);
        if (e != OC_OK) {
            fprintf(stderr, "error: failed to load model (%s)\n", oc_error_msg(e));
            return 1;
        }
        cap_model_ctx(&model, args.n_ctx);
        OcTokenizer tok;
        e = oc_tokenizer_load_from_gguf(&model.gguf.unified, &tok);
        if (e != OC_OK) {
            fprintf(stderr, "error: tokenizer load failed (%s)\n", oc_error_msg(e));
            oc_llama_free(&model);
            return 1;
        }
        const char *text = args.prompt;
        char *owned_text = NULL;
        if (!text && args.prompt_file) {
            FILE *pf = fopen(args.prompt_file, "rb");
            if (!pf || fseek(pf, 0, SEEK_END) != 0) {
                if (pf) fclose(pf);
                oc_tokenizer_free(&tok);
                oc_llama_free(&model);
                return 1;
            }
            long sz = ftell(pf);
            if (sz < 0 || fseek(pf, 0, SEEK_SET) != 0) {
                fclose(pf);
                oc_tokenizer_free(&tok);
                oc_llama_free(&model);
                return 1;
            }
            owned_text = malloc((size_t)sz + 1);
            if (!owned_text || fread(owned_text, 1, (size_t)sz, pf) != (size_t)sz) {
                free(owned_text);
                fclose(pf);
                oc_tokenizer_free(&tok);
                oc_llama_free(&model);
                return 1;
            }
            fclose(pf);
            owned_text[sz] = '\0';
            text = owned_text;
        }
        if (!text) text = "The quick brown fox jumps over the lazy dog.";

        OcPerplexityResult result;
        e = oc_perplexity_evaluate(&model, &tok, text, 0, &result);
        if (e != OC_OK) {
            fprintf(stderr, "error: perplexity evaluation failed (%s)\n", oc_error_msg(e));
        } else {
            char fmt[256];
            oc_perplexity_format(&result, fmt, sizeof(fmt));
            printf("%s\n", fmt);
        }
        free(owned_text);
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        return (e == OC_OK) ? 0 : 1;
    }

    if (args.bench) {
        if (args.bench_iterations <= 0) {
            fprintf(stderr, "error: --bench-iters must be greater than zero\n");
            return 1;
        }
        /* Benchmark mode: run N iterations and report tok/s. */
        OcLlamaModel model;
        OcError e = oc_llama_load(args.model_path, &model);
        if (e != OC_OK) {
            fprintf(stderr, "error: failed to load model (%s)\n", oc_error_msg(e));
            return 1;
        }
        cap_model_ctx(&model, args.n_ctx);
        OcKvCacheType bench_kv = oc_llama_select_kv_type(model.cfg.n_ctx, args.kv_type);
        OcTokenizer tok;
        e = oc_tokenizer_load_from_gguf(&model.gguf.unified, &tok);
        if (e != OC_OK) {
            fprintf(stderr, "error: tokenizer load failed (%s)\n", oc_error_msg(e));
            oc_llama_free(&model);
            return 1;
        }
        char *bench_file_prompt = NULL;
        const char *prompt = args.prompt;
        if (!prompt && args.prompt_file) {
            FILE *f = fopen(args.prompt_file, "rb");
            if (!f || fseek(f, 0, SEEK_END) != 0) {
                if (f) fclose(f);
                oc_tokenizer_free(&tok);
                oc_llama_free(&model);
                return 1;
            }
            long size = ftell(f);
            if (size < 0 || fseek(f, 0, SEEK_SET) != 0) {
                fclose(f);
                oc_tokenizer_free(&tok);
                oc_llama_free(&model);
                return 1;
            }
            bench_file_prompt = malloc((size_t)size + 1);
            if (!bench_file_prompt) {
                fclose(f);
                oc_tokenizer_free(&tok);
                oc_llama_free(&model);
                return 1;
            }
            size_t read = fread(bench_file_prompt, 1, (size_t)size, f);
            bench_file_prompt[read] = '\0';
            fclose(f);
            prompt = bench_file_prompt;
        }
        if (!prompt) prompt = "The quick brown fox jumps over the lazy dog.";
        uint32_t *ids = NULL;
        size_t n_ids = 0;
        OcSpecialTokenPolicy pol = tok.has_add_bos_token && tok.add_bos_token
            ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
        e = oc_tokenizer_encode(&tok, prompt, pol, &ids, &n_ids);
        if (e != OC_OK || n_ids == 0) {
            free(bench_file_prompt);
            oc_tokenizer_free(&tok);
            oc_llama_free(&model);
            return 1;
        }

        printf("benchmark: %zu prompt tokens, %d iterations, %d max tokens\n",
               n_ids, args.bench_iterations, args.n_predict);
        double best_tps = 0.0, sum_tps = 0.0;
        int completed_iterations = 0;
        for (int iter = 0; iter < args.bench_iterations; iter++) {
            OcLlamaSession sess;
            if (oc_llama_session_init_kv(&model, &sess, bench_kv) != OC_OK) break;
            float *logits = sess.logits;
            for (size_t i = 0; i + 1 < n_ids && e == OC_OK; i++)
                e = oc_llama_forward(&sess, ids[i], NULL);
            if (e == OC_OK) e = oc_llama_forward(&sess, ids[n_ids - 1], logits);
            if (e != OC_OK) {
                fprintf(stderr, "error: benchmark prefill failed (%s)\n", oc_error_msg(e));
                oc_llama_session_free(&sess);
                break;
            }
            double start = wall_now();
            size_t emitted = 0;
            while (emitted < (size_t)args.n_predict) {
                uint32_t sampled = oc_argmax(logits, model.cfg.vocab_size);
                if (tok.has_eos && sampled == tok.eos_id) break;
                emitted++;
                if (oc_llama_forward(&sess, sampled, logits) != OC_OK) break;
            }
            double elapsed = wall_now() - start;
            double tps = (elapsed > 0) ? (double)emitted / elapsed : 0.0;
            /* Also measure prompt processing (prefill) speed. */
            double pf_start = wall_now();
            OcLlamaSession pf_sess;
            memset(&pf_sess, 0, sizeof(pf_sess));
            if (oc_llama_session_init_kv(&model, &pf_sess, bench_kv) == OC_OK) {
                for (size_t i = 0; i < n_ids; i++)
                    oc_llama_forward(&pf_sess, ids[i], NULL);
            }
            double pf_elapsed = wall_now() - pf_start;
            double pf_tps = (pf_elapsed > 0) ? (double)n_ids / pf_elapsed : 0.0;
            oc_llama_session_free(&pf_sess);
            printf("  iter %d: %zu tokens in %.3fs = %.2f tok/s (prefill: %.2f tok/s)\n",
                   iter + 1, emitted, elapsed, tps, pf_tps);
            if (tps > best_tps) best_tps = tps;
            sum_tps += tps;
            completed_iterations++;
            oc_llama_session_free(&sess);
        }
        free(ids);
        free(bench_file_prompt);
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        if (completed_iterations == 0) return 1;
        printf("benchmark: best=%.2f tok/s, avg=%.2f tok/s\n",
               best_tps, sum_tps / completed_iterations);
        return 0;
    }

    OcError e = run_generation(&args);
    return (e == OC_OK) ? 0 : 1;
}
