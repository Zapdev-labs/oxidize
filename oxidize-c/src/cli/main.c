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
 * --threads). Threads/numa/autotune are accepted as flags; their effect is
 * applied by later features (autotune-plan-apply, server-http-core).
 */
#include "oxidize/activation.h"   /* ensure link for forward deps */
#include "oxidize/autotune.h"
#include "oxidize/cuda.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/http.h"
#include "oxidize/llama.h"
#include "oxidize/log.h"
#include "oxidize/oc.h"
#include "oxidize/openai.h"
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

#define OC_CLI_VERSION "0.2.0"

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
"  --threads N            CPU thread hint (0 = auto)\n"
"  --numa MODE            single | interleave | none (default none)\n"
"  --auto                 Enable autotune (detect + plan)\n"
"  --no-auto              Disable autotune\n"
"  --print-plan           Print the autotune plan and exit\n"
"  --serve-api            Start the OpenAI-compatible HTTP server\n"
"  --host HOST            Server bind host (default 127.0.0.1)\n"
"  --port PORT            Server bind port (default 8080)\n"
"  --temperature T        Sampling temperature (0 = greedy, default 0)\n"
"  --top-k K              Top-K sampling (default 40, 0 = disabled)\n"
"  --top-p P              Top-P / nucleus (default 0.95)\n"
"  --repeat-penalty P     Repeat penalty (default 1.1)\n"
"  --seed N               RNG seed (0 = greedy always)\n"
"  --backend cpu|cuda     Compute backend (default cpu)\n"
"  -v, --verbose          Verbose logging\n"
"  -h, --help             Show this help\n"
"  --version              Print version and exit\n",
    OC_CLI_VERSION);
}

/* ─── Generation ──────────────────────────────────────────────────────── */

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
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
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

    /* If --auto: detect CPU, fingerprint the model, plan, and apply the
     * memory-side policy (hugepages/mlock) to the mmap'd weights. Thread
     * and NUMA policy are read from the plan by the caller's worker pool
     * (not yet wired — single-threaded forward for now). */
    if (args->auto_tune) {
        OcCpuInfo cpu;
        OcModelFingerprint fp;
        if (oc_autotune_detect_cpu(&cpu) == OC_OK &&
            oc_autotune_fingerprint_gguf(&model.gguf, &fp) == OC_OK) {
            OcTuningPlan plan = oc_autotune_plan(&cpu, &fp);
            oc_log(OC_LOG_INFO, "autotune: %u threads, numa=%s, simd=%s",
                   plan.threads, oc_autotune_numa_name(plan.numa),
                   cpu.simd.name);
            oc_autotune_apply(&plan, &model.gguf);
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
    e = oc_llama_session_init(&model, &sess);
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
    OcSamplerConfig scfg;
    scfg.temperature    = args->temperature;
    scfg.repeat_penalty = args->repeat_penalty;
    scfg.seed           = args->seed;
    if (args->temperature <= 0.0f) {
        scfg.type = OC_SAMPLER_GREEDY;
    } else if (args->top_k > 0) {
        scfg.type   = OC_SAMPLER_TOP_K;
        scfg.top_k  = (uint32_t)args->top_k;
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

    /* Prefill: forward all but the last prompt token (no sampling needed for
     * those). The last prompt token's logits seed the generation loop. */
    float *logits = sess.logits;
    for (size_t i = 0; i + 1 < n_ids; i++) {
        if (use_cuda)
            e = oc_cuda_forward(&cuda_ctx, ids[i], sess.pos, NULL);
        else
            e = oc_llama_forward(&sess, ids[i], NULL);   /* KV cache only */
        if (e != OC_OK) {
            fprintf(stderr, "error: prefill forward failed at token %zu (%s)\n",
                    i, oc_error_msg(e));
            break;
        }
        sess.pos++;
        if (recent_len < RECENT_CAP) recent[recent_len++] = ids[i];
        else recent[i % RECENT_CAP] = ids[i];
    }

    /* Forward the last prompt token WITH logits to start generation. */
    uint32_t next_tok = ids[n_ids - 1];
    if (e == OC_OK) {
        if (use_cuda)
            e = oc_cuda_forward(&cuda_ctx, next_tok, sess.pos, logits);
        else
            e = oc_llama_forward(&sess, next_tok, logits);
        sess.pos++;
    }
    if (e != OC_OK) {
        fprintf(stderr, "error: seed forward failed (%s)\n", oc_error_msg(e));
    } else {
        /* Decode + print the prompt's last token context implicitly; generate. */
        size_t emitted = 0;
        bool eos_reached = false;
        clock_t decode_start = clock();
        while (emitted < (size_t)args->n_predict && !eos_reached && e == OC_OK) {
            uint32_t sampled = oc_sample(logits, model.cfg.vocab_size, &scfg,
                                        recent, recent_len);
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
            sess.pos++;
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
            double elapsed = (double)(clock() - decode_start) / CLOCKS_PER_SEC;
            if (elapsed > 0) {
                oc_log(OC_LOG_INFO, "speed: %.2f tok/s", (double)emitted / elapsed);
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

/* ─── Entrypoint ──────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    oc_log_init_from_env();

    OcCliArgs args;
    oc_cli_parse_args(argc, argv, &args);

    if (args.show_help)    { print_help(); return 0; }
    if (args.show_version) { printf("oxidize-c v%s\n", OC_CLI_VERSION); return 0; }

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
        OcHttpServer srv;
        OcError e = oc_http_server_start(args.host, (uint16_t)args.port, 4,
                                         oc_openai_handler, &st, &srv);
        if (e != OC_OK) {
            fprintf(stderr, "error: server start failed (%s)\n", oc_error_msg(e));
            return 1;
        }
        printf("oxidize-c server listening on http://%s:%u\n"
               "  GET  /v1/models\n"
               "  POST /v1/completions\n"
               "  POST /v1/chat/completions\n"
               "(Ctrl+C to stop)\n", args.host, srv.port);
        fflush(stdout);
        oc_http_server_join(&srv);
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

    if (args.quantize_input) {
        /* Quantize mode: re-quantize the input model to the target type. */
        const char *target = args.quantize_type ? args.quantize_type : "Q4_K_M";
        OcQuantizeConfig qcfg = {
            .input_path  = args.quantize_input,
            .output_path = args.quantize_output ? args.quantize_output : "quantized.gguf",
            .target_type = target,
            .verbose     = args.verbose,
            .n_threads   = (size_t)args.threads,
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

    if (args.bench) {
        /* Benchmark mode: run N iterations and report tok/s. */
        OcLlamaModel model;
        OcError e = oc_llama_load(args.model_path, &model);
        if (e != OC_OK) {
            fprintf(stderr, "error: failed to load model (%s)\n", oc_error_msg(e));
            return 1;
        }
        OcTokenizer tok;
        e = oc_tokenizer_load_from_gguf(&model.gguf.unified, &tok);
        if (e != OC_OK) {
            fprintf(stderr, "error: tokenizer load failed (%s)\n", oc_error_msg(e));
            oc_llama_free(&model);
            return 1;
        }
        const char *prompt = args.prompt ? args.prompt : "The quick brown fox jumps over the lazy dog.";
        uint32_t *ids = NULL;
        size_t n_ids = 0;
        OcSpecialTokenPolicy pol = tok.has_add_bos_token && tok.add_bos_token
            ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
        oc_tokenizer_encode(&tok, prompt, pol, &ids, &n_ids);
        if (n_ids == 0) { oc_tokenizer_free(&tok); oc_llama_free(&model); return 1; }

        printf("benchmark: %zu prompt tokens, %d iterations, %d max tokens\n",
               n_ids, args.bench_iterations, args.n_predict);
        double best_tps = 0.0, sum_tps = 0.0;
        for (int iter = 0; iter < args.bench_iterations; iter++) {
            OcLlamaSession sess;
            if (oc_llama_session_init(&model, &sess) != OC_OK) break;
            float *logits = sess.logits;
            for (size_t i = 0; i + 1 < n_ids; i++)
                oc_llama_forward(&sess, ids[i], NULL);
            oc_llama_forward(&sess, ids[n_ids - 1], logits);
            clock_t start = clock();
            size_t emitted = 0;
            while (emitted < (size_t)args.n_predict) {
                uint32_t sampled = oc_argmax(logits, model.cfg.vocab_size);
                if (tok.has_eos && sampled == tok.eos_id) break;
                emitted++;
                if (oc_llama_forward(&sess, sampled, logits) != OC_OK) break;
            }
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            double tps = (elapsed > 0) ? (double)emitted / elapsed : 0.0;
            printf("  iter %d: %zu tokens in %.3fs = %.2f tok/s\n",
                   iter + 1, emitted, elapsed, tps);
            if (tps > best_tps) best_tps = tps;
            sum_tps += tps;
            oc_llama_session_free(&sess);
        }
        free(ids);
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        printf("benchmark: best=%.2f tok/s, avg=%.2f tok/s\n",
               best_tps, sum_tps / args.bench_iterations);
        return 0;
    }

    OcError e = run_generation(&args);
    return (e == OC_OK) ? 0 : 1;
}
