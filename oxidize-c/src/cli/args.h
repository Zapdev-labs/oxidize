/* args.h — CLI argument parsing (extracted for testability). Kept in src/cli/ (not include/) because it's CLI-internal, not library API. */
#ifndef OXIDIZE_CLI_ARGS_H
#define OXIDIZE_CLI_ARGS_H

#include <stdbool.h>
#include <stdint.h>

typedef struct OcCliArgs {
    const char *model_path;
    const char *prompt;
    const char *prompt_file;
    uint32_t   n_predict;
    /* --ctx N: cap the KV cache context below the model's advertised */
    uint32_t   n_ctx;
    const char *kv_type;       /* --kv f32|q8; NULL = auto (q8 if ctx>=8192) */
    int        threads;
    const char *numa;
    bool       auto_tune;
    bool       no_auto;
    bool       print_plan;
    bool       serve_api;
    const char *host;
    int        port;
    const char *api_key;        /* --api-key KEY; NULL leaves auth off     */
    int        rate_limit_rpm;  /* --rate-limit N; 0 leaves it off         */
    const char *cors_origin;    /* --cors-origin ORIGIN; NULL leaves it off */
    float      temperature;
    uint32_t   top_k;
    float      top_p;
    float      repeat_penalty;
    uint64_t   seed;
    /* Speculative decoding. */
    const char *draft_model;       /* path to draft model GGUF            */
    int        draft_tokens;       /* K draft tokens per step (0=default 4) */
    const char *spec_type;         /* none|mtp|dspark (NULL = auto)       */
    /* Quantization mode. */
    const char *quantize_input;   /* input GGUF path                      */
    const char *quantize_output;  /* output GGUF path                     */
    const char *quantize_type;   /* target quant type string (Q4_0, etc.) */
    /* Streaming. */
    bool       stream;
    /* Backend selection. */
    const char *backend;         /* "cpu" (default) or "cuda"             */
    /* Benchmark mode. */
    bool       bench;            /* run benchmark instead of generation   */
    int        bench_iterations;  /* number of benchmark iterations       */
    /* Min-p sampler. */
    float      min_p;            /* 0 = disabled; (0,1] filters low-prob  */
    /* Mirostat. */
    float      mirostat_tau;     /* 0 = disabled; target surprise         */
    float      mirostat_eta;     /* learning rate (default 0.1)           */
    /* Prompt-prefill chunk: tokens fed through the weights per pass. */
    uint32_t   batch_size;
    bool       verbose;
    bool       inspect;           /* print model info and exit             */
    bool       perplexity;        /* compute perplexity on input text       */
    bool       cuda_selftest;     /* run CUDA kernel self-test and exit     */
    bool       show_help;
    bool       show_version;
} OcCliArgs;

void oc_cli_args_defaults(OcCliArgs *a);
void oc_cli_parse_args(int argc, char **argv, OcCliArgs *a);

/* Parse the `oxidize-c <subcommand> [flags]` form into an OcCliContext. */
struct OcCliContext;
bool oc_cli_context_parse(int argc, char **argv, struct OcCliContext *ctx);

#endif /* OXIDIZE_CLI_ARGS_H */
