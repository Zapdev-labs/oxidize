/*
 * cli_commands.h — CLI subcommand dispatch for oxidize-c.
 *
 * Extends the basic --prompt / --serve-api flag-based CLI with a structured
 * subcommand system modeled after the Rust `oxidize-cli` subcommands.
 * Each subcommand (bench, inspect, quantize, convert, merge, prune, etc.)
 * has a dedicated handler that wires into the existing module APIs
 * (benchmark.h, inspect.h, quantize_tool.h, safetensors_to_gguf.h,
 * merge.h, prune.h, finetune.h, perplexity.h, realtime.h, openai.h, etc.).
 *
 * The dispatch model:
 *   1. Parse the first positional argument as a subcommand name
 *      (via oc_cli_command_parse).
 *   2. Build an OcCliContext from the remaining flags
 *      (model path, host/port, output format, verbose, etc.).
 *   3. Call oc_cli_command_run(ctx) which dispatches to the handler.
 *
 * Output formats: every handler supports `--output text` (default,
 * human-readable) and `--output json` (machine-readable JSON on stdout).
 * Progress / diagnostic messages always go to stderr so stdout stays clean
 * for piping.
 *
 * Port of oxidize-cli subcommands to the C11 port.
 */
#ifndef OXIDIZE_CLI_COMMANDS_H
#define OXIDIZE_CLI_COMMANDS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Subcommand enum ────────────────────────────────────────────────────
 *
 * Numeric values are stable (callers may store them). Append-only.
 * OC_CLI_CMD_NONE is the "no subcommand" sentinel. */
typedef enum {
    OC_CLI_CMD_NONE            = 0,
    OC_CLI_CMD_PROMPT          = 1,  /* generation (default)               */
    OC_CLI_CMD_CHAT            = 2,  /* interactive chat loop              */
    OC_CLI_CMD_BENCH           = 3,  /* benchmark forward pass             */
    OC_CLI_CMD_INSPECT         = 4,  /* inspect model metadata             */
    OC_CLI_CMD_SERVE           = 5,  /* start HTTP API server              */
    OC_CLI_CMD_QUANTIZE        = 6,  /* re-quantize a GGUF model           */
    OC_CLI_CMD_CONVERT         = 7,  /* SafeTensors → GGUF                 */
    OC_CLI_CMD_MERGE           = 8,  /* merge checkpoints                  */
    OC_CLI_CMD_PRUNE           = 9,  /* prune model weights                */
    OC_CLI_CMD_FINETUNE        = 10, /* LoRA / SFT finetuning              */
    OC_CLI_CMD_LIST            = 11, /* list cached models                */
    OC_CLI_CMD_DOWNLOAD        = 12, /* download from HuggingFace          */
    OC_CLI_CMD_TOKENIZE        = 13, /* tokenize text → token IDs          */
    OC_CLI_CMD_DETOKENIZE      = 14, /* token IDs → text                   */
    OC_CLI_CMD_PERPLEXITY      = 15, /* compute perplexity                 */
    OC_CLI_CMD_SERVE_REALTIME  = 16, /* start WebSocket realtime server    */
} OcCliCommand;

/* ─── Output format ────────────────────────────────────────────────────── */
typedef enum {
    OC_CLI_OUTPUT_TEXT = 0,   /* human-readable (default)               */
    OC_CLI_OUTPUT_JSON = 1,   /* machine-readable JSON                 */
} OcCliOutputFormat;

/* ─── CLI context ────────────────────────────────────────────────────────
 *
 * Aggregates all flags needed by any subcommand handler. Fields not
 * relevant to a given command are simply ignored (NULL / 0 / false). */
typedef struct OcCliContext {
    /* Subcommand + output. */
    OcCliCommand       command;
    OcCliOutputFormat  output_format;
    bool               verbose;

    /* Model + generation. */
    const char        *model_path;     /* --model PATH                      */
    const char        *prompt;         /* --prompt TEXT / positional       */
    const char        *prompt_file;    /* --prompt-file PATH                */
    uint32_t           n_predict;      /* --n-predict N (default 128)       */
    /* --ctx N: KV context length. 0 means "use the default cap"; see
     * OC_CLI_DEFAULT_MAX_CTX below. Pass a value larger than the cap to
     * request it explicitly. */
    uint32_t           n_ctx;
    int                threads;        /* --threads N (0 = auto)            */
    const char        *numa;           /* --numa single|interleave|none     */
    bool               auto_tune;      /* --auto                            */
    bool               no_auto;        /* --no-auto                         */

    /* Sampling. */
    float              temperature;
    uint32_t           top_k;
    float              top_p;
    float              repeat_penalty;
    uint64_t           seed;
    float              min_p;
    float              mirostat_tau;
    float              mirostat_eta;

    /* Backend. */
    const char        *backend;        /* "cpu" | "cuda"                    */

    /* Server. */
    const char        *host;           /* --host (default 127.0.0.1)        */
    int                port;           /* --port (default 8080)             */
    const char        *api_key;        /* --api-key KEY (NULL = no auth)    */
    uint32_t           rate_limit_rpm; /* --rate-limit N (0 = unlimited)    */
    const char        *cors_origin;    /* --cors-origin ORIGIN (NULL = off) */

    /* Benchmark. */
    int                bench_iterations; /* --bench-iters (default 3)       */
    uint32_t           bench_warmup;     /* --bench-warmup (default 5)      */
    uint32_t           bench_tokens;     /* --bench-tokens (default 50)     */

    /* Quantize / convert / merge / prune. */
    const char        *input_path;      /* --input PATH (for convert/merge)  */
    const char        *output_path;     /* --output PATH                     */
    const char        *target_type;     /* --quant-type / --target (Q4_K_M)  */
    const char        *arch;            /* --arch (llama, qwen2, ...)        */
    const char        *merge_strategy;  /* --strategy linear|slerp|ties|dare */
    float              merge_slerp_t;   /* --slerp-t (default 0.5)           */
    float              merge_density;   /* --density (TIES, default 0.5)     */
    float              prune_sparsity;  /* --sparsity (default 0.5)          */
    const char        *prune_strategy;  /* --strategy wanda|magnitude        */

    /* Finetune. */
    const char        *dataset_path;    /* --dataset PATH (JSONL)            */
    const char        *output_dir;      /* --output-dir PATH                 */
    const char        *resume_from;     /* --resume-from PATH                */
    const char        *ft_strategy;     /* --strategy sft|self-train|dpo|ppo */
    uint32_t           lora_rank;       /* --lora-rank (default 8)           */
    uint32_t           lora_alpha;      /* --lora-alpha (default 16)         */
    uint32_t           epochs;          /* --epochs (default 3)              */
    uint32_t           batch_size;      /* --batch-size (default 1)          */
    float              learning_rate;   /* --lr (default 1e-4)               */

    /* Download. */
    const char        *hf_repo;         /* --repo (e.g. "Qwen/Qwen2.5-7B")   */
    const char        *hf_file;         /* --file (specific file in repo)    */
    const char        *cache_dir;       /* --cache-dir (default ~/.cache/oxidize/hf) */

    /* Perplexity. */
    size_t             ppl_max_tokens;  /* --max-tokens (0 = all)            */

    /* Tokenize / detokenize. */
    const char        *token_ids_str;   /* --ids "1,2,3" (for detokenize)    */
    bool               tokens_no_special; /* --no-special (disallow special) */
} OcCliContext;

/* Default cap on the KV context when --ctx is not given.
 *
 * The KV cache is allocated eagerly for the whole context, so using the
 * model's advertised context_length as the default is not viable on modern
 * long-context models: Llama-3.1-8B advertises 131072, which at its 1024-float
 * KV row is 34 GB of f32 cache for an 8B model. That allocation dominates
 * startup and thrashes memory before a single token is produced. llama.cpp
 * makes the same choice for the same reason.
 *
 * Capping is announced at INFO. `--ctx 0` restores the model's own value. */
#define OC_CLI_DEFAULT_MAX_CTX 4096u

/* Resolve the effective context for `model` under `ctx` and write it into the
 * model config, logging whenever the advertised value is reduced. Call once
 * after loading, before creating a session. */
struct OcLlamaModel;
void oc_cli_apply_ctx(const OcCliContext *ctx, struct OcLlamaModel *model);

/* ─── Enum ↔ string ───────────────────────────────────────────────────── */

/* Convert a command enum to its canonical name string.
 * Returns "none" for OC_CLI_CMD_NONE. Never returns NULL. */
const char *oc_cli_command_name(OcCliCommand cmd);

/* Parse a command name string into the enum. Case-insensitive.
 * Returns OC_CLI_CMD_NONE if the name is unrecognized. */
OcCliCommand oc_cli_command_parse(const char *name);

/* ─── Context helpers ──────────────────────────────────────────────────── */

/* Zero-initialize an OcCliContext with sensible defaults. */
void oc_cli_context_defaults(OcCliContext *ctx);

/* ─── Dispatch ─────────────────────────────────────────────────────────── */

/* Dispatch to the appropriate command handler based on ctx->command.
 * Returns OC_OK on success, or an error code on failure.
 * Prints user-friendly error messages to stderr. */
OcError oc_cli_command_run(OcCliContext *ctx);

/* Print top-level help text (subcommand list + common flags) to stdout. */
void oc_cli_command_help(void);

/* Print help text for a specific subcommand to stdout.
 * If cmd is OC_CLI_CMD_NONE, prints the top-level help. */
void oc_cli_command_help_for(OcCliCommand cmd);

/* ─── Individual command handlers ────────────────────────────────────────
 *
 * Each handler takes an OcCliContext* and returns OC_OK or an error.
 * Handlers print results to stdout (text or JSON depending on
 * ctx->output_format) and progress/diagnostics to stderr.
 * On error, a user-friendly message is printed to stderr. */

/* Run a benchmark: load model, run N iterations, print tok/s. */
OcError oc_cli_run_bench(OcCliContext *ctx);

/* Inspect a model: print architecture, layers, tensors, quant type. */
OcError oc_cli_run_inspect(OcCliContext *ctx);

/* Quantize a GGUF model to a different quantization type. */
OcError oc_cli_run_quantize(OcCliContext *ctx);

/* Convert a SafeTensors checkpoint to GGUF format. */
OcError oc_cli_run_convert(OcCliContext *ctx);

/* Merge multiple GGUF checkpoints into one. */
OcError oc_cli_run_merge(OcCliContext *ctx);

/* Prune model weights (Wanda or magnitude). */
OcError oc_cli_run_prune(OcCliContext *ctx);

/* Fine-tune a model with LoRA / SFT. */
OcError oc_cli_run_finetune(OcCliContext *ctx);

/* List available models in the cache directory. */
OcError oc_cli_run_list(OcCliContext *ctx);

/* Download a model from HuggingFace Hub. */
OcError oc_cli_run_download(OcCliContext *ctx);

/* Tokenize text and print token IDs. */
OcError oc_cli_run_tokenize(OcCliContext *ctx);

/* Detokenize token IDs and print text. */
OcError oc_cli_run_detokenize(OcCliContext *ctx);

/* Compute perplexity of a model on input text. */
OcError oc_cli_run_perplexity(OcCliContext *ctx);

/* Start the OpenAI-compatible HTTP server. */
OcError oc_cli_run_serve(OcCliContext *ctx);

/* Start the WebSocket realtime server. */
OcError oc_cli_run_serve_realtime(OcCliContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CLI_COMMANDS_H */
