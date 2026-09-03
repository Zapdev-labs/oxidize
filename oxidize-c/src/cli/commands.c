/* strdup is POSIX.1-2008; needs _POSIX_C_SOURCE to be declared. Must be
 * the first non-comment thing in the file, before any system header. */
#define _POSIX_C_SOURCE 200809L

/*
 * commands.c — CLI subcommand implementation for oxidize-c.
 *
 * Port of oxidize-cli subcommands to the C11 port. Provides:
 *   - Command name <-> enum mapping
 *   - OcCliContext initialization with defaults
 *   - Dispatch to per-command handlers
 *   - Each handler wires into the existing module APIs (benchmark.h,
 *     inspect.h, quantize_tool.h, safetensors_to_gguf.h, merge.h,
 *     prune.h, finetune.h, perplexity.h, realtime.h, openai.h, etc.)
 *   - Text and JSON output modes
 *   - Progress reporting to stderr
 *
 * Compile: cc -std=c11 -Wall -Wextra -Werror -O2 -c src/cli/commands.c -I include
 */
#include "oxidize/cli_commands.h"

#include "oxidize/benchmark.h"
#include "oxidize/error.h"
#include "oxidize/finetune.h"
#include "oxidize/gguf.h"
#include "oxidize/hf_hub.h"
#include "oxidize/http.h"
#include "oxidize/inspect.h"
#include "oxidize/llama.h"
#include "oxidize/dspark.h"
#include "oxidize/log.h"
#include "oxidize/merge.h"
#include "oxidize/openai.h"
#include "oxidize/perplexity.h"
#include "oxidize/prune.h"
#include "oxidize/quantize_tool.h"
#include "oxidize/realtime.h"
#include "oxidize/safetensors_to_gguf.h"
#include "oxidize/sampling.h"
#include "oxidize/tokenizer.h"
#include "oxidize/websocket.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static double wall_now(void)
{
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Case-insensitive string equality. */
static bool ieq(const char *a, const char *b)
{
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return false;
        a++; b++;
    }
    return *a == *b;
}

/* Print progress/diagnostic info to stderr (stdout stays clean). */
static void cli_info(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dflash2] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

/* JSON-escape a string into buf. Returns bytes written (excl NUL), 0 on overflow. */
static size_t json_escape(const char *s, char *buf, size_t cap)
{
    if (cap == 0) return 0;
    size_t n = 0;
    while (*s && n + 2 < cap) {
        char c = *s++;
        switch (c) {
        case '"':  if (n + 2 >= cap) return 0; buf[n++] = '\\'; buf[n++] = '"'; break;
        case '\\': if (n + 2 >= cap) return 0; buf[n++] = '\\'; buf[n++] = '\\'; break;
        case '\n': if (n + 2 >= cap) return 0; buf[n++] = '\\'; buf[n++] = 'n'; break;
        case '\r': if (n + 2 >= cap) return 0; buf[n++] = '\\'; buf[n++] = 'r'; break;
        case '\t': if (n + 2 >= cap) return 0; buf[n++] = '\\'; buf[n++] = 't'; break;
        default:
            if ((unsigned char)c < 0x20) {
                int w = snprintf(buf + n, cap - n, "\\u%04x", (unsigned)(unsigned char)c);
                if (w < 0 || (size_t)w >= cap - n) return 0;
                n += (size_t)w;
            } else {
                buf[n++] = c;
            }
        }
    }
    buf[n] = '\0';
    return n;
}

/* Print a progress message to stderr. */
static void progress(const OcCliContext *ctx, const char *fmt, ...)
{
    if (!ctx->verbose) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[cli] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

/* Print a user-friendly error to stderr. */
static void cli_error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static OcError cli_session_init(const OcCliContext *ctx, OcLlamaModel *model,
                                OcLlamaSession *sess)
{
    OcKvCacheType kv = oc_llama_select_kv_type(model->cfg.n_ctx, ctx->kv_type);
    return oc_llama_session_init_kv(model, sess, kv);
}

void oc_cli_apply_ctx(const OcCliContext *ctx, struct OcLlamaModel *model)
{
    if (ctx == NULL || model == NULL) return;
    OcLlamaModel *m = (OcLlamaModel *)model;
    const uint32_t advertised = m->cfg.n_ctx;
    if (advertised == 0) return;

    uint32_t want;
    if (ctx->n_ctx > 0) {
        want = ctx->n_ctx;                       /* explicit --ctx N       */
    } else {
        want = OC_CLI_DEFAULT_MAX_CTX;           /* default cap            */
    }
    /* Never extend past what the model was trained for — a larger KV cache
     * than the advertised context buys nothing and the position encodings
     * are undefined out there. */
    if (want > advertised) want = advertised;
    if (want == advertised) return;

    oc_log(OC_LOG_INFO,
           "llama: context capped to %u (model advertises %u); "
           "pass --ctx N to change, --ctx %u for the full context",
           want, advertised, advertised);
    m->cfg.n_ctx = want;
}

/* Read entire file into a malloc'd buffer. Returns NULL on error. */
static char *read_file_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

/* Get default HF cache directory. Returns a static buffer. */
static const char *default_cache_dir(void)
{
    static char buf[512];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(buf, sizeof(buf), "%s/.cache/oxidize/hf", home);
    return buf;
}

/* Check if a path is a directory. */
static bool is_directory(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

/* Check if a file has a .gguf extension (case-insensitive). */
static bool has_gguf_ext(const char *name)
{
    size_t len = strlen(name);
    if (len < 5) return false;
    return ieq(name + len - 5, ".gguf");
}

/* ─── Command name <-> enum ────────────────────────────────────────────── */

const char *oc_cli_command_name(OcCliCommand cmd)
{
    switch (cmd) {
    case OC_CLI_CMD_PROMPT:         return "prompt";
    case OC_CLI_CMD_CHAT:           return "chat";
    case OC_CLI_CMD_BENCH:          return "bench";
    case OC_CLI_CMD_INSPECT:        return "inspect";
    case OC_CLI_CMD_SERVE:          return "serve";
    case OC_CLI_CMD_QUANTIZE:       return "quantize";
    case OC_CLI_CMD_CONVERT:        return "convert";
    case OC_CLI_CMD_MERGE:          return "merge";
    case OC_CLI_CMD_PRUNE:          return "prune";
    case OC_CLI_CMD_FINETUNE:       return "finetune";
    case OC_CLI_CMD_LIST:           return "list";
    case OC_CLI_CMD_DOWNLOAD:       return "download";
    case OC_CLI_CMD_TOKENIZE:       return "tokenize";
    case OC_CLI_CMD_DETOKENIZE:     return "detokenize";
    case OC_CLI_CMD_PERPLEXITY:     return "perplexity";
    case OC_CLI_CMD_SERVE_REALTIME: return "serve-realtime";
    case OC_CLI_CMD_DFLASH2:        return "dflash2";
    case OC_CLI_CMD_NONE:           break;
    }
    return "none";
}

OcCliCommand oc_cli_command_parse(const char *name)
{
    if (!name) return OC_CLI_CMD_NONE;
    if (ieq(name, "prompt"))          return OC_CLI_CMD_PROMPT;
    if (ieq(name, "chat"))            return OC_CLI_CMD_CHAT;
    if (ieq(name, "bench"))           return OC_CLI_CMD_BENCH;
    if (ieq(name, "inspect"))         return OC_CLI_CMD_INSPECT;
    if (ieq(name, "serve"))           return OC_CLI_CMD_SERVE;
    if (ieq(name, "quantize"))       return OC_CLI_CMD_QUANTIZE;
    if (ieq(name, "convert"))         return OC_CLI_CMD_CONVERT;
    if (ieq(name, "merge"))           return OC_CLI_CMD_MERGE;
    if (ieq(name, "prune"))           return OC_CLI_CMD_PRUNE;
    if (ieq(name, "finetune") || ieq(name, "fine-tune"))
                                      return OC_CLI_CMD_FINETUNE;
    if (ieq(name, "list"))            return OC_CLI_CMD_LIST;
    if (ieq(name, "download") || ieq(name, "pull"))
                                      return OC_CLI_CMD_DOWNLOAD;
    if (ieq(name, "tokenize") || ieq(name, "tok"))
                                      return OC_CLI_CMD_TOKENIZE;
    if (ieq(name, "detokenize") || ieq(name, "detok"))
                                      return OC_CLI_CMD_DETOKENIZE;
    if (ieq(name, "perplexity") || ieq(name, "ppl"))
                                      return OC_CLI_CMD_PERPLEXITY;
    if (ieq(name, "serve-realtime") || ieq(name, "realtime"))
                                      return OC_CLI_CMD_SERVE_REALTIME;
    if (ieq(name, "dflash2"))         return OC_CLI_CMD_DFLASH2;
    return OC_CLI_CMD_NONE;
}

/* ─── Context defaults ─────────────────────────────────────────────────── */

void oc_cli_context_defaults(OcCliContext *ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->command          = OC_CLI_CMD_NONE;
    ctx->output_format    = OC_CLI_OUTPUT_TEXT;
    ctx->verbose          = false;
    ctx->n_predict        = 128;
    ctx->threads          = 0;
    ctx->numa             = "none";
    ctx->temperature      = 0.0f;
    ctx->top_k            = 40;
    ctx->top_p            = 0.95f;
    ctx->repeat_penalty   = 1.1f;
    ctx->seed             = 0;
    ctx->min_p            = 0.0f;
    ctx->mirostat_tau     = 0.0f;
    ctx->mirostat_eta     = 0.1f;
    ctx->backend          = "cpu";
    ctx->host             = "127.0.0.1";
    ctx->port             = 8080;
    ctx->bench_iterations = 3;
    ctx->bench_warmup     = 5;
    ctx->bench_tokens     = 50;
    ctx->merge_slerp_t    = 0.5f;
    ctx->merge_density    = 0.5f;
    ctx->prune_sparsity   = 0.5f;
    ctx->prune_strategy   = "wanda";
    ctx->ft_strategy      = "sft";
    ctx->lora_rank        = 8;
    ctx->lora_alpha       = 16;
    ctx->epochs           = 3;
    ctx->batch_size       = 1;
    ctx->learning_rate    = 1e-4f;
    ctx->cache_dir        = NULL; /* lazily defaulted */
    ctx->ppl_max_tokens   = 0;
}

/* ─── Dispatch ─────────────────────────────────────────────────────────── */

OcError oc_cli_command_run(OcCliContext *ctx)
{
    if (!ctx) return OC_ERR_INVALID_ARG;

    switch (ctx->command) {
    case OC_CLI_CMD_BENCH:           return oc_cli_run_bench(ctx);
    case OC_CLI_CMD_INSPECT:        return oc_cli_run_inspect(ctx);
    case OC_CLI_CMD_QUANTIZE:        return oc_cli_run_quantize(ctx);
    case OC_CLI_CMD_CONVERT:         return oc_cli_run_convert(ctx);
    case OC_CLI_CMD_MERGE:           return oc_cli_run_merge(ctx);
    case OC_CLI_CMD_PRUNE:           return oc_cli_run_prune(ctx);
    case OC_CLI_CMD_FINETUNE:        return oc_cli_run_finetune(ctx);
    case OC_CLI_CMD_LIST:            return oc_cli_run_list(ctx);
    case OC_CLI_CMD_DOWNLOAD:        return oc_cli_run_download(ctx);
    case OC_CLI_CMD_TOKENIZE:        return oc_cli_run_tokenize(ctx);
    case OC_CLI_CMD_DETOKENIZE:      return oc_cli_run_detokenize(ctx);
    case OC_CLI_CMD_PERPLEXITY:      return oc_cli_run_perplexity(ctx);
    case OC_CLI_CMD_SERVE:           return oc_cli_run_serve(ctx);
    case OC_CLI_CMD_SERVE_REALTIME:  return oc_cli_run_serve_realtime(ctx);
    case OC_CLI_CMD_DFLASH2:        return oc_cli_run_dflash2(ctx);
    /* PROMPT and CHAT are handled by the generation path in main.c. */
    case OC_CLI_CMD_PROMPT:
    case OC_CLI_CMD_CHAT:
    case OC_CLI_CMD_NONE:
        cli_error("no subcommand given (run `oxidize-c help` for usage)");
        return OC_ERR_INVALID_ARG;
    }
    return OC_ERR_INVALID_ARG;
}

/* ─── Help text ───────────────────────────────────────────────────────── */

void oc_cli_command_help(void)
{
    printf(
"oxidize-c — dependency-free C11 LLM inference\n"
"\n"
"USAGE:\n"
"  oxidize-c <SUBCOMMAND> [OPTIONS]\n"
"  oxidize-c --model <path.gguf> --prompt \"text\"  (legacy flag mode)\n"
"\n"
"SUBCOMMANDS:\n"
"  prompt           Generate text from a prompt (default)\n"
"  chat             Interactive chat session\n"
"  bench            Benchmark inference speed (tok/s)\n"
"  inspect          Inspect a model's metadata and architecture\n"
"  serve            Start the OpenAI-compatible HTTP server\n"
"  serve-realtime   Start the WebSocket realtime server\n"
"  quantize         Re-quantize a GGUF model to a different type\n"
"  convert          Convert SafeTensors checkpoint to GGUF\n"
"  merge            Merge multiple GGUF checkpoints\n"
"  prune            Prune model weights (Wanda / magnitude)\n"
"  finetune         Fine-tune a model with LoRA / SFT\n"
"  list             List available models in the cache\n"
"  download         Download a model from HuggingFace Hub\n"
"  tokenize         Tokenize text and print token IDs\n"
"  detokenize       Detokenize token IDs back to text\n"
"  perplexity       Compute perplexity on input text\n"
"\n"
"COMMON OPTIONS:\n"
"  --model PATH          GGUF model file\n"
"  --output text|json    Output format (default: text)\n"
"  --threads N           CPU thread hint (0 = auto)\n"
"  --kv f32|q8           KV cache dtype (q8 auto when ctx>=8192)\n"
"  --ctx N               KV context length (default cap 4096)\n"
"  --verbose, -v         Verbose logging to stderr\n"
"  --help, -h            Show help\n"
"  --version             Print version\n"
"\n"
"Run `oxidize-c help <SUBCOMMAND>` for per-subcommand options.\n");
}

void oc_cli_command_help_for(OcCliCommand cmd)
{
    if (cmd == OC_CLI_CMD_NONE) { oc_cli_command_help(); return; }

    printf("oxidize-c %s — ", oc_cli_command_name(cmd));
    switch (cmd) {
    case OC_CLI_CMD_PROMPT:
        printf("generate text from a prompt\n\n"
               "USAGE: oxidize-c prompt --model <path> --prompt \"text\" [OPTIONS]\n\n"
               "OPTIONS:\n"
               "  --prompt TEXT         Prompt text\n"
               "  --prompt-file PATH    Read prompt from file\n"
               "  --n-predict N         Max tokens to generate (default 128)\n"
               "  --temperature T       Sampling temperature (0 = greedy)\n"
               "  --top-k K             Top-K sampling\n"
               "  --top-p P             Top-P / nucleus sampling\n"
               "  --seed N              RNG seed\n");
        break;
    case OC_CLI_CMD_CHAT:
        printf("interactive chat session\n\n"
               "USAGE: oxidize-c chat --model <path> [OPTIONS]\n\n"
               "Type messages; Ctrl+D to exit.\n"
               "OPTIONS:\n"
               "  --temperature T       Sampling temperature\n"
               "  --top-k K             Top-K sampling\n");
        break;
    case OC_CLI_CMD_BENCH:
        printf("benchmark inference speed\n\n"
               "USAGE: oxidize-c bench --model <path> [OPTIONS]\n\n"
               "OPTIONS:\n"
               "  --bench-iters N       Number of iterations (default 3)\n"
               "  --bench-warmup N      Unreported warmup iterations (default 5)\n"
               "  --bench-tokens N      Measured tokens per iteration (default 50)\n"
               "  --bench-prompt-tokens N  Exact synthetic prompt-token count\n"
               "  --bench-decode-tokens N  Exact decode-token count\n"
               "  --bench-no-eos        Do not stop decode at EOS\n"
               "  --prompt TEXT          Prompt to use for benchmarking\n");
        break;
    case OC_CLI_CMD_INSPECT:
        printf("inspect model metadata and architecture\n\n"
               "USAGE: oxidize-c inspect --model <path> [OPTIONS]\n\n"
               "OPTIONS:\n"
               "  --output text|json    Output format (default: text)\n");
        break;
    case OC_CLI_CMD_SERVE:
        printf("start OpenAI-compatible HTTP server\n\n"
               "USAGE: oxidize-c serve --model <path> [OPTIONS]\n\n"
               "OPTIONS:\n"
               "  --host HOST           Bind host (default 127.0.0.1)\n"
               "  --port PORT           Bind port (default 8080)\n");
        break;
    case OC_CLI_CMD_SERVE_REALTIME:
        printf("start WebSocket realtime server\n\n"
               "USAGE: oxidize-c serve-realtime --model <path> [OPTIONS]\n\n"
               "OPTIONS:\n"
               "  --host HOST           Bind host (default 127.0.0.1)\n"
               "  --port PORT           Bind port (default 8080)\n");
        break;
    case OC_CLI_CMD_DFLASH2:
        printf("DFlash2 draft propose benchmark\n\n"
               "USAGE: oxidize-c dflash2 --model <dir-or-model.safetensors> [OPTIONS]\n\n"
               "OPTIONS:\n"
               "  --bench-iters N       Propose steps to time (default 20)\n"
               "  --bench-warmup N      Untimed warmup steps (default 3)\n"
               "  --bench-prompt-tokens N  Synthetic context rows per step (default 1)\n"
               "  --lm-materialize       Pre-materialize the lm_head in BF16 so the vocab\n"
               "                          scan measures real streaming bandwidth (~1.27 GB)\n"
               "  --threads N          Worker threads (default 8)\n"
               "  --seed N              Deterministic input seed (default 42)\n"
               "  --json                Machine-readable results\n");
        break;
    case OC_CLI_CMD_QUANTIZE:
        printf("re-quantize a GGUF model\n\n"
               "USAGE: oxidize-c quantize --model <input.gguf> --output <output.gguf> \\\n"
               "       --quant-type Q4_K_M\n\n"
               "OPTIONS:\n"
               "  --model PATH          Input GGUF path\n"
               "  --output PATH         Output GGUF path\n"
               "  --quant-type TYPE     Target type: Q4_0, Q4_K_M, Q4_K_S, Q8_0, F16\n");
        break;
    case OC_CLI_CMD_CONVERT:
        printf("convert SafeTensors checkpoint to GGUF\n\n"
               "USAGE: oxidize-c convert --input <model_dir/> --output <model.gguf> \\\n"
               "       --quant-type Q4_K_M --arch llama\n\n"
               "OPTIONS:\n"
               "  --input PATH          SafeTensors file or directory\n"
               "  --output PATH         Output GGUF path\n"
               "  --quant-type TYPE     Target quant type\n"
               "  --arch NAME           Architecture override (llama, qwen2, ...)\n");
        break;
    case OC_CLI_CMD_MERGE:
        printf("merge multiple GGUF checkpoints\n\n"
               "USAGE: oxidize-c merge --strategy linear --input a.gguf --input b.gguf \\\n"
               "       --output merged.gguf\n\n"
               "OPTIONS:\n"
               "  --strategy S          linear|slerp|ties|dare (default: linear)\n"
               "  --input PATH          Input GGUF (repeat for multiple)\n"
               "  --output PATH         Output GGUF path\n"
               "  --slerp-t T           SLERP interpolation (default 0.5)\n"
               "  --density F           TIES density (default 0.5)\n");
        break;
    case OC_CLI_CMD_PRUNE:
        printf("prune model weights\n\n"
               "USAGE: oxidize-c prune --model <input.gguf> --output <pruned.gguf> \\\n"
               "       --strategy wanda --sparsity 0.5\n\n"
               "OPTIONS:\n"
               "  --strategy S         wanda|magnitude (default: wanda)\n"
               "  --sparsity F         Fraction to prune [0,1) (default 0.5)\n");
        break;
    case OC_CLI_CMD_FINETUNE:
        printf("fine-tune with LoRA / SFT\n\n"
               "USAGE: oxidize-c finetune --model <base.gguf> --dataset data.jsonl \\\n"
               "       --output-dir ./adapters --strategy sft\n\n"
               "OPTIONS:\n"
               "  --strategy S         sft|self-train|dpo|ppo (default: sft)\n"
               "  --dataset PATH       JSONL training data\n"
               "  --output-dir PATH    Output directory for adapters\n"
               "  --lora-rank N        LoRA rank (default 8)\n"
               "  --lora-alpha N      LoRA alpha (default 16)\n"
               "  --epochs N           Training epochs (default 3)\n"
               "  --lr F               Learning rate (default 1e-4)\n");
        break;
    case OC_CLI_CMD_LIST:
        printf("list available models in cache\n\n"
               "USAGE: oxidize-c list [OPTIONS]\n\n"
               "OPTIONS:\n"
               "  --cache-dir PATH     Cache directory (default ~/.cache/oxidize/hf)\n"
               "  --output text|json   Output format (default: text)\n");
        break;
    case OC_CLI_CMD_DOWNLOAD:
        printf("download a model from HuggingFace Hub\n\n"
               "USAGE: oxidize-c download --repo Qwen/Qwen2.5-7B [OPTIONS]\n\n"
               "OPTIONS:\n"
               "  --repo NAME          HF repository (e.g. Qwen/Qwen2.5-7B)\n"
               "  --file NAME          Specific file in repo (default: auto-detect .gguf)\n"
               "  --cache-dir PATH     Cache directory (default ~/.cache/oxidize/hf)\n");
        break;
    case OC_CLI_CMD_TOKENIZE:
        printf("tokenize text and print token IDs\n\n"
               "USAGE: oxidize-c tokenize --model <path> --prompt \"text\"\n\n"
               "OPTIONS:\n"
               "  --prompt TEXT        Text to tokenize\n"
               "  --no-special         Disallow special tokens\n");
        break;
    case OC_CLI_CMD_DETOKENIZE:
        printf("detokenize token IDs back to text\n\n"
               "USAGE: oxidize-c detokenize --model <path> --ids \"1,2,3\"\n\n"
               "OPTIONS:\n"
               "  --ids IDS            Comma-separated token IDs\n");
        break;
    case OC_CLI_CMD_PERPLEXITY:
        printf("compute perplexity on input text\n\n"
               "USAGE: oxidize-c perplexity --model <path> --prompt \"text\"\n\n"
               "OPTIONS:\n"
               "  --prompt TEXT        Text to evaluate\n"
               "  --prompt-file PATH   Read text from file\n"
               "  --max-tokens N       Max tokens to evaluate (0 = all)\n");
        break;
    case OC_CLI_CMD_NONE:
        oc_cli_command_help();
        return;
    }
    printf("\n");
}

/* ─── bench ────────────────────────────────────────────────────────────── */

OcError oc_cli_run_bench(OcCliContext *ctx)
{
    if (!ctx->model_path) {
        cli_error("--model is required for bench");
        return OC_ERR_INVALID_ARG;
    }
    if (ctx->bench_iterations <= 0) {
        cli_error("--bench-iters must be greater than zero");
        return OC_ERR_INVALID_ARG;
    }

    progress(ctx, "loading model: %s", ctx->model_path);
    OcLlamaModel model;
    OcError e = oc_llama_load(ctx->model_path, &model);
    if (e != OC_OK) {
        cli_error("failed to load model (%s)", oc_error_msg(e));
        return e;
    }
    oc_cli_apply_ctx(ctx, &model);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&model.gguf.unified, &tok);
    if (e != OC_OK) {
        cli_error("tokenizer load failed (%s)", oc_error_msg(e));
        oc_llama_free(&model);
        return e;
    }
    const char *prompt = ctx->prompt;
    if (!prompt) prompt = "The quick brown fox jumps over the lazy dog.";

    uint32_t *ids = NULL;
    size_t n_ids = 0;
    OcSpecialTokenPolicy pol = (tok.has_add_bos_token && tok.add_bos_token)
        ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
    e = oc_tokenizer_encode(&tok, prompt, pol, &ids, &n_ids);
    if (e != OC_OK || n_ids == 0) {
        cli_error("prompt encode failed (%s)", oc_error_msg(e));
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        return e != OC_OK ? e : OC_ERR_TOKENIZER;
    }

    uint32_t *synthetic_ids = NULL;
    const uint32_t *bench_ids = ids;
    if (ctx->bench_prompt_tokens > 0) {
        synthetic_ids = malloc((size_t)ctx->bench_prompt_tokens * sizeof(*synthetic_ids));
        if (synthetic_ids == NULL) {
            free(ids);
            oc_tokenizer_free(&tok);
            oc_llama_free(&model);
            return OC_ERR_OOM;
        }
        for (uint32_t i = 0; i < ctx->bench_prompt_tokens; i++)
            synthetic_ids[i] = ids[i % n_ids];
        bench_ids = synthetic_ids;
        n_ids = ctx->bench_prompt_tokens;
    }
    uint32_t decode_tokens = ctx->bench_decode_tokens > 0
        ? ctx->bench_decode_tokens : ctx->bench_tokens;

    progress(ctx, "prompt: %zu tokens, %d iterations, %u max tokens",
             n_ids, ctx->bench_iterations, ctx->bench_tokens);

    double best_tps = 0.0, sum_tps = 0.0;
    double best_pf = 0.0, sum_pf = 0.0;
    int completed = 0;

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"bench\",\"model\":\"%s\",\"prompt_tokens\":%zu,"
               "\"iterations\":%d,\"results\":[",
               ctx->model_path, n_ids, ctx->bench_iterations);
    } else {
        printf("benchmark: %zu prompt tokens, %d iterations, %u max tokens\n",
               n_ids, ctx->bench_iterations, ctx->bench_tokens);
    }

    const uint64_t total_iterations =
        (uint64_t)ctx->bench_warmup + (uint64_t)ctx->bench_iterations;
    for (uint64_t iter = 0; iter < total_iterations; iter++) {
        OcLlamaSession sess;
        if (cli_session_init(ctx, &model, &sess) != OC_OK) break;
        float *logits = sess.logits;

        double pf_start = wall_now();
        e = oc_llama_prefill(&sess, bench_ids, n_ids, 0, logits);
        double pf_elapsed = wall_now() - pf_start;
        if (e != OC_OK) {
            cli_error("benchmark prefill failed (%s)", oc_error_msg(e));
            oc_llama_session_free(&sess);
            break;
        }
        double pf_tps = (pf_elapsed > 0)
                      ? (double)n_ids / pf_elapsed : 0.0;

        double start = wall_now();
        size_t emitted = 0;
        while (emitted < decode_tokens) {
            uint32_t toks[8];
            size_t n = 0;
            size_t want = decode_tokens - emitted;
            if (want > 8) want = 8;
            OcDsparkConfig dcfg;
            oc_dspark_config_init(&dcfg);
            if (oc_dspark_advance(&sess, logits, &dcfg, toks, want, &n, NULL) != OC_OK)
                break;
            if (n == 0) break;
            if (!ctx->bench_no_eos && tok.has_eos) {
                size_t keep = 0;
                for (; keep < n; keep++) {
                    if (toks[keep] == tok.eos_id) break;
                }
                emitted += keep;
                if (keep < n) break;
            } else {
                emitted += n;
            }
        }
        double elapsed = wall_now() - start;
        double tps = (elapsed > 0) ? (double)emitted / elapsed : 0.0;

        if (iter >= ctx->bench_warmup && ctx->output_format == OC_CLI_OUTPUT_JSON) {
            if (completed > 0) printf(",");
            printf("{\"iter\":%d,\"decode_tokens\":%zu,\"decode_time_s\":%.6f,"
                   "\"decode_tok_per_s\":%.2f,\"prefill_tok_per_s\":%.2f}",
                   completed + 1, emitted, elapsed, tps, pf_tps);
        } else if (iter >= ctx->bench_warmup) {
            printf("  iter %d: %zu tokens in %.3fs = %.2f tok/s (prefill: %.2f tok/s)\n",
                   completed + 1, emitted, elapsed, tps, pf_tps);
        }

        if (iter >= ctx->bench_warmup) {
            if (tps > best_tps) best_tps = tps;
            if (pf_tps > best_pf) best_pf = pf_tps;
            sum_tps += tps;
            sum_pf += pf_tps;
            completed++;
        }
        oc_llama_session_free(&sess);
    }

    free(synthetic_ids);
    free(ids);
    oc_tokenizer_free(&tok);
    oc_llama_free(&model);

    if (completed == 0) return OC_ERR_INTERNAL;

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("],\"best_decode\":%.2f,\"avg_decode\":%.2f,"
               "\"best_prefill\":%.2f,\"avg_prefill\":%.2f}\n",
               best_tps, sum_tps / completed, best_pf, sum_pf / completed);
    } else {
        printf("benchmark: best=%.2f tok/s, avg=%.2f tok/s (prefill best=%.2f, avg=%.2f)\n",
               best_tps, sum_tps / completed, best_pf, sum_pf / completed);
    }
    return OC_OK;
}

/* ─── inspect ─────────────────────────────────────────────────────────── */

OcError oc_cli_run_inspect(OcCliContext *ctx)
{
    if (!ctx->model_path) {
        cli_error("--model is required for inspect");
        return OC_ERR_INVALID_ARG;
    }

    progress(ctx, "inspecting: %s", ctx->model_path);
    OcModelInfo info;
    OcError e = oc_inspect_model(ctx->model_path, &info);
    if (e != OC_OK) {
        cli_error("inspect failed (%s)", oc_error_msg(e));
        return e;
    }
    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        char buf[8192];
        size_t n = oc_inspect_format_json(&info, buf, sizeof(buf));
        if (n > 0) {
            printf("%s\n", buf);
        } else {
            /* Fallback: minimal JSON. */
            printf("{\"arch\":\"%s\",\"name\":\"%s\",\"n_layer\":%u,"
                   "\"n_embd\":%u,\"n_head\":%u,\"vocab_size\":%u,"
                   "\"param_count\":%llu,\"file_size\":%llu,"
                   "\"size_gb\":%.3f,\"quant_type\":\"%s\",\"n_tensors\":%u}\n",
                   info.arch, info.name, info.n_layer, info.n_embd, info.n_head,
                   info.vocab_size, (unsigned long long)info.param_count,
                   (unsigned long long)info.file_size, info.size_gb,
                   info.quant_type, info.n_tensors);
        }
    } else {
        char buf[8192];
        oc_inspect_format(&info, buf, sizeof(buf));
        printf("%s\n", buf);
    }

    oc_inspect_free(&info);
    return OC_OK;
}

/* ─── quantize ─────────────────────────────────────────────────────────── */

OcError oc_cli_run_quantize(OcCliContext *ctx)
{
    const char *input = ctx->model_path ? ctx->model_path : ctx->input_path;
    if (!input) {
        cli_error("--model (input) is required for quantize");
        return OC_ERR_INVALID_ARG;
    }
    const char *output = ctx->output_path ? ctx->output_path : "quantized.gguf";
    const char *target = ctx->target_type ? ctx->target_type : "Q4_K_M";

    progress(ctx, "quantizing: %s → %s (%s)", input, output, target);

    OcQuantizeConfig qcfg = {
        .input_path  = input,
        .output_path = output,
        .target_type = target,
        .verbose     = ctx->verbose,
    };
    OcError e = oc_quantize_model(&qcfg);
    if (e != OC_OK) {
        cli_error("quantization failed (%s)", oc_error_msg(e));
        return e;
    }
    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"quantize\",\"input\":\"%s\",\"output\":\"%s\","
               "\"target_type\":\"%s\",\"status\":\"ok\"}\n",
               input, output, target);
    } else {
        printf("quantization complete: %s → %s (%s)\n", input, output, target);
    }
    return OC_OK;
}

/* ─── convert ──────────────────────────────────────────────────────────── */

OcError oc_cli_run_convert(OcCliContext *ctx)
{
    if (!ctx->input_path) {
        cli_error("--input is required for convert");
        return OC_ERR_INVALID_ARG;
    }
    const char *output = ctx->output_path ? ctx->output_path : "converted.gguf";
    const char *target = ctx->target_type ? ctx->target_type : "Q4_K_M";
    const char *arch = ctx->arch ? ctx->arch : "llama";

    progress(ctx, "converting: %s → %s (%s, arch=%s)",
             ctx->input_path, output, target, arch);

    OcConvertConfig ccfg = {
        .input_path  = ctx->input_path,
        .output_path = output,
        .target_type = target,
        .arch        = arch,
        .verbose     = ctx->verbose,
    };
    OcError e = oc_safetensors_to_gguf(&ccfg);
    if (e != OC_OK) {
        cli_error("conversion failed (%s)", oc_error_msg(e));
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"convert\",\"input\":\"%s\",\"output\":\"%s\","
               "\"target_type\":\"%s\",\"arch\":\"%s\",\"status\":\"ok\"}\n",
               ctx->input_path, output, target, arch);
    } else {
        printf("conversion complete: %s → %s (%s, arch=%s)\n",
               ctx->input_path, output, target, arch);
    }
    return OC_OK;
}

/* ─── merge ────────────────────────────────────────────────────────────── */

OcError oc_cli_run_merge(OcCliContext *ctx)
{
    if (!ctx->input_path) {
        cli_error("at least one --input is required for merge");
        return OC_ERR_INVALID_ARG;
    }
    if (!ctx->output_path) {
        cli_error("--output is required for merge");
        return OC_ERR_INVALID_ARG;
    }

    /* Parse merge strategy. */
    OcMergeStrategy strategy = OC_MERGE_LINEAR;
    if (ctx->merge_strategy) {
        if (ieq(ctx->merge_strategy, "linear"))       strategy = OC_MERGE_LINEAR;
        else if (ieq(ctx->merge_strategy, "slerp"))    strategy = OC_MERGE_SLERP;
        else if (ieq(ctx->merge_strategy, "ties"))     strategy = OC_MERGE_TIES;
        else if (ieq(ctx->merge_strategy, "dare"))     strategy = OC_MERGE_DARE;
        else {
            cli_error("unknown merge strategy: %s "
                      "(expected linear|slerp|ties|dare)", ctx->merge_strategy);
            return OC_ERR_INVALID_ARG;
        }
    }

    /* For now, we support a single input via --input (the CLI parser
     * accumulates multiple --input flags into input_path; a future
     * enhancement will support multiple inputs via repeated flags). */
    OcMergeInput inputs[1];
    inputs[0].path = ctx->input_path;
    inputs[0].weight = 1.0f;

    progress(ctx, "merging: %s → %s (%s)",
             ctx->input_path, ctx->output_path,
             oc_merge_strategy_name(strategy));

    OcMergeConfig mcfg = {
        .strategy     = strategy,
        .inputs       = inputs,
        .n_inputs     = 1,
        .output_path  = ctx->output_path,
        .slerp_t      = ctx->merge_slerp_t,
        .ties_density  = ctx->merge_density,
        .verbose      = ctx->verbose,
    };
    OcError e = oc_merge_models(&mcfg);
    if (e != OC_OK) {
        cli_error("merge failed (%s)", oc_error_msg(e));
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"merge\",\"input\":\"%s\",\"output\":\"%s\","
               "\"strategy\":\"%s\",\"status\":\"ok\"}\n",
               ctx->input_path, ctx->output_path,
               oc_merge_strategy_name(strategy));
    } else {
        printf("merge complete: %s → %s (%s)\n",
               ctx->input_path, ctx->output_path,
               oc_merge_strategy_name(strategy));
    }
    return OC_OK;
}

/* ─── prune ────────────────────────────────────────────────────────────── */

OcError oc_cli_run_prune(OcCliContext *ctx)
{
    const char *input = ctx->model_path ? ctx->model_path : ctx->input_path;
    if (!input) {
        cli_error("--model (input) is required for prune");
        return OC_ERR_INVALID_ARG;
    }
    if (!ctx->output_path) {
        cli_error("--output is required for prune");
        return OC_ERR_INVALID_ARG;
    }

    OcPruneStrategy strategy = OC_PRUNE_WANDA;
    if (ctx->prune_strategy) {
        if (ieq(ctx->prune_strategy, "wanda"))          strategy = OC_PRUNE_WANDA;
        else if (ieq(ctx->prune_strategy, "magnitude")) strategy = OC_PRUNE_MAGNITUDE;
        else {
            cli_error("unknown prune strategy: %s "
                      "(expected wanda|magnitude)", ctx->prune_strategy);
            return OC_ERR_INVALID_ARG;
        }
    }

    progress(ctx, "pruning: %s → %s (%s, sparsity=%.2f)",
             input, ctx->output_path,
             oc_prune_strategy_name(strategy), ctx->prune_sparsity);

    OcPruneConfig pcfg = {
        .strategy   = strategy,
        .sparsity   = ctx->prune_sparsity,
        .input_path = input,
        .output_path = ctx->output_path,
        .verbose    = ctx->verbose,
    };
    OcError e = oc_prune_model(&pcfg);
    if (e != OC_OK) {
        cli_error("prune failed (%s)", oc_error_msg(e));
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"prune\",\"input\":\"%s\",\"output\":\"%s\","
               "\"strategy\":\"%s\",\"sparsity\":%.4f,\"status\":\"ok\"}\n",
               input, ctx->output_path,
               oc_prune_strategy_name(strategy), ctx->prune_sparsity);
    } else {
        printf("prune complete: %s → %s (%s, sparsity=%.2f)\n",
               input, ctx->output_path,
               oc_prune_strategy_name(strategy), ctx->prune_sparsity);
    }
    return OC_OK;
}

/* ─── finetune ─────────────────────────────────────────────────────────── */

OcError oc_cli_run_finetune(OcCliContext *ctx)
{
    if (!ctx->model_path) {
        cli_error("--model is required for finetune");
        return OC_ERR_INVALID_ARG;
    }
    if (!ctx->dataset_path) {
        cli_error("--dataset is required for finetune");
        return OC_ERR_INVALID_ARG;
    }

    OcFtStrategy strategy = OC_FT_SFT;
    if (ctx->ft_strategy) {
        if (ieq(ctx->ft_strategy, "sft"))                  strategy = OC_FT_SFT;
        else if (ieq(ctx->ft_strategy, "self-train"))      strategy = OC_FT_SELF_TRAIN;
        else if (ieq(ctx->ft_strategy, "dpo"))              strategy = OC_FT_DPO;
        else if (ieq(ctx->ft_strategy, "ppo"))             strategy = OC_FT_PPO;
        else {
            cli_error("unknown finetune strategy: %s "
                      "(expected sft|self-train|dpo|ppo)", ctx->ft_strategy);
            return OC_ERR_INVALID_ARG;
        }
    }

    const char *out_dir = ctx->output_dir ? ctx->output_dir : "./adapters";

    progress(ctx, "finetuning: model=%s dataset=%s strategy=%s out=%s",
             ctx->model_path, ctx->dataset_path,
             oc_ft_strategy_name(strategy), out_dir);

    OcFtConfig fcfg = {
        .strategy      = strategy,
        .model_path    = ctx->model_path,
        .dataset_path  = ctx->dataset_path,
        .output_dir    = out_dir,
        .resume_from   = ctx->resume_from,
        .lora_rank     = ctx->lora_rank,
        .lora_alpha    = ctx->lora_alpha,
        .epochs        = ctx->epochs,
        .batch_size    = ctx->batch_size,
        .learning_rate = ctx->learning_rate,
        .warmup_ratio  = 0.1f,
        .weight_decay  = 0.01f,
        .max_seq_length = 2048,
        .verbose       = ctx->verbose,
    };
    OcError e = oc_finetune_run(&fcfg);
    if (e != OC_OK) {
        cli_error("finetune failed (%s)", oc_error_msg(e));
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"finetune\",\"model\":\"%s\",\"dataset\":\"%s\","
               "\"strategy\":\"%s\",\"output_dir\":\"%s\",\"status\":\"ok\"}\n",
               ctx->model_path, ctx->dataset_path,
               oc_ft_strategy_name(strategy), out_dir);
    } else {
        printf("finetune complete: model=%s strategy=%s output=%s\n",
               ctx->model_path, oc_ft_strategy_name(strategy), out_dir);
    }
    return OC_OK;
}

/* ─── list ─────────────────────────────────────────────────────────────── */

OcError oc_cli_run_list(OcCliContext *ctx)
{
    const char *cache = ctx->cache_dir ? ctx->cache_dir : default_cache_dir();

    progress(ctx, "scanning cache: %s", cache);

    DIR *d = opendir(cache);
    if (!d) {
        if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
            printf("{\"command\":\"list\",\"cache_dir\":\"%s\",\"models\":[]}\n", cache);
        } else {
            printf("No models found in %s\n", cache);
        }
        return OC_OK;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"list\",\"cache_dir\":\"%s\",\"models\":[", cache);
    } else {
        printf("Available models in %s:\n", cache);
    }

    bool first = true;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;

        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", cache, ent->d_name);

        if (is_directory(full)) {
            /* Scan subdirectory for .gguf files. */
            DIR *sub = opendir(full);
            if (!sub) continue;
            struct dirent *sent;
            while ((sent = readdir(sub)) != NULL) {
                if (!has_gguf_ext(sent->d_name)) continue;
                char sub_full[2048];
                snprintf(sub_full, sizeof(sub_full), "%s/%s", full, sent->d_name);
                struct stat st;
                if (stat(sub_full, &st) != 0) continue;

                if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
                    if (!first) printf(",");
                    printf("{\"name\":\"%s/%s\",\"path\":\"%s\",\"size\":%lld}",
                           ent->d_name, sent->d_name, sub_full,
                           (long long)st.st_size);
                } else {
                    printf("  %-40s %lld bytes\n",
                           sent->d_name, (long long)st.st_size);
                }
                first = false;
            }
            closedir(sub);
        } else if (has_gguf_ext(ent->d_name)) {
            struct stat st;
            if (stat(full, &st) != 0) continue;
            if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
                if (!first) printf(",");
                printf("{\"name\":\"%s\",\"path\":\"%s\",\"size\":%lld}",
                       ent->d_name, full, (long long)st.st_size);
            } else {
                printf("  %-40s %lld bytes\n", ent->d_name, (long long)st.st_size);
            }
            first = false;
        }
    }
    closedir(d);

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("]}\n");
    } else if (first) {
        printf("  (none)\n");
    }
    return OC_OK;
}

/* ─── download ─────────────────────────────────────────────────────────── */

OcError oc_cli_run_download(OcCliContext *ctx)
{
    if (!ctx->hf_repo) {
        cli_error("--repo is required for download "
                  "(e.g. --repo Qwen/Qwen2.5-7B)");
        return OC_ERR_INVALID_ARG;
    }

    const char *cache = ctx->cache_dir ? ctx->cache_dir : default_cache_dir();

    /* Build HF config. */
    OcHfConfig hcfg;
    oc_hf_config_init(&hcfg, cache);
    snprintf(hcfg.repo_id, sizeof(hcfg.repo_id), "%s", ctx->hf_repo);
    if (ctx->hf_file) {
        /* User specified a specific file — resolve and download it. */
        OcHfModel model;
        memset(&model, 0, sizeof(model));
        snprintf(model.repo_id, sizeof(model.repo_id), "%s", ctx->hf_repo);
        snprintf(model.filename, sizeof(model.filename), "%s", ctx->hf_file);

        /* Build download URL. repo_id and filename are already clamped to
         * their own field sizes above, but the compiler cannot see that the
         * concatenation fits, so check the result explicitly rather than
         * letting a silent truncation produce a wrong URL. */
        int url_len = snprintf(model.download_url, sizeof(model.download_url),
                               "%s/%s/resolve/main/%s",
                               hcfg.api_base[0] ? hcfg.api_base
                                                : OC_HF_DEFAULT_API_BASE,
                               ctx->hf_repo, ctx->hf_file);
        if (url_len < 0 || (size_t)url_len >= sizeof(model.download_url)) {
            cli_error("repo/file name too long for a download URL");
            return OC_ERR_INVALID_ARG;
        }

        progress(ctx, "downloading: %s/%s → %s", ctx->hf_repo, ctx->hf_file, cache);

        OcError e = oc_hf_download(&hcfg, &model, NULL, NULL);
        if (e != OC_OK) {
            cli_error("download failed (%s)", oc_error_msg(e));
            return e;
        }

        /* Get the local cache path. */
        char local_path[1024];
        oc_hf_cache_path(&hcfg, ctx->hf_repo, ctx->hf_file,
                         local_path, sizeof(local_path));

        if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
            printf("{\"command\":\"download\",\"repo\":\"%s\",\"file\":\"%s\","
                   "\"cache_dir\":\"%s\",\"local_path\":\"%s\",\"status\":\"ok\"}\n",
                   ctx->hf_repo, ctx->hf_file, cache, local_path);
        } else {
            printf("downloaded: %s/%s → %s\n",
                   ctx->hf_repo, ctx->hf_file, local_path);
        }
    } else {
        /* No specific file — list available .gguf files and download the best match. */
        progress(ctx, "listing models in %s...", ctx->hf_repo);

        OcHfModel models[OC_HF_MAX_MODELS];
        size_t n_models = OC_HF_MAX_MODELS;
        OcError e = oc_hf_list_models(&hcfg, models, &n_models);
        if (e != OC_OK) {
            cli_error("failed to list models (%s)", oc_error_msg(e));
            return e;
        }

        if (n_models == 0) {
            cli_error("no .gguf files found in repo %s", ctx->hf_repo);
            return OC_ERR_MODEL;
        }

        /* If quant_type is specified, filter for it. Otherwise pick the first. */
        size_t pick = 0;
        if (hcfg.quant_type[0]) {
            for (size_t i = 0; i < n_models; i++) {
                if (strcmp(models[i].quant_type, hcfg.quant_type) == 0) {
                    pick = i;
                    break;
                }
            }
        }

        progress(ctx, "downloading: %s/%s → %s",
                 ctx->hf_repo, models[pick].filename, cache);

        e = oc_hf_download(&hcfg, &models[pick], NULL, NULL);
        if (e != OC_OK) {
            cli_error("download failed (%s)", oc_error_msg(e));
            return e;
        }

        char local_path[1024];
        oc_hf_cache_path(&hcfg, ctx->hf_repo, models[pick].filename,
                         local_path, sizeof(local_path));

        if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
            printf("{\"command\":\"download\",\"repo\":\"%s\","
                   "\"file\":\"%s\",\"cache_dir\":\"%s\","
                   "\"local_path\":\"%s\",\"status\":\"ok\"}\n",
                   ctx->hf_repo, models[pick].filename, cache, local_path);
        } else {
            printf("downloaded: %s/%s → %s\n",
                   ctx->hf_repo, models[pick].filename, local_path);
        }
    }
    return OC_OK;
}

/* ─── tokenize ────────────────────────────────────────────────────────── */

OcError oc_cli_run_tokenize(OcCliContext *ctx)
{
    if (!ctx->model_path) {
        cli_error("--model is required for tokenize");
        return OC_ERR_INVALID_ARG;
    }

    const char *text = ctx->prompt;
    char *file_text = NULL;
    if (!text && ctx->prompt_file) {
        file_text = read_file_text(ctx->prompt_file);
        if (!file_text) {
            cli_error("cannot read prompt-file: %s", ctx->prompt_file);
            return OC_ERR_IO;
        }
        text = file_text;
    }
    if (!text) {
        cli_error("--prompt or --prompt-file is required for tokenize");
        free(file_text);
        return OC_ERR_INVALID_ARG;
    }

    progress(ctx, "loading model: %s", ctx->model_path);
    OcLlamaModel model;
    OcError e = oc_llama_load(ctx->model_path, &model);
    if (e != OC_OK) {
        cli_error("failed to load model (%s)", oc_error_msg(e));
        free(file_text);
        return e;
    }
    oc_cli_apply_ctx(ctx, &model);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&model.gguf.unified, &tok);
    if (e != OC_OK) {
        cli_error("tokenizer load failed (%s)", oc_error_msg(e));
        oc_llama_free(&model);
        free(file_text);
        return e;
    }

    OcSpecialTokenPolicy pol = ctx->tokens_no_special
        ? OC_TOK_DISALLOW_SPECIAL
        : ((tok.has_add_bos_token && tok.add_bos_token)
           ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT);

    uint32_t *ids = NULL;
    size_t n_ids = 0;
    e = oc_tokenizer_encode(&tok, text, pol, &ids, &n_ids);
    if (e != OC_OK) {
        cli_error("tokenize failed (%s)", oc_error_msg(e));
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        free(file_text);
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"tokenize\",\"count\":%zu,\"tokens\":[", n_ids);
        for (size_t i = 0; i < n_ids; i++) {
            if (i > 0) printf(",");
            printf("%u", ids[i]);
        }
        printf("]}\n");
    } else {
        printf("tokens (%zu):\n", n_ids);
        for (size_t i = 0; i < n_ids; i++) {
            /* Print ID and the decoded piece (for context). */
            char *piece = NULL;
            if (oc_tokenizer_decode(&tok, &ids[i], 1, &piece) == OC_OK && piece) {
                char esc[256];
                json_escape(piece, esc, sizeof(esc));
                printf("  %6u  %s\n", ids[i], esc);
                free(piece);
            } else {
                printf("  %6u  (decode failed)\n", ids[i]);
            }
        }
    }

    free(ids);
    oc_tokenizer_free(&tok);
    oc_llama_free(&model);
    free(file_text);
    return OC_OK;
}

/* ─── detokenize ───────────────────────────────────────────────────────── */

OcError oc_cli_run_detokenize(OcCliContext *ctx)
{
    if (!ctx->model_path) {
        cli_error("--model is required for detokenize");
        return OC_ERR_INVALID_ARG;
    }
    if (!ctx->token_ids_str) {
        cli_error("--ids is required for detokenize (e.g. --ids \"1,2,3\")");
        return OC_ERR_INVALID_ARG;
    }

    progress(ctx, "loading model: %s", ctx->model_path);
    OcLlamaModel model;
    OcError e = oc_llama_load(ctx->model_path, &model);
    if (e != OC_OK) {
        cli_error("failed to load model (%s)", oc_error_msg(e));
        return e;
    }
    oc_cli_apply_ctx(ctx, &model);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&model.gguf.unified, &tok);
    if (e != OC_OK) {
        cli_error("tokenizer load failed (%s)", oc_error_msg(e));
        oc_llama_free(&model);
        return e;
    }

    /* Parse comma-separated token IDs. */
    uint32_t ids[4096];
    size_t n_ids = 0;
    char *ids_copy = strdup(ctx->token_ids_str);
    if (!ids_copy) {
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        return OC_ERR_OOM;
    }
    char *tok_str = strtok(ids_copy, ", \t\n");
    while (tok_str && n_ids < 4096) {
        char *end = NULL;
        unsigned long val = strtoul(tok_str, &end, 10);
        if (end != tok_str) {
            ids[n_ids++] = (uint32_t)val;
        }
        tok_str = strtok(NULL, ", \t\n");
    }
    free(ids_copy);

    if (n_ids == 0) {
        cli_error("no valid token IDs found in --ids");
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        return OC_ERR_INVALID_ARG;
    }

    char *text = NULL;
    e = oc_tokenizer_decode(&tok, ids, n_ids, &text);
    if (e != OC_OK) {
        cli_error("detokenize failed (%s)", oc_error_msg(e));
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        char esc[65536];
        json_escape(text, esc, sizeof(esc));
        printf("{\"command\":\"detokenize\",\"count\":%zu,\"text\":\"%s\"}\n",
               n_ids, esc);
    } else {
        printf("%s\n", text);
    }

    free(text);
    oc_tokenizer_free(&tok);
    oc_llama_free(&model);
    return OC_OK;
}

/* ─── perplexity ───────────────────────────────────────────────────────── */

OcError oc_cli_run_perplexity(OcCliContext *ctx)
{
    if (!ctx->model_path) {
        cli_error("--model is required for perplexity");
        return OC_ERR_INVALID_ARG;
    }

    const char *text = ctx->prompt;
    char *file_text = NULL;
    if (!text && ctx->prompt_file) {
        file_text = read_file_text(ctx->prompt_file);
        if (!file_text) {
            cli_error("cannot read prompt-file: %s", ctx->prompt_file);
            return OC_ERR_IO;
        }
        text = file_text;
    }
    if (!text) text = "The quick brown fox jumps over the lazy dog.";

    progress(ctx, "loading model: %s", ctx->model_path);
    OcLlamaModel model;
    OcError e = oc_llama_load(ctx->model_path, &model);
    if (e != OC_OK) {
        cli_error("failed to load model (%s)", oc_error_msg(e));
        free(file_text);
        return e;
    }
    oc_cli_apply_ctx(ctx, &model);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&model.gguf.unified, &tok);
    if (e != OC_OK) {
        cli_error("tokenizer load failed (%s)", oc_error_msg(e));
        oc_llama_free(&model);
        free(file_text);
        return e;
    }

    OcPerplexityResult result;
    e = oc_perplexity_evaluate(&model, &tok, text, ctx->ppl_max_tokens, &result);
    if (e != OC_OK) {
        cli_error("perplexity evaluation failed (%s)", oc_error_msg(e));
        oc_tokenizer_free(&tok);
        oc_llama_free(&model);
        free(file_text);
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"perplexity\",\"ppl\":%.6f,\"avg_nll\":%.6f,"
               "\"total_nll\":%.6f,\"n_tokens\":%zu,\"eval_time_s\":%.6f,"
               "\"tokens_per_sec\":%.2f}\n",
               result.ppl, result.avg_nll, result.total_nll,
               result.n_tokens, result.eval_time_sec, result.tokens_per_sec);
    } else {
        char fmt[256];
        oc_perplexity_format(&result, fmt, sizeof(fmt));
        printf("%s\n", fmt);
    }

    oc_tokenizer_free(&tok);
    oc_llama_free(&model);
    free(file_text);
    return OC_OK;
}

/* ─── serve ────────────────────────────────────────────────────────────── */

OcError oc_cli_run_serve(OcCliContext *ctx)
{
    OcError e;
    OcOpenaiState st;
    memset(&st, 0, sizeof(st));
    st.model_loaded = false;

    if (ctx->model_path) {
        OcLlamaModel *model = calloc(1, sizeof(OcLlamaModel));
        OcTokenizer *tok = calloc(1, sizeof(OcTokenizer));
        if (!model || !tok) {
            cli_error("out of memory");
            free(model); free(tok);
            return OC_ERR_OOM;
        }
        progress(ctx, "loading model: %s", ctx->model_path);
        e = oc_llama_load(ctx->model_path, model);
        if (e != OC_OK) {
            cli_error("failed to load model (%s)", oc_error_msg(e));
            free(model); free(tok);
            return e;
        }
        oc_cli_apply_ctx(ctx, model);
        e = oc_tokenizer_load_from_gguf(&model->gguf.unified, tok);
        if (e != OC_OK) {
            cli_error("tokenizer load failed (%s)", oc_error_msg(e));
            oc_llama_free(model);
            free(model); free(tok);
            return e;
        }
        st.model = model;
        st.tokenizer = tok;
        st.model_loaded = true;
        const char *slash = strrchr(ctx->model_path, '/');
        st.model_id = strdup(slash ? slash + 1 : ctx->model_path);
        progress(ctx, "serve: model loaded, starting server on %s:%d",
                 ctx->host, ctx->port);
    } else {
        progress(ctx, "serve: no model — starting placeholder server on %s:%d",
                 ctx->host, ctx->port);
    }

    /* Middleware stack. Metrics + audit are always on so /metrics has
     * something to serve; auth and rate limiting turn on only when the
     * corresponding flag is given, so the default local-dev invocation is
     * unchanged. */
    OcMiddleware mw;
    uint32_t mw_enabled = OC_MW_METRICS | OC_MW_AUDIT;
    if (ctx->api_key != NULL)        mw_enabled |= OC_MW_AUTH;
    if (ctx->rate_limit_rpm > 0)     mw_enabled |= OC_MW_RATE_LIMIT;
    if (ctx->cors_origin != NULL)    mw_enabled |= OC_MW_CORS;
    OcRateLimitConfig rl = {
        .requests_per_minute = ctx->rate_limit_rpm,
        /* Allow a short burst of one second's worth of traffic (min 1) so
         * a browser opening several connections at once isn't throttled. */
        .burst_size = ctx->rate_limit_rpm / 60u + 1u,
        .per_ip = true,
    };
    e = oc_middleware_init(&mw, mw_enabled, ctx->api_key, &rl, ctx->cors_origin);
    if (e != OC_OK) {
        cli_error("middleware init failed (%s)", oc_error_msg(e));
        if (st.model_loaded) {
            free(st.model_id);
            oc_tokenizer_free(st.tokenizer);
            oc_llama_free(st.model);
            free(st.tokenizer);
            free(st.model);
        }
        return e;
    }
    st.mw = &mw;

    OcHttpServer srv;
    memset(&srv, 0, sizeof(srv));
    oc_openai_attach_http(&srv, &st);
    e = oc_http_server_start_configured(ctx->host, (uint16_t)ctx->port, 4,
                                        oc_openai_handler, &st, &srv);
    if (e != OC_OK) {
        cli_error("server start failed (%s)", oc_error_msg(e));
        oc_middleware_free(&mw);
        if (st.model_loaded) {
            free(st.model_id);
            oc_tokenizer_free(st.tokenizer);
            oc_llama_free(st.model);
            free(st.tokenizer);
            free(st.model);
        }
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"serve\",\"host\":\"%s\",\"port\":%u,"
               "\"model_loaded\":%s,\"model_id\":\"%s\"}\n",
               ctx->host, srv.port,
               st.model_loaded ? "true" : "false",
               st.model_id ? st.model_id : "");
    } else {
        printf("oxidize-c server listening on http://%s:%u\n"
               "  GET  /healthz  /livez  /readyz\n"
               "  GET  /metrics  /openapi.json\n"
               "  GET  /v1/models\n"
               "  POST /v1/completions\n"
               "  POST /v1/chat/completions\n"
               "  POST /v1/embeddings\n"
               "  POST /v1/responses\n"
               "%s%s"
               "(Ctrl+C to stop)\n", ctx->host, srv.port,
               ctx->api_key ? "  auth: enabled (Bearer token required)\n" : "",
               ctx->rate_limit_rpm ? "  rate limit: enabled (per-IP)\n" : "");
    }
    fflush(stdout);
    oc_http_server_join(&srv);

    oc_middleware_free(&mw);
    if (st.model_loaded) {
        free(st.model_id);
        oc_tokenizer_free(st.tokenizer);
        oc_llama_free(st.model);
        free(st.tokenizer);
        free(st.model);
    }
    return OC_OK;
}

/* ─── serve-realtime ───────────────────────────────────────────────────── */

OcError oc_cli_run_serve_realtime(OcCliContext *ctx)
{
    if (!ctx->model_path) {
        cli_error("--model is required for serve-realtime");
        return OC_ERR_INVALID_ARG;
    }

    OcLlamaModel *model = calloc(1, sizeof(OcLlamaModel));
    OcTokenizer *tok = calloc(1, sizeof(OcTokenizer));
    if (!model || !tok) {
        cli_error("out of memory");
        free(model); free(tok);
        return OC_ERR_OOM;
    }

    progress(ctx, "loading model: %s", ctx->model_path);
    OcError e = oc_llama_load(ctx->model_path, model);
    if (e != OC_OK) {
        cli_error("failed to load model (%s)", oc_error_msg(e));
        free(model); free(tok);
        return e;
    }
    oc_cli_apply_ctx(ctx, model);
    e = oc_tokenizer_load_from_gguf(&model->gguf.unified, tok);
    if (e != OC_OK) {
        cli_error("tokenizer load failed (%s)", oc_error_msg(e));
        oc_llama_free(model);
        free(model); free(tok);
        return e;
    }

    /* The realtime server uses the HTTP server to upgrade WebSocket
     * connections. We start the HTTP server with the realtime handler.
     * The actual WebSocket upgrade + realtime session handling is wired
     * via the realtime module. For now, we start the HTTP server and
     * accept connections at /v1/realtime. */
    OcHttpServer srv;
    memset(&srv, 0, sizeof(srv));
    /* We reuse the OpenAI handler as the base; WebSocket upgrade for
     * realtime is handled within the handler when path == /v1/realtime.
     * A future enhancement will add a dedicated realtime HTTP handler. */
    OcOpenaiState st;
    memset(&st, 0, sizeof(st));
    st.model = model;
    st.tokenizer = tok;
    st.model_loaded = true;
    const char *slash = strrchr(ctx->model_path, '/');
    st.model_id = strdup(slash ? slash + 1 : ctx->model_path);

    oc_openai_attach_http(&srv, &st);
    e = oc_http_server_start_configured(ctx->host, (uint16_t)ctx->port, 4,
                                        oc_openai_handler, &st, &srv);
    if (e != OC_OK) {
        cli_error("realtime server start failed (%s)", oc_error_msg(e));
        free(st.model_id);
        oc_tokenizer_free(tok);
        oc_llama_free(model);
        free(tok); free(model);
        return e;
    }

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        printf("{\"command\":\"serve-realtime\",\"host\":\"%s\",\"port\":%u,"
               "\"model_id\":\"%s\",\"endpoint\":\"ws://%s:%u/v1/realtime\"}\n",
               ctx->host, srv.port, st.model_id,
               ctx->host, srv.port);
    } else {
        printf("oxidize-c realtime server listening on ws://%s:%u/v1/realtime\n"
               "  model: %s\n"
               "(Ctrl+C to stop)\n", ctx->host, srv.port, st.model_id);
    }
    fflush(stdout);
    oc_http_server_join(&srv);

    free(st.model_id);
    oc_tokenizer_free(tok);
    oc_llama_free(model);
    free(tok); free(model);
    return OC_OK;
}

/* ─── dflash2: draft propose benchmark ───────────────────────────────── */

#include "oxidize/dflash2.h"
#include "oxidize/numa.h"
#include "oxidize/parallel.h"

#include <time.h>

static double dflash2_now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Deterministic LCG so runs are reproducible without a real target. */
static uint32_t dflash2_lcg(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

/* Deterministic pseudo-random lm_head row generator. 8 independent LCG
 * lanes seeded from the row index; each lane advances once per 8 floats,
 * so the element loop is dependency-free arithmetic (no serial chain)
 * and the scan stays dominated by the dot products like a real GEMM. */
static void dflash2_gen_lm_row(size_t row, size_t cols, float *buf, void *user)
{
    (void)user;
    uint32_t s0 = (uint32_t)row * 1664525u + 1013904223u;
    uint32_t s1 = s0 * 1664525u + 1013904223u;
    uint32_t s2 = s1 * 1664525u + 1013904223u;
    uint32_t s3 = s2 * 1664525u + 1013904223u;
    uint32_t s4 = s3 * 1664525u + 1013904223u;
    uint32_t s5 = s4 * 1664525u + 1013904223u;
    uint32_t s6 = s5 * 1664525u + 1013904223u;
    uint32_t s7 = s6 * 1664525u + 1013904223u;
    size_t c = 0;
    for (; c + 8 <= cols; c += 8) {
        s0 = s0 * 1664525u + 1013904223u;
        s1 = s1 * 1664525u + 1013904223u;
        s2 = s2 * 1664525u + 1013904223u;
        s3 = s3 * 1664525u + 1013904223u;
        s4 = s4 * 1664525u + 1013904223u;
        s5 = s5 * 1664525u + 1013904223u;
        s6 = s6 * 1664525u + 1013904223u;
        s7 = s7 * 1664525u + 1013904223u;
        buf[c + 0] = ((float)(s0 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
        buf[c + 1] = ((float)(s1 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
        buf[c + 2] = ((float)(s2 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
        buf[c + 3] = ((float)(s3 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
        buf[c + 4] = ((float)(s4 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
        buf[c + 5] = ((float)(s5 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
        buf[c + 6] = ((float)(s6 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
        buf[c + 7] = ((float)(s7 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
    }
    for (; c < cols; c++) {
        s0 = s0 * 1664525u + 1013904223u;
        buf[c] = ((float)(s0 >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.04f;
    }
}

OcError oc_cli_run_dflash2(OcCliContext *ctx)
{
    if (!ctx || !ctx->model_path) {
        cli_error("dflash2 requires --model <path/to/model.safetensors>");
        return OC_ERR_INVALID_ARG;
    }

    /* Resolve model.safetensors + config.json (same dir by default). */
    char st_path[512], cfg_path[512];
    snprintf(st_path, sizeof(st_path), "%s", ctx->model_path);
    /* Accept either the directory or the safetensors file itself. */
    struct stat sb;
    if (stat(st_path, &sb) == 0 && S_ISDIR(sb.st_mode)) {
        snprintf(st_path, sizeof(st_path), "%s/model.safetensors", ctx->model_path);
    }
    snprintf(cfg_path, sizeof(cfg_path), "%s", st_path);
    char *slash = strrchr(cfg_path, '/');
    if (slash) {
        snprintf(slash + 1, sizeof(cfg_path) - (size_t)(slash + 1 - cfg_path),
                 "config.json");
    } else {
        /* Bare filename (no directory): the sidecar config lives in the
         * current working directory. Without this, cfg_path stayed
         * pointing at the checkpoint and the loader parsed SafeTensors
         * bytes as config.json. */
        snprintf(cfg_path, sizeof(cfg_path), "config.json");
    }

    cli_info("loading DFlash2 draft from %s", st_path);
    /* Command-specific benchmark defaults when the option was omitted:
     * the shared context ships the generic bench defaults (3 iters /
     * 5 warmup), but the documented dflash2 workload is 20 / 3. */
    if (!ctx->bench_iters_set)  ctx->bench_iterations = 20;
    if (!ctx->bench_warmup_set) ctx->bench_warmup = 3;
    if (ctx->threads > 0) {
        oc_parallel_set_threads((size_t)ctx->threads);
    } else if (ctx->auto_tune ||
               (ctx->numa && strcmp(ctx->numa, "none") != 0)) {
        size_t n = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
        oc_parallel_set_threads(n > 16 ? 16 : n);
    } else {
        /* Default: keep init_compute_threads' physical-core pool (already
         * pinned one-worker-per-core by oc_parallel_set_threads). SMT
         * siblings contend on the µop-bound phases and measured 19%
         * slower. */
    }

    OcDFlash2Model model;
    double t0 = dflash2_now_sec();
    OcError e = oc_dflash2_model_load(&model, st_path, cfg_path);
    if (e != OC_OK) {
        cli_error("failed to load DFlash2 checkpoint (%s)", oc_error_msg(e));
        return e;
    }
    double load_s = dflash2_now_sec() - t0;
    cli_info("draft loaded in %.2fs (5 layers, hidden %zu, vocab %zu)",
             load_s, model.cfg.hidden_size, model.cfg.vocab_size);

    /* Synthetic target inputs: deterministic random. The block-1 drafts
     * per step: 7 for GLM-5.3-Flash-DFlash2 (block 8). */
    const size_t H = model.cfg.hidden_size;
    const size_t block = model.cfg.block_size;
    const size_t n_target_w = model.cfg.n_target_layer_ids * H;
    const size_t top_k = model.cfg.selector_top_k;
    const size_t vocab = model.cfg.vocab_size;

    /* Context rows per step: --bench-prompt-tokens drives how much target
     * context each propose consumes, exercising the full KV window (the
     * reference model would feed the target's hidden states for the last
     * `ctx_rows` positions). Capped by the KV ring capacity. */
    size_t ctx_rows = ctx->bench_prompt_tokens > 0
                    ? (size_t)ctx->bench_prompt_tokens : 1;
    if (ctx_rows > model.kv_capacity) ctx_rows = model.kv_capacity;
    if (ctx_rows < 1) ctx_rows = 1;
    const size_t ctx_cap = model.kv_capacity;

    /* Stand-in target lm_head: rows = vocab, cols = H. Too big to
     * materialize densely in f32 (154880*4096*4 = 2.4 GB) — use a
     * structured pseudo-random projection: hash rows so each "weight row"
     * is generated on the fly. That keeps memory flat while exercising
     * the same FLOPs. */
    float *noise = malloc(block * H * sizeof(float));
    uint32_t *block_ids = malloc(block * sizeof(uint32_t));
    uint32_t *anchors = malloc(sizeof(uint32_t));
    uint32_t *out_tok = malloc((block - 1) * sizeof(uint32_t));
    uint32_t *out_cand = malloc((block - 1) * top_k * sizeof(uint32_t));
    float *out_probs = malloc((block - 1) * top_k * sizeof(float));
    if (!noise || !block_ids || !anchors || !out_tok ||
        !out_cand || !out_probs) {
        free(noise); free(block_ids); free(anchors);
        free(out_tok); free(out_cand); free(out_probs);
        oc_dflash2_model_free(&model);
        return OC_ERR_OOM;
    }

    /* Synthetic lm_head. Two modes:
     *  - default: rows generated on the fly (flat memory, but the vocab
     *    scan is compute-bound on the generator instead of bandwidth).
     *  - --lm-materialize: pre-materialize the rows in BF16 (vocab*H*2
     *    bytes, ~1.27 GB for GLM-5.3-Flash) so the scan measures the real
     *    target-owned lm_head streaming cost from RAM. Same deterministic
     *    generator, so both modes produce identical rows. */
    static OcDFlash2Weight lm;
    lm.rows = vocab;
    lm.cols = H;
    lm.gen_user = NULL;
    lm.generate = dflash2_gen_lm_row;
    lm.data = NULL;
    lm.bf16 = NULL;
    if (ctx->bench_lm_materialize) {
        size_t n_w = vocab * H;
        size_t lm_bytes = n_w * sizeof(uint16_t);
        lm.bf16 = oc_dflash2_alloc_huge(lm_bytes);
        if (!lm.bf16) {
            free(noise); free(block_ids); free(anchors);
            free(out_tok); free(out_cand); free(out_probs);
            oc_dflash2_model_free(&model);
            return OC_ERR_OOM;
        }
        float *row = malloc(H * sizeof(float));
        if (!row) {
            oc_dflash2_free_huge(lm.bf16, lm_bytes); lm.bf16 = NULL;
            free(noise); free(block_ids); free(anchors);
            free(out_tok); free(out_cand); free(out_probs);
            oc_dflash2_model_free(&model);
            return OC_ERR_OOM;
        }
        for (size_t v = 0; v < vocab; v++) {
            dflash2_gen_lm_row(v, H, row, NULL);
            for (size_t c = 0; c < H; c++) {
                uint32_t bits;
                memcpy(&bits, row + c, 4);
                lm.bf16[v * H + c] = (uint16_t)(bits >> 16);
            }
        }
        free(row);
        cli_info("materialized %zu x %zu BF16 lm_head (%.2f GB), "
                 "vocab scan now bandwidth-bound",
                 vocab, H, (double)(n_w * 2) / (1024.0 * 1024.0 * 1024.0));
    }

    uint32_t seed = ctx->seed ? ctx->seed : 42u;
    const uint32_t iters = (uint32_t)(ctx->bench_iterations > 0
                                      ? ctx->bench_iterations : 20);
    const uint32_t warmup = ctx->bench_warmup;
    double best = INFINITY, total = 0.0;
    uint64_t drafted_total = 0;

    /* Pre-generate the full synthetic context once (the "prefill": the
     * target's hidden states for positions [0, ctx_rows)). Each step then
     * only produces the NEW rows (the accepted tokens of the previous
     * verify), matching dflash_generate: after the first call the draft
     * consumes just output.hidden_states[:, :produced, :]. */
    float *ctx_prefill = malloc(ctx_rows * n_target_w * sizeof(float));
    float *ctx_new = malloc((block - 1) * n_target_w * sizeof(float));
    if (!ctx_prefill || !ctx_new) {
        free(ctx_prefill); free(ctx_new);
        free(noise); free(block_ids); free(anchors);
        free(out_tok); free(out_cand); free(out_probs);
        oc_dflash2_model_free(&model);
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < ctx_rows * n_target_w; i++)
        ctx_prefill[i] = (float)(int32_t)(dflash2_lcg(&seed) >> 8) / 16777216.0f;
    e = oc_dflash2_set_context(&model, ctx_prefill, ctx_rows);
    model.next_noise_pos = (int64_t)ctx_rows;
    if (e != OC_OK) {
        free(ctx_prefill); free(ctx_new);
        free(noise); free(block_ids); free(anchors);
        free(out_tok); free(out_cand); free(out_probs);
        oc_dflash2_model_free(&model);
        return e;
    }

    /* Deterministic noise BEFORE any timing: the backbone loop below ran
     * on uninitialized malloc memory before this point, so backbone_ms
     * measured indeterminate inputs and was not reproducible. */
    for (size_t i = 0; i < block * H; i++)
        noise[i] = (float)(int32_t)(dflash2_lcg(&seed) >> 8) / 16777216.0f;

    /* Backbone-only timing (forward_debug runs the exact same layer math
     * as propose without the selector/lm_head scan). Reported separately
     * so the draft-model cost and the target-owned lm_head scan are
     * distinguishable. */
    float *bb_hidden = malloc(block * H * sizeof(float));
    double bb_total = 0.0;
    if (bb_hidden) {
        for (uint32_t it = 0; it < iters; it++) {
            double s0 = dflash2_now_sec();
            if (oc_dflash2_forward_debug(&model, noise, block, bb_hidden)
                    != OC_OK) {
                s0 = 0.0; /* exclude failed calls from the average */
                break;
            }
            bb_total += dflash2_now_sec() - s0;
        }
    }
    double bb_avg = iters ? bb_total / (double)iters : 0.0;

    cli_info("running %u propose steps (block %zu, %zu drafts/step, %zu threads, %zu ctx rows)",
             iters + warmup, block, block - 1, oc_parallel_n_threads(), ctx_rows);

    for (uint32_t it = 0; it < iters + warmup; it++) {
        /* Fresh synthetic inputs each step. */
        for (size_t i = 0; i < block * H; i++)
            noise[i] = (float)(int32_t)(dflash2_lcg(&seed) >> 8) / 16777216.0f;
        for (size_t i = 0; i < (size_t)(block - 1) * n_target_w; i++)
            ctx_new[i] = (float)(int32_t)(dflash2_lcg(&seed) >> 8) / 16777216.0f;
        for (size_t i = 0; i < block; i++)
            block_ids[i] = dflash2_lcg(&seed) % (uint32_t)vocab;

        /* Per-step: only the newly produced rows enter the context, like
         * the reference's output.hidden_states[:, :produced, :]. */
        e = oc_dflash2_set_context(&model, ctx_new, block - 1);
        if (e != OC_OK) break;
        anchors[0] = block_ids[0];

        double s0 = dflash2_now_sec();
        e = oc_dflash2_propose(&model, anchors, 1, noise, block, block_ids,
                               &lm, 0.0f, out_tok, out_cand, out_probs);
        double dt = dflash2_now_sec() - s0;
        if (e != OC_OK) break;
        if (it >= warmup) {
            total += dt;
            drafted_total += block - 1;
            if (dt < best) best = dt;
        }
        /* Advance like a server: the accepted tokens are committed and the
         * ring window slides. */
        model.next_noise_pos += (int64_t)(block - 1);
    }
    double prefill_fuse_ms = 0.0; /* reported as 0; fusion is amortized */
    (void)prefill_fuse_ms;
    free(ctx_prefill); free(ctx_new);
    if (e != OC_OK) {
        cli_error("propose failed (%s)", oc_error_msg(e));
        free(noise); free(block_ids); free(anchors);
        free(out_tok); free(out_cand); free(out_probs);
        oc_dflash2_model_free(&model);
        return e;
    }

    double avg = total / (double)iters;
    double per_draft_tok = avg / (double)(block - 1);

    if (ctx->output_format == OC_CLI_OUTPUT_JSON) {
        char model_esc[512];
        if (json_escape(ctx->model_path, model_esc, sizeof(model_esc)) == 0)
            snprintf(model_esc, sizeof(model_esc), "?");
        printf("{\"command\":\"dflash2\",\"model\":\"%s\","
               "\"block_size\":%zu,\"drafts_per_step\":%zu,"
               "\"threads\":%zu,\"ctx_rows\":%zu,\"steps\":%u,"
               "\"drafted_total\":%llu,"
               "\"avg_step_ms\":%.3f,\"best_step_ms\":%.3f,"
               "\"backbone_ms\":%.3f,"
               "\"draft_tok_per_s\":%.1f}\n",
               model_esc, block, block - 1,
               oc_parallel_n_threads(), ctx_rows, iters,
               (unsigned long long)drafted_total,
               avg * 1e3, best * 1e3, bb_avg * 1e3, 1.0 / per_draft_tok);
    } else {
        printf("DFlash2 propose benchmark — %s\n", ctx->model_path);
        printf("  block size:        %zu (%zu drafts/step)\n",
               block, block - 1);
        printf("  threads:           %zu\n", oc_parallel_n_threads());
        printf("  context rows:      %zu (KV window %zu)\n", ctx_rows, ctx_cap);
        printf("  steps:             %u (+%u warmup)\n", iters, warmup);
        printf("  avg step:          %.3f ms\n", avg * 1e3);
        printf("  best step:         %.3f ms\n", best * 1e3);
        printf("  backbone-only:     %.3f ms/step\n", bb_avg * 1e3);
        printf("  draft throughput:  %.1f draft tok/s\n", 1.0 / per_draft_tok);
        printf("  (synthetic target inputs; the GLM-5.3-Flash target is "
               "321B and not co-resident)\n");
    }

    free(noise); free(block_ids); free(anchors);
    free(out_tok); free(out_cand); free(out_probs); free(bb_hidden);
    if (lm.bf16)
        oc_dflash2_free_huge(lm.bf16, (size_t)vocab * H * sizeof(uint16_t));
    oc_dflash2_model_free(&model);
    return OC_OK;
}
