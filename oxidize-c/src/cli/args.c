/*
 * args.c — CLI argument parsing (extracted from main.c for testability).
 *
 * OcCliArgs + oc_cli_parse_args live here (no main()) so they link into both
 * the oxidize-c binary (via main.c) and the test_runner (via tests/test_cli.c).
 */
#include "args.h"

#include "oxidize/cli_commands.h"

#include <stdlib.h>
#include <string.h>

void oc_cli_args_defaults(OcCliArgs *a)
{
    memset(a, 0, sizeof(*a));
    a->n_predict      = 128;
    a->threads        = 0;
    a->numa           = "none";
    a->temperature    = 0.0f;
    a->top_k          = 40;
    a->top_p          = 0.95f;
    a->repeat_penalty = 1.1f;
    a->seed           = 0;
    a->host           = "127.0.0.1";
    a->port           = 8080;
    a->bench_iterations = 3;
    a->min_p          = 0.0f;
    a->mirostat_tau   = 0.0f;
    a->mirostat_eta   = 0.1f;
}

static bool match(const char *arg, const char *long_name)
{
    return strcmp(arg, long_name) == 0;
}

static bool parse_value_flag(OcCliArgs *a, const char *arg, const char *val,
                             bool *consumed_val)
{
    *consumed_val = false;
    if (val == NULL) return false;
    if (match(arg, "--model"))          { a->model_path = val; *consumed_val = true; }
    else if (match(arg, "--prompt"))     { a->prompt = val; *consumed_val = true; }
    else if (match(arg, "--prompt-file")){ a->prompt_file = val; *consumed_val = true; }
    else if (match(arg, "--n-predict"))  { a->n_predict = val[0] == '-' ? 0u : (uint32_t)strtoul(val, NULL, 10); *consumed_val = true; }
    else if (match(arg, "--ctx"))        { a->n_ctx = val[0] == '-' ? 0u : (uint32_t)strtoul(val, NULL, 10); *consumed_val = true; }
    else if (match(arg, "--kv"))         { a->kv_type = val; *consumed_val = true; }
    else if (match(arg, "--threads"))    { a->threads = atoi(val); *consumed_val = true; }
    else if (match(arg, "--batch-size")) { a->batch_size = val[0] == '-' ? 0u : (uint32_t)strtoul(val, NULL, 10); *consumed_val = true; }
    else if (match(arg, "--prefill-chunk-size")) { a->prefill_chunk_size = val[0] == '-' ? 0u : (uint32_t)strtoul(val, NULL, 10); *consumed_val = true; }
    else if (match(arg, "--numa"))       { a->numa = val; *consumed_val = true; }
    else if (match(arg, "--temperature")||match(arg,"--temp")){a->temperature=(float)atof(val);*consumed_val=true; }
    else if (match(arg, "--top-k"))       { a->top_k = val[0] == '-' ? 0u : (uint32_t)strtoul(val, NULL, 10); *consumed_val = true; }
    else if (match(arg, "--top-p"))       { a->top_p = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--repeat-penalty")) { a->repeat_penalty = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--seed"))        { a->seed = strtoull(val, NULL, 10); *consumed_val = true; }
    else if (match(arg, "--host"))        { a->host = val; *consumed_val = true; }
    else if (match(arg, "--port"))        { a->port = atoi(val); *consumed_val = true; }
    else if (match(arg, "--api-key"))     { a->api_key = val; *consumed_val = true; }
    else if (match(arg, "--rate-limit"))  { a->rate_limit_rpm = atoi(val); *consumed_val = true; }
    else if (match(arg, "--cors-origin")) { a->cors_origin = val; *consumed_val = true; }
    else if (match(arg, "--draft-model"))  { a->draft_model = val; *consumed_val = true; }
    else if (match(arg, "--draft-tokens")) { a->draft_tokens = atoi(val); *consumed_val = true; }
    else if (match(arg, "--spec-type"))    { a->spec_type = val; *consumed_val = true; }
    else if (match(arg, "--quantize"))     { a->quantize_input = val; *consumed_val = true; }
    else if (match(arg, "--output"))       { a->quantize_output = val; *consumed_val = true; }
    else if (match(arg, "--quant-type"))   { a->quantize_type = val; *consumed_val = true; }
    else if (match(arg, "--backend"))       { a->backend = val; *consumed_val = true; }
    else if (match(arg, "--min-p"))         { a->min_p = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--mirostat-tau"))  { a->mirostat_tau = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--mirostat-eta"))  { a->mirostat_eta = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--bench-iters"))   { a->bench_iterations = val[0] == '-' ? 0 : atoi(val); *consumed_val = true; }
    else return false;
    return true;
}

void oc_cli_parse_args(int argc, char **argv, OcCliArgs *a)
{
    oc_cli_args_defaults(a);
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (match(arg, "--auto"))            { a->auto_tune = true; continue; }
        if (match(arg, "--no-auto"))         { a->no_auto = true; continue; }
        if (match(arg, "--print-plan"))     { a->print_plan = true; continue; }
        if (match(arg, "--serve-api"))       { a->serve_api = true; continue; }
        if (match(arg, "--stream"))          { a->stream = true; continue; }
        if (match(arg, "--bench"))           { a->bench = true; continue; }
        if (match(arg, "--inspect"))         { a->inspect = true; continue; }
        if (match(arg, "--perplexity") || match(arg, "--ppl")) { a->perplexity = true; continue; }
        if (match(arg, "--cuda-selftest"))   { a->cuda_selftest = true; continue; }
        if (match(arg, "--verbose") || match(arg, "-v")) { a->verbose = true; continue; }
        if (match(arg, "--help") || match(arg, "-h"))    { a->show_help = true; continue; }
        if (match(arg, "--version"))         { a->show_version = true; continue; }

        bool consumed_val = false;
        const char *val = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (parse_value_flag(a, arg, val, &consumed_val)) {
            if (consumed_val) i++;
            continue;
        }
        if (arg[0] == '-' && arg[1] == '-') {
            /* unknown --flag: ignored (would log, but args.c has no logger to
             * keep it dependency-free for the test_runner link) */
        } else if (a->prompt == NULL) {
            a->prompt = arg;
        }
    }
}

/* ─── Subcommand parsing ──────────────────────────────────────────────────
 *
 * `oxidize-c <subcommand> [flags]` populates an OcCliContext directly rather
 * than going through OcCliArgs, which only carries the flags the legacy
 * flag-only invocation needs. The two paths coexist: main() tries the
 * subcommand form first and falls back to the flag form when argv[1] is not
 * a recognized subcommand name, so `oxidize-c --model m.gguf --prompt hi`
 * keeps working unchanged. */

bool oc_cli_context_parse(int argc, char **argv, OcCliContext *ctx)
{
    oc_cli_context_defaults(ctx);
    if (argc < 2 || argv[1][0] == '-') return false;
    OcCliCommand cmd = oc_cli_command_parse(argv[1]);
    if (cmd == OC_CLI_CMD_NONE) return false;
    ctx->command = cmd;

    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];
        /* Valueless flags. */
        if (match(arg, "--auto"))       { ctx->auto_tune = true; continue; }
        if (match(arg, "--no-auto"))    { ctx->no_auto = true; continue; }
        if (match(arg, "--json"))       { ctx->output_format = OC_CLI_OUTPUT_JSON; continue; }
        if (match(arg, "--no-special")) { ctx->tokens_no_special = true; continue; }
        if (match(arg, "--bench-no-eos")) { ctx->bench_no_eos = true; continue; }
        if (match(arg, "--verbose") || match(arg, "-v")) { ctx->verbose = true; continue; }

        const char *val = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (val == NULL) {
            /* A trailing bare word is the prompt for generation commands. */
            if (arg[0] != '-' && ctx->prompt == NULL) ctx->prompt = arg;
            continue;
        }

        /* Model + generation. */
        if      (match(arg, "--model"))          { ctx->model_path = val; i++; }
        else if (match(arg, "--prompt"))         { ctx->prompt = val; i++; }
        else if (match(arg, "--prompt-file"))    { ctx->prompt_file = val; i++; }
        else if (match(arg, "--n-predict"))      { ctx->n_predict = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--ctx"))            { ctx->n_ctx = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--kv"))             { ctx->kv_type = val; i++; }
        else if (match(arg, "--prefill-chunk-size")) { ctx->prefill_chunk_size = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--threads"))        { ctx->threads = atoi(val); i++; }
        else if (match(arg, "--numa"))           { ctx->numa = val; i++; }
        else if (match(arg, "--backend"))        { ctx->backend = val; i++; }
        /* Sampling. */
        else if (match(arg, "--temperature") || match(arg, "--temp")) { ctx->temperature = (float)atof(val); i++; }
        else if (match(arg, "--top-k"))          { ctx->top_k = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--top-p"))          { ctx->top_p = (float)atof(val); i++; }
        else if (match(arg, "--repeat-penalty")) { ctx->repeat_penalty = (float)atof(val); i++; }
        else if (match(arg, "--seed"))           { ctx->seed = strtoull(val, NULL, 10); i++; }
        else if (match(arg, "--min-p"))          { ctx->min_p = (float)atof(val); i++; }
        else if (match(arg, "--mirostat-tau"))   { ctx->mirostat_tau = (float)atof(val); i++; }
        else if (match(arg, "--mirostat-eta"))   { ctx->mirostat_eta = (float)atof(val); i++; }
        /* Server. */
        else if (match(arg, "--host"))           { ctx->host = val; i++; }
        else if (match(arg, "--port"))           { ctx->port = atoi(val); i++; }
        else if (match(arg, "--api-key"))        { ctx->api_key = val; i++; }
        else if (match(arg, "--rate-limit"))     { ctx->rate_limit_rpm = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--cors-origin"))    { ctx->cors_origin = val; i++; }
        /* Benchmark. */
        else if (match(arg, "--bench-iters"))    { ctx->bench_iterations = val[0] == '-' ? 0 : atoi(val); i++; }
        else if (match(arg, "--bench-warmup"))   { ctx->bench_warmup = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--bench-tokens"))   { ctx->bench_tokens = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--bench-prompt-tokens")) { ctx->bench_prompt_tokens = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--bench-decode-tokens")) { ctx->bench_decode_tokens = (uint32_t)strtoul(val, NULL, 10); i++; }
        /* Quantize / convert / merge / prune. */
        else if (match(arg, "--input"))          { ctx->input_path = val; i++; }
        else if (match(arg, "--output"))         { ctx->output_path = val; i++; }
        else if (match(arg, "--quant-type") || match(arg, "--target")) { ctx->target_type = val; i++; }
        else if (match(arg, "--arch"))           { ctx->arch = val; i++; }
        else if (match(arg, "--strategy")) {
            /* Shared spelling: merge, prune, and finetune each read it into
             * their own field, and only one of them runs per invocation. */
            ctx->merge_strategy = val; ctx->prune_strategy = val;
            ctx->ft_strategy = val; i++;
        }
        else if (match(arg, "--slerp-t"))        { ctx->merge_slerp_t = (float)atof(val); i++; }
        else if (match(arg, "--density"))        { ctx->merge_density = (float)atof(val); i++; }
        else if (match(arg, "--sparsity"))       { ctx->prune_sparsity = (float)atof(val); i++; }
        /* Finetune. */
        else if (match(arg, "--dataset"))        { ctx->dataset_path = val; i++; }
        else if (match(arg, "--output-dir"))     { ctx->output_dir = val; i++; }
        else if (match(arg, "--resume-from"))    { ctx->resume_from = val; i++; }
        else if (match(arg, "--lora-rank"))      { ctx->lora_rank = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--lora-alpha"))     { ctx->lora_alpha = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--epochs"))         { ctx->epochs = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--batch-size"))     { ctx->batch_size = (uint32_t)strtoul(val, NULL, 10); i++; }
        else if (match(arg, "--lr"))             { ctx->learning_rate = (float)atof(val); i++; }
        /* Download. */
        else if (match(arg, "--repo"))           { ctx->hf_repo = val; i++; }
        else if (match(arg, "--file"))           { ctx->hf_file = val; i++; }
        else if (match(arg, "--cache-dir"))      { ctx->cache_dir = val; i++; }
        /* Perplexity / tokenize. */
        else if (match(arg, "--max-tokens"))     { ctx->ppl_max_tokens = (size_t)strtoull(val, NULL, 10); i++; }
        else if (match(arg, "--ids"))            { ctx->token_ids_str = val; i++; }
        else if (arg[0] != '-' && ctx->prompt == NULL) { ctx->prompt = arg; }
    }
    return true;
}
