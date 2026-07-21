/*
 * args.c — CLI argument parsing (extracted from main.c for testability).
 *
 * OcCliArgs + oc_cli_parse_args live here (no main()) so they link into both
 * the oxidize-c binary (via main.c) and the test_runner (via tests/test_cli.c).
 */
#include "args.h"

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
    else if (match(arg, "--threads"))    { a->threads = atoi(val); *consumed_val = true; }
    else if (match(arg, "--numa"))       { a->numa = val; *consumed_val = true; }
    else if (match(arg, "--temperature")||match(arg,"--temp")){a->temperature=(float)atof(val);*consumed_val=true; }
    else if (match(arg, "--top-k"))       { a->top_k = val[0] == '-' ? 0u : (uint32_t)strtoul(val, NULL, 10); *consumed_val = true; }
    else if (match(arg, "--top-p"))       { a->top_p = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--repeat-penalty")) { a->repeat_penalty = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--seed"))        { a->seed = strtoull(val, NULL, 10); *consumed_val = true; }
    else if (match(arg, "--host"))        { a->host = val; *consumed_val = true; }
    else if (match(arg, "--port"))        { a->port = atoi(val); *consumed_val = true; }
    else if (match(arg, "--draft-model"))  { a->draft_model = val; *consumed_val = true; }
    else if (match(arg, "--draft-tokens")) { a->draft_tokens = atoi(val); *consumed_val = true; }
    else if (match(arg, "--quantize"))     { a->quantize_input = val; *consumed_val = true; }
    else if (match(arg, "--output"))       { a->quantize_output = val; *consumed_val = true; }
    else if (match(arg, "--quant-type"))   { a->quantize_type = val; *consumed_val = true; }
    else if (match(arg, "--backend"))       { a->backend = val; *consumed_val = true; }
    else if (match(arg, "--min-p"))         { a->min_p = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--mirostat-tau"))  { a->mirostat_tau = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--mirostat-eta"))  { a->mirostat_eta = (float)atof(val); *consumed_val = true; }
    else if (match(arg, "--bench-iters"))   { a->bench_iterations = atoi(val); *consumed_val = true; }
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
