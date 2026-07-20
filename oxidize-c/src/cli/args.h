/*
 * args.h — CLI argument parsing (extracted for testability).
 *
 * OcCliArgs + parse_args are used by src/cli/main.c and tests/test_cli.c.
 * Kept in src/cli/ (not include/) because it's CLI-internal, not library API.
 */
#ifndef OXIDIZE_CLI_ARGS_H
#define OXIDIZE_CLI_ARGS_H

#include <stdbool.h>
#include <stdint.h>

typedef struct OcCliArgs {
    const char *model_path;
    const char *prompt;
    const char *prompt_file;
    int        n_predict;
    int        threads;
    const char *numa;
    bool       auto_tune;
    bool       no_auto;
    bool       print_plan;
    bool       serve_api;
    const char *host;
    int        port;
    float      temperature;
    int        top_k;
    float      top_p;
    float      repeat_penalty;
    uint64_t   seed;
    bool       verbose;
    bool       show_help;
    bool       show_version;
} OcCliArgs;

void oc_cli_args_defaults(OcCliArgs *a);
void oc_cli_parse_args(int argc, char **argv, OcCliArgs *a);

#endif /* OXIDIZE_CLI_ARGS_H */
