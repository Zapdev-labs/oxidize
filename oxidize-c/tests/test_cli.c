/*
 * test_cli.c — CLI argument parser tests.
 *
 * The flag parser (oc_cli_parse_args) is extracted from main.c into
 * src/cli/args.h so it can be unit-tested without invoking the binary.
 * VAL-CLI-001..006 cover defaults, value flags, boolean flags, positional
 * prompt, unknown-flag tolerance, and sampler-config wiring.
 *
 * main.c's run_generation is exercised end-to-end by running the built
 * `oxidize-c` binary with --help / --version / --prompt (smoke, manual).
 */
#include <criterion/criterion.h>

#include "../src/cli/args.h"
#include "oxidize/cli_commands.h"

#include <string.h>

Test(cli, defaults_are_sensible)
{
    OcCliArgs a;
    oc_cli_args_defaults(&a);
    cr_assert_eq(a.n_predict, 128, "default n_predict");
    cr_assert_eq(a.top_k, 40, "default top_k");
    cr_assert_float_eq(a.top_p, 0.95f, 1e-6f, "default top_p");
    cr_assert_float_eq(a.repeat_penalty, 1.1f, 1e-6f, "default penalty");
    cr_assert_float_eq(a.temperature, 0.0f, 1e-6f, "default temp (greedy)");
    cr_assert_str_eq(a.numa, "none", "default numa");
    cr_assert_str_eq(a.host, "127.0.0.1", "default host");
    cr_assert_eq(a.port, 8080, "default port");
    cr_assert(a.model_path == NULL, "no default model");
}

Test(cli, parses_value_flags)
{
    char *argv[] = {"oxidize-c",
                    "--model", "model.gguf",
                    "--prompt", "hello",
                    "--n-predict", "50",
                    "--threads", "8",
                    "--numa", "interleave",
                    "--temperature", "0.7",
                    "--top-k", "10",
                    "--seed", "12345"};
    OcCliArgs a;
    oc_cli_parse_args(17, argv, &a);
    cr_assert_str_eq(a.model_path, "model.gguf");
    cr_assert_str_eq(a.prompt, "hello");
    cr_assert_eq(a.n_predict, 50);
    cr_assert_eq(a.threads, 8);
    cr_assert_str_eq(a.numa, "interleave");
    cr_assert_float_eq(a.temperature, 0.7f, 1e-4f);
    cr_assert_eq(a.top_k, 10);
    cr_assert_eq(a.seed, 12345ull);
}

Test(cli, parses_boolean_flags)
{
    char *argv[] = {"oxidize-c", "--auto", "--print-plan", "-v"};
    OcCliArgs a;
    oc_cli_parse_args(4, argv, &a);
    cr_assert(a.auto_tune, "--auto sets auto_tune");
    cr_assert(a.print_plan, "--print-plan sets print_plan");
    cr_assert(a.verbose, "-v sets verbose");
}

Test(cli, positional_prompt_when_no_prompt_flag)
{
    char *argv[] = {"oxidize-c", "--model", "m.gguf", "some prompt text"};
    OcCliArgs a;
    oc_cli_parse_args(4, argv, &a);
    cr_assert_str_eq(a.model_path, "m.gguf");
    cr_assert_str_eq(a.prompt, "some prompt text", "positional arg becomes prompt");
}

Test(cli, temp_short_alias_works)
{
    char *argv[] = {"oxidize-c", "--temp", "0.5"};
    OcCliArgs a;
    oc_cli_parse_args(3, argv, &a);
    cr_assert_float_eq(a.temperature, 0.5f, 1e-4f, "--temp aliases --temperature");
}

Test(cli, no_auto_flag)
{
    char *argv[] = {"oxidize-c", "--no-auto"};
    OcCliArgs a;
    oc_cli_parse_args(2, argv, &a);
    cr_assert(a.no_auto, "--no-auto sets no_auto");
    cr_assert_not(a.auto_tune, "auto_tune stays false");
}

Test(cli, help_and_version_flags)
{
    char *argv1[] = {"oxidize-c", "--help"};
    OcCliArgs a;
    oc_cli_parse_args(2, argv1, &a);
    cr_assert(a.show_help, "--help sets show_help");
    char *argv2[] = {"oxidize-c", "--version"};
    oc_cli_parse_args(2, argv2, &a);
    cr_assert(a.show_version, "--version sets show_version");
}

Test(cli, cuda_selftest_flag)
{
    char *argv[] = {"oxidize-c", "--cuda-selftest"};
    OcCliArgs a;
    oc_cli_parse_args(2, argv, &a);
    cr_assert(a.cuda_selftest);
}

Test(cli, unknown_double_dash_flag_is_ignored)
{
    /* Unknown --foo bar should NOT consume "bar" as a value, and should not
     * crash. It logs a warning and continues. */
    char *argv[] = {"oxidize-c", "--unknown-flag", "--model", "x.gguf"};
    OcCliArgs a;
    oc_cli_parse_args(4, argv, &a);
    cr_assert_str_eq(a.model_path, "x.gguf", "parsing continues after unknown flag");
}

Test(cli, serve_api_flag)
{
    char *argv[] = {"oxidize-c", "--serve-api", "--host", "0.0.0.0", "--port", "9999"};
    OcCliArgs a;
    oc_cli_parse_args(6, argv, &a);
    cr_assert(a.serve_api, "--serve-api sets serve_api");
    cr_assert_str_eq(a.host, "0.0.0.0");
    cr_assert_eq(a.port, 9999);
}

Test(cli, bench_parses_fixed_workload_flags)
{
    char *argv[] = {"oxidize-c", "bench", "--model", "m.gguf",
                    "--bench-prompt-tokens", "64",
                    "--bench-decode-tokens", "32", "--bench-no-eos",
                    "--bench-warmup", "2", "--bench-iters", "5"};
    OcCliContext ctx;
    cr_assert(oc_cli_context_parse(13, argv, &ctx));
    cr_assert_eq(ctx.bench_prompt_tokens, 64);
    cr_assert_eq(ctx.bench_decode_tokens, 32);
    cr_assert(ctx.bench_no_eos);
    cr_assert_eq(ctx.bench_warmup, 2);
    cr_assert_eq(ctx.bench_iterations, 5);
}

Test(cli, parses_kv_type)
{
    char *argv[] = {"oxidize-c", "--model", "m.gguf", "--kv", "q8"};
    OcCliArgs a;
    oc_cli_parse_args(5, argv, &a);
    cr_assert_str_eq(a.kv_type, "q8");
}
