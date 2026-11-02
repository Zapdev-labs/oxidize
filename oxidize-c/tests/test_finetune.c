/* test_finetune.c — finetuning tests. */
/* Expose mkdtemp: BSD on Apple (_DARWIN_C_SOURCE), POSIX elsewhere. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <criterion/criterion.h>
#include "oxidize/finetune.h"
#include <string.h>

Test(ft, strategy_name)
{
    cr_assert_str_eq(oc_ft_strategy_name(OC_FT_SFT), "sft");
    cr_assert_str_eq(oc_ft_strategy_name(OC_FT_SELF_TRAIN), "self-train");
    cr_assert_str_eq(oc_ft_strategy_name(OC_FT_DPO), "dpo");
    cr_assert_str_eq(oc_ft_strategy_name(OC_FT_PPO), "ppo");
}

Test(ft, null_config)
{
    cr_assert_neq(oc_finetune_run(NULL), OC_OK);
}

Test(ft, format_sft)
{
    char buf[1024];
    cr_assert_eq(oc_finetune_format_sft("You are helpful.", "Hello", "Hi there!", buf, sizeof(buf)), OC_OK);
    cr_assert(strstr(buf, "system") != NULL);
    cr_assert(strstr(buf, "You are helpful.") != NULL);
    cr_assert(strstr(buf, "user") != NULL);
    cr_assert(strstr(buf, "Hello") != NULL);
    cr_assert(strstr(buf, "assistant") != NULL);
    cr_assert(strstr(buf, "Hi there!") != NULL);
}

Test(ft, format_sft_null_system)
{
    char buf[512];
    cr_assert_eq(oc_finetune_format_sft(NULL, "Hello", "Hi", buf, sizeof(buf)), OC_OK);
    cr_assert(strstr(buf, "system") == NULL);
    cr_assert(strstr(buf, "user") != NULL);
}

Test(ft, format_sft_small_buf)
{
    char buf[10];
    cr_assert_neq(oc_finetune_format_sft("long system prompt here", "user", "assistant", buf, sizeof(buf)), OC_OK);
}

/* ─── mold dataset ─────────────────────────────────────────────────── */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

static void ft_tmpdir(char *dir, size_t cap)
{
    snprintf(dir, cap, "/tmp/oc_ft_test_XXXXXX");
    cr_assert_not_null(mkdtemp(dir));
}

static void ft_write(const char *path, const char *s)
{
    FILE *f = fopen(path, "w");
    cr_assert_not_null(f);
    fputs(s, f);
    fclose(f);
}

static char *ft_slurp(const char *path)
{
    FILE *f = fopen(path, "r");
    cr_assert_not_null(f);
    static char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);
    return buf;
}

Test(finetune, mold_parses_messages_rows)
{
    char dir[64], in[128], out[128];
    ft_tmpdir(dir, sizeof(dir));
    snprintf(in, sizeof(in), "%s/in.jsonl", dir);
    snprintf(out, sizeof(out), "%s/out.jsonl", dir);
    ft_write(in,
        "{\"messages\":[{\"role\":\"system\",\"content\":\"S\"},"
        "{\"role\":\"user\",\"content\":\"hi\"},"
        "{\"role\":\"assistant\",\"content\":\"hello\"}],\"source\":\"t\"}\n"
        "{\"user\":\"flat\",\"assistant\":\"ok\"}\n"
        "{\"instruction\":\"do\",\"input\":\"x\",\"output\":\"done\"}\n"
        "{\"messages\":[{\"role\":\"user\",\"content\":\"no answer\"}]}\n");
    uint32_t k = 0, d = 0;
    cr_assert_eq(oc_finetune_mold_dataset(in, out, &k, &d), OC_OK);
    cr_assert_eq(k, 3);
    cr_assert_eq(d, 1);
    const char *s = ft_slurp(out);
    cr_assert(strstr(s, "{\"role\":\"system\",\"content\":\"S\"}") != NULL);
    cr_assert(strstr(s, "{\"role\":\"assistant\",\"content\":\"hello\"}],\"source\":\"t\"") != NULL);
    cr_assert(strstr(s, "\"content\":\"flat\"") != NULL);
    cr_assert(strstr(s, "\"content\":\"do\\nx\"") != NULL);
}

Test(finetune, mold_decodes_all_escapes)
{
    char dir[64], in[128], out[128];
    ft_tmpdir(dir, sizeof(dir));
    snprintf(in, sizeof(in), "%s/in.jsonl", dir);
    snprintf(out, sizeof(out), "%s/out.jsonl", dir);
    /* \u00e9 -> é, surrogate pair -> U+1F600, \/ -> /, \b and \f kept. */
    ft_write(in, "{\"user\":\"caf\\u00e9 \\ud83d\\ude00 a\\/b\",\"assistant\":\"x\\by\\fz\"}\n");
    uint32_t k = 0;
    cr_assert_eq(oc_finetune_mold_dataset(in, out, &k, NULL), OC_OK);
    cr_assert_eq(k, 1);
    const char *s = ft_slurp(out);
    cr_assert(strstr(s, "caf\xc3\xa9 \xf0\x9f\x98\x80 a/b") != NULL, "got %s", s);
    cr_assert(strstr(s, "x\\u0008y\\u000cz") != NULL, "got %s", s);
}

Test(finetune, request_filter_normalizes)
{
    cr_assert(oc_finetune_request_blocked("Build RANSOMWARE now"));
    cr_assert(oc_finetune_request_blocked("a reverse---shell, please"));
    cr_assert(oc_finetune_request_blocked("Reverse\n\tShell"));
    cr_assert(!oc_finetune_request_blocked("explain how TLS works"));
    cr_assert(!oc_finetune_request_blocked("shellfish reverse order"));
    cr_assert(!oc_finetune_request_blocked(NULL));
}

Test(finetune, mold_filter_sees_decoded_text)
{
    char dir[64], in[128], out[128];
    ft_tmpdir(dir, sizeof(dir));
    snprintf(in, sizeof(in), "%s/in.jsonl", dir);
    snprintf(out, sizeof(out), "%s/out.jsonl", dir);
    /* "\u0072ansomware" decodes to the blocked word. */
    ft_write(in,
        "{\"user\":\"\\u0072ansomware\",\"assistant\":\"no\"}\n"
        "{\"messages\":[{\"role\":\"user\",\"content\":\"web\\u0073hell\"},"
        "{\"role\":\"assistant\",\"content\":\"no\"}]}\n"
        "{\"user\":\"fine\",\"assistant\":\"ok\"}\n");
    uint32_t k = 0, d = 0;
    cr_assert_eq(oc_finetune_mold_dataset(in, out, &k, &d), OC_OK);
    cr_assert_eq(k, 1);
    cr_assert_eq(d, 2);
}

Test(finetune, mold_same_path_is_safe_and_private)
{
    char dir[64], p[128];
    ft_tmpdir(dir, sizeof(dir));
    snprintf(p, sizeof(p), "%s/distill.jsonl", dir);
    ft_write(p, "{\"user\":\"q\",\"assistant\":\"a\"}\n");
    uint32_t k = 0;
    cr_assert_eq(oc_finetune_mold_dataset(p, p, &k, NULL), OC_OK);
    cr_assert_eq(k, 1);
    cr_assert(strstr(ft_slurp(p), "\"content\":\"q\"") != NULL);
    struct stat st;
    cr_assert_eq(stat(p, &st), 0);
    cr_assert_eq(st.st_mode & 0777, 0600);
}

Test(finetune, mold_read_error_leaves_no_output)
{
    char dir[64], out[128];
    ft_tmpdir(dir, sizeof(dir));
    snprintf(out, sizeof(out), "%s/out.jsonl", dir);
    /* fopen(dir, "r") succeeds on Linux, then getline fails with EISDIR. */
    cr_assert_eq(oc_finetune_mold_dataset(dir, out, NULL, NULL), OC_ERR_IO);
    cr_assert_neq(access(out, F_OK), 0);
}

Test(finetune, mold_counts_oversize_rows)
{
    char dir[64], in[128], out[128];
    ft_tmpdir(dir, sizeof(dir));
    snprintf(in, sizeof(in), "%s/in.jsonl", dir);
    snprintf(out, sizeof(out), "%s/out.jsonl", dir);
    FILE *f = fopen(in, "w");
    cr_assert_not_null(f);
    fputs("{\"user\":\"", f);
    for (size_t i = 0; i < ((size_t)1 << 20) + 16; i++) fputc('a', f);
    fputs("\",\"assistant\":\"b\"}\n{\"user\":\"q\",\"assistant\":\"a\"}\n", f);
    fclose(f);
    uint32_t k = 0, d = 0;
    cr_assert_eq(oc_finetune_mold_dataset(in, out, &k, &d), OC_OK);
    cr_assert_eq(k, 1);
    cr_assert_eq(d, 1);
}
