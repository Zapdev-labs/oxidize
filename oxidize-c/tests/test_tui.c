#include <criterion/criterion.h>

#include "oxidize/tui.h"

Test(tui, fuzzy_is_case_insensitive_substring)
{
    cr_assert(oc_tui_fuzzy("chat", "Go to Chat"));
    cr_assert(oc_tui_fuzzy("", "x"));
    cr_assert_not(oc_tui_fuzzy("xyz", "Go to chat"));
}

Test(tui, sse_delta_reads_content)
{
    char out[64];
    cr_assert(oc_tui_sse_delta("data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}", out, sizeof(out)));
    cr_assert_str_eq(out, "Hi");
    cr_assert_eq(oc_tui_sse_delta("data: [DONE]", out, sizeof(out)), 0);
    cr_assert_eq(oc_tui_sse_delta("event: ping", out, sizeof(out)), 0);
}

Test(tui, metric_parser_sums_named_series)
{
    const char *text =
        "oxidize_tokens_per_second 12.5\n"
        "oxidize_requests_total{path=\"/v1\"} 3\n"
        "# comment\n";
    cr_assert_float_eq(oc_tui_metric_value(text, "oxidize_tokens_per_second"), 12.5, 1e-6);
    cr_assert_float_eq(oc_tui_metric_value(text, "oxidize_requests_total"), 3.0, 1e-6);
}
