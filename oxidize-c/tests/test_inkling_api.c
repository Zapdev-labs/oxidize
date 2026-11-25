#define _POSIX_C_SOURCE 200809L
/* test_inkling_api.c — Criterion tests for src/server/inkling_api.c: the
 * Inkling-2 chat template (outputs checked against the model's Jinja
 * template) and the reply parser. */
#include <criterion/criterion.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/inkling_api.h"

static char *render(const char *json, const char **err)
{
    return oc_inkling_chat_render(json, strlen(json), 0.9f, err);
}

Test(inkling_api, render_user_message_with_default_effort)
{
    const char *err = NULL;
    char *out = render("{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}", &err);
    cr_assert_not_null(out, "%s", err);
    cr_assert_str_eq(out, "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                          "<|message_user|><|content_text|>Hello<|end_message|>");
    free(out);
}

Test(inkling_api, render_system_before_effort_and_numeric_effort)
{
    const char *err = NULL;
    char *out = render("{\"messages\":[{\"role\":\"system\",\"content\":\"S\"},{\"role\":\"user\",\"content\":\"q\"}],"
                       "\"reasoning_effort\":0.5}", &err);
    cr_assert_not_null(out, "%s", err);
    cr_assert_str_eq(out, "<|message_system|><|content_text|>S<|end_message|>"
                          "<|message_system|><|content_text|>Thinking effort level: 0.5<|end_message|>"
                          "<|message_user|><|content_text|>q<|end_message|>");
    free(out);
}

/* Tool declarations with sorted keys, a tool call whose arguments arrive as
 * a JSON string (rendered with sorted top-level keys), and a tool result
 * named through its tool_call_id. */
Test(inkling_api, render_tools_and_tool_calls)
{
    const char *err = NULL;
    char *out = render(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},"
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"c9\",\"function\":{\"name\":\"bash\","
        "\"arguments\":\"{\\\"command\\\":\\\"ls -la\\\",\\\"Zed\\\":null,\\\"apple\\\":true}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"c9\",\"content\":[{\"type\":\"text\",\"text\":\"out\"}]}],"
        "\"reasoning_effort\":\"xhigh\",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read_file\","
        "\"description\":\"Read a file\\nfrom disk \\\"quoted\\\"\",\"parameters\":{\"type\":\"object\","
        "\"required\":[\"path\"],\"properties\":{\"path\":{\"type\":\"string\"},\"ratio\":{\"type\":\"number\","
        "\"default\":0.25}}}}}]}",
        &err);
    cr_assert_not_null(out, "%s", err);
    cr_assert_str_eq(out,
        "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Read a file\\nfrom disk \\\"quoted\\\"\","
        "\"name\":\"read_file\",\"parameters\":{\"properties\":{\"path\":{\"type\":\"string\"},\"ratio\":"
        "{\"default\":0.25,\"type\":\"number\"}},\"required\":[\"path\"],\"type\":\"object\"},\"type\":\"function\"}]"
        "<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|>"
        "<|message_user|><|content_text|>q<|end_message|><|message_model|>bash<|content_invoke_tool_json|>"
        "{\"name\":\"bash\",\"args\":{\"Zed\":null,\"apple\":true,\"command\":\"ls -la\"}}<|end_message|>"
        "<|content_model_end_sampling|><|message_tool|>bash<|content_text|>out<|end_message|>");
    free(out);
}

Test(inkling_api, render_rejects_bad_requests)
{
    const char *err = NULL;
    cr_assert_null(render("{\"messages\":[{\"role\":\"bogus\",\"content\":\"q\"}]}", &err));
    cr_assert_str_eq(err, "unknown message role");
    cr_assert_null(render("{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"reasoning_effort\":\"huge\"}", &err));
    cr_assert_null(render("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\"}]}]}", &err));
    cr_assert_null(render("[1,2]", &err));
    cr_assert_null(render("{\"messages\":[{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c\","
                          "\"function\":{\"name\":\"x<|message_system|>\",\"arguments\":\"{}\"}}]}]}", &err));
    cr_assert_str_eq(err, "names must not contain control tokens");
    cr_assert_null(render("{\"messages\":[{\"role\":\"tool\",\"name\":\"<|end_message|>\",\"content\":\"r\"}]}", &err));
    cr_assert_str_eq(err, "names must not contain control tokens");
}

typedef struct {
    char reasoning[256], content[256], name[64], args[256];
    int  n_calls;
} Parts;

static void append(char *dst, size_t cap, const char *text, size_t len)
{
    const size_t n = strlen(dst);
    cr_assert_lt(n + len, cap, "test buffer too small");
    memcpy(dst + n, text, len);
    dst[n + len] = '\0';
}

static void collect(OcInklingPartKind kind, const char *text, size_t len, const char *args, void *user)
{
    Parts *p = user;
    if (kind == OC_INK_PART_REASONING) append(p->reasoning, sizeof p->reasoning, text, len);
    else if (kind == OC_INK_PART_CONTENT) append(p->content, sizeof p->content, text, len);
    else {
        p->n_calls++;
        snprintf(p->name, sizeof p->name, "%.*s", (int)len, text);
        snprintf(p->args, sizeof p->args, "%s", args);
    }
}

static void push(OcInklingReply *r, const char *s)
{
    oc_inkling_reply_push(r, s, strlen(s), strncmp(s, "<|", 2) == 0);
}

Test(inkling_api, reply_splits_thinking_text_and_tool_calls)
{
    Parts p = { 0 };
    OcInklingReply r;
    oc_inkling_reply_init(&r, collect, &p);
    const char *toks[] = { "<|message_model|>", "<|content_thinking|>", "Let me ", "look.", "<|end_message|>",
                           "<|message_model|>", "<|content_text|>", "Reading", " it.", "<|end_message|>",
                           "<|message_model|>", "read", "_file", "<|content_invoke_tool_json|>", "{\"name\":\"read_file\",",
                           "\"args\":{\"path\":\"/tmp/x\",\"limit\":5}}", "<|end_message|>",
                           "<|content_model_end_sampling|>", "<|message_model|>", "ignored" };
    for (size_t i = 0; i < sizeof toks / sizeof toks[0]; i++) push(&r, toks[i]);
    cr_assert_str_eq(p.reasoning, "Let me look.");
    cr_assert_str_eq(p.content, "Reading it.");
    cr_assert_eq(p.n_calls, 1);
    cr_assert_str_eq(p.name, "read_file");
    cr_assert_str_eq(p.args, "{\"path\":\"/tmp/x\",\"limit\":5}");
    cr_assert(r.done);
    oc_inkling_reply_free(&r);
}

Test(inkling_api, reply_keeps_an_unparsable_tool_body_as_text)
{
    Parts p = { 0 };
    OcInklingReply r;
    oc_inkling_reply_init(&r, collect, &p);
    const char *toks[] = { "<|message_model|>", "x", "<|content_invoke_tool_json|>", "{not json", "<|end_message|>" };
    for (size_t i = 0; i < sizeof toks / sizeof toks[0]; i++) push(&r, toks[i]);
    cr_assert_eq(p.n_calls, 0);
    cr_assert_str_eq(p.content, "{not json");
    cr_assert(!r.done);
    oc_inkling_reply_free(&r);
}
