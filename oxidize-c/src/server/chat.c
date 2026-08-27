/*
 * chat.c — chat template formatting.
 *
 * Renders OpenAI-style messages into model-specific prompt strings.
 */
#include "oxidize/chat.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static size_t append_str(char *out, size_t *pos, size_t cap, const char *s)
{
    size_t len = strlen(s);
    if (*pos + len + 1 > cap) return 0; /* truncation */
    memcpy(out + *pos, s, len);
    *pos += len;
    out[*pos] = '\0';
    return len;
}

static size_t append_fmt(char *out, size_t *pos, size_t cap, const char *fmt, ...)
{
    if (*pos + 1 > cap) return 0;
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(out + *pos, cap - *pos, fmt, args);
    va_end(args);
    if (n < 0 || (size_t)n >= cap - *pos) return 0;
    *pos += (size_t)n;
    return (size_t)n;
}

OcChatTemplate oc_chat_detect(const char *arch_str)
{
    if (!arch_str) return OC_CHAT_CHATML;
    if (strncmp(arch_str, "gemma", 5) == 0) return OC_CHAT_GEMMA;
    if (strncmp(arch_str, "llama", 5) == 0) {
        return OC_CHAT_LLAMA3;
    }
    /* Qwen, Mistral, others → ChatML. */
    return OC_CHAT_CHATML;
}

static int ascii_ieq(char a, char b)
{
    if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
    if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
    return a == b;
}

static int ascii_istrstr(const char *hay, const char *needle)
{
    if (!hay || !needle || !*needle) return 0;
    for (const char *h = hay; *h; h++) {
        const char *a = h, *b = needle;
        while (*a && *b && ascii_ieq(*a, *b)) { a++; b++; }
        if (*b == '\0') return 1;
    }
    return 0;
}

OcChatTemplate oc_chat_detect_named(const char *arch_str, const char *name)
{
    OcChatTemplate t = oc_chat_detect(arch_str);
    if (t != OC_CHAT_LLAMA3) return t;
    if (ascii_istrstr(name, "llama3") || ascii_istrstr(name, "llama-3"))
        return OC_CHAT_LLAMA3;
    if (ascii_istrstr(name, "llama-2") || ascii_istrstr(name, "llama2"))
        return OC_CHAT_LLAMA2;
    return t;
}

size_t oc_chat_render_message(OcChatTemplate template,
                              const char *role, const char *content,
                              char *out, size_t out_cap,
                              bool is_first, bool is_last)
{
    (void)is_first;
    if (!out || out_cap == 0 || !role || !content) return 0;
    size_t pos = 0;
    bool ok = true;
    out[0] = '\0';

    switch (template) {
    case OC_CHAT_CHATML:
        /* <|im_start|>role\ncontent<|im_end|>\n */
        ok = append_fmt(out, &pos, out_cap, "<|im_start|>%s\n%s<|im_end|>\n",
                        role, content) != 0;
        if (ok && is_last)
            ok = append_str(out, &pos, out_cap, "<|im_start|>assistant\n") != 0;
        break;

    case OC_CHAT_LLAMA3:
        /* <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|> */
        ok = append_fmt(out, &pos, out_cap,
                        "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>",
                        role, content) != 0;
        if (ok && is_last)
            ok = append_str(out, &pos, out_cap,
                            "<|start_header_id|>assistant<|end_header_id|>\n\n") != 0;
        break;

    case OC_CHAT_LLAMA2:
        /* [INST] content [/INST] */
        if (strcmp(role, "user") == 0) {
            ok = append_fmt(out, &pos, out_cap, "[INST] %s [/INST]", content) != 0;
        } else if (strcmp(role, "system") == 0) {
            ok = append_fmt(out, &pos, out_cap, "<<SYS>>\n%s\n<</SYS>>\n\n", content) != 0;
        } else {
            ok = append_str(out, &pos, out_cap, content) != 0 || content[0] == '\0';
        }
        if (ok && is_last) ok = append_str(out, &pos, out_cap, " ") != 0;
        break;

    case OC_CHAT_GEMMA:
        /* <start_of_turn>role\ncontent<end_of_turn>\n */
        ok = append_fmt(out, &pos, out_cap, "<start_of_turn>%s\n%s<end_of_turn>\n",
                        role, content) != 0;
        if (ok && is_last)
            ok = append_str(out, &pos, out_cap, "<start_of_turn>model\n") != 0;
        break;

    case OC_CHAT_PLAIN:
        ok = append_str(out, &pos, out_cap, content) != 0 || content[0] == '\0';
        if (ok) ok = append_str(out, &pos, out_cap, "\n") != 0;
        break;

    default:
        ok = append_str(out, &pos, out_cap, content) != 0 || content[0] == '\0';
        break;
    }

    return ok ? pos : 0;
}

size_t oc_chat_render_messages(OcChatTemplate template,
                               const char * const *roles,
                               const char * const *contents,
                               size_t n_msgs,
                               char *out, size_t out_cap)
{
    size_t total = 0;
    out[0] = '\0';

    for (size_t i = 0; i < n_msgs; i++) {
        char msg[8192];
        size_t n = oc_chat_render_message(template, roles[i], contents[i],
                                          msg, sizeof(msg),
                                          i == 0, i + 1 == n_msgs);
        if (n == 0) return 0;
        if (total + n + 1 > out_cap) return 0;
        memcpy(out + total, msg, n);
        total += n;
        out[total] = '\0';
    }

    return total;
}
