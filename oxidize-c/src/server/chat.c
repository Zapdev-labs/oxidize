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
        /* Llama-2 vs Llama-3: check for "3" in arch string.
         * In practice, the GGUF metadata doesn't distinguish; default to
         * Llama-3 format (more common in 2024+). */
        return OC_CHAT_LLAMA3;
    }
    /* Qwen, Mistral, others → ChatML. */
    return OC_CHAT_CHATML;
}

size_t oc_chat_render_message(OcChatTemplate template,
                              const char *role, const char *content,
                              char *out, size_t out_cap,
                              bool is_first, bool is_last)
{
    size_t pos = 0;
    out[0] = '\0';

    switch (template) {
    case OC_CHAT_CHATML:
        /* <|im_start|>role\ncontent<|im_end|>\n */
        append_fmt(out, &pos, out_cap, "<|im_start|>%s\n%s<|im_end|>\n",
                   role, content);
        if (is_last) append_str(out, &pos, out_cap, "<|im_start|>assistant\n");
        break;

    case OC_CHAT_LLAMA3:
        /* <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|> */
        append_fmt(out, &pos, out_cap,
                   "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>",
                   role, content);
        if (is_last) {
            append_str(out, &pos, out_cap,
                       "<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        break;

    case OC_CHAT_LLAMA2:
        /* [INST] content [/INST] */
        if (strcmp(role, "user") == 0) {
            append_fmt(out, &pos, out_cap, "[INST] %s [/INST]", content);
        } else if (strcmp(role, "system") == 0) {
            append_fmt(out, &pos, out_cap, "<<SYS>>\n%s\n<</SYS>>\n\n", content);
        } else {
            append_str(out, &pos, out_cap, content);
        }
        if (is_last) append_str(out, &pos, out_cap, " ");
        break;

    case OC_CHAT_GEMMA:
        /* <start_of_turn>role\ncontent<end_of_turn>\n */
        append_fmt(out, &pos, out_cap, "<start_of_turn>%s\n%s<end_of_turn>\n",
                   role, content);
        if (is_last) append_str(out, &pos, out_cap, "<start_of_turn>model\n");
        break;

    case OC_CHAT_PLAIN:
        append_str(out, &pos, out_cap, content);
        append_str(out, &pos, out_cap, "\n");
        break;

    default:
        append_str(out, &pos, out_cap, content);
        break;
    }

    return pos;
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
        if (n == 0) continue;
        if (total + n + 1 > out_cap) break;
        memcpy(out + total, msg, n);
        total += n;
        out[total] = '\0';
    }

    return total;
}
