/*
 * chat_template.c — Chat template rendering implementation.
 */
#include "oxidize/chat_template.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static size_t buf_append(char *buf, size_t cap, size_t *off, const char *s)
{
    size_t slen = strlen(s);
    if (*off + slen + 1 > cap) return 0;
    memcpy(buf + *off, s, slen);
    *off += slen;
    return slen;
}

static size_t buf_appendf(char *buf, size_t cap, size_t *off,
                          const char *fmt, ...)
{
    va_list args, args2;
    va_start(args, fmt);
    va_copy(args2, args);
    int n = vsnprintf(NULL, 0, fmt, args);
    va_end(args);
    if (n < 0 || *off + (size_t)n + 1 > cap) { va_end(args2); return 0; }
    vsnprintf(buf + *off, cap - *off, fmt, args2);
    va_end(args2);
    *off += n;
    return n;
}

/* ─── API ──────────────────────────────────────────────────────────────── */

OcError oc_chat_tpl_init(OcChatTemplate *tpl, OcChatTemplateType type)
{
    if (!tpl) return OC_ERR_INVALID_ARG;
    memset(tpl, 0, sizeof(*tpl));
    tpl->type = type;

    switch (type) {
    case OC_TPL_CHATML:
        strcpy(tpl->bos_token, "<|im_start|>");
        strcpy(tpl->eos_token, "<|im_end|>");
        break;
    case OC_TPL_LLAMA2:
        strcpy(tpl->bos_token, "<s>");
        strcpy(tpl->eos_token, "</s>");
        break;
    case OC_TPL_LLAMA3:
        strcpy(tpl->bos_token, "<|begin_of_text|>");
        strcpy(tpl->eos_token, "<|end_of_text|>");
        break;
    case OC_TPL_MISTRAL:
        strcpy(tpl->bos_token, "<s>");
        strcpy(tpl->eos_token, "</s>");
        break;
    case OC_TPL_ALPACA:
        strcpy(tpl->bos_token, "");
        strcpy(tpl->eos_token, "");
        break;
    case OC_TPL_VICUNA:
        strcpy(tpl->bos_token, "");
        strcpy(tpl->eos_token, "");
        break;
    case OC_TPL_ZEPHYR:
        strcpy(tpl->bos_token, "<|system|>");
        strcpy(tpl->eos_token, "<|endoftext|>");
        break;
    default:
        break;
    }
    return OC_OK;
}

OcError oc_chat_tpl_add_message(OcChatMessage *messages, size_t *n_msgs,
                                OcChatRole role, const char *content)
{
    if (!messages || !n_msgs || !content) return OC_ERR_INVALID_ARG;
    if (*n_msgs >= OC_TPL_MAX_MESSAGES) return OC_ERR_OOM;

    OcChatMessage *msg = &messages[*n_msgs];
    msg->role = role;
    size_t clen = strlen(content);
    if (clen >= OC_TPL_MAX_CONTENT_LEN) clen = OC_TPL_MAX_CONTENT_LEN - 1;
    memcpy(msg->content, content, clen);
    msg->content[clen] = '\0';
    (*n_msgs)++;
    return OC_OK;
}

static const char *role_str(OcChatRole role)
{
    switch (role) {
    case OC_ROLE_SYSTEM:    return "system";
    case OC_ROLE_USER:      return "user";
    case OC_ROLE_ASSISTANT: return "assistant";
    case OC_ROLE_TOOL:      return "tool";
    default:                return "user";
    }
}

OcError oc_chat_tpl_render(const OcChatTemplate *tpl,
                           const OcChatMessage *messages, size_t n_msgs,
                           char *out, size_t cap)
{
    if (!tpl || !out || cap == 0) return OC_ERR_INVALID_ARG;
    if (n_msgs > 0 && !messages) return OC_ERR_INVALID_ARG;

    size_t off = 0;

    switch (tpl->type) {
    case OC_TPL_CHATML:
        for (size_t i = 0; i < n_msgs; i++) {
            buf_appendf(out, cap, &off, "<|im_start|>%s\n%s<|im_end|>\n",
                       role_str(messages[i].role), messages[i].content);
        }
        buf_append(out, cap, &off, "<|im_start|>assistant\n");
        break;

    case OC_TPL_LLAMA2: {
        bool has_system = false;
        for (size_t i = 0; i < n_msgs; i++) {
            if (messages[i].role == OC_ROLE_SYSTEM) {
                buf_appendf(out, cap, &off,
                    "[INST] <<SYS>>\n%s\n<</SYS>>\n\n", messages[i].content);
                has_system = true;
                continue;
            }
            if (messages[i].role == OC_ROLE_USER) {
                if (has_system && i > 0) {
                    /* Append to existing [INST] */
                    off -= 2; /* remove trailing "\n\n" */
                    buf_appendf(out, cap, &off, "%s [/INST]", messages[i].content);
                    has_system = false;
                } else {
                    buf_appendf(out, cap, &off, "[INST] %s [/INST]",
                               messages[i].content);
                }
                continue;
            }
            if (messages[i].role == OC_ROLE_ASSISTANT) {
                buf_appendf(out, cap, &off, " %s%s",
                           messages[i].content, tpl->eos_token);
            }
        }
        break;
    }

    case OC_TPL_LLAMA3:
        for (size_t i = 0; i < n_msgs; i++) {
            buf_appendf(out, cap, &off,
                "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>",
                role_str(messages[i].role), messages[i].content);
        }
        buf_append(out, cap, &off,
            "<|start_header_id|>assistant<|end_header_id|>\n\n");
        break;

    case OC_TPL_MISTRAL:
        for (size_t i = 0; i < n_msgs; i++) {
            if (messages[i].role == OC_ROLE_SYSTEM) {
                buf_appendf(out, cap, &off, "[INST] %s\n\n",
                           messages[i].content);
            } else if (messages[i].role == OC_ROLE_USER) {
                buf_appendf(out, cap, &off, "%s [/INST]", messages[i].content);
            } else {
                buf_appendf(out, cap, &off, " %s%s",
                           messages[i].content, tpl->eos_token);
            }
        }
        break;

    case OC_TPL_ALPACA:
        for (size_t i = 0; i < n_msgs; i++) {
            if (messages[i].role == OC_ROLE_SYSTEM) {
                buf_appendf(out, cap, &off, "%s\n\n",
                           messages[i].content);
            } else if (messages[i].role == OC_ROLE_USER) {
                buf_appendf(out, cap, &off,
                    "### Instruction:\n%s\n\n", messages[i].content);
            } else {
                buf_appendf(out, cap, &off,
                    "### Response:\n%s\n\n", messages[i].content);
            }
        }
        buf_append(out, cap, &off, "### Response:\n");
        break;

    case OC_TPL_VICUNA:
        for (size_t i = 0; i < n_msgs; i++) {
            if (messages[i].role == OC_ROLE_SYSTEM) {
                buf_appendf(out, cap, &off, "%s\n",
                           messages[i].content);
            } else if (messages[i].role == OC_ROLE_USER) {
                buf_appendf(out, cap, &off, "USER: %s\n",
                           messages[i].content);
            } else {
                buf_appendf(out, cap, &off, "ASSISTANT: %s\n",
                           messages[i].content);
            }
        }
        buf_append(out, cap, &off, "ASSISTANT:");
        break;

    case OC_TPL_ZEPHYR:
        for (size_t i = 0; i < n_msgs; i++) {
            buf_appendf(out, cap, &off, "<|%s|>\n%s<|endoftext|>\n",
                       role_str(messages[i].role), messages[i].content);
        }
        buf_append(out, cap, &off, "<|assistant|>\n");
        break;

    default:
        /* Generic format. */
        for (size_t i = 0; i < n_msgs; i++) {
            buf_appendf(out, cap, &off, "[%s] %s\n",
                       role_str(messages[i].role), messages[i].content);
        }
        break;
    }

    out[off] = '\0';
    return OC_OK;
}

const char *oc_chat_tpl_type_name(OcChatTemplateType type)
{
    switch (type) {
    case OC_TPL_CHATML:    return "chatml";
    case OC_TPL_LLAMA2:    return "llama2";
    case OC_TPL_LLAMA3:    return "llama3";
    case OC_TPL_MISTRAL:   return "mistral";
    case OC_TPL_ALPACA:    return "alpaca";
    case OC_TPL_VICUNA:    return "vicuna";
    case OC_TPL_ZEPHYR:    return "zephyr";
    case OC_TPL_CUSTOM:    return "custom";
    default:               return "unknown";
    }
}

const char *oc_chat_role_name(OcChatRole role)
{
    switch (role) {
    case OC_ROLE_SYSTEM:    return "system";
    case OC_ROLE_USER:      return "user";
    case OC_ROLE_ASSISTANT: return "assistant";
    case OC_ROLE_TOOL:      return "tool";
    default:                return "unknown";
    }
}

OcChatTemplateType oc_chat_tpl_parse_type(const char *name)
{
    if (!name) return OC_TPL_CHATML;
    if (strcmp(name, "chatml") == 0) return OC_TPL_CHATML;
    if (strcmp(name, "llama2") == 0) return OC_TPL_LLAMA2;
    if (strcmp(name, "llama3") == 0) return OC_TPL_LLAMA3;
    if (strcmp(name, "mistral") == 0) return OC_TPL_MISTRAL;
    if (strcmp(name, "alpaca") == 0) return OC_TPL_ALPACA;
    if (strcmp(name, "vicuna") == 0) return OC_TPL_VICUNA;
    if (strcmp(name, "zephyr") == 0) return OC_TPL_ZEPHYR;
    if (strcmp(name, "custom") == 0) return OC_TPL_CUSTOM;
    return OC_TPL_CHATML;
}

OcError oc_chat_tpl_add_generation_prompt(const OcChatTemplate *tpl,
                                          char *out, size_t cap)
{
    if (!tpl || !out || cap == 0) return OC_ERR_INVALID_ARG;

    switch (tpl->type) {
    case OC_TPL_CHATML:
        snprintf(out, cap, "<|im_start|>assistant\n");
        break;
    case OC_TPL_LLAMA3:
        snprintf(out, cap,
            "<|start_header_id|>assistant<|end_header_id|>\n\n");
        break;
    case OC_TPL_ALPACA:
        snprintf(out, cap, "### Response:\n");
        break;
    case OC_TPL_VICUNA:
        snprintf(out, cap, "ASSISTANT:");
        break;
    case OC_TPL_ZEPHYR:
        snprintf(out, cap, "<|assistant|>\n");
        break;
    default:
        snprintf(out, cap, "");
        break;
    }
    return OC_OK;
}
