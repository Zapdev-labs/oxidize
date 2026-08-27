/*
 * chat.h — chat template formatting for OpenAI-compatible server.
 *
 * Renders OpenAI-style message arrays (role + content) into model-specific
 * prompt strings. Supports ChatML (Qwen/Mistral), Llama-3, Llama-2, and
 * Gemma formats. Auto-detected from architecture or specified via CLI.
 */
#ifndef OXIDIZE_CHAT_H
#define OXIDIZE_CHAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_CHAT_TEMPLATE_AUTO = 0,   /* detect from model arch               */
    OC_CHAT_CHATML,              /* Qwen/Mistral: <|im_start|>role\n...   */
    OC_CHAT_LLAMA3,              /* Llama-3: <|start_header_id|>role...    */
    OC_CHAT_LLAMA2,              /* Llama-2: [INST] ... [/INST]           */
    OC_CHAT_GEMMA,              /* Gemma: <start_of_turn>role\n...        */
    OC_CHAT_PLAIN,              /* Plain text concatenation              */
} OcChatTemplate;

/* Render a single message into the template buffer.
 *
 *   template  : chat template type
 *   role      : "system", "user", "assistant"
 *   content   : message text
 *   out       : output buffer (caller-allocated)
 *   out_cap   : capacity of out
 *   is_first  : true if this is the first message
 *   is_last   : true if this is the last message (adds generation prompt)
 *
 * Returns the number of bytes written (excluding NUL). Returns 0 on error
 * or truncation. */
size_t oc_chat_render_message(OcChatTemplate template,
                              const char *role, const char *content,
                              char *out, size_t out_cap,
                              bool is_first, bool is_last);

/* Render an array of messages into a single prompt string.
 *
 *   template : chat template type
 *   roles    : array of role strings (e.g. "system", "user", "assistant")
 *   contents : array of content strings
 *   n_msgs   : number of messages
 *   out      : output buffer
 *   out_cap  : capacity
 *
 * Returns bytes written (excluding NUL). */
size_t oc_chat_render_messages(OcChatTemplate template,
                               const char * const *roles,
                               const char * const *contents,
                               size_t n_msgs,
                               char *out, size_t out_cap);

/* Auto-detect chat template from architecture string. */
OcChatTemplate oc_chat_detect(const char *arch_str);

/* Like oc_chat_detect, then Llama-2 filenames (llama-2 / llama2 without llama3)
 * override the Llama-3 default. */
OcChatTemplate oc_chat_detect_named(const char *arch_str, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CHAT_H */
