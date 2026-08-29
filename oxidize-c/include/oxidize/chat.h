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

/* Render a single message into the template buffer. */
size_t oc_chat_render_message(OcChatTemplate template,
                              const char *role, const char *content,
                              char *out, size_t out_cap,
                              bool is_first, bool is_last);

/* Render an array of messages into a single prompt string. */
size_t oc_chat_render_messages(OcChatTemplate template,
                               const char * const *roles,
                               const char * const *contents,
                               size_t n_msgs,
                               char *out, size_t out_cap);

/* Auto-detect chat template from architecture string. */
OcChatTemplate oc_chat_detect(const char *arch_str);

/* Prefer tokenizer.chat_template contents when present, then filename, then arch. */
OcChatTemplate oc_chat_detect_named(const char *arch_str, const char *name);
OcChatTemplate oc_chat_detect_full(const char *arch_str, const char *name,
                                    const char *chat_template);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CHAT_H */
