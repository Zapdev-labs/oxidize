#ifndef OXIDIZE_CHAT_MODE_H
#define OXIDIZE_CHAT_MODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/chat.h"
#include "oxidize/llama.h"
#include "oxidize/tokenizer.h"
#include "oxidize/sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A single chat message in the conversation history. */
typedef struct OcChatHistoryEntry {
    char *role;       /* "system", "user", "assistant" */
    char *content;     /* message text                   */
} OcChatHistoryEntry;

/* Conversation history (dynamic array). */
typedef struct OcChatHistory {
    OcChatHistoryEntry *entries;
    size_t n_entries;
    size_t capacity;
    OcChatTemplate template;
} OcChatHistory;

/* Initialize chat history. */
void oc_chat_history_init(OcChatHistory *h, OcChatTemplate tmpl);

/* Append a message to the history. Takes ownership of role/content only on
 * OC_OK; on error (e.g. OC_ERR_OOM) the caller retains ownership and must
 * free both. */
OcError oc_chat_history_append(OcChatHistory *h, char *role, char *content);

/* Render the full conversation history as a prompt string using the
 * configured chat template. Caller frees the returned string. */
OcError oc_chat_history_render(const OcChatHistory *h, char **out_text);

/* Free the chat history. */
void oc_chat_history_free(OcChatHistory *h);

/* Run the interactive chat REPL.
 * Reads user input from stdin, generates responses, and maintains
 * conversation history. Exits on EOF or "/quit". */
OcError oc_chat_run(OcLlamaModel *model, OcTokenizer *tok,
                    const OcSamplerConfig *scfg,
                    uint32_t n_predict,
                    const char *system_prompt,
                    OcChatTemplate tmpl);

/* Process a single user input: append to history, render prompt,
 * tokenize, forward, generate response, append response to history. */
OcError oc_chat_process_input(OcLlamaModel *model, OcTokenizer *tok,
                              const OcSamplerConfig *scfg,
                              OcChatHistory *h,
                              const char *user_input,
                              uint32_t n_predict,
                              char **out_response);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CHAT_MODE_H */
