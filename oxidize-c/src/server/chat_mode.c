/*
 * chat_mode.c — Interactive chat mode REPL implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/chat_mode.h"

#include "oxidize/chat.h"
#include "oxidize/llama.h"
#include "oxidize/sampling.h"
#include "oxidize/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Chat history ──────────────────────────────────────────────────────── */

void oc_chat_history_init(OcChatHistory *h, OcChatTemplate tmpl)
{
    if (!h) return;
    h->entries = NULL;
    h->n_entries = 0;
    h->capacity = 0;
    h->template = tmpl;
}

OcError oc_chat_history_append(OcChatHistory *h, char *role, char *content)
{
    if (!h || !role || !content) return OC_ERR_INVALID_ARG;
    if (h->n_entries >= h->capacity) {
        size_t new_cap = h->capacity ? h->capacity * 2 : 8;
        OcChatHistoryEntry *p = realloc(h->entries, new_cap * sizeof(*p));
        if (!p) return OC_ERR_OOM;
        h->entries = p;
        h->capacity = new_cap;
    }
    h->entries[h->n_entries].role = role;
    h->entries[h->n_entries].content = content;
    h->n_entries++;
    return OC_OK;
}

OcError oc_chat_history_render(const OcChatHistory *h, char **out_text)
{
    if (!h || !out_text) return OC_ERR_INVALID_ARG;
    *out_text = NULL;

    /* Collect roles and contents into arrays. */
    const char **roles = malloc(h->n_entries * sizeof(char *));
    const char **contents = malloc(h->n_entries * sizeof(char *));
    if (!roles || !contents) { free(roles); free(contents); return OC_ERR_OOM; }

    for (size_t i = 0; i < h->n_entries; i++) {
        roles[i] = h->entries[i].role;
        contents[i] = h->entries[i].content;
    }

    /* Render into a buffer sized from the actual history so long
     * conversations are never silently truncated by a fixed cap. */
    size_t buf_cap = 1 << 16;
    size_t needed = 1024;
    for (size_t i = 0; i < h->n_entries; i++) {
        needed += strlen(h->entries[i].role) + strlen(h->entries[i].content) + 128;
    }
    if (needed > buf_cap) buf_cap = needed;
    char *buf = malloc(buf_cap);
    if (!buf) { free(roles); free(contents); return OC_ERR_OOM; }

    size_t written = oc_chat_render_messages(h->template, roles, contents,
                                             h->n_entries, buf, buf_cap);
    if (written == 0 && h->n_entries > 0) {
        /* Retry with larger buffer. */
        free(buf);
        buf_cap = 1 << 20;
        buf = malloc(buf_cap);
        if (!buf) { free(roles); free(contents); return OC_ERR_OOM; }
        written = oc_chat_render_messages(h->template, roles, contents,
                                          h->n_entries, buf, buf_cap);
    }

    free(roles);
    free(contents);

    if (written == 0 && h->n_entries > 0) {
        free(buf);
        return OC_ERR_OOM;
    }

    *out_text = buf;
    return OC_OK;
}

void oc_chat_history_free(OcChatHistory *h)
{
    if (!h) return;
    for (size_t i = 0; i < h->n_entries; i++) {
        free(h->entries[i].role);
        free(h->entries[i].content);
    }
    free(h->entries);
    memset(h, 0, sizeof(*h));
}

/* ─── Single input processing ───────────────────────────────────────────── */

OcError oc_chat_process_input(OcLlamaModel *model, OcTokenizer *tok,
                              const OcSamplerConfig *scfg,
                              OcChatHistory *h,
                              const char *user_input,
                              uint32_t n_predict,
                              char **out_response)
{
    if (!model || !tok || !scfg || !h || !user_input || !out_response)
        return OC_ERR_INVALID_ARG;
    *out_response = NULL;

    /* Append user message to history. */
    char *role_copy = strdup("user");
    char *content_copy = strdup(user_input);
    if (!role_copy || !content_copy) { free(role_copy); free(content_copy); return OC_ERR_OOM; }
    OcError e = oc_chat_history_append(h, role_copy, content_copy);
    if (e != OC_OK) { free(role_copy); free(content_copy); return e; }

    /* Render full conversation as prompt. */
    char *prompt = NULL;
    e = oc_chat_history_render(h, &prompt);
    if (e != OC_OK) return e;

    /* Tokenize prompt. */
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    OcSpecialTokenPolicy pol = tok->has_add_bos_token && tok->add_bos_token
        ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
    e = oc_tokenizer_encode(tok, prompt, pol, &ids, &n_ids);
    free(prompt);
    if (e != OC_OK) return e;

    /* Initialize session and forward prompt. */
    OcLlamaSession sess;
    e = oc_llama_session_init(model, &sess);
    if (e != OC_OK) { free(ids); return e; }

    if (n_ids == 0) { free(ids); oc_llama_session_free(&sess); return OC_ERR_TOKENIZER; }

    /* Prefill: all but the last prompt token without logits, then the last
     * one with logits so sampling starts from the prompt's next-token
     * distribution. */
    for (size_t i = 0; i + 1 < n_ids; i++) {
        e = oc_llama_forward(&sess, ids[i], NULL);
        if (e != OC_OK) break;
    }
    if (e == OC_OK) {
        e = oc_llama_forward(&sess, ids[n_ids - 1], sess.logits);
    }
    free(ids);
    if (e != OC_OK) { oc_llama_session_free(&sess); return e; }

    /* Generate response. */
    float *logits = sess.logits;
    size_t resp_cap = 4096;
    size_t resp_len = 0;
    char *response = malloc(resp_cap);
    if (!response) { oc_llama_session_free(&sess); return OC_ERR_OOM; }
    response[0] = '\0';

    for (uint32_t t = 0; t < n_predict; t++) {
        uint32_t token = oc_sample(logits, model->cfg.vocab_size, (OcSamplerConfig *)scfg, NULL, 0);
        if (tok->has_eos && token == tok->eos_id) break;

        char *piece = NULL;
        if (oc_tokenizer_decode(tok, &token, 1, &piece) == OC_OK && piece) {
            size_t pl = strlen(piece);
            if (resp_len + pl + 1 >= resp_cap) {
                resp_cap = resp_cap * 2 + pl;
                char *tmp = realloc(response, resp_cap);
                if (!tmp) {
                    free(piece);
                    free(response);
                    oc_llama_session_free(&sess);
                    return OC_ERR_OOM;
                }
                response = tmp;
            }
            memcpy(response + resp_len, piece, pl);
            resp_len += pl;
            response[resp_len] = '\0';
            free(piece);
        }

        /* Feed the sampled token back for the next step. */
        e = oc_llama_forward(&sess, token, logits);
        if (e != OC_OK) break;
    }

    oc_llama_session_free(&sess);

    /* Append assistant response to history. */
    char *a_role = strdup("assistant");
    char *a_content = strdup(response);
    if (a_role && a_content) {
        oc_chat_history_append(h, a_role, a_content);
    } else {
        free(a_role); free(a_content);
    }

    *out_response = response;
    return OC_OK;
}

/* ─── Interactive REPL ─────────────────────────────────────────────────── */

OcError oc_chat_run(OcLlamaModel *model, OcTokenizer *tok,
                    const OcSamplerConfig *scfg,
                    uint32_t n_predict,
                    const char *system_prompt,
                    OcChatTemplate tmpl)
{
    if (!model || !tok || !scfg) return OC_ERR_INVALID_ARG;

    OcChatHistory h;
    oc_chat_history_init(&h, tmpl);

    /* Add system prompt if provided. */
    if (system_prompt) {
        char *role = strdup("system");
        char *content = strdup(system_prompt);
        if (role && content) oc_chat_history_append(&h, role, content);
        else { free(role); free(content); }
    }

    printf("oxidize-c chat (template=%d). Type /quit to exit.\n", (int)tmpl);
    fflush(stdout);

    char input[8192];
    while (1) {
        printf("\nuser> ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;

        /* Remove trailing newline. */
        size_t len = strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r'))
            input[--len] = '\0';

        if (len == 0) continue;
        if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) break;

        char *response = NULL;
        OcError e = oc_chat_process_input(model, tok, scfg, &h,
                                          input, n_predict, &response);
        if (e == OC_OK && response) {
            printf("\nassistant> %s\n", response);
            free(response);
        } else {
            fprintf(stderr, "error: generation failed (%s)\n", oc_error_msg(e));
        }
    }

    oc_chat_history_free(&h);
    return OC_OK;
}
