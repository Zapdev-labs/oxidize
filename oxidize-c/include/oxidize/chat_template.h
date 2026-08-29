/* chat_template.h — Chat template rendering for various model formats. */
#ifndef OXIDIZE_CHAT_TEMPLATE_H
#define OXIDIZE_CHAT_TEMPLATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_TPL_MAX_ROLES 4
#define OC_TPL_MAX_MESSAGES 128
#define OC_TPL_MAX_ROLE_LEN 32
#define OC_TPL_MAX_CONTENT_LEN 8192
#define OC_TPL_MAX_OUTPUT 65536


typedef enum {
    OC_TPL_CHATML    = 0,
    OC_TPL_LLAMA2    = 1,
    OC_TPL_LLAMA3    = 2,
    OC_TPL_MISTRAL   = 3,
    OC_TPL_ALPACA    = 4,
    OC_TPL_VICUNA    = 5,
    OC_TPL_ZEPHYR    = 6,
    OC_TPL_CUSTOM    = 7,
} OcChatTemplateType;

typedef enum {
    OC_ROLE_SYSTEM    = 0,
    OC_ROLE_USER      = 1,
    OC_ROLE_ASSISTANT = 2,
    OC_ROLE_TOOL      = 3,
} OcChatRole;

typedef struct {
    OcChatRole role;
    char content[OC_TPL_MAX_CONTENT_LEN];
} OcChatMessage;

typedef struct {
    OcChatTemplateType type;
    char system_prompt[OC_TPL_MAX_CONTENT_LEN];
    char bos_token[32];
    char eos_token[32];
    /* For custom templates. */
    char user_tag[64];
    char assistant_tag[64];
    char system_tag[64];
} OcChatTemplate;


/* Initialize a template with default tokens for the given type. */
OcError oc_chat_tpl_init(OcChatTemplate *tpl, OcChatTemplateType type);

/* Add a message to the conversation. */
OcError oc_chat_tpl_add_message(OcChatMessage *messages, size_t *n_msgs,
                                OcChatRole role, const char *content);

/* Render a conversation using the template. Writes the rendered prompt
 * to `out` (cap bytes). Returns OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_chat_tpl_render(const OcChatTemplate *tpl,
                           const OcChatMessage *messages, size_t n_msgs,
                           char *out, size_t cap);

/* Get the template type name. */
const char *oc_chat_tpl_type_name(OcChatTemplateType type);

/* Get the role name string. */
const char *oc_chat_role_name(OcChatRole role);

/* Parse a template type from string. */
OcChatTemplateType oc_chat_tpl_parse_type(const char *name);

/* Add a generation prompt (assistant header) to the rendered conversation. */
OcError oc_chat_tpl_add_generation_prompt(const OcChatTemplate *tpl,
                                          char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CHAT_TEMPLATE_H */
