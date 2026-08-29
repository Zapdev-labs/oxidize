#ifndef OXIDIZE_MESH_CHAT_H
#define OXIDIZE_MESH_CHAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_MESH_CHAT_MAX_MESSAGES 256
#define OC_MESH_CHAT_CONTENT_LEN  4096
#define OC_MESH_CHAT_ID_LEN      64

/* Conversation participant roles. */
typedef enum {
    OC_MESH_CHAT_ROLE_SYSTEM = 0,
    OC_MESH_CHAT_ROLE_USER = 1,
    OC_MESH_CHAT_ROLE_ASSISTANT = 2,
    OC_MESH_CHAT_ROLE__COUNT, /* sentinel; not a valid role */
} OcMeshChatRole;

/* A single chat message. All string fields are NUL-terminated fixed-size
 * buffers; truncation is silent (long inputs are clipped). */
typedef struct {
    OcMeshChatRole role;
    char content[OC_MESH_CHAT_CONTENT_LEN];
    char sender_id[OC_MESH_CHAT_ID_LEN];
    uint64_t timestamp;        /* monotonic counter assigned by add_message */
} OcMeshChatMessage;

/* A conversation holds up to OC_MESH_CHAT_MAX_MESSAGES messages. */
typedef struct {
    OcMeshChatMessage messages[OC_MESH_CHAT_MAX_MESSAGES];
    size_t count;
    char conversation_id[OC_MESH_CHAT_ID_LEN];
    uint64_t next_timestamp; /* next timestamp to assign; increments by 1 */
} OcMeshChatConversation;

/* Initialize a conversation. conversation_id may be NULL (an "anonymous"
 * id is assigned). Resets all state including the timestamp counter. */
OcError oc_mesh_chat_init(OcMeshChatConversation *conv, const char *conversation_id);

/* Append a message to the conversation. `content` and `sender_id` may be
 * NULL (treated as empty). Returns OC_ERR_OOM when the conversation is
 * full. Returns OC_ERR_INVALID_ARG on an invalid role. */
OcError oc_mesh_chat_add_message(OcMeshChatConversation *conv, OcMeshChatRole role,
                                 const char *content, const char *sender_id);

/* Copy up to `max_out` messages into `out`, filtered by `filter_role`.
 * Pass OC_MESH_CHAT_ROLE__COUNT to disable filtering. Writes the number of
 * messages copied to `*n_out`. */
OcError oc_mesh_chat_get_messages(const OcMeshChatConversation *conv,
                                  OcMeshChatRole filter_role,
                                  OcMeshChatMessage *out, size_t max_out,
                                  size_t *n_out);

/* Human-readable role name ("system", "user", "assistant"). Returns
 * "unknown" for invalid roles. Never returns NULL. */
const char *oc_mesh_chat_role_name(OcMeshChatRole role);

/* Count messages in the conversation matching `filter_role`. Pass
 * OC_MESH_CHAT_ROLE__COUNT to count all roles. */
size_t oc_mesh_chat_message_count(const OcMeshChatConversation *conv, OcMeshChatRole filter_role);

/* Remove all messages from the conversation, keeping the conversation_id
 * and resetting the timestamp counter to zero. */
OcError oc_mesh_chat_clear(OcMeshChatConversation *conv);

/* Serialize the conversation into a text format into `out`. Each message */
OcError oc_mesh_chat_serialize(const OcMeshChatConversation *conv,
                               char *out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MESH_CHAT_H */
