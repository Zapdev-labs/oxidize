/* mesh_chat.c — Chat message routing over the mesh network. */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/mesh_chat.h"

#include <stdio.h>
#include <string.h>

/* Helpers.                                                            */

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0) return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static bool role_valid(OcMeshChatRole r)
{
    return (size_t)r < OC_MESH_CHAT_ROLE__COUNT;
}

/* Public API.                                                         */

OcError oc_mesh_chat_init(OcMeshChatConversation *conv, const char *conversation_id)
{
    if (!conv) return OC_ERR_INVALID_ARG;
    memset(conv, 0, sizeof(*conv));
    if (conversation_id) {
        copy_str(conv->conversation_id, sizeof(conv->conversation_id), conversation_id);
    } else {
        copy_str(conv->conversation_id, sizeof(conv->conversation_id), "anonymous");
    }
    conv->count = 0;
    conv->next_timestamp = 0;
    return OC_OK;
}

OcError oc_mesh_chat_add_message(OcMeshChatConversation *conv, OcMeshChatRole role,
                                 const char *content, const char *sender_id)
{
    if (!conv) return OC_ERR_INVALID_ARG;
    if (!role_valid(role)) return OC_ERR_INVALID_ARG;
    if (conv->count >= OC_MESH_CHAT_MAX_MESSAGES) return OC_ERR_OOM;

    OcMeshChatMessage *m = &conv->messages[conv->count];
    memset(m, 0, sizeof(*m));
    m->role = role;
    copy_str(m->content, sizeof(m->content), content ? content : "");
    copy_str(m->sender_id, sizeof(m->sender_id), sender_id ? sender_id : "");
    m->timestamp = conv->next_timestamp++;
    conv->count++;
    return OC_OK;
}

OcError oc_mesh_chat_get_messages(const OcMeshChatConversation *conv,
                                  OcMeshChatRole filter_role,
                                  OcMeshChatMessage *out, size_t max_out,
                                  size_t *n_out)
{
    if (!conv || !out || !n_out) return OC_ERR_INVALID_ARG;
    size_t copied = 0;
    bool filter = role_valid(filter_role);
    for (size_t i = 0; i < conv->count && copied < max_out; i++) {
        const OcMeshChatMessage *m = &conv->messages[i];
        if (filter && m->role != filter_role) continue;
        out[copied++] = *m;
    }
    *n_out = copied;
    return OC_OK;
}

const char *oc_mesh_chat_role_name(OcMeshChatRole role)
{
    switch (role) {
    case OC_MESH_CHAT_ROLE_SYSTEM:    return "system";
    case OC_MESH_CHAT_ROLE_USER:       return "user";
    case OC_MESH_CHAT_ROLE_ASSISTANT:  return "assistant";
    default:                           return "unknown";
    }
}

size_t oc_mesh_chat_message_count(const OcMeshChatConversation *conv, OcMeshChatRole filter_role)
{
    if (!conv) return 0;
    bool filter = role_valid(filter_role);
    size_t n = 0;
    for (size_t i = 0; i < conv->count; i++) {
        if (!filter || conv->messages[i].role == filter_role) n++;
    }
    return n;
}

OcError oc_mesh_chat_clear(OcMeshChatConversation *conv)
{
    if (!conv) return OC_ERR_INVALID_ARG;
    char id[OC_MESH_CHAT_ID_LEN];
    memcpy(id, conv->conversation_id, sizeof(id));
    memset(conv->messages, 0, sizeof(conv->messages));
    conv->count = 0;
    conv->next_timestamp = 0;
    memcpy(conv->conversation_id, id, sizeof(id));
    return OC_OK;
}

OcError oc_mesh_chat_serialize(const OcMeshChatConversation *conv,
                               char *out, size_t out_len)
{
    if (!conv || !out || out_len == 0) return OC_ERR_INVALID_ARG;

    size_t off = 0;
    int n = snprintf(out + off, out_len - off,
                     "conversation: %s\n", conv->conversation_id);
    if (n < 0) return OC_ERR_INTERNAL;
    off += (size_t)n;
    if (off >= out_len) return OC_ERR_INTERNAL;

    for (size_t i = 0; i < conv->count; i++) {
        const OcMeshChatMessage *m = &conv->messages[i];
        n = snprintf(out + off, out_len - off, "%s: %s\n",
                     oc_mesh_chat_role_name(m->role), m->content);
        if (n < 0) return OC_ERR_INTERNAL;
        if ((size_t)n >= out_len - off) return OC_ERR_INTERNAL;
        off += (size_t)n;
    }
    return OC_OK;
}
