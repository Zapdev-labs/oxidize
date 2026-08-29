/* test_mesh_chat.c — Chat routing over mesh tests. */
#include "framework.h"
#include "oxidize/mesh_chat.h"
#include <string.h>

/* ----------------------------------------------------------------- */
/* init.                                                              */
/* ----------------------------------------------------------------- */

Test(mesh_chat, init)
{
    OcMeshChatConversation conv;
    cr_assert_eq(oc_mesh_chat_init(&conv, "conv-1"), OC_OK);
    cr_assert_str_eq(conv.conversation_id, "conv-1");
    cr_assert_eq(conv.count, 0u);
    cr_assert_eq(conv.next_timestamp, 0u);
}

Test(mesh_chat, init_null_id)
{
    OcMeshChatConversation conv;
    cr_assert_eq(oc_mesh_chat_init(&conv, NULL), OC_OK);
    cr_assert_str_eq(conv.conversation_id, "anonymous");
    cr_assert_eq(conv.count, 0u);
}

OC_TEST_NULL_SAFE(mesh_chat, init_null_conv,
        cr_assert_neq(oc_mesh_chat_init(NULL, "x"), OC_OK);)

/* ----------------------------------------------------------------- */
/* add_message.                                                       */
/* ----------------------------------------------------------------- */

Test(mesh_chat, add_message_basic)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    cr_assert_eq(oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER,
                                          "hello", "alice"), OC_OK);
    cr_assert_eq(conv.count, 1u);
    cr_assert_eq(conv.messages[0].role, OC_MESH_CHAT_ROLE_USER);
    cr_assert_str_eq(conv.messages[0].content, "hello");
    cr_assert_str_eq(conv.messages[0].sender_id, "alice");
    cr_assert_eq(conv.messages[0].timestamp, 0u);
    cr_assert_eq(conv.next_timestamp, 1u);
}

Test(mesh_chat, add_message_increments_timestamp)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "a", "x");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_ASSISTANT, "b", "y");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_SYSTEM, "c", "z");
    cr_assert_eq(conv.messages[0].timestamp, 0u);
    cr_assert_eq(conv.messages[1].timestamp, 1u);
    cr_assert_eq(conv.messages[2].timestamp, 2u);
    cr_assert_eq(conv.next_timestamp, 3u);
}

Test(mesh_chat, add_message_null_content_sender)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    cr_assert_eq(oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER,
                                          NULL, NULL), OC_OK);
    cr_assert_str_eq(conv.messages[0].content, "");
    cr_assert_str_eq(conv.messages[0].sender_id, "");
}

Test(mesh_chat, add_message_invalid_role)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    cr_assert_neq(oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE__COUNT,
                                            "x", "y"), OC_OK);
    cr_assert_eq(conv.count, 0u);
}

Test(mesh_chat, add_message_null_conv)
{
    cr_assert_neq(oc_mesh_chat_add_message(NULL, OC_MESH_CHAT_ROLE_USER,
                                           "x", "y"), OC_OK);
}

/* ----------------------------------------------------------------- */
/* get_messages + filtering.                                          */
/* ----------------------------------------------------------------- */

Test(mesh_chat, get_messages_all)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u1", "a");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_ASSISTANT, "a1", "b");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u2", "c");

    OcMeshChatMessage out[8];
    size_t n = 0;
    cr_assert_eq(oc_mesh_chat_get_messages(&conv, OC_MESH_CHAT_ROLE__COUNT,
                                           out, 8, &n), OC_OK);
    cr_assert_eq(n, 3u);
    cr_assert_str_eq(out[0].content, "u1");
    cr_assert_str_eq(out[2].content, "u2");
}

Test(mesh_chat, get_messages_filter_role)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u1", "a");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_ASSISTANT, "a1", "b");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u2", "c");

    OcMeshChatMessage out[8];
    size_t n = 0;
    cr_assert_eq(oc_mesh_chat_get_messages(&conv, OC_MESH_CHAT_ROLE_USER,
                                           out, 8, &n), OC_OK);
    cr_assert_eq(n, 2u);
    cr_assert_str_eq(out[0].content, "u1");
    cr_assert_str_eq(out[1].content, "u2");
}

Test(mesh_chat, get_messages_respects_max)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u1", "a");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u2", "b");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u3", "c");

    OcMeshChatMessage out[2];
    size_t n = 99;
    cr_assert_eq(oc_mesh_chat_get_messages(&conv, OC_MESH_CHAT_ROLE__COUNT,
                                           out, 2, &n), OC_OK);
    cr_assert_eq(n, 2u);
}

Test(mesh_chat, get_messages_null)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    OcMeshChatMessage out[1];
    size_t n;
    cr_assert_neq(oc_mesh_chat_get_messages(NULL, OC_MESH_CHAT_ROLE__COUNT,
                                            out, 1, &n), OC_OK);
    cr_assert_neq(oc_mesh_chat_get_messages(&conv, OC_MESH_CHAT_ROLE__COUNT,
                                            NULL, 1, &n), OC_OK);
    cr_assert_neq(oc_mesh_chat_get_messages(&conv, OC_MESH_CHAT_ROLE__COUNT,
                                            out, 1, NULL), OC_OK);
}

/* ----------------------------------------------------------------- */
/* message_count.                                                     */
/* ----------------------------------------------------------------- */

Test(mesh_chat, message_count_all)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u1", "a");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_ASSISTANT, "a1", "b");
    cr_assert_eq(oc_mesh_chat_message_count(&conv, OC_MESH_CHAT_ROLE__COUNT), 2u);
}

Test(mesh_chat, message_count_by_role)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u1", "a");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_ASSISTANT, "a1", "b");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u2", "c");
    cr_assert_eq(oc_mesh_chat_message_count(&conv, OC_MESH_CHAT_ROLE_USER), 2u);
    cr_assert_eq(oc_mesh_chat_message_count(&conv, OC_MESH_CHAT_ROLE_ASSISTANT), 1u);
    cr_assert_eq(oc_mesh_chat_message_count(&conv, OC_MESH_CHAT_ROLE_SYSTEM), 0u);
}

OC_TEST_NULL_SAFE(mesh_chat, message_count_null_conv,
        cr_assert_eq(oc_mesh_chat_message_count(NULL, OC_MESH_CHAT_ROLE__COUNT), 0u);)

/* ----------------------------------------------------------------- */
/* role_name.                                                         */
/* ----------------------------------------------------------------- */

Test(mesh_chat, role_name)
{
    cr_assert_str_eq(oc_mesh_chat_role_name(OC_MESH_CHAT_ROLE_SYSTEM), "system");
    cr_assert_str_eq(oc_mesh_chat_role_name(OC_MESH_CHAT_ROLE_USER), "user");
    cr_assert_str_eq(oc_mesh_chat_role_name(OC_MESH_CHAT_ROLE_ASSISTANT), "assistant");
    cr_assert_str_eq(oc_mesh_chat_role_name(OC_MESH_CHAT_ROLE__COUNT), "unknown");
}

/* ----------------------------------------------------------------- */
/* clear.                                                             */
/* ----------------------------------------------------------------- */

Test(mesh_chat, clear)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "keep-me");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u1", "a");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u2", "b");
    cr_assert_eq(conv.count, 2u);

    cr_assert_eq(oc_mesh_chat_clear(&conv), OC_OK);
    cr_assert_eq(conv.count, 0u);
    cr_assert_eq(conv.next_timestamp, 0u);
    /* conversation_id is preserved. */
    cr_assert_str_eq(conv.conversation_id, "keep-me");

    /* Re-adding after clear restarts timestamps from 0. */
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "u3", "c");
    cr_assert_eq(conv.messages[0].timestamp, 0u);
}

OC_TEST_NULL_SAFE(mesh_chat, clear_null,
        cr_assert_neq(oc_mesh_chat_clear(NULL), OC_OK);)

/* ----------------------------------------------------------------- */
/* serialize.                                                         */
/* ----------------------------------------------------------------- */

Test(mesh_chat, serialize_basic)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "conv-xyz");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER, "hi", "alice");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_ASSISTANT, "hello", "bob");

    char buf[1024];
    cr_assert_eq(oc_mesh_chat_serialize(&conv, buf, sizeof(buf)), OC_OK);
    cr_assert_gt(strlen(buf), 0u);
    cr_assert_not_null(strstr(buf, "conv-xyz"));
    cr_assert_not_null(strstr(buf, "user: hi"));
    cr_assert_not_null(strstr(buf, "assistant: hello"));
}

Test(mesh_chat, serialize_empty)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "empty");
    char buf[256];
    cr_assert_eq(oc_mesh_chat_serialize(&conv, buf, sizeof(buf)), OC_OK);
    cr_assert_not_null(strstr(buf, "empty"));
}

Test(mesh_chat, serialize_null)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c");
    char buf[64];
    cr_assert_neq(oc_mesh_chat_serialize(NULL, buf, sizeof(buf)), OC_OK);
    cr_assert_neq(oc_mesh_chat_serialize(&conv, NULL, 0), OC_OK);
}

Test(mesh_chat, serialize_overflow)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c");
    oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER,
                             "this is a long content message", "a");
    char buf[16];
    cr_assert_neq(oc_mesh_chat_serialize(&conv, buf, sizeof(buf)), OC_OK);
}

/* ----------------------------------------------------------------- */
/* Overflow: max 256 messages.                                        */
/* ----------------------------------------------------------------- */

Test(mesh_chat, overflow_max_messages)
{
    OcMeshChatConversation conv;
    oc_mesh_chat_init(&conv, "c1");
    for (size_t i = 0; i < OC_MESH_CHAT_MAX_MESSAGES; i++) {
        cr_assert_eq(oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER,
                                              "msg", "s"), OC_OK);
    }
    cr_assert_eq(conv.count, OC_MESH_CHAT_MAX_MESSAGES);
    /* Adding one more must fail with OOM. */
    cr_assert_eq(oc_mesh_chat_add_message(&conv, OC_MESH_CHAT_ROLE_USER,
                                          "extra", "s"), OC_ERR_OOM);
    cr_assert_eq(conv.count, OC_MESH_CHAT_MAX_MESSAGES);
}
