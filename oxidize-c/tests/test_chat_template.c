/* test_chat_template.c — Chat template tests. */
#include "framework.h"
#include "oxidize/chat_template.h"
#include <string.h>

Test(tpl, init_chatml)
{
    OcChatTemplate tpl;
    cr_assert_eq(oc_chat_tpl_init(&tpl, OC_TPL_CHATML), OC_OK);
    cr_assert_str_eq(tpl.bos_token, "<|im_start|>");
    cr_assert_str_eq(tpl.eos_token, "<|im_end|>");
}

Test(tpl, init_llama2)
{
    OcChatTemplate tpl;
    cr_assert_eq(oc_chat_tpl_init(&tpl, OC_TPL_LLAMA2), OC_OK);
    cr_assert_str_eq(tpl.bos_token, "<s>");
    cr_assert_str_eq(tpl.eos_token, "</s>");
}

OC_TEST_NULL_SAFE(tpl, init_null,
        cr_assert_neq(oc_chat_tpl_init(NULL, OC_TPL_CHATML), OC_OK);)

Test(tpl, add_message)
{
    OcChatMessage msgs[10];
    size_t n = 0;
    cr_assert_eq(oc_chat_tpl_add_message(msgs, &n, OC_ROLE_USER, "hello"), OC_OK);
    cr_assert_eq(n, 1);
    cr_assert_str_eq(msgs[0].content, "hello");
}

OC_TEST_NULL_SAFE(tpl, add_message_null,
        cr_assert_neq(oc_chat_tpl_add_message(NULL, NULL, OC_ROLE_USER, "x"), OC_OK);)

Test(tpl, render_chatml)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_CHATML);
    OcChatMessage msgs[2];
    size_t n = 0;
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_USER, "hello");
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_ASSISTANT, "hi there");
    char out[4096];
    cr_assert_eq(oc_chat_tpl_render(&tpl, msgs, n, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "<|im_start|>user") != NULL);
    cr_assert(strstr(out, "hello") != NULL);
    cr_assert(strstr(out, "<|im_start|>assistant") != NULL);
}

Test(tpl, render_llama2)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_LLAMA2);
    OcChatMessage msgs[1];
    size_t n = 0;
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_USER, "hello");
    char out[4096];
    cr_assert_eq(oc_chat_tpl_render(&tpl, msgs, n, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "[INST]") != NULL);
    cr_assert(strstr(out, "hello") != NULL);
}

Test(tpl, render_mistral)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_MISTRAL);
    OcChatMessage msgs[1];
    size_t n = 0;
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_USER, "test");
    char out[4096];
    cr_assert_eq(oc_chat_tpl_render(&tpl, msgs, n, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "test") != NULL);
}

Test(tpl, render_alpaca)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_ALPACA);
    OcChatMessage msgs[2];
    size_t n = 0;
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_USER, "do something");
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_ASSISTANT, "done");
    char out[4096];
    cr_assert_eq(oc_chat_tpl_render(&tpl, msgs, n, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "### Instruction:") != NULL);
    cr_assert(strstr(out, "### Response:") != NULL);
}

Test(tpl, render_vicuna)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_VICUNA);
    OcChatMessage msgs[2];
    size_t n = 0;
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_USER, "hi");
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_ASSISTANT, "hello");
    char out[4096];
    cr_assert_eq(oc_chat_tpl_render(&tpl, msgs, n, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "USER:") != NULL);
    cr_assert(strstr(out, "ASSISTANT:") != NULL);
}

Test(tpl, render_system)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_CHATML);
    OcChatMessage msgs[2];
    size_t n = 0;
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_SYSTEM, "you are helpful");
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_USER, "hello");
    char out[4096];
    cr_assert_eq(oc_chat_tpl_render(&tpl, msgs, n, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "you are helpful") != NULL);
}

Test(tpl, type_name)
{
    cr_assert_str_eq(oc_chat_tpl_type_name(OC_TPL_CHATML), "chatml");
    cr_assert_str_eq(oc_chat_tpl_type_name(OC_TPL_LLAMA2), "llama2");
    cr_assert_str_eq(oc_chat_tpl_type_name(OC_TPL_LLAMA3), "llama3");
    cr_assert_str_eq(oc_chat_tpl_type_name(OC_TPL_MISTRAL), "mistral");
}

Test(tpl, role_name)
{
    cr_assert_str_eq(oc_chat_role_name(OC_ROLE_SYSTEM), "system");
    cr_assert_str_eq(oc_chat_role_name(OC_ROLE_USER), "user");
    cr_assert_str_eq(oc_chat_role_name(OC_ROLE_ASSISTANT), "assistant");
}

Test(tpl, parse_type)
{
    cr_assert_eq(oc_chat_tpl_parse_type("chatml"), OC_TPL_CHATML);
    cr_assert_eq(oc_chat_tpl_parse_type("llama2"), OC_TPL_LLAMA2);
    cr_assert_eq(oc_chat_tpl_parse_type("mistral"), OC_TPL_MISTRAL);
    cr_assert_eq(oc_chat_tpl_parse_type("unknown"), OC_TPL_CHATML);
    cr_assert_eq(oc_chat_tpl_parse_type(NULL), OC_TPL_CHATML);
}

Test(tpl, gen_prompt_chatml)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_CHATML);
    char out[256];
    cr_assert_eq(oc_chat_tpl_add_generation_prompt(&tpl, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "assistant") != NULL);
}

Test(tpl, gen_prompt_alpaca)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_ALPACA);
    char out[256];
    cr_assert_eq(oc_chat_tpl_add_generation_prompt(&tpl, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "### Response:") != NULL);
}

OC_TEST_NULL_SAFE(tpl, render_null,
        cr_assert_neq(oc_chat_tpl_render(NULL, NULL, 0, NULL, 0), OC_OK);)

Test(tpl, render_empty)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_CHATML);
    char out[256];
    cr_assert_eq(oc_chat_tpl_render(&tpl, NULL, 0, out, sizeof(out)), OC_OK);
    /* Should render just the assistant prompt. */
    cr_assert(strlen(out) > 0);
}

Test(tpl, render_llama3)
{
    OcChatTemplate tpl;
    oc_chat_tpl_init(&tpl, OC_TPL_LLAMA3);
    OcChatMessage msgs[1];
    size_t n = 0;
    oc_chat_tpl_add_message(msgs, &n, OC_ROLE_USER, "test");
    char out[4096];
    cr_assert_eq(oc_chat_tpl_render(&tpl, msgs, n, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "user") != NULL);
    cr_assert(strstr(out, "test") != NULL);
}
