/*
 * test_chat.c — chat template formatting tests.
 */
#include <criterion/criterion.h>

#include "oxidize/chat.h"

#include <string.h>

Test(chat, chatml_user_message)
{
    char out[1024];
    size_t n = oc_chat_render_message(OC_CHAT_CHATML, "user", "Hello",
                                      out, sizeof(out), true, false);
    cr_assert_gt(n, 0);
    cr_assert_str_eq(out, "<|im_start|>user\nHello<|im_end|>\n");
}

Test(chat, chatml_last_message_adds_assistant_prompt)
{
    char out[1024];
    size_t n = oc_chat_render_message(OC_CHAT_CHATML, "user", "Hi",
                                      out, sizeof(out), true, true);
    cr_assert_gt(n, 0);
    cr_assert(strstr(out, "<|im_start|>assistant\n") != NULL,
              "last message should add assistant prompt");
}

Test(chat, llama3_format)
{
    char out[1024];
    size_t n = oc_chat_render_message(OC_CHAT_LLAMA3, "user", "Hello",
                                      out, sizeof(out), true, false);
    cr_assert_gt(n, 0);
    cr_assert(strstr(out, "<|start_header_id|>user<|end_header_id|>") != NULL);
    cr_assert(strstr(out, "Hello") != NULL);
    cr_assert(strstr(out, "<|eot_id|>") != NULL);
}

Test(chat, llama2_format)
{
    char out[1024];
    size_t n = oc_chat_render_message(OC_CHAT_LLAMA2, "user", "Hello",
                                      out, sizeof(out), true, true);
    cr_assert_gt(n, 0);
    cr_assert(strstr(out, "[INST] Hello [/INST]") != NULL);
}

Test(chat, gemma_format)
{
    char out[1024];
    size_t n = oc_chat_render_message(OC_CHAT_GEMMA, "user", "Hello",
                                      out, sizeof(out), true, true);
    cr_assert_gt(n, 0);
    cr_assert(strstr(out, "<start_of_turn>user\nHello<end_of_turn>") != NULL);
    cr_assert(strstr(out, "<start_of_turn>model\n") != NULL,
              "last message should add model prompt");
}

Test(chat, render_multiple_messages)
{
    const char *roles[] = {"system", "user"};
    const char *contents[] = {"You are helpful.", "Hello"};
    char out[4096];
    size_t n = oc_chat_render_messages(OC_CHAT_CHATML, roles, contents, 2,
                                      out, sizeof(out));
    cr_assert_gt(n, 0);
    cr_assert(strstr(out, "<|im_start|>system\nYou are helpful.<|im_end|>") != NULL);
    cr_assert(strstr(out, "<|im_start|>user\nHello<|im_end|>") != NULL);
    cr_assert(strstr(out, "<|im_start|>assistant\n") != NULL);
}

Test(chat, auto_detect)
{
    cr_assert_eq(oc_chat_detect("qwen2"), OC_CHAT_CHATML);
    cr_assert_eq(oc_chat_detect("mistral"), OC_CHAT_CHATML);
    cr_assert_eq(oc_chat_detect("gemma2"), OC_CHAT_GEMMA);
    cr_assert_eq(oc_chat_detect("llama"), OC_CHAT_LLAMA3);
    cr_assert_eq(oc_chat_detect(NULL), OC_CHAT_CHATML);
}

Test(chat, render_message_reports_overflow)
{
    char out[16];
    cr_assert_eq(oc_chat_render_message(OC_CHAT_CHATML, "user",
                                        "this does not fit", out, sizeof(out),
                                        true, true), 0);
}
