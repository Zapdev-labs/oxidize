/* test_finetune.c — finetuning tests. */
#include "framework.h"
#include "oxidize/finetune.h"
#include <string.h>

Test(ft, strategy_name)
{
    cr_assert_str_eq(oc_ft_strategy_name(OC_FT_SFT), "sft");
    cr_assert_str_eq(oc_ft_strategy_name(OC_FT_SELF_TRAIN), "self-train");
    cr_assert_str_eq(oc_ft_strategy_name(OC_FT_DPO), "dpo");
    cr_assert_str_eq(oc_ft_strategy_name(OC_FT_PPO), "ppo");
}

Test(ft, null_config)
{
    cr_assert_neq(oc_finetune_run(NULL), OC_OK);
}

Test(ft, format_sft)
{
    char buf[1024];
    cr_assert_eq(oc_finetune_format_sft("You are helpful.", "Hello", "Hi there!", buf, sizeof(buf)), OC_OK);
    cr_assert(strstr(buf, "system") != NULL);
    cr_assert(strstr(buf, "You are helpful.") != NULL);
    cr_assert(strstr(buf, "user") != NULL);
    cr_assert(strstr(buf, "Hello") != NULL);
    cr_assert(strstr(buf, "assistant") != NULL);
    cr_assert(strstr(buf, "Hi there!") != NULL);
}

Test(ft, format_sft_null_system)
{
    char buf[512];
    cr_assert_eq(oc_finetune_format_sft(NULL, "Hello", "Hi", buf, sizeof(buf)), OC_OK);
    cr_assert(strstr(buf, "system") == NULL);
    cr_assert(strstr(buf, "user") != NULL);
}

Test(ft, format_sft_small_buf)
{
    char buf[10];
    cr_assert_neq(oc_finetune_format_sft("long system prompt here", "user", "assistant", buf, sizeof(buf)), OC_OK);
}
