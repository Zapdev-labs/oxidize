/*
 * test_qwen.c — Qwen2/Qwen3 config-prefix support tests.
 */
#include <criterion/criterion.h>

#include "oxidize/model.h"

Test(qwen, arch_name_for_config_prefix)
{
    /* oc_llama_load reads general.architecture directly from GGUF metadata */
    const char *enum_name = oc_model_arch_name(OC_ARCH_QWEN);
    cr_assert_str_eq(enum_name, "qwen",
        "enum name is 'qwen' (config MUST use the raw GGUF arch string instead)");
}
