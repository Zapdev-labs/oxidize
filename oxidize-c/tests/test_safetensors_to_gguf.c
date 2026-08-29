/* test_safetensors_to_gguf.c — SafeTensors to GGUF conversion tests. */
#include "framework.h"
#include "oxidize/safetensors_to_gguf.h"
#include <string.h>

OC_TEST_NULL_SAFE(st_to_gguf, null_config,
        cr_assert_neq(oc_safetensors_to_gguf(NULL), OC_OK);)

Test(st_to_gguf, null_paths)
{
    OcConvertConfig cfg = {0};
    cr_assert_neq(oc_safetensors_to_gguf(&cfg), OC_OK);
}

Test(st_to_gguf, detect_arch_llama)
{
    const char *names[] = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight"
    };
    cr_assert_str_eq(oc_detect_arch_from_tensors(names, 5), "llama");
}

Test(st_to_gguf, detect_arch_qwen)
{
    const char *names[] = {"qwen_model.weight"};
    cr_assert_str_eq(oc_detect_arch_from_tensors(names, 1), "qwen2");
}

Test(st_to_gguf, detect_arch_gemma)
{
    const char *names[] = {"gemma_model.weight"};
    cr_assert_str_eq(oc_detect_arch_from_tensors(names, 1), "gemma");
}

Test(st_to_gguf, detect_arch_default)
{
    const char *names[] = {"some_unknown_tensor"};
    cr_assert_str_eq(oc_detect_arch_from_tensors(names, 1), "unknown");
}

Test(st_to_gguf, map_tensor_name_embed)
{
    cr_assert_str_eq(oc_map_tensor_name("model.embed_tokens.weight", "llama"),
                     "token_embd.weight");
}

Test(st_to_gguf, map_tensor_name_lm_head)
{
    cr_assert_str_eq(oc_map_tensor_name("lm_head.weight", "llama"),
                     "output.weight");
}

Test(st_to_gguf, map_tensor_name_norm)
{
    cr_assert_str_eq(oc_map_tensor_name("model.norm.weight", "llama"),
                     "output_norm.weight");
}

Test(st_to_gguf, map_tensor_name_q_proj)
{
    const char *mapped = oc_map_tensor_name(
        "model.layers.5.self_attn.q_proj.weight", "llama");
    cr_assert_str_eq(mapped, "blk.5.attn_q.weight");
}

Test(st_to_gguf, map_tensor_name_mlp_gate)
{
    const char *mapped = oc_map_tensor_name(
        "model.layers.3.mlp.gate_proj.weight", "llama");
    cr_assert_str_eq(mapped, "blk.3.ffn_gate.weight");
}

Test(st_to_gguf, map_tensor_name_mlp_down)
{
    const char *mapped = oc_map_tensor_name(
        "model.layers.10.mlp.down_proj.weight", "llama");
    cr_assert_str_eq(mapped, "blk.10.ffn_down.weight");
}

Test(st_to_gguf, map_tensor_name_unmapped)
{
    const char *mapped = oc_map_tensor_name("custom_tensor.name", "llama");
    cr_assert_str_eq(mapped, "custom_tensor.name");
}

Test(st_to_gguf, map_tensor_name_mla)
{
    const char *mapped = oc_map_tensor_name(
        "model.layers.0.self_attn.q_a_proj.weight", "deepseek2");
    cr_assert_str_eq(mapped, "blk.0.attn_q_a.weight");
}
