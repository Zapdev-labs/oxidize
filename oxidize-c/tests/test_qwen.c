/*
 * test_qwen.c — Qwen2/Qwen3 dense forward-path support tests.
 *
 * Qwen2 and Qwen3-dense use the SAME architecture as Llama (RMSNorm + GQA
 * + RoPE + SwiGLU), so oc_llama_forward handles them without arch-specific
 * code. This file asserts the architectural-detection and config-prefix
 * plumbing that makes a Qwen2 GGUF load correctly:
 *   - oc_model_arch_from_str("qwen2") == OC_ARCH_QWEN
 *   - the model is recognized as dense (not MoE, not MLA)
 *   - tensor-name mapping covers Qwen2's HF self_attn names
 *
 * Qwen3-MoE (shared + routed experts) is a SEPARATE feature (cpu-qwen3-moe);
 * it is NOT covered here.
 */
#include <criterion/criterion.h>

#include "oxidize/model.h"

#include "oxidize/arena.h"
#include <string.h>

Test(qwen, arch_from_str_recognizes_qwen2)
{
    cr_assert_eq(oc_model_arch_from_str("qwen2"), OC_ARCH_QWEN,
                 "qwen2 must map to OC_ARCH_QWEN");
    cr_assert_eq(oc_model_arch_from_str("qwen3"), OC_ARCH_QWEN,
                 "qwen3 also maps to OC_ARCH_QWEN (same enum slot)");
}

Test(qwen, qwen_is_dense_not_moe_not_mla)
{
    /* Qwen2/3 dense: no MoE, no MLA, no Alibi. The shared Llama forward path
     * applies. */
    cr_assert_not(oc_model_arch_uses_moe(OC_ARCH_QWEN),
                  "Qwen dense must not report MoE");
    cr_assert_not(oc_model_arch_uses_mla(OC_ARCH_QWEN),
                  "Qwen must not report MLA");
    cr_assert_not(oc_model_arch_uses_alibi(OC_ARCH_QWEN),
                  "Qwen uses RoPE, not Alibi");
}

Test(qwen, tensor_name_mapping_covers_hf_self_attn)
{
    /* Qwen2 HF tensor names follow the standard model.layers.N.* pattern.
     * The shared HF-decoder mapping must canonicalize them to blk.N.* so
     * oc_llama_load's assign_tensor() finds them. */
    OcArena *arena = oc_arena_new(4096);
    cr_assert_not_null(arena);

    const char *mapped = oc_gguf_map_tensor_name(OC_ARCH_QWEN,
        "model.layers.5.self_attn.q_proj.weight", arena);
    cr_assert_str_eq(mapped, "blk.5.attn_q.weight", "q_proj mapping");

    mapped = oc_gguf_map_tensor_name(OC_ARCH_QWEN,
        "model.layers.5.self_attn.k_proj.weight", arena);
    cr_assert_str_eq(mapped, "blk.5.attn_k.weight", "k_proj mapping");

    mapped = oc_gguf_map_tensor_name(OC_ARCH_QWEN,
        "model.layers.5.mlp.gate_proj.weight", arena);
    cr_assert_str_eq(mapped, "blk.5.ffn_gate.weight", "gate_proj mapping");

    mapped = oc_gguf_map_tensor_name(OC_ARCH_QWEN,
        "model.layers.5.mlp.down_proj.weight", arena);
    cr_assert_str_eq(mapped, "blk.5.ffn_down.weight", "down_proj mapping");

    mapped = oc_gguf_map_tensor_name(OC_ARCH_QWEN,
        "model.embed_tokens.weight", arena);
    cr_assert_str_eq(mapped, "tok_embeddings.weight", "embed mapping");

    oc_arena_free(arena);
}

Test(qwen, arch_name_for_config_prefix)
{
    /* oc_llama_load reads general.architecture directly from GGUF metadata
     * (not oc_model_arch_name) so the prefix is "qwen2." matching the on-disk
     * keys. This test documents that contract: the enum name is "qwen" but
     * the GGUF key prefix is the raw "qwen2" string. */
    const char *enum_name = oc_model_arch_name(OC_ARCH_QWEN);
    cr_assert_str_eq(enum_name, "qwen",
        "enum name is 'qwen' (config MUST use the raw GGUF arch string instead)");
}
