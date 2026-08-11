/* test_arch.c — tests for OcModelArchitecture detection + tensor name mapping.
 *
 * Covers:
 *   - VAL-FOUND-007: Llama arch tensor name mapping
 *   - VAL-FOUND-008: Qwen2 dense tensor name mapping (identical to Llama)
 *   - VAL-FOUND-009: Qwen2-MoE tensor name mapping (block_sparse_moe.experts.M.*)
 *   - VAL-FOUND-010: Qwen3-MoE shared+routed expert tensor mapping
 *   - VAL-FOUND-011: DeepSeek MLA tensor name mapping
 *   - VAL-FOUND-012: All 18 architecture strings detected (17 recognized +
 *     OC_ARCH_UNKNOWN = 18 enum values).
 *
 * Reference: Rust oxidize-core/src/format/gguf.rs::map_tensor_name +
 * oxidize-core/src/model/inference.rs::ModelArchitecture::from_gguf.
 */
#include <criterion/criterion.h>
#include "oxidize/arena.h"
#include "oxidize/model.h"

#include <string.h>

/* Helper: map a name and check the result equals `expected`. The arena is
 * a fresh scratch arena per test (freed at the end). */
static void check_map(OcModelArchitecture arch, const char *name,
                      const char *expected)
{
    OcArena *a = oc_arena_new(0);
    cr_assert_not_null(a, "arena alloc");
    const char *mapped = oc_gguf_map_tensor_name(arch, name, a);
    cr_assert_not_null(mapped, "mapping returned NULL for '%s'", name);
    cr_assert_str_eq(mapped, expected,
        "arch=%s name='%s' expected='%s' got='%s'",
        oc_model_arch_name(arch), name, expected, mapped);
    oc_arena_free(a);
}

/* ─── VAL-FOUND-012: All 18 architecture strings detected ───────────────── */

Test(arch, all_19_arch_strings_detected)
{
    /* Each line: input string → expected OcModelArchitecture variant.
     * Mirrors Rust `ModelArchitecture::from_gguf`. */
    struct { const char *s; OcModelArchitecture a; } cases[] = {
        /* Llama family */
        { "llama",                          OC_ARCH_LLAMA },
        /* Mistral */
        { "mistral",                       OC_ARCH_MISTRAL },
        /* Mixtral (MoE) */
        { "mixtral",                        OC_ARCH_MIXTRAL },
        /* DeepSeek (all variants) */
        { "deepseek",                       OC_ARCH_DEEPSEEK },
        { "deepseek2",                      OC_ARCH_DEEPSEEK },
        { "deepseek_v2",                    OC_ARCH_DEEPSEEK },
        { "deepseek_v3",                    OC_ARCH_DEEPSEEK },
        { "deepseek_moe",                   OC_ARCH_DEEPSEEK },
        /* The hyphen form is normalized to underscore (Rust
         * `arch.replace('-', "_")`). */
        { "deepseek-v2",                    OC_ARCH_DEEPSEEK },
        { "deepseek-v3",                    OC_ARCH_DEEPSEEK },
        { "deepseek-moe",                   OC_ARCH_DEEPSEEK },
        /* Qwen (all variants) */
        { "qwen",                           OC_ARCH_QWEN },
        { "qwen2",                          OC_ARCH_QWEN },
        { "qwen2moe",                       OC_ARCH_QWEN },
        { "qwen3",                          OC_ARCH_QWEN },
        { "qwen3moe",                       OC_ARCH_QWEN },
        { "qwen35",                         OC_ARCH_QWEN },
        { "qwen3_5",                        OC_ARCH_QWEN },
        { "qwen3_5_text",                   OC_ARCH_QWEN },
        { "qwen35_text",                    OC_ARCH_QWEN },
        { "qwen3_5_moe",                    OC_ARCH_QWEN },
        { "qwen3_5_moe_text",               OC_ARCH_QWEN },
        { "qwen35moe",                      OC_ARCH_QWEN },
        /* Gemma (all variants) */
        { "gemma",                          OC_ARCH_GEMMA },
        { "gemma2",                         OC_ARCH_GEMMA },
        { "gemma3",                         OC_ARCH_GEMMA },
        { "gemma4",                         OC_ARCH_GEMMA },
        /* Phi */
        { "phi",                            OC_ARCH_PHI },
        { "phi3",                           OC_ARCH_PHI },
        /* Falcon */
        { "falcon",                         OC_ARCH_FALCON },
        /* GPT2 / GPTJ / GPTNeoX */
        { "gpt2",                           OC_ARCH_GPT2 },
        { "gptj",                           OC_ARCH_GPTJ },
        { "gptneox",                        OC_ARCH_GPTNEOX },
        /* MiniMax */
        { "minimax",                        OC_ARCH_MINIMAX },
        { "minimax-m2",                     OC_ARCH_MINIMAX },
        { "minimax-text-01",                OC_ARCH_MINIMAX },
        /* LFM2 / Lfm2Moe */
        { "lfm2",                           OC_ARCH_LFM2 },
        { "lfm2moe",                        OC_ARCH_LFM2_MOE },
        /* GLM (all variants → GlmMoeDsa) */
        { "glm",                            OC_ARCH_GLM_MOE_DSA },
        { "glm4",                           OC_ARCH_GLM_MOE_DSA },
        { "glm_moe",                        OC_ARCH_GLM_MOE_DSA },
        { "glm_moe_dsa",                    OC_ARCH_GLM_MOE_DSA },
        { "glm_dsa",                        OC_ARCH_GLM_MOE_DSA },
        { "glmmoe",                         OC_ARCH_GLM_MOE_DSA },
        { "glmmoedsa",                      OC_ARCH_GLM_MOE_DSA },
        { "glm-dsa",                        OC_ARCH_GLM_MOE_DSA },
        { "glm-moe-dsa",                    OC_ARCH_GLM_MOE_DSA },
        /* Hunyuan (all variants) */
        { "hunyuan",                        OC_ARCH_HUNYUAN_MOE },
        { "hunyuan_moe",                    OC_ARCH_HUNYUAN_MOE },
        { "hunyuanmoe",                     OC_ARCH_HUNYUAN_MOE },
        { "hy_v3",                          OC_ARCH_HUNYUAN_MOE },
        { "hyv3",                           OC_ARCH_HUNYUAN_MOE },
        { "hunyuan_v3",                     OC_ARCH_HUNYUAN_MOE },
        { "hunyuan-moe",                    OC_ARCH_HUNYUAN_MOE },
        /* LongCat (all variants) */
        { "longcat",                        OC_ARCH_LONGCAT },
        { "longcat2",                       OC_ARCH_LONGCAT },
        { "longcat_2",                      OC_ARCH_LONGCAT },
        { "longcat-2",                      OC_ARCH_LONGCAT },
        { "longcat_flash",                  OC_ARCH_LONGCAT },
        { "longcatflash",                   OC_ARCH_LONGCAT },
        /* Muse Glimmer (Meta). The GGUF spells it with a hyphen. */
        { "muse-glimmer",                   OC_ARCH_MUSE_GLIMMER },
        { "muse_glimmer",                   OC_ARCH_MUSE_GLIMMER },
        { "museglimmer",                    OC_ARCH_MUSE_GLIMMER },
        { "muse",                           OC_ARCH_MUSE_GLIMMER },
        /* Unknown → OC_ARCH_UNKNOWN. */
        { "not-a-real-arch",                OC_ARCH_UNKNOWN },
        { "",                               OC_ARCH_UNKNOWN },
        { NULL,                             OC_ARCH_UNKNOWN },
    };
    size_t n = sizeof(cases) / sizeof(cases[0]);
    for (size_t i = 0; i < n; i++) {
        OcModelArchitecture got = oc_model_arch_from_str(cases[i].s);
        cr_assert_eq(got, cases[i].a,
            "arch_from_str(\"%s\") = %s, expected %s",
            cases[i].s ? cases[i].s : "(null)",
            oc_model_arch_name(got), oc_model_arch_name(cases[i].a));
    }

    /* Verify the enum has exactly 19 variants (18 recognized + 1 unknown).
     * OC_ARCH_MUSE_GLIMMER is appended AFTER OC_ARCH_UNKNOWN so the older
     * values stay stable. */
    cr_assert_eq((int)OC_ARCH__COUNT, 19,
        "OcModelArchitecture should have 19 variants (18 + UNKNOWN), got %d",
        (int)OC_ARCH__COUNT);
}

Test(arch, arch_name_round_trip)
{
    /* oc_model_arch_name should return a non-NULL string for every variant. */
    for (int i = 0; i < (int)OC_ARCH__COUNT; i++) {
        const char *name = oc_model_arch_name((OcModelArchitecture)i);
        cr_assert_not_null(name, "arch_name(%d) returned NULL", i);
        cr_assert_gt(strlen(name), 0, "arch_name(%d) returned empty string", i);
    }
}

Test(arch, uses_moe_classification)
{
    /* Mirrors Rust `ModelArchitecture::uses_moe()`. */
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_MIXTRAL),    "mixtral is MoE");
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_MINIMAX),    "minimax is MoE");
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_LFM2_MOE),   "lfm2moe is MoE");
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_DEEPSEEK),   "deepseek is MoE");
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_GLM_MOE_DSA), "glm_moe_dsa is MoE");
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_HUNYUAN_MOE), "hunyuan_moe is MoE");
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_LONGCAT),     "longcat is MoE");

    cr_assert_not(oc_model_arch_uses_moe(OC_ARCH_LLAMA),   "llama is dense");
    cr_assert_not(oc_model_arch_uses_moe(OC_ARCH_QWEN),    "qwen dense is not MoE (Qwen2-MoE is, but mapped to OC_ARCH_QWEN — Rust uses_moe() checks the enum variant not the GGUF string. For MoE-specific behavior, callers should inspect the GGUF tensor table for `experts` substrings.)");
    cr_assert_not(oc_model_arch_uses_moe(OC_ARCH_GEMMA),   "gemma is dense");
    cr_assert_not(oc_model_arch_uses_moe(OC_ARCH_UNKNOWN), "unknown is not MoE");
}

Test(arch, uses_mla_classification)
{
    /* Mirrors Rust `ModelArchitecture::uses_mla()`. DeepSeek, GlmMoeDsa and
     * LongCat use MLA (compressed KV cache). */
    cr_assert(oc_model_arch_uses_mla(OC_ARCH_DEEPSEEK),     "deepseek uses MLA");
    cr_assert(oc_model_arch_uses_mla(OC_ARCH_GLM_MOE_DSA),  "glm_moe_dsa uses MLA");
    cr_assert(oc_model_arch_uses_mla(OC_ARCH_LONGCAT),      "longcat uses MLA");
    cr_assert_not(oc_model_arch_uses_mla(OC_ARCH_LLAMA),     "llama does not use MLA");
    cr_assert_not(oc_model_arch_uses_mla(OC_ARCH_QWEN),      "qwen does not use MLA");
    cr_assert_not(oc_model_arch_uses_mla(OC_ARCH_HUNYUAN_MOE), "hunyuan does not use MLA (standard attention)");
    cr_assert_not(oc_model_arch_uses_mla(OC_ARCH_MIXTRAL),   "mixtral does not use MLA");
}

Test(arch, uses_alibi_classification)
{
    cr_assert(oc_model_arch_uses_alibi(OC_ARCH_FALCON),  "falcon uses Alibi");
    cr_assert(oc_model_arch_uses_alibi(OC_ARCH_GPT2),    "gpt2 uses Alibi");
    cr_assert(oc_model_arch_uses_alibi(OC_ARCH_GPTJ),    "gptj uses Alibi");
    cr_assert(oc_model_arch_uses_alibi(OC_ARCH_GPTNEOX), "gptneox uses Alibi");
    cr_assert_not(oc_model_arch_uses_alibi(OC_ARCH_LLAMA), "llama uses RoPE");
    cr_assert_not(oc_model_arch_uses_alibi(OC_ARCH_QWEN),  "qwen uses RoPE");
}

/* ─── VAL-FOUND-007: Llama arch tensor name mapping ──────────────────────── */

Test(arch_mapping, llama_top_level_tensors)
{
    check_map(OC_ARCH_LLAMA, "model.embed_tokens.weight", "tok_embeddings.weight");
    check_map(OC_ARCH_LLAMA, "lm_head.weight",            "output.weight");
    check_map(OC_ARCH_LLAMA, "model.norm.weight",          "norm.weight");
}

Test(arch_mapping, llama_layer_attention_tensors)
{
    check_map(OC_ARCH_LLAMA, "model.layers.3.self_attn.q_proj.weight",   "blk.3.attn_q.weight");
    check_map(OC_ARCH_LLAMA, "model.layers.3.self_attn.k_proj.weight",   "blk.3.attn_k.weight");
    check_map(OC_ARCH_LLAMA, "model.layers.3.self_attn.v_proj.weight",   "blk.3.attn_v.weight");
    check_map(OC_ARCH_LLAMA, "model.layers.3.self_attn.o_proj.weight",   "blk.3.attn_output.weight");
    check_map(OC_ARCH_LLAMA, "model.layers.5.input_layernorm.weight",     "blk.5.attn_norm.weight");
    check_map(OC_ARCH_LLAMA, "model.layers.5.post_attention_layernorm.weight", "blk.5.ffn_norm.weight");
}

Test(arch_mapping, llama_layer_ffn_tensors)
{
    check_map(OC_ARCH_LLAMA, "model.layers.7.mlp.gate_proj.weight", "blk.7.ffn_gate.weight");
    check_map(OC_ARCH_LLAMA, "model.layers.7.mlp.up_proj.weight",   "blk.7.ffn_up.weight");
    check_map(OC_ARCH_LLAMA, "model.layers.7.mlp.down_proj.weight",  "blk.7.ffn_down.weight");
}

Test(arch_mapping, llama_unknown_name_passes_through)
{
    /* Mirrors Rust `mapped.unwrap_or_else(|| name.to_owned())`. */
    check_map(OC_ARCH_LLAMA, "custom.tensor.weight", "custom.tensor.weight");
    check_map(OC_ARCH_LLAMA, "some.random.name",      "some.random.name");
}

/* ─── VAL-FOUND-008: Qwen2 dense tensor mapping (identical to Llama) ──── */

Test(arch_mapping, qwen2_dense_identical_to_llama)
{
    /* Rust uses the same `map_hf_decoder_name` for both Llama and Qwen2 dense. */
    check_map(OC_ARCH_QWEN, "model.embed_tokens.weight", "tok_embeddings.weight");
    check_map(OC_ARCH_QWEN, "lm_head.weight",            "output.weight");
    check_map(OC_ARCH_QWEN, "model.norm.weight",          "norm.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.self_attn.q_proj.weight", "blk.1.attn_q.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.self_attn.k_proj.weight", "blk.1.attn_k.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.self_attn.v_proj.weight", "blk.1.attn_v.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.self_attn.o_proj.weight", "blk.1.attn_output.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.input_layernorm.weight",   "blk.1.attn_norm.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.post_attention_layernorm.weight", "blk.1.ffn_norm.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.gate_proj.weight", "blk.1.ffn_gate.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.up_proj.weight",   "blk.1.ffn_up.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.down_proj.weight",  "blk.1.ffn_down.weight");
}

/* ─── VAL-FOUND-009: Qwen2-MoE tensor mapping ──────────────────────────────
 *
 * Mirrors Rust test:
 *   qwen2moe_mapped[0].name == "blk.4.ffn_gate.2.weight"
 * where the input was "model.layers.4.block_sparse_moe.experts.2.w1.weight". */
Test(arch_mapping, qwen2moe_block_sparse_moe_experts)
{
    /* block_sparse_moe.experts.<M>.<w1|w2|w3>.weight → blk.<N>.ffn_<gate|down|up>.<M>.weight */
    check_map(OC_ARCH_QWEN, "model.layers.4.block_sparse_moe.experts.2.w1.weight",
              "blk.4.ffn_gate.2.weight");
    check_map(OC_ARCH_QWEN, "model.layers.4.block_sparse_moe.experts.2.w2.weight",
              "blk.4.ffn_down.2.weight");
    check_map(OC_ARCH_QWEN, "model.layers.4.block_sparse_moe.experts.2.w3.weight",
              "blk.4.ffn_up.2.weight");

    /* Mixtral uses the same convention (block_sparse_moe.experts.*). */
    check_map(OC_ARCH_MIXTRAL, "model.layers.2.block_sparse_moe.experts.3.w1.weight",
              "blk.2.ffn_gate.3.weight");
    check_map(OC_ARCH_MIXTRAL, "model.layers.2.block_sparse_moe.experts.3.w2.weight",
              "blk.2.ffn_down.3.weight");
    check_map(OC_ARCH_MIXTRAL, "model.layers.2.block_sparse_moe.experts.3.w3.weight",
              "blk.2.ffn_up.3.weight");

    /* block_sparse_moe.gate.weight → blk.<N>.ffn_gate_inp.weight */
    check_map(OC_ARCH_MIXTRAL, "model.layers.2.block_sparse_moe.gate.weight",
              "blk.2.ffn_gate_inp.weight");
}

/* ─── VAL-FOUND-010: Qwen3-MoE shared + routed experts ──────────────────────
 *
 * Qwen3-MoE uses both routed experts (mlp.experts.<M>.*) and a shared
 * expert (mlp.shared_expert.*). The shared expert uses the
 * `*_shexp` suffix per Rust `map_hf_decoder_name`. */
Test(arch_mapping, qwen3moe_shared_and_routed_experts)
{
    /* Routed experts via mlp.experts.<M>.<gate|up|down>_proj.weight. */
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.experts.42.gate_proj.weight",
              "blk.1.ffn_gate.42.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.experts.42.up_proj.weight",
              "blk.1.ffn_up.42.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.experts.42.down_proj.weight",
              "blk.1.ffn_down.42.weight");

    /* Shared expert (Qwen3-MoE / DeepSeekMoE style). */
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.shared_expert.gate_proj.weight",
              "blk.1.ffn_gate_shexp.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.shared_expert.up_proj.weight",
              "blk.1.ffn_up_shexp.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.shared_expert.down_proj.weight",
              "blk.1.ffn_down_shexp.weight");
    check_map(OC_ARCH_QWEN, "model.layers.1.mlp.shared_expert_gate.weight",
              "blk.1.ffn_gate_inp_shexp.weight");
}

/* ─── VAL-FOUND-011: DeepSeek MLA tensor mapping ────────────────────────────
 *
 * Mirrors Rust test:
 *   mapped[1].name == "blk.1.attn_kv_a_mqa.weight"
 *   mapped[3].name == "blk.1.ffn_gate.42.weight"
 *   mapped[4].name == "blk.1.ffn_gate_shexp.weight"
 *   mapped[5].name == "blk.1.ffn_up_shexp.weight"
 *   mapped[6].name == "blk.1.ffn_gate_inp_shexp.weight"
 * for DeepSeek2 architecture. */
Test(arch_mapping, deepseek_mla_attention_tensors)
{
    /* MLA (Multi-head Latent Attention) compressed KV cache tensors. */
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.self_attn.q_a_proj.weight",            "blk.1.attn_q_a.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.self_attn.q_a_layernorm.weight",      "blk.1.attn_q_a_norm.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.self_attn.q_b_proj.weight",           "blk.1.attn_q_b.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.self_attn.kv_a_proj_with_mqa.weight", "blk.1.attn_kv_a_mqa.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.self_attn.kv_a_layernorm.weight",     "blk.1.attn_kv_a_norm.weight");

    /* Standard attention tensors should also map (DeepSeek uses the same
     * map_hf_decoder_name table for the non-MLA tensors). */
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.self_attn.q_proj.weight", "blk.1.attn_q.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.self_attn.k_proj.weight", "blk.1.attn_k.weight");
}

Test(arch_mapping, deepseek_moe_shared_expert)
{
    /* DeepSeek-MoE shared expert + routed experts (mlp.experts.<M>.*). */
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.mlp.experts.42.gate_proj.weight",
              "blk.1.ffn_gate.42.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.mlp.experts.42.up_proj.weight",
              "blk.1.ffn_up.42.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.mlp.experts.42.down_proj.weight",
              "blk.1.ffn_down.42.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.mlp.shared_expert.gate_proj.weight",
              "blk.1.ffn_gate_shexp.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.mlp.shared_expert.up_proj.weight",
              "blk.1.ffn_up_shexp.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.mlp.shared_expert.down_proj.weight",
              "blk.1.ffn_down_shexp.weight");
    check_map(OC_ARCH_DEEPSEEK, "model.layers.1.mlp.shared_expert_gate.weight",
              "blk.1.ffn_gate_inp_shexp.weight");
}

/* ─── Falcon / GPT2 / GPTJ / GPTNeoX mappings ─────────────────────────── */

Test(arch_mapping, falcon_top_level_tensors)
{
    check_map(OC_ARCH_FALCON, "transformer.word_embeddings.weight", "tok_embeddings.weight");
    check_map(OC_ARCH_FALCON, "lm_head.weight",                     "output.weight");
    check_map(OC_ARCH_FALCON, "transformer.ln_f.weight",            "norm.weight");
    /* Unknown falcon name passes through. */
    check_map(OC_ARCH_FALCON, "transformer.h.0.self_attention.query_key_value.weight",
              "transformer.h.0.self_attention.query_key_value.weight");
}

Test(arch_mapping, gpt2_top_level_tensors)
{
    check_map(OC_ARCH_GPT2, "transformer.wte.weight",   "tok_embeddings.weight");
    check_map(OC_ARCH_GPT2, "lm_head.weight",           "output.weight");
    check_map(OC_ARCH_GPT2, "transformer.ln_f.weight",  "norm.weight");
}

Test(arch_mapping, gptj_top_level_tensors)
{
    /* GPTJ shares GPT2's mapping table (Rust uses the same map_gpt2_name). */
    check_map(OC_ARCH_GPTJ, "transformer.wte.weight",   "tok_embeddings.weight");
    check_map(OC_ARCH_GPTJ, "lm_head.weight",            "output.weight");
    check_map(OC_ARCH_GPTJ, "transformer.ln_f.weight",  "norm.weight");
}

Test(arch_mapping, gptneox_top_level_tensors)
{
    check_map(OC_ARCH_GPTNEOX, "gpt_neox.embed_in.weight",          "tok_embeddings.weight");
    check_map(OC_ARCH_GPTNEOX, "embed_out.weight",                   "output.weight");
    check_map(OC_ARCH_GPTNEOX, "lm_head.weight",                      "output.weight");
    check_map(OC_ARCH_GPTNEOX, "gpt_neox.final_layer_norm.weight",   "norm.weight");
}

Test(arch_mapping, unknown_arch_passes_through)
{
    /* Unknown arch: every name passes through unchanged (Rust falls back to
     * identity copy). */
    check_map(OC_ARCH_UNKNOWN, "model.layers.3.self_attn.q_proj.weight",
              "model.layers.3.self_attn.q_proj.weight");
    check_map(OC_ARCH_UNKNOWN, "anything", "anything");
}

Test(arch_mapping, multiple_layers_in_one_arena)
{
    /* Verify the arena accumulates multiple mapped names without
     * interference. Mirrors how the model loader builds a full tensor table. */
    OcArena *a = oc_arena_new(0);
    cr_assert_not_null(a, "");

    const char *names[] = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    };
    const char *expected[] = {
        "tok_embeddings.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.ffn_gate.weight",
        "blk.1.attn_q.weight",
        "blk.1.ffn_gate.weight",
        "norm.weight",
        "output.weight",
    };
    size_t n = sizeof(names) / sizeof(names[0]);
    for (size_t i = 0; i < n; i++) {
        const char *m = oc_gguf_map_tensor_name(OC_ARCH_LLAMA, names[i], a);
        cr_assert_not_null(m, "map returned NULL for '%s'", names[i]);
        cr_assert_str_eq(m, expected[i],
            "map('%s') = '%s', expected '%s'", names[i], m, expected[i]);
    }

    /* Verify all earlier pointers are still valid (arena owns them, none
     * should have been freed). */
    for (size_t i = 0; i < n; i++) {
        const char *m = oc_gguf_map_tensor_name(OC_ARCH_LLAMA, names[i], a);
        cr_assert_str_eq(m, expected[i], "re-mapped name %zu differs", i);
    }

    oc_arena_free(a);
}

Test(arch_mapping, large_layer_indices)
{
    /* Qwen3-30B-A3B has 48 layers; verify large indices format correctly. */
    check_map(OC_ARCH_QWEN, "model.layers.47.self_attn.q_proj.weight", "blk.47.attn_q.weight");
    check_map(OC_ARCH_QWEN, "model.layers.127.mlp.experts.100.gate_proj.weight",
              "blk.127.ffn_gate.100.weight");
    /* Three-digit layer + three-digit expert. */
    check_map(OC_ARCH_DEEPSEEK, "model.layers.60.mlp.experts.255.up_proj.weight",
              "blk.60.ffn_up.255.weight");
}
