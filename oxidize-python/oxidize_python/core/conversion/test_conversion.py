from oxidize_python.core.conversion.conversion import map_hf_tensor_name


def test_map_hf_tensor_name_weights():
    assert map_hf_tensor_name("model.embed_tokens.weight") == "token_embd.weight"
    assert map_hf_tensor_name("model.norm.weight") == "output_norm.weight"
    assert map_hf_tensor_name("lm_head.weight") == "output.weight"
    assert (
        map_hf_tensor_name("model.layers.0.self_attn.q_proj.weight")
        == "blk.0.attn_q.weight"
    )
    assert (
        map_hf_tensor_name("model.layers.1.mlp.gate_proj.weight")
        == "blk.1.ffn_gate.weight"
    )


def test_map_hf_attention_biases_keep_bias_suffix():
    # Qwen2-style attention biases must map to attn_*.bias, not collide with
    # the weight tensor (which silently breaks attention).
    assert (
        map_hf_tensor_name("model.layers.0.self_attn.q_proj.bias")
        == "blk.0.attn_q.bias"
    )
    assert (
        map_hf_tensor_name("model.layers.0.self_attn.k_proj.bias")
        == "blk.0.attn_k.bias"
    )
    assert (
        map_hf_tensor_name("model.layers.0.self_attn.v_proj.bias")
        == "blk.0.attn_v.bias"
    )
    assert (
        map_hf_tensor_name("model.layers.0.self_attn.o_proj.bias")
        == "blk.0.attn_output.bias"
    )
