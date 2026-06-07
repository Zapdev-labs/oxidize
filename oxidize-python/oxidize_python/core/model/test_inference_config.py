from oxidize_python.core.model.inference_config import (
    InferenceConfig,
    _apply_token_embedding_dims,
)


def test_apply_token_embedding_dims_preserves_qwen3_metadata() -> None:
    cfg = InferenceConfig(hidden_size=2560, vocab_size=151936)
    _apply_token_embedding_dims(cfg, [2560, 151936])
    assert cfg.hidden_size == 2560
    assert cfg.vocab_size == 151936


def test_apply_token_embedding_dims_fills_missing() -> None:
    cfg = InferenceConfig(hidden_size=0, vocab_size=0)
    _apply_token_embedding_dims(cfg, [2560, 151936])
    assert cfg.hidden_size == 2560
    assert cfg.vocab_size == 151936


def test_gemma_decoder_config_layer_pattern() -> None:
    from oxidize_python.core.model.llama_decoder import LlamaDecoderConfig

    cfg = LlamaDecoderConfig(
        hidden_size=8,
        layer_count=12,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        key_value_head_dim=4,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        sliding_window=1024,
        rope_theta_swa=10_000.0,
        sliding_window_pattern=6,
    )
    for layer in range(5):
        assert not cfg.layer_is_global(layer)
        assert cfg.layer_rope_theta(layer) == 10_000.0
        assert cfg.layer_sliding_window(layer) == 1024
    assert cfg.layer_is_global(5)
    assert cfg.layer_rope_theta(5) == 1_000_000.0
    assert cfg.layer_sliding_window(5) == 0


def test_mistral_uniform_window_still_applies() -> None:
    from oxidize_python.core.model.llama_decoder import LlamaDecoderConfig

    cfg = LlamaDecoderConfig(
        hidden_size=8,
        layer_count=4,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        key_value_head_dim=4,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        sliding_window=4096,
    )
    for layer in range(4):
        assert not cfg.layer_is_global(layer)
        assert cfg.layer_sliding_window(layer) == 4096
