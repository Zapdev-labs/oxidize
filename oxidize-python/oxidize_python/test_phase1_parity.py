"""Layer-wise and LoRA parity tests."""

from __future__ import annotations

from oxidize_python.core.model.inference import InferenceConfig, InferenceModel, WeightStorage
from oxidize_python.core.model.layer_wise import LayerWiseModel, new_layer_wise_from_inference
from oxidize_python.core.model.lora import LoraLayer, new_lora_layer
from oxidize_python.core.model.model import Session


def test_layer_wise_delegates_to_inner() -> None:
    cfg = InferenceConfig(hidden_size=8, vocab_size=4, layer_count=2, context_size=16)
    inner = InferenceModel(config=cfg, storage=WeightStorage(), stack=None)
    wrapped = new_layer_wise_from_inference(inner, 2)
    assert wrapped.inner is inner
    logits = wrapped.forward([1], Session())
    assert len(logits) == cfg.vocab_size


def test_lora_low_rank_delta() -> None:
    layer = new_lora_layer("test", rank=2, alpha=4.0, base_shape=[4, 4])
    layer.set_low_rank_weights(
        up=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        down=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        in_dim=4,
        out_dim=4,
    )
    x = [1.0, 2.0, 3.0, 4.0]
    out = [0.0, 0.0, 0.0, 0.0]
    layer.apply_low_rank_delta(x, out)
    assert any(v != 0.0 for v in out)
