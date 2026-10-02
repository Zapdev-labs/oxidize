import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


SCRIPT = Path(__file__).with_name("k2_merge_lora.py")
SPEC = importlib.util.spec_from_file_location("k2_merge_lora", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_target_name_maps_peft_keys_to_base_weights():
    assert MODULE.target_name(
        "base_model.model.model.layers.19.self_attn.q_proj.lora_A.weight"
    ) == ("model.layers.19.self_attn.q_proj.weight", "lora_A")
    assert MODULE.target_name(
        "base_model.model.model.layers.19.self_attn.o_proj.lora_B.weight"
    ) == ("model.layers.19.self_attn.o_proj.weight", "lora_B")


def test_target_name_rejects_unknown_keys():
    assert MODULE.target_name("model.layers.0.weight") is None


def test_merges_lora_delta_and_preserves_other_tensors(tmp_path, monkeypatch):
    base_dir = tmp_path / "base"
    output_dir = tmp_path / "output"
    base_dir.mkdir()
    target = "model.layers.0.self_attn.q_proj.weight"
    shard = "model-00001-of-00001.safetensors"
    base_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    untouched = torch.tensor([5.0], dtype=torch.bfloat16)
    save_file({target: base_weight, "model.norm.weight": untouched}, base_dir / shard)
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 10},
                "weight_map": {
                    target: shard,
                    "model.norm.weight": shard,
                },
            }
        )
    )
    (base_dir / "config.json").write_text('{"model_type":"k2_horizon"}')

    a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    b = torch.tensor([[3.0], [4.0]], dtype=torch.float32)
    adapter_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
        },
        adapter_path,
    )
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text('{"r":1,"lora_alpha":2}')

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--base",
            str(base_dir),
            "--adapter",
            str(adapter_path),
            "--adapter-config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
    )
    MODULE.main()

    merged = load_file(output_dir / shard)
    expected = base_weight.float() + torch.matmul(b, a) * 2.0
    torch.testing.assert_close(merged[target].float(), expected.to(torch.bfloat16).float())
    torch.testing.assert_close(merged["model.norm.weight"], untouched)
    assert json.loads((output_dir / "merge_summary.json").read_text())[
        "merged_tensor_count"
    ] == 1
    assert (output_dir / "config.json").read_text() == '{"model_type":"k2_horizon"}'
