---
base_model: empero-ai/Qwythos-9B-Claude-Mythos-5-1M
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- coding-agent
- qwen3_5
- unsloth
---

# qwen35-9b-self-train-lora

Self-train LoRA fine-tune of a 9B hybrid (Gated DeltaNet + Gated Attention)
model for coding-agent behavior. This repo ships the adapter plus the full
merged and base weights.

## Contents

| Path | What it is |
|------|------------|
| `adapter/` | LoRA weights only (~233 MB) — needs the base model |
| `merged/` | Full merged safetensors (base + LoRA baked in, ~17 GB) — standalone |
| `base/` | Original pre-LoRA safetensors (~18 GB) |

## Training

- **Method:** BF16 LoRA SFT (Unsloth)
- **LoRA config:** rank 32, alpha 64, targets `q/k/v/o_proj` + `gate/up/down_proj`
- **Schedule:** 1 epoch, 400 steps, lr 2e-4 cosine, `adamw_8bit`, seq len 2048
- **Data:** 1,600 coding-agent trajectories (tool use, file edits, build/test loops)
- **Hardware:** single A100 40GB (~15 min, final train loss ≈ 0.82)

## Usage (adapter on top of base)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "empero-ai/Qwythos-9B-Claude-Mythos-5-1M"
adapter = "freakyskittle/qwen35-9b-self-train-lora"

tok = AutoTokenizer.from_pretrained(adapter, subfolder="adapter", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter, subfolder="adapter")
```

## Usage (merged, no adapter)

Use the standalone merged repo
[`freakyskittle/qwen35-9b-self-train-merged`](https://huggingface.co/freakyskittle/qwen35-9b-self-train-merged)
or the `merged/` subfolder here.

## Notes

Requires `transformers>=5.0`. For the linear-attention fast path, install
`flash-linear-attention` and `causal-conv1d`.
