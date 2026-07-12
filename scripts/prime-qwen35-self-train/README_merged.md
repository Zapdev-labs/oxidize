---
base_model: empero-ai/Qwythos-9B-Claude-Mythos-5-1M
library_name: transformers
pipeline_tag: text-generation
tags:
- lora
- sft
- coding-agent
- qwen3_5
- unsloth
---

# qwen35-9b-self-train-merged

A 9B hybrid (Gated DeltaNet + Gated Attention) model fine-tuned for
coding-agent behavior, with the trained LoRA merged back into the base
weights. Load it directly — no adapter required.

## Details

- **Architecture:** Qwen3.5-style hybrid (3:1 linear/full attention), 32 layers, hidden 4096, vocab 248320
- **Training:** BF16 LoRA SFT (Unsloth), merged with `merge_and_unload()`
- **LoRA config:** rank 32, alpha 64, targets `q/k/v/o_proj` + `gate/up/down_proj`
- **Schedule:** 1 epoch, 400 steps, lr 2e-4 cosine, `adamw_8bit`, seq len 2048
- **Data:** 1,600 coding-agent trajectories (tool use, file edits, build/test loops)
- **Hardware:** single A100 40GB (~15 min, final train loss ≈ 0.82)
- **Precision:** bf16, 4 safetensors shards

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "freakyskittle/qwen35-9b-self-train-merged"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

msgs = [{"role": "user", "content": "Write a Python function to reverse a linked list."}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Notes

Requires `transformers>=5.0`. For the linear-attention fast path, install
`flash-linear-attention` and `causal-conv1d`. The LoRA-only adapter and the
original base weights are available at
[`freakyskittle/qwen35-9b-self-train-lora`](https://huggingface.co/freakyskittle/qwen35-9b-self-train-lora).
