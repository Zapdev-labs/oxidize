"""Check that a train.py adapter loads in stock PEFT on the full Qwen3.8 VLM class and
reproduces train.py's own LoRA forward:
    python check_adapter.py MODEL_DIR ADAPTER_DIR
"""
import os, sys

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OXFT_WEBHOOK", "")
model_dir, ad = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir)
ids = tok("Write a haiku about pipelines.<|im_end|>", return_tensors="pt").input_ids.cuda()

vlm = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float16, device_map="cuda:0")
with torch.no_grad():
    base = vlm(input_ids=ids).logits.float()
pm = PeftModel.from_pretrained(vlm, ad)
n_lora = sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A"))
with torch.no_grad():
    peft_out = pm(input_ids=ids).logits.float()

# train.py's own module on the text tower, weights from the same file
import train as T  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq  # noqa: E402

del pm, vlm
torch.cuda.empty_cache()
top = mq.Qwen3_5ForConditionalGeneration.config_class.from_pretrained(model_dir)
tc = top.text_config
tc.quantization_config = top.quantization_config
lm = mq.Qwen3_5ForCausalLM.from_pretrained(model_dir, config=tc, dtype=torch.float16, device_map="cuda:0")
sd = load_file(os.path.join(ad, "adapter_model.safetensors"))
layers = sorted({int(k.split(".")[5]) for k in sd})
_, names = T.add_lora(lm, layers[0])
for n, m in names:
    m.lora_A.data.copy_(sd[f"base_model.model.{n}.lora_A.weight"].float())
    m.lora_B.data.copy_(sd[f"base_model.model.{n}.lora_B.weight"].float())
with torch.no_grad():
    own = lm(input_ids=ids).logits.float()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
print(f"lora modules in peft model: {n_lora} (adapter tensors {len(sd) // 2}, ours {len(names)})")
print(f"peft vs base rel diff {rel(peft_out, base):.4f}; ours vs peft rel diff {rel(own, peft_out):.5f}")
# fp16 rounding (scale applied before vs after the cast) compounds over layers; a wrong
# key/layout mapping would miss by ~100% of the adapter's effect, not a few percent
effect = rel(peft_out, base)
assert n_lora == len(names) and effect > 1e-2 and rel(own, peft_out) < 0.03 * effect
print("adapter OK")
