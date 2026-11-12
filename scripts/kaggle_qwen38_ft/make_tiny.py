"""Build a tiny random Qwen3.5 (qwen3_5) VLM checkpoint, NF4-prequantized like the
unsloth 27B one, plus the real Qwen3.8 tokenizer, for local pipeline smoke tests:
    python make_tiny.py /tmp/tiny-q38 && OXFT_MODEL_DIR=/tmp/tiny-q38 OXFT_DEVICES=cuda:0 python train.py
"""
import json, os, sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

REPO = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
out = sys.argv[1]
raw, fp16 = out + "-fp16", out
cfg_path = hf_hub_download(REPO, "config.json")
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    try:
        p = hf_hub_download(REPO, f)
        for d in (raw, out):
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f), "wb").write(open(p, "rb").read())
    except Exception as e:  # noqa: BLE001
        print("skip", f, e)
c = json.load(open(cfg_path))
c.pop("quantization_config", None)
tc = c["text_config"]
L = int(os.environ.get("TINY_LAYERS", 8))
tc.update(hidden_size=512, intermediate_size=1024, num_hidden_layers=L, num_attention_heads=4,
          num_key_value_heads=1, linear_num_key_heads=2, linear_num_value_heads=4,
          layer_types=tc["layer_types"][:L], mtp_num_hidden_layers=0)
vc = c["vision_config"]
vc.update(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=512)
os.makedirs(raw, exist_ok=True)
json.dump(c, open(os.path.join(raw, "config.json"), "w"))
cfg = AutoConfig.from_pretrained(raw)
torch.manual_seed(0)
m = Qwen3_5ForConditionalGeneration(cfg).half()
m.save_pretrained(raw)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16,
                       llm_int8_skip_modules=["lm_head", "model.visual", ".*\\.visual\\..*", "in_proj_a",
                                              "in_proj_b", "in_proj_qkv"])
mq = Qwen3_5ForConditionalGeneration.from_pretrained(raw, quantization_config=q, dtype=torch.float16, device_map="cuda:0")
mq.save_pretrained(out)
print("saved", out, sorted(os.listdir(out)))
