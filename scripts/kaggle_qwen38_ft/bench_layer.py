"""Time real-size Qwen3.8-27B decoder layers (NF4 + LoRA, gdn_t4 kernel) on one T4.
Builds a 4-layer (3 GDN + 1 attention) random model at the real text dims, NF4-quantizes it the
same way as the unsloth checkpoint, and reports per-layer fwd / train (ckpt fwd+recompute+bwd)
times plus the ChunkedCE lm_head cost, so the 10-minute token budget can be planned:
    python bench_layer.py [/tmp/bench-q38]
"""
import json, os, sys, time

import torch
from huggingface_hub import hf_hub_download

os.environ.setdefault("OXFT_DEVICES", "cuda:0")
os.environ.setdefault("OXFT_WEBHOOK", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402
from transformers import AutoConfig, BitsAndBytesConfig, Qwen3_5ForCausalLM  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench-q38"
if not os.path.exists(os.path.join(out, "model.safetensors")):
    c = json.load(open(hf_hub_download("unsloth/Qwen3.8-27B-unsloth-bnb-4bit", "config.json")))
    tc = c["text_config"]
    tc.update(num_hidden_layers=4, layer_types=tc["layer_types"][:4], mtp_num_hidden_layers=0, vocab_size=4096)
    os.makedirs(out + "-fp16", exist_ok=True)
    json.dump(tc, open(out + "-fp16/config.json", "w"))
    cfg = AutoConfig.from_pretrained(out + "-fp16")
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(cfg).half()
    for p in m.parameters():
        p.data.normal_(0, 0.02)
    m.save_pretrained(out + "-fp16")
    del m
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.float16,
                           llm_int8_skip_modules=["lm_head", "in_proj_a", "in_proj_b", "in_proj_qkv"])
    mq = Qwen3_5ForCausalLM.from_pretrained(out + "-fp16", quantization_config=q, dtype=torch.float16,
                                            device_map="cuda:0")
    mq.save_pretrained(out)
    del mq
    torch.cuda.empty_cache()

import gdn_t4  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq  # noqa: E402

mq.torch_chunk_gated_delta_rule = gdn_t4.chunk_gated_delta_rule
model = Qwen3_5ForCausalLM.from_pretrained(out, dtype=torch.float16, device_map="cuda:0")
for p in model.parameters():
    p.requires_grad_(False)
params, names = T.add_lora(model, 0)
pipe = T.Pipe(model, ["cuda:0"], 4, 0)
dev = torch.device("cuda:0")
D, V = model.config.hidden_size, 248320
res = {"gpu": torch.cuda.get_device_name(0), "mem_weights_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}


def timeit(fn, n=3):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


for B, Tn in [(2, 2048), (8, 512)]:
    h0 = torch.randn(B, Tn, D, device=dev, dtype=torch.float16) * 0.5
    cos, sin, pid = pipe._pos(B, Tn, dev)
    for i, name in [(0, "gdn"), (3, "attn")]:
        L = pipe.layers[i]

        def fwd():
            with torch.no_grad():
                T.Pipe._layer(L, h0, cos, sin, pid)

        def fwd_frozen():
            T.LoRA.enabled = False
            fwd()
            T.LoRA.enabled = True

        def trn():
            h = h0.clone().requires_grad_(True)
            y = T.checkpoint(T.Pipe._layer, L, h, cos, sin, pid, use_reentrant=False)
            y.float().square().mean().backward()

        def trn_nockpt():
            h = h0.clone().requires_grad_(True)
            y = T.Pipe._layer(L, h, cos, sin, pid)
            y.float().square().mean().backward()

        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        r = {"fwd_ms": timeit(fwd), "fwd_frozen_ms": timeit(fwd_frozen), "train_ckpt_ms": timeit(trn), "train_nockpt_ms": timeit(trn_nockpt),
             "peak_extra_gib": round((torch.cuda.max_memory_allocated() - base) / 2**30, 2)}
        res[f"{name}_B{B}_T{Tn}"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()}
        print(name, B, Tn, res[f"{name}_B{B}_T{Tn}"], flush=True)
        model.zero_grad(set_to_none=True)

# lm_head ChunkedCE cost per 1024 supervised rows at the real vocab
W = torch.randn(V, D, device=dev, dtype=torch.float16) * 0.02
hs = torch.randn(1024, D, device=dev, dtype=torch.float16, requires_grad=True)
lab = torch.randint(0, V, (1024,), device=dev)
res["ce_ms_per_1k_rows"] = round(timeit(lambda: T.ChunkedCE.apply(hs, W, lab, 256, 1.0).backward()), 1)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(out) or ".", "bench_layer.json"), "w"), indent=1)

if os.environ.get("BENCH_PROF"):
    from torch.profiler import ProfilerActivity, profile
    B, Tn = 2, 2048
    h0 = torch.randn(B, Tn, D, device=dev, dtype=torch.float16) * 0.5
    cos, sin, pid = pipe._pos(B, Tn, dev)
    for i in (0, 3):
        L = pipe.layers[i]

        def trn():
            h = h0.clone().requires_grad_(True)
            y = T.checkpoint(T.Pipe._layer, L, h, cos, sin, pid, use_reentrant=False)
            y.float().square().mean().backward()
        trn(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            trn(); torch.cuda.synchronize()
        print(f"--- layer {i}")
        rows = [e for e in pr.key_averages(group_by_input_shape=True) if e.key.startswith("aten::") or "Backward" in e.key]
        rows.sort(key=lambda e: -e.self_device_time_total)
        for e in rows[:40]:
            print(f"{e.self_device_time_total/1e3:8.1f} ms  n={e.count:4d}  {e.key[:40]:40s} {str(e.input_shapes)[:110]}")

if os.environ.get("BENCH_MEM"):
    for B, Tn in [(2, 2048), (1, 4096)]:
        h0 = torch.randn(B, Tn, D, device=dev, dtype=torch.float16) * 0.5
        cos, sin, pid = pipe._pos(B, Tn, dev)
        for i in (0, 3):
            torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
            h = h0.clone().requires_grad_(True)
            y = T.Pipe._layer(pipe.layers[i], h, cos, sin, pid)
            torch.cuda.synchronize()
            print(f"saved_gib layer{i} B{B} T{Tn}: {(torch.cuda.memory_allocated() - base) / 2**30:.2f}", flush=True)
            del y, h
            torch.cuda.empty_cache()
    for i in (0, 3):   # forward-only (stage 0) transient peak
        h0 = torch.randn(1, 4096, D, device=dev, dtype=torch.float16) * 0.5
        cos, sin, pid = pipe._pos(1, 4096, dev)
        T.LoRA.enabled = False
        torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            y = T.Pipe._layer(pipe.layers[i], h0, cos, sin, pid)
        torch.cuda.synchronize()
        print(f"fwd_peak_gib layer{i} T4096: {(torch.cuda.max_memory_allocated() - base) / 2**30:.2f}", flush=True)
        T.LoRA.enabled = True
        del y
