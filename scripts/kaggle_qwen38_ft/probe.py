"""Kaggle T4x2 capability probe for Qwen3.8-27B fast finetuning.

Measures: fp16 GEMM throughput per GPU (alone + concurrent), bnb NF4 linear
fwd/bwd cost, FLA gated-delta-rule kernels on sm75 (correctness + speed),
HF download bandwidth for the pre-quantized checkpoint, and a 2-GPU model
load + forward/backward smoke test.

Secrets are injected at push time by push_kernel.py (never committed).
"""
import json, os, subprocess, sys, time, threading, traceback, urllib.request

WH = os.environ.get("OXFT_WEBHOOK", "__WEBHOOK__")
HF_TOKEN = os.environ.get("HF_TOKEN", "__HF_TOKEN__")
os.environ["HF_TOKEN"] = HF_TOKEN
MODEL = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
MODEL_DIR = "/root/qwen38-bnb4"
RESULTS = {}


def post(stage, **kw):
    RESULTS[stage] = kw
    print(f"[{stage}] {json.dumps(kw)[:2000]}", flush=True)
    if WH.startswith("__"):
        return
    try:
        body = json.dumps({"stage": stage, "ts": time.time(), **kw}).encode()
        req = urllib.request.Request(WH, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # noqa: BLE001
        print("webhook failed:", e, flush=True)


def sh(cmd):
    t = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-3000:], round(time.time() - t, 1)


def guarded(name):
    def deco(fn):
        def run():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                post(name + "_error", err=repr(e), tb=traceback.format_exc()[-2500:])
        return run
    return deco


post("start", host=os.uname().nodename)
_, smi, _ = sh("nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm --format=csv")
_, mem, _ = sh("free -g; nproc; df -h /root /kaggle/working /tmp | tail -3")
post("env", smi=smi, mem=mem, py=sys.version)

# --- background download of the checkpoint (measures bandwidth) ---
dl = {}


def download():
    t = time.time()
    rc, out, _ = sh("pip install -q -U 'huggingface_hub[hf_xet]' && "
                    f"hf download {MODEL} --local-dir {MODEL_DIR}")
    dl.update(rc=rc, secs=round(time.time() - t, 1), tail=out[-600:])
    post("download_done", **dl)


dl_thread = threading.Thread(target=download, daemon=True)
if os.environ.get("OXFT_SKIP_DOWNLOAD") != "1":
    dl_thread.start()

rc, out, secs = sh("pip install -q -U 'transformers>=5.20' peft bitsandbytes accelerate "
                   "flash-linear-attention 2>&1 | tail -5")
post("pip", rc=rc, secs=secs, tail=out)
_, vers, _ = sh("python -c \"import torch,triton,transformers,peft,bitsandbytes;"
                "print(torch.__version__, torch.version.cuda, triton.__version__, transformers.__version__,"
                " peft.__version__, bitsandbytes.__version__)\"")
post("versions", v=vers)

import torch  # noqa: E402


def bench(fn, iters=10, warm=3, devs=(0,)):
    for _ in range(warm):
        fn()
    for d in devs:
        torch.cuda.synchronize(d)
    t = time.time()
    for _ in range(iters):
        fn()
    for d in devs:
        torch.cuda.synchronize(d)
    return (time.time() - t) / iters


@guarded("gemm")
def gemm():
    M, K, N = 4096, 5120, 17408
    flop = 2 * M * K * N
    res = {}
    mats = []
    ngpu = torch.cuda.device_count()
    for d in range(ngpu):
        a = torch.randn(M, K, device=f"cuda:{d}", dtype=torch.float16)
        b = torch.randn(K, N, device=f"cuda:{d}", dtype=torch.float16)
        mats.append((a, b))
        s = bench(lambda: a @ b, iters=20, devs=(d,))
        res[f"gpu{d}_tflops"] = round(flop / s / 1e12, 1)
    if ngpu > 1:
        s = bench(lambda: [x @ y for x, y in mats], iters=20, devs=(0, 1))
        res["concurrent_total_tflops"] = round(2 * flop / s / 1e12, 1)
    # sustained (thermal/power throttle) 15 s
    a, b = mats[0]
    t = time.time(); n = 0
    while time.time() - t < 15:
        a @ b; n += 1
        if n % 20 == 0:
            torch.cuda.synchronize(0)
    torch.cuda.synchronize(0)
    res["gpu0_sustained_15s_tflops"] = round(n * flop / (time.time() - t) / 1e12, 1)
    post("gemm", **res)


@guarded("bnb")
def bnb():
    import bitsandbytes as bnbm
    M, K, N = 4096, 5120, 17408
    lin = bnbm.nn.Linear4bit(K, N, bias=False, compute_dtype=torch.float16, quant_type="nf4")
    lin = lin.to("cuda:0")
    x = torch.randn(M, K, device="cuda:0", dtype=torch.float16, requires_grad=True)
    f = bench(lambda: lin(x), iters=10)
    def fb():
        y = lin(x); y.backward(torch.ones_like(y))
    fbt = bench(fb, iters=10)
    W = lin.weight
    deq = bench(lambda: bnbm.functional.dequantize_4bit(W.data, W.quant_state), iters=20)
    flop = 2 * M * K * N
    post("bnb", fwd_ms=round(f * 1e3, 1), fwd_tflops=round(flop / f / 1e12, 1),
         fwdbwd_ms=round(fbt * 1e3, 1), dequant_ms=round(deq * 1e3, 2))


@guarded("fla")
def fla():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import torch.nn.functional as F
    B, T, H, K, V = 1, 4096, 48, 128, 128
    dev = "cuda:0"
    def mk(T):
        q = F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).half().requires_grad_()
        k = F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).half().requires_grad_()
        v = torch.randn(B, T, H, V, device=dev).half().requires_grad_()
        g = (-torch.rand(B, T, H, device=dev) * 0.1).float().requires_grad_()
        beta = torch.rand(B, T, H, device=dev).half().requires_grad_()
        return q, k, v, g, beta
    # correctness vs naive fp32 recurrence on short seq
    q, k, v, g, beta = mk(256)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=K ** -0.5)
    S = torch.zeros(B, H, K, V, device=dev)
    qf, kf, vf, gf, bf = (t.float() for t in (q, k, v, g, beta))
    outs = []
    for t in range(256):
        S = S * gf[:, t].exp()[..., None, None]
        kt, vt = kf[:, t], vf[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, (vt - pred) * bf[:, t][..., None])
        outs.append(torch.einsum("bhk,bhkv->bhv", qf[:, t] * K ** -0.5, S))
    ref = torch.stack(outs, 1)
    err = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    q, k, v, g, beta = mk(T)
    fw = bench(lambda: chunk_gated_delta_rule(q, k, v, g, beta, scale=K ** -0.5), iters=5)
    def fb():
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=K ** -0.5)
        o.sum().backward()
    fbt = bench(fb, iters=5)
    post("fla", rel_err=err, fwd_ms=round(fw * 1e3, 1), fwdbwd_ms=round(fbt * 1e3, 1), T=T)


gemm(); bnb(); fla()

if dl_thread.is_alive():
    dl_thread.join(timeout=3600)


@guarded("model")
def model_smoke():
    from transformers import AutoModelForCausalLM, AutoConfig
    t = time.time()
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    n = cfg.text_config.num_hidden_layers if hasattr(cfg, "text_config") else cfg.num_hidden_layers
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", dtype=torch.float16,
                                             max_memory={0: "13GiB", 1: "13GiB", "cpu": "28GiB"})
    load = round(time.time() - t, 1)
    dm = {}
    for k, v in getattr(m, "hf_device_map", {}).items():
        dm.setdefault(str(v), 0); dm[str(v)] += 1
    post("model_loaded", secs=load, cls=type(m).__name__, layers=n, device_counts=dm,
         mem=[round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in (0, 1)])
    ids = torch.randint(0, 200000, (1, 1024), device="cuda:0")
    with torch.no_grad():
        t = time.time(); out = m(input_ids=ids, logits_to_keep=1); torch.cuda.synchronize(); fwd = time.time() - t
        t = time.time(); out = m(input_ids=ids, logits_to_keep=1); torch.cuda.synchronize(); fwd2 = time.time() - t
    post("model_fwd", first_s=round(fwd, 2), warm_s=round(fwd2, 2), tokens=1024,
         tok_per_s=round(1024 / fwd2, 1), finite=bool(torch.isfinite(out.logits).all()),
         peak=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in (0, 1)])
    print(m, file=open("/kaggle/working/model_repr.txt", "w"))


if dl.get("rc") == 0 and torch.cuda.device_count() > 1:
    model_smoke()
json.dump(RESULTS, open(os.environ.get("OXFT_OUT", "/kaggle/working") + "/probe_results.json", "w"), indent=1)
post("done")
