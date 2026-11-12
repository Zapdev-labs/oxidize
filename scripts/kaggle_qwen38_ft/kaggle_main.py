"""Kaggle T4x2 entry point: fetch the NF4 checkpoint, pin the stack, run train.py.

Pushed by push_kernel.py, which inlines gdn_t4.py / train.py as base64 and injects
secrets (never committed). The fine-tune wall clock is measured by train.py itself
(process start -> adapter saved, includes model load); the checkpoint download runs
before it and is reported separately.
"""
import base64, json, os, subprocess, sys, time, urllib.request

WH = os.environ.get("OXFT_WEBHOOK", "__WEBHOOK__")
HF_TOKEN = os.environ.get("HF_TOKEN", "__HF_TOKEN__")
HF_REPO = "__HF_REPO__"                                         # optional private adapter upload
_EXTRA = "__EXTRA_ENV__"                                        # JSON of OXFT_* overrides
EXTRA_ENV = {} if _EXTRA.startswith("__") else json.loads(_EXTRA)
MODEL = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
MODEL_DIR = "/root/qwen38-bnb4"
WORK = "/root/oxft"
OUT = "/kaggle/working"
FILES = {"gdn_t4.py": "@@B64 gdn_t4.py@@", "train.py": "@@B64 train.py@@"}
T0 = time.time()


def post(stage, **kw):
    kw = {"t": round(time.time() - T0, 1), **kw}
    print(f"[{stage}] {json.dumps(kw)[:3000]}", flush=True)
    if not WH.startswith("http"):
        return
    try:
        req = urllib.request.Request(WH, data=json.dumps({"stage": "main_" + stage, **kw}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # noqa: BLE001
        print("webhook failed:", e, flush=True)


def sh(cmd, env=None):
    t = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout + r.stderr)[-4000:], round(time.time() - t, 1)


os.environ["HF_TOKEN"] = HF_TOKEN
_, smi, _ = sh("nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm --format=csv,noheader")
_, mem, _ = sh("free -g | head -2; nproc; df -h /root | tail -1")
post("env", smi=smi, mem=mem)

# 1) checkpoint download (xet) in the background while pip runs
dl = subprocess.Popen(f"pip install -q 'huggingface_hub[hf_xet]>=1.0,<2' && hf download {MODEL} --local-dir {MODEL_DIR}",
                      shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
rc, out, secs = sh("pip install -q 'transformers==5.22.2' 'bitsandbytes>=0.48' 'accelerate>=1.10' "
                   "'datasets>=4' safetensors 2>&1 | tail -3; pip uninstall -y -q torchao 2>&1 | tail -1")
_, vers, _ = sh("python -c \"import torch,triton,transformers,bitsandbytes;print(torch.__version__,"
                "torch.version.cuda,triton.__version__,transformers.__version__,bitsandbytes.__version__)\"")
post("pip", rc=rc, secs=secs, tail=out[-500:], versions=vers.strip())
dl_out = dl.communicate()[0]
post("download", rc=dl.returncode, secs=round(time.time() - T0, 1), tail=dl_out[-400:],
     gib=round(sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR)
                   if os.path.isfile(os.path.join(MODEL_DIR, f))) / 2**30, 2) if os.path.isdir(MODEL_DIR) else 0)
if dl.returncode:
    sys.exit("download failed")
# page the weights in once so train.py's load measures the box, not cold network storage
sh(f"cat {MODEL_DIR}/*.safetensors > /dev/null")

# 2) trainer
os.makedirs(WORK, exist_ok=True)
for name, b64 in FILES.items():
    open(os.path.join(WORK, name), "wb").write(base64.b64decode(b64))
env = dict(os.environ, OXFT_MODEL_DIR=MODEL_DIR, OXFT_OUT=OUT, OXFT_WEBHOOK=WH, PYTHONUNBUFFERED="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false", **EXTRA_ENV)
t = time.time()
p = subprocess.Popen([sys.executable, "train.py"], cwd=WORK, env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True)
log = open(os.path.join(OUT, "train_stdout.txt"), "w")
for line in p.stdout:
    log.write(line); log.flush()
    if not line.startswith("["):
        print(line, end="", flush=True)
p.wait()
post("train_exit", rc=p.returncode, secs=round(time.time() - t, 1))

# 3) optional private upload of the adapter + report
rep = os.path.join(OUT, "oxft_report.json")
ad = os.path.join(OUT, "qwen38-27b-oxft-lora")
if p.returncode == 0 and HF_REPO and not HF_REPO.startswith("__") and os.path.isdir(ad):
    import shutil
    shutil.copy(rep, ad)
    code = ("from huggingface_hub import HfApi; a=HfApi(); "
            f"a.create_repo('{HF_REPO}', private=True, exist_ok=True); "
            f"a.upload_folder(folder_path='{ad}', repo_id='{HF_REPO}', commit_message='oxft Kaggle T4x2 run')")
    rc, out, secs = sh(f"python -c \"{code}\"")
    post("hf_upload", rc=rc, secs=secs, repo=HF_REPO, tail=out[-300:])
post("done", report=json.load(open(rep)).get("gate") if os.path.exists(rep) else None)
