"""Kaggle TPU v5e-8 runner: K2-Horizon-MoVA-36B-A4B on vLLM-TPU.

Start this notebook from the Kaggle UI with Accelerator = "TPU v5e-8" (API-pushed
batch runs on this account land on a CPU box). It:
  1. waits for the 8 TPU chips,
  2. builds a vllm-tpu venv and downloads the bf16 safetensors in parallel,
  3. installs the K2Horizon vLLM plugin (embedded below),
  4. runs bench.py (perplexity + chat sanity + batch 1/8/32/64 speed),
  5. then serves as a job runner: HMAC-signed scripts posted to an ntfy topic are
     executed in the venv and their logs are posted back, until `stop` or the
     deadline. Progress lines + logs go to the OUT topic.
Filled in by build_notebook.py: CFG below.
"""
import base64
import hashlib
import hmac
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

CFG = {}  # __CFG__
CFG = {**CFG, **globals().get("K2_SECRETS", {})}

VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
LOG = WORK / "runner.log"
T0 = time.time()
os.environ.update({
    "HF_HOME": "/tmp/hf",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_TOKEN": CFG["hf_token"],
    "VLLM_XLA_CACHE_PATH": "/tmp/xla_cache",
    "MODEL_IMPL_TYPE": "vllm",
})
os.environ.pop("TPU_LIBRARY_PATH", None)   # the venv ships its own libtpu
_log = open(LOG, "a", buffering=1)


def ntfy(topic, body, filename=None, title=None):
    try:
        headers = {}
        if filename:
            headers["Filename"] = filename
        if title:
            headers["Title"] = title
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode()
                                     if isinstance(body, str) else body,
                                     headers=headers, method="PUT")
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:  # noqa: BLE001
        print(f"(ntfy failed: {e})", flush=True)


def log(msg, push=True):
    line = f"[{time.strftime('%H:%M:%S')} +{int(time.time() - T0)}s] {msg}"
    print(line, flush=True)
    _log.write(line + "\n")
    if push:
        ntfy(CFG["out_topic"], line[:3900])


def run(cmd, tag, env=None, push_prefixes=("RESULT",), timeout=None):
    """Run a command, stream to console+log, forward matching lines to ntfy."""
    out = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})}, bufsize=1)
    deadline = time.time() + timeout if timeout else None
    for line in p.stdout:
        out.append(line)
        _log.write(f"[{tag}] {line}")
        print(f"[{tag}] {line}", end="", flush=True)
        if line.startswith(push_prefixes):
            ntfy(CFG["out_topic"], f"[{tag}] {line.strip()}"[:3900])
        if deadline and time.time() > deadline:
            p.kill()
            out.append(f"\n[runner] killed after {timeout}s\n")
            break
    p.wait()
    return p.returncode, "".join(out)


def tpu_check():
    code = "import jax; d=jax.devices(); print('TPU_CHECK', len(d), d[0].platform, d[0].device_kind)"
    for attempt in range(24):
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        if "TPU_CHECK 8 tpu" in r.stdout:
            log(r.stdout.strip()[-200:])
            return True
        log(f"TPU check {attempt + 1}: {tail[0][:200]}")
        time.sleep(20)
    return False


# ------------------------------------------------------------------ 1. hardware
log("runner start")
_, hw = run(["bash", "-c", "nproc; free -g | head -2; df -h /tmp | tail -1; "
                         "ls /dev/vfio /dev/accel* 2>&1 | tr '\\n' ' '; echo; "
                         "env | grep -i -E 'tpu|pjrt' | grep -v -i token | tr '\\n' ' '"],
            "hw", push_prefixes=())
log("hardware: " + " | ".join(hw.split("\n")))

# ------------------------------------------------------------------ 2. venv + weights (parallel)
state = {}


def build_venv():
    t = time.time()
    rc = 1
    if run([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip", push_prefixes=())[0] == 0 \
            and run([sys.executable, "-m", "uv", "venv", VENV, "--python", sys.executable, "-q"],
                    "uv", push_prefixes=())[0] == 0:
        rc = run([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
                  "--torch-backend=cpu", f"vllm-tpu=={CFG['vllm_tpu_version']}"],
                 "uv", push_prefixes=())[0]
    state["venv"] = (rc, int(time.time() - t))


def download():
    t = time.time()
    try:
        from huggingface_hub import snapshot_download
        state["model"] = snapshot_download(CFG["hf_model_id"], max_workers=16)
        state["dl"] = int(time.time() - t)
    except Exception as e:  # noqa: BLE001
        state["model"] = None
        state["dl_err"] = repr(e)


th = [threading.Thread(target=build_venv), threading.Thread(target=download)]
for x in th:
    x.start()
while any(x.is_alive() for x in th):
    time.sleep(60)
    log(f"setup: venv={'done' if 'venv' in state else '...'} "
        f"download={'done' if 'model' in state else '...'} "
        f"(/tmp/hf {subprocess.run(['du', '-sh', '/tmp/hf'], capture_output=True, text=True).stdout.split()[:1]})")
log(f"venv rc/secs={state.get('venv')}  model={state.get('model')} dl_secs={state.get('dl')} "
    f"err={state.get('dl_err')}")
if state.get("venv", (1,))[0] != 0 or not state.get("model"):
    log("setup FAILED; see runner.log")
    ntfy(CFG["out_topic"], LOG.read_text()[-200000:], filename="runner.log")
    sys.exit(1)
if not tpu_check():
    log("NO TPU visible to the venv's libtpu; see hardware line above.")
    sys.exit(1)
log("TPU OK: 8 chips")
MODEL = state["model"]
os.environ["K2_MODEL"] = MODEL

# ------------------------------------------------------------------ 3. plugin
plug = Path("/tmp/k2plugin")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CFG["plugin_b64"]))).extractall(plug)
rc, o = run([sys.executable, "-m", "uv", "pip", "install", "--python", PY, str(plug)],
            "plugin", push_prefixes=())
log(f"plugin install rc={rc}")
Path("/tmp/bench.py").write_text(base64.b64decode(CFG["bench_b64"]).decode())
run([PY, "-c", "import vllm, tpu_inference, jax; print('VERSIONS', vllm.__version__, "
     "jax.__version__)"], "ver", push_prefixes=("VERSIONS",))

# ------------------------------------------------------------------ 4. baseline benchmark
log("baseline bench starting (compile takes a while)")
rc, out = run([PY, "/tmp/bench.py"], "bench", timeout=3 * 3600)
log(f"baseline bench rc={rc}")
ntfy(CFG["out_topic"], out[-2_000_000:], filename="bench.log", title=f"bench rc={rc}")

# ------------------------------------------------------------------ 5. job loop
key = CFG["hmac_key"].encode()
since = str(int(time.time()) - 5)
deadline = T0 + CFG["keepalive_min"] * 60
jobs = 0
log(f"job loop: listening on the IN topic until {time.strftime('%H:%M', time.localtime(deadline))}")
while time.time() < deadline:
    try:
        raw = urllib.request.urlopen(
            f"https://ntfy.sh/{CFG['in_topic']}/json?poll=1&since={since}", timeout=60).read()
    except Exception as e:  # noqa: BLE001
        print(f"(poll failed: {e})", flush=True)
        time.sleep(15)
        continue
    for ln in raw.decode().splitlines():
        m = json.loads(ln)
        if m.get("event") != "message":
            continue
        since = m["id"]
        if (m.get("message") or "").strip() == "stop":
            log("stop received")
            deadline = 0
            break
        att = m.get("attachment")
        if not att:
            continue
        body = urllib.request.urlopen(att["url"], timeout=60).read()
        sig = (m.get("title") or "").strip()
        if not hmac.compare_digest(sig, hmac.new(key, body, hashlib.sha256).hexdigest()):
            log(f"rejected unsigned job {att.get('name')}")
            continue
        jobs += 1
        name = att.get("name", f"job{jobs}.py")
        path = Path(f"/tmp/jobs/{jobs:03d}_{name}")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(body)
        cmd = ["bash", str(path)] if name.endswith(".sh") else [PY, str(path)]
        log(f"job {jobs} {name} start")
        rc, out = run(cmd, f"job{jobs}", timeout=2 * 3600)
        log(f"job {jobs} {name} rc={rc}")
        ntfy(CFG["out_topic"], out[-2_000_000:] or "(no output)", filename=f"job{jobs}.log",
             title=f"job {jobs} {name} rc={rc}")
    time.sleep(10)
log("runner exit")
