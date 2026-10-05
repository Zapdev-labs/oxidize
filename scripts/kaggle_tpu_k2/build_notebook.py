"""Assemble runner.py + plugin + bench.py into a private Kaggle TPU notebook.

    python build_notebook.py --revision <hf commit sha> [--push]
    python build_notebook.py --allow-unpinned [--push]   # moving main branch, on purpose

The model loads with trust_remote_code, so pin --revision (or K2_REVISION) to a
reviewed commit of the HF repo. The ntfy topics and HMAC key live in .state.json
(gitignored). The generated notebook holds no secrets: before running it, attach
Kaggle Secrets HF_TOKEN (your HF token) and K2_HMAC_KEY (the hmac_key value from
.state.json); runner.py reads both at runtime.
"""
import argparse
import base64
import io
import json
import os
import secrets
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / ".state.json"
BUILD = HERE / "build"
SLUG = "k2-horizon-vllm-tpu"

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--revision", default=os.environ.get("K2_REVISION", ""),
                help="HF commit hash of the model repo to pin (env K2_REVISION)")
ap.add_argument("--allow-unpinned", action="store_true",
                help="build without --revision; the runner then pulls remote code from main")
ap.add_argument("--push", action="store_true", help="kaggle kernels push the build")
args = ap.parse_args()
if not args.revision and not args.allow_unpinned:
    sys.exit("refusing to build: pass --revision <commit sha> (or K2_REVISION), or "
             "--allow-unpinned to load trust_remote_code from the moving main branch")

state = json.loads(STATE.read_text()) if STATE.exists() else {}
state.setdefault("in_topic", "k2in-" + secrets.token_hex(16))
state.setdefault("out_topic", "k2out-" + secrets.token_hex(16))
state.setdefault("hmac_key", secrets.token_hex(32))
STATE.write_text(json.dumps(state, indent=1))

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    src = HERE / "k2_horizon_vllm"
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".egg-info"):
            tf.add(p, arcname=str(p.relative_to(src)))

# No secrets here: runner.py reads HF_TOKEN and K2_HMAC_KEY from Kaggle Secrets.
cfg = {
    "hf_model_id": "InfinimindCreations/K2-Horizon-MoVA-36B-A4B-uncensored",
    "hf_revision": args.revision,
    "allow_unpinned": bool(args.allow_unpinned and not args.revision),
    "vllm_tpu_version": "0.29.0",
    "keepalive_min": 510,
    "in_topic": state["in_topic"],
    "out_topic": state["out_topic"],
    "plugin_b64": base64.b64encode(buf.getvalue()).decode(),
    "bench_b64": base64.b64encode((HERE / "bench.py").read_bytes()).decode(),
}
code = (HERE / "runner.py").read_text().replace("CFG = {}  # __CFG__", f"CFG = {cfg!r}")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": code}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
BUILD.mkdir(exist_ok=True)
(BUILD / f"{SLUG}.ipynb").write_text(json.dumps(nb))
user = os.environ.get("KAGGLE_USER", "otdoges")
(BUILD / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{SLUG}", "title": SLUG.replace("-", " "), "code_file": f"{SLUG}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": False, "enable_tpu": True, "enable_internet": True,
    "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    "model_sources": [], "machine_shape": "TpuV5E8"}))
assert state["hmac_key"] not in code, "HMAC key leaked into the notebook source"
_hf_token = os.environ.get("HF_TOKEN")
assert not _hf_token or _hf_token not in code, "HF token leaked into the notebook source"
print("built", BUILD / f"{SLUG}.ipynb", f"({len(code) // 1024} KB)")
print("attach Kaggle Secrets HF_TOKEN and K2_HMAC_KEY (hmac_key in .state.json) to the notebook")
if args.push:
    subprocess.run(["kaggle", "kernels", "push", "-p", str(BUILD)], check=True)
