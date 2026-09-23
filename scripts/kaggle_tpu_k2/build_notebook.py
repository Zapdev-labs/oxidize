"""Assemble runner.py + plugin + bench.py into a private Kaggle TPU notebook.

    HF_TOKEN=hf_... python build_notebook.py [--push]

Secrets (HF token, ntfy topics, HMAC key) go to .state.json / build/ (gitignored).
"""
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

cfg = {
    "hf_token": os.environ.get("HF_TOKEN", ""),
    "hf_model_id": "InfinimindCreations/K2-Horizon-MoVA-36B-A4B-uncensored",
    "vllm_tpu_version": "0.29.0",
    "keepalive_min": 510,
    "in_topic": state["in_topic"],
    "out_topic": state["out_topic"],
    "hmac_key": state["hmac_key"],
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
print("built", BUILD / f"{SLUG}.ipynb", f"({len(code) // 1024} KB)")
if "--push" in sys.argv:
    subprocess.run(["kaggle", "kernels", "push", "-p", str(BUILD)], check=True)
