"""
Train a custom DFlash draft for Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16 on Modal.

Usage:
    modal run modal_dflash_train.py --smoke
    modal run modal_dflash_train.py
    OXIDIZE_MODAL_GPU=H100 modal run modal_dflash_train.py --max-steps 2000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

GPU = os.environ.get("OXIDIZE_MODAL_GPU", "A10G")
REPO_ROOT = "/workspace"
HF_CACHE = "/root/.cache/huggingface"
OUT_DIR = "/vol/dflash-out"

IGNORE = [
    "target/**",
    ".git/**",
    "models/**",
    "dist/**",
    "node_modules/**",
    "**/*.gguf.bak",
    "rust_out/**",
    ".omo/**",
    ".cursor/**",
    ".claude/**",
    "deploy/**",
    "uv.lock",
    "bun.lock",
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
    "**/*.log",
    "**/*.o",
    "**/*.a",
    "oxidize-c/oxidize-c",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "pkg-config")
    .pip_install(
        "torch==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate==1.15.0",
        "bitsandbytes==0.50.2",
        "datasets==5.0.1",
        "gguf==0.19.0",
        "huggingface_hub==1.31.0",
        "protobuf==6.31.1",
        "safetensors==0.8.0",
        "sentencepiece==0.2.1",
        # Target config.json was written by 5.8.0.dev0 (model_type qwen3_5).
        "transformers==5.8.0",
    )
    .add_local_dir(".", REPO_ROOT, ignore=IGNORE, copy=True)
)

hf_cache = modal.Volume.from_name("oxidize-model-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("oxidize-dflash-qwen38", create_if_missing=True)

app = modal.App("oxidize-dflash-qwen38")


@app.function(image=image, cpu=4.0, memory=8192, timeout=600)
def unit_tests() -> str:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "scripts.test_dflash_train", "-v"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    print(text, flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"unit tests failed:\n{text}")
    return "unit tests OK"


def _cfg(smoke: bool, max_steps: int | None, max_samples: int | None):
    import sys

    sys.path.insert(0, REPO_ROOT)
    from scripts.dflash.config import DFlashTrainConfig

    cfg = DFlashTrainConfig()
    if GPU in {"A10G", "T4", "L4", "L4:1"}:
        cfg.max_seq_len = min(cfg.max_seq_len, 256)
        cfg.max_anchors = min(cfg.max_anchors, 4)
        cfg.grad_accum = max(cfg.grad_accum, 8)
        cfg.max_samples = min(cfg.max_samples, 256)
    if smoke:
        cfg.max_steps = 4
        cfg.max_samples = 16
        cfg.max_seq_len = 256
        cfg.max_anchors = 8
        cfg.grad_accum = 1
    if max_steps is not None:
        cfg.max_steps = max_steps
    if max_samples is not None:
        cfg.max_samples = max_samples
    return cfg


@app.function(
    image=image,
    timeout=180,
    cpu=2.0,
    memory=4096,
    secrets=[modal.Secret.from_name("hf-token")],
)
def hf_probe() -> str:
    import os

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    me = api.whoami()
    namespaces: list[str] = []
    if me.get("name"):
        namespaces.append(str(me["name"]))
    for org in me.get("orgs") or []:
        if isinstance(org, dict) and org.get("name"):
            namespaces.append(str(org["name"]))
    print({"name": me.get("name"), "orgs": namespaces, "auth": me.get("auth")}, flush=True)
    last = ""
    for ns in namespaces:
        repo_id = f"{ns}/Qwen3.8-27B-ABLITERATED-DFlash"
        try:
            api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
            print(f"writable {repo_id}", flush=True)
            return repo_id
        except HfHubHTTPError as exc:
            last = f"{repo_id}: {exc}"
            print(last, flush=True)
            try:
                api.repo_info(repo_id, repo_type="model")
                print(f"existing {repo_id}", flush=True)
                return repo_id
            except Exception:
                continue
    print(f"no writable repo yet; training will keep artifacts on the Modal volume. last={last}", flush=True)
    return namespaces[0] + "/Qwen3.8-27B-ABLITERATED-DFlash" if namespaces else ""


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 60 * 60,
    memory=131072,
    cpu=8.0,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={
        HF_CACHE: hf_cache,
        OUT_DIR: out_vol,
    },
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)
def dump_hiddens_job(smoke: bool = False, max_samples: int | None = None) -> int:
    import json
    import sys

    sys.path.insert(0, REPO_ROOT)
    from scripts.dflash.train import dump_hiddens

    cfg = _cfg(smoke, None, max_samples)
    cache_dir = Path(OUT_DIR) / "hidden-cache"
    manifest_path = cache_dir / "manifest.json"
    from huggingface_hub import HfApi

    # Resolve the exact target and dataset commits, so an upstream change at the
    # same repo id invalidates the cache. The effective target layer ids are a
    # pure function of the target config and num_target_layers, so the target
    # commit covers them too.
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    target_sha = api.model_info(cfg.target_model, revision=cfg.target_revision or None).sha
    dataset_sha = api.dataset_info(cfg.dataset_name).sha
    # Snapshot before dump_hiddens: load_target() mutates cfg from the target config.
    want = {
        "target_model": cfg.target_model,
        "target_sha": target_sha,
        "dataset_sha": dataset_sha,
        "dataset_name": cfg.dataset_name,
        "dataset_split": cfg.dataset_split,
        "max_seq_len": cfg.max_seq_len,
        "max_samples": cfg.max_samples,
        "num_target_layers": cfg.num_target_layers,
    }
    existing = sorted(cache_dir.glob("[0-9]*.pt"))
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        manifest = None
    if (
        isinstance(manifest, dict)
        and manifest.get("config") == want
        and manifest.get("n") == len(existing)
        and len(existing) >= 8
    ):
        print(f"cache already has {len(existing)} sequences matching config", flush=True)
        return len(existing)
    if existing or manifest is not None:
        print(f"hidden cache stale or mismatched (manifest={manifest}); rebuilding", flush=True)
        manifest_path.unlink(missing_ok=True)
        for p in existing:
            p.unlink()
    n = dump_hiddens(cfg, cache_dir, cfg.max_samples)
    manifest_path.write_text(json.dumps({"config": want, "n": n}, indent=2))
    out_vol.commit()
    hf_cache.commit()
    return n


@app.function(
    image=image,
    gpu=GPU,
    timeout=12 * 60 * 60,
    memory=131072,
    cpu=8.0,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={
        HF_CACHE: hf_cache,
        OUT_DIR: out_vol,
    },
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)
def train_dflash(
    smoke: bool = False,
    max_steps: int | None = None,
    max_samples: int | None = None,
    hub_repo: str = "",
) -> dict[str, Any]:

    cfg = _cfg(smoke, max_steps, max_samples)
    if hub_repo:
        cfg.hub_repo = hub_repo
    try:
        from scripts.dflash.train import train

        summary = train(cfg, Path(OUT_DIR))
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from None
    finally:
        out_vol.commit()
        hf_cache.commit()
    return summary


@app.function(
    image=image,
    timeout=2 * 60 * 60,
    cpu=4.0,
    memory=16384,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={OUT_DIR: out_vol},
)
def upload_job(hub_repo: str = "freakyskittle/Qwen3.8-27B-ABLITERATED-DFlash") -> dict[str, Any]:
    import os
    import sys

    sys.path.insert(0, REPO_ROOT)
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    from scripts.dflash.config import DFlashTrainConfig
    from scripts.dflash.export_gguf import export_dflash_gguf
    from scripts.dflash.model import DFlashDraftModel
    from scripts.dflash.train import push_to_hub

    out_dir = Path(OUT_DIR)
    files = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    print("volume files:", files[:80], flush=True)
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    me = api.whoami()
    print({"name": me.get("name"), "orgs": [o.get("name") if isinstance(o, dict) else o for o in (me.get("orgs") or [])]}, flush=True)
    try:
        info = api.repo_info(hub_repo, repo_type="model")
        print(f"repo exists private={info.private} id={info.id}", flush=True)
    except Exception as exc:
        print(f"repo_info {hub_repo}: {exc}", flush=True)
        try:
            api.create_repo(hub_repo, repo_type="model", private=True, exist_ok=True)
            print(f"created {hub_repo}", flush=True)
        except HfHubHTTPError as create_exc:
            print(f"create_repo failed: {create_exc}", flush=True)

    gguf = out_dir / "Qwen3.8-27B-ABLITERATED-DFlash-F16.gguf"
    final_ckpt = out_dir / "dflash_draft.pt"

    def _step_of(p: Path) -> int:
        suffix = p.stem[len("draft-step"):]
        return int(suffix) if suffix.isdigit() else -1

    step_ckpts = sorted(out_dir.glob("draft-step*.pt"), key=_step_of)
    ckpts = step_ckpts + ([final_ckpt] if final_ckpt.exists() else [])
    if not gguf.exists() and ckpts:
        import torch

        latest = final_ckpt if final_ckpt.exists() else step_ckpts[-1]
        print(f"exporting GGUF from {latest.name}", flush=True)
        # Checkpoints are {"cfg": plain dict, "draft": state_dict}; weights_only suffices.
        packed = torch.load(latest, map_location="cpu", weights_only=True)
        cfg = DFlashTrainConfig(**{k: v for k, v in packed["cfg"].items() if k in DFlashTrainConfig.__dataclass_fields__})
        draft = DFlashDraftModel(cfg)
        draft.load_state_dict(packed["draft"], strict=True)
        export_dflash_gguf(draft, cfg, gguf)
        out_vol.commit()
    if not gguf.exists() and not ckpts:
        raise RuntimeError("no GGUF or draft checkpoint on volume oxidize-dflash-qwen38 yet")
    cfg = DFlashTrainConfig(hub_repo=hub_repo)
    url = push_to_hub(out_dir, cfg, {"steps": "upload-only"})
    out_vol.commit()
    return {"hub_url": url, "files": files[:80]}


@app.local_entrypoint()
def main(smoke: bool = False, max_steps: int = 0, skip_tests: bool = False, upload_only: bool = False) -> None:
    if upload_only:
        print(upload_job.remote(), flush=True)
        return
    if not skip_tests:
        print(unit_tests.remote(), flush=True)
    steps = None if max_steps <= 0 else max_steps
    repo = hf_probe.remote()
    print(f"hub repo={repo}", flush=True)
    dumped = dump_hiddens_job.remote(smoke=smoke)
    print(f"hidden cache sequences={dumped}", flush=True)
    summary = train_dflash.remote(smoke=smoke, max_steps=steps, hub_repo=repo)
    print(summary, flush=True)
