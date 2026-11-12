#!/usr/bin/env python3
"""Stage a Kaggle script kernel with secrets injected and push it.

Secrets come from the local environment (HF_TOKEN, OXFT_WEBHOOK) and are only
written into scripts/local/kaggle/<slug>/ (gitignored), never into tracked
files. Usage:

    HF_TOKEN=... OXFT_WEBHOOK=https://webhook.site/<uuid> \
      python3 scripts/kaggle_qwen38_ft/push_kernel.py probe.py otdoges/oxft-probe \
      [--extra file ...]

Extra files (e.g. the trainer module) are inlined, since script kernels only ship
a single file: `# @@INLINE <name>@@` is replaced by the file's source and the string
literal "@@B64 <name>@@" by the file's base64. Optional: OXFT_HF_REPO (private
adapter upload) and OXFT_EXTRA_ENV (JSON of OXFT_* overrides for train.py).
"""
import argparse, base64, json, os, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("slug", help="owner/kernel-slug")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = (HERE / a.script).read_text()
    for name in a.extra:
        body = (HERE / name).read_text()
        src = src.replace(f"# @@INLINE {name}@@", body)
        src = src.replace(f'"@@B64 {name}@@"', json.dumps(base64.b64encode(body.encode()).decode()))
    if "# @@INLINE" in src or '"@@B64 ' in src:
        sys.exit("unresolved INLINE/B64 marker; pass it via --extra")
    for env, marker in (("OXFT_HF_REPO", "__HF_REPO__"), ("OXFT_EXTRA_ENV", "__EXTRA_ENV__")):
        if os.environ.get(env):
            src = src.replace(f'"{marker}"', json.dumps(os.environ[env]))
    for env, marker in (("HF_TOKEN", "__HF_TOKEN__"), ("OXFT_WEBHOOK", "__WEBHOOK__")):
        val = os.environ.get(env)
        if not val:
            sys.exit(f"{env} not set")
        src = src.replace(f'"{marker}"', json.dumps(val))

    owner, slug = a.slug.split("/")
    out = ROOT / "scripts" / "local" / "kaggle" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.py").write_text(src)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": a.slug,
        "title": slug,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not a.no_gpu,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=1))
    if a.dry_run:
        print(out)
        return
    cmd = ["kaggle", "kernels", "push", "-p", str(out), "--accelerator", "NvidiaTeslaT4"]
    r = subprocess.run(cmd, text=True, capture_output=True)
    print(r.stdout, r.stderr)
    if r.returncode and "accelerator" in r.stderr:
        subprocess.run(cmd[:-2], check=True)
    elif r.returncode:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
