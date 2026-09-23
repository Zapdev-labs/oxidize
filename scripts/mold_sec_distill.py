#!/usr/bin/env python3
"""Pull a defensive security SFT set and a general replay set, write flat JSONL.

The C finetune --strategy distill step turns this file into chat messages.
Reads HF_TOKEN from the environment. Does not print the token.
"""

import json
import os
import sys

from datasets import load_dataset


def row_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def emit(out, system, user, assistant, source):
    user = row_text(user)
    assistant = row_text(assistant)
    if not user or not assistant:
        return 0
    rec = {
        "system": row_text(system),
        "user": user,
        "assistant": assistant,
        "source": source,
    }
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 1


def load_security(limit):
    candidates = [
        ("ansulev/Cybersecurity-Dataset-Fenrir-v2.1", None),
        ("AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1", None),
    ]
    last = None
    for name, config in candidates:
        try:
            ds = load_dataset(name, config, split="train", streaming=True)
            return name, ds
        except Exception as exc:
            last = exc
            print(f"skip {name}: {exc}", file=sys.stderr)
    raise SystemExit(f"no security dataset loaded: {last}")


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "sec_distill_raw.jsonl"
    sec_limit = int(os.environ.get("SEC_LIMIT", "2000"))
    gen_limit = int(os.environ.get("GEN_LIMIT", "800"))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE")

    n_sec = 0
    n_gen = 0
    with open(out_path, "w", encoding="utf-8") as out:
        sec_name, sec = load_security(sec_limit)
        for row in sec:
            if n_sec >= sec_limit:
                break
            n_sec += emit(
                out,
                row.get("system") or "You are a defensive security assistant. Explain weaknesses and how to fix them. Refuse requests to build exploits or malware.",
                row.get("user") or row.get("instruction"),
                row.get("assistant") or row.get("output"),
                sec_name,
            )
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        for row in dolly:
            if n_gen >= gen_limit:
                break
            user = row_text(row.get("instruction"))
            context = row_text(row.get("context"))
            if context:
                user = user + "\n" + context
            n_gen += emit(out, "", user, row.get("response"), "databricks/databricks-dolly-15k")

    print(f"wrote {out_path} security={n_sec} general={n_gen}")
    if os.environ.get("HF_UPLOAD") != "1":
        return
    if not token:
        raise SystemExit("HF_TOKEN is required to upload")
    from huggingface_hub import HfApi

    repo = os.environ.get("HF_DATASET_REPO", "freakyskittle/sec-distill-sft")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo="raw.jsonl",
        repo_id=repo,
        repo_type="dataset",
    )
    card = """---
license: cc-by-sa-4.0
task_categories:
- text-generation
language:
- en
tags:
- security
- distillation
- defensive
size_categories:
- 1K<n<10K
---

# Defensive security distillation mix

Chat-ready traces for a small student model. Security rows come from a defensive instruction set. General rows are a replay slice of Dolly so the student keeps everyday instruction following while it absorbs security knowledge.

This is not a 27B weight checkpoint and it does not include exploit-construction tasks. The oxidize-c `finetune --strategy distill` command rewrites `raw.jsonl` into `messages` JSONL and drops requests that ask for weaponized output.

Sources: a Fenrir or Heimdall defensive cybersecurity set (Apache-2.0) and `databricks/databricks-dolly-15k` (CC BY-SA). The mix follows the stricter share-alike terms.
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"uploaded private dataset {repo}")


if __name__ == "__main__":
    main()
