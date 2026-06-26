#!/usr/bin/env python3
"""Stream-merge two HF SafeTensors models when full B + output won't fit on disk.

Keeps model A resident. Downloads model B one shard at a time, blends tensors,
writes sharded output, then deletes each B shard after use.

Peak disk ≈ |A| + |one B shard| + |output| (fits ~3TB for GLM-5.x).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from huggingface_hub import hf_hub_download
    from safetensors import serialize
except ImportError:
    print("Install: pip install huggingface_hub safetensors numpy", file=sys.stderr)
    sys.exit(1)


DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.dtype(np.uint16),  # handled specially
}


def classify_tensor(name: str) -> str:
    lower = name.lower()
    attn = (
        "self_attn",
        ".attn.",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "query_proj",
        "key_proj",
        "value_proj",
    )
    if any(x in lower for x in attn):
        return "attention"
    mlp = (
        "mlp",
        "ffn",
        "feed_forward",
        "expert",
        "gate_proj",
        "up_proj",
        "down_proj",
        "w1",
        "w2",
        "w3",
    )
    if any(x in lower for x in mlp):
        return "mlp"
    return "other"


def t_for_tensor(name: str, attention_t: float, mlp_t: float, other_t: float) -> float:
    cat = classify_tensor(name)
    if cat == "attention":
        return attention_t
    if cat == "mlp":
        return mlp_t
    return other_t


def bf16_bytes_to_f32(data: bytes) -> np.ndarray:
    u16 = np.frombuffer(data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()



def read_tensor_bytes(path: Path, name: str) -> tuple[bytes, str, list[int]]:
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    if name not in header:
        raise KeyError(name)
    info = header[name]
    dtype = info["dtype"]
    shape = [int(x) for x in info["shape"]]
    start, end = info["data_offsets"]
    data_start = 8 + header_len
    with open(path, "rb") as f:
        f.seek(data_start + start)
        data = f.read(end - start)
    return data, dtype, shape


def linear_bytes(dtype: str, a: bytes, b: bytes, t: float) -> bytes:
    if dtype == "BF16":
        va = bf16_bytes_to_f32(a)
        vb = bf16_bytes_to_f32(b)
        out = (1.0 - t) * va + t * vb
        return f32_to_bf16_bytes(out)
    np_dtype = DTYPE_MAP.get(dtype)
    if np_dtype is None:
        return a
    va = np.frombuffer(a, dtype=np_dtype).astype(np.float64)
    vb = np.frombuffer(b, dtype=np_dtype).astype(np.float64)
    out = ((1.0 - t) * va + t * vb).astype(np_dtype)
    return out.tobytes()


def slerp_bytes(dtype: str, a: bytes, b: bytes, t: float) -> bytes:
    if dtype == "BF16":
        va = bf16_bytes_to_f32(a)
        vb = bf16_bytes_to_f32(b)
    else:
        np_dtype = DTYPE_MAP.get(dtype, np.float32)
        va = np.frombuffer(a, dtype=np_dtype).astype(np.float64)
        vb = np.frombuffer(b, dtype=np_dtype).astype(np.float64)
    dot = float(np.dot(va, vb))
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 and nb == 0.0:
        out = np.zeros_like(va)
    elif na == 0.0:
        out = vb
    elif nb == 0.0:
        out = va
    else:
        cos_theta = max(-1.0, min(1.0, dot / (na * nb)))
        theta = math.acos(cos_theta)
        if theta < 1e-8:
            out = (1.0 - t) * va + t * vb
        else:
            sin_theta = math.sin(theta)
            out = (
                math.sin((1.0 - t) * theta) / sin_theta * va
                + math.sin(t * theta) / sin_theta * vb
            )
    if dtype == "BF16":
        return f32_to_bf16_bytes(out.astype(np.float32))
    np_dtype = DTYPE_MAP.get(dtype, np.float32)
    return out.astype(np_dtype).tobytes()


def is_blendable(dtype: str) -> bool:
    return dtype in ("F32", "F16", "BF16")


def load_weight_index(model_dir: Path) -> dict[str, str]:
    for p in sorted(model_dir.glob("*.safetensors.index.json")):
        data = json.loads(p.read_text())
        return data["weight_map"]
    raise FileNotFoundError(f"no safetensors index in {model_dir}")


def shard_for_tensor(weight_map: dict[str, str], name: str) -> str | None:
    return weight_map.get(name)


def group_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        out[shard].append(name)
    for shard in out:
        out[shard].sort()
    return dict(out)


class ShardWriter:
    def __init__(self, out_dir: Path, max_shard_bytes: int, metadata: dict[str, str]):
        self.out_dir = out_dir
        self.max_shard_bytes = max_shard_bytes
        self.metadata = metadata
        self.current: list[tuple[str, bytes, str, list[int]]] = []
        self.current_bytes = 0
        self.shard_index = 0
        self.weight_map: dict[str, str] = {}

    def push(self, name: str, data: bytes, dtype: str, shape: list[int]) -> None:
        nbytes = len(data)
        if self.current and self.current_bytes + nbytes > self.max_shard_bytes:
            self.flush()
        self.current.append((name, data, dtype, shape))
        self.current_bytes += nbytes

    def flush(self) -> None:
        if not self.current:
            return
        shard_name = f"model-{self.shard_index:05d}-of-?????.safetensors"
        path = self.out_dir / shard_name
        tensor_dict: dict[str, dict[str, object]] = {}
        for name, data, dtype, shape in self.current:
            tensor_dict[name] = {"dtype": dtype, "shape": shape, "data": data}
        meta = self.metadata if self.shard_index == 0 else None
        path.write_bytes(serialize(tensor_dict, metadata=meta))
        for name, _, _, _ in self.current:
            self.weight_map[name] = shard_name
        self.shard_index += 1
        self.current.clear()
        self.current_bytes = 0

    def finish(self) -> None:
        self.flush()
        total = self.shard_index
        final_map: dict[str, str] = {}
        for name, shard in self.weight_map.items():
            new_shard = shard.replace("of-?????", f"of-{total:05d}")
            old = self.out_dir / shard
            new = self.out_dir / new_shard
            if old.exists() and old != new:
                old.rename(new)
            final_map[name] = new_shard
        index = {"metadata": self.metadata, "weight_map": final_map}
        (self.out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2)
        )


def download_b_shard(repo: str, shard: str, local_dir: Path) -> Path:
    path = hf_hub_download(
        repo_id=repo,
        filename=shard,
        local_dir=str(local_dir),
    )
    return Path(path)


def write_progress(path: str | None, payload: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.write_text(json.dumps({**payload, "updated_at": time.time()}))


def main() -> None:
    p = argparse.ArgumentParser(description="Stream-merge GLM SafeTensors checkpoints")
    p.add_argument("--a-dir", type=Path, required=True, help="Local GLM-5.1 directory")
    p.add_argument("--b-repo", default="zai-org/GLM-5.2")
    p.add_argument("--b-cache", type=Path, required=True, help="Temp dir for B shards")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("slerp", "linear"), default="slerp")
    p.add_argument("--attention-t", type=float, default=0.35)
    p.add_argument("--mlp-t", type=float, default=0.55)
    p.add_argument("--other-t", type=float, default=0.45)
    p.add_argument("--max-shard-gib", type=int, default=5)
    p.add_argument("--progress-file", type=Path, default=None, help="JSON progress for Grafana exporter")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    progress_file = str(args.progress_file) if args.progress_file else os.environ.get("GLM_MERGE_PROGRESS")

    a_map = load_weight_index(args.a_dir)
    # fetch B index only
    args.b_cache.mkdir(parents=True, exist_ok=True)
    idx_path = hf_hub_download(
        repo_id=args.b_repo,
        filename="model.safetensors.index.json",
        local_dir=str(args.b_cache),
    )
    b_map = json.loads(Path(idx_path).read_text())["weight_map"]
    b_by_shard = group_by_shard(b_map)

    all_names = sorted(set(a_map) | set(b_map))
    print(f"tensors: {len(all_names)}  B shards: {len(b_by_shard)}")

    if args.dry_run:
        merged = sum(
            1
            for n in all_names
            if n in a_map and n in b_map
        )
        print(f"dry-run: would merge {merged} tensors")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "oxidize-merge.method": args.method,
        "oxidize-merge.attention_t": str(args.attention_t),
        "oxidize-merge.mlp_t": str(args.mlp_t),
        "oxidize-merge.other_t": str(args.other_t),
        "oxidize-merge.model_a": str(args.a_dir),
        "oxidize-merge.model_b": args.b_repo,
        "oxidize-merge.streamed": "true",
    }
    writer = ShardWriter(args.output, args.max_shard_gib * 1024**3, meta)
    merged = copied_a = copied_b = 0
    write_progress(
        progress_file,
        {
            "phase": "merge",
            "complete": False,
            "b_shard_index": 0,
            "b_shards_total": len(b_by_shard),
            "tensors_merged": 0,
        },
    )

    # tensors only in A (no B shard download needed)
    b_only_shards = set(b_by_shard)
    for name in all_names:
        if name in a_map and name not in b_map:
            a_shard = args.a_dir / a_map[name]
            data, dtype, shape = read_tensor_bytes(a_shard, name)
            writer.push(name, data, dtype, shape)
            copied_a += 1

    for shard in sorted(b_by_shard):
        shard_idx = int(shard.split("-")[1]) if "-" in shard else 0
        print(f"B shard {shard} ({len(b_by_shard[shard])} tensors)", flush=True)
        write_progress(
            progress_file,
            {
                "phase": "merge",
                "complete": False,
                "b_shard": shard,
                "b_shard_index": shard_idx,
                "b_shards_total": len(b_by_shard),
                "tensors_merged": merged,
            },
        )
        b_path = download_b_shard(args.b_repo, shard, args.b_cache)
        for name in b_by_shard[shard]:
            data_b, dtype_b, shape_b = read_tensor_bytes(b_path, name)
            if name in a_map:
                a_shard = args.a_dir / a_map[name]
                data_a, dtype_a, shape_a = read_tensor_bytes(a_shard, name)
                if dtype_a != dtype_b or shape_a != shape_b:
                    raise ValueError(f"mismatch {name}: {dtype_a}/{shape_a} vs {dtype_b}/{shape_b}")
                if is_blendable(dtype_a):
                    t = t_for_tensor(name, args.attention_t, args.mlp_t, args.other_t)
                    blend = slerp_bytes if args.method == "slerp" else linear_bytes
                    out = blend(dtype_a, data_a, data_b, t)
                    merged += 1
                else:
                    out = data_a
                    copied_a += 1
                writer.push(name, out, dtype_a, shape_a)
            else:
                writer.push(name, data_b, dtype_b, shape_b)
                copied_b += 1
        b_path.unlink(missing_ok=True)
        # remove empty parent dirs huggingface_hub may create
        try:
            for root, dirs, files in os.walk(args.b_cache, topdown=False):
                if not dirs and not files and root != str(args.b_cache):
                    os.rmdir(root)
        except OSError:
            pass

    writer.finish()
    write_progress(
        progress_file,
        {
            "phase": "complete",
            "complete": True,
            "b_shard_index": len(b_by_shard),
            "b_shards_total": len(b_by_shard),
            "tensors_merged": merged,
            "copied_a": copied_a,
            "copied_b": copied_b,
        },
    )
    print(
        f"done: merged={merged} copied_a={copied_a} copied_b={copied_b} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
