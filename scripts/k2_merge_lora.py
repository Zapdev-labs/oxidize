#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into sharded K2 Horizon SafeTensors weights.

The merger preserves the source shard names and weight index. It materializes
one source shard at a time, so peak memory is bounded by one shard plus one
float32 LoRA update.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass(frozen=True)
class LoraPair:
    a: torch.Tensor
    b: torch.Tensor


def target_name(adapter_key: str) -> tuple[str, str] | None:
    prefix = "base_model.model."
    if not adapter_key.startswith(prefix):
        return None
    key = adapter_key[len(prefix) :]
    for kind in ("lora_A", "lora_B"):
        suffix = f".{kind}.weight"
        if key.endswith(suffix):
            return key[: -len(suffix)] + ".weight", kind
    return None


def load_adapter(path: Path) -> dict[str, LoraPair]:
    partial: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            mapped = target_name(key)
            if mapped is None:
                raise ValueError(f"unsupported adapter tensor key: {key}")
            target, kind = mapped
            partial.setdefault(target, {})[kind] = tensors.get_tensor(key)

    pairs: dict[str, LoraPair] = {}
    for target, values in partial.items():
        if set(values) != {"lora_A", "lora_B"}:
            raise ValueError(f"incomplete LoRA pair for {target}: {sorted(values)}")
        pairs[target] = LoraPair(values["lora_A"], values["lora_B"])
    return pairs


def copy_auxiliary_files(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if path.name.endswith(".safetensors") or path.name.endswith(
            ".safetensors.index.json"
        ):
            continue
        destination = output / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.adapter_config.read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scale = alpha / rank
    pairs = load_adapter(args.adapter)
    if not pairs:
        raise ValueError("adapter contains no LoRA tensors")

    index_path = args.base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    missing = sorted(set(pairs) - set(weight_map))
    if missing:
        raise ValueError(f"adapter targets missing from base index: {missing[:8]}")

    args.output.mkdir(parents=True, exist_ok=True)
    copy_auxiliary_files(args.base, args.output)
    consumed: set[str] = set()
    shard_names = sorted(set(weight_map.values()))
    merge_stats: list[dict[str, object]] = []

    for shard_index, shard_name in enumerate(shard_names, 1):
        source_path = args.base / shard_name
        destination_path = args.output / shard_name
        temporary_path = destination_path.with_suffix(".safetensors.tmp")
        output_tensors: dict[str, torch.Tensor] = {}

        with safe_open(source_path, framework="pt", device="cpu") as source:
            metadata = source.metadata()
            for name in source.keys():
                base = source.get_tensor(name)
                pair = pairs.get(name)
                if pair is None:
                    output_tensors[name] = base
                    continue

                if base.ndim != 2:
                    raise ValueError(f"LoRA target is not a matrix: {name} {base.shape}")
                if pair.a.shape[1] != base.shape[1]:
                    raise ValueError(
                        f"A/base input mismatch for {name}: {pair.a.shape} vs {base.shape}"
                    )
                if pair.b.shape[0] != base.shape[0]:
                    raise ValueError(
                        f"B/base output mismatch for {name}: {pair.b.shape} vs {base.shape}"
                    )
                if pair.a.shape[0] != pair.b.shape[1]:
                    raise ValueError(
                        f"LoRA rank mismatch for {name}: {pair.a.shape} vs {pair.b.shape}"
                    )

                delta = torch.matmul(pair.b.float(), pair.a.float()).mul_(scale)
                merged = base.float().add_(delta).to(base.dtype)
                if not torch.isfinite(merged.float()).all():
                    raise ValueError(f"non-finite merged tensor: {name}")
                output_tensors[name] = merged
                consumed.add(name)
                merge_stats.append(
                    {
                        "tensor": name,
                        "base_shape": list(base.shape),
                        "rank": pair.a.shape[0],
                        "delta_l2": float(torch.linalg.vector_norm(delta)),
                        "delta_max_abs": float(delta.abs().max()),
                    }
                )

            save_file(output_tensors, temporary_path, metadata=metadata)
        temporary_path.replace(destination_path)
        print(
            f"merged shard {shard_index}/{len(shard_names)}: {shard_name}",
            flush=True,
        )

    unconsumed = sorted(set(pairs) - consumed)
    if unconsumed:
        raise ValueError(f"adapter targets were not merged: {unconsumed[:8]}")
    if len(consumed) != len(pairs):
        raise AssertionError("adapter pair accounting failed")

    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    summary = {
        "base": str(args.base),
        "adapter": str(args.adapter),
        "rank": rank,
        "alpha": alpha,
        "scale": scale,
        "merged_tensor_count": len(consumed),
        "all_deltas_finite": all(
            math.isfinite(float(item["delta_l2"])) for item in merge_stats
        ),
        "tensors": merge_stats,
    }
    (args.output / "merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"merge complete: {len(consumed)} tensors across {len(shard_names)} shards",
        flush=True,
    )


if __name__ == "__main__":
    main()
