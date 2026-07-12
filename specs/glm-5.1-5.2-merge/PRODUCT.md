# GLM-5.1 × GLM-5.2 Merge — Product Spec

**Commit:** `6dc59873a6127453be67530497bf90e661585d70`  
**Host:** `ai@192.168.1.122` (password auth; ~16 TB free, no GPU)

## Summary

Produce a single BF16 SafeTensors checkpoint that SLERP-blends `zai-org/GLM-5.1` (A) with `zai-org/GLM-5.2` (B), then ships with GLM-5.2 tokenizer/config so the result loads as a GLM-5.2-family model (IndexShare + 1M context).

## Goals / Non-goals

**Goals**

- One merged BF16 checkpoint under `~/models/GLM-5.1-5.2-merged` on `.122`
- Shared tensors blended; 5.1-only IndexShare leftovers copied from A
- Runtime metadata from 5.2 (config, tokenizer, chat template)

**Non-goals**

- Merging FP8 variants (`GLM-5.1-FP8` / `GLM-5.2-FP8`)
- Cross-family merges (MiniMax ↔ GLM)
- GGUF quant / publish in this phase (follow-up)
- GPU finetune / distillation

## Behavior

1. **Inputs:** Local `~/models/GLM-5.1` (full BF16, 282 shards) and streamed `zai-org/GLM-5.2` shards (no full B tree required on disk).
2. **Blend:** For every tensor present in both models with matching dtype+shape, apply SLERP with category weights: attention `t=0.35`, MLP/expert `t=0.55`, other `t=0.45` (higher `t` → more 5.2).
3. **Missing tensors:** The 285 DSA indexer tensors that exist only in 5.1 (IndexShare removed them in 5.2) are copied from A unchanged. No tensors exist only in 5.2.
4. **Output:** Sharded SafeTensors (~1.5 TB) plus `config.json`, tokenizer files, and chat template copied from GLM-5.2.
5. **Dry-run first:** A dry-run reports tensor counts and exits without writing weights.
6. **Disk envelope:** Peak usage ≈ `|A| + one B shard + |output|` (~3 TB), not `|A|+|B|+|output|`.
7. **Failure:** Any shared-name dtype/shape mismatch aborts the merge; partial output is not published as complete.
8. **License:** Both parents are MIT; merged artifact inherits MIT.

## Open questions

1. After merge, should we immediately quantize to GGUF (AL5 / Q4_K) on `.122`, or stop at BF16?
2. Private HF upload target for the merged checkpoint (if any)?
