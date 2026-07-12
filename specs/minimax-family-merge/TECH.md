# MiniMax Family Merge — Technical Spec

**Commit:** `6dc59873a6127453be67530497bf90e661585d70`  
**Remote:** `ai@192.168.1.122`

## Context

oxidize-merge blends only when name, dtype, and shape match ([`oxidize-merge/src/merge.rs`](https://github.com/Zapdev-labs/oxidize/blob/6dc59873a6127453be67530497bf90e661585d70/oxidize-merge/src/merge.rs)). Non-blendable dtypes (including `F8_E4M3`) are copied from A ([`is_blendable`](https://github.com/Zapdev-labs/oxidize/blob/6dc59873a6127453be67530497bf90e661585d70/oxidize-merge/src/index.rs)).

### Compatibility matrix

| Pair | Arch match | Name ∩ | `--missing error` | Verdict |
|------|------------|--------|-------------------|---------|
| M2.5 × M3 | No | 0 | FAIL | **Impossible** |
| M2.7 × M3 | No | 0 | FAIL | **Impossible** |
| Community M2.75 × M3 | No | 0 | FAIL | **Impossible** |
| M2.5 × M2.7 | Yes | full | OK | **Possible** (FP8 mostly copy) |
| M3 × GLM-5.2 | No | ~0 | FAIL | **Impossible** |
| GLM-5.1 × GLM-5.2 | Yes | 59585 | need `--missing a` | See `specs/glm-5.1-5.2-merge/` |

### Why M2.x / M2.75 × M3 fails

| Check | M2.5 / M2.7 | M3 |
|-------|-------------|-----|
| `model_type` | `minimax_m2` | `minimax_m3_vl` |
| Prefix | `model.layers.*` | `language_model.model.layers.*` + vision |
| `hidden_size` | 3072 | 6144 |
| Layers | 62 | 60 |
| Experts / top-k | 256 / 8 (M2.75 community: 512 / 16) | 128 / 4 + shared |
| Dtype | mostly FP8 | mostly BF16 |
| Modality | text | text + image + video |

Even stripping prefixes, `lm_head.weight` alone still mismatches shape `[200064,3072]` vs `[200064,6144]`.

Repo IDs:

- M3: `MiniMaxAI/MiniMax-M3` (~869 GB BF16)
- M2.5: `MiniMaxAI/MiniMax-M2.5` (~466 GB)
- M2.7: `MiniMaxAI/MiniMax-M2.7` (~481 GB)
- “M2.75”: **no** `MiniMaxAI/MiniMax-M2.75`; community `selimaktas/MiniMax-M2.75-460B-A20B`

Neither MiniMax family is present on `.122` today.

## Proposed changes

### Path A — Do not merge M2.75 × M3 (default)

No download. Spec stands as the decision record. Prefer GLM-5.1×5.2 merge for “combine two strong open checkpoints.”

### Path B — Optional M2.5 × M2.7 (only if user confirms)

```bash
# on ai@192.168.1.122 — after GLM merge frees disk, or with ≥1.5 TB free
hf download MiniMaxAI/MiniMax-M2.5 --local-dir ~/models/MiniMax-M2.5
hf download MiniMaxAI/MiniMax-M2.7 --local-dir ~/models/MiniMax-M2.7

cargo run -p oxidize-merge --release --manifest-path ~/oxidize/Cargo.toml -- \
  --a ~/models/MiniMax-M2.5 \
  --b ~/models/MiniMax-M2.7 \
  --output ~/models/MiniMax-M2.5-2.7-merged \
  --method slerp --t 0.5 \
  --missing error \
  --dry-run
```

Expectation: FP8 expert weights copied from A; norms/embeds that are BF16/F32 SLERP. Copy tokenizer/config from the preferred parent (usually M2.7).

True float SLERP of the M2 line requires a separate dequant-to-BF16 step (not in oxidize-merge today).

**Update (2026-07-09):** implemented and executed via `oxidize-c/oc-merge` (C11
port of oxidize-merge, `make merge`), which adds FP8 block-aware blending the
Rust crate lacks: each `F8_E4M3` weight + F32 `*_scale_inv` (128×128 block)
pair is dequantized on both sides, SLERP'd in f32, and requantized against
fresh per-block `amax/448` scales — a true weight blend, not copy-from-A.
Dry-run on the real checkpoints: 96,103/96,103 tensors blend (47,864 FP8
pairs), 0 copied, full name intersection. Both models are FP8 (~215 GiB each,
not the 466/481 GB listed above). Output: `~/models/MiniMax-M2.6-merged` on
`.122`, config/tokenizer copied from M2.7. A C++ port (`oxidize-cpp-merge`,
Rust-parity, no FP8 path yet) also exists.

### Path C — Alternatives if “combine M3 + something” is the real goal

1. Distill / LoRA on M3 (or GLM) using the other as teacher — needs GPU (not on `.122`)
2. Dual-model routing at inference
3. Speculative decoding drafts (repo already has MiniMax-M3 EAGLE3 scripts) — not a weight merge

## Testing and validation

| Case | Check |
|------|--------|
| M2×M3 rejected | Dry-run / name-set script shows ∩ = 0; do not start multi-TB download |
| M2.5×M2.7 dry-run | `oxidize-merge --dry-run` (or equivalent) reports full name match |
| FP8 behavior | Merged FP8 tensors byte-identical to A; BF16 tensors differ from both parents |
| No false “M2.75 official” | Docs/scripts use official IDs only unless community repo is explicitly named |

## Parallelization

Not beneficial until Path B is approved. GLM merge on `.122` should own the disk first.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| User expects M2.75×M3 hybrid | Spec + fail-fast; offer Path B or distillation |
| FP8 “merge” looks like SLERP but isn’t | Document copy-from-A; optional BF16 dequant follow-up |
| Disk contention with GLM merge | Sequence: GLM first, MiniMax optional later |
