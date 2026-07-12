# MiniMax Family Merge — Product Spec

**Commit:** `6dc59873a6127453be67530497bf90e661585d70`  
**Host:** `ai@192.168.1.122`

## Summary

Document which MiniMax pairs can be weight-merged with oxidize-merge / stream-merge, and which cannot. **MiniMax M2.75 × MiniMax-M3 cannot be merged.** Same-family M2.5 × M2.7 is the only official oxidize-merge candidate in this line (with FP8 caveats). Cross-family MiniMax × GLM remains out of scope.

## Goals / Non-goals

**Goals**

- Clear go/no-go for M2.75↔M3 and M2.5↔M2.7
- If a MiniMax merge is approved later, a recipe that matches oxidize-merge constraints

**Non-goals**

- Inventing a custom architecture remapper / layer transplant
- Merging M3 with any M2.x checkpoint
- Treating community `selimaktas/MiniMax-M2.75-460B-A20B` as an official MiniMax release

## Behavior

1. **M2.75 × M3 — rejected:** Official `MiniMaxAI/MiniMax-M2.75` does not exist. Community “M2.75” and official M2.5/M2.7 use `minimax_m2` (hidden 3072, 62 layers, text, mostly FP8). M3 uses `minimax_m3_vl` (hidden 6144, 60 layers, multimodal BF16). Tensor name intersection is empty; every `--missing` policy fails.
2. **M3 × GLM-5.x — rejected:** Different families (`minimax_m3_vl` vs `glm_moe_dsa`); see GLM merge spec for the GLM-only path.
3. **M2.5 × M2.7 — allowed (limited):** Same architecture and tensor names. oxidize-merge can run with `--missing error`. Most weights are `F8_E4M3` and are **copied from A**, not SLERP’d; only BF16/F32 tensors truly blend.
4. **Community M2.75:** Already an expert-injection merge of M2.5 experts onto M2.7. Do not re-merge it with M3; optional merge with M2.7 needs `--missing a` if A is the wider expert set.
5. **User-facing outcome if M2↔M3 is requested:** Fail fast with an architecture incompatibility message; do not download ~1.3 TB of weights “to try.”

## Open questions

1. Is the intended MiniMax pair actually **M2.5 × M2.7** (possible) rather than M2.75 × M3?
2. If yes, is FP8 copy-from-A + BF16 SLERP acceptable, or must we dequant to BF16 first (outside oxidize-merge)?
3. Should MiniMax work wait until the GLM-5.1×5.2 merge on `.122` finishes (disk contention)?
