# `oxidize-prune` Agent Notes

## What this crate does

`oxidize-prune` reads a GGUF file, optionally prunes linear weights, and writes a new GGUF. Three pruning methods are supported:

1. **`name-filter`** (legacy, default). Substring `keep` / `drop` pattern matching on tensor names. Bytes are copied verbatim — no weight-level work, fast even on 30 GB models.
2. **`wanda`** (Sun et al. 2023, ICLR 2024 — `arxiv:2306.11695`). Per-output-row pruning by `|W_ij| · ‖X_j‖_2`, where `‖X_j‖_2` is the per-input-neuron L2 norm of the calibration activations. One forward pass of calibration data, no weight update, no Hessian inverse. 300× faster than SparseGPT (`arxiv:2301.00774`) at the same perplexity.
3. **`magnitude`** (Han et al. 2015, with the per-output-row comparison group from Wanda Table 7). No calibration required.

## Public API surface

- `prune_gguf(PruneOptions) -> Result<PruneSummary>` (`gguf_copy.rs`) — name-filter path.
- `wanda_prune(WandaOptions) -> Result<PruneReport>` (`wanda.rs`) — Wanda.
- `magnitude_prune(WandaOptions) -> Result<PruneReport>` (`wanda.rs`) — magnitude.
- `magnitude_mask(weights, rows, cols, sparsity) -> Vec<bool>` (`mask.rs`).
- `wanda_mask(weights, norms, rows, cols, sparsity) -> Vec<bool>` (`mask.rs`).
- `apply_nm_pattern(mask, rows, cols, pattern, score_fn) -> Result<()>` (`mask.rs`).
- `load_l2_norms_cache(path) -> Result<BTreeMap<String, Vec<f32>>>` (`wanda.rs`).
- `write_l2_norms_cache(path, norms) -> Result<()>` (`wanda.rs`).
- `validate_calibration(cache, gguf_bytes) -> Result<()>` (`wanda.rs`).
- `SparsityPattern::{Unstructured, N2of4, N4of8}` (`mask.rs`).

## CLI

```text
oxidize-prune --input <model.gguf> --output <out.gguf>
              --method {name-filter|wanda|magnitude}   [default: name-filter]
              [--calibration <l2_norms.txt>]            (Wanda only)
              [--sparsity 0.5]                          (Wanda / magnitude)
              [--pattern {unstructured|n2of4|n4of8}]    (Wanda / magnitude)
              [--joint-quantize Q4_K_M]                 (Wanda / magnitude)
              [--keep-name <substring>]                 (repeatable, default: token_embd, output, rope, norm)
              [--dry-run]
              [--timing]                                (prints dequant/mask/requant ms)
```

## L2-norms cache format (for `--calibration`)

```text
# oxidize-prune L2 norms cache
# one row per linear weight tensor, N f32 values per row
blk.0.attn_q.weight 0.012 0.018 0.011 ...
blk.0.ffn_gate.weight 0.040 0.052 0.038 ...
```

One row per GGUF weight tensor name; N space-separated `f32` values, one per input column of the linear layer. The runner that produces this cache is described in `oxidize-core/src/compute/activation_stats.rs` and the layer-instrumented calibration forward is being added incrementally to `LayerWiseModel`.

## Reference papers

- Wanda: `arxiv:2306.11695` (Sun, Liu, Bair, Kolter — ICLR 2024)
- SparseGPT: `arxiv:2301.00774` (Frantar, Alistarh — ICML 2023)
- LLM.int8(): `arxiv:2208.07339` (Dettmers et al. — NeurIPS 2022)
- 50%-sparse OPT-175B runs at 0.21 PPL above dense on WikiText; 50%-sparse LLaMA-2-70B at 0.05 mean acc above dense (Wanda Table 3 / Table 26).
