# GLM-5.1 × GLM-5.2 Merge — Technical Spec

**Commit:** `6dc59873a6127453be67530497bf90e661585d70`  
**Remote:** `ai@192.168.1.122` (not `.121`)

## Context

`oxidize-merge` requires exact tensor name + dtype + shape to blend. GLM-5.1 and GLM-5.2 share architecture `GlmMoeDsaForCausalLM` / `glm_moe_dsa` with identical MoE/MLA dims:

| Field | 5.1 | 5.2 |
|-------|-----|-----|
| hidden / layers / vocab | 6144 / 78 / 154880 | same |
| MoE | 256 routed, top-8, shared=1 | same |
| MLA ranks | q_lora 2048, kv_lora 512 | same |
| Shared tensors | 59585 | 59585 (all match dtype+shape) |
| Only-in-5.1 | 285 indexer tensors | — |
| Only-in-5.2 | 0 | — |
| IndexShare | per-layer indexer | every 4 layers share |
| Context / rope_theta | ~200K / 1e6 | 1M / 8e6 |

Relevant code:

- [`scripts/glm_stream_merge.py`](https://github.com/Zapdev-labs/oxidize/blob/6dc59873a6127453be67530497bf90e661585d70/scripts/glm_stream_merge.py) — streamed SLERP (A resident, B one shard at a time)
- [`scripts/run_glm_merge_remote.sh`](https://github.com/Zapdev-labs/oxidize/blob/6dc59873a6127453be67530497bf90e661585d70/scripts/run_glm_merge_remote.sh) — download wait + merge + config copy
- [`oxidize-merge/src/merge.rs`](https://github.com/Zapdev-labs/oxidize/blob/6dc59873a6127453be67530497bf90e661585d70/oxidize-merge/src/merge.rs) — reference blend semantics (`validate_compatible`, `--missing a`)

Stream merge already implements `--missing a` behavior (copy A-only, blend intersection). Prefer it over `oxidize-merge` at this scale to avoid holding two full 1.5 TB trees.

## Proposed changes

No new Rust features required. Execution plan on `.122`:

### 1. Host prep

```bash
# on ai@192.168.1.122
mkdir -p ~/models ~/work ~/.venvs/merge
python3 -m venv ~/.venvs/merge
~/.venvs/merge/bin/pip install -U huggingface_hub safetensors numpy
cp ~/oxidize/scripts/glm_stream_merge.py ~/work/
cp ~/oxidize/scripts/run_glm_merge_remote.sh ~/work/
# ensure HF token: ~/.cache/huggingface/token (already present)
```

Update `run_glm_merge_remote.sh` host comments if any still say `.121`.

### 2. Download GLM-5.1 (~1.5 TB)

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
nohup hf download zai-org/GLM-5.1 --local-dir ~/models/GLM-5.1 \
  >> ~/work/glm51-download.log 2>&1 &
```

Wait until 282 `model-*.safetensors` shards are present.

### 3. Dry-run then merge

```bash
bash ~/work/run_glm_merge_remote.sh
# or manually:
python3 ~/work/glm_stream_merge.py \
  --a-dir ~/models/GLM-5.1 \
  --b-repo zai-org/GLM-5.2 \
  --b-cache ~/models/GLM-5.2-cache \
  --output ~/models/GLM-5.1-5.2-merged \
  --method slerp \
  --attention-t 0.35 --mlp-t 0.55 --other-t 0.45 \
  --dry-run

# then without --dry-run
```

### 4. Ship 5.2 runtime files

Script already copies `config.json`, tokenizer, `generation_config.json`, `chat_template.jinja` from `zai-org/GLM-5.2` into the output dir. Serving must use **5.2 config** (IndexShare + 1M RoPE). A-only indexer weights are inert under IndexShare.

### 5. Optional oxidize-merge path (if both trees fully local)

```bash
cargo run -p oxidize-merge --release -- \
  --a ~/models/GLM-5.1 \
  --b ~/models/GLM-5.2 \
  --output ~/models/GLM-5.1-5.2-merged-oxm \
  --method slerp \
  --attention-t 0.35 --mlp-t 0.55 --other-t 0.45 \
  --missing a
```

Requires ~4.5 TB free for A+B+out; not preferred on this box.

## Testing and validation

| PRODUCT invariant | Check |
|-------------------|--------|
| 1–2 Blend recipe | Dry-run tensor count ≈ 59585 merged; progress JSON shows `tensors_merged` |
| 3 Missing A-only | `copied_a` includes 285 indexer tensors; no crash |
| 4 Output + 5.2 config | `config.json` has `indexer_types` / 1M `max_position_embeddings`; index lists all tensors |
| 5 Dry-run | Exits 0 without writing weight shards |
| 6 Disk | `df` stays within ~3 TB delta during merge |
| 7 Mismatch abort | Injected shape mismatch in a test shard fails hard (unit-level on `glm_stream_merge` if added) |
| Smoke load | Load merged config + first shard header with `safetensors` / transformers (CPU, no full forward required on this host) |

## Parallelization

| Step | Parallel? | Notes |
|------|-----------|-------|
| Download 5.1 | Sequential (single `hf download`) | Dominates wall clock |
| Dry-run | After 5.1 complete | Fast |
| Stream merge | Sequential | B shards downloaded one-by-one |
| Config copy | After merge | Seconds |

No multi-agent fan-out: I/O-bound single-host job. Parent agent monitors logs on `.122`.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| ~3 TB disk during merge | Confirm ≥4 TB free before start; stream B |
| IndexShare + leftover 5.1 indexers | Serve with 5.2 config; extras unused |
| Network flake mid-download | `run_glm_merge_remote.sh` restarts incomplete `hf download` |
| No GPU on `.122` | BF16 merge is CPU/numpy; fine. Quant/bench later on `.132` or rented GPU |
| HF rate limits | Use existing token; resume downloads |

## Follow-ups

- Quant merged BF16 → GGUF (AL5 / Q4_K) via oxidize-c / convert path
- Private HF publish
- Optional smoke generation on a machine with enough RAM/GPU
