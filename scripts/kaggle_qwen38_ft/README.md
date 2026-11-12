# Qwen3.8-27B QLoRA on Kaggle 2x T4 in under 10 minutes

Spec and implementation for fine-tuning `Qwen/Qwen3.8-27B` on a free Kaggle T4x2 box,
with the process wall clock (start to adapter saved, model load included) under
600 s, and without degrading the base model.

## Constraints

| | |
|---|---|
| GPUs | 2x Tesla T4 (sm75): 14.56 GiB usable each, fp16 tensor cores only (no bf16), ~20 TFLOPS sustained fp16 GEMM |
| Host | 4 vCPU, ~31 GB RAM, ~95 GB writable outside `/kaggle/working` |
| Model | 64 layers = 16 x [3 x (Gated DeltaNet + MLP), (gated attention + MLP)], hidden 5120, FFN 17408, vocab 248320, untied lm_head |
| Weights | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` (NF4, 22.3 GB file): layers 15.2 GiB (GDN layer 0.257, attention layer 0.179), lm_head 2.37, embeddings 2.37 |

A 27B forward costs ~220 TFLOP per 4096 tokens, so two T4s can process only a few
hundred tokens per second. The 10-minute budget therefore buys roughly 150k
training tokens. Everything below is about making those tokens count and not
wasting GPU time.

## Design

**Base.** Pre-quantized NF4 checkpoint, fp16 compute. Only the text tower is loaded
(`Qwen3_5ForCausalLM` + `text_config`; the vision tower and MTP head are skipped).
Embeddings stay on the CPU (lookup only).

**Pipeline, "stream" mode (default).** A 4-bit model cannot be duplicated on 15 GB,
so the two GPUs form a 2-stage layer pipeline. Stage 0 (GPU0) holds a frozen prefix
and runs forward-only in its own thread. Stage 1 (GPU1) holds the LoRA layers and
the lm_head and does forward + backward. Stage 0 never waits on a backward, so
there is no pipeline bubble and every micro-batch can be its own optimizer step.
The split balances measured per-layer costs under GPU0's weight budget
(`plan_split`). With the real checkpoint it picks layer 49: GPU0 holds 49 frozen
layers (11.9 GiB) and GPU1 trains layers 49-63 (15 layers, 25.5M LoRA params).
`OXFT_MODE=1f1b` trains all 64 layers with a 1F1B schedule instead. That mode is
about 2x slower per token and has bubbles.

Measured on a T4, real dims, 4096 tokens per layer:

| per layer | ms |
|---|---|
| frozen forward (stage 0) | 220 |
| LoRA train, checkpointed (fwd + recompute + bwd) | 800 |
| LoRA train, no checkpoint | 560 (saves 1.95 GiB GDN / 1.16 GiB attn) |
| lm_head + CE per 1k supervised rows | 440 |

Stage 0 costs 49 x 220 ms = 10.8 s and stage 1 costs about 11.3 s per 4096-token
micro-batch, so the run does ~360 tok/s. A real-weight run on one T4 (20 real
layers with the same 15-layer LoRA stage) measured 12.4 s per step with both
stages serialized on one GPU, against a prediction of 13.3 s.

**Kernels.**
- `gdn_t4.py`: chunked Gated DeltaNet forward/backward for Turing. It has Triton
  chunk scans with quarter-split state tiles (no register spills), a fused
  `[B,T,H,D] -> chunks` + L2-norm kernel, per-row scalings folded into C x C
  matrices, and a torch-bmm backward scan where that is faster. At T=4096, H=48
  it does fwd 32 ms / fwd+bwd 115 ms with a 2.7 GiB peak. FLA on sm75 does
  96 / 301 ms, and the HF torch fallback does 61 / 707 ms. Gradients match an fp32
  recurrence to ~4e-7. On real weights, base NLL matches the HF reference kernel
  to 4 decimals. It is monkeypatched into `modeling_qwen3_5`. Triton JIT warm-up
  runs in a thread during the model load.
- `LoRA` (in `train.py`): fp32 master A/B, fp16 tensor-core compute. peft's bnb LoRA
  casts every activation to fp32 and runs SGEMMs, which took ~30% of a layer step.
- `ChunkedCE`: CE over assistant tokens only, in 256-row vocab chunks. The
  gradient is produced in the forward pass, so `[n, 248k]` fp32 logits are never
  kept.

**Data (2 sets, mixed by token share).**
- `HuggingFaceH4/no_robots`: 9.5k human-written instruction/chat examples (~305 tok).
- `NousResearch/hermes-function-calling-v1` (`func_calling_singleturn`): 1.9k
  examples (~970 tok). They are converted to Qwen-native `tools=` / `tool_calls`
  so the model's own chat template renders them (`OXFT_MIX_TOOLS=0.35`).

Training uses non-thinking mode, so the prompt is a strict prefix of the
conversation and only the final assistant turn is supervised. Batches are
length-bucketed and right-padded with no attention mask (every layer is causal,
so this is exact).

**Optimization.** LoRA r=16, alpha=32, on GDN in/out, attention q/k/v/o and MLP
projections of the trained layers. AdamW 8-bit at lr 2e-4, warmup + cosine over
the time remaining before the deadline (`OXFT_DEADLINE`, default 570 s since process
start), so the schedule adapts to whatever the box delivers. Datasets and
tokenization load in a thread while the model loads.

**Quality gates.** These run after the adapter is saved, so they are outside the
10-minute clock. On held-out sets of both datasets, assistant-token NLL is measured
with LoRA off and on, plus KL(base || tuned) on the renormalized base top-32.
Stage 0 is frozen, so its activations are computed once and reused for both passes.
The run passes if both NLL deltas are <= 0, chat KL < 0.15, and start-to-adapter
time < 600 s. The report goes to `oxft_report.json`.

**Output.** A PEFT adapter keyed for the full `Qwen3_5ForConditionalGeneration`
(`model.language_model.layers.*`). `check_adapter.py` verifies that stock PEFT
loads it and reproduces `train.py`'s forward.

## Files

| file | role |
|---|---|
| `train.py` | trainer (env knobs `OXFT_*`, see `Cfg`) |
| `gdn_t4.py`, `test_gdn_t4.py` | GDN kernel + gradient/speed tests vs FLA and HF |
| `kaggle_main.py` | Kaggle entry: download, pin stack, run `train.py`, optional private HF upload |
| `push_kernel.py` | stages the single-file kernel with secrets into gitignored `scripts/local/kaggle/` and pushes it |
| `bench_layer.py` | per-layer cost / memory bench used for the split planner |
| `make_tiny.py`, `check_adapter.py` | tiny NF4 checkpoint for smoke tests; adapter/PEFT parity check |
| `probe.py` | Kaggle env / throughput probe |

## Run

```bash
# Kaggle T4x2 (secrets come from the environment and never land in tracked files)
HF_TOKEN=... OXFT_WEBHOOK=https://webhook.site/<uuid> \
OXFT_HF_REPO=<user>/Qwen3.8-27B-oxft-t4x2-lora \
  python3 scripts/kaggle_qwen38_ft/push_kernel.py kaggle_main.py <kaggle-user>/oxft-qwen38-train \
  --extra gdn_t4.py train.py
kaggle kernels output <kaggle-user>/oxft-qwen38-train -p out/   # oxft_report.json, adapter, logs

# local smoke test on one GPU
python make_tiny.py /tmp/tiny-q38
OXFT_MODEL_DIR=/tmp/tiny-q38 OXFT_DEVICES=cuda:0 OXFT_OUT=/tmp/oxft OXFT_TRAIN_SECONDS=40 python train.py
python check_adapter.py /tmp/tiny-q38 /tmp/oxft/qwen38-27b-oxft-lora
# real weights, first N layers only, one GPU
OXFT_MODEL_DIR=/path/qwen38-bnb4 OXFT_TRUNC_LAYERS=20 OXFT_SPLIT=5 OXFT_DEVICES=cuda:0 python train.py
```

## Status

Validated on a single T4: kernel gradients and speed, tiny-model end to end in
both modes, PEFT adapter parity, real-weight GDN parity, and a real-weight stream
run with the full stage-1 workload. The full 64-layer 2-GPU run on Kaggle has not
run yet because the account's weekly Kaggle GPU quota is used up. The numbers
above for the full run (split 49, ~360 tok/s, ~150k tokens) are projections from
measured per-layer costs, not a completed run.
