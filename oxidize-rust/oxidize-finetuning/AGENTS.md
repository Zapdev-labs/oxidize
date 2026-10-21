# oxidize-finetuning

**Domain:** CPU LoRA / SFT / self-train fine-tuning for GGUF models

## OVERVIEW
Rust CLI + library for LoRA / SFT / self-regressing fine-tuning of oxidize GGUF models on CPU, built on `oxidize-core` (`LayerWiseModel`, tokenizer, GGUF). The `sft`, `self-train`, and `merge` subcommands are fully wired; `dpo` and `ppo` are intentional stubs that `bail!` "not implemented".

## STRUCTURE
```
oxidize-finetuning/
├── Cargo.toml         # bin: oxidize-finetuning; deps: oxidize-core, clap, rayon, serde
└── src/
    ├── main.rs        # clap CLI: Sft / Dpo / Ppo / SelfTrain / Merge
    ├── lib.rs         # public API re-exports
    ├── trainer.rs     # SftTrainer (forward/backward, AdamW, packing)
    ├── self_train.rs  # SelfTrainLoop / SelfTrainConfig (iterative rounds)
    ├── lora.rs        # LoRAAdapter
    ├── merge.rs       # AdapterMerger / MergeStrategy
    ├── dataset.rs     # JSONL loaders (Alpaca/ShareGPT/{text}/{messages}), pack_chunks
    ├── export.rs      # export_lora_gguf, adapter manifest I/O
    ├── generate.rs    # self-dialogue synthetic data
    ├── fused.rs / qlora.rs / rlhf.rs / dpo.rs   # kernels + (partial) RL paths
    ├── config.rs / telemetry.rs / error.rs
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| SFT training loop | `src/trainer.rs` | `SftTrainer` |
| Self-train rounds | `src/self_train.rs` | per-round checkpoints, self-dialogue, self-critique |
| LoRA math | `src/lora.rs`, `src/fused.rs` | |
| Adapter merge/export | `src/merge.rs`, `src/export.rs` | GGUF adapter output |
| Dataset formats | `src/dataset.rs` | `load_jsonl_sft`, `load_jsonl_dpo`, `pack_chunks` |

## CLI
```text
oxidize-finetuning [--threads N] <subcommand>
  sft         --model <gguf> --dataset <jsonl> [--output lora-out --lora-rank 16
              --lora-alpha 32 --learning-rate 2e-4 --epochs 1 --max-seq-len 512 --window 64]
  self-train  --model <gguf> ... [--resume-from <ckpt>]
  merge       ...                # merge multiple LoRA adapter GGUFs
  dpo / ppo   ...                # stubs (not implemented)
```

## BUILD / TEST / RUN
```bash
cargo build -p oxidize-finetuning
cargo test  -p oxidize-finetuning
cargo run   -p oxidize-finetuning -- self-train --model base.gguf --dataset data.jsonl
```

## NOTES
- Build/test this crate individually: a pre-existing borrow-check error in `src/qlora.rs` currently breaks whole-workspace `cargo build --workspace`.
