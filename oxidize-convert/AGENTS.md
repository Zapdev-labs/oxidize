# oxidize-convert

**Domain:** SafeTensors → GGUF conversion with optional pruning

## OVERVIEW
Rust CLI that converts HuggingFace SafeTensors (single file or model directory with `config.json`) to GGUF, with optional in-pass Wanda/magnitude pruning and joint quantization. Thin frontend over `oxidize_core::safetensors_to_gguf` and `oxidize-prune`.

## STRUCTURE
```
oxidize-convert/
├── Cargo.toml            # bin: oxidize-convert; deps: oxidize-core, oxidize-prune, clap, anyhow
└── src/
    ├── main.rs           # clap Args; 2-phase pipeline (convert, then optional prune)
    ├── run.rs            # ConvertOptions/ConvertSummary + convert()
    └── quantization.rs   # parse_target(): string -> GgufQuantizationType
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add CLI flag | `src/main.rs` | clap `Args` |
| Conversion logic | `oxidize-core/src/format/safetensors_to_gguf.rs` | Core lives in `oxidize-core` |
| Add quant target | `src/quantization.rs` | Map string to `GgufQuantizationType` |
| Prune-during-convert | `src/main.rs` (Phase 2) | Delegates to `oxidize-prune` |

## CLI
```text
oxidize-convert --input <safetensors|hf-dir> --output <out.gguf>
                [--arch <name>] [--config <config.json>] [--no-hf-names]
                [--target F32|F16|Q4_0|Q4_K_S|Q4_K_M|Q6_K|Q8_0]
                [--prune wanda|magnitude] [--prune-calibration <l2-norms>]
                [--prune-sparsity 0.5] [--prune-pattern unstructured|n2of4|n4of8]
                [--prune-joint-quantize <QTYPE>]
```
Phase 1 writes SafeTensors→GGUF. If `--prune` is set, Phase 2 writes an intermediate `<output>.prerun.gguf`, prunes to the final output, then removes the intermediate. Wanda requires `--prune-calibration`.

## BUILD / TEST / RUN
```bash
cargo build -p oxidize-convert
cargo test -p oxidize-convert
cargo run -p oxidize-convert -- --input model/ --output model.gguf --target Q4_K_M
```
