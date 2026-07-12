# oxidize-merge

**Domain:** SafeTensors checkpoint merging (linear / SLERP)

## OVERVIEW
Library + CLI that merges two HuggingFace SafeTensors checkpoints with linear or SLERP blending, per-tensor-category weights, and optional preset recipes. Handles sharded input/output and missing-tensor policies.

## STRUCTURE
```
oxidize-merge/
├── Cargo.toml
└── src/
    ├── main.rs      # clap Args -> MergeOptions/MergeRecipe -> merge_models()
    ├── lib.rs       # public API re-exports
    ├── merge.rs     # merge_models(), MergeOptions, MissingTensorPolicy
    ├── blend.rs     # linear + SLERP blend math
    ├── recipe.rs    # MergeMethod, MergeRecipe, category presets
    ├── index.rs     # SafeTensors shard index handling
    └── writer.rs    # sharded SafeTensors output
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Blend algorithm | `src/blend.rs` | linear / SLERP |
| Per-category weights | `src/recipe.rs` | attention vs MLP/expert `t` |
| Shard reading | `src/index.rs` | HF `model.safetensors.index.json` |
| Output writing | `src/writer.rs` | default max shard 5 GiB |

## CLI
```text
oxidize-merge --a <model> --b <model> --output <path>
              [--method linear|slerp]        (default: slerp)
              [--preset <name>] [--t <0..1>]
              [--attention-t 0.3] [--mlp-t 0.5] ...
```
`--a`/`--b` accept a `.safetensors` file or a HuggingFace model directory. `--output` is a `.safetensors` file or a directory for sharded output.

## BUILD / TEST / RUN
```bash
cargo build -p oxidize-merge
cargo test  -p oxidize-merge
cargo run   -p oxidize-merge -- --a modelA/ --b modelB/ --output merged/ --method slerp --t 0.5
```
