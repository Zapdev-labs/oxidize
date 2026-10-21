# oxidize-quantize

**Generated:** 2026-06-03
**Domain:** Offline weight quantization utility (GGUF → GGUF with different q-types)

## OVERVIEW
Standalone binary for converting model weights between quantization formats. Supports all `GgufQuantizationType` variants. Can also append pre-encoded tensors to existing GGUF files without requantizing.

## STRUCTURE
```
oxidize-quantize/
├── Cargo.toml
└── src/
    └── main.rs    # Single-file tool: clap args, quantize loop, tensor append (620 lines)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add quantization type | `main.rs` + `oxidize-core/src/compute/quantization.rs` | Extend `parse_quantization_type()` match + core impl |
| Source/target format | `main.rs` | `--source` and `--target` clap args |
| Append tensor mode | `main.rs` | `--append-tensor name:path:dim0,dim1:type` |
| Quantization logic | `oxidize-core/src/compute/quantization.rs` | `quantize_scalar()`, `quantized_size()` |

## RUN
```bash
# Full requantization
sfw cargo run -p oxidize-quantize -- --input in.gguf --output out.gguf --source F32 --target Q4_0

# Append a pre-encoded tensor
sfw cargo run -p oxidize-quantize -- --input base.gguf --output out.gguf --append-tensor "layer.0.weight:layer0.bin:4096,4096:F32"
```

## SUPPORTED QUANTIZATION TYPES
F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K

## CONVENTIONS
- **Thin wrapper**: All heavy lifting is in `oxidize-core/src/compute/quantization.rs`. This crate is just CLI glue.
- **Validation**: Input and output paths are validated; source/target formats is checked against GGUF spec.
- **Progress**: Writes progress to stderr for large files.

## ANTI-PATTERNS
- `main.rs` at 620 lines for a simple converter — could be shorter with better abstractions.
- No dry-run mode to preview quantization size savings before writing.
- Error messages on unsupported q-type mismatch are not user-friendly.
- Does not validate that `--source` matches actual input tensor types — trust but verify flag.
