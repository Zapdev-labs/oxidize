# oxidize-golang

**Generated:** 2026-06-03
**Domain:** Go port of oxidize-core (feature parity work in progress)

## OVERVIEW
Pure-Go reimplementation of `oxidize-core` functionality. Ports GGUF parsing, tensor ops, quantization, model inference, tokenizers, and backends. Active development branch: `go/initial-oxidize-port`.

## STRUCTURE
```
oxidize-golang/
├── go.mod                       # Module: github.com/Zapdev-labs/oxidize/golang
├── cmd/                         # CLI entry points (to be added)
├── core/                        # Core library (ports oxidize-core/src/)
│   ├── tensor/                  # Tensor ops, dtype, GEMV
│   ├── simd/                    # SIMD wrappers (AVX-512, AVX2, NEON)
│   ├── quantization/            # Quantize/dequantize, k-quant types
│   ├── model/                   # Inference, DFlash speculative, sampling, generation
│   ├── flash_attention/         # Flash attention, ALiBi, sliding window
│   ├── kv_cache/                # KV cache management
│   ├── tokenizer/               # SP, WordPiece, BPE, Tiktoken
│   ├── ggufcore/                # GGUF v3 parser
│   ├── safetensors/             # SafeTensors format
│   ├── mesh/                    # Distributed inference mesh
│   ├── vision/                  # Vision encoder / multimodal
│   ├── turboquant/              # Block-wise INT4/INT8 for GEMV
│   ├── backends/                # Compute backend abstractions
│   └── workspace/               # Training workspace utilities
├── internal/                    # Private packages
│   ├── api/                     # OpenAI-compatible schema types
│   ├── auth/                    # API key authentication
│   ├── cli/                     # CLI command handling
│   ├── gguf/                    # GGUF writer, reader, metadata
│   ├── generate/                # Completion generation runtime
│   ├── server/                  # HTTP server (OpenAI-compatible)
│   ├── serviceinfo/             # Health/metrics/models endpoints
│   └── buildinfo/               # Compile-time feature detection
├── hf/                          # HuggingFace hub download client
└── scripts/                     # Manual QA test scripts
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Port GGUF parser | `core/ggufcore/` / `internal/gguf/` | Mirror Rust `gguf.rs` API |
| Port tensor ops | `core/tensor/` | Mirror `tensor.rs`, `cpu_kernels.rs`, `simd.rs` |
| Port quantization | `core/quantization/` | Mirror `quantization.rs`, types from `gguf.rs` |
| Port inference | `core/model/` | Mirror `inference.rs`, `generation.rs`, `sampling.rs` |
| Port DFlash | `core/model/dflash*.go` | Mirror `dflash.rs`, `speculative.rs` |
| Port tokenizer | `core/tokenizer/` | Mirror `tokenizer.rs` (4 formats) |
| Port server | `internal/server/` | Mirror `oxidize-server` Axum routes |
| Port mesh | `core/mesh/` | Mirror `mesh/` libp2p equivalent |
| Add backend | `core/backends/` | Implement Go `ComputeBackend` interface |

## RUN
```bash
# Build
cd oxidize-golang && go build ./...

# Test
cd oxidize-golang && go test ./...

# Run server
cd oxidize-golang && go run ./cmd/server/main.go
```

## CONVENTIONS
- **Mirror Rust structure**: Go packages map 1:1 to Rust modules where possible.
- **Interface for traits**: Rust traits become Go interfaces (e.g., `ComputeBackend`).
- **Error handling**: Use `fmt.Errorf("...: %w", err)` for error wrapping; avoid `panic()` in library code.
- **Test fixtures**: Reuse `oxidize-core/tests/fixtures/valid-v3.gguf` for cross-language parity tests.
- **Build info**: `internal/buildinfo/` mirrors Rust `XxxBuildInfo` pattern for feature detection.

## ANTI-PATTERNS
- Some Go files are placeholders — check for `TODO(port)` comments before relying on them.
- `core/model/` is very large — consider splitting `inference.go` into smaller files.
- No `go vet` / `golint` CI enforcement yet — add before merging to master.
- Error messages sometimes lose Rust-style context during port — maintain `cause` chain.

## SYNC NOTES
When `oxidize-core` master gets new features, bring them into `oxidize-golang` before expanding `oxidize-python`. DFlash speculative decoding is a current priority port target.
