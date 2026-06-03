# oxidize-core/src/format/

**Generated:** 2026-06-03
**Domain:** Model format parsers (GGUF, SafeTensors, tokenizers)

## OVERVIEW
Format parsing and serialization. GGUF v3 parser (`gguf.rs`), SafeTensors (`safetensors.rs`), and 4 tokenizer formats. Central type hub: `GgufQuantizationType` in `gguf.rs` (20+ cross-module refs).

## STRUCTURE
```
format/
├── gguf.rs         # GGUF v3 parser, metadata, tensor info, quantization types
├── safetensors.rs  # HuggingFace SafeTensors format support
├── tokenizer.rs    # 4 tokenizer formats: SP, WordPiece, BPE, Tiktoken
└── conversion.rs   # Format conversion utilities (GGUF ↔ SafeTensors)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add quantization type | `gguf.rs` | Extend `GgufQuantizationType` enum (central hub) |
| GGUF metadata read | `gguf.rs` | `GgufMetadataValue`, `GgufMetadataArray` |
| Tokenizer format | `tokenizer.rs` | `TokenizerFormat::{SentencePiece,WordPiece,Bpe,Tiktoken}` |
| SafeTensors mmap | `safetensors.rs` | Memory-mapped tensor loading |
| Format conversion | `conversion.rs` | `convert_gguf_to_safetensors()`, `convert_safetensors_to_gguf()` |
| Add new format | New `*.rs` + update `lib.rs` path | Follow `GgufFile` / `SafeTensorsFile` pattern |

## CONVENTIONS
- **Central type hub**: `GgufQuantizationType` is the single source of truth for all quantization format constants. 20+ modules reference it.
- **Mmap-first**: GGUF and SafeTensors both use memory-mapped I/O for large files.
- **Tokenizer factory**: `load_tokenizer_from_gguf_metadata()` dispatches to format-specific loaders.
- **Error per format**: `GgufError`, `SafeTensorsError`, `TokenizerError` each have `From` impls.

## ANTI-PATTERNS
- `GgufQuantizationType` is referenced everywhere but block-size constants are duplicated in `tensor.rs` and `quantization.rs` — should be unified.
- SafeTensors validator does not check tensor name collisions — add hash-based dedup.
- Tokenizer `load()` returns `Box<dyn>` without `Send` bound — limits async usage.

## GGUF FIXTURES
Tests reuse fixtures under `oxidize-core/tests/fixtures/` (e.g., `valid-v3.gguf`).
Go and Python ports also load these same fixtures for cross-language parity tests.
