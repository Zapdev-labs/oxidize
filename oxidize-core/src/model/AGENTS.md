# oxidize-core/src/model/

**Generated:** 2026-05-26
**Domain:** Inference engine, model loading, speculative decoding

## OVERVIEW
Inference engine directory: 5 `Model` trait implementations, speculative decoding, sampling strategies, and memory optimization.

## STRUCTURE
```
model/
├── model.rs          # Model trait, Session, Token, Logits types
├── inference.rs      # InferenceModel (main impl), ModelArchitecture enum (12 variants)
├── llama.rs          # LlamaConfig, LlamaArchitecture (legacy)
├── layer_wise.rs     # LayerWiseModel with LRU layer cache
├── mlx_inference.rs  # MlxInferenceModel (macOS-only, #[cfg gated])
├── dflash.rs         # DFlashDraftModel, DFlashConfig (speculative draft model)
├── speculative.rs    # SpeculativeDecoder, SpeculativeConfig (orchestrates draft+target)
├── generation.rs     # GenerationConfig, GenerationError, generation loop
├── sampling.rs       # SamplingConfig, GrammarConstraint, sample(), speculative_decode()
├── loader.rs         # ModelLoader trait, GgufModelLoader, BaselineGgufModel
├── offload.rs        # LayerOffloadPlan, MultiGpuConfig, ParallelismStrategy
├── lora.rs           # LoraPlan, AdapterKind, plan_lora_application()
├── prefix_cache.rs   # PrefixCache, CachedPrefix (KV cache for common prefixes)
└── advanced_features.rs # (placeholder / misc extensions)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add model architecture | `inference.rs` | Extend `ModelArchitecture` enum + `from_gguf()` match |
| Add sampling strategy | `sampling.rs` | ~1,355 lines, includes grammar constraints |
| Change generation loop | `generation.rs` | Handles streaming, stop sequences, prefill batching |
| Enable speculative decoding | `speculative.rs` + `dflash.rs` | Draft model loads from GGUF/SafeTensors |
| macOS acceleration | `mlx_inference.rs` | Entire file is `#[cfg(target_os = "macos")]` |
| Layer offloading | `offload.rs` | Tensor vs Pipeline parallelism strategies |
| LoRA adapter loading | `lora.rs` | Supports both LoRA and QLoRA |
| Prefix caching | `prefix_cache.rs` | KV cache snapshotting for repeated prompts |

## CONVENTIONS
- **Model trait is the boundary**: All inference paths implement `Model` (forward/forward_many/vocab_size/context_size/layer_count).
- **Config + Error per file**: Each model variant has its own `XxxConfig` and `XxxError`.
- **Session tracks token consumption**: `Session.consumed_tokens` is incremented by forward calls; use `rewind_to()` for speculative rollback.
- **Architecture detection from GGUF**: `ModelArchitecture::from_gguf()` is the single source of truth for mapping GGUF arch strings to enum variants.
- **MLX is fully gated**: `mlx_inference.rs` uses `#[cfg(target_os = "macos")]` on every item; no stub impls.

## ANTI-PATTERNS
- `dflash.rs` contains hardcoded debug logging to `/home/dih/oxidize/.cursor/debug-49b0b9.log` — should use tracing.
- `inference.rs` and `dflash.rs` duplicate similar tensor dequantization kernels — refactor candidate.
- `llama.rs` architecture enum shadows `inference.rs` `ModelArchitecture` — potential confusion.
- `layer_wise.rs` uses `HashMap<String, GgufTensorRef>` per layer — consider indexed arrays for cache locality.
