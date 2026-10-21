# oxidize-core/src/vision

**Domain:** CPU image preprocessing + CLIP-style vision encoding for multimodal inference

## OVERVIEW
CPU-side image preprocessing and CLIP-style vision encoding that projects images into the language-model hidden space for multimodal (LLaVA / Qwen-VL) inference. Standalone/CPU-only — no backend abstraction. The encoder is a foundational/partial CLIP path (patch-embed → project; no transformer self-attention stack yet).

## STRUCTURE
```
vision/
├── mod.rs         # module root, re-exports, decode_image_bytes() stub
├── config.rs      # VisionConfig + presets + token math
├── preprocess.rs  # ImagePreprocessor / ImagePatches (resize + patchify + normalize)
├── encoder.rs     # VisionEncoder (patch embed → +pos → norm → projection)
├── prompt.rs      # MultimodalPrompt (interleave text + image embeddings)
└── error.rs       # VisionError enum
```

## KEY TYPES
| Type | Role |
|------|------|
| `VisionConfig` | Geometry + normalization; presets: `clip_large()` (default, 336/14, 576 tokens), `clip_base()`, `llava_1_5()`, `qwen_vl()` |
| `ImagePreprocessor` | `preprocess_rgb()` / `preprocess_from_rgba()` → `ImagePatches` (nearest-neighbor resize, rayon parallel normalize) |
| `VisionEncoder` | `load_weights(...)`, `encode(&ImagePatches) -> Vec<f32>`; output is `num_patches * projection_dim` |
| `MultimodalPrompt` | `add_text()`, `add_image()`, `build_sequence(...)` — interleaves image embeddings at segment boundaries |
| `count_image_tokens(image_size, patch_size)` | Free fn, panic-free token count |

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add vision preset | `config.rs` | Extend `VisionConfig` presets |
| Change resize/normalize | `preprocess.rs` | Currently nearest-neighbor |
| Encoder math | `encoder.rs` | Uses `crate::tensor::{gemm_f32, rms_norm_f32}` |
| Multimodal prompt assembly | `prompt.rs` | Feeds embeddings into LM via segment boundaries |
| Image codec support | `mod.rs` | `decode_image_bytes()` currently returns `UnsupportedFormat` |

## NOTES
- Post-layernorm uses `rms_norm_f32`; no transformer self-attention stack yet.
- For video multimodal, see `oxidize-core/src/video/AGENTS.md`.
