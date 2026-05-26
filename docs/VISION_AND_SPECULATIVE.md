# Vision and Speculative Decoding in Oxidize

This document describes the vision/multimodal and speculative decoding features available in the `oxidize` inference engine.

## Table of Contents

- [Overview](#overview)
- [Vision/Multimodal Support](#visionmultimodal-support)
  - [Architecture](#architecture)
  - [Image Preprocessing](#image-preprocessing)
  - [Vision Encoder](#vision-encoder)
  - [Multimodal Prompt Builder](#multimodal-prompt-builder)
  - [Supported Configurations](#supported-configurations)
  - [CLI Usage](#cli-usage)
  - [Server API Usage](#server-api-usage)
- [Speculative Decoding](#speculative-decoding)
  - [Architecture](#architecture-1)
  - [DFlash Draft Models](#dflash-draft-models)
  - [Token Acceptance and Rejection](#token-acceptance-and-rejection)
  - [Configuration](#configuration)
  - [Performance Monitoring](#performance-monitoring)
  - [CLI Usage](#cli-usage-1)
  - [Fallback Mode](#fallback-mode)
- [Examples](#examples)
  - [Vision Inference Example](#vision-inference-example)
  - [Speculative Decoding Example](#speculative-decoding-example)
  - [Combined Example](#combined-example)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Overview

`oxidize` supports two advanced inference features:

1. **Vision/Multimodal Inference**: Process images alongside text prompts using vision encoders (CLIP-style) that project image patches into the language model's embedding space.

2. **Speculative Decoding**: Accelerate inference by using a smaller, faster "draft" model (such as DFlash) to predict multiple tokens ahead, which are then verified by the full target model in parallel. Accepted tokens are committed; rejected tokens trigger resampling.

Both features can be used independently or together.

## Vision/Multimodal Support

### Architecture

The vision pipeline consists of three stages:

```
Image File → Preprocessor → Patch Embedding → Vision Encoder → Projection → LLM
```

1. **Image Preprocessor**: Loads raw image data, resizes to the target resolution, normalizes pixel values, and splits into non-overlapping patches.
2. **Patch Embedding**: Converts each patch into an embedding vector via learned projection.
3. **Vision Encoder**: Processes patch embeddings through transformer layers to produce contextualized image representations.
4. **Projection**: Maps the vision encoder output into the language model's embedding space.
5. **Multimodal Prompt Builder**: Interleaves text token embeddings with image embeddings to create a unified input sequence.

### Image Preprocessing

The `ImagePreprocessor` handles:

- **Resize**: Nearest-neighbor resizing to the model's expected input size (e.g., 224x224 for CLIP, 448x448 for Qwen-VL).
- **Normalization**: Scales pixel values from `[0, 255]` to `[0, 1]` and applies mean/std normalization using ImageNet statistics by default.
- **Patchification**: Splits the image into non-overlapping square patches (e.g., 14x14 pixels for CLIP-Large).

```rust
use oxidize_core::vision::{ImagePreprocessor, VisionConfig};

let config = VisionConfig::clip_large();
let preprocessor = ImagePreprocessor::new(&config);
let image_data = std::fs::read("image.jpg").unwrap();
let patches = preprocessor.preprocess(&image_data).unwrap();
```

### Vision Encoder

The `VisionEncoder` processes image patches through:

1. **Patch Embedding**: Linear projection of flattened patches.
2. **Positional Encoding**: Learned position embeddings added to patch embeddings.
3. **Transformer Layers**: Self-attention and feed-forward layers (simplified in current implementation; full attention requires weight loading infrastructure).
4. **Output Projection**: Maps to the language model's embedding dimension.

```rust
use oxidize_core::vision::{VisionEncoder, VisionConfig};

let config = VisionConfig::clip_large();
let encoder = VisionEncoder::new(&config);
let image_embeddings = encoder.encode(&patches).unwrap();
// image_embeddings shape: [num_patches, projection_dim]
```

### Multimodal Prompt Builder

The `MultimodalPrompt` builder interleaves text and image content:

```rust
use oxidize_core::vision::MultimodalPrompt;

let prompt = MultimodalPrompt::new()
    .add_text("Describe this image:")
    .add_image(image_embeddings)
    .add_text("What objects do you see?");

let sequence = prompt.build_sequence(tokenizer, embedding_table);
```

### Supported Configurations

Three preset configurations are provided:

| Configuration | Image Size | Patch Size | Hidden Dim | Projection Dim | Num Layers |
|--------------|------------|------------|------------|----------------|------------|
| `clip_large` | 224x224 | 14x14 | 1024 | 768 | 24 |
| `clip_base` | 224x224 | 16x16 | 768 | 512 | 12 |
| `qwen_vl` | 448x448 | 14x14 | 1024 | 4096 | 24 |

Custom configurations can be created via `VisionConfig::new(...)`.

### CLI Usage

Enable vision mode with the `--vision` flag and provide an image with `--image`:

```bash
# Single image inference
cargo run -p oxidize-cli -- \
  --model /path/to/multimodal-model.gguf \
  --vision \
  --image /path/to/image.jpg \
  --prompt "What is in this image?"

# Chat mode with vision
cargo run -p oxidize-cli -- \
  --model /path/to/multimodal-model.gguf \
  --chat \
  --vision
```

### Server API Usage

The chat completions API accepts an optional `images` field in messages:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "What do you see in this image?",
        "images": ["https://example.com/image.jpg"]
      }
    ]
  }'
```

Images can be provided as:
- URLs (http/https)
- Base64-encoded data URIs (`data:image/jpeg;base64,...`)
- Local file paths (when running server locally)

## Speculative Decoding

### Architecture

Speculative decoding uses two models:

1. **Target Model**: The full, high-quality model (slow but accurate).
2. **Draft Model**: A smaller, faster model (e.g., DFlash) that generates candidate tokens quickly.

The algorithm works as follows:

```
1. Draft model generates K candidate tokens autoregressively.
2. Target model evaluates all K positions in parallel (single forward pass).
3. For each position:
   - If target agrees with draft (probabilistic acceptance), commit the token.
   - If target disagrees, reject and resample from the residual distribution.
4. If all K tokens accepted, draft model generates K more.
5. If acceptance rate drops below threshold, fall back to direct target sampling.
```

### DFlash Draft Models

DFlash models are small, fast transformer models specifically designed for speculative decoding:

- Typically 1-2 layers (vs. 24-80 for full models)
- Shared vocabulary with the target model
- Optimized for single-token latency
- Can be loaded from GGUF files

```rust
use oxidize_core::speculative::{SpeculativeDecoder, SpeculativeConfig};
use oxidize_core::model_loader::GgufModelLoader;

// Load target model
let target = GgufModelLoader::new().load("target-model.gguf")?;

// Load draft model (DFlash)
let draft = GgufModelLoader::new().load("dflash-draft.gguf")?;

// Create speculative decoder
let config = SpeculativeConfig::aggressive();
let mut decoder = SpeculativeDecoder::new(target, draft, config)?;

// Generate with speculative decoding
let tokens = decoder.generate("Hello, world!", 100)?;
```

### Token Acceptance and Rejection

The acceptance criterion uses the ratio of target and draft probabilities:

```
acceptance_probability = min(1.0, target_prob / draft_prob)
```

If accepted, the draft token is committed. If rejected:

1. The position is rewound to before the rejected token.
2. A new token is sampled from the residual distribution: `residual = (target - draft).relu().normalize()`.
3. The draft model's KV cache is invalidated from the rejection point.

This guarantees that the output distribution matches the target model exactly (no quality loss).

### Configuration

Two preset configurations are provided:

#### Conservative (default)
- `draft_tokens`: 2
- `min_acceptance_rate`: 0.5
- `temperature`: 1.0
- Suitable when draft model quality is uncertain

#### Aggressive
- `draft_tokens`: 8
- `min_acceptance_rate`: 0.2
- `temperature`: 1.0
- Maximum speedup when draft model is well-aligned

Custom configuration via builder:

```rust
use oxidize_core::speculative::SpeculativeConfig;

let config = SpeculativeConfig::builder()
    .draft_tokens(4)
    .min_acceptance_rate(0.3)
    .temperature(0.8)
    .build();
```

### Performance Monitoring

The `SpeculativeStats` struct tracks:

- `drafted_tokens`: Total tokens generated by draft model
- `accepted_tokens`: Tokens accepted by target model
- `rejected_tokens`: Tokens rejected and resampled
- `fallback_steps`: Times fallback to direct sampling occurred
- `acceptance_rate`: Ratio of accepted to drafted tokens
- `speedup_estimate`: Estimated speedup vs. direct sampling

```rust
let stats = decoder.stats();
println!("Acceptance rate: {:.1}%", stats.acceptance_rate() * 100.0);
println!("Estimated speedup: {:.2}x", stats.speedup_estimate());
```

### CLI Usage

Enable speculative decoding with `--draft-model`:

```bash
# Basic speculative decoding
cargo run -p oxidize-cli -- \
  --model /path/to/target-model.gguf \
  --draft-model /path/to/dflash-draft.gguf \
  --prompt "Explain quantum computing"

# Adjust draft tokens per step
cargo run -p oxidize-cli -- \
  --model /path/to/target-model.gguf \
  --draft-model /path/to/dflash-draft.gguf \
  --draft-tokens 8 \
  --prompt "Write a story about robots"
```

### Fallback Mode

When the acceptance rate drops below `min_acceptance_rate` (default 0.3), the decoder automatically falls back to direct target model sampling for one step. This prevents wasting time on low-quality draft predictions.

Fallback is triggered when:
- Running acceptance rate over last 10 steps < threshold
- All K draft tokens rejected in a single step
- Draft model encounters an error

The decoder automatically resumes speculative decoding once acceptance rate recovers.

## Examples

### Vision Inference Example

```rust
use oxidize_core::vision::{ImagePreprocessor, VisionEncoder, VisionConfig, MultimodalPrompt};
use oxidize_core::model::{Model, Session};

// Load image
let image_data = std::fs::read("photo.jpg")?;

// Initialize vision pipeline
let config = VisionConfig::clip_large();
let preprocessor = ImagePreprocessor::new(&config);
let encoder = VisionEncoder::new(&config);

// Process image
let patches = preprocessor.preprocess(&image_data)?;
let image_embeddings = encoder.encode(&patches)?;

// Build multimodal prompt
let prompt = MultimodalPrompt::new()
    .add_text("Describe this image in detail:")
    .add_image(image_embeddings);

// Generate response
let mut session = Session::new();
let tokens = model.generate(&mut session, prompt.build_sequence(...)?, 200)?;
```

### Speculative Decoding Example

```rust
use oxidize_core::speculative::{SpeculativeDecoder, SpeculativeConfig};

// Load models
let target = load_model("llama-7b.gguf")?;
let draft = load_model("dflash-1l.gguf")?;

// Configure for aggressive speedup
let config = SpeculativeConfig::aggressive();
let mut decoder = SpeculativeDecoder::new(target, draft, config)?;

// Generate with monitoring
let tokens = decoder.generate("Once upon a time", 500)?;

// Print statistics
let stats = decoder.stats();
println!("Generated {} tokens", tokens.len());
println!("Drafted: {}, Accepted: {}, Rejected: {}",
    stats.drafted_tokens, stats.accepted_tokens, stats.rejected_tokens);
println!("Acceptance rate: {:.1}%", stats.acceptance_rate() * 100.0);
println!("Estimated speedup: {:.2}x", stats.speedup_estimate());
```

### Combined Example

```bash
# Vision + Speculative decoding together
cargo run -p oxidize-cli -- \
  --model /path/to/multimodal-model.gguf \
  --draft-model /path/to/dflash-draft.gguf \
  --vision \
  --image /path/to/image.jpg \
  --prompt "What is unusual about this image?"
```

## Performance Tuning

### Vision Tuning

1. **Image resolution**: Higher resolution = more patches = more compute. Use the lowest resolution that preserves task-relevant details.
2. **Patch size**: Smaller patches capture finer details but increase sequence length.
3. **Encoder layers**: If using a custom vision encoder, fewer layers are faster but may miss high-level features.

### Speculative Decoding Tuning

1. **Draft model selection**: The draft model should be 10-100x faster than the target per token. DFlash 1-2 layer models work well for 7B-70B targets.
2. **Draft tokens (`K`)**: Start with 4. Increase if acceptance rate > 70%; decrease if < 30%.
3. **Temperature**: Lower temperatures (0.5-0.8) often improve acceptance rates by making the draft model's predictions more deterministic.
4. **Fallback threshold**: Lower threshold = more aggressive speculative decoding. Raise if you see quality degradation.

Expected speedups:
- Well-aligned draft model: 1.5x - 2.5x
- Perfect alignment (synthetic): Up to 3x
- Poor alignment: 0.8x - 1.0x (fallback dominates)

## Troubleshooting

### Vision Issues

- **"Invalid image format"**: Ensure the image is a valid JPEG, PNG, or BMP. The preprocessor does not handle animated formats.
- **"Image too large"**: Resize images before preprocessing if they exceed the model's expected input size.
- **"Vision encoder not initialized"**: The vision encoder requires weight loading infrastructure not yet fully implemented. Use the simplified encoder for testing.

### Speculative Decoding Issues

- **"Draft model vocabulary mismatch"**: The draft and target models must share the same tokenizer/vocabulary. Use the `--tokenizer-model` flag to specify a shared tokenizer.
- **Low acceptance rate**: Try a larger draft model (more layers) or reduce `draft_tokens`. Ensure the draft model was trained on similar data.
- **Slower than direct sampling**: If acceptance rate < 20%, speculative decoding overhead exceeds gains. Use conservative config or disable.
- **"KV cache exhausted"**: Speculative decoding uses additional KV cache for draft tokens. Increase context size or reduce `draft_tokens`.

### General

- **Model path errors**: Use absolute paths or verify relative paths from the workspace root.
- **Memory errors**: Vision encoders and speculative decoding both increase memory usage. Monitor with `--profile perf`.
- **Build errors**: Ensure all workspace crates are built: `cargo build --workspace`.

## References

- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- LLaVA: Liu et al., "Visual Instruction Tuning" (2023)
- Speculative Decoding: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2022)
- DFlash: Zapdev Labs, "DFlash: Sub-Second Draft Models for Speculative Decoding" (2024)
