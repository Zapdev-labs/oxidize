//! Vision module for multimodal LLM inference.
//!
//! Provides image preprocessing, CLIP-style patch embedding, and vision-language
//! connector layers for architectures like LLaVA and Qwen-VL.
//!
//! # Architecture
//!
//! ```text
//! Image (H×W×3) → Resize → Normalize → Patchify → Vision Encoder → Projection → LLM
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxidize_core::vision::{ImagePreprocessor, VisionConfig, VisionEncoder};
//!
//! let config = VisionConfig::llava_1_5();
//! let preprocessor = ImagePreprocessor::new(config.image_size, config.patch_size);
//! let patches = preprocessor.preprocess_rgb(&image_data, width, height);
//! ```

use crate::tensor::{DType, gemm_f32, gemv_f32, rms_norm_f32};
use rayon::prelude::*;

/// Configuration for vision encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    /// Input image size (square, e.g., 336 for CLIP-Large).
    pub image_size: usize,
    /// Patch size (e.g., 14 for CLIP-Large).
    pub patch_size: usize,
    /// Hidden dimension of vision encoder (e.g., 1024 for CLIP-Large).
    pub hidden_size: usize,
    /// Number of attention heads in vision encoder.
    pub num_attention_heads: usize,
    /// Number of layers in vision encoder.
    pub num_hidden_layers: usize,
    /// Intermediate size for vision encoder FFN.
    pub intermediate_size: usize,
    /// Layer norm epsilon.
    pub layer_norm_eps: f32,
    /// Projection dimension to LLM hidden size.
    pub projection_dim: usize,
    /// Image mean for normalization (RGB).
    pub image_mean: [f32; 3],
    /// Image std for normalization (RGB).
    pub image_std: [f32; 3],
    /// Number of image tokens per image (precomputed: (image_size/patch_size)^2).
    pub num_image_tokens: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self::clip_large()
    }
}

impl VisionConfig {
    /// CLIP-Large configuration (used by LLaVA-1.5).
    pub fn clip_large() -> Self {
        let image_size = 336;
        let patch_size = 14;
        let num_patches = image_size / patch_size;
        Self {
            image_size,
            patch_size,
            hidden_size: 1024,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            intermediate_size: 4096,
            layer_norm_eps: 1e-5,
            projection_dim: 4096, // LLaVA-1.5 projects to LLM hidden size
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
            num_image_tokens: num_patches * num_patches,
        }
    }

    /// CLIP-Base configuration (used by smaller multimodal models).
    pub fn clip_base() -> Self {
        let image_size = 224;
        let patch_size = 14;
        let num_patches = image_size / patch_size;
        Self {
            image_size,
            patch_size,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            layer_norm_eps: 1e-5,
            projection_dim: 2048,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
            num_image_tokens: num_patches * num_patches,
        }
    }

    /// Qwen-VL configuration.
    pub fn qwen_vl() -> Self {
        let image_size = 448;
        let patch_size = 14;
        let num_patches = image_size / patch_size;
        Self {
            image_size,
            patch_size,
            hidden_size: 1664,
            num_attention_heads: 16,
            num_hidden_layers: 48,
            intermediate_size: 6656,
            layer_norm_eps: 1e-6,
            projection_dim: 4096,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
            num_image_tokens: num_patches * num_patches,
        }
    }

    /// Number of patches per side.
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of patches (image tokens).
    pub fn num_patches(&self) -> usize {
        self.num_patches_per_side().pow(2)
    }

    /// Patch dimension (channels × patch_size²).
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
}

/// Preprocessed image patches ready for vision encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePatches {
    /// Flattened patch embeddings [num_patches, patch_dim].
    pub data: Vec<f32>,
    /// Number of patches.
    pub num_patches: usize,
    /// Patch dimension.
    pub patch_dim: usize,
    /// Original image width.
    pub original_width: usize,
    /// Original image height.
    pub original_height: usize,
}

/// Image preprocessor: resize, normalize, and patchify.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePreprocessor {
    config: VisionConfig,
}

impl ImagePreprocessor {
    pub fn new(config: VisionConfig) -> Self {
        Self { config }
    }

    /// Preprocess RGB image data into normalized patches.
    ///
    /// `image_data` is row-major RGB: [R, G, B, R, G, B, ...].
    /// `width` and `height` are the original image dimensions.
    pub fn preprocess_rgb(
        &self,
        image_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<ImagePatches, VisionError> {
        if image_data.len() != width * height * 3 {
            return Err(VisionError::InvalidImageDimensions {
                expected: width * height * 3,
                actual: image_data.len(),
            });
        }

        // Resize to target image_size
        let resized = resize_rgb_nearest(
            image_data,
            width,
            height,
            self.config.image_size,
            self.config.image_size,
        );

        // Normalize and patchify
        let num_patches = self.config.num_patches();
        let patch_dim = self.config.patch_dim();
        let mut patches = vec![0.0_f32; num_patches * patch_dim];

        let patch_size = self.config.patch_size;
        let patches_per_side = self.config.num_patches_per_side();
        let img_size = self.config.image_size;

        for py in 0..patches_per_side {
            for px in 0..patches_per_side {
                let patch_idx = py * patches_per_side + px;
                let y_start = py * patch_size;
                let x_start = px * patch_size;

                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let y = y_start + dy;
                        let x = x_start + dx;
                        let pixel_idx = (y * img_size + x) * 3;

                        for c in 0..3 {
                            let val = resized[pixel_idx + c] as f32 / 255.0;
                            let normalized = (val - self.config.image_mean[c])
                                / self.config.image_std[c];
                            let elem_idx =
                                patch_idx * patch_dim + (dy * patch_size + dx) * 3 + c;
                            patches[elem_idx] = normalized;
                        }
                    }
                }
            }
        }

        Ok(ImagePatches {
            data: patches,
            num_patches,
            patch_dim,
            original_width: width,
            original_height: height,
        })
    }

    /// Preprocess from common image formats (PNG/JPEG decoded bytes).
    pub fn preprocess_from_rgba(
        &self,
        rgba_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<ImagePatches, VisionError> {
        if rgba_data.len() != width * height * 4 {
            return Err(VisionError::InvalidImageDimensions {
                expected: width * height * 4,
                actual: rgba_data.len(),
            });
        }

        // Convert RGBA to RGB
        let mut rgb = vec![0u8; width * height * 3];
        for i in 0..(width * height) {
            rgb[i * 3] = rgba_data[i * 4];
            rgb[i * 3 + 1] = rgba_data[i * 4 + 1];
            rgb[i * 3 + 2] = rgba_data[i * 4 + 2];
        }

        self.preprocess_rgb(&rgb, width, height)
    }
}

/// Simple nearest-neighbor resize for RGB images.
fn resize_rgb_nearest(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sy = (dy as f32 * y_ratio) as usize;
            let sx = (dx as f32 * x_ratio) as usize;
            let src_idx = (sy * src_w + sx) * 3;
            let dst_idx = (dy * dst_w + dx) * 3;
            dst[dst_idx] = src[src_idx];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        }
    }
    dst
}

/// Vision encoder: CLIP-style transformer for image patches.
///
/// This is a simplified implementation that can be extended with
/// actual weight loading from vision model checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionEncoder {
    config: VisionConfig,
    /// Patch embedding projection [patch_dim, hidden_size].
    patch_embedding: Vec<f32>,
    /// Class token [hidden_size].
    class_token: Vec<f32>,
    /// Position embeddings [num_patches + 1, hidden_size].
    position_embeddings: Vec<f32>,
    /// Post-encoder layer norm weight.
    post_layernorm_weight: Vec<f32>,
    /// Projection to LLM space [hidden_size, projection_dim].
    projection: Vec<f32>,
}

impl VisionEncoder {
    pub fn new(config: VisionConfig) -> Self {
        let patch_dim = config.patch_dim();
        let num_patches = config.num_patches();
        Self {
            config: config.clone(),
            patch_embedding: vec![0.0; patch_dim * config.hidden_size],
            class_token: vec![0.0; config.hidden_size],
            position_embeddings: vec![0.0; (num_patches + 1) * config.hidden_size],
            post_layernorm_weight: vec![1.0; config.hidden_size],
            projection: vec![0.0; config.hidden_size * config.projection_dim],
        }
    }

    /// Load weights from a checkpoint (placeholder for actual weight loading).
    pub fn load_weights(
        &mut self,
        patch_embedding: Vec<f32>,
        class_token: Vec<f32>,
        position_embeddings: Vec<f32>,
        post_layernorm_weight: Vec<f32>,
        projection: Vec<f32>,
    ) -> Result<(), VisionError> {
        let expected_patch_emb = self.config.patch_dim() * self.config.hidden_size;
        if patch_embedding.len() != expected_patch_emb {
            return Err(VisionError::WeightShapeMismatch {
                name: "patch_embedding",
                expected: expected_patch_emb,
                actual: patch_embedding.len(),
            });
        }
        self.patch_embedding = patch_embedding;
        self.class_token = class_token;
        self.position_embeddings = position_embeddings;
        self.post_layernorm_weight = post_layernorm_weight;
        self.projection = projection;
        Ok(())
    }

    /// Encode image patches into LLM-compatible embeddings.
    ///
    /// Returns [num_image_tokens, projection_dim] flattened row-major.
    pub fn encode(&self, patches: &ImagePatches) -> Result<Vec<f32>, VisionError> {
        let num_patches = patches.num_patches;
        let h = self.config.hidden_size;
        let patch_dim = patches.patch_dim;

        // 1. Patch embedding: [num_patches, patch_dim] × [patch_dim, h] → [num_patches, h]
        let mut embeddings = vec![0.0_f32; num_patches * h];
        gemm_f32(
            &patches.data,
            num_patches,
            patch_dim,
            &self.patch_embedding,
            h,
            &mut embeddings,
        )
        .map_err(|e| VisionError::InferenceFailed(format!("patch embedding: {:?}", e)))?;

        // 2. Add position embeddings (skip class token, use patch positions)
        for p in 0..num_patches {
            for i in 0..h {
                embeddings[p * h + i] += self.position_embeddings[(p + 1) * h + i];
            }
        }

        // 3. Apply transformer layers (simplified: just layer norm + projection)
        // In a full implementation, this would run N transformer layers with
        // self-attention and FFN.
        let mut normalized = vec![0.0_f32; num_patches * h];
        for p in 0..num_patches {
            rms_norm_f32(
                &embeddings[p * h..(p + 1) * h],
                &self.post_layernorm_weight,
                self.config.layer_norm_eps,
                &mut normalized[p * h..(p + 1) * h],
            )
            .map_err(|e| VisionError::InferenceFailed(format!("layer norm: {:?}", e)))?;
        }

        // 4. Project to LLM space: [num_patches, h] × [h, projection_dim]
        let proj_dim = self.config.projection_dim;
        let mut output = vec![0.0_f32; num_patches * proj_dim];
        gemm_f32(
            &normalized,
            num_patches,
            h,
            &self.projection,
            proj_dim,
            &mut output,
        )
        .map_err(|e| VisionError::InferenceFailed(format!("projection: {:?}", e)))?;

        Ok(output)
    }
}

/// Multimodal prompt builder: interleaves text and image tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalPrompt {
    /// Text segments (will be tokenized).
    pub text_segments: Vec<String>,
    /// Image patch embeddings (from VisionEncoder).
    pub image_embeddings: Vec<Vec<f32>>,
    /// Which segment index each image corresponds to (for ordering).
    pub image_positions: Vec<usize>,
}

impl MultimodalPrompt {
    pub fn new() -> Self {
        Self {
            text_segments: Vec::new(),
            image_embeddings: Vec::new(),
            image_positions: Vec::new(),
        }
    }

    /// Add a text segment.
    pub fn add_text(&mut self, text: impl Into<String>) {
        self.text_segments.push(text.into());
    }

    /// Add an image (embeddings from vision encoder).
    pub fn add_image(&mut self, embeddings: Vec<f32>, position: usize) {
        self.image_embeddings.push(embeddings);
        self.image_positions.push(position);
    }

    /// Build a flat sequence of embeddings interleaving text and images.
    ///
    /// Text embeddings are looked up from the tokenizer's embedding table.
    /// Image embeddings are inserted at the specified positions.
    pub fn build_sequence(
        &self,
        text_token_ids: Vec<Vec<u32>>,
        text_embedding_table: &[f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut sequence: Vec<f32> = Vec::new();
        let mut image_idx = 0;

        for (seg_idx, token_ids) in text_token_ids.iter().enumerate() {
            // Check if there's an image before this text segment
            while image_idx < self.image_positions.len()
                && self.image_positions[image_idx] == seg_idx
            {
                if let Some(img_emb) = self.image_embeddings.get(image_idx) {
                    sequence.extend_from_slice(img_emb);
                }
                image_idx += 1;
            }

            // Add text token embeddings
            for &token_id in token_ids {
                let idx = (token_id as usize).min(vocab_size - 1);
                let emb_start = idx * hidden_size;
                let emb_end = emb_start + hidden_size;
                if emb_end <= text_embedding_table.len() {
                    sequence.extend_from_slice(&text_embedding_table[emb_start..emb_end]);
                }
            }
        }

        // Add any remaining images at the end
        while image_idx < self.image_positions.len() {
            if let Some(img_emb) = self.image_embeddings.get(image_idx) {
                sequence.extend_from_slice(img_emb);
            }
            image_idx += 1;
        }

        sequence
    }
}

impl Default for MultimodalPrompt {
    fn default() -> Self {
        Self::new()
    }
}

/// Vision-related errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionError {
    InvalidImageDimensions { expected: usize, actual: usize },
    WeightShapeMismatch { name: &'static str, expected: usize, actual: usize },
    InferenceFailed(String),
    UnsupportedFormat(String),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidImageDimensions { expected, actual } => {
                write!(f, "invalid image dimensions: expected {} bytes, got {}", expected, actual)
            }
            Self::WeightShapeMismatch { name, expected, actual } => {
                write!(f, "weight shape mismatch for {}: expected {}, got {}", name, expected, actual)
            }
            Self::InferenceFailed(msg) => write!(f, "vision inference failed: {}", msg),
            Self::UnsupportedFormat(fmt) => write!(f, "unsupported image format: {}", fmt),
        }
    }
}

impl std::error::Error for VisionError {}

/// Utility: Convert a simple image format to RGB bytes.
///
/// For production use, integrate with `image` crate or similar.
pub fn decode_image_bytes(data: &[u8]) -> Result<(Vec<u8>, usize, usize), VisionError> {
    // Placeholder: assume raw RGB data for now
    // In production, detect format and decode PNG/JPEG/etc.
    Err(VisionError::UnsupportedFormat(
        "raw image decoding not implemented; provide RGB bytes directly".to_string(),
    ))
}

/// Compute the number of image tokens for a given vision config.
pub fn count_image_tokens(image_size: usize, patch_size: usize) -> usize {
    let patches_per_side = image_size / patch_size;
    patches_per_side * patches_per_side
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_config_clip_large() {
        let config = VisionConfig::clip_large();
        assert_eq!(config.image_size, 336);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.num_patches(), 24 * 24);
        assert_eq!(config.num_image_tokens, 576);
        assert_eq!(config.patch_dim(), 3 * 14 * 14);
    }

    #[test]
    fn vision_config_qwen_vl() {
        let config = VisionConfig::qwen_vl();
        assert_eq!(config.image_size, 448);
        assert_eq!(config.num_patches(), 32 * 32);
    }

    #[test]
    fn preprocessor_rejects_wrong_size() {
        let preprocessor = ImagePreprocessor::new(VisionConfig::clip_large());
        let result = preprocessor.preprocess_rgb(&[0u8; 100], 10, 10);
        assert!(matches!(result, Err(VisionError::InvalidImageDimensions { .. })));
    }

    #[test]
    fn preprocessor_creates_correct_patch_count() {
        let config = VisionConfig::clip_large();
        let preprocessor = ImagePreprocessor::new(config.clone());
        let img_size = config.image_size;
        let data = vec![128u8; img_size * img_size * 3];
        let patches = preprocessor.preprocess_rgb(&data, img_size, img_size).unwrap();
        assert_eq!(patches.num_patches, config.num_patches());
        assert_eq!(patches.patch_dim, config.patch_dim());
    }

    #[test]
    fn encoder_output_shape() {
        let config = VisionConfig::clip_large();
        let encoder = VisionEncoder::new(config.clone());
        let patches = ImagePatches {
            data: vec![0.5_f32; config.num_patches() * config.patch_dim()],
            num_patches: config.num_patches(),
            patch_dim: config.patch_dim(),
            original_width: 336,
            original_height: 336,
        };
        let output = encoder.encode(&patches).unwrap();
        assert_eq!(output.len(), config.num_patches() * config.projection_dim);
    }

    #[test]
    fn multimodal_prompt_builds_sequence() {
        let mut prompt = MultimodalPrompt::new();
        prompt.add_text("Describe this image:");
        prompt.add_image(vec![1.0_f32; 4096 * 576], 1);
        prompt.add_text("What do you see?");

        let text_tokens = vec![vec![1u32, 2, 3], vec![4u32, 5, 6]];
        let embedding_table = vec![0.1_f32; 32000 * 4096];
        let sequence = prompt.build_sequence(text_tokens, &embedding_table, 32000, 4096);

        // Should have: text1 embeddings + image embeddings + text2 embeddings
        let expected_len = 3 * 4096 + 576 * 4096 + 3 * 4096;
        assert_eq!(sequence.len(), expected_len);
    }

    #[test]
    fn count_image_tokens_matches_config() {
        let config = VisionConfig::clip_large();
        assert_eq!(
            count_image_tokens(config.image_size, config.patch_size),
            config.num_patches()
        );
    }

    #[test]
    fn resize_smaller_image() {
        let src = vec![255u8; 100 * 100 * 3]; // 100x100 white
        let dst = resize_rgb_nearest(&src, 100, 100, 50, 50);
        assert_eq!(dst.len(), 50 * 50 * 3);
        assert_eq!(dst[0], 255); // Should still be white
    }
}
