use super::VisionError;

pub(crate) const RGB_CHANNELS: usize = 3;
pub(crate) const RGBA_CHANNELS: usize = 4;

const CLIP_IMAGE_MEAN: [f32; RGB_CHANNELS] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_IMAGE_STD: [f32; RGB_CHANNELS] = [0.26862954, 0.261_302_6, 0.275_777_1];

#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub projection_dim: usize,
    pub image_mean: [f32; RGB_CHANNELS],
    pub image_std: [f32; RGB_CHANNELS],
    pub num_image_tokens: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self::clip_large()
    }
}

impl VisionConfig {
    pub fn clip_large() -> Self {
        Self::with_clip_normalization(336, 14, 1024, 16, 24, 4096, 1e-5, 4096)
    }

    pub fn llava_1_5() -> Self {
        Self::clip_large()
    }

    pub fn clip_base() -> Self {
        Self::with_clip_normalization(224, 14, 768, 12, 12, 3072, 1e-5, 2048)
    }

    pub fn qwen_vl() -> Self {
        Self::with_clip_normalization(448, 14, 1664, 16, 48, 6656, 1e-6, 4096)
    }

    #[allow(clippy::too_many_arguments)]
    fn with_clip_normalization(
        image_size: usize,
        patch_size: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        num_hidden_layers: usize,
        intermediate_size: usize,
        layer_norm_eps: f32,
        projection_dim: usize,
    ) -> Self {
        Self {
            image_size,
            patch_size,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            intermediate_size,
            layer_norm_eps,
            projection_dim,
            image_mean: CLIP_IMAGE_MEAN,
            image_std: CLIP_IMAGE_STD,
            num_image_tokens: count_image_tokens(image_size, patch_size),
        }
    }

    pub fn num_patches_per_side(&self) -> usize {
        self.image_size.checked_div(self.patch_size).unwrap_or(0)
    }

    pub fn num_patches(&self) -> usize {
        let side = self.num_patches_per_side();
        side * side
    }

    pub fn patch_dim(&self) -> usize {
        RGB_CHANNELS * self.patch_size * self.patch_size
    }

    pub fn validate(&self) -> Result<(), VisionError> {
        if self.image_size == 0 {
            return invalid_config("image_size must be non-zero");
        }
        if self.patch_size == 0 {
            return invalid_config("patch_size must be non-zero");
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return invalid_config("image_size must be divisible by patch_size");
        }
        if self.hidden_size == 0 || self.projection_dim == 0 {
            return invalid_config("hidden_size and projection_dim must be non-zero");
        }
        if self.num_image_tokens != self.num_patches() {
            return Err(VisionError::InvalidConfig(format!(
                "num_image_tokens must be {}, got {}",
                self.num_patches(),
                self.num_image_tokens
            )));
        }
        if self.image_std.contains(&0.0) {
            return invalid_config("image_std entries must be non-zero");
        }
        Ok(())
    }
}

fn invalid_config<T>(message: &str) -> Result<T, VisionError> {
    Err(VisionError::InvalidConfig(message.into()))
}

pub fn count_image_tokens(image_size: usize, patch_size: usize) -> usize {
    if patch_size == 0 {
        return 0;
    }
    let patches_per_side = image_size / patch_size;
    patches_per_side * patches_per_side
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_large_shape_matches_llava() {
        let config = VisionConfig::clip_large();
        assert_eq!(config.image_size, 336);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.num_patches(), 24 * 24);
        assert_eq!(config.num_image_tokens, 576);
        assert_eq!(config.patch_dim(), 3 * 14 * 14);
        config.validate().unwrap();
    }

    #[test]
    fn qwen_vl_shape_matches_expected_grid() {
        let config = VisionConfig::qwen_vl();
        assert_eq!(config.image_size, 448);
        assert_eq!(config.num_patches(), 32 * 32);
        config.validate().unwrap();
    }

    #[test]
    fn zero_patch_size_has_zero_tokens_without_panicking() {
        assert_eq!(count_image_tokens(336, 0), 0);
    }
}
