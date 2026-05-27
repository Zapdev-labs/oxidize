use super::config::{RGB_CHANNELS, RGBA_CHANNELS};
use super::{VisionConfig, VisionError};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct ImagePatches {
    pub data: Vec<f32>,
    pub num_patches: usize,
    pub patch_dim: usize,
    pub original_width: usize,
    pub original_height: usize,
}

impl ImagePatches {
    pub(crate) fn validate_for(&self, config: &VisionConfig) -> Result<(), VisionError> {
        check_shape(self.num_patches, config.num_patches())?;
        check_shape(self.patch_dim, config.patch_dim())?;
        check_shape(self.data.len(), config.num_patches() * config.patch_dim())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImagePreprocessor {
    config: VisionConfig,
}

impl ImagePreprocessor {
    pub fn new(config: VisionConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub fn preprocess_rgb(
        &self,
        image_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<ImagePatches, VisionError> {
        self.config.validate()?;
        validate_image_len(image_data.len(), width, height, RGB_CHANNELS)?;

        let resized = resize_rgb_nearest(
            image_data,
            width,
            height,
            self.config.image_size,
            self.config.image_size,
        )?;

        Ok(ImagePatches {
            data: patchify_rgb(&resized, &self.config),
            num_patches: self.config.num_patches(),
            patch_dim: self.config.patch_dim(),
            original_width: width,
            original_height: height,
        })
    }

    pub fn preprocess_from_rgba(
        &self,
        rgba_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<ImagePatches, VisionError> {
        validate_image_len(rgba_data.len(), width, height, RGBA_CHANNELS)?;

        let mut rgb = vec![0_u8; checked_image_len(width, height, RGB_CHANNELS)?];
        for (src, dst) in rgba_data
            .chunks_exact(RGBA_CHANNELS)
            .zip(rgb.chunks_exact_mut(RGB_CHANNELS))
        {
            dst.copy_from_slice(&src[..RGB_CHANNELS]);
        }

        self.preprocess_rgb(&rgb, width, height)
    }
}

fn check_shape(actual: usize, expected: usize) -> Result<(), VisionError> {
    if actual == expected {
        Ok(())
    } else {
        Err(VisionError::PatchShapeMismatch { expected, actual })
    }
}

fn checked_image_len(width: usize, height: usize, channels: usize) -> Result<usize, VisionError> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| VisionError::InvalidConfig("image dimensions overflow usize".into()))
}

fn validate_image_len(
    actual: usize,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<(), VisionError> {
    let expected = checked_image_len(width, height, channels)?;
    if actual != expected {
        return Err(VisionError::InvalidImageDimensions { expected, actual });
    }
    if width == 0 || height == 0 {
        return Err(VisionError::InvalidConfig(
            "image width and height must be non-zero".into(),
        ));
    }
    Ok(())
}

fn resize_rgb_nearest(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Result<Vec<u8>, VisionError> {
    validate_image_len(src.len(), src_w, src_h, RGB_CHANNELS)?;
    let mut dst = vec![0_u8; checked_image_len(dst_w, dst_h, RGB_CHANNELS)?];

    dst.par_chunks_exact_mut(RGB_CHANNELS)
        .enumerate()
        .for_each(|(dst_pixel, out)| {
            let dy = dst_pixel / dst_w;
            let dx = dst_pixel % dst_w;
            let sy = dy * src_h / dst_h;
            let sx = dx * src_w / dst_w;
            let src_idx = (sy * src_w + sx) * RGB_CHANNELS;
            out.copy_from_slice(&src[src_idx..src_idx + RGB_CHANNELS]);
        });

    Ok(dst)
}

fn patchify_rgb(resized: &[u8], config: &VisionConfig) -> Vec<f32> {
    let patch_dim = config.patch_dim();
    let mut patches = vec![0.0_f32; config.num_patches() * patch_dim];

    patches
        .par_chunks_exact_mut(patch_dim)
        .enumerate()
        .for_each(|(patch_idx, patch)| normalize_patch(resized, config, patch_idx, patch));

    patches
}

fn normalize_patch(resized: &[u8], config: &VisionConfig, patch_idx: usize, patch: &mut [f32]) {
    let patch_size = config.patch_size;
    let patch_y = patch_idx / config.num_patches_per_side();
    let patch_x = patch_idx % config.num_patches_per_side();

    for y_in_patch in 0..patch_size {
        let y = patch_y * patch_size + y_in_patch;
        for x_in_patch in 0..patch_size {
            let x = patch_x * patch_size + x_in_patch;
            let pixel_idx = (y * config.image_size + x) * RGB_CHANNELS;
            let patch_offset = (y_in_patch * patch_size + x_in_patch) * RGB_CHANNELS;

            for channel in 0..RGB_CHANNELS {
                let value = resized[pixel_idx + channel] as f32 / 255.0;
                patch[patch_offset + channel] =
                    (value - config.image_mean[channel]) / config.image_std[channel];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> VisionConfig {
        VisionConfig {
            image_size: 4,
            patch_size: 2,
            hidden_size: 4,
            num_attention_heads: 1,
            num_hidden_layers: 1,
            intermediate_size: 8,
            layer_norm_eps: 1e-5,
            projection_dim: 3,
            image_mean: [0.0; RGB_CHANNELS],
            image_std: [1.0; RGB_CHANNELS],
            num_image_tokens: 4,
        }
    }

    #[test]
    fn rejects_wrong_rgb_size() {
        let preprocessor = ImagePreprocessor::new(tiny_config());
        let result = preprocessor.preprocess_rgb(&[0_u8; 10], 2, 2);
        assert!(matches!(
            result,
            Err(VisionError::InvalidImageDimensions { .. })
        ));
    }

    #[test]
    fn creates_normalized_patches() {
        let config = tiny_config();
        let data = vec![128_u8; config.image_size * config.image_size * RGB_CHANNELS];
        let patches = ImagePreprocessor::new(config.clone())
            .preprocess_rgb(&data, config.image_size, config.image_size)
            .unwrap();

        assert_eq!(patches.num_patches, config.num_patches());
        assert_eq!(patches.patch_dim, config.patch_dim());
        assert!(
            patches
                .data
                .iter()
                .all(|value| (*value - 128.0 / 255.0).abs() < 1e-6)
        );
    }

    #[test]
    fn rgba_preprocessing_drops_alpha() {
        let config = tiny_config();
        let rgba = vec![64_u8; config.image_size * config.image_size * RGBA_CHANNELS];
        let patches = ImagePreprocessor::new(config.clone())
            .preprocess_from_rgba(&rgba, config.image_size, config.image_size)
            .unwrap();

        assert_eq!(patches.num_patches, config.num_patches());
        assert!(
            patches
                .data
                .iter()
                .all(|value| (*value - 64.0 / 255.0).abs() < 1e-6)
        );
    }

    #[test]
    fn resize_smaller_image() {
        let src = vec![255_u8; 4 * 4 * RGB_CHANNELS];
        let dst = resize_rgb_nearest(&src, 4, 4, 2, 2).unwrap();
        assert_eq!(dst.len(), 2 * 2 * RGB_CHANNELS);
        assert_eq!(dst[0], 255);
    }
}
