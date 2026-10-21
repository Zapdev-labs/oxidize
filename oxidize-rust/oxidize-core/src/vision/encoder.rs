use super::{ImagePatches, VisionConfig, VisionError};
use crate::tensor::{gemm_f32, rms_norm_f32};

#[derive(Debug, Clone, PartialEq)]
pub struct VisionEncoder {
    config: VisionConfig,
    patch_embedding: Vec<f32>,
    class_token: Vec<f32>,
    position_embeddings: Vec<f32>,
    post_layernorm_weight: Vec<f32>,
    projection: Vec<f32>,
}

impl VisionEncoder {
    pub fn new(config: VisionConfig) -> Self {
        let patch_dim = config.patch_dim();
        let num_patches = config.num_patches();

        Self {
            patch_embedding: vec![0.0; patch_dim * config.hidden_size],
            class_token: vec![0.0; config.hidden_size],
            position_embeddings: vec![0.0; (num_patches + 1) * config.hidden_size],
            post_layernorm_weight: vec![1.0; config.hidden_size],
            projection: vec![0.0; config.hidden_size * config.projection_dim],
            config,
        }
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub fn load_weights(
        &mut self,
        patch_embedding: Vec<f32>,
        class_token: Vec<f32>,
        position_embeddings: Vec<f32>,
        post_layernorm_weight: Vec<f32>,
        projection: Vec<f32>,
    ) -> Result<(), VisionError> {
        self.check_weight(
            "patch_embedding",
            patch_embedding.len(),
            self.patch_embedding_len(),
        )?;
        self.check_weight("class_token", class_token.len(), self.config.hidden_size)?;
        self.check_weight(
            "position_embeddings",
            position_embeddings.len(),
            self.position_embeddings_len(),
        )?;
        self.check_weight(
            "post_layernorm_weight",
            post_layernorm_weight.len(),
            self.config.hidden_size,
        )?;
        self.check_weight("projection", projection.len(), self.projection_len())?;

        self.patch_embedding = patch_embedding;
        self.class_token = class_token;
        self.position_embeddings = position_embeddings;
        self.post_layernorm_weight = post_layernorm_weight;
        self.projection = projection;
        Ok(())
    }

    pub fn encode(&self, patches: &ImagePatches) -> Result<Vec<f32>, VisionError> {
        self.config.validate()?;
        self.validate_weights()?;
        patches.validate_for(&self.config)?;

        let hidden = self.config.hidden_size;
        let num_patches = self.config.num_patches();
        let mut patch_embeddings = vec![0.0_f32; num_patches * hidden];

        gemm_f32(
            &patches.data,
            num_patches,
            self.config.patch_dim(),
            &self.patch_embedding,
            hidden,
            &mut patch_embeddings,
        )
        .map_err(|err| VisionError::InferenceFailed(format!("patch embedding: {err:?}")))?;

        let mut tokens = vec![0.0_f32; (num_patches + 1) * hidden];
        tokens[..hidden].copy_from_slice(&self.class_token);
        tokens[hidden..].copy_from_slice(&patch_embeddings);
        add_position_embeddings(&mut tokens, &self.position_embeddings, hidden);

        let mut normalized = vec![0.0_f32; tokens.len()];
        for (src, dst) in tokens
            .chunks_exact(hidden)
            .zip(normalized.chunks_exact_mut(hidden))
        {
            rms_norm_f32(
                src,
                &self.post_layernorm_weight,
                self.config.layer_norm_eps,
                dst,
            )
            .map_err(|err| VisionError::InferenceFailed(format!("layer norm: {err:?}")))?;
        }

        let mut output = vec![0.0_f32; num_patches * self.config.projection_dim];
        gemm_f32(
            &normalized[hidden..],
            num_patches,
            hidden,
            &self.projection,
            self.config.projection_dim,
            &mut output,
        )
        .map_err(|err| VisionError::InferenceFailed(format!("projection: {err:?}")))?;

        Ok(output)
    }

    fn validate_weights(&self) -> Result<(), VisionError> {
        self.check_weight(
            "patch_embedding",
            self.patch_embedding.len(),
            self.patch_embedding_len(),
        )?;
        self.check_weight(
            "class_token",
            self.class_token.len(),
            self.config.hidden_size,
        )?;
        self.check_weight(
            "position_embeddings",
            self.position_embeddings.len(),
            self.position_embeddings_len(),
        )?;
        self.check_weight(
            "post_layernorm_weight",
            self.post_layernorm_weight.len(),
            self.config.hidden_size,
        )?;
        self.check_weight("projection", self.projection.len(), self.projection_len())
    }

    fn check_weight(
        &self,
        name: &'static str,
        actual: usize,
        expected: usize,
    ) -> Result<(), VisionError> {
        if actual == expected {
            Ok(())
        } else {
            Err(VisionError::WeightShapeMismatch {
                name,
                expected,
                actual,
            })
        }
    }

    fn patch_embedding_len(&self) -> usize {
        self.config.patch_dim() * self.config.hidden_size
    }

    fn position_embeddings_len(&self) -> usize {
        (self.config.num_patches() + 1) * self.config.hidden_size
    }

    fn projection_len(&self) -> usize {
        self.config.hidden_size * self.config.projection_dim
    }
}

fn add_position_embeddings(tokens: &mut [f32], positions: &[f32], hidden_size: usize) {
    for (token, position) in tokens
        .chunks_exact_mut(hidden_size)
        .zip(positions.chunks_exact(hidden_size))
    {
        for (value, pos) in token.iter_mut().zip(position) {
            *value += *pos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::config::RGB_CHANNELS;

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
    fn validates_loaded_weight_shapes() {
        let mut encoder = VisionEncoder::new(tiny_config());
        let err = encoder
            .load_weights(vec![0.0; 1], vec![], vec![], vec![], vec![])
            .unwrap_err();

        assert!(matches!(
            err,
            VisionError::WeightShapeMismatch {
                name: "patch_embedding",
                ..
            }
        ));
    }

    #[test]
    fn output_shape_matches_patch_projection() {
        let config = tiny_config();
        let encoder = VisionEncoder::new(config.clone());
        let patches = ImagePatches {
            data: vec![0.5_f32; config.num_patches() * config.patch_dim()],
            num_patches: config.num_patches(),
            patch_dim: config.patch_dim(),
            original_width: config.image_size,
            original_height: config.image_size,
        };

        let output = encoder.encode(&patches).unwrap();
        assert_eq!(output.len(), config.num_patches() * config.projection_dim);
    }
}
