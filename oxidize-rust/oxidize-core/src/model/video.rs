//! CPU-first video model wrapper.
//!
//! The existing [`Model`](crate::model::Model) trait is text-token oriented, so
//! this wrapper keeps language generation compatible with the current runtime
//! while exposing explicit video encoding APIs. In practice a caller:
//!
//! 1. Decodes/samples/preprocesses RGB frames with [`encode_video_frames`].
//! 2. Inserts the returned video-token embeddings into a multimodal prompt.
//! 3. Continues normal token generation through the wrapped language model.

use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::video::{
    DecodedFrame, FrameSamplingStrategy, VideoConfig, VideoEncoder, VideoEncoderWorkspace,
    VideoError, VideoPreprocessor, luma_histogram_rgb, sample_indices, sample_indices_adaptive,
};

/// CPU video understanding wrapper around an existing language model.
pub struct VideoModel<M: Model> {
    text_model: M,
    encoder: VideoEncoder,
    preprocessor: VideoPreprocessor,
    workspace: VideoEncoderWorkspace,
}

impl<M: Model> VideoModel<M> {
    pub fn new(text_model: M, encoder: VideoEncoder) -> Self {
        let config = encoder.config().clone();
        Self {
            text_model,
            encoder,
            preprocessor: VideoPreprocessor::new(config.vision.clone()),
            workspace: VideoEncoderWorkspace::for_config(&config),
        }
    }

    pub fn config(&self) -> &VideoConfig {
        self.encoder.config()
    }

    pub fn text_model(&self) -> &M {
        &self.text_model
    }

    pub fn text_model_mut(&mut self) -> &mut M {
        &mut self.text_model
    }

    /// Sample and encode decoded RGB frames into video token embeddings.
    ///
    /// Returned layout is `[sampled_frames, llm_hidden_size]` row-major.
    pub fn encode_video_frames(&mut self, frames: &[DecodedFrame]) -> Result<Vec<f32>, VideoError> {
        if frames.is_empty() {
            return Err(VideoError::FrameCountOutOfRange {
                requested: 0,
                min: 1,
                max: self.config().temporal.max_frames,
            });
        }

        let indices = match self.config().sampling {
            FrameSamplingStrategy::Adaptive => {
                let mut hists = Vec::with_capacity(frames.len() * 16);
                for frame in frames {
                    hists.extend(luma_histogram_rgb(&frame.data, frame.width, frame.height));
                }
                sample_indices_adaptive(frames.len(), self.config().target_frames, &hists)?
            }
            strategy => sample_indices(frames.len(), self.config().target_frames, strategy)?,
        };
        let sampled: Vec<DecodedFrame> =
            indices.into_iter().map(|idx| frames[idx].clone()).collect();
        let preprocessed = self.preprocessor.preprocess(&sampled)?;
        self.encoder.encode(&preprocessed, &mut self.workspace)
    }
}

impl<M: Model> Model for VideoModel<M> {
    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError> {
        self.text_model.forward(tokens, session)
    }

    fn vocab_size(&self) -> usize {
        self.text_model.vocab_size()
    }

    fn context_size(&self) -> usize {
        self.text_model.context_size()
    }

    fn layer_count(&self) -> usize {
        self.text_model.layer_count()
    }

    fn forward_many(
        &mut self,
        tokens: &[Token],
        session: &mut Session,
    ) -> Result<Vec<Logits>, ModelError> {
        self.text_model.forward_many(tokens, session)
    }

    fn rewind_to(&mut self, consumed_tokens: usize) -> Result<(), ModelError> {
        self.text_model.rewind_to(consumed_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelError;
    use crate::video::{TemporalConfig, TemporalPool};
    use crate::vision::{VisionConfig, VisionEncoder};

    struct MockTextModel;

    impl Model for MockTextModel {
        fn forward(
            &mut self,
            tokens: &[Token],
            session: &mut Session,
        ) -> Result<Logits, ModelError> {
            if tokens.is_empty() {
                return Err(ModelError::EmptyInput);
            }
            session.record_tokens(tokens.len());
            Ok(vec![0.0, 1.0, 2.0])
        }

        fn vocab_size(&self) -> usize {
            3
        }
        fn context_size(&self) -> usize {
            16
        }
        fn layer_count(&self) -> usize {
            1
        }
    }

    fn tiny_config() -> VideoConfig {
        let vision = VisionConfig {
            image_size: 4,
            patch_size: 2,
            hidden_size: 4,
            num_attention_heads: 1,
            num_hidden_layers: 1,
            intermediate_size: 8,
            layer_norm_eps: 1e-5,
            projection_dim: 4,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            num_image_tokens: 4,
        };
        let temporal = TemporalConfig {
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 8,
            rms_norm_eps: 1e-5,
            max_frames: 4,
            rope_theta: 10000.0,
            use_cls_token: false,
            layer_dropout: 0.0,
        };
        VideoConfig {
            vision,
            temporal,
            sampling: FrameSamplingStrategy::Uniform,
            target_frames: 2,
            llm_hidden_size: 4,
            pool: TemporalPool::Mean,
            video_start_token_id: 0,
            video_end_token_id: 0,
        }
    }

    #[test]
    fn model_trait_delegates_to_text_model() {
        let cfg = tiny_config();
        let encoder =
            VideoEncoder::new(cfg.clone(), VisionEncoder::new(cfg.vision.clone())).unwrap();
        let mut model = VideoModel::new(MockTextModel, encoder);
        let mut session = Session::new();
        let logits = model.forward(&[1, 2], &mut session).unwrap();
        assert_eq!(logits, vec![0.0, 1.0, 2.0]);
        assert_eq!(session.consumed_tokens(), 2);
    }

    #[test]
    fn empty_video_is_rejected() {
        let cfg = tiny_config();
        let encoder =
            VideoEncoder::new(cfg.clone(), VisionEncoder::new(cfg.vision.clone())).unwrap();
        let mut model = VideoModel::new(MockTextModel, encoder);
        let err = model.encode_video_frames(&[]).unwrap_err();
        assert!(matches!(err, VideoError::FrameCountOutOfRange { .. }));
    }
}
