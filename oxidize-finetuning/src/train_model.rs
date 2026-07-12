//! Training backend: CPU layer-wise (default) or CUDA inference engine.

use oxidize_core::gguf::MappedGgufFile;
use oxidize_core::inference::{InferenceConfig, InferenceModel};
use oxidize_core::layer_wise::LayerWiseModel;
use oxidize_core::model::{Model, Session, Token};

use crate::error::{FinetuneError, Result};

pub enum TrainModel {
    Cpu(LayerWiseModel),
    #[cfg(feature = "cuda")]
    Gpu(GpuTrainModel),
}

pub struct GpuTrainModel {
    inner: InferenceModel,
    session: Session,
}

impl GpuTrainModel {
    pub fn load(mapped: &MappedGgufFile, config: InferenceConfig) -> Result<Self> {
        let inner = InferenceModel::load_from_gguf(mapped, config, true)
            .map_err(|e| FinetuneError::Model(e))?;
        Ok(Self {
            inner,
            session: Session::new(),
        })
    }

    pub fn config(&self) -> &InferenceConfig {
        self.inner.config()
    }

    pub fn warm_layer_cache(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn rewind_to(&mut self, consumed_tokens: usize) -> Result<()> {
        Model::rewind_to(&mut self.inner, consumed_tokens)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        self.session.rewind_to(consumed_tokens);
        Ok(())
    }

    pub fn forward_normed_hidden(&mut self, tokens: &[Token], start_pos: usize) -> Result<Vec<f32>> {
        if self.session.consumed_tokens() != start_pos {
            self.rewind_to(start_pos)?;
        }
        let out = self
            .inner
            .forward_normed_hidden_train(tokens, start_pos)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        self.session.record_tokens(tokens.len());
        Ok(out)
    }

    pub fn lm_head_logits_batch(
        &self,
        normed_all: &[f32],
        count: usize,
        logits_out: &mut [f32],
    ) -> Result<()> {
        self.inner
            .lm_head_logits_batch_from_normed(normed_all, count, logits_out)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))
    }
}

impl TrainModel {
    pub fn load_cpu(mapped: &MappedGgufFile, config: InferenceConfig, cache_slots: usize) -> Result<Self> {
        let model = LayerWiseModel::load_from_gguf(mapped, config, cache_slots)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        Ok(Self::Cpu(model))
    }

    #[cfg(feature = "cuda")]
    pub fn load_gpu(mapped: &MappedGgufFile, config: InferenceConfig) -> Result<Self> {
        Ok(Self::Gpu(GpuTrainModel::load(mapped, config)?))
    }

    pub fn config(&self) -> &InferenceConfig {
        match self {
            Self::Cpu(m) => m.config(),
            #[cfg(feature = "cuda")]
            Self::Gpu(m) => m.config(),
        }
    }

    pub fn warm_layer_cache(&mut self) -> Result<()> {
        match self {
            Self::Cpu(m) => m
                .warm_layer_cache()
                .map_err(|e| FinetuneError::Model(format!("{e:?}"))),
            #[cfg(feature = "cuda")]
            Self::Gpu(m) => m.warm_layer_cache(),
        }
    }

    pub fn rewind_to(&mut self, consumed_tokens: usize) -> Result<()> {
        match self {
            Self::Cpu(m) => m
                .rewind_to(consumed_tokens)
                .map_err(|e| FinetuneError::Model(format!("{e:?}"))),
            #[cfg(feature = "cuda")]
            Self::Gpu(m) => m.rewind_to(consumed_tokens),
        }
    }

    pub fn forward_normed_hidden(&mut self, tokens: &[Token], start_pos: usize) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(m) => m
                .forward_normed_hidden(tokens, start_pos)
                .map_err(|e| FinetuneError::Model(format!("{e:?}"))),
            #[cfg(feature = "cuda")]
            Self::Gpu(m) => m.forward_normed_hidden(tokens, start_pos),
        }
    }

    pub fn lm_head_logits_batch(
        &self,
        normed_all: &[f32],
        count: usize,
        logits_out: &mut [f32],
    ) -> Result<()> {
        match self {
            Self::Cpu(m) => m
                .lm_head_logits_batch(normed_all, count, logits_out)
                .map_err(|e| FinetuneError::Model(format!("{e:?}"))),
            #[cfg(feature = "cuda")]
            Self::Gpu(m) => m.lm_head_logits_batch(normed_all, count, logits_out),
        }
    }
}
