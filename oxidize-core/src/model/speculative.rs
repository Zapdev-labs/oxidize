//! Speculative decoding integration for oxidize.
//!
//! Provides end-to-end speculative decoding using DFlash draft models to accelerate
//! inference on full target models. The draft model generates candidate tokens which
//! are then verified by the target model in parallel.
//!
//! # Architecture
//!
//! ```text
//! Prompt → Target Model (prefill) → Draft generates K tokens → Target verifies K tokens
//!                                      ↑___________________________________________↓
//!                                           (accept/reject, update caches)
//! ```
//!
//! # Usage
//!
//! ```rust
//! use oxidize_core::speculative::{SpeculativeDecoder, SpeculativeConfig};
//! use oxidize_core::dflash::DFlashDraftModel;
//! use oxidize_core::model::Model;
//!
//! let config = SpeculativeConfig::default();
//! let mut decoder = SpeculativeDecoder::new(target_model, draft_model, config);
//! let tokens = decoder.generate(prompt_tokens, max_tokens)?;
//! ```

use crate::dflash::{DFlashConfig, DFlashDraftModel};
use crate::generation::{GenerationConfig, GenerationError};
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::sampling::{SamplingConfig, SamplingError, sample, speculative_decode};
use std::collections::VecDeque;

/// Configuration for speculative decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate per speculative step.
    pub draft_tokens_per_step: usize,
    /// Maximum total tokens to generate (including prompt).
    pub max_new_tokens: usize,
    /// Sampling configuration for both draft and target.
    pub sampling: SamplingConfig,
    /// Stop token ID (optional).
    pub stop_token: Option<Token>,
    /// Whether to use strict mode (reject on first mismatch) or lenient mode.
    pub strict_mode: bool,
    /// Minimum acceptance rate before falling back to greedy decoding.
    pub min_acceptance_rate: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_tokens_per_step: 4,
            max_new_tokens: 128,
            sampling: SamplingConfig::default(),
            stop_token: None,
            strict_mode: false,
            min_acceptance_rate: 0.3,
        }
    }
}

impl SpeculativeConfig {
    /// Conservative config: fewer draft tokens, higher quality.
    pub fn conservative() -> Self {
        Self {
            draft_tokens_per_step: 2,
            max_new_tokens: 128,
            sampling: SamplingConfig {
                temperature: 0.8,
                top_p: Some(0.95),
                ..Default::default()
            },
            stop_token: None,
            strict_mode: true,
            min_acceptance_rate: 0.5,
        }
    }

    /// Aggressive config: more draft tokens, faster but potentially more waste.
    pub fn aggressive() -> Self {
        Self {
            draft_tokens_per_step: 8,
            max_new_tokens: 256,
            sampling: SamplingConfig {
                temperature: 1.0,
                ..Default::default()
            },
            stop_token: None,
            strict_mode: false,
            min_acceptance_rate: 0.2,
        }
    }
}

/// Statistics for speculative decoding performance monitoring.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpeculativeStats {
    /// Total number of draft tokens generated.
    pub total_draft_tokens: usize,
    /// Total number of draft tokens accepted by target.
    pub accepted_draft_tokens: usize,
    /// Total number of target model forward passes.
    pub target_forward_passes: usize,
    /// Total number of draft model forward passes.
    pub draft_forward_passes: usize,
    /// Number of fallback tokens (sampled from target without draft).
    pub fallback_tokens: usize,
}

impl SpeculativeStats {
    /// Acceptance rate: accepted / total draft tokens.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.accepted_draft_tokens as f32 / self.total_draft_tokens as f32
    }

    /// Average accepted tokens per target forward pass.
    pub fn tokens_per_target_forward(&self) -> f32 {
        if self.target_forward_passes == 0 {
            return 0.0;
        }
        (self.accepted_draft_tokens + self.fallback_tokens) as f32
            / self.target_forward_passes as f32
    }

    /// Speedup estimate: (accepted + fallback) / target_forward_passes.
    /// Ideal speedup is draft_tokens_per_step + 1.
    pub fn estimated_speedup(&self) -> f32 {
        if self.target_forward_passes == 0 {
            return 1.0;
        }
        (self.accepted_draft_tokens + self.fallback_tokens) as f32
            / self.target_forward_passes as f32
    }
}

/// Speculative decoder that uses a DFlash draft model to accelerate target model inference.
pub struct SpeculativeDecoder<'a, T: Model> {
    target_model: &'a mut T,
    draft_model: &'a mut DFlashDraftModel,
    config: SpeculativeConfig,
    stats: SpeculativeStats,
    /// Buffer for emitted tokens waiting to be returned.
    emit_buffer: VecDeque<Token>,
    /// Recent tokens for repetition penalty.
    recent_tokens: Vec<Token>,
    /// Current generation state.
    state: DecoderState,
    /// Target model session for KV cache.
    target_session: Session,
    /// Whether the last token needs KV cache update in target.
    last_token_pending_kv: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DecoderState {
    Prefill,
    Speculating,
    Fallback,
    Done,
}

impl<'a, T: Model> SpeculativeDecoder<'a, T> {
    /// Create a new speculative decoder.
    pub fn new(
        target_model: &'a mut T,
        draft_model: &'a mut DFlashDraftModel,
        config: SpeculativeConfig,
    ) -> Self {
        Self {
            target_model,
            draft_model,
            config,
            stats: SpeculativeStats::default(),
            emit_buffer: VecDeque::with_capacity(16),
            recent_tokens: Vec::with_capacity(256),
            state: DecoderState::Prefill,
            target_session: Session::new(),
            last_token_pending_kv: false,
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Generate tokens from a prompt using speculative decoding.
    ///
    /// Returns all generated tokens (not including the prompt).
    pub fn generate(&mut self, prompt_tokens: &[Token]) -> Result<Vec<Token>, SpeculativeError> {
        let mut generated = Vec::with_capacity(self.config.max_new_tokens);

        // Prefill: run prompt through target model
        if !prompt_tokens.is_empty() {
            self.prefill(prompt_tokens)?;
        }

        // Generate tokens
        while generated.len() < self.config.max_new_tokens {
            match self.generate_one_token()? {
                Some(token) => {
                    if self.config.stop_token == Some(token) {
                        break;
                    }
                    generated.push(token);
                }
                None => break,
            }
        }

        Ok(generated)
    }

    /// Generate a single token, using speculative decoding when possible.
    fn generate_one_token(&mut self) -> Result<Option<Token>, SpeculativeError> {
        // Emit buffered tokens first
        if let Some(token) = self.emit_buffer.pop_front() {
            return Ok(Some(token));
        }

        if matches!(self.state, DecoderState::Done) {
            return Ok(None);
        }

        match self.state {
            DecoderState::Prefill => {
                // After prefill, switch to speculative mode
                self.state = DecoderState::Speculating;
                self.run_speculative_step()
            }
            DecoderState::Speculating => {
                if self.stats.acceptance_rate() < self.config.min_acceptance_rate {
                    // Fall back to direct target sampling if acceptance rate is too low
                    self.state = DecoderState::Fallback;
                    self.run_fallback_step()
                } else {
                    self.run_speculative_step()
                }
            }
            DecoderState::Fallback => {
                if self.stats.acceptance_rate() >= self.config.min_acceptance_rate {
                    // Return to speculative mode
                    self.state = DecoderState::Speculating;
                    self.run_speculative_step()
                } else {
                    self.run_fallback_step()
                }
            }
            DecoderState::Done => Ok(None),
        }
    }

    /// Run the prompt through the target model to initialize KV caches.
    fn prefill(&mut self, prompt_tokens: &[Token]) -> Result<(), SpeculativeError> {
        // Run target model prefill
        let batch_size = 256;
        for chunk in prompt_tokens.chunks(batch_size) {
            self.target_model
                .forward(chunk, &mut self.target_session)
                .map_err(SpeculativeError::Model)?;
        }

        // Get the last token's logits and sample
        let last_logits = self
            .target_model
            .forward(
                &[prompt_tokens[prompt_tokens.len() - 1]],
                &mut self.target_session,
            )
            .map_err(SpeculativeError::Model)?;

        let token = sample(&last_logits, self.config.sampling, fastrand::f32())
            .map_err(SpeculativeError::Sampling)?;

        self.recent_tokens.push(token);
        self.emit_buffer.push_back(token);
        self.stats.fallback_tokens += 1;
        self.stats.target_forward_passes += 1;

        // Initialize draft model with the same context
        self.draft_model.reset_cache();
        for &token in prompt_tokens {
            let _ = self.draft_model.forward_token(token, None);
        }

        Ok(())
    }

    /// Run one speculative decoding step.
    ///
    /// 1. Draft model generates K tokens autoregressively.
    /// 2. Target model verifies all K+1 positions in parallel (batched).
    /// 3. Accept/reject based on probability ratios.
    fn run_speculative_step(&mut self) -> Result<Option<Token>, SpeculativeError> {
        let k = self.config.draft_tokens_per_step;
        let start_token = *self
            .recent_tokens
            .last()
            .ok_or(SpeculativeError::NoContext)?;

        // 1. Draft model generates K tokens
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_logits = Vec::with_capacity(k);
        let mut current_token = start_token;

        for _ in 0..k {
            let hidden = self
                .draft_model
                .forward_token(current_token, None)
                .map_err(|e| SpeculativeError::DraftModel(e))?;
            let logits = self
                .draft_model
                .logits(&hidden)
                .map_err(|e| SpeculativeError::DraftModel(e))?;

            let token = sample(&logits, self.config.sampling, fastrand::f32())
                .map_err(SpeculativeError::Sampling)?;

            draft_tokens.push(token);
            draft_logits.push(logits);
            current_token = token;

            self.stats.draft_forward_passes += 1;
        }

        self.stats.total_draft_tokens += k;

        // 2. Target model verifies: get logits for each draft position
        let mut target_logits = Vec::with_capacity(k + 1);

        // First position: start token
        let verify_start = self.target_session.consumed_tokens();
        let start_logits = self
            .target_model
            .forward(
                &[if self.last_token_pending_kv {
                    start_token
                } else {
                    // Start token is already in cache, just get next logits
                    start_token
                }],
                &mut self.target_session,
            )
            .map_err(SpeculativeError::Model)?;
        target_logits.push(start_logits);

        // Subsequent positions: verify each draft token
        let mut verify_sequence = vec![start_token];
        verify_sequence.extend_from_slice(&draft_tokens);

        for i in 1..=k {
            // Rewind to verification start to ensure clean state
            self.target_model
                .rewind_to(verify_start)
                .map_err(SpeculativeError::Model)?;
            self.target_session.rewind_to(verify_start);

            let logits = self
                .target_model
                .forward(&verify_sequence[..i + 1], &mut self.target_session)
                .map_err(SpeculativeError::Model)?;
            target_logits.push(logits);
        }

        self.stats.target_forward_passes += 1;

        // 3. Speculative decode: accept/reject
        let randoms: Vec<f32> = (0..=k).map(|_| fastrand::f32()).collect();

        let result = speculative_decode(
            &draft_tokens,
            &draft_logits,
            &target_logits[1..], // Skip start token logits
            self.config.sampling,
            &randoms,
        )
        .map_err(SpeculativeError::Sampling)?;

        // 4. Update target model with accepted sequence
        let accepted_count = result.accepted_draft_tokens;
        self.stats.accepted_draft_tokens += accepted_count;

        // Build accepted sequence
        let mut accepted_tokens = Vec::new();
        if self.last_token_pending_kv {
            accepted_tokens.push(start_token);
        }
        accepted_tokens.extend_from_slice(&result.tokens);

        // Update target model KV cache with accepted tokens
        if !accepted_tokens.is_empty() {
            self.target_model
                .rewind_to(verify_start)
                .map_err(SpeculativeError::Model)?;
            self.target_session.rewind_to(verify_start);
            self.target_model
                .forward(&accepted_tokens, &mut self.target_session)
                .map_err(SpeculativeError::Model)?;
        }

        self.last_token_pending_kv = false;

        // 5. Update draft model KV cache to match accepted tokens
        self.draft_model.reset_cache();
        let mut replay_token = start_token;
        for &dt in draft_tokens.iter().take(accepted_count) {
            let _ = self.draft_model.forward_token(replay_token, None);
            replay_token = dt;
        }

        // 6. Queue accepted tokens for emission
        for token in result.tokens {
            self.recent_tokens.push(token);
            self.emit_buffer.push_back(token);
        }

        // Return first emitted token
        Ok(self.emit_buffer.pop_front())
    }

    /// Run direct target model sampling without speculative decoding.
    fn run_fallback_step(&mut self) -> Result<Option<Token>, SpeculativeError> {
        let last_token = *self
            .recent_tokens
            .last()
            .ok_or(SpeculativeError::NoContext)?;

        let logits = self
            .target_model
            .forward(
                &[if self.last_token_pending_kv {
                    last_token
                } else {
                    last_token
                }],
                &mut self.target_session,
            )
            .map_err(SpeculativeError::Model)?;

        let token = sample(&logits, self.config.sampling, fastrand::f32())
            .map_err(SpeculativeError::Sampling)?;

        self.recent_tokens.push(token);
        self.emit_buffer.push_back(token);
        self.stats.fallback_tokens += 1;
        self.stats.target_forward_passes += 1;
        self.last_token_pending_kv = false;

        Ok(Some(token))
    }
}

/// Errors during speculative decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativeError {
    Model(ModelError),
    DraftModel(String),
    Sampling(SamplingError),
    NoContext,
    ConfigError(String),
}

impl From<ModelError> for SpeculativeError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<SamplingError> for SpeculativeError {
    fn from(value: SamplingError) -> Self {
        Self::Sampling(value)
    }
}

impl std::fmt::Display for SpeculativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(e) => write!(f, "target model error: {:?}", e),
            Self::DraftModel(e) => write!(f, "draft model error: {}", e),
            Self::Sampling(e) => write!(f, "sampling error: {:?}", e),
            Self::NoContext => write!(f, "no context available for generation"),
            Self::ConfigError(e) => write!(f, "configuration error: {}", e),
        }
    }
}

impl std::error::Error for SpeculativeError {}

/// Builder for creating speculative decoder configurations.
pub struct SpeculativeConfigBuilder {
    config: SpeculativeConfig,
}

impl SpeculativeConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: SpeculativeConfig::default(),
        }
    }

    pub fn draft_tokens_per_step(mut self, n: usize) -> Self {
        self.config.draft_tokens_per_step = n;
        self
    }

    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.config.max_new_tokens = n;
        self
    }

    pub fn sampling(mut self, sampling: SamplingConfig) -> Self {
        self.config.sampling = sampling;
        self
    }

    pub fn stop_token(mut self, token: Token) -> Self {
        self.config.stop_token = Some(token);
        self
    }

    pub fn strict_mode(mut self, strict: bool) -> Self {
        self.config.strict_mode = strict;
        self
    }

    pub fn min_acceptance_rate(mut self, rate: f32) -> Self {
        self.config.min_acceptance_rate = rate;
        self
    }

    pub fn build(self) -> SpeculativeConfig {
        self.config
    }
}

impl Default for SpeculativeConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility: Create a DFlash draft model from a GGUF file for speculative decoding.
pub fn load_draft_model_for_speculative(
    model_path: &std::path::Path,
) -> Result<DFlashDraftModel, String> {
    use crate::dflash::DFlashConfig;
    use crate::gguf::MappedGgufFile;
    use crate::model_loader::{GgufModelLoader, ModelLoader};

    let loader = GgufModelLoader;
    let mapped: MappedGgufFile = loader
        .load(model_path)
        .map_err(|e| format!("failed to load draft model: {}", e))?;

    let config = DFlashConfig::from_gguf(&mapped);
    DFlashDraftModel::load_from_gguf(&mapped, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::{LlamaConfig, LlamaModel};

    #[test]
    fn speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.draft_tokens_per_step, 4);
        assert_eq!(config.max_new_tokens, 128);
        assert!(!config.strict_mode);
    }

    #[test]
    fn speculative_config_conservative() {
        let config = SpeculativeConfig::conservative();
        assert_eq!(config.draft_tokens_per_step, 2);
        assert!(config.strict_mode);
        assert!(config.min_acceptance_rate >= 0.5);
    }

    #[test]
    fn speculative_config_aggressive() {
        let config = SpeculativeConfig::aggressive();
        assert_eq!(config.draft_tokens_per_step, 8);
        assert!(!config.strict_mode);
    }

    #[test]
    fn stats_acceptance_rate() {
        let mut stats = SpeculativeStats::default();
        assert_eq!(stats.acceptance_rate(), 0.0);

        stats.total_draft_tokens = 100;
        stats.accepted_draft_tokens = 75;
        assert_eq!(stats.acceptance_rate(), 0.75);
    }

    #[test]
    fn stats_speedup() {
        let mut stats = SpeculativeStats::default();
        assert_eq!(stats.estimated_speedup(), 1.0);

        stats.accepted_draft_tokens = 300;
        stats.fallback_tokens = 100;
        stats.target_forward_passes = 100;
        assert_eq!(stats.estimated_speedup(), 4.0);
    }

    #[test]
    fn config_builder_chain() {
        let config = SpeculativeConfigBuilder::new()
            .draft_tokens_per_step(6)
            .max_new_tokens(200)
            .strict_mode(true)
            .min_acceptance_rate(0.4)
            .build();

        assert_eq!(config.draft_tokens_per_step, 6);
        assert_eq!(config.max_new_tokens, 200);
        assert!(config.strict_mode);
        assert_eq!(config.min_acceptance_rate, 0.4);
    }

    #[test]
    fn error_display() {
        let err = SpeculativeError::NoContext;
        assert_eq!(format!("{}", err), "no context available for generation");
    }
}
