use crate::dflash::DFlashDraftModel;
use crate::eagle3::Eagle3DraftModel;
use crate::inference::InferenceModel;
use crate::model::{Model, ModelError, Session, Token};
use crate::sampling::{SamplingConfig, SamplingError, sample, speculative_decode};
use futures_core::Stream;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Stop-sequence + stop-token tracking shared by every generation stream.
/// Keeps a bounded ring of recent tokens sized to the longest stop sequence
/// and checks every emitted token against it.
struct StopTracker {
    recent_tokens: Vec<Token>,
    max_len: usize,
}

impl StopTracker {
    fn new(stop_sequences: &[Vec<Token>]) -> Self {
        // Bound the ring by the same cap the spec streams use: a stop
        // sequence longer than this can never match a realistic window and
        // an untrusted config must not drive unbounded pre-allocation.
        const MAX_STOP_SEQUENCE_LEN: usize = 4096;
        let max_len = stop_sequences
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .min(MAX_STOP_SEQUENCE_LEN);
        Self {
            recent_tokens: Vec::with_capacity(max_len),
            max_len,
        }
    }

    /// Record an emitted token; returns true when generation should stop.
    fn push(
        &mut self,
        token: Token,
        stop_token: Option<Token>,
        stop_sequences: &[Vec<Token>],
    ) -> bool {
        if self.max_len > 0 {
            self.recent_tokens.push(token);
            if self.recent_tokens.len() > self.max_len {
                let to_drop = self.recent_tokens.len() - self.max_len;
                self.recent_tokens.drain(..to_drop);
            }
        }
        // Sequences longer than the ring can never match (their tail is
        // always truncated), so skip them explicitly instead of relying on
        // ends_with to fail — keeps the ring/match invariant explicit.
        let matched_stop_sequence = stop_sequences
            .iter()
            .filter(|sequence| !sequence.is_empty() && sequence.len() <= self.max_len)
            .any(|sequence| self.recent_tokens.ends_with(sequence));
        stop_token == Some(token) || matched_stop_sequence
    }
}

/// Speculative-decoding health bookkeeping shared by the DFlash / MTP /
/// EAGLE-3 streams: drafted/accepted totals, zero-acceptance streak, and
/// the disable decision (2 zero rounds, or ≥4 steps of samples below 0.2).
struct SpeculationHealth {
    drafted_tokens: usize,
    accepted_draft_tokens: usize,
    zero_acceptance_rounds: usize,
    speculation_disabled: bool,
}

impl SpeculationHealth {
    fn new() -> Self {
        Self {
            drafted_tokens: 0,
            accepted_draft_tokens: 0,
            zero_acceptance_rounds: 0,
            speculation_disabled: false,
        }
    }

    fn update(&mut self, drafted: usize, accepted: usize, cap: usize) {
        self.drafted_tokens = self.drafted_tokens.saturating_add(drafted);
        self.accepted_draft_tokens = self.accepted_draft_tokens.saturating_add(accepted);
        if accepted == 0 {
            self.zero_acceptance_rounds = self.zero_acceptance_rounds.saturating_add(1);
        } else {
            self.zero_acceptance_rounds = 0;
        }
        let enough_samples = self.drafted_tokens >= cap.max(1) * 4;
        let acceptance_rate = if self.drafted_tokens == 0 {
            1.0
        } else {
            self.accepted_draft_tokens as f32 / self.drafted_tokens as f32
        };
        if self.zero_acceptance_rounds >= 2 || (enough_samples && acceptance_rate < 0.2) {
            self.speculation_disabled = true;
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub stop_token: Option<Token>,
    pub stop_sequences: Vec<Vec<Token>>,
    pub prefill_batch_size: usize,
    pub sampling: SamplingConfig,
    pub suppressed_tokens: Vec<Token>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            stop_token: None,
            stop_sequences: Vec::new(),
            prefill_batch_size: 256,
            sampling: SamplingConfig::default(),
            suppressed_tokens: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationError {
    Model(ModelError),
    Sampling(SamplingError),
}

impl From<ModelError> for GenerationError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<SamplingError> for GenerationError {
    fn from(value: SamplingError) -> Self {
        Self::Sampling(value)
    }
}

/// Upper bound on draft tokens per speculative step.
/// Untrusted configs must not drive unbounded `Vec::with_capacity`.
pub const MAX_DRAFT_TOKENS_PER_STEP: usize = 64;

/// Speculative generation configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeGenerationConfig {
    pub generation: GenerationConfig,
    /// Number of tokens the draft model generates per speculative step.
    pub draft_tokens_per_step: usize,
    /// QuantSpec: draft with hierarchical I8 TurboQuant KV (self-speculative MTP path).
    pub quantspec_draft_kv: bool,
}

impl SpeculativeGenerationConfig {
    fn capped_draft_tokens_per_step(&self) -> usize {
        self.draft_tokens_per_step.min(MAX_DRAFT_TOKENS_PER_STEP)
    }
}

impl Default for SpeculativeGenerationConfig {
    fn default() -> Self {
        Self {
            generation: GenerationConfig::default(),
            draft_tokens_per_step: 4,
            quantspec_draft_kv: false,
        }
    }
}

/// A speculative generation stream that uses a DFlash draft model to accelerate
/// decoding via speculative decoding.
pub struct SpeculativeGenerationStream<'a, T: Model + ?Sized> {
    target_model: Option<&'a mut T>,
    draft_model: Option<&'a mut DFlashDraftModel>,
    session: Option<&'a mut Session>,
    prompt: &'a [Token],
    state: GenerationState,
    config: SpeculativeGenerationConfig,
    generated: usize,
    last_token: Option<Token>,
    stops: StopTracker,
    random: Box<dyn FnMut() -> f32 + 'a>,
    /// Buffer for draft tokens generated in the current speculative step.
    draft_token_buffer: Vec<Token>,
    /// Buffer for accepted tokens waiting to be emitted.
    emit_buffer: VecDeque<Token>,
    /// True when `last_token` was sampled but not yet written to the target KV cache.
    last_token_pending_kv: bool,
    /// Target logits for the token immediately after the committed prefix.
    pending_target_logits: Option<Vec<f32>>,
    health: SpeculationHealth,
}

impl<'a, T: Model + ?Sized> SpeculativeGenerationStream<'a, T> {
    pub fn new(
        target_model: &'a mut T,
        draft_model: &'a mut DFlashDraftModel,
        session: &'a mut Session,
        prompt: &'a [Token],
        config: SpeculativeGenerationConfig,
        random: impl FnMut() -> f32 + 'a,
    ) -> Self {
        // Allocation sizes here derive from the capped value so an
        // untrusted config cannot request an unbounded buffer (CodeQL:
        // rust/uncontrolled-allocation-size).
        let draft_tokens_per_step = config.capped_draft_tokens_per_step();
        let stops = StopTracker::new(&config.generation.stop_sequences);
        debug_assert!(draft_tokens_per_step <= MAX_DRAFT_TOKENS_PER_STEP);
        Self {
            target_model: Some(target_model),
            draft_model: Some(draft_model),
            session: Some(session),
            prompt,
            state: GenerationState::Prefill,
            config,
            generated: 0,
            last_token: None,
            stops,
            random: Box::new(random),
            draft_token_buffer: Vec::with_capacity(
                draft_tokens_per_step.min(MAX_DRAFT_TOKENS_PER_STEP),
            ),
            emit_buffer: VecDeque::with_capacity(
                draft_tokens_per_step
                    .saturating_add(1)
                    .min(MAX_DRAFT_TOKENS_PER_STEP + 1),
            ),
            last_token_pending_kv: false,
            pending_target_logits: None,
            health: SpeculationHealth::new(),
        }
    }

    fn emit_token(&mut self, token: Token) -> Option<Result<Token, GenerationError>> {
        self.generated = self.generated.saturating_add(1);
        self.last_token = Some(token);
        let stop_token = self.config.generation.stop_token;
        if self
            .stops
            .push(token, stop_token, &self.config.generation.stop_sequences)
        {
            self.state = GenerationState::Done;
        }
        Some(Ok(token))
    }

    fn run_target_step(&mut self) -> Result<(), GenerationError> {
        let target_model = self.target_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "target model missing".to_string(),
            ))
        })?;
        let session = self.session.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("session missing".to_string()))
        })?;
        let last_token = self.last_token.ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("no last token".to_string()))
        })?;

        let logits = if self.last_token_pending_kv {
            self.pending_target_logits = None;
            target_model
                .forward(&[last_token], session)
                .map_err(GenerationError::Model)?
        } else if let Some(logits) = self.pending_target_logits.take() {
            logits
        } else {
            target_model
                .forward(&[last_token], session)
                .map_err(GenerationError::Model)?
        };

        let random = (self.random.as_mut())();
        let token = sample(&logits, self.config.generation.sampling, random)
            .map_err(GenerationError::Sampling)?;
        self.last_token_pending_kv = true;
        self.emit_buffer.push_back(token);
        self.target_model = Some(target_model);
        self.session = Some(session);
        Ok(())
    }

    fn update_speculation_health(&mut self, drafted: usize, accepted: usize) {
        let cap = self.config.capped_draft_tokens_per_step();
        self.health.update(drafted, accepted, cap);
    }

    fn run_speculative_step(&mut self) -> Result<(), GenerationError> {
        let draft_model = self.draft_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "draft model missing".to_string(),
            ))
        })?;
        let target_model = self.target_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "target model missing".to_string(),
            ))
        })?;
        let session = self.session.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("session missing".to_string()))
        })?;

        let start_token = self.last_token.ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("no last token".to_string()))
        })?;

        // 1. Draft model generates K tokens autoregressively.
        let k = self.config.capped_draft_tokens_per_step();
        let mut draft_tokens = std::mem::take(&mut self.draft_token_buffer);
        draft_tokens.clear();
        let mut draft_logits = Vec::with_capacity(k);
        let mut current_token = start_token;

        draft_model.reset_cache();
        draft_model.reserve_cache_tokens(k);
        for _ in 0..k {
            let hidden = draft_model
                .forward_token(current_token, None)
                .map_err(|e| GenerationError::Model(ModelError::InferenceFailed(e)))?;
            let logits = draft_model
                .logits(&hidden)
                .map_err(|e| GenerationError::Model(ModelError::InferenceFailed(e)))?;
            let random = (self.random.as_mut())();
            let token = sample(&logits, self.config.generation.sampling, random)
                .map_err(GenerationError::Sampling)?;
            draft_tokens.push(token);
            draft_logits.push(logits);
            current_token = token;
        }

        // 2. Target model verifies draft tokens from a fixed KV checkpoint.
        let verify_start = session.consumed_tokens();
        let mut target_logits = Vec::with_capacity(draft_tokens.len() + 1);
        if self.last_token_pending_kv {
            target_model
                .rewind_to(verify_start)
                .map_err(GenerationError::Model)?;
            session.rewind_to(verify_start);
            let logits = target_model
                .forward(&[start_token], session)
                .map_err(GenerationError::Model)?;
            target_logits.push(logits);
        } else if let Some(logits) = self.pending_target_logits.take() {
            target_logits.push(logits);
        } else {
            return Err(GenerationError::Model(ModelError::InferenceFailed(
                "missing target logits for speculative verification".to_string(),
            )));
        }

        let verified_logits = target_model
            .forward_many(&draft_tokens, session)
            .map_err(GenerationError::Model)?;
        target_logits.extend(verified_logits);

        let mut accepted_sequence = if self.last_token_pending_kv {
            vec![start_token]
        } else {
            Vec::new()
        };

        // 3. Speculative decode: accept/reject draft tokens.
        let randoms: Vec<f32> = (0..=2 * draft_tokens.len())
            .map(|_| (self.random.as_mut())())
            .collect();

        let result = speculative_decode(
            &draft_tokens,
            &draft_logits,
            &target_logits,
            self.config.generation.sampling,
            &randoms,
        )
        .map_err(GenerationError::Sampling)?;

        accepted_sequence.extend_from_slice(&result.tokens);
        target_model
            .rewind_to(verify_start)
            .map_err(GenerationError::Model)?;
        session.rewind_to(verify_start);
        let next_target_logits = target_model
            .forward(&accepted_sequence, session)
            .map_err(GenerationError::Model)?;
        self.pending_target_logits = Some(next_target_logits);
        self.last_token_pending_kv = false;

        let accepted_count = result.accepted_draft_tokens;
        self.update_speculation_health(draft_tokens.len(), accepted_count);
        for token in result.tokens {
            self.emit_buffer.push_back(token);
        }

        draft_tokens.clear();
        self.draft_token_buffer = draft_tokens;
        self.draft_model = Some(draft_model);
        self.target_model = Some(target_model);
        self.session = Some(session);
        Ok(())
    }
}

impl<T: Model + ?Sized> Stream for SpeculativeGenerationStream<'_, T> {
    type Item = Result<Token, GenerationError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Emit buffered tokens first.
        if let Some(token) = self.emit_buffer.pop_front() {
            return Poll::Ready(self.emit_token(token));
        }

        if self.generated >= self.config.generation.max_new_tokens
            || matches!(self.state, GenerationState::Done)
        {
            self.state = GenerationState::Done;
            return Poll::Ready(None);
        }

        match self.state {
            GenerationState::Prefill => {
                self.state = GenerationState::Decode;
                let target_model = self.target_model.take().unwrap();
                let session = self.session.take().unwrap();
                let prompt = self.prompt;

                let logits = if prompt.is_empty() {
                    target_model
                        .forward(prompt, session)
                        .map_err(GenerationError::Model)?
                } else {
                    let batch_size = self.config.generation.prefill_batch_size.max(1);
                    let mut last_logits = None;
                    for chunk in prompt.chunks(batch_size) {
                        last_logits = Some(
                            target_model
                                .forward(chunk, session)
                                .map_err(GenerationError::Model)?,
                        );
                    }
                    last_logits.expect("non-empty prompt should produce logits")
                };

                let random = (self.random.as_mut())();
                let token = sample(&logits, self.config.generation.sampling, random)
                    .map_err(GenerationError::Sampling)?;

                self.target_model = Some(target_model);
                self.session = Some(session);
                self.last_token = Some(token);
                self.last_token_pending_kv = true;
                self.pending_target_logits = None;
                Poll::Ready(self.emit_token(token))
            }
            GenerationState::Decode => {
                let result = if self.health.speculation_disabled {
                    self.run_target_step()
                } else {
                    self.run_speculative_step()
                };
                if let Err(e) = result {
                    self.state = GenerationState::Done;
                    return Poll::Ready(Some(Err(e)));
                }
                if let Some(token) = self.emit_buffer.pop_front() {
                    return Poll::Ready(self.emit_token(token));
                }
                self.state = GenerationState::Done;
                Poll::Ready(None)
            }
            GenerationState::Done => Poll::Ready(None),
        }
    }
}

/// Speculative generation using a native in-GGUF MTP/nextn block on the target
/// model (Qwen3.5/Qwen3.6 `nextn_predict_layers`). Unlike an autoregressive
/// external draft model, MTP drafts from the last committed target token plus
/// that token's output-normalized hidden state, so the prompt prefill itself
/// provides the first draft anchor.
pub struct MtpGenerationStream<'a> {
    target_model: Option<&'a mut InferenceModel>,
    session: Option<&'a mut Session>,
    prompt: &'a [Token],
    state: GenerationState,
    config: SpeculativeGenerationConfig,
    generated: usize,
    last_token: Option<Token>,
    stops: StopTracker,
    random: Box<dyn FnMut() -> f32 + 'a>,
    draft_token_buffer: Vec<Token>,
    emit_buffer: VecDeque<Token>,
    pending_target_logits: Option<Vec<f32>>,
    health: SpeculationHealth,
}

impl<'a> MtpGenerationStream<'a> {
    pub fn new(
        target_model: &'a mut InferenceModel,
        session: &'a mut Session,
        prompt: &'a [Token],
        config: SpeculativeGenerationConfig,
        random: impl FnMut() -> f32 + 'a,
    ) -> Self {
        // Allocation sizes here derive from the capped value so an
        // untrusted config cannot request an unbounded buffer (CodeQL:
        // rust/uncontrolled-allocation-size).
        let draft_tokens_per_step = config.capped_draft_tokens_per_step();
        let stops = StopTracker::new(&config.generation.stop_sequences);
        debug_assert!(draft_tokens_per_step <= MAX_DRAFT_TOKENS_PER_STEP);
        Self {
            target_model: Some(target_model),
            session: Some(session),
            prompt,
            state: GenerationState::Prefill,
            config,
            generated: 0,
            last_token: None,
            stops,
            random: Box::new(random),
            draft_token_buffer: Vec::with_capacity(
                draft_tokens_per_step.min(MAX_DRAFT_TOKENS_PER_STEP),
            ),
            emit_buffer: VecDeque::with_capacity(
                draft_tokens_per_step
                    .saturating_add(1)
                    .min(MAX_DRAFT_TOKENS_PER_STEP + 1),
            ),
            pending_target_logits: None,
            health: SpeculationHealth::new(),
        }
    }

    fn emit_token(&mut self, token: Token) -> Option<Result<Token, GenerationError>> {
        self.generated = self.generated.saturating_add(1);
        self.last_token = Some(token);
        let stop_token = self.config.generation.stop_token;
        if self
            .stops
            .push(token, stop_token, &self.config.generation.stop_sequences)
        {
            self.state = GenerationState::Done;
        }
        Some(Ok(token))
    }

    fn update_speculation_health(&mut self, drafted: usize, accepted: usize) {
        let cap = self.config.capped_draft_tokens_per_step();
        self.health.update(drafted, accepted, cap);
    }

    fn run_target_step(&mut self) -> Result<(), GenerationError> {
        let target_model = self.target_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "target model missing".to_string(),
            ))
        })?;
        let session = self.session.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("session missing".to_string()))
        })?;
        let logits = self.pending_target_logits.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "missing target logits for MTP fallback".to_string(),
            ))
        })?;
        let token = sample(
            &logits,
            self.config.generation.sampling,
            (self.random.as_mut())(),
        )
        .map_err(GenerationError::Sampling)?;
        let next_logits = target_model
            .forward(&[token], session)
            .map_err(GenerationError::Model)?;
        self.pending_target_logits = Some(next_logits);
        self.emit_buffer.push_back(token);
        self.target_model = Some(target_model);
        self.session = Some(session);
        Ok(())
    }

    fn run_mtp_step(&mut self) -> Result<(), GenerationError> {
        let target_model = self.target_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "target model missing".to_string(),
            ))
        })?;
        let session = self.session.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("session missing".to_string()))
        })?;
        let start_token = self.last_token.ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "no MTP anchor token".to_string(),
            ))
        })?;
        let anchor_hidden = target_model.last_output_hidden().to_vec();
        if anchor_hidden.is_empty() {
            return Err(GenerationError::Model(ModelError::InferenceFailed(
                "missing MTP anchor hidden state".to_string(),
            )));
        }

        let k = self.config.capped_draft_tokens_per_step().max(1);
        let mut draft_tokens = std::mem::take(&mut self.draft_token_buffer);
        draft_tokens.clear();
        let (sampled_draft_tokens, draft_logits) = target_model
            .draft_mtp_tokens(
                start_token,
                &anchor_hidden,
                k,
                self.config.generation.sampling,
                self.random.as_mut(),
                self.config.quantspec_draft_kv,
            )
            .map_err(GenerationError::Model)?;
        draft_tokens.extend_from_slice(&sampled_draft_tokens);

        let verify_start = session.consumed_tokens();
        let mut target_logits = Vec::with_capacity(draft_tokens.len() + 1);
        let first_logits = self.pending_target_logits.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "missing target logits for MTP verification".to_string(),
            ))
        })?;
        target_logits.push(first_logits);
        let verified_logits = target_model
            .forward_many(&draft_tokens, session)
            .map_err(GenerationError::Model)?;
        target_logits.extend(verified_logits);

        let randoms: Vec<f32> = (0..=2 * draft_tokens.len())
            .map(|_| (self.random.as_mut())())
            .collect();
        let result = speculative_decode(
            &draft_tokens,
            &draft_logits,
            &target_logits,
            self.config.generation.sampling,
            &randoms,
        )
        .map_err(GenerationError::Sampling)?;

        target_model
            .rewind_to(verify_start)
            .map_err(GenerationError::Model)?;
        session.rewind_to(verify_start);
        let next_target_logits = target_model
            .forward(&result.tokens, session)
            .map_err(GenerationError::Model)?;
        self.pending_target_logits = Some(next_target_logits);

        let accepted_count = result.accepted_draft_tokens;
        self.update_speculation_health(draft_tokens.len(), accepted_count);
        for token in result.tokens {
            self.emit_buffer.push_back(token);
        }

        draft_tokens.clear();
        self.draft_token_buffer = draft_tokens;
        self.target_model = Some(target_model);
        self.session = Some(session);
        Ok(())
    }

    fn prefill(&mut self) -> Result<(), GenerationError> {
        let target_model = self.target_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "target model missing".to_string(),
            ))
        })?;
        let session = self.session.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("session missing".to_string()))
        })?;
        if self.prompt.is_empty() {
            return Err(GenerationError::Model(ModelError::EmptyInput));
        }
        let batch_size = self.config.generation.prefill_batch_size.max(1);
        let mut logits = None;
        for chunk in self.prompt.chunks(batch_size) {
            logits = Some(
                target_model
                    .forward(chunk, session)
                    .map_err(GenerationError::Model)?,
            );
        }
        self.pending_target_logits = logits;
        self.last_token = self.prompt.last().copied();
        self.target_model = Some(target_model);
        self.session = Some(session);
        self.state = GenerationState::Decode;
        Ok(())
    }
}

impl Stream for MtpGenerationStream<'_> {
    type Item = Result<Token, GenerationError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Terminate before draining buffered tokens. One MTP step can enqueue
        // several tokens at once (accepted drafts + the bonus token), so the
        // budget/stop checks must gate every emitted token — not just run
        // between steps. Otherwise a request with max_new_tokens=1 and
        // draft_tokens=4 would emit up to 5 tokens, and a stop/EOS token popped
        // from the buffer (which sets Done in `emit_token`) would not prevent
        // the trailing buffered tokens from being returned.
        if self.generated >= self.config.generation.max_new_tokens
            || matches!(self.state, GenerationState::Done)
        {
            self.state = GenerationState::Done;
            self.emit_buffer.clear();
            return Poll::Ready(None);
        }

        if let Some(token) = self.emit_buffer.pop_front() {
            return Poll::Ready(self.emit_token(token));
        }

        if matches!(self.state, GenerationState::Prefill)
            && let Err(e) = self.prefill()
        {
            self.state = GenerationState::Done;
            return Poll::Ready(Some(Err(e)));
        }

        let result = if self.health.speculation_disabled {
            self.run_target_step()
        } else {
            self.run_mtp_step()
        };
        if let Err(e) = result {
            self.state = GenerationState::Done;
            return Poll::Ready(Some(Err(e)));
        }
        if let Some(token) = self.emit_buffer.pop_front() {
            return Poll::Ready(self.emit_token(token));
        }
        self.state = GenerationState::Done;
        Poll::Ready(None)
    }
}

/// Speculative generation with an external EAGLE3 draft model fused from target
/// layer hidden states captured during target forward.
pub struct Eagle3GenerationStream<'a> {
    target_model: Option<&'a mut InferenceModel>,
    draft_model: Option<&'a mut Eagle3DraftModel>,
    session: Option<&'a mut Session>,
    prompt: &'a [Token],
    state: GenerationState,
    config: SpeculativeGenerationConfig,
    generated: usize,
    last_token: Option<Token>,
    stops: StopTracker,
    random: Box<dyn FnMut() -> f32 + 'a>,
    draft_token_buffer: Vec<Token>,
    emit_buffer: VecDeque<Token>,
    last_token_pending_kv: bool,
    pending_target_logits: Option<Vec<f32>>,
    health: SpeculationHealth,
}

impl<'a> Eagle3GenerationStream<'a> {
    pub fn new(
        target_model: &'a mut InferenceModel,
        draft_model: &'a mut Eagle3DraftModel,
        session: &'a mut Session,
        prompt: &'a [Token],
        config: SpeculativeGenerationConfig,
        random: impl FnMut() -> f32 + 'a,
    ) -> Self {
        target_model.set_eagle3_capture_layers(draft_model.config.extract_layers.clone());
        // Allocation sizes here derive from the capped value so an
        // untrusted config cannot request an unbounded buffer (CodeQL:
        // rust/uncontrolled-allocation-size).
        let draft_tokens_per_step = config.capped_draft_tokens_per_step();
        let stops = StopTracker::new(&config.generation.stop_sequences);
        debug_assert!(draft_tokens_per_step <= MAX_DRAFT_TOKENS_PER_STEP);
        Self {
            target_model: Some(target_model),
            draft_model: Some(draft_model),
            session: Some(session),
            prompt,
            state: GenerationState::Prefill,
            config,
            generated: 0,
            last_token: None,
            stops,
            random: Box::new(random),
            draft_token_buffer: Vec::with_capacity(
                draft_tokens_per_step.min(MAX_DRAFT_TOKENS_PER_STEP),
            ),
            emit_buffer: VecDeque::with_capacity(
                draft_tokens_per_step
                    .saturating_add(1)
                    .min(MAX_DRAFT_TOKENS_PER_STEP + 1),
            ),
            last_token_pending_kv: false,
            pending_target_logits: None,
            health: SpeculationHealth::new(),
        }
    }

    fn emit_token(&mut self, token: Token) -> Option<Result<Token, GenerationError>> {
        self.generated = self.generated.saturating_add(1);
        self.last_token = Some(token);
        let stop_token = self.config.generation.stop_token;
        if self
            .stops
            .push(token, stop_token, &self.config.generation.stop_sequences)
        {
            self.state = GenerationState::Done;
        }
        Some(Ok(token))
    }

    fn update_speculation_health(&mut self, drafted: usize, accepted: usize) {
        let cap = self.config.capped_draft_tokens_per_step();
        self.health.update(drafted, accepted, cap);
    }

    fn run_target_step(&mut self) -> Result<(), GenerationError> {
        let target_model = self.target_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "target model missing".to_string(),
            ))
        })?;
        let session = self.session.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("session missing".to_string()))
        })?;
        let last_token = self.last_token.ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("no last token".to_string()))
        })?;
        let logits = if self.last_token_pending_kv {
            self.pending_target_logits = None;
            target_model
                .forward(&[last_token], session)
                .map_err(GenerationError::Model)?
        } else if let Some(logits) = self.pending_target_logits.take() {
            logits
        } else {
            target_model
                .forward(&[last_token], session)
                .map_err(GenerationError::Model)?
        };
        let token = sample(
            &logits,
            self.config.generation.sampling,
            (self.random.as_mut())(),
        )
        .map_err(GenerationError::Sampling)?;
        self.last_token_pending_kv = true;
        self.emit_buffer.push_back(token);
        self.target_model = Some(target_model);
        self.session = Some(session);
        Ok(())
    }

    fn run_eagle3_step(&mut self) -> Result<(), GenerationError> {
        let draft_model = self.draft_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "EAGLE3 draft model missing".to_string(),
            ))
        })?;
        let target_model = self.target_model.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed(
                "target model missing".to_string(),
            ))
        })?;
        let session = self.session.take().ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("session missing".to_string()))
        })?;
        let start_token = self.last_token.ok_or_else(|| {
            GenerationError::Model(ModelError::InferenceFailed("no last token".to_string()))
        })?;

        let features = target_model
            .concat_eagle3_features()
            .map_err(GenerationError::Model)?;
        draft_model
            .encode_features(&features)
            .map_err(|e| GenerationError::Model(ModelError::InferenceFailed(e)))?;

        let k = self.config.capped_draft_tokens_per_step();
        let mut draft_tokens = std::mem::take(&mut self.draft_token_buffer);
        draft_tokens.clear();
        let mut draft_logits = Vec::with_capacity(k);
        let mut current_token = start_token;
        draft_model.reset_cache();
        draft_model.reserve_cache_tokens(k);
        for _ in 0..k {
            let (_, logits) = draft_model
                .forward_decoder(current_token)
                .map_err(|e| GenerationError::Model(ModelError::InferenceFailed(e)))?;
            let token = sample(
                &logits,
                self.config.generation.sampling,
                (self.random.as_mut())(),
            )
            .map_err(GenerationError::Sampling)?;
            draft_tokens.push(token);
            draft_logits.push(logits);
            current_token = token;
        }

        let verify_start = session.consumed_tokens();
        let mut target_logits = Vec::with_capacity(draft_tokens.len() + 1);
        if self.last_token_pending_kv {
            target_model
                .rewind_to(verify_start)
                .map_err(GenerationError::Model)?;
            session.rewind_to(verify_start);
            let logits = target_model
                .forward(&[start_token], session)
                .map_err(GenerationError::Model)?;
            target_logits.push(logits);
        } else if let Some(logits) = self.pending_target_logits.take() {
            target_logits.push(logits);
        } else {
            return Err(GenerationError::Model(ModelError::InferenceFailed(
                "missing target logits for EAGLE3 verification".to_string(),
            )));
        }
        target_logits.extend(
            target_model
                .forward_many(&draft_tokens, session)
                .map_err(GenerationError::Model)?,
        );

        let randoms: Vec<f32> = (0..=2 * draft_tokens.len())
            .map(|_| (self.random.as_mut())())
            .collect();
        let result = speculative_decode(
            &draft_tokens,
            &draft_logits,
            &target_logits,
            self.config.generation.sampling,
            &randoms,
        )
        .map_err(GenerationError::Sampling)?;

        target_model
            .rewind_to(verify_start)
            .map_err(GenerationError::Model)?;
        session.rewind_to(verify_start);
        let next_target_logits = target_model
            .forward(&result.tokens, session)
            .map_err(GenerationError::Model)?;
        self.pending_target_logits = Some(next_target_logits);
        self.last_token_pending_kv = false;
        self.update_speculation_health(draft_tokens.len(), result.accepted_draft_tokens);
        for token in result.tokens {
            self.emit_buffer.push_back(token);
        }
        draft_tokens.clear();
        self.draft_token_buffer = draft_tokens;
        self.draft_model = Some(draft_model);
        self.target_model = Some(target_model);
        self.session = Some(session);
        Ok(())
    }
}

impl Stream for Eagle3GenerationStream<'_> {
    type Item = Result<Token, GenerationError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.generated >= self.config.generation.max_new_tokens
            || matches!(self.state, GenerationState::Done)
        {
            self.state = GenerationState::Done;
            self.emit_buffer.clear();
            return Poll::Ready(None);
        }

        if let Some(token) = self.emit_buffer.pop_front() {
            return Poll::Ready(self.emit_token(token));
        }

        let Some(target_model) = self.target_model.take() else {
            self.state = GenerationState::Done;
            return Poll::Ready(None);
        };
        let Some(session) = self.session.take() else {
            self.state = GenerationState::Done;
            return Poll::Ready(None);
        };

        match self.state {
            GenerationState::Prefill => {
                self.state = GenerationState::Decode;
                let prompt = self.prompt;
                let logits = if prompt.is_empty() {
                    target_model
                        .forward(prompt, session)
                        .map_err(GenerationError::Model)?
                } else {
                    let batch_size = self.config.generation.prefill_batch_size.max(1);
                    let mut last_logits = None;
                    for chunk in prompt.chunks(batch_size) {
                        last_logits = Some(
                            target_model
                                .forward(chunk, session)
                                .map_err(GenerationError::Model)?,
                        );
                    }
                    last_logits.ok_or_else(|| {
                        GenerationError::Model(ModelError::InferenceFailed(
                            "empty prefill logits".to_string(),
                        ))
                    })?
                };
                let token = sample(
                    &logits,
                    self.config.generation.sampling,
                    (self.random.as_mut())(),
                )
                .map_err(GenerationError::Sampling)?;
                self.last_token = Some(token);
                self.pending_target_logits = Some(logits);
                self.last_token_pending_kv = false;
                self.emit_buffer.push_back(token);
                self.target_model = Some(target_model);
                self.session = Some(session);
                if let Some(token) = self.emit_buffer.pop_front() {
                    return Poll::Ready(self.emit_token(token));
                }
                Poll::Pending
            }
            GenerationState::Decode => {
                self.target_model = Some(target_model);
                self.session = Some(session);
                let result = if self.health.speculation_disabled {
                    self.run_target_step()
                } else {
                    self.run_eagle3_step()
                };
                if let Err(e) = result {
                    self.state = GenerationState::Done;
                    return Poll::Ready(Some(Err(e)));
                }
                if let Some(token) = self.emit_buffer.pop_front() {
                    return Poll::Ready(self.emit_token(token));
                }
                self.state = GenerationState::Done;
                Poll::Ready(None)
            }
            GenerationState::Done => Poll::Ready(None),
        }
    }
}

enum GenerationState {
    Prefill,
    Decode,
    Done,
}

pub struct GenerationStream<'a, M: Model + ?Sized> {
    model: Option<&'a mut M>,
    session: Option<&'a mut Session>,
    prompt: &'a [Token],
    state: GenerationState,
    config: GenerationConfig,
    generated: usize,
    last_token: Option<Token>,
    stops: StopTracker,
    random: Box<dyn FnMut() -> f32 + 'a>,
}

impl<'a, M: Model + ?Sized> GenerationStream<'a, M> {
    pub fn new(
        model: &'a mut M,
        session: &'a mut Session,
        prompt: &'a [Token],
        config: GenerationConfig,
        random: impl FnMut() -> f32 + 'a,
    ) -> Self {
        let stops = StopTracker::new(&config.stop_sequences);
        Self {
            model: Some(model),
            session: Some(session),
            prompt,
            state: GenerationState::Prefill,
            config,
            generated: 0,
            last_token: None,
            stops,
            random: Box::new(random),
        }
    }
}

impl<M: Model + ?Sized> Stream for GenerationStream<'_, M> {
    type Item = Result<Token, GenerationError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.generated >= self.config.max_new_tokens
            || matches!(self.state, GenerationState::Done)
        {
            self.state = GenerationState::Done;
            return Poll::Ready(None);
        }

        let Some(model) = self.model.take() else {
            self.state = GenerationState::Done;
            return Poll::Ready(None);
        };
        let Some(session) = self.session.take() else {
            self.state = GenerationState::Done;
            return Poll::Ready(None);
        };

        let mut logits = match self.state {
            GenerationState::Prefill => {
                self.state = GenerationState::Decode;
                let prompt = self.prompt;
                if prompt.is_empty() {
                    match model.forward(prompt, session) {
                        Ok(logits) => logits,
                        Err(err) => {
                            self.model = Some(model);
                            self.session = Some(session);
                            self.state = GenerationState::Done;
                            return Poll::Ready(Some(Err(err.into())));
                        }
                    }
                } else {
                    let batch_size = self.config.prefill_batch_size.max(1);
                    let mut logits = None;
                    for chunk in prompt.chunks(batch_size) {
                        match model.forward(chunk, session) {
                            Ok(chunk_logits) => logits = Some(chunk_logits),
                            Err(err) => {
                                self.model = Some(model);
                                self.session = Some(session);
                                self.state = GenerationState::Done;
                                return Poll::Ready(Some(Err(err.into())));
                            }
                        }
                    }
                    logits.expect("non-empty prompt chunks should produce logits")
                }
            }
            GenerationState::Decode => {
                let Some(last_token) = self.last_token else {
                    self.model = Some(model);
                    self.session = Some(session);
                    self.state = GenerationState::Done;
                    return Poll::Ready(None);
                };
                match model.forward(&[last_token], session) {
                    Ok(logits) => logits,
                    Err(err) => {
                        self.model = Some(model);
                        self.session = Some(session);
                        self.state = GenerationState::Done;
                        return Poll::Ready(Some(Err(err.into())));
                    }
                }
            }
            GenerationState::Done => return Poll::Ready(None),
        };
        for &token in &self.config.suppressed_tokens {
            if let Some(logit) = logits.get_mut(token as usize) {
                *logit = f32::NEG_INFINITY;
            }
        }

        let random = (self.random.as_mut())();
        let token = match sample(&logits, self.config.sampling, random) {
            Ok(token) => token,
            Err(err) => {
                self.model = Some(model);
                self.session = Some(session);
                self.state = GenerationState::Done;
                return Poll::Ready(Some(Err(err.into())));
            }
        };

        self.model = Some(model);
        self.session = Some(session);
        self.generated = self.generated.saturating_add(1);
        self.last_token = Some(token);
        let stop_token = self.config.stop_token;
        let stop_sequences = self.config.stop_sequences.clone();
        if self.stops.push(token, stop_token, &stop_sequences) {
            self.state = GenerationState::Done;
        }

        Poll::Ready(Some(Ok(token)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::{LlamaConfig, LlamaModel};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::{Wake, Waker};

    #[derive(Default)]
    struct NoopWaker;

    #[allow(unknown_lints, clippy::manual_noop_waker)]
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    fn collect_stream<M: Model>(
        stream: &mut GenerationStream<'_, M>,
    ) -> Vec<Result<Token, GenerationError>> {
        let waker: Waker = Waker::from(Arc::new(NoopWaker));
        let mut cx = Context::from_waker(&waker);
        let mut pinned = Pin::new(stream);
        let mut items = Vec::new();
        loop {
            match Stream::poll_next(pinned.as_mut(), &mut cx) {
                Poll::Ready(Some(item)) => items.push(item),
                Poll::Ready(None) => break,
                Poll::Pending => panic!("generation stream should never pend"),
            }
        }
        items
    }

    #[test]
    fn streams_up_to_max_new_tokens() {
        let mut model = LlamaModel::new(LlamaConfig::llama2(16, 32, 2));
        let mut session = Session::new();
        let mut stream = GenerationStream::new(
            &mut model,
            &mut session,
            &[1, 2, 3],
            GenerationConfig {
                max_new_tokens: 3,
                sampling: SamplingConfig {
                    temperature: 0.01,
                    ..SamplingConfig::default()
                },
                ..GenerationConfig::default()
            },
            || 0.5,
        );

        let items = collect_stream(&mut stream);
        assert_eq!(items, vec![Ok(3), Ok(3), Ok(3)]);
    }

    #[test]
    fn stops_when_stop_token_is_generated() {
        let mut model = LlamaModel::new(LlamaConfig::llama2(16, 32, 2));
        let mut session = Session::new();
        let mut stream = GenerationStream::new(
            &mut model,
            &mut session,
            &[5],
            GenerationConfig {
                max_new_tokens: 8,
                stop_token: Some(5),
                sampling: SamplingConfig {
                    temperature: 0.01,
                    ..SamplingConfig::default()
                },
                ..GenerationConfig::default()
            },
            || 0.1,
        );

        let items = collect_stream(&mut stream);
        assert_eq!(items, vec![Ok(5)]);
    }

    #[test]
    fn stops_when_stop_sequence_is_generated() {
        let mut model = LlamaModel::new(LlamaConfig::llama2(16, 32, 2));
        let mut session = Session::new();
        let mut stream = GenerationStream::new(
            &mut model,
            &mut session,
            &[3],
            GenerationConfig {
                max_new_tokens: 8,
                stop_sequences: vec![vec![3, 3]],
                sampling: SamplingConfig {
                    temperature: 0.01,
                    ..SamplingConfig::default()
                },
                ..GenerationConfig::default()
            },
            || 0.1,
        );

        let items = collect_stream(&mut stream);
        assert_eq!(items, vec![Ok(3), Ok(3)]);
    }

    #[test]
    fn ignores_empty_stop_sequences() {
        let mut model = LlamaModel::new(LlamaConfig::llama2(16, 32, 2));
        let mut session = Session::new();
        let mut stream = GenerationStream::new(
            &mut model,
            &mut session,
            &[3],
            GenerationConfig {
                max_new_tokens: 3,
                stop_sequences: vec![Vec::new()],
                sampling: SamplingConfig {
                    temperature: 0.01,
                    ..SamplingConfig::default()
                },
                ..GenerationConfig::default()
            },
            || 0.1,
        );

        let items = collect_stream(&mut stream);
        assert_eq!(items, vec![Ok(3), Ok(3), Ok(3)]);
    }

    #[test]
    fn yields_model_error_for_empty_prompt() {
        let mut model = LlamaModel::new(LlamaConfig::llama2(16, 32, 2));
        let mut session = Session::new();
        let mut stream = GenerationStream::new(
            &mut model,
            &mut session,
            &[],
            GenerationConfig {
                max_new_tokens: 4,
                stop_token: None,
                sampling: SamplingConfig::default(),
                ..GenerationConfig::default()
            },
            || 0.2,
        );

        let items = collect_stream(&mut stream);
        assert_eq!(
            items,
            vec![Err(GenerationError::Model(ModelError::EmptyInput))]
        );
    }

    #[derive(Debug)]
    struct RecordingModel {
        vocab_size: usize,
        context_size: usize,
        layer_count: usize,
        batch_sizes: Rc<RefCell<Vec<usize>>>,
    }

    impl Model for RecordingModel {
        fn forward(
            &mut self,
            tokens: &[Token],
            session: &mut Session,
        ) -> Result<Vec<f32>, ModelError> {
            self.batch_sizes.borrow_mut().push(tokens.len());
            if tokens.is_empty() {
                return Err(ModelError::EmptyInput);
            }
            let requested_total_tokens = session.consumed_tokens().saturating_add(tokens.len());
            if requested_total_tokens > self.context_size {
                return Err(ModelError::ContextExceeded {
                    context_size: self.context_size,
                    requested_total_tokens,
                });
            }
            session.record_tokens(tokens.len());
            let mut logits = vec![0.0; self.vocab_size];
            let next_token = (tokens[tokens.len() - 1] as usize) % self.vocab_size;
            logits[next_token] = 1.0;
            Ok(logits)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn context_size(&self) -> usize {
            self.context_size
        }

        fn layer_count(&self) -> usize {
            self.layer_count
        }
    }

    #[test]
    fn prefill_processes_prompt_in_batches() {
        let batch_sizes = Rc::new(RefCell::new(Vec::new()));
        let mut model = RecordingModel {
            vocab_size: 32,
            context_size: 64,
            layer_count: 2,
            batch_sizes: Rc::clone(&batch_sizes),
        };
        let mut session = Session::new();
        let mut stream = GenerationStream::new(
            &mut model,
            &mut session,
            &[1, 2, 3, 4, 5],
            GenerationConfig {
                max_new_tokens: 1,
                prefill_batch_size: 2,
                sampling: SamplingConfig {
                    temperature: 0.01,
                    ..SamplingConfig::default()
                },
                ..GenerationConfig::default()
            },
            || 0.4,
        );

        let items = collect_stream(&mut stream);
        assert_eq!(items, vec![Ok(5)]);
        assert_eq!(*batch_sizes.borrow(), vec![2, 2, 1]);
    }

    #[test]
    fn caps_untrusted_draft_tokens_per_step() {
        let cfg = SpeculativeGenerationConfig {
            draft_tokens_per_step: usize::MAX,
            ..SpeculativeGenerationConfig::default()
        };
        assert_eq!(
            cfg.capped_draft_tokens_per_step(),
            MAX_DRAFT_TOKENS_PER_STEP
        );
        const _: () = assert!(MAX_DRAFT_TOKENS_PER_STEP < 1024);
    }

    #[test]
    fn stop_tracker_matches_sequences_within_ring() {
        let stop = vec![3_u32, 4, 5];
        let mut tracker = StopTracker::new(std::slice::from_ref(&stop));
        let seqs = [stop.clone()];
        assert!(!tracker.push(1, None, &seqs));
        assert!(!tracker.push(2, None, &seqs));
        assert!(!tracker.push(3, None, &seqs)); // partial only
        assert!(!tracker.push(4, None, &seqs)); // partial only
        assert!(tracker.push(5, None, &seqs)); // completes [3, 4, 5]
    }

    #[test]
    fn stop_tracker_ignores_sequences_longer_than_ring() {
        // A pathological config: one sequence longer than the 4096 ring cap
        // (impossible to match) plus one normal sequence. The over-long one
        // must not stop generation, and the ring cap must bound allocation.
        let overlong = vec![7_u32; 5000];
        let normal = vec![9_u32, 9];
        let mut tracker = StopTracker::new(&[overlong, normal.clone()]);
        assert!(tracker.recent_tokens.capacity() <= 4096);
        assert!(!tracker.push(7, None, &[vec![7_u32; 5000]])); // never matches
        assert!(!tracker.push(9, None, &[vec![7_u32; 5000]]));
        assert!(tracker.push(9, None, &[normal]));
    }
}
