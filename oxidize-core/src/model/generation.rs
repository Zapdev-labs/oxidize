use crate::dflash::DFlashDraftModel;
use crate::model::{Model, ModelError, Session, Token};
use crate::sampling::{SamplingConfig, SamplingError, sample, speculative_decode};
use futures_core::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

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

/// Speculative generation configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeGenerationConfig {
    pub generation: GenerationConfig,
    /// Number of tokens the draft model generates per speculative step.
    pub draft_tokens_per_step: usize,
}

impl Default for SpeculativeGenerationConfig {
    fn default() -> Self {
        Self {
            generation: GenerationConfig::default(),
            draft_tokens_per_step: 4,
        }
    }
}

/// A speculative generation stream that uses a DFlash draft model to accelerate
/// decoding via speculative decoding.
pub struct SpeculativeGenerationStream<'a, T: Model> {
    target_model: Option<&'a mut T>,
    draft_model: Option<&'a mut DFlashDraftModel>,
    session: Option<&'a mut Session>,
    prompt: &'a [Token],
    state: GenerationState,
    config: SpeculativeGenerationConfig,
    generated: usize,
    last_token: Option<Token>,
    recent_tokens: Vec<Token>,
    max_stop_sequence_len: usize,
    random: Box<dyn FnMut() -> f32 + 'a>,
    /// Buffer for draft tokens generated in the current speculative step.
    draft_token_buffer: Vec<Token>,
    /// Buffer for accepted tokens waiting to be emitted.
    emit_buffer: Vec<Token>,
}

impl<'a, T: Model> SpeculativeGenerationStream<'a, T> {
    pub fn new(
        target_model: &'a mut T,
        draft_model: &'a mut DFlashDraftModel,
        session: &'a mut Session,
        prompt: &'a [Token],
        config: SpeculativeGenerationConfig,
        random: impl FnMut() -> f32 + 'a,
    ) -> Self {
        let max_stop_sequence_len = config
            .generation
            .stop_sequences
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        Self {
            target_model: Some(target_model),
            draft_model: Some(draft_model),
            session: Some(session),
            prompt,
            state: GenerationState::Prefill,
            config,
            generated: 0,
            last_token: None,
            recent_tokens: Vec::with_capacity(max_stop_sequence_len),
            max_stop_sequence_len,
            random: Box::new(random),
            draft_token_buffer: Vec::new(),
            emit_buffer: Vec::new(),
        }
    }

    fn emit_token(&mut self, token: Token) -> Option<Result<Token, GenerationError>> {
        self.generated = self.generated.saturating_add(1);
        self.last_token = Some(token);
        if self.max_stop_sequence_len > 0 {
            self.recent_tokens.push(token);
            if self.recent_tokens.len() > self.max_stop_sequence_len {
                let to_drop = self.recent_tokens.len() - self.max_stop_sequence_len;
                self.recent_tokens.drain(..to_drop);
            }
        }
        let matched_stop_sequence = self
            .config
            .generation
            .stop_sequences
            .iter()
            .filter(|sequence| !sequence.is_empty())
            .any(|sequence| self.recent_tokens.ends_with(sequence));
        if self.config.generation.stop_token == Some(token) || matched_stop_sequence {
            self.state = GenerationState::Done;
        }
        Some(Ok(token))
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
        let k = self.config.draft_tokens_per_step;
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_logits = Vec::with_capacity(k);
        let mut current_token = start_token;

        draft_model.reset_cache();
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

        // 2. Target model verifies draft tokens: replay each prefix from a fixed KV checkpoint.
        let verify_start = session.consumed_tokens();
        let mut verify_sequence = vec![start_token];
        verify_sequence.extend_from_slice(&draft_tokens);

        let mut target_logits = Vec::with_capacity(draft_tokens.len() + 1);
        for i in 0..=draft_tokens.len() {
            target_model.rewind_to(verify_start);
            session.rewind_to(verify_start);
            let logits = target_model
                .forward(&verify_sequence[..=i], session)
                .map_err(GenerationError::Model)?;
            target_logits.push(logits);
        }

        let mut accepted_sequence = vec![start_token];

        // 3. Speculative decode: accept/reject draft tokens.
        let randoms: Vec<f32> = (0..=draft_tokens.len())
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
        target_model.rewind_to(verify_start);
        session.rewind_to(verify_start);
        target_model
            .forward(&accepted_sequence, session)
            .map_err(GenerationError::Model)?;

        for token in result.tokens {
            self.emit_buffer.push(token);
        }

        // 5. Update draft model KV cache to match accepted tokens.
        // Reset and replay accepted tokens through draft model.
        draft_model.reset_cache();
        let accepted_count = result.accepted_draft_tokens;
        let mut replay_token = start_token;
        for i in 0..accepted_count {
            let _ = draft_model
                .forward_token(replay_token, None)
                .map_err(|e| GenerationError::Model(ModelError::InferenceFailed(e)))?;
            replay_token = draft_tokens[i];
        }

        self.draft_model = Some(draft_model);
        self.target_model = Some(target_model);
        self.session = Some(session);
        Ok(())
    }
}

impl<T: Model> Stream for SpeculativeGenerationStream<'_, T> {
    type Item = Result<Token, GenerationError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Emit buffered tokens first.
        if !self.emit_buffer.is_empty() {
            let token = self.emit_buffer.remove(0);
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
                return Poll::Ready(self.emit_token(token));
            }
            GenerationState::Decode => {
                if let Err(e) = self.run_speculative_step() {
                    self.state = GenerationState::Done;
                    return Poll::Ready(Some(Err(e)));
                }
                if !self.emit_buffer.is_empty() {
                    let token = self.emit_buffer.remove(0);
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

pub struct GenerationStream<'a, M: Model> {
    model: Option<&'a mut M>,
    session: Option<&'a mut Session>,
    prompt: &'a [Token],
    state: GenerationState,
    config: GenerationConfig,
    generated: usize,
    last_token: Option<Token>,
    recent_tokens: Vec<Token>,
    max_stop_sequence_len: usize,
    random: Box<dyn FnMut() -> f32 + 'a>,
}

impl<'a, M: Model> GenerationStream<'a, M> {
    pub fn new(
        model: &'a mut M,
        session: &'a mut Session,
        prompt: &'a [Token],
        config: GenerationConfig,
        random: impl FnMut() -> f32 + 'a,
    ) -> Self {
        let max_stop_sequence_len = config
            .stop_sequences
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        Self {
            model: Some(model),
            session: Some(session),
            prompt,
            state: GenerationState::Prefill,
            config,
            generated: 0,
            last_token: None,
            recent_tokens: Vec::with_capacity(max_stop_sequence_len),
            max_stop_sequence_len,
            random: Box::new(random),
        }
    }
}

impl<M: Model> Stream for GenerationStream<'_, M> {
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
        if self.max_stop_sequence_len > 0 {
            self.recent_tokens.push(token);
            if self.recent_tokens.len() > self.max_stop_sequence_len {
                let to_drop = self.recent_tokens.len() - self.max_stop_sequence_len;
                self.recent_tokens.drain(..to_drop);
            }
        }
        let matched_stop_sequence = self
            .config
            .stop_sequences
            .iter()
            .filter(|sequence| !sequence.is_empty())
            .any(|sequence| self.recent_tokens.ends_with(sequence));
        if self.config.stop_token == Some(token) || matched_stop_sequence {
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
}
