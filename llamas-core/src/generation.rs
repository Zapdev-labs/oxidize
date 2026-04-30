use crate::model::{Model, ModelError, Session, Token};
use crate::sampling::{SamplingConfig, SamplingError, sample};
use futures_core::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub stop_token: Option<Token>,
    pub stop_sequences: Vec<Vec<Token>>,
    pub sampling: SamplingConfig,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            stop_token: None,
            stop_sequences: Vec::new(),
            sampling: SamplingConfig::default(),
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
        if self.generated >= self.config.max_new_tokens || matches!(self.state, GenerationState::Done)
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

        let logits = match self.state {
            GenerationState::Prefill => {
                self.state = GenerationState::Decode;
                let prompt = self.prompt;
                match model.forward(prompt, session) {
                    Ok(logits) => logits,
                    Err(err) => {
                        self.model = Some(model);
                        self.session = Some(session);
                        self.state = GenerationState::Done;
                        return Poll::Ready(Some(Err(err.into())));
                    }
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
    use std::sync::Arc;
    use std::task::{Wake, Waker};

    #[derive(Default)]
    struct NoopWaker;

    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    fn collect_stream<M: Model>(stream: &mut GenerationStream<'_, M>) -> Vec<Result<Token, GenerationError>> {
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
        assert_eq!(items, vec![Err(GenerationError::Model(ModelError::EmptyInput))]);
    }
}
