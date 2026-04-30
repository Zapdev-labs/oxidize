#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    consumed_tokens: usize,
}

impl Session {
    pub fn new() -> Self {
        Self { consumed_tokens: 0 }
    }

    pub fn consumed_tokens(&self) -> usize {
        self.consumed_tokens
    }

    pub fn record_tokens(&mut self, token_count: usize) {
        self.consumed_tokens = self.consumed_tokens.saturating_add(token_count);
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

pub type Token = u32;
pub type Logits = Vec<f32>;

pub trait Model {
    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError>;
    fn vocab_size(&self) -> usize;
    fn context_size(&self) -> usize;
    fn layer_count(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    EmptyInput,
    ContextExceeded {
        context_size: usize,
        requested_total_tokens: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockModel {
        vocab_size: usize,
        context_size: usize,
        layer_count: usize,
    }

    impl Model for MockModel {
        fn forward(
            &mut self,
            tokens: &[Token],
            session: &mut Session,
        ) -> Result<Logits, ModelError> {
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
            Ok((0..self.vocab_size).map(|idx| idx as f32).collect())
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
    fn session_tracks_consumed_token_count() {
        let mut session = Session::new();
        assert_eq!(session.consumed_tokens(), 0);

        session.record_tokens(3);
        session.record_tokens(2);
        assert_eq!(session.consumed_tokens(), 5);
    }

    #[test]
    fn model_trait_supports_forward_and_metadata_queries() {
        let mut model = MockModel {
            vocab_size: 4,
            context_size: 8,
            layer_count: 2,
        };
        let mut session = Session::default();

        let logits = model
            .forward(&[1, 2, 3], &mut session)
            .expect("forward should return logits");

        assert_eq!(model.vocab_size(), 4);
        assert_eq!(model.context_size(), 8);
        assert_eq!(model.layer_count(), 2);
        assert_eq!(session.consumed_tokens(), 3);
        assert_eq!(logits, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn forward_rejects_empty_input_and_context_overflow() {
        let mut model = MockModel {
            vocab_size: 8,
            context_size: 4,
            layer_count: 1,
        };
        let mut session = Session::new();

        let empty_err = model
            .forward(&[], &mut session)
            .expect_err("empty input should fail");
        assert_eq!(empty_err, ModelError::EmptyInput);

        let context_err = model
            .forward(&[1, 2, 3, 4, 5], &mut session)
            .expect_err("input beyond context limit should fail");
        assert_eq!(
            context_err,
            ModelError::ContextExceeded {
                context_size: 4,
                requested_total_tokens: 5,
            }
        );
    }
}
