use crate::model::{Logits, Model, ModelError, Session, Token};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaArchitecture {
    Llama2,
    Llama3,
    Mistral,
    Mixtral,
    Qwen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaConfig {
    pub architecture: LlamaArchitecture,
    pub vocab_size: usize,
    pub context_size: usize,
    pub layer_count: usize,
}

impl LlamaConfig {
    pub fn llama2(vocab_size: usize, context_size: usize, layer_count: usize) -> Self {
        Self {
            architecture: LlamaArchitecture::Llama2,
            vocab_size,
            context_size,
            layer_count,
        }
    }

    pub fn llama3(vocab_size: usize, context_size: usize, layer_count: usize) -> Self {
        Self {
            architecture: LlamaArchitecture::Llama3,
            vocab_size,
            context_size,
            layer_count,
        }
    }

    pub fn mistral(vocab_size: usize, context_size: usize, layer_count: usize) -> Self {
        Self {
            architecture: LlamaArchitecture::Mistral,
            vocab_size,
            context_size,
            layer_count,
        }
    }

    pub fn mixtral(vocab_size: usize, context_size: usize, layer_count: usize) -> Self {
        Self {
            architecture: LlamaArchitecture::Mixtral,
            vocab_size,
            context_size,
            layer_count,
        }
    }

    pub fn qwen(vocab_size: usize, context_size: usize, layer_count: usize) -> Self {
        Self {
            architecture: LlamaArchitecture::Qwen,
            vocab_size,
            context_size,
            layer_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaModel {
    config: LlamaConfig,
}

impl LlamaModel {
    pub fn new(config: LlamaConfig) -> Self {
        Self { config }
    }

    pub fn architecture(&self) -> LlamaArchitecture {
        self.config.architecture
    }
}

impl Model for LlamaModel {
    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        let requested_total_tokens = session.consumed_tokens().saturating_add(tokens.len());
        if requested_total_tokens > self.config.context_size {
            return Err(ModelError::ContextExceeded {
                context_size: self.config.context_size,
                requested_total_tokens,
            });
        }

        session.record_tokens(tokens.len());

        let mut logits = vec![0.0; self.config.vocab_size];
        let next_token = (tokens[tokens.len() - 1] as usize) % self.config.vocab_size;
        logits[next_token] = 1.0;
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn context_size(&self) -> usize {
        self.config.context_size
    }

    fn layer_count(&self) -> usize {
        self.config.layer_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_llama2_llama3_mistral_mixtral_and_qwen_configs() {
        let llama2 = LlamaModel::new(LlamaConfig::llama2(32_000, 4096, 32));
        let llama3 = LlamaModel::new(LlamaConfig::llama3(128_256, 8192, 32));
        let mistral = LlamaModel::new(LlamaConfig::mistral(32_000, 32_768, 32));
        let mixtral = LlamaModel::new(LlamaConfig::mixtral(32_000, 32_768, 32));
        let qwen = LlamaModel::new(LlamaConfig::qwen(151_936, 32_768, 28));

        assert_eq!(llama2.architecture(), LlamaArchitecture::Llama2);
        assert_eq!(llama3.architecture(), LlamaArchitecture::Llama3);
        assert_eq!(mistral.architecture(), LlamaArchitecture::Mistral);
        assert_eq!(mixtral.architecture(), LlamaArchitecture::Mixtral);
        assert_eq!(qwen.architecture(), LlamaArchitecture::Qwen);
        assert_eq!(llama2.vocab_size(), 32_000);
        assert_eq!(llama3.vocab_size(), 128_256);
        assert_eq!(mistral.context_size(), 32_768);
        assert_eq!(mixtral.layer_count(), 32);
        assert_eq!(qwen.vocab_size(), 151_936);
    }

    #[test]
    fn forward_tracks_session_and_returns_logits() {
        let mut model = LlamaModel::new(LlamaConfig::llama3(16, 8, 2));
        let mut session = Session::new();

        let logits = model
            .forward(&[1, 7, 3], &mut session)
            .expect("forward should succeed");

        assert_eq!(session.consumed_tokens(), 3);
        assert_eq!(logits.len(), 16);
        assert_eq!(logits[3], 1.0);
        assert_eq!(logits.iter().filter(|value| **value == 1.0).count(), 1);
    }

    #[test]
    fn forward_rejects_empty_input_and_context_overflow() {
        let mut model = LlamaModel::new(LlamaConfig::llama2(32, 2, 2));
        let mut session = Session::new();

        assert_eq!(model.forward(&[], &mut session), Err(ModelError::EmptyInput));
        assert_eq!(
            model.forward(&[4, 5, 6], &mut session),
            Err(ModelError::ContextExceeded {
                context_size: 2,
                requested_total_tokens: 3
            })
        );
    }
}
