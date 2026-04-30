use crate::generation::{GenerationConfig, GenerationError, GenerationStream};
use crate::llama::{LlamaConfig, LlamaModel};
use crate::model::{Session, Token};
use crate::sampling::SamplingConfig;
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerModelConfig {
    pub vocab_size: usize,
    pub context_size: usize,
    pub layer_count: usize,
}

impl Default for WorkerModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            context_size: 4096,
            layer_count: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerInferenceRequest {
    pub prompt_tokens: Vec<Token>,
    pub max_new_tokens: usize,
    pub model: WorkerModelConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerInferenceResponse {
    pub generated_tokens: Vec<Token>,
    pub consumed_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerMessageResponse {
    pub response: Option<WorkerInferenceResponse>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerInferenceError {
    EmptyPrompt,
    InvalidModelConfig,
    Generation(GenerationError),
}

impl std::fmt::Display for WorkerInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPrompt => write!(f, "prompt_tokens must not be empty"),
            Self::InvalidModelConfig => write!(f, "model config values must be greater than zero"),
            Self::Generation(err) => write!(f, "generation error: {err:?}"),
        }
    }
}

impl std::error::Error for WorkerInferenceError {}

pub fn run_inference_in_background_worker(
    request: &WorkerInferenceRequest,
) -> Result<WorkerInferenceResponse, WorkerInferenceError> {
    if request.prompt_tokens.is_empty() {
        return Err(WorkerInferenceError::EmptyPrompt);
    }
    if request.model.vocab_size == 0 || request.model.context_size == 0 || request.model.layer_count == 0
    {
        return Err(WorkerInferenceError::InvalidModelConfig);
    }

    let mut model = LlamaModel::new(LlamaConfig::llama2(
        request.model.vocab_size,
        request.model.context_size,
        request.model.layer_count,
    ));
    let mut session = Session::new();
    let mut stream = GenerationStream::new(
        &mut model,
        &mut session,
        &request.prompt_tokens,
        GenerationConfig {
            max_new_tokens: request.max_new_tokens,
            sampling: SamplingConfig {
                temperature: 0.01,
                ..SamplingConfig::default()
            },
            ..GenerationConfig::default()
        },
        || 0.5,
    );

    let mut generated_tokens = Vec::with_capacity(request.max_new_tokens);
    while generated_tokens.len() < request.max_new_tokens {
        match futures_core::Stream::poll_next(
            std::pin::Pin::new(&mut stream),
            &mut std::task::Context::from_waker(std::task::Waker::noop()),
        ) {
            std::task::Poll::Ready(Some(Ok(token))) => generated_tokens.push(token),
            std::task::Poll::Ready(Some(Err(err))) => {
                return Err(WorkerInferenceError::Generation(err));
            }
            std::task::Poll::Ready(None) => break,
            std::task::Poll::Pending => break,
        }
    }
    drop(stream);

    Ok(WorkerInferenceResponse {
        generated_tokens,
        consumed_tokens: session.consumed_tokens(),
    })
}

pub fn handle_worker_message(request_json: &str) -> String {
    let message = match serde_json::from_str::<WorkerInferenceRequest>(request_json) {
        Ok(request) => match run_inference_in_background_worker(&request) {
            Ok(response) => WorkerMessageResponse {
                response: Some(response),
                error: None,
            },
            Err(err) => WorkerMessageResponse {
                response: None,
                error: Some(err.to_string()),
            },
        },
        Err(err) => WorkerMessageResponse {
            response: None,
            error: Some(format!("invalid request json: {err}")),
        },
    };

    serde_json::to_string(&message).unwrap_or_else(|_| {
        "{\"response\":null,\"error\":\"failed to serialize worker response\"}".to_string()
    })
}

#[cfg_attr(all(target_arch = "wasm32", feature = "wasm"), wasm_bindgen)]
pub fn wasm_handle_worker_message(request_json: &str) -> String {
    handle_worker_message(request_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_worker_inference_generates_expected_number_of_tokens() {
        let request = WorkerInferenceRequest {
            prompt_tokens: vec![7, 9],
            max_new_tokens: 3,
            model: WorkerModelConfig::default(),
        };

        let response = run_inference_in_background_worker(&request)
            .expect("worker inference should produce response");

        assert_eq!(response.generated_tokens, vec![9, 9, 9]);
        assert_eq!(response.consumed_tokens, 4);
    }

    #[test]
    fn background_worker_inference_rejects_invalid_requests() {
        let empty_prompt = WorkerInferenceRequest {
            prompt_tokens: Vec::new(),
            max_new_tokens: 1,
            model: WorkerModelConfig::default(),
        };
        assert_eq!(
            run_inference_in_background_worker(&empty_prompt),
            Err(WorkerInferenceError::EmptyPrompt)
        );

        let invalid_model = WorkerInferenceRequest {
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            model: WorkerModelConfig {
                vocab_size: 0,
                ..WorkerModelConfig::default()
            },
        };
        assert_eq!(
            run_inference_in_background_worker(&invalid_model),
            Err(WorkerInferenceError::InvalidModelConfig)
        );
    }

    #[test]
    fn worker_message_handler_returns_json_response_payload() {
        let request = WorkerInferenceRequest {
            prompt_tokens: vec![3],
            max_new_tokens: 2,
            model: WorkerModelConfig::default(),
        };
        let request_json =
            serde_json::to_string(&request).expect("request serialization should succeed");

        let response_json = handle_worker_message(&request_json);
        let message: WorkerMessageResponse =
            serde_json::from_str(&response_json).expect("response json should decode");

        assert_eq!(
            message.response,
            Some(WorkerInferenceResponse {
                generated_tokens: vec![3, 3],
                consumed_tokens: 2
            })
        );
        assert_eq!(message.error, None);
    }
}
