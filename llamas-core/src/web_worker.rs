use crate::generation::{GenerationConfig, GenerationError, GenerationStream};
use crate::llama::{LlamaConfig, LlamaModel};
use crate::model::{Session, Token};
use crate::sampling::SamplingConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
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
pub struct WorkerStreamChunk {
    pub token: Token,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerMessageResponse {
    pub response: Option<WorkerInferenceResponse>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerModelCacheAction {
    CacheDownloadedModel,
    GetCachedModel,
    RemoveCachedModel,
    ClearCache,
    CacheStats,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerModelCacheRequest {
    pub action: WorkerModelCacheAction,
    pub model_id: Option<String>,
    pub model_bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerModelCacheResponse {
    pub model_id: Option<String>,
    pub model_bytes: Option<Vec<u8>>,
    pub cached: Option<bool>,
    pub cached_models: usize,
    pub cached_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerModelCacheMessageResponse {
    pub response: Option<WorkerModelCacheResponse>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerModelCacheError {
    MissingModelId,
    MissingModelBytes,
}

impl std::fmt::Display for WorkerModelCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingModelId => write!(f, "model_id is required for this action"),
            Self::MissingModelBytes => write!(f, "model_bytes is required for cache_downloaded_model"),
        }
    }
}

impl std::error::Error for WorkerModelCacheError {}

fn worker_model_cache() -> &'static Mutex<HashMap<String, Vec<u8>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Vec<u8>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_stats(cache: &HashMap<String, Vec<u8>>) -> (usize, usize) {
    (cache.len(), cache.values().map(std::vec::Vec::len).sum())
}

pub fn handle_worker_model_cache_message(request_json: &str) -> String {
    fn run_request(
        request: WorkerModelCacheRequest,
    ) -> Result<WorkerModelCacheResponse, WorkerModelCacheError> {
        let mut cache = worker_model_cache()
            .lock()
            .expect("worker model cache mutex should not be poisoned");
        match request.action {
            WorkerModelCacheAction::CacheDownloadedModel => {
                let model_id = request.model_id.ok_or(WorkerModelCacheError::MissingModelId)?;
                let model_bytes = request
                    .model_bytes
                    .ok_or(WorkerModelCacheError::MissingModelBytes)?;
                cache.insert(model_id.clone(), model_bytes.clone());
                let (cached_models, cached_bytes) = cache_stats(&cache);
                Ok(WorkerModelCacheResponse {
                    model_id: Some(model_id),
                    model_bytes: Some(model_bytes),
                    cached: Some(true),
                    cached_models,
                    cached_bytes,
                })
            }
            WorkerModelCacheAction::GetCachedModel => {
                let model_id = request.model_id.ok_or(WorkerModelCacheError::MissingModelId)?;
                let model_bytes = cache.get(&model_id).cloned();
                let (cached_models, cached_bytes) = cache_stats(&cache);
                Ok(WorkerModelCacheResponse {
                    model_id: Some(model_id),
                    cached: Some(model_bytes.is_some()),
                    model_bytes,
                    cached_models,
                    cached_bytes,
                })
            }
            WorkerModelCacheAction::RemoveCachedModel => {
                let model_id = request.model_id.ok_or(WorkerModelCacheError::MissingModelId)?;
                let cached = cache.remove(&model_id).is_some();
                let (cached_models, cached_bytes) = cache_stats(&cache);
                Ok(WorkerModelCacheResponse {
                    model_id: Some(model_id),
                    model_bytes: None,
                    cached: Some(cached),
                    cached_models,
                    cached_bytes,
                })
            }
            WorkerModelCacheAction::ClearCache => {
                cache.clear();
                Ok(WorkerModelCacheResponse {
                    model_id: None,
                    model_bytes: None,
                    cached: None,
                    cached_models: 0,
                    cached_bytes: 0,
                })
            }
            WorkerModelCacheAction::CacheStats => {
                let (cached_models, cached_bytes) = cache_stats(&cache);
                Ok(WorkerModelCacheResponse {
                    model_id: None,
                    model_bytes: None,
                    cached: None,
                    cached_models,
                    cached_bytes,
                })
            }
        }
    }

    let message = match serde_json::from_str::<WorkerModelCacheRequest>(request_json) {
        Ok(request) => match run_request(request) {
            Ok(response) => WorkerModelCacheMessageResponse {
                response: Some(response),
                error: None,
            },
            Err(err) => WorkerModelCacheMessageResponse {
                response: None,
                error: Some(err.to_string()),
            },
        },
        Err(err) => WorkerModelCacheMessageResponse {
            response: None,
            error: Some(format!("invalid request json: {err}")),
        },
    };

    serde_json::to_string(&message).unwrap_or_else(|_| {
        "{\"response\":null,\"error\":\"failed to serialize worker model cache response\"}"
            .to_string()
    })
}

pub fn run_inference_in_background_worker(
    request: &WorkerInferenceRequest,
) -> Result<WorkerInferenceResponse, WorkerInferenceError> {
    run_inference_streaming_in_background_worker(request, |_| {})
}

pub fn run_inference_streaming_in_background_worker<F>(
    request: &WorkerInferenceRequest,
    mut on_token: F,
) -> Result<WorkerInferenceResponse, WorkerInferenceError>
where
    F: FnMut(WorkerStreamChunk),
{
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
            std::task::Poll::Ready(Some(Ok(token))) => {
                generated_tokens.push(token);
                on_token(WorkerStreamChunk {
                    token,
                    index: generated_tokens.len() - 1,
                });
            }
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

pub fn collect_worker_stream(request_json: &str) -> String {
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct WorkerStreamResponse {
        chunks: Vec<WorkerStreamChunk>,
        response: Option<WorkerInferenceResponse>,
        error: Option<String>,
    }

    let mut chunks = Vec::new();
    let message = match serde_json::from_str::<WorkerInferenceRequest>(request_json) {
        Ok(request) => match run_inference_streaming_in_background_worker(&request, |chunk| {
            chunks.push(chunk);
        }) {
            Ok(response) => WorkerStreamResponse {
                chunks,
                response: Some(response),
                error: None,
            },
            Err(err) => WorkerStreamResponse {
                chunks,
                response: None,
                error: Some(err.to_string()),
            },
        },
        Err(err) => WorkerStreamResponse {
            chunks,
            response: None,
            error: Some(format!("invalid request json: {err}")),
        },
    };

    serde_json::to_string(&message).unwrap_or_else(|_| {
        "{\"chunks\":[],\"response\":null,\"error\":\"failed to serialize worker stream response\"}"
            .to_string()
    })
}

pub const WASM_WORKER_TYPESCRIPT_BINDINGS: &str = r#"
export interface LlamasWorkerModelConfig {
  vocab_size: number;
  context_size: number;
  layer_count: number;
}

export interface LlamasWorkerInferenceRequest {
  prompt_tokens: number[];
  max_new_tokens: number;
  model: LlamasWorkerModelConfig;
}

export interface LlamasWorkerInferenceResponse {
  generated_tokens: number[];
  consumed_tokens: number;
}

export interface LlamasWorkerStreamChunk {
  token: number;
  index: number;
}

export interface LlamasWorkerMessageResponse {
  response: LlamasWorkerInferenceResponse | null;
  error: string | null;
}

export type LlamasWorkerModelCacheAction =
  | "cache_downloaded_model"
  | "get_cached_model"
  | "remove_cached_model"
  | "clear_cache"
  | "cache_stats";

export interface LlamasWorkerModelCacheRequest {
  action: LlamasWorkerModelCacheAction;
  model_id?: string;
  model_bytes?: number[];
}

export interface LlamasWorkerModelCacheResponse {
  model_id: string | null;
  model_bytes: number[] | null;
  cached: boolean | null;
  cached_models: number;
  cached_bytes: number;
}

export interface LlamasWorkerModelCacheMessageResponse {
  response: LlamasWorkerModelCacheResponse | null;
  error: string | null;
}

export interface LlamasWorkerStreamResponse {
  chunks: LlamasWorkerStreamChunk[];
  response: LlamasWorkerInferenceResponse | null;
  error: string | null;
}
"#;

#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    wasm_bindgen(typescript_custom_section)
)]
pub const WASM_WORKER_TYPES: &str = WASM_WORKER_TYPESCRIPT_BINDINGS;

#[cfg_attr(all(target_arch = "wasm32", feature = "wasm"), wasm_bindgen)]
pub fn wasm_handle_worker_message(request_json: &str) -> String {
    handle_worker_message(request_json)
}

#[cfg_attr(all(target_arch = "wasm32", feature = "wasm"), wasm_bindgen)]
pub fn wasm_collect_worker_stream(request_json: &str) -> String {
    collect_worker_stream(request_json)
}

#[cfg_attr(all(target_arch = "wasm32", feature = "wasm"), wasm_bindgen)]
pub fn wasm_handle_worker_model_cache_message(request_json: &str) -> String {
    handle_worker_model_cache_message(request_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker_model_cache_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_worker_model_cache_for_test() {
        let response_json = handle_worker_model_cache_message(r#"{"action":"clear_cache"}"#);
        let message: WorkerModelCacheMessageResponse =
            serde_json::from_str(&response_json).expect("clear cache response should decode");
        assert_eq!(message.error, None);
    }

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

    #[test]
    fn streaming_worker_inference_emits_chunks_for_each_generated_token() {
        let request = WorkerInferenceRequest {
            prompt_tokens: vec![5],
            max_new_tokens: 3,
            model: WorkerModelConfig::default(),
        };

        let mut chunks = Vec::new();
        let response = run_inference_streaming_in_background_worker(&request, |chunk| {
            chunks.push(chunk);
        })
        .expect("streaming worker inference should produce response");

        assert_eq!(
            chunks,
            vec![
                WorkerStreamChunk { token: 5, index: 0 },
                WorkerStreamChunk { token: 5, index: 1 },
                WorkerStreamChunk { token: 5, index: 2 },
            ]
        );
        assert_eq!(response.generated_tokens, vec![5, 5, 5]);
    }

    #[test]
    fn collect_worker_stream_returns_chunks_and_final_response() {
        let request = WorkerInferenceRequest {
            prompt_tokens: vec![2],
            max_new_tokens: 2,
            model: WorkerModelConfig::default(),
        };
        let request_json =
            serde_json::to_string(&request).expect("request serialization should succeed");

        let response_json = collect_worker_stream(&request_json);
        let response: serde_json::Value =
            serde_json::from_str(&response_json).expect("stream response should decode");

        assert_eq!(response["error"], serde_json::Value::Null);
        assert_eq!(response["chunks"][0]["token"], 2);
        assert_eq!(response["chunks"][0]["index"], 0);
        assert_eq!(response["chunks"][1]["token"], 2);
        assert_eq!(response["chunks"][1]["index"], 1);
        assert_eq!(response["response"]["generated_tokens"], serde_json::json!([2, 2]));
    }

    #[test]
    fn worker_typescript_bindings_include_typed_contracts() {
        assert!(WASM_WORKER_TYPESCRIPT_BINDINGS.contains("interface LlamasWorkerInferenceRequest"));
        assert!(WASM_WORKER_TYPESCRIPT_BINDINGS.contains("interface LlamasWorkerMessageResponse"));
        assert!(WASM_WORKER_TYPESCRIPT_BINDINGS.contains("interface LlamasWorkerStreamResponse"));
        assert!(WASM_WORKER_TYPESCRIPT_BINDINGS.contains("type LlamasWorkerModelCacheAction"));
        assert!(WASM_WORKER_TYPESCRIPT_BINDINGS.contains("interface LlamasWorkerModelCacheRequest"));
        assert!(
            WASM_WORKER_TYPESCRIPT_BINDINGS
                .contains("interface LlamasWorkerModelCacheMessageResponse")
        );
    }

    #[test]
    fn worker_model_cache_can_store_and_read_model_bytes() {
        let _guard = worker_model_cache_test_lock()
            .lock()
            .expect("test lock should not be poisoned");
        clear_worker_model_cache_for_test();

        let put_response_json = handle_worker_model_cache_message(
            r#"{"action":"cache_downloaded_model","model_id":"tiny","model_bytes":[1,2,3]}"#,
        );
        let put_message: WorkerModelCacheMessageResponse =
            serde_json::from_str(&put_response_json).expect("cache put response should decode");
        assert_eq!(put_message.error, None);

        let put_response = put_message.response.expect("cache put should have response");
        assert_eq!(put_response.model_id, Some("tiny".to_string()));
        assert_eq!(put_response.model_bytes, Some(vec![1, 2, 3]));
        assert_eq!(put_response.cached, Some(true));
        assert_eq!(put_response.cached_models, 1);
        assert_eq!(put_response.cached_bytes, 3);

        let get_response_json =
            handle_worker_model_cache_message(r#"{"action":"get_cached_model","model_id":"tiny"}"#);
        let get_message: WorkerModelCacheMessageResponse =
            serde_json::from_str(&get_response_json).expect("cache get response should decode");
        assert_eq!(get_message.error, None);

        let get_response = get_message.response.expect("cache get should have response");
        assert_eq!(get_response.model_id, Some("tiny".to_string()));
        assert_eq!(get_response.model_bytes, Some(vec![1, 2, 3]));
        assert_eq!(get_response.cached, Some(true));
        assert_eq!(get_response.cached_models, 1);
        assert_eq!(get_response.cached_bytes, 3);
    }

    #[test]
    fn worker_model_cache_remove_and_clear_update_cache_stats() {
        let _guard = worker_model_cache_test_lock()
            .lock()
            .expect("test lock should not be poisoned");
        clear_worker_model_cache_for_test();
        handle_worker_model_cache_message(
            r#"{"action":"cache_downloaded_model","model_id":"tiny","model_bytes":[1,2,3]}"#,
        );
        handle_worker_model_cache_message(
            r#"{"action":"cache_downloaded_model","model_id":"small","model_bytes":[7,8]}"#,
        );

        let stats_response_json = handle_worker_model_cache_message(r#"{"action":"cache_stats"}"#);
        let stats_message: WorkerModelCacheMessageResponse =
            serde_json::from_str(&stats_response_json).expect("stats response should decode");
        let stats = stats_message.response.expect("stats should be present");
        assert_eq!(stats.cached_models, 2);
        assert_eq!(stats.cached_bytes, 5);

        let remove_response_json = handle_worker_model_cache_message(
            r#"{"action":"remove_cached_model","model_id":"tiny"}"#,
        );
        let remove_message: WorkerModelCacheMessageResponse =
            serde_json::from_str(&remove_response_json).expect("remove response should decode");
        let remove = remove_message.response.expect("remove should be present");
        assert_eq!(remove.cached, Some(true));
        assert_eq!(remove.cached_models, 1);
        assert_eq!(remove.cached_bytes, 2);

        let clear_response_json = handle_worker_model_cache_message(r#"{"action":"clear_cache"}"#);
        let clear_message: WorkerModelCacheMessageResponse =
            serde_json::from_str(&clear_response_json).expect("clear response should decode");
        let clear = clear_message.response.expect("clear should be present");
        assert_eq!(clear.cached_models, 0);
        assert_eq!(clear.cached_bytes, 0);
    }

    #[test]
    fn worker_model_cache_rejects_missing_required_fields() {
        let _guard = worker_model_cache_test_lock()
            .lock()
            .expect("test lock should not be poisoned");
        clear_worker_model_cache_for_test();

        let missing_id_json =
            handle_worker_model_cache_message(r#"{"action":"get_cached_model"}"#);
        let missing_id: WorkerModelCacheMessageResponse =
            serde_json::from_str(&missing_id_json).expect("missing-id response should decode");
        assert_eq!(
            missing_id.error,
            Some("model_id is required for this action".to_string())
        );

        let missing_bytes_json = handle_worker_model_cache_message(
            r#"{"action":"cache_downloaded_model","model_id":"tiny"}"#,
        );
        let missing_bytes: WorkerModelCacheMessageResponse =
            serde_json::from_str(&missing_bytes_json).expect("missing-bytes response should decode");
        assert_eq!(
            missing_bytes.error,
            Some("model_bytes is required for cache_downloaded_model".to_string())
        );
    }
}
