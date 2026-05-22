use std::{
    collections::BTreeMap,
    collections::VecDeque,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use futures_util::{Stream, stream};
use oxidize_core::{
    generation::{GenerationConfig, GenerationStream},
    gguf::{GgufMetadataValue, MappedGgufFile},
    inference::{InferenceConfig, InferenceModel},
    layer_wise::LayerWiseModel,
    model::{Model, Session, Token},
    model_loader::{GgufModelLoader, ModelLoader},
    sampling::SamplingConfig,
    tensor::DType,
    tokenizer::{
        ChatMessage, EncodeOptions, LoadedTokenizer, load_tokenizer_from_gguf_metadata,
        process_chat_template,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Mutex as StdMutex;
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::{Instant as TokioInstant, sleep_until};

#[derive(Debug, Parser)]
#[command(name = "oxidize-server")]
struct Args {
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value = "oxidize-default")]
    model_id: String,
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    ctx_size: Option<usize>,
    #[arg(long, default_value_t = false)]
    cpu_optimized: bool,
    #[arg(long, default_value_t = false)]
    ram_offload: bool,
    #[arg(long, default_value_t = false)]
    mmap_prefetch: bool,
    #[arg(long, default_value_t = false)]
    mmap_hugepages: bool,
    #[arg(long, default_value_t = false)]
    layer_wise: bool,
    #[arg(long, default_value_t = 1)]
    layer_cache: usize,
}

#[derive(Debug, Clone, Copy)]
struct RequestLimitConfig {
    requests_per_second: usize,
    max_in_flight: usize,
    max_queue: usize,
}

impl Default for RequestLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 64,
            max_in_flight: 8,
            max_queue: 128,
        }
    }
}

#[derive(Clone)]
struct AppState {
    limiter: Arc<RequestLimiter>,
    batcher: Arc<ContinuousBatcher>,
    auth: AuthConfig,
    model: Option<Arc<ModelRuntime>>,
}

#[derive(Clone)]
struct AuthConfig {
    api_key: Option<Arc<str>>,
}

struct ModelRuntime {
    id: String,
    tokenizer: LoadedTokenizer,
    chat_template: Option<String>,
    model: StdMutex<LoadedModel>,
    defaults: GenerationDefaults,
}

#[derive(Debug, Clone, Copy)]
struct GenerationDefaults {
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
}

enum LoadedModel {
    Inference(Box<InferenceModel>),
    LayerWise(Box<LayerWiseModel>),
}

impl Model for LoadedModel {
    fn forward(
        &mut self,
        tokens: &[Token],
        session: &mut Session,
    ) -> Result<Vec<f32>, oxidize_core::model::ModelError> {
        match self {
            Self::Inference(model) => model.forward(tokens, session),
            Self::LayerWise(model) => model.forward(tokens, session),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Inference(model) => model.vocab_size(),
            Self::LayerWise(model) => model.vocab_size(),
        }
    }

    fn context_size(&self) -> usize {
        match self {
            Self::Inference(model) => model.context_size(),
            Self::LayerWise(model) => model.context_size(),
        }
    }

    fn layer_count(&self) -> usize {
        match self {
            Self::Inference(model) => model.layer_count(),
            Self::LayerWise(model) => model.layer_count(),
        }
    }

    fn rewind_to(&mut self, consumed_tokens: usize) {
        match self {
            Self::Inference(model) => model.rewind_to(consumed_tokens),
            Self::LayerWise(model) => model.rewind_to(consumed_tokens),
        }
    }
}

#[cfg(test)]
fn build_app() -> Router {
    let api_key = std::env::var("OXIDIZE_API_KEY")
        .ok()
        .filter(|value| !value.is_empty());
    build_app_with_config(RequestLimitConfig::default(), api_key, None)
}

#[cfg(test)]
fn build_app_with_limits(config: RequestLimitConfig) -> Router {
    build_app_with_config(config, None, None)
}

fn build_app_with_config(
    config: RequestLimitConfig,
    api_key: Option<String>,
    model: Option<Arc<ModelRuntime>>,
) -> Router {
    let state = AppState {
        limiter: Arc::new(RequestLimiter::new(config)),
        batcher: Arc::new(ContinuousBatcher::default()),
        auth: AuthConfig {
            api_key: api_key.map(Arc::<str>::from),
        },
        model,
    };
    build_app_with_state(state)
}

fn build_app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/openapi.json", get(openapi))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            enforce_request_limits,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            enforce_api_key,
        ))
        .layer(middleware::from_fn(log_request_response))
        .with_state(state)
}

async fn healthz() -> StatusCode {
    StatusCode::OK
}

async fn livez() -> StatusCode {
    StatusCode::OK
}

async fn readyz() -> StatusCode {
    StatusCode::OK
}

async fn openapi() -> Json<Value> {
    Json(openapi_spec())
}

fn openapi_spec() -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "oxidize-server API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "OpenAI-compatible endpoints exposed by oxidize-server."
        },
        "servers": [{ "url": "/" }],
        "paths": {
            "/healthz": {
                "get": {
                    "summary": "Health check",
                    "responses": {
                        "200": { "description": "OK" }
                    }
                }
            },
            "/livez": {
                "get": {
                    "summary": "Liveness check",
                    "responses": {
                        "200": { "description": "OK" }
                    }
                }
            },
            "/readyz": {
                "get": {
                    "summary": "Readiness check",
                    "responses": {
                        "200": { "description": "OK" }
                    }
                }
            },
            "/v1/chat/completions": {
                "post": {
                    "summary": "Create chat completion",
                    "security": [{ "ApiKeyAuth": [] }, { "BearerAuth": [] }],
                    "responses": {
                        "200": { "description": "Chat completion response" },
                        "401": { "description": "Invalid API key" }
                    }
                }
            },
            "/v1/completions": {
                "post": {
                    "summary": "Create text completion",
                    "security": [{ "ApiKeyAuth": [] }, { "BearerAuth": [] }],
                    "responses": {
                        "200": { "description": "Completion response" },
                        "401": { "description": "Invalid API key" }
                    }
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "List models",
                    "security": [{ "ApiKeyAuth": [] }, { "BearerAuth": [] }],
                    "responses": {
                        "200": { "description": "Model list" },
                        "401": { "description": "Invalid API key" }
                    }
                }
            },
            "/v1/embeddings": {
                "post": {
                    "summary": "Create embeddings",
                    "security": [{ "ApiKeyAuth": [] }, { "BearerAuth": [] }],
                    "responses": {
                        "200": { "description": "Embeddings response" },
                        "401": { "description": "Invalid API key" }
                    }
                }
            }
        },
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "x-api-key"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
        }
    })
}

async fn log_request_response(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    tracing::info!("{}", request_log_message(method.as_ref(), &path));
    let response = next.run(request).await;
    tracing::info!(
        "{}",
        response_log_message(method.as_ref(), &path, response.status())
    );
    response
}

struct RequestLimiter {
    config: RequestLimitConfig,
    queue_slots: Arc<Semaphore>,
    active_slots: Arc<Semaphore>,
    accepted_at: Mutex<VecDeque<Instant>>,
}

struct RequestPermit {
    _queue: OwnedSemaphorePermit,
    _active: OwnedSemaphorePermit,
}

#[derive(Debug, Clone, Copy)]
struct ContinuousBatchConfig {
    max_batch_size: usize,
    max_wait: Duration,
}

impl Default for ContinuousBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_wait: Duration::from_millis(5),
        }
    }
}

struct ContinuousBatcher {
    config: ContinuousBatchConfig,
    state: Mutex<BatchState>,
}

struct BatchState {
    open: Option<OpenBatch>,
}

struct OpenBatch {
    size: usize,
    deadline: TokioInstant,
    notify: Arc<Notify>,
}

impl Default for ContinuousBatcher {
    fn default() -> Self {
        Self::new(ContinuousBatchConfig::default())
    }
}

impl ContinuousBatcher {
    fn new(config: ContinuousBatchConfig) -> Self {
        Self {
            config,
            state: Mutex::new(BatchState { open: None }),
        }
    }

    async fn wait_turn(&self) {
        let (deadline, notify) = {
            let now = TokioInstant::now();
            let mut state = self.state.lock().await;
            let max_batch_size = self.config.max_batch_size.max(1);

            if state
                .open
                .as_ref()
                .is_some_and(|batch| batch.deadline <= now)
            {
                state.open = None;
            }

            if state.open.is_none() {
                state.open = Some(OpenBatch {
                    size: 0,
                    deadline: now + self.config.max_wait,
                    notify: Arc::new(Notify::new()),
                });
            }

            let batch = state.open.as_mut().expect("batch should exist");
            batch.size += 1;
            let is_full = batch.size >= max_batch_size;
            let deadline = batch.deadline;
            let notify = Arc::clone(&batch.notify);

            if is_full {
                state.open = None;
                notify.notify_waiters();
                return;
            }

            (deadline, notify)
        };

        tokio::select! {
            _ = sleep_until(deadline) => {}
            _ = notify.notified() => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestLimitError {
    RateLimited,
    QueueFull,
}

impl RequestLimiter {
    fn new(config: RequestLimitConfig) -> Self {
        let total_slots = config.max_in_flight.saturating_add(config.max_queue).max(1);
        let active_slots = config.max_in_flight.max(1);
        Self {
            config,
            queue_slots: Arc::new(Semaphore::new(total_slots)),
            active_slots: Arc::new(Semaphore::new(active_slots)),
            accepted_at: Mutex::new(VecDeque::new()),
        }
    }

    async fn try_acquire(&self) -> Result<RequestPermit, RequestLimitError> {
        let queue = self
            .queue_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| RequestLimitError::QueueFull)?;
        if !self.try_accept_rate().await {
            return Err(RequestLimitError::RateLimited);
        }
        let active = self
            .active_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| RequestLimitError::QueueFull)?;
        Ok(RequestPermit {
            _queue: queue,
            _active: active,
        })
    }

    async fn try_accept_rate(&self) -> bool {
        if self.config.requests_per_second == 0 {
            return false;
        }
        let now = Instant::now();
        let mut accepted = self.accepted_at.lock().await;
        let cutoff = now.checked_sub(Duration::from_secs(1)).unwrap_or(now);
        while accepted.front().is_some_and(|instant| *instant <= cutoff) {
            accepted.pop_front();
        }
        if accepted.len() >= self.config.requests_per_second {
            return false;
        }
        accepted.push_back(now);
        true
    }
}

async fn enforce_request_limits(
    axum::extract::State(state): axum::extract::State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if is_generation_route(request.uri().path()) {
        state.batcher.wait_turn().await;
    }
    match state.limiter.try_acquire().await {
        Ok(_permit) => next.run(request).await,
        Err(RequestLimitError::RateLimited) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "rate limit exceeded"})),
        )
            .into_response(),
        Err(RequestLimitError::QueueFull) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "request queue full"})),
        )
            .into_response(),
    }
}

fn is_generation_route(path: &str) -> bool {
    matches!(path, "/v1/chat/completions" | "/v1/completions")
}

async fn enforce_api_key(
    axum::extract::State(state): axum::extract::State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    if !path.starts_with("/v1/") {
        return next.run(request).await;
    }
    let Some(expected_key) = state.auth.api_key.as_deref() else {
        return next.run(request).await;
    };
    if request_has_api_key(request.headers(), expected_key) {
        return next.run(request).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "invalid api key"})),
    )
        .into_response()
}

fn request_has_api_key(headers: &axum::http::HeaderMap, expected_key: &str) -> bool {
    headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == expected_key)
        || headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .is_some_and(|token| token == expected_key)
}

fn request_log_message(method: &str, path: &str) -> String {
    format!("request {method} {path}")
}

fn response_log_message(method: &str, path: &str, status: StatusCode) -> String {
    format!("response {method} {path} {}", status.as_u16())
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessageInput>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ChatMessageInput {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Value },
}

impl ResponseFormat {
    fn output_text(&self) -> &'static str {
        match self {
            Self::Text => "",
            Self::JsonObject => "{}",
            Self::JsonSchema { json_schema } => {
                let _ = json_schema;
                "{}"
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingsRequest {
    model: String,
    input: Value,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelData>,
}

#[derive(Debug, Serialize)]
struct ModelData {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Debug, Clone)]
struct GenerationRequest {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
}

#[derive(Debug, Clone)]
struct GenerationResult {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    if let Some(runtime) = state.model.clone() {
        if payload.model != runtime.id {
            return model_not_found(&payload.model);
        }
        let prompt = render_chat_prompt(&runtime, &payload.messages);
        let model = payload.model;
        let stream = payload.stream;
        let generated = generate_text(
            runtime,
            GenerationRequest {
                prompt,
                max_tokens: payload.max_tokens,
                temperature: payload.temperature,
                top_p: payload.top_p,
                top_k: payload.top_k,
            },
        )
        .await;
        return match generated {
            Ok(result) if stream => chat_completion_stream_response(model, result.text),
            Ok(result) => chat_completion_response(model, result),
            Err(error) => generation_error_response(error),
        };
    }

    let _input_size: usize = payload
        .messages
        .iter()
        .map(|message| message.role.len() + message.content.len())
        .sum();
    let response_content = payload
        .response_format
        .as_ref()
        .map_or("", ResponseFormat::output_text);

    if payload.stream {
        let model = payload.model;
        let model_for_end = model.clone();
        let stream = stream::iter(vec![
            Ok::<Event, std::convert::Infallible>(
                Event::default().data(
                    json!({
                        "id": "chatcmpl-placeholder",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": { "role": "assistant", "content": response_content },
                                "finish_reason": null
                            }
                        ]
                    })
                    .to_string(),
                ),
            ),
            Ok(Event::default().data(
                json!({
                    "id": "chatcmpl-placeholder",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model_for_end,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
                })
                .to_string(),
            )),
            Ok(Event::default().data("[DONE]")),
        ]);

        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    (
        StatusCode::OK,
        Json(json!({
            "id": "chatcmpl-placeholder",
            "object": "chat.completion",
            "created": 0,
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })),
    )
        .into_response()
}

async fn completions(
    State(state): State<AppState>,
    Json(payload): Json<CompletionRequest>,
) -> Response {
    if let Some(runtime) = state.model.clone() {
        if payload.model != runtime.id {
            return model_not_found(&payload.model);
        }
        let model = payload.model;
        let stream = payload.stream;
        let generated = generate_text(
            runtime,
            GenerationRequest {
                prompt: payload.prompt,
                max_tokens: payload.max_tokens,
                temperature: payload.temperature,
                top_p: payload.top_p,
                top_k: payload.top_k,
            },
        )
        .await;
        return match generated {
            Ok(result) if stream => completion_stream_response(model, result.text),
            Ok(result) => completion_response(model, result),
            Err(error) => generation_error_response(error),
        };
    }

    let _ = &payload.prompt;
    let response_text = payload
        .response_format
        .as_ref()
        .map_or("", ResponseFormat::output_text);

    if payload.stream {
        let model = payload.model;
        let model_for_end = model.clone();
        let stream = stream::iter(vec![
            Ok::<Event, std::convert::Infallible>(
                Event::default().data(
                    json!({
                        "id": "cmpl-placeholder",
                        "object": "text_completion",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "text": response_text,
                                "finish_reason": null
                            }
                        ]
                    })
                    .to_string(),
                ),
            ),
            Ok(Event::default().data(
                json!({
                    "id": "cmpl-placeholder",
                    "object": "text_completion",
                    "created": 0,
                    "model": model_for_end,
                    "choices": [{ "index": 0, "text": response_text, "finish_reason": "stop" }]
                })
                .to_string(),
            )),
            Ok(Event::default().data("[DONE]")),
        ]);

        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    (
        StatusCode::OK,
        Json(json!({
            "id": "cmpl-placeholder",
            "object": "text_completion",
            "created": 0,
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "text": response_text,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })),
    )
        .into_response()
}

fn render_chat_prompt(runtime: &ModelRuntime, messages: &[ChatMessageInput]) -> String {
    let chat_messages = messages
        .iter()
        .map(|message| ChatMessage {
            role: message.role.as_str(),
            content: message.content.as_str(),
        })
        .collect::<Vec<_>>();

    runtime.chat_template.as_ref().map_or_else(
        || {
            let mut prompt = String::new();
            for message in messages {
                prompt.push_str(&message.role);
                prompt.push_str(": ");
                prompt.push_str(&message.content);
                prompt.push('\n');
            }
            prompt.push_str("assistant: ");
            prompt
        },
        |template| process_chat_template(template, &chat_messages, true),
    )
}

async fn generate_text(
    runtime: Arc<ModelRuntime>,
    request: GenerationRequest,
) -> Result<GenerationResult, String> {
    tokio::task::spawn_blocking(move || generate_text_blocking(&runtime, request))
        .await
        .map_err(|error| format!("generation task failed: {error}"))?
}

fn generate_text_blocking(
    runtime: &ModelRuntime,
    request: GenerationRequest,
) -> Result<GenerationResult, String> {
    let mut model = runtime
        .model
        .lock()
        .map_err(|_| "model lock poisoned".to_owned())?;
    model.rewind_to(0);
    let mut session = Session::new();
    let prompt_tokens = runtime.tokenizer.encode_with_special_tokens(
        &request.prompt,
        EncodeOptions {
            add_bos: true,
            add_eos: false,
            pad_to: None,
        },
    );
    let max_tokens = request.max_tokens.unwrap_or(runtime.defaults.max_tokens);
    let temperature = request.temperature.unwrap_or(runtime.defaults.temperature);
    let top_p = request.top_p.or(runtime.defaults.top_p);
    let top_k = request.top_k.or(runtime.defaults.top_k);
    let config = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: runtime.tokenizer.special_tokens().eos,
        suppressed_tokens: suppressed_generation_tokens(&runtime.tokenizer, model.vocab_size()),
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            ..SamplingConfig::default()
        },
        ..GenerationConfig::default()
    };
    let mut rng = rand::thread_rng();
    let mut stream =
        GenerationStream::new(&mut *model, &mut session, &prompt_tokens, config, || {
            rand::Rng::r#gen::<f32>(&mut rng)
        });
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);
    let mut generated_tokens = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => generated_tokens.push(token),
            Poll::Ready(Some(Err(error))) => return Err(format!("generation error: {error:?}")),
            Poll::Ready(None) | Poll::Pending => break,
        }
    }

    let text = runtime
        .tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    Ok(GenerationResult {
        text,
        prompt_tokens: prompt_tokens.len(),
        completion_tokens: generated_tokens.len(),
    })
}

fn chat_completion_response(model: String, result: GenerationResult) -> Response {
    (
        StatusCode::OK,
        Json(json!({
            "id": "chatcmpl-oxidize",
            "object": "chat.completion",
            "created": unix_timestamp(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": usage_json(result.prompt_tokens, result.completion_tokens)
        })),
    )
        .into_response()
}

fn completion_response(model: String, result: GenerationResult) -> Response {
    (
        StatusCode::OK,
        Json(json!({
            "id": "cmpl-oxidize",
            "object": "text_completion",
            "created": unix_timestamp(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": result.text,
                    "finish_reason": "stop"
                }
            ],
            "usage": usage_json(result.prompt_tokens, result.completion_tokens)
        })),
    )
        .into_response()
}

fn chat_completion_stream_response(model: String, content: String) -> Response {
    let model_for_end = model.clone();
    let stream = stream::iter(vec![
        Ok::<Event, std::convert::Infallible>(
            Event::default().data(
                json!({
                    "id": "chatcmpl-oxidize",
                    "object": "chat.completion.chunk",
                    "created": unix_timestamp(),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": { "role": "assistant", "content": content },
                            "finish_reason": null
                        }
                    ]
                })
                .to_string(),
            ),
        ),
        Ok(Event::default().data(
            json!({
                "id": "chatcmpl-oxidize",
                "object": "chat.completion.chunk",
                "created": unix_timestamp(),
                "model": model_for_end,
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
            })
            .to_string(),
        )),
        Ok(Event::default().data("[DONE]")),
    ]);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn completion_stream_response(model: String, text: String) -> Response {
    let model_for_end = model.clone();
    let stream = stream::iter(vec![
        Ok::<Event, std::convert::Infallible>(
            Event::default().data(
                json!({
                    "id": "cmpl-oxidize",
                    "object": "text_completion",
                    "created": unix_timestamp(),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": null
                        }
                    ]
                })
                .to_string(),
            ),
        ),
        Ok(Event::default().data(
            json!({
                "id": "cmpl-oxidize",
                "object": "text_completion",
                "created": unix_timestamp(),
                "model": model_for_end,
                "choices": [{ "index": 0, "text": "", "finish_reason": "stop" }]
            })
            .to_string(),
        )),
        Ok(Event::default().data("[DONE]")),
    ]);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn usage_json(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens.saturating_add(completion_tokens)
    })
}

fn generation_error_response(error: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": error, "type": "generation_error"}})),
    )
        .into_response()
}

fn model_not_found(model: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": format!("model '{model}' is not loaded"),
                "type": "model_not_found"
            }
        })),
    )
        .into_response()
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelData {
            id: state.model.as_ref().map_or_else(
                || "oxidize-default".to_owned(),
                |runtime| runtime.id.clone(),
            ),
            object: "model",
            owned_by: "oxidize",
        }],
    })
}

async fn embeddings(Json(payload): Json<EmbeddingsRequest>) -> (StatusCode, Json<Value>) {
    let _ = &payload.input;
    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [],
                    "index": 0
                }
            ],
            "model": payload.model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        })),
    )
}

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

fn suppressed_generation_tokens(tokenizer: &LoadedTokenizer, vocab_size: usize) -> Vec<u32> {
    let special_tokens = tokenizer.special_tokens();
    let mut suppressed = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for token in [
        special_tokens.unknown,
        special_tokens.bos,
        special_tokens.pad,
        special_tokens.separator,
        special_tokens.cls,
        special_tokens.mask,
    ]
    .into_iter()
    .flatten()
    {
        if seen.insert(token) {
            suppressed.push(token);
        }
    }

    for token in 0..vocab_size {
        let token = token as u32;
        if seen.contains(&token) || special_tokens.eos == Some(token) {
            continue;
        }
        if let Ok(piece) = tokenizer.decode(&[token])
            && (piece.starts_with("[PAD") || (piece.starts_with('<') && piece.ends_with('>')))
            && seen.insert(token)
        {
            suppressed.push(token);
        }
    }
    suppressed
}

fn load_model_runtime(args: &Args) -> Result<Option<Arc<ModelRuntime>>, String> {
    let Some(model_path) = args.model.as_ref() else {
        return Ok(None);
    };
    let loader = GgufModelLoader;
    let mapped = loader
        .load_with_progress(model_path, |progress| {
            tracing::info!(
                stage = progress.stage,
                percent = progress.percent,
                "loading model"
            );
        })
        .map_err(|error| format!("failed to load model: {error:?}"))?;
    optimize_mapped_model_memory(&mapped, args);
    let metadata = &mapped.parsed().metadata;
    let config = inference_config_from_gguf(&mapped, args.ctx_size);
    let tokenizer = load_tokenizer_from_gguf_metadata(metadata)
        .map_err(|error| format!("failed to load tokenizer: {error:?}"))?;
    let chat_template = metadata
        .get("tokenizer.chat_template")
        .or_else(|| metadata.get("tokenizer.ggml.chat_template"))
        .and_then(|value| match value {
            GgufMetadataValue::String(template) => Some(template.clone()),
            _ => None,
        });
    let model = if args.layer_wise {
        LoadedModel::LayerWise(Box::new(
            LayerWiseModel::load_from_gguf(&mapped, config, args.layer_cache)
                .map_err(|error| format!("failed to load layer-wise model: {error}"))?,
        ))
    } else {
        LoadedModel::Inference(Box::new(
            InferenceModel::load_from_gguf(&mapped, config, args.cpu_optimized)
                .map_err(|error| format!("failed to load model weights: {error}"))?,
        ))
    };

    Ok(Some(Arc::new(ModelRuntime {
        id: args.model_id.clone(),
        tokenizer,
        chat_template,
        model: StdMutex::new(model),
        defaults: GenerationDefaults {
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
        },
    })))
}

fn optimize_mapped_model_memory(mapped: &MappedGgufFile, args: &Args) {
    let apply_hints =
        args.cpu_optimized || args.ram_offload || args.mmap_prefetch || args.mmap_hugepages;
    if !apply_hints {
        return;
    }

    if let Err(error) = mapped.advise_random_access() {
        tracing::warn!(%error, "mmap random-access hint failed");
    }
    if (args.cpu_optimized || args.ram_offload || args.mmap_prefetch)
        && let Err(error) = mapped.advise_will_need()
    {
        tracing::warn!(%error, "mmap prefetch hint failed");
    }
    if (args.cpu_optimized || args.mmap_hugepages)
        && let Err(error) = mapped.advise_huge_pages()
    {
        tracing::warn!(%error, "mmap hugepage hint failed");
    }
    if args.ram_offload {
        let started = Instant::now();
        let checksum = mapped.prefault_pages();
        tracing::info!(
            gib = mapped.bytes().len() as f64 / 1024.0 / 1024.0 / 1024.0,
            elapsed_ms = started.elapsed().as_millis(),
            checksum,
            "ram offload prefaulted model pages"
        );
    }
}

fn inference_config_from_gguf(mapped: &MappedGgufFile, ctx_size: Option<usize>) -> InferenceConfig {
    let metadata = &mapped.parsed().metadata;
    let vocab_size = metadata_u32(metadata, "llama.vocab_size")
        .or_else(|| metadata_u32(metadata, "qwen35.vocab_size"))
        .or_else(|| metadata_u32(metadata, "qwen2.vocab_size"))
        .or_else(|| metadata_u32(metadata, "qwen.vocab_size"))
        .or_else(|| metadata_u32(metadata, "general.vocab_size"))
        .or_else(|| metadata_u32(metadata, "tokenizer.ggml.tokens.count"))
        .or_else(|| {
            tensor_dims(mapped, "tok_embeddings.weight")
                .and_then(|dims| dims.get(1).copied())
                .map(|value| value as u32)
        })
        .or_else(|| {
            tensor_dims(mapped, "token_embd.weight")
                .and_then(|dims| dims.get(1).copied())
                .map(|value| value as u32)
        })
        .unwrap_or(32000) as usize;
    let context_size = metadata_u32(metadata, "llama.context_length")
        .or_else(|| metadata_u32(metadata, "qwen35.context_length"))
        .or_else(|| metadata_u32(metadata, "qwen2.context_length"))
        .or_else(|| metadata_u32(metadata, "qwen.context_length"))
        .or_else(|| metadata_u32(metadata, "gemma4.context_length"))
        .or_else(|| metadata_u32(metadata, "gemma.context_length"))
        .or_else(|| metadata_u32(metadata, "llama.embedding_length"))
        .map(|value| value as usize)
        .unwrap_or(4096);
    let layer_count = metadata_u32(metadata, "llama.block_count")
        .or_else(|| metadata_u32(metadata, "qwen35.block_count"))
        .or_else(|| metadata_u32(metadata, "qwen2.block_count"))
        .or_else(|| metadata_u32(metadata, "qwen.block_count"))
        .or_else(|| metadata_u32(metadata, "gemma4.block_count"))
        .or_else(|| metadata_u32(metadata, "gemma.block_count"))
        .unwrap_or(32) as usize;
    let hidden_size = metadata_u32(metadata, "llama.embedding_length")
        .or_else(|| metadata_u32(metadata, "qwen35.embedding_length"))
        .or_else(|| metadata_u32(metadata, "qwen2.embedding_length"))
        .or_else(|| metadata_u32(metadata, "qwen.embedding_length"))
        .or_else(|| metadata_u32(metadata, "gemma4.embedding_length"))
        .or_else(|| metadata_u32(metadata, "gemma.embedding_length"))
        .or_else(|| {
            tensor_dims(mapped, "tok_embeddings.weight")
                .and_then(|dims| dims.first().copied())
                .map(|value| value as u32)
        })
        .or_else(|| {
            tensor_dims(mapped, "token_embd.weight")
                .and_then(|dims| dims.first().copied())
                .map(|value| value as u32)
        })
        .unwrap_or(4096) as usize;
    let intermediate_size = metadata_u32(metadata, "llama.feed_forward_length")
        .or_else(|| metadata_u32(metadata, "qwen35.feed_forward_length"))
        .or_else(|| metadata_u32(metadata, "qwen2.feed_forward_length"))
        .or_else(|| metadata_u32(metadata, "qwen.feed_forward_length"))
        .or_else(|| metadata_u32(metadata, "gemma4.feed_forward_length"))
        .or_else(|| metadata_u32(metadata, "gemma.feed_forward_length"))
        .unwrap_or(11008) as usize;
    let num_attention_heads = metadata_u32(metadata, "llama.attention.head_count")
        .or_else(|| metadata_u32(metadata, "qwen35.attention.head_count"))
        .or_else(|| metadata_u32(metadata, "qwen2.attention.head_count"))
        .or_else(|| metadata_u32(metadata, "qwen.attention.head_count"))
        .or_else(|| metadata_u32(metadata, "gemma4.attention.head_count"))
        .or_else(|| metadata_u32(metadata, "gemma.attention.head_count"))
        .unwrap_or(32) as usize;
    let num_key_value_heads = metadata_u32(metadata, "llama.attention.head_count_kv")
        .or_else(|| metadata_u32(metadata, "qwen35.attention.head_count_kv"))
        .or_else(|| metadata_u32(metadata, "qwen2.attention.head_count_kv"))
        .or_else(|| metadata_u32(metadata, "qwen.attention.head_count_kv"))
        .or_else(|| metadata_u32(metadata, "gemma4.attention.head_count_kv"))
        .or_else(|| metadata_u32(metadata, "gemma.attention.head_count_kv"))
        .unwrap_or(num_attention_heads as u32) as usize;
    let key_value_head_dim = first_layer_tensor_dims(mapped, "attn_k.weight")
        .and_then(|dims| dims.get(1).copied())
        .and_then(|width| width.checked_div(num_key_value_heads as u64))
        .and_then(|value| value.try_into().ok())
        .or_else(|| metadata_u32(metadata, "llama.attention.key_length"))
        .or_else(|| metadata_u32(metadata, "qwen35.attention.key_length"))
        .or_else(|| metadata_u32(metadata, "qwen2.attention.key_length"))
        .or_else(|| metadata_u32(metadata, "qwen.attention.key_length"))
        .unwrap_or((hidden_size / num_attention_heads) as u32)
        as usize;
    let rms_norm_eps = metadata_f32(metadata, "llama.attention.layer_norm_rms_epsilon")
        .or_else(|| metadata_f32(metadata, "qwen35.attention.layer_norm_rms_epsilon"))
        .or_else(|| metadata_f32(metadata, "qwen2.attention.layer_norm_rms_epsilon"))
        .or_else(|| metadata_f32(metadata, "qwen.attention.layer_norm_rms_epsilon"))
        .or_else(|| metadata_f32(metadata, "gemma4.attention.layer_norm_rms_epsilon"))
        .or_else(|| metadata_f32(metadata, "gemma.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    let rope_theta = metadata_f32(metadata, "llama.rope.freq_base")
        .or_else(|| metadata_f32(metadata, "qwen35.rope.freq_base"))
        .or_else(|| metadata_f32(metadata, "qwen2.rope.freq_base"))
        .or_else(|| metadata_f32(metadata, "qwen.rope.freq_base"))
        .or_else(|| metadata_f32(metadata, "gemma4.rope.freq_base"))
        .or_else(|| metadata_f32(metadata, "gemma.rope.freq_base"))
        .unwrap_or(10000.0);

    InferenceConfig {
        vocab_size,
        context_size: ctx_size.unwrap_or(context_size),
        layer_count,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        key_value_head_dim,
        kv_cache_dtype: DType::F32,
        rms_norm_eps,
        rope_theta,
    }
}

fn metadata_u32(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<u32> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Uint8(value)) => Some((*value).into()),
        Some(GgufMetadataValue::Uint16(value)) => Some((*value).into()),
        Some(GgufMetadataValue::Uint32(value)) => Some(*value),
        Some(GgufMetadataValue::Uint64(value)) => (*value).try_into().ok(),
        Some(GgufMetadataValue::Int8(value)) if *value >= 0 => Some((*value as u8).into()),
        Some(GgufMetadataValue::Int16(value)) if *value >= 0 => Some((*value as u16).into()),
        Some(GgufMetadataValue::Int32(value)) if *value >= 0 => (*value).try_into().ok(),
        Some(GgufMetadataValue::Int64(value)) if *value >= 0 => (*value).try_into().ok(),
        _ => None,
    }
}

fn metadata_f32(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<f32> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Float32(value)) => Some(*value),
        Some(GgufMetadataValue::Float64(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Int8(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Int16(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Int32(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Int64(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Uint8(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Uint16(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Uint32(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Uint64(value)) => Some(*value as f32),
        _ => None,
    }
}

fn tensor_dims(mapped: &MappedGgufFile, name: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|tensor| tensor.name == name)
        .map(|tensor| tensor.dimensions.clone())
}

fn first_layer_tensor_dims(mapped: &MappedGgufFile, suffix: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|tensor| tensor.name.starts_with("blk.") && tensor.name.ends_with(suffix))
        .map(|tensor| tensor.dimensions.clone())
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let model = load_model_runtime(&args)
        .unwrap_or_else(|error| panic!("failed to initialize model runtime: {error}"));
    let api_key = std::env::var("OXIDIZE_API_KEY")
        .ok()
        .filter(|value| !value.is_empty());
    let app = build_app_with_config(RequestLimitConfig::default(), api_key, model);
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .expect("failed to bind TCP listener");
    axum::serve(listener, app)
        .await
        .expect("server runtime error");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, header},
    };
    use serde_json::Value;
    use tokio::time::{Duration, sleep};
    use tower::ServiceExt;

    #[test]
    fn args_use_expected_defaults() {
        let args = Args::parse_from(["oxidize-server"]);
        assert_eq!(args.host, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(args.port, 8080);
    }

    #[test]
    fn args_accept_explicit_values() {
        let args = Args::parse_from(["oxidize-server", "--host", "0.0.0.0", "--port", "3000"]);
        assert_eq!(args.host, IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        assert_eq!(args.port, 3000);
    }

    #[test]
    fn app_builds() {
        let _ = build_app();
    }

    #[tokio::test]
    async fn healthz_returns_200() {
        let response = build_app()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn livez_returns_200() {
        let response = build_app()
            .oneshot(
                Request::builder()
                    .uri("/livez")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn readyz_returns_200() {
        let response = build_app()
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn openapi_endpoint_returns_expected_shape() {
        let response = build_app()
            .oneshot(
                Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        assert_eq!(parsed["openapi"], "3.1.0");
        assert_eq!(
            parsed["paths"]["/v1/chat/completions"]["post"]["summary"],
            "Create chat completion"
        );
        assert_eq!(
            parsed["components"]["securitySchemes"]["ApiKeyAuth"]["name"],
            "x-api-key"
        );
    }

    #[tokio::test]
    async fn openapi_endpoint_lists_all_public_routes() {
        let response = build_app()
            .oneshot(
                Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");

        for path in [
            "/healthz",
            "/livez",
            "/readyz",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/models",
            "/v1/embeddings",
        ] {
            assert!(
                parsed["paths"].get(path).is_some(),
                "openapi spec should include {path}"
            );
        }
    }

    #[test]
    fn request_log_message_has_expected_shape() {
        let message = request_log_message("GET", "/healthz");
        assert_eq!(message, "request GET /healthz");
    }

    #[test]
    fn response_log_message_has_expected_shape() {
        let message = response_log_message("GET", "/healthz", StatusCode::OK);
        assert_eq!(message, "response GET /healthz 200");
    }

    #[tokio::test]
    async fn chat_completions_returns_openai_shape() {
        let request_body = json!({
            "model": "oxidize-default",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = build_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        assert_eq!(parsed["object"], "chat.completion");
        assert_eq!(parsed["model"], "oxidize-default");
    }

    #[tokio::test]
    async fn completions_returns_openai_shape() {
        let request_body = json!({
            "model": "oxidize-default",
            "prompt": "hello"
        });
        let response = build_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        assert_eq!(parsed["object"], "text_completion");
        assert_eq!(parsed["model"], "oxidize-default");
    }

    #[tokio::test]
    async fn chat_completions_json_mode_returns_json_content() {
        let request_body = json!({
            "model": "oxidize-default",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": { "type": "json_object" }
        });
        let response = build_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        let content = parsed["choices"][0]["message"]["content"]
            .as_str()
            .expect("content should be a string");
        let parsed_content: Value = serde_json::from_str(content).expect("content should be json");
        assert_eq!(parsed_content, json!({}));
    }

    #[tokio::test]
    async fn completions_json_schema_mode_returns_json_text() {
        let request_body = json!({
            "model": "oxidize-default",
            "prompt": "hello",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "demo",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": { "type": "string" }
                        }
                    }
                }
            }
        });
        let response = build_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        let text = parsed["choices"][0]["text"]
            .as_str()
            .expect("text should be a string");
        let parsed_text: Value = serde_json::from_str(text).expect("text should be json");
        assert_eq!(parsed_text, json!({}));
    }

    #[tokio::test]
    async fn chat_completions_stream_returns_sse_events() {
        let request_body = json!({
            "model": "oxidize-default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        });
        let response = build_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&header::HeaderValue::from_static("text/event-stream"))
        );

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body = String::from_utf8(bytes.to_vec()).expect("sse body should be utf-8");
        assert!(body.contains("\"object\":\"chat.completion.chunk\""));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn chat_completions_stream_json_mode_returns_json_chunk() {
        let request_body = json!({
            "model": "oxidize-default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "response_format": { "type": "json_object" }
        });
        let response = build_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body = String::from_utf8(bytes.to_vec()).expect("sse body should be utf-8");
        assert!(body.contains("\"content\":\"{}\""));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn completions_stream_returns_sse_events() {
        let request_body = json!({
            "model": "oxidize-default",
            "prompt": "hello",
            "stream": true
        });
        let response = build_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&header::HeaderValue::from_static("text/event-stream"))
        );

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body = String::from_utf8(bytes.to_vec()).expect("sse body should be utf-8");
        assert!(body.contains("\"object\":\"text_completion\""));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn models_returns_openai_shape() {
        let response = build_app()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        assert_eq!(parsed["object"], "list");
        assert_eq!(parsed["data"][0]["object"], "model");
    }

    #[tokio::test]
    async fn embeddings_returns_openai_shape() {
        let request_body = json!({
            "model": "oxidize-default",
            "input": "hello"
        });
        let response = build_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        assert_eq!(parsed["object"], "list");
        assert_eq!(parsed["model"], "oxidize-default");
        assert_eq!(parsed["data"][0]["object"], "embedding");
    }

    #[tokio::test]
    async fn api_key_auth_rejects_missing_key_on_v1_routes() {
        let response =
            build_app_with_config(RequestLimitConfig::default(), Some("secret".into()), None)
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .body(Body::empty())
                        .expect("valid request"),
                )
                .await
                .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn api_key_auth_allows_x_api_key_header() {
        let response =
            build_app_with_config(RequestLimitConfig::default(), Some("secret".into()), None)
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .header("x-api-key", "secret")
                        .body(Body::empty())
                        .expect("valid request"),
                )
                .await
                .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_key_auth_allows_bearer_token() {
        let response =
            build_app_with_config(RequestLimitConfig::default(), Some("secret".into()), None)
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .header(header::AUTHORIZATION, "Bearer secret")
                        .body(Body::empty())
                        .expect("valid request"),
                )
                .await
                .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_key_auth_does_not_gate_health_endpoints() {
        let response =
            build_app_with_config(RequestLimitConfig::default(), Some("secret".into()), None)
                .oneshot(
                    Request::builder()
                        .uri("/healthz")
                        .body(Body::empty())
                        .expect("valid request"),
                )
                .await
                .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rate_limit_returns_429_when_window_is_exceeded() {
        let app = build_app_with_limits(RequestLimitConfig {
            requests_per_second: 1,
            max_in_flight: 4,
            max_queue: 4,
        });
        let first = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        let second = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");

        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn limiter_queues_one_request_and_rejects_when_queue_is_full() {
        let limiter = Arc::new(RequestLimiter::new(RequestLimitConfig {
            requests_per_second: 100,
            max_in_flight: 1,
            max_queue: 1,
        }));

        let first = limiter
            .try_acquire()
            .await
            .expect("first request should acquire active slot");

        let queued_limiter = Arc::clone(&limiter);
        let queued_task = tokio::spawn(async move { queued_limiter.try_acquire().await });
        sleep(Duration::from_millis(20)).await;

        let rejected = limiter.try_acquire().await;
        assert!(matches!(rejected, Err(RequestLimitError::QueueFull)));

        drop(first);
        let queued = queued_task
            .await
            .expect("queued task should complete")
            .expect("queued request should eventually acquire");
        drop(queued);
    }

    #[tokio::test]
    async fn limiter_allows_concurrent_in_flight_requests_up_to_limit() {
        let limiter = Arc::new(RequestLimiter::new(RequestLimitConfig {
            requests_per_second: 100,
            max_in_flight: 2,
            max_queue: 0,
        }));

        let first = limiter
            .try_acquire()
            .await
            .expect("first request should acquire active slot");
        let second = limiter
            .try_acquire()
            .await
            .expect("second request should acquire active slot");

        let third = limiter.try_acquire().await;
        assert!(matches!(third, Err(RequestLimitError::QueueFull)));

        drop(first);
        drop(second);
    }

    #[tokio::test]
    async fn queue_full_does_not_consume_rate_limit_capacity() {
        let limiter = Arc::new(RequestLimiter::new(RequestLimitConfig {
            requests_per_second: 2,
            max_in_flight: 1,
            max_queue: 0,
        }));

        let held = limiter
            .try_acquire()
            .await
            .expect("first request should acquire active slot");

        let rejected_while_full = limiter.try_acquire().await;
        assert!(matches!(
            rejected_while_full,
            Err(RequestLimitError::QueueFull)
        ));
        drop(held);

        let second = limiter
            .try_acquire()
            .await
            .expect("queue-full rejection should not consume rate budget");
        drop(second);

        let third = limiter.try_acquire().await;
        assert!(matches!(third, Err(RequestLimitError::RateLimited)));
    }

    #[tokio::test]
    async fn continuous_batcher_waits_for_batch_window() {
        let batcher = Arc::new(ContinuousBatcher::new(ContinuousBatchConfig {
            max_batch_size: 4,
            max_wait: Duration::from_millis(40),
        }));
        let started = TokioInstant::now();

        let first = {
            let batcher = Arc::clone(&batcher);
            tokio::spawn(async move {
                batcher.wait_turn().await;
                TokioInstant::now()
            })
        };

        sleep(Duration::from_millis(5)).await;

        let second = {
            let batcher = Arc::clone(&batcher);
            tokio::spawn(async move {
                batcher.wait_turn().await;
                TokioInstant::now()
            })
        };

        let first_done = first.await.expect("first task should complete");
        let second_done = second.await.expect("second task should complete");
        let first_elapsed = first_done.duration_since(started);
        let second_elapsed = second_done.duration_since(started);
        assert!(first_elapsed >= Duration::from_millis(30));
        assert!(second_elapsed >= Duration::from_millis(30));
    }

    #[tokio::test]
    async fn continuous_batcher_releases_early_when_batch_is_full() {
        let batcher = Arc::new(ContinuousBatcher::new(ContinuousBatchConfig {
            max_batch_size: 2,
            max_wait: Duration::from_millis(200),
        }));
        let started = TokioInstant::now();

        let first = {
            let batcher = Arc::clone(&batcher);
            tokio::spawn(async move {
                batcher.wait_turn().await;
                TokioInstant::now()
            })
        };

        sleep(Duration::from_millis(20)).await;
        batcher.wait_turn().await;

        let first_done = first.await.expect("first task should complete");
        let elapsed = first_done.duration_since(started);
        assert!(elapsed < Duration::from_millis(150));
    }
}
