use std::{
    collections::BTreeMap,
    collections::VecDeque,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use clap::{Parser, ValueEnum};
use futures_util::{Stream, StreamExt, stream};
use oxidize_core::{
    generation::{GenerationConfig, GenerationStream},
    gguf::{GgufMetadataValue, MappedGgufFile},
    inference::{InferenceConfig, InferenceModel},
    layer_wise::LayerWiseModel,
    model::{Model, ModelError, Session, Token},
    model_loader::{GgufModelLoader, ModelLoader},
    paged_attention::{
        BlockPool, BlockPoolConfig, Scheduler, SchedulerConfig, Sequence,
    },
    sampling::{SamplingConfig, sample},
    tensor::DType,
    tokenizer::{
        ChatMessage, EncodeOptions, LoadedTokenizer, load_tokenizer_from_gguf_metadata,
        process_chat_template,
    },
};

mod mesh_cluster;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Mutex as StdMutex;
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::{Instant as TokioInstant, sleep_until};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    Cpu,
    Metal,
    /// macOS only
    Mlx,
    Cuda,
}

impl Backend {
    fn to_core_backend(self) -> oxidize_core::backend::Backend {
        match self {
            Backend::Cpu => oxidize_core::backend::Backend::Cpu,
            Backend::Metal => oxidize_core::backend::Backend::Metal,
            Backend::Mlx => oxidize_core::backend::Backend::Mlx,
            Backend::Cuda => oxidize_core::backend::Backend::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BatchMode {
    Sequential,
    Paged,
}

#[derive(Debug, Parser)]
#[command(name = "oxidize-server")]
struct Args {
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,
    #[arg(long, value_enum, default_value_t = BatchMode::Sequential)]
    batch_mode: BatchMode,
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
    #[arg(long, default_value_t = 512)]
    prefill_batch_size: usize,
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
    /// Enable mesh cluster mode: this node becomes the master that routes
    /// OpenAI-compatible requests to worker shards over the mesh data plane.
    #[arg(long, default_value_t = false)]
    mesh: bool,
    /// Port for the mesh libp2p listener (0 = ephemeral).
    #[arg(long, default_value_t = 0)]
    mesh_port: u16,
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
    paged: Option<Arc<PagedModelRuntime>>,
    mesh: Option<mesh_cluster::MeshClusterState>,
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
    prefill_batch_size: usize,
}

/// Runtime state for PagedAttention-based generation.
///
/// Holds a [`Scheduler`] alongside the loaded model so that each request
/// becomes a [`Sequence`] with tracked block allocation.  The scheduler
/// enforces token budgets, handles block reclamation, and provides accurate
/// usage counts.
struct PagedModelRuntime {
    runtime: Arc<ModelRuntime>,
    scheduler: StdMutex<Scheduler>,
    next_seq_id: AtomicU64,
    block_size: usize,
}

enum LoadedModel {
    Inference(Box<InferenceModel>),
    LayerWise(Box<LayerWiseModel>),
    #[cfg(target_os = "macos")]
    Mlx(Box<oxidize_core::mlx_inference::MlxInferenceModel>),
    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)]
    Mlx(Box<InferenceModel>),
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
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.forward(tokens, session),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.forward(tokens, session),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Inference(model) => model.vocab_size(),
            Self::LayerWise(model) => model.vocab_size(),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.vocab_size(),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.vocab_size(),
        }
    }

    fn context_size(&self) -> usize {
        match self {
            Self::Inference(model) => model.context_size(),
            Self::LayerWise(model) => model.context_size(),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.context_size(),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.context_size(),
        }
    }

    fn layer_count(&self) -> usize {
        match self {
            Self::Inference(model) => model.layer_count(),
            Self::LayerWise(model) => model.layer_count(),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.layer_count(),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.layer_count(),
        }
    }

    fn rewind_to(&mut self, consumed_tokens: usize) -> Result<(), ModelError> {
        match self {
            Self::Inference(model) => model.rewind_to(consumed_tokens),
            Self::LayerWise(model) => model.rewind_to(consumed_tokens),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.rewind_to(consumed_tokens),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.rewind_to(consumed_tokens),
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

#[allow(dead_code)]
fn build_app_with_config(
    config: RequestLimitConfig,
    api_key: Option<String>,
    model: Option<Arc<ModelRuntime>>,
) -> Router {
    build_app_with_full_config(config, api_key, model, None)
}

#[allow(dead_code)]
fn build_app_with_full_config(
    config: RequestLimitConfig,
    api_key: Option<String>,
    model: Option<Arc<ModelRuntime>>,
    mesh: Option<mesh_cluster::MeshClusterState>,
) -> Router {
    let state = AppState {
        limiter: Arc::new(RequestLimiter::new(config)),
        batcher: Arc::new(ContinuousBatcher::default()),
        auth: AuthConfig {
            api_key: api_key.map(Arc::<str>::from),
        },
        model: model.clone(),
        paged: None,
        mesh,
    };
    build_app_with_state(state)
}

const MAX_BODY_SIZE_BYTES: usize = 10 * 1024 * 1024;

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
        .route("/v1/mesh/chat/completions", post(mesh_chat_completions_handler))
        .layer(DefaultBodyLimit::max(MAX_BODY_SIZE_BYTES))
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

async fn mesh_chat_completions_handler(
    State(state): State<AppState>,
    Json(payload): Json<mesh_cluster::MeshChatRequest>,
) -> Response {
    let Some(mesh) = state.mesh.clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "mesh not enabled", "type": "mesh_disabled"}})),
        )
            .into_response();
    };
    mesh_cluster::mesh_chat_completions(axum::extract::State(mesh), Json(payload)).await
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
    fn constant_time_eq(a: &str, b: &str) -> bool {
        use subtle::ConstantTimeEq;
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();
        a_bytes.ct_eq(b_bytes).into()
    }

    headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| constant_time_eq(value, expected_key))
        || headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .is_some_and(|token| constant_time_eq(token, expected_key))
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
    guided_json: Option<Value>,
    #[serde(default)]
    json_schema: Option<Value>,
    #[serde(default)]
    guided_regex: Option<String>,
    #[serde(default)]
    guided_choice: Option<Vec<String>>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    typical_p: Option<f32>,
    #[serde(default)]
    tail_free_z: Option<f32>,
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    best_of: Option<usize>,
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
    guided_json: Option<Value>,
    #[serde(default)]
    json_schema: Option<Value>,
    #[serde(default)]
    guided_regex: Option<String>,
    #[serde(default)]
    guided_choice: Option<Vec<String>>,
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
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    typical_p: Option<f32>,
    #[serde(default)]
    tail_free_z: Option<f32>,
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    echo: bool,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    best_of: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
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
    min_p: Option<f32>,
    typical_p: Option<f32>,
    tail_free_z: Option<f32>,
    stop: Vec<String>,
    seed: Option<u64>,
    echo: bool,
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
    if let Some(response) = validate_candidate_count(payload.n, payload.best_of) {
        return response;
    }

    let model_id = payload.model.clone();
    let stream = payload.stream;

    // --- PagedAttention path ------------------------------------------------
    if let Some(paged) = state.paged.clone() {
        if payload.model != paged.runtime.id {
            return model_not_found(&payload.model);
        }
        let prompt = render_chat_prompt(&paged.runtime, &payload.messages);
        let req = GenerationRequest {
            prompt,
            max_tokens: payload.max_tokens.or(payload.max_completion_tokens),
            temperature: payload.temperature,
            top_p: payload.top_p,
            top_k: payload.top_k,
            min_p: payload.min_p,
            typical_p: payload.typical_p,
            tail_free_z: payload.tail_free_z,
            stop: payload.stop.map(StopSequences::into_vec).unwrap_or_default(),
            seed: payload.seed,
            echo: false,
        };

        if stream {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, GenerationError>>(128);
            let cancel = Arc::new(AtomicBool::new(false));
            let cancel_for_task = Arc::clone(&cancel);
            let paged_for_task = Arc::clone(&paged);
            tokio::task::spawn_blocking(move || {
                generate_with_scheduler_streaming_blocking(paged_for_task, req, tx, cancel_for_task);
            });
            return chat_completion_stream_response_paged(model_id, rx, cancel);
        }

        let generated = tokio::task::spawn_blocking(move || {
            generate_with_scheduler_blocking(&paged, req)
        })
        .await
        .map_err(|e| GenerationError::Other(format!("generation task failed: {e}")));

        return match generated {
            Ok(Ok(result)) => chat_completion_response(model_id, result),
            Ok(Err(err)) => generation_error_response(err),
            Err(err) => generation_error_response(err),
        };
    }

    // --- Sequential fallback path -------------------------------------------
    if let Some(runtime) = state.model.clone() {
        if payload.model != runtime.id {
            return model_not_found(&payload.model);
        }
        let prompt = render_chat_prompt(&runtime, &payload.messages);
        let generated = generate_text(
            runtime,
            GenerationRequest {
                prompt,
                max_tokens: payload.max_tokens.or(payload.max_completion_tokens),
                temperature: payload.temperature,
                top_p: payload.top_p,
                top_k: payload.top_k,
                min_p: payload.min_p,
                typical_p: payload.typical_p,
                tail_free_z: payload.tail_free_z,
                stop: payload
                    .stop
                    .map(StopSequences::into_vec)
                    .unwrap_or_default(),
                seed: payload.seed,
                echo: false,
            },
        )
        .await;
        return match generated {
            Ok(result) if stream => chat_completion_stream_response(model_id, result.text),
            Ok(result) => chat_completion_response(model_id, result),
            Err(error) => generation_error_response(error),
        };
    }

    // --- No-model placeholder fallback --------------------------------------
    let response_content = payload
        .response_format
        .as_ref()
        .map_or("", ResponseFormat::output_text);
    let response_content = if let Some(choices) = payload.guided_choice.as_ref()
        && let Some(choice) = choices.first()
    {
        choice.as_str()
    } else if payload.guided_json.is_some() || payload.json_schema.is_some() {
        "{}"
    } else if let Some(regex) = payload.guided_regex.as_ref() {
        regex.as_str()
    } else {
        response_content
    };

    if stream {
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
    if let Some(response) = validate_candidate_count(payload.n, payload.best_of) {
        return response;
    }

    let model_id = payload.model.clone();
    let stream = payload.stream;

    // --- PagedAttention path ------------------------------------------------
    if let Some(paged) = state.paged.clone() {
        if payload.model != paged.runtime.id {
            return model_not_found(&payload.model);
        }
        let req = GenerationRequest {
            prompt: payload.prompt,
            max_tokens: payload.max_tokens,
            temperature: payload.temperature,
            top_p: payload.top_p,
            top_k: payload.top_k,
            min_p: payload.min_p,
            typical_p: payload.typical_p,
            tail_free_z: payload.tail_free_z,
            stop: payload.stop.map(StopSequences::into_vec).unwrap_or_default(),
            seed: payload.seed,
            echo: payload.echo,
        };

        if stream {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, GenerationError>>(128);
            let cancel = Arc::new(AtomicBool::new(false));
            let cancel_for_task = Arc::clone(&cancel);
            let paged_for_task = Arc::clone(&paged);
            tokio::task::spawn_blocking(move || {
                generate_with_scheduler_streaming_blocking(paged_for_task, req, tx, cancel_for_task);
            });
            return chat_completion_stream_response_paged(model_id, rx, cancel);
        }

        let generated = tokio::task::spawn_blocking(move || {
            generate_with_scheduler_blocking(&paged, req)
        })
        .await
        .map_err(|e| GenerationError::Other(format!("generation task failed: {e}")));

        return match generated {
            Ok(Ok(result)) => completion_response(model_id, result),
            Ok(Err(err)) => generation_error_response(err),
            Err(err) => generation_error_response(err),
        };
    }

    // --- Sequential fallback path -------------------------------------------
    if let Some(runtime) = state.model.clone() {
        if payload.model != runtime.id {
            return model_not_found(&payload.model);
        }
        let generated = generate_text(
            runtime,
            GenerationRequest {
                prompt: payload.prompt,
                max_tokens: payload.max_tokens,
                temperature: payload.temperature,
                top_p: payload.top_p,
                top_k: payload.top_k,
                min_p: payload.min_p,
                typical_p: payload.typical_p,
                tail_free_z: payload.tail_free_z,
                stop: payload
                    .stop
                    .map(StopSequences::into_vec)
                    .unwrap_or_default(),
                seed: payload.seed,
                echo: payload.echo,
            },
        )
        .await;
        return match generated {
            Ok(result) if stream => completion_stream_response(model_id, result.text),
            Ok(result) => completion_response(model_id, result),
            Err(error) => generation_error_response(error),
        };
    }

    // --- No-model placeholder fallback --------------------------------------
    let response_text = payload
        .response_format
        .as_ref()
        .map_or("", ResponseFormat::output_text);
    let response_text = if let Some(choices) = payload.guided_choice.as_ref()
        && let Some(choice) = choices.first()
    {
        choice.as_str()
    } else if payload.guided_json.is_some() || payload.json_schema.is_some() {
        "{}"
    } else if let Some(regex) = payload.guided_regex.as_ref() {
        regex.as_str()
    } else {
        response_text
    };

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
) -> Result<GenerationResult, GenerationError> {
    tokio::task::spawn_blocking(move || generate_text_blocking(&runtime, request))
        .await
        .map_err(|error| GenerationError::Other(format!("generation task failed: {error}")))?
}

fn generate_text_blocking(
    runtime: &ModelRuntime,
    request: GenerationRequest,
) -> Result<GenerationResult, GenerationError> {
    let mut model = runtime
        .model
        .lock()
        .map_err(|_| GenerationError::Other("model lock poisoned".to_owned()))?;
    model
        .rewind_to(0)
        .map_err(|e| GenerationError::Other(format!("failed to reset model KV cache: {e:?}")))?;
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
    let stop_sequences = request
        .stop
        .iter()
        .map(|stop| {
            runtime.tokenizer.encode_with_special_tokens(
                stop,
                EncodeOptions {
                    add_bos: false,
                    add_eos: false,
                    pad_to: None,
                },
            )
        })
        .filter(|tokens| !tokens.is_empty())
        .collect();
    let config = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: runtime.tokenizer.special_tokens().eos,
        stop_sequences,
        prefill_batch_size: runtime.defaults.prefill_batch_size,
        suppressed_tokens: suppressed_generation_tokens(&runtime.tokenizer, model.vocab_size()),
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            min_p: request.min_p,
            typical_p: request.typical_p,
            tail_free_z: request.tail_free_z,
            ..SamplingConfig::default()
        },
    };
    let mut seeded_rng = request.seed.map(StdRng::seed_from_u64);
    let mut thread_rng = rand::thread_rng();
    let mut stream =
        GenerationStream::new(&mut *model, &mut session, &prompt_tokens, config, || {
            seeded_rng.as_mut().map_or_else(
                || rand::Rng::r#gen::<f32>(&mut thread_rng),
                rand::Rng::r#gen::<f32>,
            )
        });
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);
    let mut generated_tokens = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => generated_tokens.push(token),
            Poll::Ready(Some(Err(error))) => {
                return Err(GenerationError::Other(format!("generation error: {error:?}")))
            }
            Poll::Ready(None) | Poll::Pending => break,
        }
    }

    let text = runtime
        .tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    let text = trim_stop_text(&text, &request.stop);
    let text = if request.echo {
        format!("{}{}", request.prompt, text)
    } else {
        text
    };
    Ok(GenerationResult {
        text,
        prompt_tokens: prompt_tokens.len(),
        completion_tokens: generated_tokens.len(),
    })
}

/// Run generation through the PagedAttention scheduler.
///
/// Creates a [`Sequence`], allocates blocks via the scheduler, runs prefill +
/// decode one token at a time, and returns accurate usage counts derived from
/// the sequence state.  On any error the sequence is removed so blocks are
/// reclaimed immediately.
fn generate_with_scheduler_blocking(
    paged: &PagedModelRuntime,
    request: GenerationRequest,
) -> Result<GenerationResult, GenerationError> {
    let mut model = paged
        .runtime
        .model
        .lock()
        .map_err(|_| GenerationError::Other("model lock poisoned".to_owned()))?;
    model
        .rewind_to(0)
        .map_err(|e| GenerationError::Other(format!("failed to reset model KV cache: {e:?}")))?;

    let mut session = Session::new();
    let prompt_tokens = paged
        .runtime
        .tokenizer
        .encode_with_special_tokens(
            &request.prompt,
            EncodeOptions {
                add_bos: true,
                add_eos: false,
                pad_to: None,
            },
        );

    let max_tokens = request.max_tokens.unwrap_or(paged.runtime.defaults.max_tokens);
    let temperature = request.temperature.unwrap_or(paged.runtime.defaults.temperature);
    let top_p = request.top_p.or(paged.runtime.defaults.top_p);
    let top_k = request.top_k.or(paged.runtime.defaults.top_k);
    let stop_token = paged.runtime.tokenizer.special_tokens().eos;
    let sampling = SamplingConfig {
        temperature,
        top_p,
        top_k,
        min_p: request.min_p,
        typical_p: request.typical_p,
        tail_free_z: request.tail_free_z,
        ..SamplingConfig::default()
    };

    let seq_id = paged.next_seq_id.fetch_add(1, Ordering::SeqCst);
    let mut scheduler = paged
        .scheduler
        .lock()
        .map_err(|_| GenerationError::Other("scheduler lock poisoned".to_owned()))?;

    let seq = Sequence::new(
        seq_id,
        prompt_tokens.clone(),
        paged.block_size,
        max_tokens,
        stop_token,
        sampling,
    );
    scheduler.add_sequence(seq).map_err(|e| {
        GenerationError::Other(format!("scheduler add_sequence failed: {e}"))
    })?;

    let mut generated_tokens: Vec<Token> = Vec::new();

    // Prefill step: scheduler allocates blocks for prompt tokens.
    let step_result = scheduler.step().map_err(|e| {
        GenerationError::Other(format!("scheduler step failed: {e}"))
    })?;

    if !step_result.scheduled_seq_ids.contains(&seq_id) {
        // Could not schedule (e.g. OOM) — clean up and return error.
        let _ = scheduler.remove_sequence(seq_id);
        return Err(GenerationError::KvCacheExhausted);
    }

    // Run prefill through the model (single-sequence, no true batched forward yet).
    let prefill_logits = model
        .forward(&prompt_tokens, &mut session)
        .map_err(|e| GenerationError::Other(format!("model forward failed: {e:?}")))?;

    // Sample the first token.
    let mut rng = rand::thread_rng();
    let first_token = sample(
        &prefill_logits,
        sampling,
        rand::Rng::r#gen::<f32>(&mut rng),
    )
    .map_err(|e| GenerationError::Other(format!("sampling failed: {e:?}")))?;

    let mut sampled = std::collections::HashMap::new();
    sampled.insert(seq_id, first_token);
    scheduler.postprocess_step(&sampled).map_err(|e| {
        GenerationError::Other(format!("scheduler postprocess_step failed: {e}"))
    })?;
    generated_tokens.push(first_token);

    // Decode loop.
    loop {
        let seq = scheduler.get_sequence(seq_id);
        if seq.is_none() || seq.unwrap().is_finished() {
            break;
        }

        let step_result = scheduler.step().map_err(|e| {
            GenerationError::Other(format!("scheduler step failed: {e}"))
        })?;

        if !step_result.scheduled_seq_ids.contains(&seq_id) {
            break;
        }

        let decode_logits = model
            .forward(
                &[*generated_tokens.last().unwrap_or(&first_token)],
                &mut session,
            )
            .map_err(|e| {
                GenerationError::Other(format!("model forward failed: {e:?}"))
            })?;

        let token = sample(
            &decode_logits,
            sampling,
            rand::Rng::r#gen::<f32>(&mut rng),
        )
        .map_err(|e| GenerationError::Other(format!("sampling failed: {e:?}")))?;

        let mut sampled = std::collections::HashMap::new();
        sampled.insert(seq_id, token);
        scheduler.postprocess_step(&sampled).map_err(|e| {
            GenerationError::Other(format!("scheduler postprocess_step failed: {e}"))
        })?;
        generated_tokens.push(token);
    }

    let seq = scheduler.get_sequence(seq_id);
    let prompt_tokens_count = seq.map(|s| s.prompt_len()).unwrap_or(prompt_tokens.len());
    let completion_tokens_count =
        seq.map(|s| s.generated_len()).unwrap_or(generated_tokens.len());

    let text = paged
        .runtime
        .tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    let text = trim_stop_text(&text, &request.stop);
    let text = if request.echo {
        format!("{}{}", request.prompt, text)
    } else {
        text
    };

    let result = Ok(GenerationResult {
        text,
        prompt_tokens: prompt_tokens_count,
        completion_tokens: completion_tokens_count,
    });

    // Remove sequence to free blocks immediately.
    let _ = scheduler.remove_sequence(seq_id);
    result
}

/// Streaming generation via PagedAttention scheduler.
///
/// Yields each generated token as it is produced.  The caller can wrap this
/// in an SSE stream.  If the generation task detects `cancel` has been set
/// (e.g. because the client disconnected), it stops immediately and frees
/// the sequence's blocks.
fn generate_with_scheduler_streaming_blocking(
    paged: Arc<PagedModelRuntime>,
    request: GenerationRequest,
    tx: tokio::sync::mpsc::Sender<Result<String, GenerationError>>,
    cancel: Arc<AtomicBool>,
) {
    let result = generate_with_scheduler_streaming_inner(&paged, request, &tx, cancel);
    // Ensure the channel is closed after generation finishes or errors.
    let _ = tx.blocking_send(result.map(|_| String::new()));
}

fn generate_with_scheduler_streaming_inner(
    paged: &PagedModelRuntime,
    request: GenerationRequest,
    tx: &tokio::sync::mpsc::Sender<Result<String, GenerationError>>,
    cancel: Arc<AtomicBool>,
) -> Result<(), GenerationError> {
    let mut model = paged
        .runtime
        .model
        .lock()
        .map_err(|_| GenerationError::Other("model lock poisoned".to_owned()))?;
    model
        .rewind_to(0)
        .map_err(|e| GenerationError::Other(format!("failed to reset model KV cache: {e:?}")))?;

    let mut session = Session::new();
    let prompt_tokens = paged
        .runtime
        .tokenizer
        .encode_with_special_tokens(
            &request.prompt,
            EncodeOptions {
                add_bos: true,
                add_eos: false,
                pad_to: None,
            },
        );

    let max_tokens = request.max_tokens.unwrap_or(paged.runtime.defaults.max_tokens);
    let temperature = request.temperature.unwrap_or(paged.runtime.defaults.temperature);
    let top_p = request.top_p.or(paged.runtime.defaults.top_p);
    let top_k = request.top_k.or(paged.runtime.defaults.top_k);
    let stop_token = paged.runtime.tokenizer.special_tokens().eos;
    let sampling = SamplingConfig {
        temperature,
        top_p,
        top_k,
        min_p: request.min_p,
        typical_p: request.typical_p,
        tail_free_z: request.tail_free_z,
        ..SamplingConfig::default()
    };

    let seq_id = paged.next_seq_id.fetch_add(1, Ordering::SeqCst);
    let mut scheduler = paged
        .scheduler
        .lock()
        .map_err(|_| GenerationError::Other("scheduler lock poisoned".to_owned()))?;

    let seq = Sequence::new(
        seq_id,
        prompt_tokens.clone(),
        paged.block_size,
        max_tokens,
        stop_token,
        sampling,
    );
    if let Err(e) = scheduler.add_sequence(seq) {
        return Err(GenerationError::Other(format!(
            "scheduler add_sequence failed: {e}"
        )));
    }

    // Helper to clean up on error or cancel.
    let cleanup = |sched: &mut Scheduler| {
        let _ = sched.remove_sequence(seq_id);
    };

    let step_result = match scheduler.step() {
        Ok(r) => r,
        Err(e) => {
            cleanup(&mut scheduler);
            return Err(GenerationError::Other(format!("scheduler step failed: {e}")));
        }
    };

    if !step_result.scheduled_seq_ids.contains(&seq_id) {
        cleanup(&mut scheduler);
        return Err(GenerationError::KvCacheExhausted);
    }

    let prefill_logits = match model.forward(&prompt_tokens, &mut session) {
        Ok(l) => l,
        Err(e) => {
            cleanup(&mut scheduler);
            return Err(GenerationError::Other(format!("model forward failed: {e:?}")));
        }
    };

    let mut rng = rand::thread_rng();
    let first_token = match sample(
        &prefill_logits,
        sampling,
        rand::Rng::r#gen::<f32>(&mut rng),
    ) {
        Ok(t) => t,
        Err(e) => {
            cleanup(&mut scheduler);
            return Err(GenerationError::Other(format!("sampling failed: {e:?}")));
        }
    };

    let mut sampled = std::collections::HashMap::new();
    sampled.insert(seq_id, first_token);
    if let Err(e) = scheduler.postprocess_step(&sampled) {
        cleanup(&mut scheduler);
        return Err(GenerationError::Other(format!(
            "scheduler postprocess_step failed: {e}"
        )));
    }

    let piece = paged.runtime.tokenizer.decode(&[first_token]).unwrap_or_default();
    if tx.blocking_send(Ok(piece)).is_err() {
        // Receiver dropped (client disconnected) — clean up and exit.
        cleanup(&mut scheduler);
        return Ok(());
    }

    // Decode loop with cancel checking.
    loop {
        if cancel.load(Ordering::Relaxed) {
            cleanup(&mut scheduler);
            return Ok(());
        }

        let seq = scheduler.get_sequence(seq_id);
        if seq.is_none() || seq.unwrap().is_finished() {
            break;
        }

        let step_result = match scheduler.step() {
            Ok(r) => r,
            Err(e) => {
                cleanup(&mut scheduler);
                return Err(GenerationError::Other(format!("scheduler step failed: {e}")));
            }
        };

        if !step_result.scheduled_seq_ids.contains(&seq_id) {
            break;
        }

        let last_token = *scheduler
            .get_sequence(seq_id)
            .and_then(|s| s.generated_tokens().last())
            .unwrap_or(&first_token);

        let decode_logits = match model.forward(&[last_token], &mut session) {
            Ok(l) => l,
            Err(e) => {
                cleanup(&mut scheduler);
                return Err(GenerationError::Other(format!(
                    "model forward failed: {e:?}"
                )));
            }
        };

        let token = match sample(
            &decode_logits,
            sampling,
            rand::Rng::r#gen::<f32>(&mut rng),
        ) {
            Ok(t) => t,
            Err(e) => {
                cleanup(&mut scheduler);
                return Err(GenerationError::Other(format!(
                    "sampling failed: {e:?}"
                )));
            }
        };

        let mut sampled = std::collections::HashMap::new();
        sampled.insert(seq_id, token);
        if let Err(e) = scheduler.postprocess_step(&sampled) {
            cleanup(&mut scheduler);
            return Err(GenerationError::Other(format!(
                "scheduler postprocess_step failed: {e}"
            )));
        }

        let piece = paged.runtime.tokenizer.decode(&[token]).unwrap_or_default();
        if tx.blocking_send(Ok(piece)).is_err() {
            // Client disconnected.
            cleanup(&mut scheduler);
            return Ok(());
        }
    }

    cleanup(&mut scheduler);
    Ok(())
}

fn trim_stop_text(text: &str, stop: &[String]) -> String {
    let Some(idx) = stop
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
    else {
        return text.to_owned();
    };
    text[..idx].to_owned()
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

/// Token-by-token SSE stream driven by a tokio mpsc channel.
///
/// The channel yields individual token strings.  Each token is emitted as an
/// OpenAI-compatible `chat.completion.chunk` event.  When the channel closes,
/// a `finish_reason: stop` chunk and `[DONE]` are emitted.
fn chat_completion_stream_response_paged(
    model: String,
    rx: tokio::sync::mpsc::Receiver<Result<String, GenerationError>>,
    cancel: Arc<AtomicBool>,
) -> Response {
    let stream = stream::unfold(
        (model.clone(), false, rx, cancel),
        move |(model, done, mut rx, cancel)| async move {
            if done {
                return None;
            }
            match rx.recv().await {
                Some(Ok(piece)) => {
                    let event = Event::default().data(
                        json!({
                            "id": "chatcmpl-oxidize",
                            "object": "chat.completion.chunk",
                            "created": unix_timestamp(),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": { "content": piece },
                                    "finish_reason": null
                                }
                            ]
                        })
                        .to_string(),
                    );
                    Some((Ok::<Event, std::convert::Infallible>(event), (model, false, rx, cancel)))
                }
                Some(Err(_error)) => {
                    // Error during generation — emit finish_reason=stop and close.
                    let event = Event::default().data(
                        json!({
                            "id": "chatcmpl-oxidize",
                            "object": "chat.completion.chunk",
                            "created": unix_timestamp(),
                            "model": model,
                            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
                        })
                        .to_string(),
                    );
                    Some((Ok(event), (model, true, rx, cancel)))
                }
                None => {
                    // Channel closed — emit finish_reason=stop and [DONE].
                    let event = Event::default().data(
                        json!({
                            "id": "chatcmpl-oxidize",
                            "object": "chat.completion.chunk",
                            "created": unix_timestamp(),
                            "model": model,
                            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
                        })
                        .to_string(),
                    );
                    Some((Ok(event), (model, true, rx, cancel)))
                }
            }
        },
    );

    let done_stream = stream::iter(vec![Ok(Event::default().data("[DONE]"))]);
    Sse::new(stream.chain(done_stream))
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

/// Structured generation error so the HTTP layer can distinguish KV-cache
/// exhaustion from other failures and return the correct status code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationError {
    KvCacheExhausted,
    Other(String),
}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerationError::KvCacheExhausted => write!(f, "KV cache exhausted"),
            GenerationError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

fn generation_error_response(error: GenerationError) -> Response {
    match error {
        GenerationError::KvCacheExhausted => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    "message": "KV cache exhausted",
                    "type": "insufficient_memory"
                }
            })),
        )
            .into_response(),
        GenerationError::Other(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": msg, "type": "generation_error"}})),
        )
            .into_response(),
    }
}

fn validate_candidate_count(n: Option<usize>, best_of: Option<usize>) -> Option<Response> {
    let n = n.unwrap_or(1);
    let best_of = best_of.unwrap_or(n);
    if n == 1 && best_of == 1 {
        return None;
    }
    Some(
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "oxidize-server currently supports only n=1 and best_of=1",
                    "type": "unsupported_parameter"
                }
            })),
        )
            .into_response(),
    )
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
            id: state
                .paged
                .as_ref()
                .map(|p| p.runtime.id.clone())
                .or_else(|| state.model.as_ref().map(|r| r.id.clone()))
                .unwrap_or_else(|| "oxidize-default".to_owned()),
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
    let (effective_backend, warning) = args.backend.to_core_backend().effective();
    if let Some(msg) = warning {
        tracing::warn!("{msg}");
    }
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
    if args.ctx_size == Some(0) {
        return Err("invalid --ctx-size: must be greater than 0".into());
    }
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
    } else if effective_backend == oxidize_core::backend::Backend::Mlx {
        #[cfg(target_os = "macos")]
        {
            match oxidize_core::mlx_inference::MlxInferenceModel::load_from_gguf(&mapped, config) {
                Ok(m) => {
                    tracing::info!("MLX backend: loaded model into unified memory");
                    LoadedModel::Mlx(Box::new(m))
                }
                Err(error) => {
                    tracing::warn!("MLX initialization failed: {error}; falling back to CPU");
                    LoadedModel::Inference(Box::new(
                        InferenceModel::load_from_gguf(&mapped, config, args.cpu_optimized)
                            .map_err(|error| format!("failed to load model weights: {error}"))?,
                    ))
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            tracing::warn!("MLX backend requested but unavailable on Linux; falling back to CPU");
            LoadedModel::Inference(Box::new(
                InferenceModel::load_from_gguf(&mapped, config, args.cpu_optimized)
                    .map_err(|error| format!("failed to load model weights: {error}"))?,
            ))
        }
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
            prefill_batch_size: args.prefill_batch_size,
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
        ..Default::default()
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
    let (effective_backend, warning) = args.backend.to_core_backend().effective();
    if let Some(msg) = warning {
        tracing::warn!("{msg}");
    }
    tracing::info!(
        backend = effective_backend.as_str(),
        batch_mode = args.batch_mode.as_str(),
        platform = if cfg!(target_os = "macos") { "macos" } else { "linux" },
        "starting oxidize-server"
    );
    let model = load_model_runtime(&args)
        .unwrap_or_else(|error| panic!("failed to initialize model runtime: {error}"));
    let api_key = std::env::var("OXIDIZE_API_KEY")
        .ok()
        .filter(|value| !value.is_empty());

    let (model_opt, paged_opt) = if args.batch_mode == BatchMode::Paged {
        if let Some(runtime) = model {
            let paged = build_paged_runtime(&args, runtime.clone());
            (None, Some(paged))
        } else {
            (None, None)
        }
    } else {
        (model, None)
    };

    let mesh = if args.mesh {
        let state = mesh_cluster::MeshClusterState::new();
        let is_master = Arc::clone(&state.is_master);
        let mesh_handle = state.mesh_handle.clone();
        let port = args.mesh_port;
        tokio::spawn(async move {
            let result = oxidize_core::mesh::run_mesh_node(port, Some(is_master)).await;
            if let Err(ref e) = result {
                tracing::error!("mesh node error: {}", e);
            }
            let mut lock = mesh_handle.lock().await;
            *lock = None;
        });
        Some(state)
    } else {
        None
    };

    let state = AppState {
        limiter: Arc::new(RequestLimiter::new(RequestLimitConfig::default())),
        batcher: Arc::new(ContinuousBatcher::default()),
        auth: AuthConfig {
            api_key: api_key.map(Arc::<str>::from),
        },
        model: model_opt,
        paged: paged_opt,
        mesh,
    };
    let app = build_app_with_state(state);
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .expect("failed to bind TCP listener");
    axum::serve(listener, app)
        .await
        .expect("server runtime error");
}

impl BatchMode {
    fn as_str(&self) -> &'static str {
        match self {
            BatchMode::Sequential => "sequential",
            BatchMode::Paged => "paged",
        }
    }
}

fn build_paged_runtime(args: &Args, runtime: Arc<ModelRuntime>) -> Arc<PagedModelRuntime> {
    let inference_model = runtime
        .model
        .lock()
        .expect("model lock poisoned");
    let config = match inference_model.context_size().checked_div(16).unwrap_or(0) {
        0 => BlockPoolConfig::default(),
        blocks => BlockPoolConfig {
            block_size: 16,
            num_blocks: blocks * 4, // heuristic: 4x the context-size-in-blocks
            num_layers: inference_model.layer_count(),
            num_kv_heads: 0, // will be updated from model metadata below
            head_dim: 0,
            dtype: DType::F32,
        },
    };
    drop(inference_model);

    // Attempt to read num_kv_heads and head_dim from the model's config
    // by locking again and matching on LoadedModel to access InferenceConfig.
    let (num_kv_heads, head_dim) = {
        let model_guard = runtime.model.lock().expect("model lock poisoned");
        match &*model_guard {
            LoadedModel::Inference(m) => {
                let cfg = m.config();
                (
                    cfg.num_key_value_heads,
                    cfg.kv_head_dim(),
                )
            }
            LoadedModel::LayerWise(m) => {
                // LayerWiseModel doesn't expose config directly; use defaults.
                let cfg = m.config();
                (
                    cfg.num_key_value_heads,
                    cfg.kv_head_dim(),
                )
            }
            #[cfg(target_os = "macos")]
            LoadedModel::Mlx(m) => {
                let cfg = m.config();
                (
                    cfg.num_key_value_heads,
                    cfg.kv_head_dim(),
                )
            }
            #[cfg(not(target_os = "macos"))]
            LoadedModel::Mlx(m) => {
                let cfg = m.config();
                (
                    cfg.num_key_value_heads,
                    cfg.kv_head_dim(),
                )
            }
        }
    };

    let config = BlockPoolConfig {
        num_kv_heads,
        head_dim,
        ..config
    };

    let block_pool = BlockPool::new(config);
    let scheduler_config = SchedulerConfig {
        max_num_batched_tokens: args.prefill_batch_size,
        prefill_chunk_size: 16,
        max_num_running_seqs: 8,
    };
    let scheduler = Scheduler::new(scheduler_config, block_pool);

    Arc::new(PagedModelRuntime {
        runtime,
        scheduler: StdMutex::new(scheduler),
        next_seq_id: AtomicU64::new(1),
        block_size: config.block_size,
    })
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
    async fn chat_completions_accepts_vllm_sglang_guided_choice() {
        let request_body = json!({
            "model": "oxidize-default",
            "messages": [{"role": "user", "content": "pick"}],
            "guided_choice": ["yes", "no"],
            "top_k": 20,
            "min_p": 0.05,
            "typical_p": 0.9,
            "tail_free_z": 0.95,
            "stop": ["\n"],
            "seed": 7
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
        assert_eq!(parsed["choices"][0]["message"]["content"], "yes");
    }

    #[tokio::test]
    async fn completions_rejects_multiple_candidates_until_supported() {
        let request_body = json!({
            "model": "oxidize-default",
            "prompt": "hello",
            "n": 2
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
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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

    // === Validation-contract assertions ===

    /// VAL-SEC-001: Request body size limit (10MB default).
    /// A 15MB JSON body must be rejected with HTTP 413 before deserialization.
    #[tokio::test]
    async fn oversized_request_body_returns_413() {
        let app = build_app_with_config(RequestLimitConfig::default(), None, None);
        let big_payload = "x".repeat(15 * 1024 * 1024);
        let request_body = format!("{{\"model\":\"oxidize-default\",\"prompt\":\"{}\"}}", big_payload);

        let started = TokioInstant::now();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        let elapsed = started.elapsed();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(
            elapsed < Duration::from_millis(100),
            "413 should be returned within 100ms, got {:?}",
            elapsed
        );
    }

    /// VAL-SEC-001 (complementary): A body just under 10MB should succeed.
    #[tokio::test]
    async fn under_limit_request_body_is_accepted() {
        let app = build_app_with_config(RequestLimitConfig::default(), None, None);
        // 9.5 MB body (under the 10 MB limit).
        let big_payload = "x".repeat(9_500_000);
        let request_body = format!("{{\"model\":\"oxidize-default\",\"prompt\":\"{}\"}}", big_payload);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");

        // Without a model it falls through to placeholder, but the request body
        // limit layer must NOT reject it.
        assert_ne!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// VAL-SEC-002: Constant-time API key comparison.
    /// Measure comparison time across 1000 runs with correct vs incorrect keys.
    /// Variance between the two should be < 5%.
    #[test]
    fn api_key_comparison_is_constant_time() {
        use std::time::Instant;

        // Long key to amortise test overhead and stress the comparison loop.
        let expected_key = "a".repeat(256);
        let correct_key = expected_key.clone();
        let incorrect_key = format!("{}x", &expected_key[..255]);

        let mut headers_correct = axum::http::HeaderMap::new();
        headers_correct.insert("x-api-key", correct_key.parse().unwrap());

        let mut headers_incorrect = axum::http::HeaderMap::new();
        headers_incorrect.insert("x-api-key", incorrect_key.parse().unwrap());

        let runs = 1000usize;
        let mut correct_durations = Vec::with_capacity(runs);
        let mut incorrect_durations = Vec::with_capacity(runs);

        // Warm-up to stabilise branch-predictor state.
        for _ in 0..100 {
            let _ = request_has_api_key(&headers_correct, &expected_key);
            let _ = request_has_api_key(&headers_incorrect, &expected_key);
        }

        for _ in 0..runs {
            let start = Instant::now();
            let _ = request_has_api_key(&headers_correct, &expected_key);
            correct_durations.push(start.elapsed().as_nanos() as f64);

            let start = Instant::now();
            let _ = request_has_api_key(&headers_incorrect, &expected_key);
            incorrect_durations.push(start.elapsed().as_nanos() as f64);
        }

        let avg_correct = correct_durations.iter().sum::<f64>() / runs as f64;
        let avg_incorrect = incorrect_durations.iter().sum::<f64>() / runs as f64;

        // Allow up to 5% variance between correct and incorrect timings.
        let ratio = if avg_correct > avg_incorrect {
            avg_incorrect / avg_correct
        } else {
            avg_correct / avg_incorrect
        };
        assert!(
            ratio >= 0.95,
            "constant-time comparison variance exceeded 5%: avg_correct={avg_correct:.0}ns avg_incorrect={avg_incorrect:.0}ns ratio={ratio:.4}"
        );
    }

    /// VAL-PAGED-009 (server-level): Streaming SSE terminates with finish_reason and [DONE].
    #[tokio::test]
    async fn chat_completions_stream_returns_finish_reason_and_done() {
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
        assert!(body.contains("\"finish_reason\":\"stop\""), "stream should contain finish_reason=stop");
        assert!(body.contains("data: [DONE]"), "stream should end with [DONE]");
    }

    /// VAL-PAGED-014 (server-level): Usage counts are present in non-streaming response.
    #[tokio::test]
    async fn chat_completions_non_stream_has_usage_counts() {
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

        assert!(parsed["usage"]["prompt_tokens"].is_number(), "usage.prompt_tokens should be a number");
        assert!(parsed["usage"]["completion_tokens"].is_number(), "usage.completion_tokens should be a number");
        assert!(parsed["usage"]["total_tokens"].is_number(), "usage.total_tokens should be a number");
    }

    // === Mesh server tests ===

    #[tokio::test]
    async fn mesh_chat_completions_returns_503_when_mesh_disabled() {
        let app = build_app();
        let request_body = json!({
            "model": "oxidize-default",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/mesh/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        assert_eq!(parsed["error"]["type"], "mesh_disabled");
    }

    #[tokio::test]
    async fn mesh_chat_completions_returns_503_when_not_master() {
        let mesh = mesh_cluster::MeshClusterState::new();
        let app = build_app_with_full_config(
            RequestLimitConfig::default(),
            None,
            None,
            Some(mesh),
        );
        let request_body = json!({
            "model": "oxidize-default",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/mesh/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request_body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("request should be handled");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid json response");
        assert_eq!(parsed["error"]["type"], "mesh_not_master");
        assert!(
            parsed["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("contact the master"),
            "503 should suggest contacting the master node"
        );
    }

    #[tokio::test]
    async fn mesh_chat_completions_returns_echo_when_master() {
        let mesh = mesh_cluster::MeshClusterState::new();
        mesh.is_master
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let app = build_app_with_full_config(
            RequestLimitConfig::default(),
            None,
            None,
            Some(mesh),
        );
        let request_body = json!({
            "model": "oxidize-default",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/mesh/chat/completions")
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
        assert!(
            parsed["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .contains("mesh echo:"),
            "master should echo the request"
        );
        assert!(parsed["usage"]["prompt_tokens"].is_number());
        assert!(parsed["usage"]["completion_tokens"].is_number());
    }
}
