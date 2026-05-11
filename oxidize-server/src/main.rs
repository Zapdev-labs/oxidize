use std::{
    collections::VecDeque,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::Request,
    http::StatusCode,
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::{Instant as TokioInstant, sleep_until};

#[derive(Debug, Parser)]
#[command(name = "oxidize-server")]
struct Args {
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
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
}

#[derive(Clone)]
struct AuthConfig {
    api_key: Option<Arc<str>>,
}

fn build_app() -> Router {
    let api_key = std::env::var("OXIDIZE_API_KEY")
        .ok()
        .filter(|value| !value.is_empty());
    build_app_with_config(RequestLimitConfig::default(), api_key)
}

#[cfg(test)]
fn build_app_with_limits(config: RequestLimitConfig) -> Router {
    build_app_with_config(config, None)
}

fn build_app_with_config(config: RequestLimitConfig, api_key: Option<String>) -> Router {
    let state = AppState {
        limiter: Arc::new(RequestLimiter::new(config)),
        batcher: Arc::new(ContinuousBatcher::default()),
        auth: AuthConfig {
            api_key: api_key.map(Arc::<str>::from),
        },
    };
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
    id: &'static str,
    object: &'static str,
    owned_by: &'static str,
}

async fn chat_completions(Json(payload): Json<ChatCompletionRequest>) -> Response {
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

async fn completions(Json(payload): Json<CompletionRequest>) -> Response {
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

async fn models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelData {
            id: "oxidize-default",
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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .expect("failed to bind TCP listener");
    axum::serve(listener, build_app())
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
        let response = build_app_with_config(RequestLimitConfig::default(), Some("secret".into()))
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
        let response = build_app_with_config(RequestLimitConfig::default(), Some("secret".into()))
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
        let response = build_app_with_config(RequestLimitConfig::default(), Some("secret".into()))
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
        let response = build_app_with_config(RequestLimitConfig::default(), Some("secret".into()))
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
