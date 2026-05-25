//! `AppState` and Axum router construction.

use std::sync::Arc;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware,
    routing::{get, post},
};

use crate::auth::{AuthConfig, enforce_api_key};
#[cfg(test)]
use crate::limits::RequestLimitConfig;
use crate::limits::{ContinuousBatcher, RequestLimiter, enforce_request_limits};
use crate::logging::log_request_response;
use crate::mesh_cluster::MeshClusterState;
use crate::openapi::openapi;
use crate::routes::{
    chat::chat_completions,
    completions::completions,
    embeddings::embeddings,
    health::{healthz, livez, readyz},
    mesh::mesh_chat_completions_handler,
    models::models,
};
use crate::runtime::model::ModelRuntime;
use crate::runtime::paged::PagedModelRuntime;

pub const MAX_BODY_SIZE_BYTES: usize = 10 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub limiter: Arc<RequestLimiter>,
    pub batcher: Arc<ContinuousBatcher>,
    pub auth: AuthConfig,
    pub model: Option<Arc<ModelRuntime>>,
    pub paged: Option<Arc<PagedModelRuntime>>,
    pub mesh: Option<MeshClusterState>,
}

pub fn build_app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/openapi.json", get(openapi))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .route(
            "/v1/mesh/chat/completions",
            post(mesh_chat_completions_handler),
        )
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

#[cfg(test)]
pub fn build_app() -> Router {
    let api_key = std::env::var("OXIDIZE_API_KEY")
        .ok()
        .filter(|value| !value.is_empty());
    build_app_with_config(RequestLimitConfig::default(), api_key, None)
}

#[cfg(test)]
pub fn build_app_with_limits(config: RequestLimitConfig) -> Router {
    build_app_with_config(config, None, None)
}

#[cfg(test)]
pub fn build_app_with_config(
    config: RequestLimitConfig,
    api_key: Option<String>,
    model: Option<Arc<ModelRuntime>>,
) -> Router {
    build_app_with_full_config(config, api_key, model, None)
}

#[cfg(test)]
pub fn build_app_with_full_config(
    config: RequestLimitConfig,
    api_key: Option<String>,
    model: Option<Arc<ModelRuntime>>,
    mesh: Option<MeshClusterState>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode, header},
    };
    use serde_json::{Value, json};
    use tower::ServiceExt;

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
                        "properties": { "answer": { "type": "string" } }
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

    /// VAL-SEC-001: Request body size limit (10MB default).
    /// A 15MB JSON body must be rejected with HTTP 413 before deserialization.
    #[tokio::test]
    async fn oversized_request_body_returns_413() {
        use std::time::Duration;
        use tokio::time::Instant as TokioInstant;

        let app = build_app_with_config(RequestLimitConfig::default(), None, None);
        let big_payload = "x".repeat(15 * 1024 * 1024);
        let request_body = format!(
            "{{\"model\":\"oxidize-default\",\"prompt\":\"{}\"}}",
            big_payload
        );

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
        let big_payload = "x".repeat(9_500_000);
        let request_body = format!(
            "{{\"model\":\"oxidize-default\",\"prompt\":\"{}\"}}",
            big_payload
        );

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

        assert_ne!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
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
        assert!(
            body.contains("\"finish_reason\":\"stop\""),
            "stream should contain finish_reason=stop"
        );
        assert!(
            body.contains("data: [DONE]"),
            "stream should end with [DONE]"
        );
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

        assert!(parsed["usage"]["prompt_tokens"].is_number());
        assert!(parsed["usage"]["completion_tokens"].is_number());
        assert!(parsed["usage"]["total_tokens"].is_number());
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
        let mesh = MeshClusterState::new();
        let app = build_app_with_full_config(RequestLimitConfig::default(), None, None, Some(mesh));
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
        let mesh = MeshClusterState::new();
        mesh.is_master
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let app = build_app_with_full_config(RequestLimitConfig::default(), None, None, Some(mesh));
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
