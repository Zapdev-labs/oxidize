use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use axum::{
    Json, Router,
    http::StatusCode,
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Parser)]
#[command(name = "llamas-server")]
struct Args {
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

fn build_app() -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
}

async fn healthz() -> StatusCode {
    StatusCode::OK
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessageInput>,
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

async fn chat_completions(
    Json(payload): Json<ChatCompletionRequest>,
) -> (StatusCode, Json<Value>) {
    let _input_size: usize = payload
        .messages
        .iter()
        .map(|message| message.role.len() + message.content.len())
        .sum();
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
                        "content": ""
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
}

async fn completions(Json(payload): Json<CompletionRequest>) -> (StatusCode, Json<Value>) {
    let _ = &payload.prompt;
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
                    "text": "",
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
}

async fn models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelData {
            id: "llamas-default",
            object: "model",
            owned_by: "llamas",
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
    use tower::ServiceExt;

    #[test]
    fn args_use_expected_defaults() {
        let args = Args::parse_from(["llamas-server"]);
        assert_eq!(args.host, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(args.port, 8080);
    }

    #[test]
    fn args_accept_explicit_values() {
        let args = Args::parse_from(["llamas-server", "--host", "0.0.0.0", "--port", "3000"]);
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
    async fn chat_completions_returns_openai_shape() {
        let request_body = json!({
            "model": "llamas-default",
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
        assert_eq!(parsed["model"], "llamas-default");
    }

    #[tokio::test]
    async fn completions_returns_openai_shape() {
        let request_body = json!({
            "model": "llamas-default",
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
        assert_eq!(parsed["model"], "llamas-default");
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
            "model": "llamas-default",
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
        assert_eq!(parsed["model"], "llamas-default");
        assert_eq!(parsed["data"][0]["object"], "embedding");
    }
}
