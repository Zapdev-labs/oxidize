use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use axum::{
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
    Json, Router,
    http::StatusCode,
};
use clap::Parser;
use futures_util::stream;
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
    stream: bool,
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

    if payload.stream {
        let model = payload.model;
        let model_for_end = model.clone();
        let stream = stream::iter(vec![
            Ok::<Event, std::convert::Infallible>(
                Event::default().data(json!({
                    "id": "chatcmpl-placeholder",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": { "role": "assistant", "content": "" },
                            "finish_reason": null
                        }
                    ]
                }).to_string()),
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
        .into_response()
}

async fn completions(Json(payload): Json<CompletionRequest>) -> Response {
    let _ = &payload.prompt;

    if payload.stream {
        let model = payload.model;
        let model_for_end = model.clone();
        let stream = stream::iter(vec![
            Ok::<Event, std::convert::Infallible>(
                Event::default().data(json!({
                    "id": "cmpl-placeholder",
                    "object": "text_completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "finish_reason": null
                        }
                    ]
                }).to_string()),
            ),
            Ok(Event::default().data(
                json!({
                    "id": "cmpl-placeholder",
                    "object": "text_completion",
                    "created": 0,
                    "model": model_for_end,
                    "choices": [{ "index": 0, "text": "", "finish_reason": "stop" }]
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
        .into_response()
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
    async fn chat_completions_stream_returns_sse_events() {
        let request_body = json!({
            "model": "llamas-default",
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
    async fn completions_stream_returns_sse_events() {
        let request_body = json!({
            "model": "llamas-default",
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
