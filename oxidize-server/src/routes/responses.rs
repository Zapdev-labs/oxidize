//! Shared response builders for chat/completions endpoints.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use axum::{
    Json,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::{StreamExt, stream};
use serde_json::{Value, json};

use crate::runtime::generate::{GenerationError, GenerationResult};

pub fn chat_completion_response(model: String, result: GenerationResult) -> Response {
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
                    "message": { "role": "assistant", "content": result.text },
                    "finish_reason": "stop"
                }
            ],
            "usage": usage_json(result.prompt_tokens, result.completion_tokens)
        })),
    )
        .into_response()
}

pub fn completion_response(model: String, result: GenerationResult) -> Response {
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

pub fn chat_completion_stream_response(model: String, content: String) -> Response {
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

pub fn completion_stream_response(model: String, text: String) -> Response {
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
                        { "index": 0, "text": text, "finish_reason": null }
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
pub fn chat_completion_stream_response_paged(
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
                    Some((
                        Ok::<Event, std::convert::Infallible>(event),
                        (model, false, rx, cancel),
                    ))
                }
                Some(Err(_error)) => {
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

pub fn usage_json(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens.saturating_add(completion_tokens)
    })
}

pub fn generation_error_response(error: GenerationError) -> Response {
    match error {
        GenerationError::KvCacheExhausted => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": { "message": "KV cache exhausted", "type": "insufficient_memory" }
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

pub fn validate_candidate_count(n: Option<usize>, best_of: Option<usize>) -> Option<Response> {
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

pub fn model_not_found(model: &str) -> Response {
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

pub fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
