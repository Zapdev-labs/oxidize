//! `POST /v1/embeddings` placeholder.

use axum::{Json, http::StatusCode};
use serde_json::{Value, json};

use crate::schema::EmbeddingsRequest;

pub async fn embeddings(Json(payload): Json<EmbeddingsRequest>) -> (StatusCode, Json<Value>) {
    let _ = &payload.input;
    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": [
                { "object": "embedding", "embedding": [], "index": 0 }
            ],
            "model": payload.model,
            "usage": { "prompt_tokens": 0, "total_tokens": 0 }
        })),
    )
}
