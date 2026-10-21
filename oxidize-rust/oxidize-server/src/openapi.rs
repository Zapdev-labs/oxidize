//! OpenAPI 3.1 spec handler.

use axum::Json;
use serde_json::{Value, json};

pub async fn openapi() -> Json<Value> {
    Json(openapi_spec())
}

pub fn openapi_spec() -> Value {
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
                    "responses": { "200": { "description": "OK" } }
                }
            },
            "/livez": {
                "get": {
                    "summary": "Liveness check",
                    "responses": { "200": { "description": "OK" } }
                }
            },
            "/readyz": {
                "get": {
                    "summary": "Readiness check",
                    "responses": { "200": { "description": "OK" } }
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
