//! `POST /v1/mesh/chat/completions` thin wrapper around `mesh_cluster::mesh_chat_completions`.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::app::AppState;
use crate::mesh_cluster::{self, MeshChatRequest};

pub async fn mesh_chat_completions_handler(
    State(state): State<AppState>,
    Json(payload): Json<MeshChatRequest>,
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
