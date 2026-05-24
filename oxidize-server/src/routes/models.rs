//! `GET /v1/models` handler.

use axum::{Json, extract::State};

use crate::app::AppState;
use crate::schema::{ModelData, ModelsResponse};

pub async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
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
