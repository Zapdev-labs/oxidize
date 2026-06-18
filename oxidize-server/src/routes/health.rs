//! Liveness/readiness probes.
//!
//! `healthz`/`livez` return immediately; `readyz` only reports ready once a
//! model runtime has finished loading. This prevents Kubernetes from routing
//! traffic to a pod that cannot yet serve inference.

use axum::extract::State;
use axum::http::StatusCode;

use crate::app::AppState;

pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

pub async fn livez() -> StatusCode {
    StatusCode::OK
}

pub async fn readyz(State(state): State<AppState>) -> StatusCode {
    if state.model.is_some() || state.paged.is_some() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}
