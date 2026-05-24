//! Liveness/readiness probes. All return 200 immediately.

use axum::http::StatusCode;

pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

pub async fn livez() -> StatusCode {
    StatusCode::OK
}

pub async fn readyz() -> StatusCode {
    StatusCode::OK
}
