//! `oxidize-server` library: HTTP serving surface for the inference engine.
//!
//! The binary in `main.rs` is a thin wrapper that parses CLI args, loads the
//! model, and binds the Axum router built here.

pub mod app;
pub mod audit;
pub mod auth;
pub mod cli;
pub mod limits;
pub mod logging;
pub mod mesh_cluster;
pub mod metrics;
pub mod openapi;
pub mod realtime;
pub mod routes;
pub mod runtime;
pub mod schema;
pub mod shutdown;

pub use app::{AppState, MAX_BODY_SIZE_BYTES, build_app_with_state};
pub use auth::AuthConfig;
pub use cli::{Args, Backend, BatchMode};
pub use limits::{ContinuousBatchConfig, ContinuousBatcher, RequestLimitConfig, RequestLimiter};
pub use runtime::generate::GenerationError;
pub use runtime::model::{LoadedModel, ModelRuntime, load_model_runtime};
pub use runtime::paged::{PagedModelRuntime, build_paged_runtime};
