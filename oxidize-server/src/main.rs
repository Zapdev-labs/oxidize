//! `oxidize-server` binary: parses args, loads the model, and serves on Axum.
//!
//! All the actual logic lives in the [`oxidize_server`] library.

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;

use oxidize_server::{
    AppState, Args, AuthConfig, BatchMode, ContinuousBatcher, RequestLimitConfig, RequestLimiter,
    audit::AuditLogger, build_app_with_state, build_paged_runtime, load_model_runtime,
    mesh_cluster::MeshClusterState, metrics::MetricsRegistry,
    shutdown::serve_with_graceful_shutdown,
};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let (effective_backend, warning) = args.backend.to_core_backend().effective();
    if let Some(msg) = warning {
        tracing::warn!("{msg}");
    }
    tracing::info!(
        backend = effective_backend.as_str(),
        batch_mode = args.batch_mode.as_str(),
        platform = if cfg!(target_os = "macos") {
            "macos"
        } else {
            "linux"
        },
        "starting oxidize-server"
    );

    let model = match load_model_runtime(&args) {
        Ok(m) => m,
        Err(error) => {
            tracing::error!("failed to initialize model runtime: {error}");
            std::process::exit(1);
        }
    };
    let api_key = std::env::var("OXIDIZE_API_KEY")
        .ok()
        .filter(|value| !value.is_empty());

    let (model_opt, paged_opt) = if args.batch_mode == BatchMode::Paged {
        if let Some(runtime) = model {
            let paged = build_paged_runtime(&args, runtime.clone());
            (None, Some(paged))
        } else {
            (None, None)
        }
    } else {
        (model, None)
    };

    let mesh = if args.mesh {
        let state = MeshClusterState::new();
        let is_master = Arc::clone(&state.is_master);
        let mesh_handle = state.mesh_handle.clone();
        let port = args.mesh_port;
        tokio::spawn(async move {
            let result = oxidize_core::mesh::run_mesh_node(port, Some(is_master), None, None).await;
            if let Err(ref e) = result {
                tracing::error!("mesh node error: {}", e);
            }
            let mut lock = mesh_handle.lock().await;
            *lock = None;
        });
        Some(state)
    } else {
        None
    };

    let state = AppState {
        limiter: Arc::new(RequestLimiter::new(RequestLimitConfig::default())),
        batcher: Arc::new(ContinuousBatcher::default()),
        auth: AuthConfig {
            api_key: api_key.map(Arc::<str>::from),
        },
        model: model_opt,
        paged: paged_opt,
        mesh,
        audit: Arc::new(AuditLogger::new()),
        metrics: Arc::new(MetricsRegistry::new().expect("failed to create metrics registry")),
    };
    let app = build_app_with_state(state);
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .expect("failed to bind TCP listener");
    let shutdown_signal = oxidize_server::shutdown::ShutdownSignal::new();
    serve_with_graceful_shutdown(listener, app, shutdown_signal).await;
}
