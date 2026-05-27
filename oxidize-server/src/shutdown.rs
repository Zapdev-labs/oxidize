//! Graceful shutdown with request draining.
//!
//! Handles SIGTERM/SIGINT signals and drains in-flight requests before exiting.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use tokio::sync::watch;

/// Shutdown signal that can be sent to trigger graceful shutdown.
#[derive(Clone)]
pub struct ShutdownSignal {
    sender: watch::Sender<bool>,
}

impl ShutdownSignal {
    pub fn new() -> Self {
        let (sender, _receiver) = watch::channel(false);
        Self { sender }
    }

    pub fn trigger(&self) {
        let _ = self.sender.send(true);
        self.sender.send_modify(|v| *v = true);
    }

    pub fn is_shutting_down(&self) -> bool {
        *self.sender.borrow()
    }
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// Future that resolves when shutdown is triggered.
pub struct ShutdownFuture {
    receiver: watch::Receiver<bool>,
}

impl Future for ShutdownFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.receiver.has_changed() {
            Ok(true) if *self.receiver.borrow_and_update() => Poll::Ready(()),
            Ok(true) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Ok(false) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(_) => Poll::Ready(()),
        }
    }
}

/// Create a shutdown future that resolves on SIGTERM or SIGINT.
pub async fn wait_for_shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

/// Middleware that rejects new requests during shutdown.
pub async fn shutdown_guard_middleware(
    signal: axum::extract::State<Arc<ShutdownSignal>>,
    request: Request,
    next: Next,
) -> Response {
    if signal.is_shutting_down() {
        return (StatusCode::SERVICE_UNAVAILABLE, "Server is shutting down").into_response();
    }

    let response = next.run(request).await;

    if signal.is_shutting_down() {
        return (StatusCode::SERVICE_UNAVAILABLE, "Server is shutting down").into_response();
    }

    response
}

/// Run the server with graceful shutdown support.
pub async fn serve_with_graceful_shutdown(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    signal: ShutdownSignal,
) {
    let shutdown_future = async move {
        wait_for_shutdown_signal().await;
        tracing::info!("shutdown signal received, starting graceful shutdown");
        signal.trigger();
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_future)
        .await
        .expect("server runtime error");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn shutdown_signal_triggers() {
        let signal = ShutdownSignal::new();
        assert!(!signal.is_shutting_down());
        signal.trigger();
        assert!(signal.is_shutting_down());
    }

    #[tokio::test]
    async fn shutdown_future_resolves() {
        let signal = ShutdownSignal::new();
        let mut future = ShutdownFuture {
            receiver: signal.sender.subscribe(),
        };

        signal.trigger();

        let result = std::future::poll_fn(|cx| Pin::new(&mut future).poll(cx)).await;
        assert_eq!(result, ());
    }
}
