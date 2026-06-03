//! Integration test for the realtime websocket endpoint using a real WS client.

use std::net::SocketAddr;

use futures_util::{SinkExt, StreamExt};
use oxidize_server::app::{AppState, build_app_with_state};
use oxidize_server::audit::AuditLogger;
use oxidize_server::auth::AuthConfig;
use oxidize_server::limits::{ContinuousBatcher, RequestLimitConfig, RequestLimiter};
use oxidize_server::metrics::MetricsRegistry;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio_tungstenite::tungstenite::Message;

fn test_state() -> AppState {
    AppState {
        limiter: Arc::new(RequestLimiter::new(RequestLimitConfig::default())),
        batcher: Arc::new(ContinuousBatcher::default()),
        auth: AuthConfig::default(),
        model: None,
        paged: None,
        mesh: None,
        audit: Arc::new(AuditLogger::new()),
        metrics: Arc::new(MetricsRegistry::new().expect("metrics")),
    }
}

async fn spawn_server() -> SocketAddr {
    let app = build_app_with_state(test_state());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

#[tokio::test]
async fn realtime_lifecycle_emits_session_created_and_response_events() {
    let addr = spawn_server().await;
    let url = format!("ws://{addr}/v1/realtime");
    let (mut socket, _) = tokio_tungstenite::connect_async(url).await.expect("connect");

    // First server event must be session.created.
    let first = next_json(&mut socket).await;
    assert_eq!(first["type"], "session.created");

    // Add a user message, then request a response.
    socket
        .send(Message::Text(
            json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "hi" }]
                }
            })
            .to_string(),
        ))
        .await
        .unwrap();
    let created = next_json(&mut socket).await;
    assert_eq!(created["type"], "conversation.item.created");

    socket
        .send(Message::Text(json!({ "type": "response.create" }).to_string()))
        .await
        .unwrap();

    // response.created should arrive, then (no model) an error event.
    let response_created = next_json(&mut socket).await;
    assert_eq!(response_created["type"], "response.created");

    let next = next_json(&mut socket).await;
    assert_eq!(next["type"], "error");
    assert_eq!(next["error"]["message"], "no model loaded");
}

async fn next_json<S>(socket: &mut S) -> Value
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        let message = socket.next().await.expect("stream open").expect("ws ok");
        if let Message::Text(text) = message {
            return serde_json::from_str(&text).expect("valid json");
        }
    }
}
