//! Structured audit logging for production observability.
//!
//! Provides request-scoped audit events with unique IDs, token counts,
//! timing, and structured JSON output suitable for ingestion into ELK,
//! Loki, Datadog, or similar log aggregation systems.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{ConnectInfo, Request},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use serde::Serialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{Level, info, span};
use uuid::Uuid;

/// Unique identifier for a request.
pub type RequestId = String;

/// Audit event severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditSeverity {
    Info,
    Warning,
    Error,
    Security,
}

/// Structured audit event for a single request.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    /// Unique request identifier.
    pub request_id: RequestId,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
    /// Event type.
    pub event_type: String,
    /// Severity level.
    pub severity: AuditSeverity,
    /// Client IP address.
    pub client_ip: Option<String>,
    /// API key identifier (hashed for security).
    pub api_key_hash: Option<String>,
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// Model identifier.
    pub model: Option<String>,
    /// Tokens in the prompt.
    pub prompt_tokens: Option<usize>,
    /// Tokens generated.
    pub completion_tokens: Option<usize>,
    /// Total tokens (prompt + completion).
    pub total_tokens: Option<usize>,
    /// Request duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// HTTP status code.
    pub status_code: Option<u16>,
    /// Sampling temperature used.
    pub temperature: Option<f32>,
    /// Stop reason (e.g., "stop", "length", "content_filter").
    pub stop_reason: Option<String>,
    /// Whether the request was streamed.
    pub streamed: Option<bool>,
    /// Error message if the request failed.
    pub error: Option<String>,
    /// Rate limit hit.
    pub rate_limited: Option<bool>,
    /// Additional custom fields.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl AuditEvent {
    pub fn new(request_id: RequestId, event_type: &str) -> Self {
        Self {
            request_id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            event_type: event_type.to_string(),
            severity: AuditSeverity::Info,
            client_ip: None,
            api_key_hash: None,
            method: String::new(),
            path: String::new(),
            model: None,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            duration_ms: None,
            status_code: None,
            temperature: None,
            stop_reason: None,
            streamed: None,
            error: None,
            rate_limited: None,
            extra: HashMap::new(),
        }
    }

    pub fn with_client_ip(mut self, ip: &str) -> Self {
        self.client_ip = Some(ip.to_string());
        self
    }

    pub fn with_api_key(mut self, key: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        self.api_key_hash = Some(format!("{:x}", hasher.finish()));
        self
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    pub fn with_tokens(mut self, prompt: usize, completion: usize) -> Self {
        self.prompt_tokens = Some(prompt);
        self.completion_tokens = Some(completion);
        self.total_tokens = Some(prompt + completion);
        self
    }

    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ms = Some(duration.as_millis() as u64);
        self
    }

    pub fn with_status(mut self, status: StatusCode) -> Self {
        self.status_code = Some(status.as_u16());
        self
    }

    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_string());
        self.severity = AuditSeverity::Error;
        self
    }

    pub fn with_temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_stop_reason(mut self, stop_reason: &str) -> Self {
        self.stop_reason = Some(stop_reason.to_string());
        self
    }

    pub fn with_streamed(mut self, streamed: bool) -> Self {
        self.streamed = Some(streamed);
        self
    }

    pub fn with_security_event(mut self) -> Self {
        self.severity = AuditSeverity::Security;
        self
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            json!({"error": "failed to serialize audit event"}).to_string()
        })
    }
}

/// Audit logger that buffers and flushes events.
pub struct AuditLogger {
    sender: mpsc::UnboundedSender<AuditEvent>,
}

impl AuditLogger {
    pub fn new() -> Self {
        let (sender, mut receiver) = mpsc::unbounded_channel::<AuditEvent>();

        #[cfg(not(test))]
        {
            tokio::spawn(async move {
                while let Some(event) = receiver.recv().await {
                    let json = event.to_json();
                    match event.severity {
                        AuditSeverity::Error => tracing::error!(target: "audit", "{}", json),
                        AuditSeverity::Warning => tracing::warn!(target: "audit", "{}", json),
                        AuditSeverity::Security => tracing::error!(target: "audit_security", "{}", json),
                        AuditSeverity::Info => tracing::info!(target: "audit", "{}", json),
                    }
                }
            });
        }

        #[cfg(test)]
        {
            drop(receiver);
        }

        Self { sender }
    }

    pub fn log(&self, event: AuditEvent) {
        let _ = self.sender.send(event);
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait to extract request ID from request extensions.
pub trait RequestIdExt {
    fn request_id(&self) -> Option<&RequestId>;
}

impl RequestIdExt for Request {
    fn request_id(&self) -> Option<&RequestId> {
        self.extensions().get::<RequestId>()
    }
}

/// Middleware that assigns a unique request ID and captures audit data.
pub async fn audit_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    mut request: Request,
    next: Next,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let start = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let client_ip = addr.ip().to_string();

    request.extensions_mut().insert(request_id.clone());

    let api_key = request
        .headers()
        .get("x-api-key")
        .or_else(|| request.headers().get("authorization"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let response = next.run(request).await;
    let duration = start.elapsed();
    let status = response.status();

    let mut event = AuditEvent::new(request_id.clone(), "http_request")
        .with_client_ip(&client_ip)
        .with_duration(duration)
        .with_status(status);

    event.method = method.to_string();
    event.path = path;

    if let Some(key) = api_key {
        event = event.with_api_key(&key);
    }

    if status.is_server_error() || status.is_client_error() {
        event.severity = if status == StatusCode::TOO_MANY_REQUESTS {
            event.rate_limited = Some(true);
            AuditSeverity::Warning
        } else if status == StatusCode::UNAUTHORIZED {
            event.severity = AuditSeverity::Security;
            AuditSeverity::Security
        } else {
            AuditSeverity::Error
        };
    }

    let json = event.to_json();
    match event.severity {
        AuditSeverity::Error => tracing::error!(target: "audit", "{}", json),
        AuditSeverity::Warning => tracing::warn!(target: "audit", "{}", json),
        AuditSeverity::Security => tracing::error!(target: "audit_security", "{}", json),
        AuditSeverity::Info => tracing::info!(target: "audit", "{}", json),
    }

    response
}

/// Log a generation completion event.
pub fn log_generation_event(
    logger: &AuditLogger,
    request_id: RequestId,
    model: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    duration: Duration,
    temperature: Option<f32>,
    stop_reason: &str,
    streamed: bool,
) {
    let event = AuditEvent::new(request_id, "generation_complete")
        .with_model(model)
        .with_tokens(prompt_tokens, completion_tokens)
        .with_duration(duration)
        .with_temperature(temperature)
        .with_stop_reason(stop_reason)
        .with_streamed(streamed);

    logger.log(event);
}

/// Log a security event (auth failure, etc.).
pub fn log_security_event(
    logger: &AuditLogger,
    request_id: RequestId,
    event_type: &str,
    client_ip: &str,
    details: &str,
) {
    let event = AuditEvent::new(request_id, event_type)
        .with_client_ip(client_ip)
        .with_error(details)
        .with_security_event();

    logger.log(event);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_event_serializes_to_json() {
        let event = AuditEvent::new("req-123".to_string(), "test_event")
            .with_model("llama-7b")
            .with_tokens(10, 20)
            .with_duration(Duration::from_millis(150))
            .with_status(StatusCode::OK);

        let json = event.to_json();
        assert!(json.contains("\"request_id\":\"req-123\""));
        assert!(json.contains("\"event_type\":\"test_event\""));
        assert!(json.contains("\"model\":\"llama-7b\""));
        assert!(json.contains("\"prompt_tokens\":10"));
        assert!(json.contains("\"completion_tokens\":20"));
        assert!(json.contains("\"total_tokens\":30"));
        assert!(json.contains("\"duration_ms\":150"));
        assert!(json.contains("\"status_code\":200"));
    }

    #[test]
    fn api_key_hashing_is_deterministic() {
        let event1 = AuditEvent::new("req-1".to_string(), "test").with_api_key("secret-key-123");
        let event2 = AuditEvent::new("req-2".to_string(), "test").with_api_key("secret-key-123");
        assert_eq!(event1.api_key_hash, event2.api_key_hash);
        assert_ne!(event1.api_key_hash, Some("secret-key-123".to_string()));
    }

    #[test]
    fn severity_escalates_on_error() {
        let event = AuditEvent::new("req-1".to_string(), "test").with_error("something failed");
        assert_eq!(event.severity, AuditSeverity::Error);
    }

    #[test]
    fn security_event_has_security_severity() {
        let event = AuditEvent::new("req-1".to_string(), "auth_failure")
            .with_client_ip("192.168.1.1")
            .with_error("invalid api key")
            .with_security_event();
        assert_eq!(event.severity, AuditSeverity::Security);
    }
}
