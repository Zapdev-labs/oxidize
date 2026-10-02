//! Request/response tracing middleware.
//!
//! Logged targets include the query string with credential parameters
//! (`api_key`, see [`redact_query`]) scrubbed.

use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};

/// Query parameters whose values never reach the logs.
const REDACTED_QUERY_PARAMS: &[&str] = &["api_key"];

/// Render `path?query` with credential parameter values replaced by `[REDACTED]`.
pub fn redact_query(path: &str, query: Option<&str>) -> String {
    let Some(query) = query.filter(|query| !query.is_empty()) else {
        return path.to_owned();
    };
    let redacted = query
        .split('&')
        .map(|pair| {
            let name = pair.split_once('=').map_or(pair, |(name, _)| name);
            if REDACTED_QUERY_PARAMS.contains(&name) {
                format!("{name}=[REDACTED]")
            } else {
                pair.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("&");
    format!("{path}?{redacted}")
}

pub async fn log_request_response(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = redact_query(request.uri().path(), request.uri().query());
    tracing::info!("{}", request_log_message(method.as_ref(), &path));
    let response = next.run(request).await;
    tracing::info!(
        "{}",
        response_log_message(method.as_ref(), &path, response.status())
    );
    response
}

pub fn request_log_message(method: &str, path: &str) -> String {
    format!("request {method} {path}")
}

pub fn response_log_message(method: &str, path: &str, status: StatusCode) -> String {
    format!("response {method} {path} {}", status.as_u16())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_log_message_has_expected_shape() {
        let message = request_log_message("GET", "/healthz");
        assert_eq!(message, "request GET /healthz");
    }

    #[test]
    fn query_api_key_is_redacted() {
        assert_eq!(
            redact_query("/v1/realtime", Some("api_key=hunter2")),
            "/v1/realtime?api_key=[REDACTED]"
        );
        assert_eq!(
            redact_query("/v1/realtime", Some("model=x&api_key=hunter2&b=2")),
            "/v1/realtime?model=x&api_key=[REDACTED]&b=2"
        );
        assert_eq!(redact_query("/healthz", None), "/healthz");
        assert_eq!(redact_query("/healthz", Some("")), "/healthz");
        assert_eq!(redact_query("/x", Some("my_api_key=v")), "/x?my_api_key=v");
        let message = request_log_message(
            "GET",
            &redact_query("/v1/realtime", Some("api_key=hunter2")),
        );
        assert!(!message.contains("hunter2"));
    }

    #[test]
    fn response_log_message_has_expected_shape() {
        let message = response_log_message("GET", "/healthz", StatusCode::OK);
        assert_eq!(message, "response GET /healthz 200");
    }
}
