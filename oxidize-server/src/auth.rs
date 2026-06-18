//! API-key auth middleware.
//!
//! Two header forms accepted: `x-api-key: <key>` or `Authorization: Bearer <key>`.
//! Comparison is constant-time (see [`request_has_api_key`]).

use std::sync::Arc;

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::app::AppState;

#[derive(Clone, Default)]
pub struct AuthConfig {
    pub api_key: Option<Arc<str>>,
    pub api_keys: Arc<[Arc<str>]>,
}

impl AuthConfig {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn from_keys(keys: impl IntoIterator<Item = String>) -> Self {
        let api_keys: Vec<Arc<str>> = keys
            .into_iter()
            .map(|key| key.trim().to_owned())
            .filter(|key| !key.is_empty())
            .map(Arc::<str>::from)
            .collect();

        Self {
            api_key: api_keys.first().cloned(),
            api_keys: Arc::from(api_keys),
        }
    }

    pub fn from_env() -> Self {
        let keys = std::env::var("OXIDIZE_API_KEYS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|key| !key.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .filter(|keys| !keys.is_empty())
            .or_else(|| {
                std::env::var("OXIDIZE_API_KEY")
                    .ok()
                    .map(|value| vec![value])
            })
            .unwrap_or_default();

        Self::from_keys(keys)
    }

    pub fn is_enabled(&self) -> bool {
        self.keys().next().is_some()
    }

    /// Iterate configured API keys without allocating per call.
    fn keys(&self) -> impl Iterator<Item = &str> {
        // `api_keys` is the source of truth when present; otherwise fall back
        // to the single `api_key`. Exactly one branch yields items.
        let from_list = self.api_keys.iter().map(AsRef::as_ref);
        let from_single = if self.api_keys.is_empty() {
            self.api_key.as_deref()
        } else {
            None
        };
        from_list.chain(from_single)
    }
}

pub async fn enforce_api_key(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    if !path.starts_with("/v1/") {
        return next.run(request).await;
    }
    if !state.auth.is_enabled() {
        return next.run(request).await;
    };
    let query = request.uri().query().map(str::to_owned);
    if state.auth.keys().into_iter().any(|expected_key| {
        request_has_api_key(request.headers(), expected_key)
            || query_has_api_key(query.as_deref(), expected_key)
    }) {
        return next.run(request).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "invalid api key"})),
    )
        .into_response()
}

pub fn request_has_api_key(headers: &axum::http::HeaderMap, expected_key: &str) -> bool {
    fn constant_time_eq(a: &str, b: &str) -> bool {
        use subtle::ConstantTimeEq;
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();
        a_bytes.ct_eq(b_bytes).into()
    }

    headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| constant_time_eq(value, expected_key))
        || headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .is_some_and(|token| constant_time_eq(token, expected_key))
}

/// Constant-time check of an `api_key=<key>` query parameter (WebSocket browser
/// fallback, since browsers cannot set custom headers on a WS upgrade).
pub fn query_has_api_key(query: Option<&str>, expected_key: &str) -> bool {
    use subtle::ConstantTimeEq;
    let Some(query) = query else {
        return false;
    };
    query
        .split('&')
        .filter_map(|pair| pair.strip_prefix("api_key="))
        .any(|value| value.as_bytes().ct_eq(expected_key.as_bytes()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VAL-SEC-002: Constant-time API key comparison.
    /// Measure comparison time across 1000 runs with correct vs incorrect keys.
    /// Variance between the two should be < 5%.
    #[test]
    fn api_key_comparison_is_constant_time() {
        use std::time::Instant;

        let expected_key = "a".repeat(256);
        let correct_key = expected_key.clone();
        let incorrect_key = format!("{}x", &expected_key[..255]);

        let mut headers_correct = axum::http::HeaderMap::new();
        headers_correct.insert("x-api-key", correct_key.parse().unwrap());

        let mut headers_incorrect = axum::http::HeaderMap::new();
        headers_incorrect.insert("x-api-key", incorrect_key.parse().unwrap());

        let runs = 1000usize;
        let mut correct_durations = Vec::with_capacity(runs);
        let mut incorrect_durations = Vec::with_capacity(runs);

        for _ in 0..100 {
            let _ = request_has_api_key(&headers_correct, &expected_key);
            let _ = request_has_api_key(&headers_incorrect, &expected_key);
        }

        for _ in 0..runs {
            let start = Instant::now();
            let _ = request_has_api_key(&headers_correct, &expected_key);
            correct_durations.push(start.elapsed().as_nanos() as f64);

            let start = Instant::now();
            let _ = request_has_api_key(&headers_incorrect, &expected_key);
            incorrect_durations.push(start.elapsed().as_nanos() as f64);
        }

        let avg_correct = correct_durations.iter().sum::<f64>() / runs as f64;
        let avg_incorrect = incorrect_durations.iter().sum::<f64>() / runs as f64;

        let ratio = if avg_correct > avg_incorrect {
            avg_incorrect / avg_correct
        } else {
            avg_correct / avg_incorrect
        };
        assert!(
            ratio >= 0.95,
            "constant-time comparison variance exceeded 5%: avg_correct={avg_correct:.0}ns avg_incorrect={avg_incorrect:.0}ns ratio={ratio:.4}"
        );
    }

    #[test]
    fn query_param_api_key_is_accepted() {
        assert!(query_has_api_key(Some("api_key=secret"), "secret"));
        assert!(query_has_api_key(
            Some("foo=1&api_key=secret&bar=2"),
            "secret"
        ));
        assert!(!query_has_api_key(Some("api_key=wrong"), "secret"));
        assert!(!query_has_api_key(None, "secret"));
    }

    #[test]
    fn auth_config_accepts_multiple_keys() {
        let auth = AuthConfig::from_keys(["alpha".to_string(), "bravo".to_string()]);
        assert!(auth.is_enabled());
        assert_eq!(auth.keys().collect::<Vec<_>>(), vec!["alpha", "bravo"]);
        assert_eq!(auth.api_key.as_deref(), Some("alpha"));
    }

    #[test]
    fn auth_config_ignores_empty_keys() {
        let auth = AuthConfig::from_keys([" alpha ".to_string(), "".to_string(), " ".to_string()]);
        assert_eq!(auth.keys().collect::<Vec<_>>(), vec!["alpha"]);
    }
}
