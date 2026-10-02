//! API-key auth middleware.
//!
//! Two header forms accepted: `x-api-key: <key>` or `Authorization: Bearer <key>`.
//! The `api_key=<key>` query parameter is honored only on the WebSocket upgrade
//! route ([`QUERY_KEY_ROUTES`]), since browsers cannot set headers there.
//! Comparison is constant-time (see [`request_has_api_key`]).
//!
//! When no key is configured every route is open. [`check_bind_exposure`] is
//! run at startup to warn (or, with `--require-auth-on-public-bind`, refuse)
//! when that open server is bound to a non-loopback address.

use std::net::IpAddr;
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

/// Routes where the `api_key` query parameter is accepted in place of a header.
pub const QUERY_KEY_ROUTES: &[&str] = &["/v1/realtime"];

/// Whether `path` requires an API key (when auth is enabled). Health probes and
/// the OpenAPI document stay open; `/metrics` is gated with the API.
pub fn path_requires_auth(path: &str) -> bool {
    path.starts_with("/v1/") || path == "/metrics"
}

/// Outcome of the startup bind/auth posture check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindExposure {
    /// Auth is enabled or the bind is loopback-only.
    Ok,
    /// Auth disabled on a routable bind; serve, but log this warning.
    Warn(String),
    /// Auth disabled on a routable bind under `--require-auth-on-public-bind`.
    Refuse(String),
}

/// Classify the bind address against the auth posture. Pure; callers log the
/// warning or exit with status 2 on [`BindExposure::Refuse`].
pub fn check_bind_exposure(
    host: IpAddr,
    auth: &AuthConfig,
    require_auth_on_public_bind: bool,
) -> BindExposure {
    if auth.is_enabled() || host.is_loopback() {
        return BindExposure::Ok;
    }
    let message = format!(
        "API auth is disabled (OXIDIZE_API_KEY/OXIDIZE_API_KEYS unset) and the server is bound to \
         non-loopback address {host}: /v1/* inference, /v1/realtime, /v1/mesh/* and /metrics are \
         reachable without credentials by any host that can route to it"
    );
    if require_auth_on_public_bind {
        BindExposure::Refuse(format!(
            "{message}; refusing to start (--require-auth-on-public-bind). Set OXIDIZE_API_KEY or bind 127.0.0.1"
        ))
    } else {
        BindExposure::Warn(message)
    }
}

pub async fn enforce_api_key(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    if !path_requires_auth(path) {
        return next.run(request).await;
    }
    if !state.auth.is_enabled() {
        return next.run(request).await;
    };
    let query = QUERY_KEY_ROUTES
        .contains(&path)
        .then(|| request.uri().query().map(str::to_owned))
        .flatten();
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

fn constant_time_eq(a: &str, b: &str) -> bool {
    use subtle::ConstantTimeEq;
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

pub fn request_has_api_key(headers: &axum::http::HeaderMap, expected_key: &str) -> bool {
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
    let Some(query) = query else {
        return false;
    };
    query
        .split('&')
        .filter_map(|pair| pair.strip_prefix("api_key="))
        .any(|value| constant_time_eq(value, expected_key))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VAL-SEC-002: Constant-time API key comparison.
    #[test]
    fn api_key_comparison_is_constant_time() {
        assert!(constant_time_eq("secret", "secret"));
        assert!(!constant_time_eq("Secret", "secret"));
        assert!(!constant_time_eq("secreu", "secret"));
        assert!(!constant_time_eq("secret!", "secret"));

        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-api-key", "secret".parse().unwrap());
        assert!(request_has_api_key(&headers, "secret"));
        assert!(!request_has_api_key(&headers, "secreu"));

        headers.clear();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer secret".parse().unwrap(),
        );
        assert!(request_has_api_key(&headers, "secret"));
        assert!(!request_has_api_key(&headers, "secret!"));
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
    fn metrics_is_gated_and_probes_are_not() {
        assert!(path_requires_auth("/metrics"));
        assert!(path_requires_auth("/v1/models"));
        assert!(!path_requires_auth("/healthz"));
        assert!(!path_requires_auth("/livez"));
        assert!(!path_requires_auth("/readyz"));
    }

    #[test]
    fn bind_exposure_warns_only_on_open_public_bind() {
        let open = AuthConfig::disabled();
        let keyed = AuthConfig::from_keys(["k".to_string()]);
        let any: IpAddr = "0.0.0.0".parse().unwrap();
        let lan: IpAddr = "192.168.1.15".parse().unwrap();
        let lo4: IpAddr = "127.0.0.1".parse().unwrap();
        let lo6: IpAddr = "::1".parse().unwrap();

        assert!(
            matches!(check_bind_exposure(any, &open, false), BindExposure::Warn(m) if m.contains("0.0.0.0"))
        );
        assert!(matches!(
            check_bind_exposure(lan, &open, false),
            BindExposure::Warn(_)
        ));
        assert!(matches!(
            check_bind_exposure(any, &open, true),
            BindExposure::Refuse(_)
        ));
        assert_eq!(check_bind_exposure(lo4, &open, true), BindExposure::Ok);
        assert_eq!(check_bind_exposure(lo6, &open, false), BindExposure::Ok);
        assert_eq!(check_bind_exposure(any, &keyed, true), BindExposure::Ok);
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
