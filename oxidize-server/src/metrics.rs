//! Prometheus metrics endpoint for production monitoring.
//!
//! Exposes request latency histograms, token counters, queue depth gauges,
//! cache hit ratios, and GPU memory usage via the standard `/metrics` endpoint.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, Histogram, HistogramOpts, HistogramVec, IntCounter,
    IntGauge, Opts, Registry, TextEncoder,
};

use crate::app::AppState;

/// Metrics registry holding all prometheus metrics.
pub struct MetricsRegistry {
    pub registry: Registry,

    // Request metrics
    pub requests_total: CounterVec,
    pub request_duration: HistogramVec,
    pub requests_in_flight: IntGauge,

    // Token metrics
    pub tokens_generated_total: IntCounter,
    pub tokens_prompt_total: IntCounter,
    pub tokens_per_second: Gauge,

    // Queue metrics
    pub queue_depth: IntGauge,
    pub queue_wait_duration: Histogram,

    // Cache metrics
    pub cache_hits_total: IntCounter,
    pub cache_misses_total: IntCounter,
    pub cache_hit_ratio: Gauge,

    // KV cache metrics
    pub kv_cache_size_bytes: Gauge,
    pub kv_cache_blocks_used: IntGauge,
    pub kv_cache_blocks_total: IntGauge,

    // Error metrics
    pub errors_total: CounterVec,

    // Rate limit metrics
    pub rate_limits_total: IntCounter,

    // Model metrics
    pub model_load_duration: Histogram,
    pub model_inference_duration: Histogram,
}

impl MetricsRegistry {
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        let requests_total = CounterVec::new(
            Opts::new("oxidize_requests_total", "Total HTTP requests"),
            &["method", "path", "status"],
        )?;
        registry.register(Box::new(requests_total.clone()))?;

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                "oxidize_request_duration_seconds",
                "HTTP request duration in seconds",
            )
            .buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &["method", "path"],
        )?;
        registry.register(Box::new(request_duration.clone()))?;

        let requests_in_flight = IntGauge::with_opts(Opts::new(
            "oxidize_requests_in_flight",
            "Number of requests currently being processed",
        ))?;
        registry.register(Box::new(requests_in_flight.clone()))?;

        let tokens_generated_total = IntCounter::with_opts(Opts::new(
            "oxidize_tokens_generated_total",
            "Total number of completion tokens generated",
        ))?;
        registry.register(Box::new(tokens_generated_total.clone()))?;

        let tokens_prompt_total = IntCounter::with_opts(Opts::new(
            "oxidize_tokens_prompt_total",
            "Total number of prompt tokens processed",
        ))?;
        registry.register(Box::new(tokens_prompt_total.clone()))?;

        let tokens_per_second = Gauge::with_opts(Opts::new(
            "oxidize_tokens_per_second",
            "Current token generation throughput",
        ))?;
        registry.register(Box::new(tokens_per_second.clone()))?;

        let queue_depth = IntGauge::with_opts(Opts::new(
            "oxidize_queue_depth",
            "Number of requests waiting in the queue",
        ))?;
        registry.register(Box::new(queue_depth.clone()))?;

        let queue_wait_duration = Histogram::with_opts(
            HistogramOpts::new(
                "oxidize_queue_wait_duration_seconds",
                "Time spent waiting in the request queue",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        )?;
        registry.register(Box::new(queue_wait_duration.clone()))?;

        let cache_hits_total = IntCounter::with_opts(Opts::new(
            "oxidize_cache_hits_total",
            "Total number of KV cache hits",
        ))?;
        registry.register(Box::new(cache_hits_total.clone()))?;

        let cache_misses_total = IntCounter::with_opts(Opts::new(
            "oxidize_cache_misses_total",
            "Total number of KV cache misses",
        ))?;
        registry.register(Box::new(cache_misses_total.clone()))?;

        let cache_hit_ratio = Gauge::with_opts(Opts::new(
            "oxidize_cache_hit_ratio",
            "Ratio of cache hits to total cache lookups",
        ))?;
        registry.register(Box::new(cache_hit_ratio.clone()))?;

        let kv_cache_size_bytes = Gauge::with_opts(Opts::new(
            "oxidize_kv_cache_size_bytes",
            "Current KV cache size in bytes",
        ))?;
        registry.register(Box::new(kv_cache_size_bytes.clone()))?;

        let kv_cache_blocks_used = IntGauge::with_opts(Opts::new(
            "oxidize_kv_cache_blocks_used",
            "Number of KV cache blocks currently in use",
        ))?;
        registry.register(Box::new(kv_cache_blocks_used.clone()))?;

        let kv_cache_blocks_total = IntGauge::with_opts(Opts::new(
            "oxidize_kv_cache_blocks_total",
            "Total number of KV cache blocks available",
        ))?;
        registry.register(Box::new(kv_cache_blocks_total.clone()))?;

        let errors_total = CounterVec::new(
            Opts::new("oxidize_errors_total", "Total number of errors"),
            &["type"],
        )?;
        registry.register(Box::new(errors_total.clone()))?;

        let rate_limits_total = IntCounter::with_opts(Opts::new(
            "oxidize_rate_limits_total",
            "Total number of rate-limited requests",
        ))?;
        registry.register(Box::new(rate_limits_total.clone()))?;

        let model_load_duration = Histogram::with_opts(
            HistogramOpts::new(
                "oxidize_model_load_duration_seconds",
                "Time to load a model",
            )
            .buckets(vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]),
        )?;
        registry.register(Box::new(model_load_duration.clone()))?;

        let model_inference_duration = Histogram::with_opts(
            HistogramOpts::new(
                "oxidize_model_inference_duration_seconds",
                "Time for a single model forward pass",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        )?;
        registry.register(Box::new(model_inference_duration.clone()))?;

        Ok(Self {
            registry,
            requests_total,
            request_duration,
            requests_in_flight,
            tokens_generated_total,
            tokens_prompt_total,
            tokens_per_second,
            queue_depth,
            queue_wait_duration,
            cache_hits_total,
            cache_misses_total,
            cache_hit_ratio,
            kv_cache_size_bytes,
            kv_cache_blocks_used,
            kv_cache_blocks_total,
            errors_total,
            rate_limits_total,
            model_load_duration,
            model_inference_duration,
        })
    }

    pub fn record_request(&self, method: &str, path: &str, status: u16, duration_secs: f64) {
        self.requests_total
            .with_label_values(&[method, path, &status.to_string()])
            .inc();
        self.request_duration
            .with_label_values(&[method, path])
            .observe(duration_secs);
    }

    pub fn record_tokens(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
        duration_secs: f64,
    ) {
        self.tokens_prompt_total.inc_by(prompt_tokens as u64);
        self.tokens_generated_total.inc_by(completion_tokens as u64);
        if duration_secs > 0.0 {
            let tps = completion_tokens as f64 / duration_secs;
            self.tokens_per_second.set(tps);
        }
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits_total.inc();
        self.update_cache_ratio();
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses_total.inc();
        self.update_cache_ratio();
    }

    fn update_cache_ratio(&self) {
        let hits = self.cache_hits_total.get() as f64;
        let misses = self.cache_misses_total.get() as f64;
        let total = hits + misses;
        if total > 0.0 {
            self.cache_hit_ratio.set(hits / total);
        }
    }

    pub fn record_error(&self, error_type: &str) {
        self.errors_total.with_label_values(&[error_type]).inc();
    }

    pub fn record_rate_limit(&self) {
        self.rate_limits_total.inc();
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new().expect("failed to create metrics registry")
    }
}

/// Handler for the `/metrics` endpoint.
pub async fn metrics_handler(State(state): State<AppState>) -> Response {
    let encoder = TextEncoder::new();
    let metric_families = state.metrics.registry.gather();
    let mut buffer = Vec::new();

    match encoder.encode(&metric_families, &mut buffer) {
        Ok(()) => (
            StatusCode::OK,
            [("Content-Type", encoder.format_type())],
            buffer,
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to encode metrics: {error}"),
        )
            .into_response(),
    }
}

/// Middleware that tracks request metrics.
pub async fn metrics_middleware(
    State(state): State<AppState>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let start = std::time::Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();

    state.metrics.requests_in_flight.inc();

    let response = next.run(request).await;
    let duration = start.elapsed();
    let status = response.status().as_u16();

    state.metrics.requests_in_flight.dec();
    state
        .metrics
        .record_request(method.as_str(), &path, status, duration.as_secs_f64());

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_registry_creates_successfully() {
        let metrics = MetricsRegistry::new().unwrap();
        assert_eq!(metrics.requests_in_flight.get(), 0);
    }

    #[test]
    fn request_metrics_increment() {
        let metrics = MetricsRegistry::new().unwrap();
        metrics.record_request("POST", "/v1/chat/completions", 200, 0.5);
        let count = metrics
            .requests_total
            .with_label_values(&["POST", "/v1/chat/completions", "200"])
            .get();
        assert!(count >= 1.0);
    }

    #[test]
    fn token_metrics_update() {
        let metrics = MetricsRegistry::new().unwrap();
        metrics.record_tokens(100, 50, 2.0);
        assert_eq!(metrics.tokens_prompt_total.get(), 100);
        assert_eq!(metrics.tokens_generated_total.get(), 50);
        assert_eq!(metrics.tokens_per_second.get(), 25.0);
    }

    #[test]
    fn cache_ratio_updates() {
        let metrics = MetricsRegistry::new().unwrap();
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        assert_eq!(metrics.cache_hit_ratio.get(), 0.5);
    }
}
