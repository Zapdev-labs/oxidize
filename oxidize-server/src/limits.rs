//! Rate limiting + continuous-batching wait queue.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::{Instant as TokioInstant, sleep_until};

use crate::app::AppState;

#[derive(Debug, Clone, Copy)]
pub struct RequestLimitConfig {
    pub requests_per_second: usize,
    pub max_in_flight: usize,
    pub max_queue: usize,
}

impl Default for RequestLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 64,
            max_in_flight: 8,
            max_queue: 128,
        }
    }
}

pub struct RequestLimiter {
    config: RequestLimitConfig,
    queue_slots: Arc<Semaphore>,
    active_slots: Arc<Semaphore>,
    accepted_at: Mutex<VecDeque<Instant>>,
}

pub struct RequestPermit {
    _queue: OwnedSemaphorePermit,
    _active: OwnedSemaphorePermit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestLimitError {
    RateLimited,
    QueueFull,
}

impl RequestLimiter {
    pub fn new(config: RequestLimitConfig) -> Self {
        let total_slots = config.max_in_flight.saturating_add(config.max_queue).max(1);
        let active_slots = config.max_in_flight.max(1);
        Self {
            config,
            queue_slots: Arc::new(Semaphore::new(total_slots)),
            active_slots: Arc::new(Semaphore::new(active_slots)),
            accepted_at: Mutex::new(VecDeque::new()),
        }
    }

    pub async fn try_acquire(&self) -> Result<RequestPermit, RequestLimitError> {
        let queue = self
            .queue_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| RequestLimitError::QueueFull)?;
        if !self.try_accept_rate().await {
            return Err(RequestLimitError::RateLimited);
        }
        let active = self
            .active_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| RequestLimitError::QueueFull)?;
        Ok(RequestPermit {
            _queue: queue,
            _active: active,
        })
    }

    async fn try_accept_rate(&self) -> bool {
        if self.config.requests_per_second == 0 {
            return false;
        }
        let now = Instant::now();
        let mut accepted = self.accepted_at.lock().await;
        let cutoff = now.checked_sub(Duration::from_secs(1)).unwrap_or(now);
        while accepted.front().is_some_and(|instant| *instant <= cutoff) {
            accepted.pop_front();
        }
        if accepted.len() >= self.config.requests_per_second {
            return false;
        }
        accepted.push_back(now);
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ContinuousBatchConfig {
    pub max_batch_size: usize,
    pub max_wait: Duration,
}

impl Default for ContinuousBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_wait: Duration::from_millis(5),
        }
    }
}

pub struct ContinuousBatcher {
    config: ContinuousBatchConfig,
    state: Mutex<BatchState>,
}

struct BatchState {
    open: Option<OpenBatch>,
}

struct OpenBatch {
    size: usize,
    deadline: TokioInstant,
    notify: Arc<Notify>,
}

impl Default for ContinuousBatcher {
    fn default() -> Self {
        Self::new(ContinuousBatchConfig::default())
    }
}

impl ContinuousBatcher {
    pub fn new(config: ContinuousBatchConfig) -> Self {
        Self {
            config,
            state: Mutex::new(BatchState { open: None }),
        }
    }

    pub async fn wait_turn(&self) {
        let (deadline, notify) = {
            let now = TokioInstant::now();
            let mut state = self.state.lock().await;
            let max_batch_size = self.config.max_batch_size.max(1);

            if state
                .open
                .as_ref()
                .is_some_and(|batch| batch.deadline <= now)
            {
                state.open = None;
            }

            if state.open.is_none() {
                state.open = Some(OpenBatch {
                    size: 0,
                    deadline: now + self.config.max_wait,
                    notify: Arc::new(Notify::new()),
                });
            }

            let batch = state.open.as_mut().expect("batch should exist");
            batch.size += 1;
            let is_full = batch.size >= max_batch_size;
            let deadline = batch.deadline;
            let notify = Arc::clone(&batch.notify);

            if is_full {
                state.open = None;
                notify.notify_waiters();
                return;
            }

            (deadline, notify)
        };

        tokio::select! {
            _ = sleep_until(deadline) => {}
            _ = notify.notified() => {}
        }
    }
}

pub async fn enforce_request_limits(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if is_generation_route(request.uri().path()) {
        state.batcher.wait_turn().await;
    }
    match state.limiter.try_acquire().await {
        Ok(_permit) => next.run(request).await,
        Err(RequestLimitError::RateLimited) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "rate limit exceeded"})),
        )
            .into_response(),
        Err(RequestLimitError::QueueFull) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "request queue full"})),
        )
            .into_response(),
    }
}

fn is_generation_route(path: &str) -> bool {
    matches!(path, "/v1/chat/completions" | "/v1/completions")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn limiter_queues_one_request_and_rejects_when_queue_is_full() {
        let limiter = Arc::new(RequestLimiter::new(RequestLimitConfig {
            requests_per_second: 100,
            max_in_flight: 1,
            max_queue: 1,
        }));

        let first = limiter
            .try_acquire()
            .await
            .expect("first request should acquire active slot");

        let queued_limiter = Arc::clone(&limiter);
        let queued_task = tokio::spawn(async move { queued_limiter.try_acquire().await });
        sleep(Duration::from_millis(20)).await;

        let rejected = limiter.try_acquire().await;
        assert!(matches!(rejected, Err(RequestLimitError::QueueFull)));

        drop(first);
        let queued = queued_task
            .await
            .expect("queued task should complete")
            .expect("queued request should eventually acquire");
        drop(queued);
    }

    #[tokio::test]
    async fn limiter_allows_concurrent_in_flight_requests_up_to_limit() {
        let limiter = Arc::new(RequestLimiter::new(RequestLimitConfig {
            requests_per_second: 100,
            max_in_flight: 2,
            max_queue: 0,
        }));

        let first = limiter
            .try_acquire()
            .await
            .expect("first request should acquire active slot");
        let second = limiter
            .try_acquire()
            .await
            .expect("second request should acquire active slot");

        let third = limiter.try_acquire().await;
        assert!(matches!(third, Err(RequestLimitError::QueueFull)));

        drop(first);
        drop(second);
    }

    #[tokio::test]
    async fn queue_full_does_not_consume_rate_limit_capacity() {
        let limiter = Arc::new(RequestLimiter::new(RequestLimitConfig {
            requests_per_second: 2,
            max_in_flight: 1,
            max_queue: 0,
        }));

        let held = limiter
            .try_acquire()
            .await
            .expect("first request should acquire active slot");

        let rejected_while_full = limiter.try_acquire().await;
        assert!(matches!(
            rejected_while_full,
            Err(RequestLimitError::QueueFull)
        ));
        drop(held);

        let second = limiter
            .try_acquire()
            .await
            .expect("queue-full rejection should not consume rate budget");
        drop(second);

        let third = limiter.try_acquire().await;
        assert!(matches!(third, Err(RequestLimitError::RateLimited)));
    }

    #[tokio::test]
    async fn continuous_batcher_waits_for_batch_window() {
        let batcher = Arc::new(ContinuousBatcher::new(ContinuousBatchConfig {
            max_batch_size: 4,
            max_wait: Duration::from_millis(40),
        }));
        let started = TokioInstant::now();

        let first = {
            let batcher = Arc::clone(&batcher);
            tokio::spawn(async move {
                batcher.wait_turn().await;
                TokioInstant::now()
            })
        };

        sleep(Duration::from_millis(5)).await;

        let second = {
            let batcher = Arc::clone(&batcher);
            tokio::spawn(async move {
                batcher.wait_turn().await;
                TokioInstant::now()
            })
        };

        let first_done = first.await.expect("first task should complete");
        let second_done = second.await.expect("second task should complete");
        let first_elapsed = first_done.duration_since(started);
        let second_elapsed = second_done.duration_since(started);
        assert!(first_elapsed >= Duration::from_millis(30));
        assert!(second_elapsed >= Duration::from_millis(30));
    }

    #[tokio::test]
    async fn continuous_batcher_releases_early_when_batch_is_full() {
        let batcher = Arc::new(ContinuousBatcher::new(ContinuousBatchConfig {
            max_batch_size: 2,
            max_wait: Duration::from_millis(200),
        }));
        let started = TokioInstant::now();

        let first = {
            let batcher = Arc::clone(&batcher);
            tokio::spawn(async move {
                batcher.wait_turn().await;
                TokioInstant::now()
            })
        };

        sleep(Duration::from_millis(20)).await;
        batcher.wait_turn().await;

        let first_done = first.await.expect("first task should complete");
        let elapsed = first_done.duration_since(started);
        assert!(elapsed < Duration::from_millis(150));
    }
}
