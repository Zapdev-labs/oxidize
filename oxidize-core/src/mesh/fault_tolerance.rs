//! Fault tolerance and deadlock prevention for the distributed mesh.
//!
//! Provides `eval_with_timeout` — a wrapper that kills hung distributed
//! operations after a configurable timeout — and `RunnerStatus` events
//! that the master uses to trigger recovery (re-shard / shutdown).

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::time::Duration;
use tokio::time::timeout;

/// Default timeout for distributed collectives (all_sum, all_gather, …).
pub const DEFAULT_COLLECTIVE_TIMEOUT: Duration = Duration::from_secs(60);

/// Status of a model-shard runner on a single mesh node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunnerStatus {
    /// Runner is healthy and processing inference.
    Healthy,
    /// Runner failed (e.g. hung collective, OOM, panic).
    RunnerFailed { reason: String },
    /// Runner is shutting down (cleanup in progress).
    ShuttingDown,
    /// Runner has finished cleanup and exited.
    Offline,
}

/// Event emitted when a runner's status changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunnerStatusUpdated {
    pub peer_id: String,
    pub status: RunnerStatus,
    pub clock: u64,
}

/// Event emitted by the master ordering a worker to shut down its shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShutdownTask {
    pub instance_id: String,
    pub reason: String,
    pub clock: u64,
}

/// Result of a timed distributed evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimedResult<T> {
    /// Operation completed successfully within the deadline.
    Ok(T),
    /// Operation was killed because it exceeded the timeout.
    TimedOut,
    /// An error occurred during execution.
    Err(String),
}

/// Evaluate an async future with a hard timeout.
///
/// If the future does not complete within `deadline`, it is cancelled and
/// `TimedResult::TimedOut` is returned.  This prevents deadlocks when a
/// ring neighbour becomes unreachable mid-collective.
///
/// # Example
/// ```ignore
/// let result = eval_with_timeout(
///     ring.all_sum(&mut data),
///     DEFAULT_COLLECTIVE_TIMEOUT,
/// ).await;
/// ```
pub async fn eval_with_timeout<F, T>(
    fut: F,
    deadline: Duration,
) -> TimedResult<T>
where
    F: Future<Output = Result<T, crate::mesh::ring::RingError>>,
{
    match timeout(deadline, fut).await {
        Ok(Ok(value)) => TimedResult::Ok(value),
        Ok(Err(e)) => TimedResult::Err(e.to_string()),
        Err(_) => TimedResult::TimedOut,
    }
}

/// Convenience wrapper that also emits a [`RunnerStatusUpdated`] when
/// the operation times out.
pub async fn eval_with_timeout_and_notify<F, T>(
    fut: F,
    deadline: Duration,
    peer_id: &str,
    clock: u64,
    on_status: impl FnOnce(RunnerStatusUpdated),
) -> TimedResult<T>
where
    F: Future<Output = Result<T, crate::mesh::ring::RingError>>,
{
    let result = eval_with_timeout(fut, deadline).await;
    if matches!(result, TimedResult::TimedOut) {
        on_status(RunnerStatusUpdated {
            peer_id: peer_id.to_string(),
            status: RunnerStatus::RunnerFailed {
                reason: format!("collective timed out after {}s", deadline.as_secs()),
            },
            clock,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn eval_with_timeout_succeeds_quickly() {
        let fut = async { Ok::<_, crate::mesh::ring::RingError>(42) };
        let result = eval_with_timeout(fut, Duration::from_secs(5)).await;
        assert_eq!(result, TimedResult::Ok(42));
    }

    #[tokio::test]
    async fn eval_with_timeout_kills_slow_future() {
        let fut = async {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok::<_, crate::mesh::ring::RingError>(())
        };
        let result = eval_with_timeout(fut, Duration::from_millis(50)).await;
        assert_eq!(result, TimedResult::TimedOut);
    }

    #[tokio::test]
    async fn eval_with_timeout_propagates_error() {
        let fut = async { Err::<(), _>(crate::mesh::ring::RingError::NotConnected) };
        let result = eval_with_timeout(fut, Duration::from_secs(5)).await;
        assert_eq!(result, TimedResult::Err("ring transport not connected".to_string()));
    }

    #[tokio::test]
    async fn eval_with_timeout_notifies_on_timeout() {
        let mut received = None;
        let fut = async {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok::<_, crate::mesh::ring::RingError>(())
        };
        let result = eval_with_timeout_and_notify(
            fut,
            Duration::from_millis(50),
            "peer-a",
            7,
            |ev| received = Some(ev),
        )
        .await;
        assert_eq!(result, TimedResult::TimedOut);
        let ev = received.unwrap();
        assert_eq!(ev.peer_id, "peer-a");
        assert_eq!(ev.clock, 7);
        assert!(matches!(ev.status, RunnerStatus::RunnerFailed { .. }));
    }

    #[test]
    fn runner_status_serializes_roundtrip() {
        let statuses = vec![
            RunnerStatus::Healthy,
            RunnerStatus::RunnerFailed { reason: "oom".into() },
            RunnerStatus::ShuttingDown,
            RunnerStatus::Offline,
        ];
        for s in statuses {
            let json = serde_json::to_string(&s).unwrap();
            let back: RunnerStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn shutdown_task_serializes_roundtrip() {
        let task = ShutdownTask {
            instance_id: "inst-1".into(),
            reason: "node lost".into(),
            clock: 3,
        };
        let json = serde_json::to_string(&task).unwrap();
        let back: ShutdownTask = serde_json::from_str(&json).unwrap();
        assert_eq!(task, back);
    }
}
