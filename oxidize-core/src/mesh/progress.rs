//! Distributed progress indicators for model loading across the mesh.
//!
//! Each worker node reports per-shard progress via `LOCAL_EVENTS`.
//! The master aggregates these reports into a cluster-wide progress bar.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Progress report sent by a single worker node while loading its shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadProgressReport {
    pub peer_id: String,
    /// Human-readable stage (e.g. "mapping", "downloading", "quantizing").
    pub stage: String,
    /// Percent complete for this shard (0–100).
    pub percent: u8,
    /// Layers loaded so far.
    pub layers_loaded: usize,
    /// Total layers in this shard.
    pub total_layers: usize,
    /// Bytes downloaded / processed.
    pub bytes_processed: u64,
    /// Total bytes expected for this shard.
    pub total_bytes: u64,
}

/// Aggregated view of loading progress across the whole cluster.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AggregatedProgress {
    /// Latest report per peer.
    pub reports: HashMap<String, LoadProgressReport>,
    /// Total number of workers expected to report.
    pub total_workers: usize,
}

impl AggregatedProgress {
    /// Number of peers that have reported any progress.
    pub fn ready_workers(&self) -> usize {
        self.reports.len()
    }

    /// True when every expected worker has reached 100 %.
    pub fn is_complete(&self) -> bool {
        if self.total_workers == 0 {
            return false;
        }
        self.reports.len() >= self.total_workers && self.reports.values().all(|r| r.percent >= 100)
    }

    /// Mean percent across all known reports.
    pub fn mean_percent(&self) -> u8 {
        if self.reports.is_empty() {
            return 0;
        }
        let sum: u32 = self.reports.values().map(|r| r.percent as u32).sum();
        (sum / self.reports.len() as u32).min(100) as u8
    }
}

/// Merge a fresh worker report into the aggregated state.
pub fn aggregate_progress(agg: &mut AggregatedProgress, report: LoadProgressReport) {
    agg.reports.insert(report.peer_id.clone(), report);
}

/// Render a simple ASCII progress bar for the cluster.
///
/// Returns a string like `[###--] 3/5 nodes ready  (mean 60%)`.
pub fn render_cluster_progress_bar(agg: &AggregatedProgress) -> String {
    let ready = agg.ready_workers();
    let total = agg.total_workers.max(1);
    let bar_len = 10usize;
    let filled = (ready * bar_len) / total;
    let empty = bar_len.saturating_sub(filled);
    let bar = format!("[{}{}]", "#".repeat(filled), "-".repeat(empty));
    format!(
        "{bar} {ready}/{total} nodes ready  (mean {}%)",
        agg.mean_percent()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_report(peer_id: &str, percent: u8) -> LoadProgressReport {
        LoadProgressReport {
            peer_id: peer_id.to_string(),
            stage: "loading".to_string(),
            percent,
            layers_loaded: 0,
            total_layers: 4,
            bytes_processed: percent as u64 * 1024,
            total_bytes: 100 * 1024,
        }
    }

    #[test]
    fn aggregate_tracks_latest_report_per_peer() {
        let mut agg = AggregatedProgress {
            total_workers: 2,
            ..Default::default()
        };
        aggregate_progress(&mut agg, dummy_report("a", 50));
        assert_eq!(agg.ready_workers(), 1);
        assert_eq!(agg.mean_percent(), 50);

        aggregate_progress(&mut agg, dummy_report("a", 75));
        assert_eq!(agg.ready_workers(), 1);
        assert_eq!(agg.mean_percent(), 75);
    }

    #[test]
    fn aggregate_completes_when_all_at_100() {
        let mut agg = AggregatedProgress {
            total_workers: 2,
            ..Default::default()
        };
        aggregate_progress(&mut agg, dummy_report("a", 100));
        assert!(!agg.is_complete());
        aggregate_progress(&mut agg, dummy_report("b", 100));
        assert!(agg.is_complete());
    }

    #[test]
    fn aggregate_not_complete_with_zero_workers() {
        let agg = AggregatedProgress::default();
        assert!(!agg.is_complete());
    }

    #[test]
    fn render_progress_bar() {
        let mut agg = AggregatedProgress {
            total_workers: 5,
            ..Default::default()
        };
        aggregate_progress(&mut agg, dummy_report("a", 50));
        aggregate_progress(&mut agg, dummy_report("b", 100));
        aggregate_progress(&mut agg, dummy_report("c", 30));
        let bar = render_cluster_progress_bar(&agg);
        assert!(bar.contains("[######----]"), "actual bar: {bar}");
        assert!(bar.contains("3/5 nodes ready"));
        assert!(bar.contains("(mean 60%)"));
    }

    #[test]
    fn load_progress_report_serializes_roundtrip() {
        let report = LoadProgressReport {
            peer_id: "p".into(),
            stage: "quantizing".into(),
            percent: 42,
            layers_loaded: 2,
            total_layers: 8,
            bytes_processed: 1024,
            total_bytes: 4096,
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: LoadProgressReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, back);
    }
}
