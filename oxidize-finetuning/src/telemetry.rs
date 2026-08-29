//! Training telemetry: metrics collection, CSV export, ASCII sparklines,
//! live progress reporting, and early-stopping logic.
//!
//! All implemented with the standard library only — no external TUI crates.

use std::io::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};


/// A snapshot of training state at a single optimizer step.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub step: usize,
    pub epoch: usize,
    pub loss: f32,
    pub lr: f32,
    pub tokens_per_sec: f32,
    pub grad_norm: f32,
    pub eval_loss: Option<f32>,
    /// Wall-clock milliseconds since the Unix epoch.
    pub timestamp_ms: u128,
}

impl TrainingMetrics {
    /// Convenience constructor that fills `timestamp_ms` automatically.
    pub fn new(
        step: usize,
        epoch: usize,
        loss: f32,
        lr: f32,
        tokens_per_sec: f32,
        grad_norm: f32,
        eval_loss: Option<f32>,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        Self {
            step,
            epoch,
            loss,
            lr,
            tokens_per_sec,
            grad_norm,
            eval_loss,
            timestamp_ms,
        }
    }
}


/// Ordered history of training metrics with analysis helpers.
#[derive(Debug, Default)]
pub struct MetricsLog {
    pub history: Vec<TrainingMetrics>,
}

impl MetricsLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new metrics snapshot.
    pub fn record(&mut self, m: TrainingMetrics) {
        self.history.push(m);
    }

    /// Mean loss over the last `n` recorded steps (or all steps if fewer).
    pub fn mean_loss_last_n(&self, n: usize) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let slice = if n == 0 || n >= self.history.len() {
            &self.history[..]
        } else {
            &self.history[self.history.len() - n..]
        };
        let sum: f32 = slice.iter().map(|m| m.loss).sum();
        sum / slice.len() as f32
    }

    /// Lowest training loss seen so far (f32::MAX if history is empty).
    pub fn best_loss(&self) -> f32 {
        self.history.iter().map(|m| m.loss).fold(f32::MAX, f32::min)
    }

    /// Serialise the full history to a CSV string.
    ///
    /// Columns: `timestamp_ms,epoch,step,loss,lr,tokens_per_sec,grad_norm,eval_loss`
    pub fn to_csv(&self) -> String {
        let mut out = String::with_capacity(self.history.len() * 80 + 128);
        out.push_str("timestamp_ms,epoch,step,loss,lr,tokens_per_sec,grad_norm,eval_loss\n");
        for m in &self.history {
            let eval = m.eval_loss.map(|v| format!("{v:.6}")).unwrap_or_default();
            out.push_str(&format!(
                "{},{},{},{:.6},{:.6e},{:.2},{:.6},{}\n",
                m.timestamp_ms, m.epoch, m.step, m.loss, m.lr, m.tokens_per_sec, m.grad_norm, eval,
            ));
        }
        out
    }

    /// Write the CSV to `path`, creating or overwriting the file.
    pub fn save_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        let csv = self.to_csv();
        let mut f = std::fs::File::create(path)?;
        f.write_all(csv.as_bytes())?;
        Ok(())
    }

    /// Render an ASCII sparkline of training loss over recorded steps.
    ///
    /// The plot is `width` columns wide and `height` rows tall (minimum 1×1).
    /// Each column represents one or more steps (bucketed by mean loss).
    /// Block characters `▁▂▃▄▅▆▇█` are used to fill cells.
    pub fn plot_ascii(&self, width: usize, height: usize) -> String {
        const BLOCKS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        let width = width.max(1);
        let height = height.max(1);

        if self.history.is_empty() {
            let blank_row = " ".repeat(width);
            return (0..height)
                .map(|_| blank_row.as_str())
                .collect::<Vec<_>>()
                .join("\n");
        }

        // Bucket the history into `width` columns.
        let n = self.history.len();
        let bucket_size = (n as f64 / width as f64).max(1.0);
        let mut buckets: Vec<f32> = (0..width)
            .map(|col| {
                let start = (col as f64 * bucket_size) as usize;
                let end = (((col + 1) as f64 * bucket_size) as usize).min(n);
                if start >= end {
                    return f32::NAN;
                }
                let vals = &self.history[start..end];
                vals.iter().map(|m| m.loss).sum::<f32>() / vals.len() as f32
            })
            .collect();

        // Replace NaN (empty buckets) with the last valid value or 0.
        let mut last_valid = 0.0_f32;
        for v in &mut buckets {
            if v.is_nan() {
                *v = last_valid;
            } else {
                last_valid = *v;
            }
        }

        let min_loss = buckets.iter().cloned().fold(f32::MAX, f32::min);
        let max_loss = buckets.iter().cloned().fold(f32::MIN, f32::max);
        let range = (max_loss - min_loss).max(1e-9);

        // For each cell row (top = high loss, bottom = low loss), decide which
        // block character fills the column by comparing the bucket value to the
        // y-range slice this row represents.
        let mut rows: Vec<String> = Vec::with_capacity(height);
        for row in 0..height {
            // Normalised threshold for this row (0 = bottom, 1 = top).
            let row_lo = (height - 1 - row) as f32 / height as f32;
            let row_hi = (height - row) as f32 / height as f32;

            let mut line = String::with_capacity(width);
            for &val in &buckets {
                let norm = (val - min_loss) / range; // 0..=1
                if norm < row_lo {
                    line.push(' ');
                } else if norm >= row_hi {
                    line.push('█');
                } else {
                    // Partial fill: map fractional position within row to a block.
                    let frac = (norm - row_lo) / (row_hi - row_lo);
                    let idx = ((frac * BLOCKS.len() as f32) as usize).min(BLOCKS.len() - 1);
                    line.push(BLOCKS[idx]);
                }
            }
            rows.push(line);
        }

        // Append a simple axis label.
        rows.push(format!(
            "loss {:.4}..{:.4}  steps {}",
            min_loss, max_loss, n
        ));
        rows.join("\n")
    }
}


/// Prints live training progress to stdout using only `print!` / `println!`.
#[derive(Debug)]
pub struct ProgressReporter {
    total_steps: usize,
    total_epochs: usize,
    bar_width: usize,
}

impl ProgressReporter {
    /// Create a reporter.  `bar_width` is the number of characters in the
    /// filled/empty progress bar (defaults internally to 20).
    pub fn new(total_steps: usize, total_epochs: usize) -> Self {
        Self {
            total_steps: total_steps.max(1),
            total_epochs: total_epochs.max(1),
            bar_width: 20,
        }
    }

    /// Override the width of the ASCII progress bar.
    pub fn with_bar_width(mut self, w: usize) -> Self {
        self.bar_width = w.max(4);
        self
    }

    /// Print a single-line progress update.
    ///
    /// Example output:
    /// ```text
    /// Epoch 1/3  Step  123/500  loss=1.2345  lr=1.00e-4  tok/s=1234  [████████░░] 45%
    /// ```
    pub fn report(&self, metrics: &TrainingMetrics) {
        let pct = (metrics.step as f64 / self.total_steps as f64).min(1.0);
        let filled = (pct * self.bar_width as f64).round() as usize;
        let empty = self.bar_width.saturating_sub(filled);
        let bar: String = "█".repeat(filled) + &"░".repeat(empty);

        let eval_str = metrics
            .eval_loss
            .map(|v| format!("  eval={v:.4}"))
            .unwrap_or_default();

        println!(
            "Epoch {}/{}  Step {:>5}/{}  loss={:.4}  lr={:.2e}  tok/s={:.0}  [{}] {:.0}%{}",
            metrics.epoch,
            self.total_epochs,
            metrics.step,
            self.total_steps,
            metrics.loss,
            metrics.lr,
            metrics.tokens_per_sec,
            bar,
            pct * 100.0,
            eval_str,
        );
    }

    /// Print a final summary after training is complete.
    pub fn summary(&self, log: &MetricsLog) {
        println!();
        println!("─── Training summary ─────────────────────────────");
        println!("  Total steps   : {}", log.history.len());
        println!("  Best loss     : {:.6}", log.best_loss());
        println!(
            "  Mean loss (all): {:.6}",
            log.mean_loss_last_n(log.history.len())
        );
        if let Some(last) = log.history.last() {
            println!("  Final lr      : {:.3e}", last.lr);
            println!("  Final tok/s   : {:.1}", last.tokens_per_sec);
        }
        println!("──────────────────────────────────────────────────");
    }
}


/// Monitors validation (or training) loss and signals when to stop training
/// early because no meaningful improvement has occurred for `patience` steps.
#[derive(Debug)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f32,
    best_loss: f32,
    no_improve_count: usize,
}

impl EarlyStopping {
    /// Create a new monitor.
    ///
    /// * `patience`  — how many consecutive non-improving calls before stopping.
    /// * `min_delta` — the minimum absolute improvement required to reset the
    ///                 patience counter.
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta: min_delta.max(0.0),
            best_loss: f32::MAX,
            no_improve_count: 0,
        }
    }

    /// Feed the latest loss.  Returns `true` when training should stop.
    ///
    /// Improvement is defined as `loss < best_loss - min_delta`.
    pub fn should_stop(&mut self, loss: f32) -> bool {
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.no_improve_count = 0;
            false
        } else {
            self.no_improve_count += 1;
            self.no_improve_count >= self.patience
        }
    }

    /// The lowest loss observed so far.
    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }

    /// How many consecutive non-improving steps have been seen.
    pub fn no_improve_count(&self) -> usize {
        self.no_improve_count
    }

    /// Reset state (useful when switching from training to eval loss tracking).
    pub fn reset(&mut self) {
        self.best_loss = f32::MAX;
        self.no_improve_count = 0;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(step: usize, loss: f32) -> TrainingMetrics {
        TrainingMetrics {
            step,
            epoch: 1,
            loss,
            lr: 1e-4,
            tokens_per_sec: 500.0,
            grad_norm: 1.0,
            eval_loss: None,
            timestamp_ms: 0,
        }
    }

    #[test]
    fn mean_loss_last_n_basic() {
        let mut log = MetricsLog::new();
        for i in 1..=5 {
            log.record(make_metrics(i, i as f32));
        }
        // Last 2: loss 4 + 5 = 9 / 2 = 4.5
        let mean = log.mean_loss_last_n(2);
        assert!((mean - 4.5).abs() < 1e-5, "got {mean}");
    }

    #[test]
    fn best_loss() {
        let mut log = MetricsLog::new();
        for loss in [3.0_f32, 1.5, 2.5, 0.8, 1.2] {
            log.record(make_metrics(1, loss));
        }
        assert!((log.best_loss() - 0.8).abs() < 1e-5);
    }

    #[test]
    fn csv_round_trip() {
        let mut log = MetricsLog::new();
        log.record(TrainingMetrics::new(1, 1, 0.5, 1e-4, 300.0, 0.9, Some(0.6)));
        let csv = log.to_csv();
        assert!(csv.starts_with("timestamp_ms,epoch,step"));
        assert!(csv.contains("0.500000"));
    }

    #[test]
    fn plot_ascii_dimensions() {
        let mut log = MetricsLog::new();
        for i in 0..40 {
            log.record(make_metrics(i, (i as f32) * 0.1));
        }
        let plot = log.plot_ascii(20, 4);
        // Should have `height + 1` lines (extra axis line).
        let lines: Vec<&str> = plot.lines().collect();
        assert_eq!(lines.len(), 5, "expected 5 lines, got {}", lines.len());
        // Each data row should be exactly `width` chars wide.
        for row in &lines[..4] {
            assert_eq!(row.chars().count(), 20, "unexpected row width: {row:?}");
        }
    }

    #[test]
    fn early_stopping_basic() {
        let mut es = EarlyStopping::new(3, 0.01);
        assert!(!es.should_stop(1.0));
        assert!(!es.should_stop(1.0)); // no improvement
        assert!(!es.should_stop(1.0)); // no improvement (count=2, patience=3)
        assert!(es.should_stop(1.0)); // count hits patience
    }

    #[test]
    fn early_stopping_improves_resets() {
        let mut es = EarlyStopping::new(2, 0.0);
        assert!(!es.should_stop(2.0));
        assert!(!es.should_stop(1.9)); // improved
        assert!(!es.should_stop(1.9)); // no improvement, count=1
        assert!(!es.should_stop(1.8)); // improved again, count reset
        assert_eq!(es.no_improve_count(), 0);
    }

    #[test]
    fn progress_reporter_smoke() {
        // Just verify it doesn't panic.
        let r = ProgressReporter::new(500, 3);
        let m = TrainingMetrics::new(123, 1, 1.234, 1e-4, 1234.0, 0.95, None);
        r.report(&m);
        let mut log = MetricsLog::new();
        log.record(m);
        r.summary(&log);
    }
}
