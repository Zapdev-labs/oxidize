//! CPU-friendly frame sampling strategies.
//!
//! All strategies take the total number of frames available and the desired
//! number of output frames and return a sorted, deduplicated `Vec<usize>` of
//! frame indices.

use super::config::FrameSamplingStrategy;
use super::error::VideoError;

/// Sample frame indices from `[0, total_frames)` according to `strategy`.
///
/// The returned vector is sorted ascending and contains at most
/// `target_frames` indices, all strictly less than `total_frames`.
///
/// # Errors
///
/// Returns `VideoError::FrameCountOutOfRange` if `total_frames` is 0 or if
/// `target_frames` is 0. Returns `VideoError::EmptySample` only if the
/// resulting sample is empty.
pub fn sample_indices(
    total_frames: usize,
    target_frames: usize,
    strategy: FrameSamplingStrategy,
) -> Result<Vec<usize>, VideoError> {
    if total_frames == 0 {
        return Err(VideoError::FrameCountOutOfRange {
            requested: 0,
            min: 1,
            max: usize::MAX,
        });
    }
    if target_frames == 0 {
        return Err(VideoError::FrameCountOutOfRange {
            requested: 0,
            min: 1,
            max: usize::MAX,
        });
    }

    let indices = match strategy {
        FrameSamplingStrategy::Uniform => uniform(total_frames, target_frames),
        FrameSamplingStrategy::Dense { stride } => {
            dense(total_frames, target_frames, stride.max(1))
        }
        // Adaptive needs per-frame metadata; from a sampler that only knows
        // counts we fall back to uniform. The adaptive code path lives in
        // `sample_indices_adaptive` below.
        FrameSamplingStrategy::Adaptive => uniform(total_frames, target_frames),
    };

    if indices.is_empty() {
        return Err(VideoError::EmptySample);
    }
    Ok(indices)
}

/// Sample `target_frames` indices using a cheap keyframe heuristic based on
/// per-frame luma histograms. The first and last frames are always kept; the
/// remaining slots are filled by frames whose 16-bin luma histogram is
/// furthest (L1) from any already-chosen frame.
///
/// `luma_hists` is laid out as `[frame_idx * 16 .. (frame_idx + 1) * 16]` with
/// values in `[0, 1]`. Pass empty slice to fall back to uniform sampling.
pub fn sample_indices_adaptive(
    total_frames: usize,
    target_frames: usize,
    luma_hists: &[f32],
) -> Result<Vec<usize>, VideoError> {
    if total_frames == 0 || target_frames == 0 {
        return Err(VideoError::FrameCountOutOfRange {
            requested: target_frames,
            min: 1,
            max: usize::MAX,
        });
    }
    if luma_hists.is_empty() || luma_hists.len() < total_frames * 16 {
        return sample_indices(total_frames, target_frames, FrameSamplingStrategy::Adaptive);
    }
    if total_frames <= target_frames {
        return Ok((0..total_frames).collect());
    }

    let mut chosen: Vec<usize> = Vec::with_capacity(target_frames);
    chosen.push(0);
    chosen.push(total_frames - 1);

    while chosen.len() < target_frames {
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for candidate in 0..total_frames {
            if chosen.contains(&candidate) {
                continue;
            }
            let cand_hist = &luma_hists[candidate * 16..(candidate + 1) * 16];
            let score = min_l1_to_set(cand_hist, &chosen, luma_hists);
            if score > best_score {
                best_score = score;
                best_idx = candidate;
            }
        }
        if best_score <= f32::NEG_INFINITY {
            // No more distinct frames — fall back to filling the rest
            // uniformly.
            let remaining = target_frames - chosen.len();
            let mut extra = uniform_avoiding(total_frames, remaining, &chosen);
            chosen.append(&mut extra);
            break;
        }
        chosen.push(best_idx);
    }
    chosen.sort_unstable();
    Ok(chosen)
}

fn min_l1_to_set(cand_hist: &[f32], chosen: &[usize], all_hists: &[f32]) -> f32 {
    let mut min_score = f32::INFINITY;
    for &idx in chosen {
        let hist = &all_hists[idx * 16..(idx + 1) * 16];
        let mut sum = 0.0_f32;
        for (a, b) in cand_hist.iter().zip(hist.iter()) {
            sum += (a - b).abs();
        }
        if sum < min_score {
            min_score = sum;
        }
    }
    if !min_score.is_finite() {
        f32::NEG_INFINITY
    } else {
        min_score
    }
}

/// Pick `n` indices evenly spaced across `[0, total)`. Always includes
/// the first and last index when `n > 1`.
fn uniform(total: usize, n: usize) -> Vec<usize> {
    if total <= n {
        return (0..total).collect();
    }
    if n == 1 {
        return vec![total / 2];
    }
    let step = (total - 1) as f64 / (n - 1) as f64;
    (0..n).map(|i| (i as f64 * step).round() as usize).collect()
}

/// Pick `n` indices by stepping through the range with a fixed `stride`.
fn dense(total: usize, n: usize, stride: usize) -> Vec<usize> {
    if total <= n {
        return (0..total).collect();
    }
    let mut out: Vec<usize> = (0..total).step_by(stride).take(n).collect();
    // Always include the last frame so we don't lose the tail.
    if let Some(last) = out.last().copied()
        && last != total - 1
    {
        out.pop();
        out.push(total - 1);
    }
    out
}

/// Like [`uniform`] but avoids any index already in `skip`.
fn uniform_avoiding(total: usize, n: usize, skip: &[usize]) -> Vec<usize> {
    let full = uniform(total, n + skip.len());
    let mut out = Vec::with_capacity(n);
    for idx in full {
        if !skip.contains(&idx) {
            out.push(idx);
            if out.len() == n {
                break;
            }
        }
    }
    out
}

/// Compute a 16-bin luma histogram for an RGB frame. Values are in `[0, 1]`.
///
/// The bin index is `floor(0.299 * R + 0.587 * G + 0.114 * B) / 16` clamped
/// to `[0, 15]`.
pub fn luma_histogram_rgb(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    let mut bins = [0.0_f32; 16];
    if width == 0 || height == 0 {
        return bins.to_vec();
    }
    let mut total = 0.0_f64;
    for chunk in rgb.chunks_exact(3) {
        let r = chunk[0] as f64 / 255.0;
        let g = chunk[1] as f64 / 255.0;
        let b = chunk[2] as f64 / 255.0;
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        let bin = (luma * 16.0).floor().clamp(0.0, 15.0) as usize;
        bins[bin] += 1.0;
        total += 1.0;
    }
    if total > 0.0 {
        let inv = 1.0 / total as f32;
        for bin in &mut bins {
            *bin *= inv;
        }
    }
    bins.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_includes_endpoints_and_matches_count() {
        let idx = sample_indices(100, 8, FrameSamplingStrategy::Uniform).unwrap();
        assert_eq!(idx.len(), 8);
        assert_eq!(idx[0], 0);
        assert_eq!(*idx.last().unwrap(), 99);
        // strictly non-decreasing
        for w in idx.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn uniform_returns_all_when_target_exceeds_total() {
        let idx = sample_indices(4, 16, FrameSamplingStrategy::Uniform).unwrap();
        assert_eq!(idx, vec![0, 1, 2, 3]);
    }

    #[test]
    fn dense_stride_picks_evenly_and_keeps_last() {
        let idx = sample_indices(10, 4, FrameSamplingStrategy::Dense { stride: 2 }).unwrap();
        assert_eq!(idx.len(), 4);
        assert_eq!(*idx.last().unwrap(), 9);
    }

    #[test]
    fn empty_total_is_error() {
        let err = sample_indices(0, 4, FrameSamplingStrategy::Uniform).unwrap_err();
        assert!(matches!(err, VideoError::FrameCountOutOfRange { .. }));
    }

    #[test]
    fn adaptive_falls_back_to_uniform_without_histograms() {
        let idx = sample_indices_adaptive(50, 6, &[]).unwrap();
        assert_eq!(idx.len(), 6);
    }

    #[test]
    fn adaptive_picks_distinct_frames() {
        // Two visually different halves.
        let mut hists = vec![0.0_f32; 50 * 16];
        for frame in 0..25 {
            hists[frame * 16] = 1.0; // bin 0
        }
        for frame in 25..50 {
            hists[frame * 16 + 15] = 1.0; // bin 15
        }
        let idx = sample_indices_adaptive(50, 4, &hists).unwrap();
        assert_eq!(idx.len(), 4);
        // Should include at least one frame from each half.
        let first_half = idx.iter().any(|&i| i < 25);
        let second_half = idx.iter().any(|&i| i >= 25);
        assert!(first_half && second_half);
    }

    #[test]
    fn luma_histogram_is_normalized() {
        let frame = vec![0u8; 4 * 4 * 3]; // all black
        let h = luma_histogram_rgb(&frame, 4, 4);
        let sum: f32 = h.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    }
}
