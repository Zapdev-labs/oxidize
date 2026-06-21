use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use super::frames::clip_to_tensor;
use super::manifest::{VideoSample, build_manifest};
use super::{FrameConfig, LabelTask, VideoError, VideoTrainingConfig};

/// A fully materialized set of clips: patch tensors + integer labels.
#[derive(Debug, Clone)]
pub struct VideoDataset {
    data: Vec<f32>,
    labels: Vec<usize>,
    class_names: Vec<String>,
    frames: FrameConfig,
    samples: Vec<VideoSample>,
}

impl VideoDataset {
    pub fn from_raw(
        data: Vec<f32>,
        labels: Vec<usize>,
        class_names: Vec<String>,
        frames: FrameConfig,
    ) -> Self {
        Self {
            data,
            labels,
            class_names,
            frames,
            samples: Vec::new(),
        }
    }

    pub fn num_clips(&self) -> usize {
        self.labels.len()
    }

    pub fn classes(&self) -> usize {
        self.class_names.len()
    }

    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }

    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    pub fn frames(&self) -> FrameConfig {
        self.frames
    }

    pub fn samples(&self) -> &[VideoSample] {
        &self.samples
    }

    pub fn clip_floats(&self) -> usize {
        self.frames.tokens_per_clip() * self.frames.patch_dim()
    }

    pub fn clip(&self, index: usize) -> &[f32] {
        let span = self.clip_floats();
        &self.data[index * span..(index + 1) * span]
    }

    /// Count of clips per class, indexed by class id.
    pub fn class_histogram(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.classes()];
        for &label in &self.labels {
            counts[label] += 1;
        }
        counts
    }
}

/// Build a `VideoDataset` from a tt-downloader-style directory: discover clips,
/// extract/cache frames in parallel, then derive labels for the chosen task.
pub fn load_dataset(config: &VideoTrainingConfig) -> Result<VideoDataset, VideoError> {
    config.validate()?;

    let mut samples = build_manifest(&config.data_root)?;
    filter_samples(&mut samples, &config.exclude, config.merge_aliases);
    if let Some(cap) = config.max_videos {
        let mut rng = StdRng::seed_from_u64(config.seed);
        samples.shuffle(&mut rng);
        samples.truncate(cap);
        samples.sort_by(|a, b| a.path.cmp(&b.path));
    }

    let cache_dir = if config.cache_dir.as_os_str().is_empty() {
        config.data_root.join(".oxidize-frames")
    } else {
        config.cache_dir.clone()
    };
    std::fs::create_dir_all(&cache_dir).map_err(|source| VideoError::Io {
        path: cache_dir.clone(),
        source,
    })?;

    let total = samples.len();
    let frames_cfg = config.frames.validate()?;
    eprintln!(
        "oxidize-train video: extracting frames for {total} clips ({} frames @ {} each)…",
        frames_cfg.num_frames,
        frames_cfg.aspect_label()
    );

    let done = AtomicUsize::new(0);
    let cache_ref = cache_dir.as_path();
    let extracted: Vec<Option<Vec<f32>>> = samples
        .par_iter()
        .map(|sample| {
            let result = clip_to_tensor(sample, frames_cfg, cache_ref);
            let n = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if n.is_multiple_of(100) || n == total {
                eprintln!("  …{n}/{total} clips ready");
            }
            match result {
                Ok(tensor) => Some(tensor),
                Err(err) => {
                    eprintln!("  skip {}: {err}", sample.path.display());
                    None
                }
            }
        })
        .collect();

    let span = frames_cfg.tokens_per_clip() * frames_cfg.patch_dim();
    let mut kept_tensors: Vec<Vec<f32>> = Vec::with_capacity(total);
    let mut kept_samples = Vec::with_capacity(total);
    for (sample, tensor) in samples.into_iter().zip(extracted) {
        if let Some(tensor) = tensor
            && tensor.len() == span
        {
            kept_tensors.push(tensor);
            kept_samples.push(sample);
        }
    }

    if kept_samples.is_empty() {
        return Err(VideoError::NoSamples(config.data_root.clone()));
    }

    let (labels, class_names) = assign_labels(&kept_samples, config.task, config.buckets);
    if class_names.len() < 2 {
        return Err(VideoError::DegenerateLabels {
            classes: class_names.len(),
        });
    }

    let selection = if config.balance {
        balanced_indices(&labels, class_names.len(), config.seed)
    } else {
        (0..labels.len()).collect()
    };

    let mut data = Vec::with_capacity(selection.len() * span);
    let mut final_labels = Vec::with_capacity(selection.len());
    let mut final_samples = Vec::with_capacity(selection.len());
    for &index in &selection {
        data.extend_from_slice(&kept_tensors[index]);
        final_labels.push(labels[index]);
        final_samples.push(kept_samples[index].clone());
    }

    eprintln!(
        "oxidize-train video: {} clips, {} classes ({}) for task '{}'{}",
        final_samples.len(),
        class_names.len(),
        class_names.join(", "),
        config.task.as_str(),
        if config.balance { " [balanced]" } else { "" }
    );

    Ok(VideoDataset {
        data,
        labels: final_labels,
        class_names,
        frames: frames_cfg,
        samples: final_samples,
    })
}

/// Pick indices so each class contributes the same number of clips (the size
/// of the smallest class), selected deterministically from a seeded shuffle.
fn balanced_indices(labels: &[usize], classes: usize, seed: u64) -> Vec<usize> {
    let mut per_class: Vec<Vec<usize>> = vec![Vec::new(); classes];
    for (index, &label) in labels.iter().enumerate() {
        per_class[label].push(index);
    }
    let min_count = per_class.iter().map(Vec::len).min().unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9e3779b9);
    let mut selection = Vec::with_capacity(min_count * classes);
    for group in &mut per_class {
        group.shuffle(&mut rng);
        selection.extend_from_slice(&group[..min_count]);
    }
    selection.sort_unstable();
    selection
}

pub fn filter_samples(samples: &mut Vec<VideoSample>, exclude: &[String], merge_aliases: bool) {
    let exclude_lc: Vec<String> = exclude.iter().map(|s| s.to_lowercase()).collect();
    samples.retain(|s| !exclude_lc.contains(&s.username.to_lowercase()));
    if merge_aliases {
        for sample in samples.iter_mut() {
            if sample.username.eq_ignore_ascii_case("sanymaaa") {
                sample.username = "sanymaa".to_string();
            }
        }
    }
}

fn assign_labels(
    samples: &[VideoSample],
    task: LabelTask,
    buckets: usize,
) -> (Vec<usize>, Vec<String>) {
    match task {
        LabelTask::Creator => {
            let names: Vec<String> = samples
                .iter()
                .map(|s| s.username.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let index: HashMap<&str, usize> = names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.as_str(), i))
                .collect();
            let labels = samples.iter().map(|s| index[s.username.as_str()]).collect();
            (labels, names)
        }
        LabelTask::Virality => {
            let values: Vec<f64> = samples.iter().map(|s| s.view_count as f64).collect();
            bucketize(&values, buckets, "views")
        }
        LabelTask::Engagement => {
            let values: Vec<f64> = samples.iter().map(VideoSample::engagement_ratio).collect();
            bucketize(&values, buckets, "like_rate")
        }
    }
}

/// Rank-based quantile bucketing: roughly equal-sized classes, low→high.
fn bucketize(values: &[f64], buckets: usize, prefix: &str) -> (Vec<usize>, Vec<String>) {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));

    let mut labels = vec![0usize; n];
    for (rank, &idx) in order.iter().enumerate() {
        labels[idx] = (rank * buckets / n).min(buckets - 1);
    }

    let tier = ["low", "mid", "high", "top", "elite"];
    let names = (0..buckets)
        .map(|b| {
            let label = tier.get(b).copied().unwrap_or("tier");
            format!("{prefix}_{label}{b}")
        })
        .collect();
    (labels, names)
}

/// Deterministic train/validation split returning index lists.
pub fn split_indices(num_clips: usize, val_split: f32, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..num_clips).collect();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5f3759df);
    indices.shuffle(&mut rng);
    let val_count = ((num_clips as f32) * val_split).round() as usize;
    let val_count = val_count.min(num_clips.saturating_sub(1));
    let val = indices[..val_count].to_vec();
    let train = indices[val_count..].to_vec();
    (train, val)
}

/// Path where a trained model is written by default for a config.
pub fn default_model_path(config: &VideoTrainingConfig) -> PathBuf {
    config
        .data_root
        .join(format!("oxidize-video-{}.json", config.task.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucketize_balances_classes() {
        let values: Vec<f64> = (0..9).map(|v| v as f64).collect();
        let (labels, names) = bucketize(&values, 3, "views");
        assert_eq!(names.len(), 3);
        let mut counts = [0usize; 3];
        for l in labels {
            counts[l] += 1;
        }
        assert_eq!(counts, [3, 3, 3]);
    }

    #[test]
    fn split_is_deterministic_and_disjoint() {
        let (train, val) = split_indices(100, 0.2, 7);
        assert_eq!(train.len() + val.len(), 100);
        assert_eq!(val.len(), 20);
        let (train2, val2) = split_indices(100, 0.2, 7);
        assert_eq!(train, train2);
        assert_eq!(val, val2);
        let mut all = train;
        all.extend(val);
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 100);
    }
}
