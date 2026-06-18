use std::path::Path;

use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{AdamW, Matrix};

use super::dataset::split_indices;
use super::{FrameConfig, VideoClassifier, VideoDataset, VideoError, VideoTrainingConfig};

/// Summary of a finished training run; serialized alongside the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoTrainingReport {
    pub task: String,
    pub clips: usize,
    pub train_clips: usize,
    pub val_clips: usize,
    pub classes: usize,
    pub class_names: Vec<String>,
    pub class_histogram: Vec<usize>,
    pub epochs: usize,
    pub final_train_loss: f32,
    pub train_accuracy: f32,
    pub val_accuracy: f32,
    pub majority_baseline: f32,
    pub epoch_losses: Vec<f32>,
}

/// Model plus everything needed to interpret its outputs later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedModel {
    pub classifier: VideoClassifier,
    pub task: String,
    pub class_names: Vec<String>,
    pub frames: FrameConfig,
}

pub fn train_video_classifier(
    dataset: &VideoDataset,
    config: &VideoTrainingConfig,
) -> Result<(VideoClassifier, VideoTrainingReport), VideoError> {
    let frames = dataset.frames();
    let (train_idx, val_idx) = split_indices(dataset.num_clips(), config.val_split, config.seed);

    let mut model = VideoClassifier::new(
        frames.patch_dim(),
        config.embed_dim,
        config.hidden_size,
        dataset.classes(),
        frames.num_frames,
        frames.tokens_per_clip(),
        config.seed,
    );
    let mut optimizer = AdamW::new(config.learning_rate, config.weight_decay);
    let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(1));
    let mut order = train_idx.clone();
    let mut epoch_losses = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        order.shuffle(&mut rng);
        let mut weighted_loss = 0.0;
        let mut seen = 0usize;

        for batch in order.chunks(config.batch_size) {
            let (input, labels) = gather(dataset, batch)?;
            let loss = model.train_step(&input, &labels, &mut optimizer);
            weighted_loss += loss * batch.len() as f32;
            seen += batch.len();
        }

        let train_loss = weighted_loss / seen.max(1) as f32;
        epoch_losses.push(train_loss);
        let val_acc = accuracy(&model, dataset, &val_idx, config.batch_size)?;
        eprintln!(
            "  epoch {:>3}/{}  train_loss={train_loss:.4}  val_acc={val_acc:.4}",
            epoch + 1,
            config.epochs
        );
    }

    let train_accuracy = accuracy(&model, dataset, &train_idx, config.batch_size)?;
    let val_accuracy = accuracy(&model, dataset, &val_idx, config.batch_size)?;
    let majority_baseline = majority_baseline(dataset, &val_idx);

    let report = VideoTrainingReport {
        task: config.task.as_str().to_string(),
        clips: dataset.num_clips(),
        train_clips: train_idx.len(),
        val_clips: val_idx.len(),
        classes: dataset.classes(),
        class_names: dataset.class_names().to_vec(),
        class_histogram: dataset.class_histogram(),
        epochs: config.epochs,
        final_train_loss: epoch_losses.last().copied().unwrap_or(f32::NAN),
        train_accuracy,
        val_accuracy,
        majority_baseline,
        epoch_losses,
    };

    Ok((model, report))
}

fn gather(dataset: &VideoDataset, indices: &[usize]) -> Result<(Matrix, Vec<usize>), VideoError> {
    let frames = dataset.frames();
    let tokens = frames.tokens_per_clip();
    let patch_dim = frames.patch_dim();
    let mut data = Vec::with_capacity(indices.len() * tokens * patch_dim);
    let mut labels = Vec::with_capacity(indices.len());
    for &index in indices {
        data.extend_from_slice(dataset.clip(index));
        labels.push(dataset.labels()[index]);
    }
    let matrix = Matrix::from_vec(indices.len() * tokens, patch_dim, data)
        .map_err(|err| VideoError::Config(err.to_string()))?;
    Ok((matrix, labels))
}

fn accuracy(
    model: &VideoClassifier,
    dataset: &VideoDataset,
    indices: &[usize],
    batch_size: usize,
) -> Result<f32, VideoError> {
    if indices.is_empty() {
        return Ok(0.0);
    }
    let mut correct = 0usize;
    for batch in indices.chunks(batch_size.max(1)) {
        let (input, labels) = gather(dataset, batch)?;
        for (predicted, expected) in model.predict(&input).into_iter().zip(labels) {
            if predicted == expected {
                correct += 1;
            }
        }
    }
    Ok(correct as f32 / indices.len() as f32)
}

fn majority_baseline(dataset: &VideoDataset, val_idx: &[usize]) -> f32 {
    if val_idx.is_empty() {
        return 0.0;
    }
    let mut counts = vec![0usize; dataset.classes()];
    for &index in val_idx {
        counts[dataset.labels()[index]] += 1;
    }
    counts.into_iter().max().unwrap_or(0) as f32 / val_idx.len() as f32
}

pub fn save_model(
    path: &Path,
    model: &VideoClassifier,
    dataset: &VideoDataset,
    config: &VideoTrainingConfig,
) -> Result<(), VideoError> {
    let saved = SavedModel {
        classifier: model.clone(),
        task: config.task.as_str().to_string(),
        class_names: dataset.class_names().to_vec(),
        frames: dataset.frames(),
    };
    let json = serde_json::to_vec_pretty(&saved).map_err(|source| VideoError::Metadata {
        path: path.to_path_buf(),
        source,
    })?;
    std::fs::write(path, json).map_err(|source| VideoError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::LabelTask;

    fn synthetic_dataset() -> VideoDataset {
        let frames = FrameConfig {
            num_frames: 2,
            frame_size: 4,
            patch_size: 2,
        };
        let tokens = frames.tokens_per_clip();
        let patch_dim = frames.patch_dim();
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for class in 0..3 {
            for _ in 0..12 {
                let value = class as f32 - 1.0;
                for _ in 0..(tokens * patch_dim) {
                    data.push(value);
                }
                labels.push(class);
            }
        }
        VideoDataset::from_raw(
            data,
            labels,
            vec!["a".into(), "b".into(), "c".into()],
            frames,
        )
    }

    #[test]
    fn trains_and_separates_synthetic_classes() {
        let dataset = synthetic_dataset();
        let config = VideoTrainingConfig {
            task: LabelTask::Creator,
            epochs: 40,
            batch_size: 8,
            learning_rate: 0.03,
            val_split: 0.25,
            ..VideoTrainingConfig::default()
        };
        let (_, report) = train_video_classifier(&dataset, &config).expect("training");
        assert_eq!(report.classes, 3);
        assert!(
            report.val_accuracy >= report.majority_baseline,
            "val_acc {} should beat baseline {}",
            report.val_accuracy,
            report.majority_baseline
        );
        assert!(report.train_accuracy > 0.8, "train_acc={}", report.train_accuracy);
    }
}
