use std::collections::BTreeMap;
use std::path::Path;

use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{AdamW, Linear, Matrix};

use super::dataset::filter_samples;
use super::frames::{
    clip_to_tensor, clips_to_tensors_parallel, ffmpeg_available, patches_to_image,
};
use super::manifest::build_manifest;
use super::{FrameConfig, VideoError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenTrainingConfig {
    pub data_root: std::path::PathBuf,
    pub cache_dir: std::path::PathBuf,
    pub frames: FrameConfig,
    pub context_frames: usize,
    pub hidden_size: usize,
    pub patch_hidden: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
    pub val_split: f32,
    pub max_videos: Option<usize>,
    pub exclude: Vec<String>,
    pub merge_aliases: bool,
}

impl Default for GenTrainingConfig {
    fn default() -> Self {
        Self {
            data_root: std::path::PathBuf::from("."),
            cache_dir: std::path::PathBuf::new(),
            frames: FrameConfig::portrait_9_16(72, 8, 12),
            context_frames: 4,
            hidden_size: 256,
            patch_hidden: 128,
            epochs: 40,
            batch_size: 64,
            learning_rate: 1e-3,
            weight_decay: 0.001,
            seed: 42,
            val_split: 0.1,
            max_videos: None,
            exclude: vec!["cellow111".into()],
            merge_aliases: true,
        }
    }
}

impl GenTrainingConfig {
    pub fn validate(&self) -> Result<(), VideoError> {
        self.frames.validate()?;
        if self.context_frames == 0 {
            return Err(VideoError::Config("context_frames must be > 0".into()));
        }
        if self.frames.num_frames <= self.context_frames {
            return Err(VideoError::Config(format!(
                "num_frames ({}) must exceed context_frames ({})",
                self.frames.num_frames, self.context_frames
            )));
        }
        if self.hidden_size == 0 || self.patch_hidden == 0 || self.batch_size == 0 {
            return Err(VideoError::Config(
                "hidden_size, patch_hidden, batch_size must be > 0".into(),
            ));
        }
        if self.epochs == 0 {
            return Err(VideoError::Config("epochs must be > 0".into()));
        }
        if !(0.0..1.0).contains(&self.val_split) {
            return Err(VideoError::Config("val_split must be in [0.0, 1.0)".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GenDataset {
    context: Vec<Vec<f32>>,
    target: Vec<Vec<f32>>,
    pair_clip: Vec<usize>,
    frames: FrameConfig,
    context_frames: usize,
}

impl GenDataset {
    pub fn len(&self) -> usize {
        self.context.len()
    }

    pub fn is_empty(&self) -> bool {
        self.context.is_empty()
    }

    pub fn frames(&self) -> FrameConfig {
        self.frames
    }

    pub fn context_frames(&self) -> usize {
        self.context_frames
    }

    pub fn patch_dim(&self) -> usize {
        self.frames.patch_dim()
    }

    pub fn num_patches(&self) -> usize {
        self.frames.num_patches()
    }

    pub fn context_flat_dim(&self) -> usize {
        self.context_frames * self.num_patches() * self.patch_dim()
    }

    pub fn target_flat_dim(&self) -> usize {
        self.num_patches() * self.patch_dim()
    }
}

pub fn load_gen_dataset(config: &GenTrainingConfig) -> Result<GenDataset, VideoError> {
    config.validate()?;
    if !ffmpeg_available() {
        return Err(VideoError::FfmpegMissing);
    }

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

    let frames_cfg = config.frames.validate()?;
    let ctx = config.context_frames;
    let patch_dim = frames_cfg.patch_dim();
    let patches = frames_cfg.num_patches();
    let frame_flat = patches * patch_dim;

    eprintln!(
        "oxidize-train gen: extracting {} clips @ {} (9:16 portrait)…",
        samples.len(),
        frames_cfg.aspect_label()
    );

    let extracted = clips_to_tensors_parallel(&samples, frames_cfg, &cache_dir);
    let mut context = Vec::new();
    let mut target = Vec::new();
    let mut pair_clip = Vec::new();

    for (clip_idx, tensor) in extracted.into_iter().enumerate() {
        let Some(tensor) = tensor else {
            continue;
        };
        let num_frames = frames_cfg.num_frames;
        if tensor.len() != num_frames * frame_flat {
            continue;
        }
        for start in 0..=(num_frames - ctx - 1) {
            let ctx_start = start * frame_flat;
            let ctx_end = (start + ctx) * frame_flat;
            let tgt_start = ctx_end;
            let tgt_end = tgt_start + frame_flat;
            context.push(tensor[ctx_start..ctx_end].to_vec());
            target.push(tensor[tgt_start..tgt_end].to_vec());
            pair_clip.push(clip_idx);
        }
    }

    if context.is_empty() {
        return Err(VideoError::NoSamples(config.data_root.clone()));
    }

    eprintln!(
        "oxidize-train gen: {} next-frame training pairs from {} clips",
        context.len(),
        samples.len()
    );

    Ok(GenDataset {
        context,
        target,
        pair_clip,
        frames: frames_cfg,
        context_frames: ctx,
    })
}

/// Autoregressive next-frame generator: encodes a context window of patch
/// tokens, fuses them temporally, then decodes the next frame's patches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoGenerator {
    patch_dim: usize,
    num_patches: usize,
    context_frames: usize,
    hidden_size: usize,
    patch_hidden: usize,
    patch_enc: Linear,
    temporal: Linear,
    temporal2: Linear,
    frame_dec: Linear,
    patch_dec: Linear,
}

struct GenForward {
    ctx_matrix: Matrix,
    temporal_in: Matrix,
    t1_pre: Matrix,
    t1: Matrix,
    t2_pre: Matrix,
    t2: Matrix,
    patch_states: Matrix,
    patch_logits: Matrix,
}

impl VideoGenerator {
    pub fn new(config: &GenTrainingConfig, seed: u64) -> Self {
        let frames = config.frames.validate().expect("validated in train/load");
        let patch_dim = frames.patch_dim();
        let num_patches = frames.num_patches();
        let context_frames = config.context_frames;
        let hidden = config.hidden_size;
        let patch_hidden = config.patch_hidden;
        let mut rng = StdRng::seed_from_u64(seed);

        Self {
            patch_dim,
            num_patches,
            context_frames,
            hidden_size: hidden,
            patch_hidden,
            patch_enc: Linear::new(patch_dim, patch_hidden, &mut rng),
            temporal: Linear::new(patch_hidden * context_frames, hidden, &mut rng),
            temporal2: Linear::new(hidden, hidden, &mut rng),
            frame_dec: Linear::new(hidden, patch_hidden * num_patches, &mut rng),
            patch_dec: Linear::new(patch_hidden, patch_dim, &mut rng),
        }
    }

    fn forward(&self, context_flat: &[f32]) -> GenForward {
        let ctx_patches = self.context_frames * self.num_patches;
        let ctx_matrix = Matrix::from_vec(ctx_patches, self.patch_dim, context_flat.to_vec())
            .expect("context shape");

        let mut patch_h = Matrix::zeros(ctx_patches, self.patch_hidden);
        self.patch_enc.forward(&ctx_matrix, &mut patch_h);

        let mut frame_h = Matrix::zeros(self.context_frames, self.patch_hidden);
        mean_pool_rows(&patch_h, self.num_patches, &mut frame_h);

        let temporal_in = Matrix::from_vec(1, frame_h.data().len(), frame_h.data().to_vec())
            .expect("temporal in");
        let mut t1_pre = Matrix::zeros(1, self.hidden_size);
        self.temporal.forward(&temporal_in, &mut t1_pre);
        let mut t1 = t1_pre.clone();
        relu_in_place(t1.data_mut());

        let mut t2_pre = Matrix::zeros(1, self.hidden_size);
        self.temporal2.forward(&t1, &mut t2_pre);
        let mut t2 = t2_pre.clone();
        relu_in_place(t2.data_mut());

        let mut frame_states_flat = Matrix::zeros(1, self.patch_hidden * self.num_patches);
        self.frame_dec.forward(&t2, &mut frame_states_flat);
        let patch_states = Matrix::from_vec(
            self.num_patches,
            self.patch_hidden,
            frame_states_flat.data().to_vec(),
        )
        .expect("patch states");

        let mut patch_logits = Matrix::zeros(self.num_patches, self.patch_dim);
        self.patch_dec.forward(&patch_states, &mut patch_logits);

        GenForward {
            ctx_matrix,
            temporal_in,
            t1_pre,
            t1,
            t2_pre,
            t2,
            patch_states,
            patch_logits,
        }
    }

    pub fn predict_frame(&self, context_flat: &[f32]) -> Vec<f32> {
        self.forward(context_flat).patch_logits.data().to_vec()
    }

    pub(crate) fn zero_grad(&mut self) {
        self.patch_enc.zero_grad();
        self.temporal.zero_grad();
        self.temporal2.zero_grad();
        self.frame_dec.zero_grad();
        self.patch_dec.zero_grad();
    }

    pub(crate) fn backward_step(
        &mut self,
        context_flat: &[f32],
        target_flat: &[f32],
        grad_scale: f32,
    ) -> f32 {
        let cache = self.forward(context_flat);
        let target = Matrix::from_vec(self.num_patches, self.patch_dim, target_flat.to_vec())
            .expect("target shape");

        let mut grad = Matrix::zeros(self.num_patches, self.patch_dim);
        let loss = mse_loss(&cache.patch_logits, &target, &mut grad);
        if grad_scale != 1.0 {
            for g in grad.data_mut() {
                *g *= grad_scale;
            }
        }

        let mut patch_states_grad = Matrix::zeros(self.num_patches, self.patch_hidden);
        self.patch_dec
            .backward(&cache.patch_states, &grad, Some(&mut patch_states_grad));

        let mut frame_states_grad = Matrix::zeros(1, self.patch_hidden * self.num_patches);
        frame_states_grad
            .data_mut()
            .copy_from_slice(patch_states_grad.data());

        let mut t2_grad = Matrix::zeros(1, self.hidden_size);
        self.frame_dec
            .backward(&cache.t2, &frame_states_grad, Some(&mut t2_grad));
        relu_backward(&cache.t2_pre, &mut t2_grad);

        let mut t1_grad = Matrix::zeros(1, self.hidden_size);
        self.temporal2
            .backward(&cache.t1, &t2_grad, Some(&mut t1_grad));
        relu_backward(&cache.t1_pre, &mut t1_grad);

        let mut temporal_in_grad = Matrix::zeros(1, self.patch_hidden * self.context_frames);
        self.temporal
            .backward(&cache.temporal_in, &t1_grad, Some(&mut temporal_in_grad));

        let mut frame_grad = Matrix::zeros(self.context_frames, self.patch_hidden);
        for g in 0..self.context_frames {
            for h in 0..self.patch_hidden {
                frame_grad.data_mut()[g * self.patch_hidden + h] =
                    temporal_in_grad.data()[g * self.patch_hidden + h];
            }
        }

        let mut patch_h_grad =
            Matrix::zeros(self.context_frames * self.num_patches, self.patch_hidden);
        unpool_rows(&frame_grad, self.context_frames, &mut patch_h_grad);

        self.patch_enc
            .backward(&cache.ctx_matrix, &patch_h_grad, None);

        loss
    }

    pub(crate) fn apply_optimizer(&mut self, optimizer: &mut AdamW) {
        optimizer.next_step();
        optimizer.step(&mut self.patch_enc);
        optimizer.step(&mut self.temporal);
        optimizer.step(&mut self.temporal2);
        optimizer.step(&mut self.frame_dec);
        optimizer.step(&mut self.patch_dec);
    }

    #[allow(dead_code)]
    pub(crate) fn train_step(
        &mut self,
        context_flat: &[f32],
        target_flat: &[f32],
        optimizer: &mut AdamW,
    ) -> f32 {
        self.zero_grad();
        let loss = self.backward_step(context_flat, target_flat, 1.0);
        self.apply_optimizer(optimizer);
        loss
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenTrainingReport {
    pub pairs: usize,
    pub train_pairs: usize,
    pub val_pairs: usize,
    pub epochs: usize,
    pub final_train_loss: f32,
    pub val_loss: f32,
    pub best_val_loss: f32,
    pub epoch_losses: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedGenerator {
    pub generator: VideoGenerator,
    pub frames: FrameConfig,
    pub context_frames: usize,
}

pub fn train_generator(
    dataset: &GenDataset,
    config: &GenTrainingConfig,
) -> Result<(VideoGenerator, GenTrainingReport), VideoError> {
    let mut model = VideoGenerator::new(config, config.seed);
    let mut optimizer = AdamW::new(config.learning_rate, config.weight_decay);
    let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(1));

    let n = dataset.len();
    let mut by_clip: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (pair_idx, &clip) in dataset.pair_clip.iter().enumerate() {
        by_clip.entry(clip).or_default().push(pair_idx);
    }
    let mut clip_ids: Vec<usize> = by_clip.keys().copied().collect();
    clip_ids.shuffle(&mut rng);
    let val_clip_count = ((clip_ids.len() as f32) * config.val_split).round() as usize;
    let val_clip_count = val_clip_count.min(clip_ids.len().saturating_sub(1));
    let (val_clips, train_clips) = clip_ids.split_at(val_clip_count);
    let val_idx: Vec<usize> = val_clips
        .iter()
        .flat_map(|clip| &by_clip[clip])
        .copied()
        .collect();
    let train_idx: Vec<usize> = train_clips
        .iter()
        .flat_map(|clip| &by_clip[clip])
        .copied()
        .collect();
    let track_val = !val_idx.is_empty();

    let mut epoch_losses = Vec::with_capacity(config.epochs);
    let mut best_val = f32::MAX;
    let mut best_model = model.clone();

    for epoch in 0..config.epochs {
        let mut order = train_idx.to_vec();
        order.shuffle(&mut rng);
        let mut weighted = 0.0f32;
        let mut seen = 0usize;

        for batch in order.chunks(config.batch_size) {
            model.zero_grad();
            let mut batch_loss = 0.0f32;
            let scale = 1.0 / batch.len() as f32;
            for &idx in batch {
                batch_loss +=
                    model.backward_step(&dataset.context[idx], &dataset.target[idx], scale);
            }
            model.apply_optimizer(&mut optimizer);
            weighted += batch_loss / batch.len() as f32;
            seen += batch.len();
        }
        let train_loss = weighted / seen.max(1) as f32;
        epoch_losses.push(train_loss);

        let val_loss = if track_val {
            eval_loss(&model, dataset, &val_idx)
        } else {
            f32::NAN
        };
        if track_val && val_loss < best_val {
            best_val = val_loss;
            best_model = model.clone();
        }
        eprintln!(
            "  epoch {:>3}/{}  train_mse={train_loss:.5}  val_mse={}  best={}",
            epoch + 1,
            config.epochs,
            if track_val {
                format!("{val_loss:.5}")
            } else {
                "n/a".into()
            },
            if track_val {
                format!("{best_val:.5}")
            } else {
                "n/a".into()
            },
        );
    }

    model = if track_val { best_model } else { model };
    let val_loss = if track_val {
        eval_loss(&model, dataset, &val_idx)
    } else {
        f32::NAN
    };
    let best_val_loss = if track_val { best_val } else { f32::NAN };

    Ok((
        model,
        GenTrainingReport {
            pairs: n,
            train_pairs: train_idx.len(),
            val_pairs: val_idx.len(),
            epochs: config.epochs,
            final_train_loss: epoch_losses.last().copied().unwrap_or(f32::NAN),
            val_loss,
            best_val_loss,
            epoch_losses,
        },
    ))
}

fn eval_loss(model: &VideoGenerator, dataset: &GenDataset, indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for &idx in indices {
        let pred = model.predict_frame(&dataset.context[idx]);
        let target = &dataset.target[idx];
        sum += mse_value(&pred, target);
    }
    sum / indices.len() as f32
}

pub fn save_generator(
    path: &Path,
    model: &VideoGenerator,
    config: &GenTrainingConfig,
) -> Result<(), VideoError> {
    let saved = SavedGenerator {
        generator: model.clone(),
        frames: config.frames.validate()?,
        context_frames: config.context_frames,
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

pub fn load_generator(path: &Path) -> Result<SavedGenerator, VideoError> {
    let bytes = std::fs::read(path).map_err(|source| VideoError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut saved: SavedGenerator =
        serde_json::from_slice(&bytes).map_err(|source| VideoError::Metadata {
            path: path.to_path_buf(),
            source,
        })?;
    saved.frames = saved.frames.validate()?;
    Ok(saved)
}

pub fn default_generator_path(config: &GenTrainingConfig) -> std::path::PathBuf {
    config.data_root.join("oxidize-video-generator.json")
}

#[derive(Debug, Clone)]
pub struct GenerateReport {
    pub frames: usize,
    pub out: std::path::PathBuf,
    pub seed_clip: String,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_video(
    model: &VideoGenerator,
    saved: &SavedGenerator,
    data_root: &Path,
    cache_dir: &Path,
    exclude: &[String],
    merge_aliases: bool,
    output_frames: usize,
    temperature: f32,
    fps: u32,
    out: &Path,
    seed: u64,
    upscale_height: u32,
) -> Result<GenerateReport, VideoError> {
    if !ffmpeg_available() {
        return Err(VideoError::FfmpegMissing);
    }

    let frames_cfg = saved.frames;
    let ctx = saved.context_frames;
    let patch_dim = frames_cfg.patch_dim();
    let patches = frames_cfg.num_patches();
    let frame_flat = patches * patch_dim;

    eprintln!(
        "oxidize-train generate: model resolution {} (output upscale height={upscale_height})",
        frames_cfg.aspect_label()
    );

    let mut samples = build_manifest(data_root)?;
    filter_samples(&mut samples, exclude, merge_aliases);
    if samples.is_empty() {
        return Err(VideoError::NoSamples(data_root.to_path_buf()));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    samples.shuffle(&mut rng);
    let seed_sample = samples
        .iter()
        .find(|s| clip_to_tensor(s, frames_cfg, cache_dir).is_ok())
        .ok_or_else(|| VideoError::NoFrames(data_root.to_path_buf()))?;
    let seed_tensor = clip_to_tensor(seed_sample, frames_cfg, cache_dir)?;

    let mut timeline: Vec<Vec<f32>> = (0..ctx)
        .map(|i| seed_tensor[i * frame_flat..(i + 1) * frame_flat].to_vec())
        .collect();

    eprintln!(
        "oxidize-train generate: seed clip {} → autoregress {} new frames",
        seed_sample.id, output_frames
    );

    for step in 0..output_frames {
        let mut ctx_flat = Vec::with_capacity(ctx * frame_flat);
        for frame in timeline.iter().rev().take(ctx).rev() {
            ctx_flat.extend_from_slice(frame);
        }
        let mut next = model.predict_frame(&ctx_flat);
        if temperature > 0.0 {
            for v in &mut next {
                *v += rng.gen_range(-1.0..=1.0) * temperature;
            }
        }
        timeline.push(next);
        if (step + 1) % 8 == 0 || step + 1 == output_frames {
            eprintln!("  …generated {}/{} frames", step + 1, output_frames);
        }
    }

    let work = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(".oxidize-generate");
    std::fs::create_dir_all(&work).map_err(|source| VideoError::Io {
        path: work.clone(),
        source,
    })?;

    for (i, frame) in timeline.iter().enumerate() {
        let img = patches_to_image(frame, frames_cfg);
        let path = work.join(format!("gen_{:04}.jpg", i + 1));
        img.save(&path).map_err(|source| VideoError::Decode {
            path: path.clone(),
            source,
        })?;
    }

    encode_generated(&work, out, fps, upscale_height)?;

    Ok(GenerateReport {
        frames: timeline.len(),
        out: out.to_path_buf(),
        seed_clip: seed_sample.id.clone(),
    })
}

fn encode_generated(
    work: &Path,
    out: &Path,
    fps: u32,
    upscale_height: u32,
) -> Result<(), VideoError> {
    let pattern = work.join("gen_%04d.jpg");
    let vf = if upscale_height > 0 {
        format!(
            "scale=-2:{upscale_height}:flags=lanczos,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc,format=yuv420p"
        )
    } else {
        "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc,format=yuv420p".to_string()
    };
    let output = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate"])
        .arg(fps.to_string())
        .args(["-i"])
        .arg(&pattern)
        .args(["-vf", &vf])
        .args(["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"])
        .arg(out)
        .output()
        .map_err(|source| VideoError::Io {
            path: out.to_path_buf(),
            source,
        })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(VideoError::Ffmpeg {
            path: out.to_path_buf(),
            message: String::from_utf8_lossy(&output.stderr)
                .lines()
                .last()
                .unwrap_or("ffmpeg encode failed")
                .to_string(),
        })
    }
}

fn mean_pool_rows(input: &Matrix, group: usize, output: &mut Matrix) {
    let hidden = input.cols;
    let groups = input.rows / group;
    for g in 0..groups {
        for h in 0..hidden {
            let mut sum = 0.0f32;
            for p in 0..group {
                sum += input.data()[(g * group + p) * hidden + h];
            }
            output.data_mut()[g * hidden + h] = sum / group as f32;
        }
    }
}

fn unpool_rows(grad: &Matrix, groups: usize, output: &mut Matrix) {
    let hidden = grad.cols;
    let group = output.rows / groups;
    for g in 0..groups {
        for h in 0..hidden {
            let v = grad.data()[g * hidden + h] / group as f32;
            for p in 0..group {
                output.data_mut()[(g * group + p) * hidden + h] = v;
            }
        }
    }
}

fn relu_in_place(data: &mut [f32]) {
    for v in data {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

fn relu_backward(pre: &Matrix, grad: &mut Matrix) {
    for (g, p) in grad.data_mut().iter_mut().zip(pre.data().iter()) {
        if *p <= 0.0 {
            *g = 0.0;
        }
    }
}

fn mse_loss(pred: &Matrix, target: &Matrix, grad: &mut Matrix) -> f32 {
    let n = pred.data().len().max(1) as f32;
    let inv = 2.0 / n;
    let mut sum = 0.0f32;
    for (g, (p, t)) in grad
        .data_mut()
        .iter_mut()
        .zip(pred.data().iter().zip(target.data()))
    {
        let diff = *p - *t;
        sum += diff * diff;
        *g = diff * inv;
    }
    sum / n
}

fn mse_value(pred: &[f32], target: &[f32]) -> f32 {
    let n = pred.len().max(1) as f32;
    pred.iter()
        .zip(target)
        .map(|(p, t)| {
            let d = *p - *t;
            d * d
        })
        .sum::<f32>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> GenTrainingConfig {
        GenTrainingConfig {
            frames: FrameConfig {
                num_frames: 6,
                frame_width: 8,
                frame_height: 8,
                frame_size: 0,
                patch_size: 4,
            },
            context_frames: 2,
            hidden_size: 16,
            patch_hidden: 8,
            epochs: 8,
            batch_size: 1,
            ..GenTrainingConfig::default()
        }
    }

    fn synthetic_dataset() -> GenDataset {
        let cfg = tiny_config();
        let patch_dim = cfg.frames.patch_dim();
        let patches = cfg.frames.num_patches();
        let frame = patches * patch_dim;
        let ctx = cfg.context_frames * frame;
        let mut context = Vec::new();
        let mut target = Vec::new();
        let mut pair_clip = Vec::new();
        for i in 0..20 {
            let phase = i as f32 * 0.1;
            let mut ctx_vec = vec![0.0; ctx];
            for j in 0..cfg.context_frames {
                for k in 0..frame {
                    ctx_vec[j * frame + k] = (phase + j as f32 * 0.05 + k as f32 * 0.001).sin();
                }
            }
            let mut tgt = vec![0.0; frame];
            for k in 0..frame {
                tgt[k] = (phase + 0.15 + k as f32 * 0.001).sin();
            }
            context.push(ctx_vec);
            target.push(tgt);
            pair_clip.push(i / 4);
        }
        GenDataset {
            context,
            target,
            pair_clip,
            frames: cfg.frames,
            context_frames: cfg.context_frames,
        }
    }

    #[test]
    fn generator_learns_synthetic_dynamics() {
        let cfg = tiny_config();
        let dataset = synthetic_dataset();
        let (model, report) = train_generator(&dataset, &cfg).expect("train");
        assert!(
            report.best_val_loss < 0.5,
            "val_mse={}",
            report.best_val_loss
        );
        let pred = model.predict_frame(&dataset.context[0]);
        let err = mse_value(&pred, &dataset.target[0]);
        assert!(err < 0.5, "sample err={err}");
    }
}
