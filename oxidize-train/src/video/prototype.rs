use std::path::{Path, PathBuf};
use std::process::Command;

use image::imageops::FilterType;
use image::RgbImage;
use rayon::prelude::*;

use super::frames::{clip_to_frames, ffmpeg_available};
use super::manifest::build_manifest;
use super::{FrameConfig, VideoError};

/// Summary of a generated prototype ("base") clip.
#[derive(Debug, Clone)]
pub struct PrototypeReport {
    pub clips: usize,
    pub frames: usize,
    pub creators: Vec<String>,
    pub out: PathBuf,
    pub sheet: Option<PathBuf>,
}

/// Build a "base" video by averaging, per temporal slot, the frames of every
/// clip whose creator is not excluded — the canonical data-mean baseline.
#[allow(clippy::too_many_arguments)]
pub fn build_prototype(
    data_root: &Path,
    cache_dir: &Path,
    frames_cfg: FrameConfig,
    exclude: &[String],
    out: &Path,
    sheet: Option<&Path>,
    upscale: u32,
    fps: u32,
) -> Result<PrototypeReport, VideoError> {
    if frames_cfg.num_frames == 0 || frames_cfg.frame_size == 0 {
        return Err(VideoError::Config(
            "num_frames and frame_size must be > 0".into(),
        ));
    }
    if !ffmpeg_available() {
        return Err(VideoError::FfmpegMissing);
    }

    let exclude_lc: Vec<String> = exclude.iter().map(|s| s.to_lowercase()).collect();
    let samples: Vec<_> = build_manifest(data_root)?
        .into_iter()
        .filter(|s| !exclude_lc.contains(&s.username.to_lowercase()))
        .collect();
    if samples.is_empty() {
        return Err(VideoError::NoSamples(data_root.to_path_buf()));
    }

    let mut creators: Vec<String> = samples.iter().map(|s| s.username.clone()).collect();
    creators.sort();
    creators.dedup();
    eprintln!(
        "oxidize-train prototype: averaging {} clips from {} creator(s): {}",
        samples.len(),
        creators.len(),
        creators.join(", ")
    );

    let n = frames_cfg.num_frames;
    let size = frames_cfg.frame_size;
    let pixels = size * size * 3;

    let loaded: Vec<Vec<RgbImage>> = samples
        .par_iter()
        .filter_map(|sample| clip_to_frames(sample, frames_cfg, cache_dir).ok())
        .collect();
    if loaded.is_empty() {
        return Err(VideoError::NoFrames(data_root.to_path_buf()));
    }

    let mut sums = vec![vec![0f64; pixels]; n];
    let mut counts = vec![0usize; n];
    for clip in &loaded {
        for (slot, img) in clip.iter().enumerate().take(n) {
            let acc = &mut sums[slot];
            for (dst, &value) in acc.iter_mut().zip(img.as_raw()) {
                *dst += value as f64;
            }
            counts[slot] += 1;
        }
    }

    let mean_frames: Vec<RgbImage> = (0..n)
        .map(|slot| {
            let inv = 1.0 / counts[slot].max(1) as f64;
            let data: Vec<u8> = sums[slot]
                .iter()
                .map(|v| (v * inv).round().clamp(0.0, 255.0) as u8)
                .collect();
            RgbImage::from_raw(size as u32, size as u32, data)
                .unwrap_or_else(|| RgbImage::new(size as u32, size as u32))
        })
        .collect();

    let work = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(".oxidize-prototype");
    std::fs::create_dir_all(&work).map_err(|source| VideoError::Io {
        path: work.clone(),
        source,
    })?;
    for (i, img) in mean_frames.iter().enumerate() {
        let up = image::imageops::resize(img, upscale, upscale, FilterType::Lanczos3);
        let path = work.join(format!("mean_{:03}.jpg", i + 1));
        up.save(&path).map_err(|source| VideoError::Decode {
            path: path.clone(),
            source,
        })?;
    }

    encode_video(&work, out, fps)?;

    let sheet_out = match sheet {
        Some(path) => {
            write_contact_sheet(&mean_frames, path)?;
            Some(path.to_path_buf())
        }
        None => None,
    };

    Ok(PrototypeReport {
        clips: loaded.len(),
        frames: n,
        creators,
        out: out.to_path_buf(),
        sheet: sheet_out,
    })
}

/// Encode the mean frames into a smooth mp4 (blended interpolation between slots).
fn encode_video(work: &Path, out: &Path, fps: u32) -> Result<(), VideoError> {
    let pattern = work.join("mean_%03d.jpg");
    let vf = format!("minterpolate=fps={fps}:mi_mode=blend,format=yuv420p");
    let output = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", "2", "-i"])
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
                .unwrap_or("ffmpeg failed to encode the prototype video")
                .to_string(),
        })
    }
}

/// Save a horizontal contact sheet of the mean frames for quick inspection.
fn write_contact_sheet(frames: &[RgbImage], out: &Path) -> Result<(), VideoError> {
    let cell = 160u32;
    let mut sheet = RgbImage::new(cell * frames.len().max(1) as u32, cell);
    for (i, img) in frames.iter().enumerate() {
        let up = image::imageops::resize(img, cell, cell, FilterType::Lanczos3);
        image::imageops::overlay(&mut sheet, &up, i as i64 * cell as i64, 0);
    }
    sheet.save(out).map_err(|source| VideoError::Decode {
        path: out.to_path_buf(),
        source,
    })
}
