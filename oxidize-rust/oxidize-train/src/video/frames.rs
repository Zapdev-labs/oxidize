use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use image::RgbImage;
use image::imageops::FilterType;
use rayon::prelude::*;

use super::{FrameConfig, VideoError, VideoSample};

pub fn ffmpeg_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

pub fn clip_to_tensor(
    sample: &VideoSample,
    cfg: FrameConfig,
    cache_dir: &Path,
) -> Result<Vec<f32>, VideoError> {
    let cfg = cfg.validate()?;
    let frame_dir = cache_dir.join(&sample.id);
    let mut frames = existing_frames(&frame_dir);
    if frames.is_empty() {
        frames = extract_frames(sample, cfg, &frame_dir)?;
    }
    if frames.is_empty() {
        return Err(VideoError::NoFrames(sample.path.clone()));
    }
    decode_clip(&frames, cfg)
}

#[allow(dead_code)]
pub fn clip_to_frames(
    sample: &VideoSample,
    cfg: FrameConfig,
    cache_dir: &Path,
) -> Result<Vec<RgbImage>, VideoError> {
    let cfg = cfg.validate()?;
    let frame_dir = cache_dir.join(&sample.id);
    let mut frames = existing_frames(&frame_dir);
    if frames.is_empty() {
        frames = extract_frames(sample, cfg, &frame_dir)?;
    }
    if frames.is_empty() {
        return Err(VideoError::NoFrames(sample.path.clone()));
    }
    let chosen = pick_indices(frames.len(), cfg.num_frames);
    chosen
        .into_iter()
        .map(|idx| decode_frame(&frames[idx], cfg))
        .collect()
}

fn existing_frames(frame_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(frame_dir) else {
        return Vec::new();
    };
    let mut frames: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jpg"))
        .collect();
    frames.sort();
    frames
}

fn extract_frames(
    sample: &VideoSample,
    cfg: FrameConfig,
    frame_dir: &Path,
) -> Result<Vec<PathBuf>, VideoError> {
    if !ffmpeg_available() {
        return Err(VideoError::FfmpegMissing);
    }
    std::fs::create_dir_all(frame_dir).map_err(|source| VideoError::Io {
        path: frame_dir.to_path_buf(),
        source,
    })?;

    let w = cfg.frame_width;
    let h = cfg.frame_height;
    let fps = if sample.duration > 0.1 {
        (cfg.num_frames as f64 / sample.duration).max(0.1)
    } else {
        2.0
    };
    let vf =
        format!("scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps:.4}");
    let pattern = frame_dir.join("f_%03d.jpg");

    let output = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&sample.path)
        .args(["-vf", &vf])
        .args(["-frames:v", &cfg.num_frames.to_string()])
        .args(["-q:v", "3"])
        .arg(&pattern)
        .output()
        .map_err(|source| VideoError::Io {
            path: sample.path.clone(),
            source,
        })?;

    if !output.status.success() {
        return Err(VideoError::Ffmpeg {
            path: sample.path.clone(),
            message: String::from_utf8_lossy(&output.stderr)
                .lines()
                .last()
                .unwrap_or("ffmpeg exited with a non-zero status")
                .to_string(),
        });
    }

    Ok(existing_frames(frame_dir))
}

fn decode_clip(frames: &[PathBuf], cfg: FrameConfig) -> Result<Vec<f32>, VideoError> {
    let chosen = pick_indices(frames.len(), cfg.num_frames);
    let mut tensor = Vec::with_capacity(cfg.tokens_per_clip() * cfg.patch_dim());
    for &idx in &chosen {
        let img = decode_frame(&frames[idx], cfg)?;
        patchify_into(&img, cfg, &mut tensor);
    }
    Ok(tensor)
}

fn decode_frame(path: &Path, cfg: FrameConfig) -> Result<RgbImage, VideoError> {
    let img = image::open(path)
        .map_err(|source| VideoError::Decode {
            path: path.to_path_buf(),
            source,
        })?
        .to_rgb8();
    let w = cfg.frame_width as u32;
    let h = cfg.frame_height as u32;
    if img.width() == w && img.height() == h {
        Ok(img)
    } else {
        Ok(image::imageops::resize(&img, w, h, FilterType::Triangle))
    }
}

fn patchify_into(img: &RgbImage, cfg: FrameConfig, out: &mut Vec<f32>) {
    let px = cfg.patches_x();
    let py = cfg.patches_y();
    let p = cfg.patch_size as u32;
    for gy in 0..py as u32 {
        for gx in 0..px as u32 {
            for ly in 0..p {
                for lx in 0..p {
                    let px_x = gx * p + lx;
                    let px_y = gy * p + ly;
                    let pixel = img.get_pixel(px_x, px_y);
                    for channel in 0..3 {
                        out.push(pixel[channel] as f32 / 127.5 - 1.0);
                    }
                }
            }
        }
    }
}

pub fn patches_to_image(patches: &[f32], cfg: FrameConfig) -> RgbImage {
    let cfg = cfg.resolve_dimensions();
    let px = cfg.patches_x();
    let py = cfg.patches_y();
    let p = cfg.patch_size as u32;
    let mut img = RgbImage::new(cfg.frame_width as u32, cfg.frame_height as u32);
    let mut idx = 0usize;
    for gy in 0..py as u32 {
        for gx in 0..px as u32 {
            for ly in 0..p {
                for lx in 0..p {
                    let r = denorm(*patches.get(idx).unwrap_or(&0.0));
                    let g = denorm(*patches.get(idx + 1).unwrap_or(&0.0));
                    let b = denorm(*patches.get(idx + 2).unwrap_or(&0.0));
                    img.put_pixel(gx * p + lx, gy * p + ly, image::Rgb([r, g, b]));
                    idx += 3;
                }
            }
        }
    }
    img
}

#[inline]
fn denorm(v: f32) -> u8 {
    ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8
}

fn pick_indices(available: usize, want: usize) -> Vec<usize> {
    if available == 0 || want == 0 {
        return Vec::new();
    }
    (0..want)
        .map(|i| {
            if want == 1 {
                0
            } else {
                let pos = i as f64 * (available - 1) as f64 / (want - 1) as f64;
                pos.round() as usize
            }
        })
        .collect()
}

/// Extract tensors from all samples in parallel (ffmpeg + decode).
pub fn clips_to_tensors_parallel(
    samples: &[VideoSample],
    cfg: FrameConfig,
    cache_dir: &Path,
) -> Vec<Option<Vec<f32>>> {
    samples
        .par_iter()
        .map(|sample| clip_to_tensor(sample, cfg, cache_dir).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portrait_9_16_dimensions() {
        let cfg = FrameConfig::portrait_9_16(72, 8, 8);
        assert_eq!(cfg.frame_width, 72);
        assert_eq!(cfg.frame_height, 128);
        assert_eq!(cfg.patches_x(), 9);
        assert_eq!(cfg.patches_y(), 16);
        assert_eq!(cfg.num_patches(), 144);
    }

    #[test]
    fn legacy_frame_size_resolves_square() {
        let cfg = FrameConfig {
            num_frames: 4,
            frame_width: 0,
            frame_height: 0,
            frame_size: 64,
            patch_size: 8,
        }
        .validate()
        .expect("valid");
        assert_eq!(cfg.frame_width, 64);
        assert_eq!(cfg.frame_height, 64);
    }

    #[test]
    fn patchify_portrait_length() {
        let cfg = FrameConfig::portrait_9_16(72, 8, 1);
        let img = RgbImage::new(72, 128);
        let mut out = Vec::new();
        patchify_into(&img, cfg, &mut out);
        assert_eq!(out.len(), cfg.num_patches() * cfg.patch_dim());
    }
}
