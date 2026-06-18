use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use image::imageops::FilterType;
use image::RgbImage;

use super::{FrameConfig, VideoError, VideoSample};

/// Returns true if an `ffmpeg` binary is callable on the PATH.
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

/// Extract (or reuse cached) frames for `sample`, then decode them into a
/// normalized patch-token tensor of length `cfg.tokens_per_clip() * cfg.patch_dim()`.
pub fn clip_to_tensor(
    sample: &VideoSample,
    cfg: FrameConfig,
    cache_dir: &Path,
) -> Result<Vec<f32>, VideoError> {
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

/// Like [`clip_to_tensor`], but returns the decoded RGB frames themselves
/// (used to build prototype/average clips rather than training tensors).
pub fn clip_to_frames(
    sample: &VideoSample,
    cfg: FrameConfig,
    cache_dir: &Path,
) -> Result<Vec<RgbImage>, VideoError> {
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
        .map(|idx| decode_frame(&frames[idx], cfg.frame_size))
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

    let size = cfg.frame_size;
    // Sample roughly `num_frames` frames spread across the whole clip.
    let fps = if sample.duration > 0.1 {
        (cfg.num_frames as f64 / sample.duration).max(0.1)
    } else {
        2.0
    };
    let vf = format!(
        "scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size},fps={fps:.4}"
    );
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
        let img = decode_frame(&frames[idx], cfg.frame_size)?;
        patchify_into(&img, cfg, &mut tensor);
    }
    Ok(tensor)
}

/// Decode a single cached JPEG to an RGB image of exactly `size`×`size`.
fn decode_frame(path: &Path, size: usize) -> Result<RgbImage, VideoError> {
    let img = image::open(path)
        .map_err(|source| VideoError::Decode {
            path: path.to_path_buf(),
            source,
        })?
        .to_rgb8();
    if img.width() as usize == size && img.height() as usize == size {
        Ok(img)
    } else {
        Ok(image::imageops::resize(
            &img,
            size as u32,
            size as u32,
            FilterType::Triangle,
        ))
    }
}

/// Split one frame into row-major patches and append normalized RGB values
/// (scaled to roughly [-1, 1]) to `out`.
fn patchify_into(img: &RgbImage, cfg: FrameConfig, out: &mut Vec<f32>) {
    let side = cfg.patches_per_side();
    let p = cfg.patch_size as u32;
    for gy in 0..side as u32 {
        for gx in 0..side as u32 {
            for ly in 0..p {
                for lx in 0..p {
                    let px = img.get_pixel(gx * p + lx, gy * p + ly);
                    for channel in 0..3 {
                        out.push(px[channel] as f32 / 127.5 - 1.0);
                    }
                }
            }
        }
    }
}

/// Evenly spaced indices into `available` items (with repetition if scarce).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_evenly_spaced_indices() {
        assert_eq!(pick_indices(10, 1), vec![0]);
        assert_eq!(pick_indices(5, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(pick_indices(9, 5), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn repeats_when_too_few_frames() {
        assert_eq!(pick_indices(1, 4), vec![0, 0, 0, 0]);
        assert_eq!(pick_indices(2, 4), vec![0, 0, 1, 1]);
    }

    #[test]
    fn patchify_produces_expected_length() {
        let cfg = FrameConfig {
            num_frames: 1,
            frame_size: 4,
            patch_size: 2,
        };
        let img = RgbImage::new(4, 4);
        let mut out = Vec::new();
        patchify_into(&img, cfg, &mut out);
        assert_eq!(out.len(), cfg.num_patches() * cfg.patch_dim());
        // A black pixel maps to -1.0 under the normalization.
        assert!(out.iter().all(|&v| (v + 1.0).abs() < 1e-6));
    }
}
