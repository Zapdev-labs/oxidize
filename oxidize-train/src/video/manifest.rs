use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use walkdir::WalkDir;

use super::VideoError;

/// A single clip paired with the metadata used to derive its label.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoSample {
    pub path: PathBuf,
    pub id: String,
    pub username: String,
    pub view_count: u64,
    pub like_count: u64,
    pub duration: f64,
    pub title: String,
}

impl VideoSample {
    /// like/view ratio, guarded against divide-by-zero.
    pub fn engagement_ratio(&self) -> f64 {
        if self.view_count == 0 {
            0.0
        } else {
            self.like_count as f64 / self.view_count as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
struct MetaEntry {
    username: String,
    view_count: u64,
    like_count: u64,
    duration: f64,
    title: String,
}

/// Scan `data_root` for `*_metadata.json` files and the `*.mp4` clips they
/// describe, matching clips to metadata by the numeric id in their filename.
pub fn build_manifest(data_root: &Path) -> Result<Vec<VideoSample>, VideoError> {
    let index = load_metadata_index(data_root)?;
    let mut samples = Vec::new();

    for entry in WalkDir::new(data_root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if !is_video_file(path) {
            continue;
        }
        let Some(id) = id_from_filename(path) else {
            continue;
        };
        if let Some(meta) = index.get(&id) {
            samples.push(VideoSample {
                path: path.to_path_buf(),
                id,
                username: meta.username.clone(),
                view_count: meta.view_count,
                like_count: meta.like_count,
                duration: meta.duration,
                title: meta.title.clone(),
            });
        }
    }

    samples.sort_by(|a, b| a.path.cmp(&b.path));
    if samples.is_empty() {
        return Err(VideoError::NoSamples(data_root.to_path_buf()));
    }
    Ok(samples)
}

fn load_metadata_index(data_root: &Path) -> Result<HashMap<String, MetaEntry>, VideoError> {
    let mut index = HashMap::new();

    for entry in WalkDir::new(data_root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if !is_metadata_file(path) {
            continue;
        }

        let text = std::fs::read_to_string(path).map_err(|source| VideoError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let value: Value = serde_json::from_str(&text).map_err(|source| VideoError::Metadata {
            path: path.to_path_buf(),
            source,
        })?;

        let username = value
            .get("username")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();

        let Some(videos) = value.get("videos").and_then(Value::as_array) else {
            continue;
        };

        for video in videos {
            let Some(id) = json_id(video.get("id")) else {
                continue;
            };
            index.insert(
                id,
                MetaEntry {
                    username: username.clone(),
                    view_count: json_u64(video.get("view_count")),
                    like_count: json_u64(video.get("like_count")),
                    duration: json_f64(video.get("duration")),
                    title: video
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                },
            );
        }
    }

    Ok(index)
}

fn is_metadata_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.ends_with("_metadata.json"))
        .unwrap_or(false)
}

fn is_video_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("mp4" | "mov" | "webm" | "mkv" | "m4v")
    )
}

/// Pull the numeric id out of names like `001_7325519850367913224.mp4`.
fn id_from_filename(path: &Path) -> Option<String> {
    let stem = path.file_stem().and_then(|s| s.to_str())?;
    let candidate = stem.rsplit('_').next().unwrap_or(stem);
    let digits: String = candidate.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        Some(digits)
    }
}

fn json_id(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        _ => None,
    }
}

fn json_u64(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Number(n)) => n
            .as_u64()
            .or_else(|| n.as_f64().map(|f| f.max(0.0) as u64))
            .unwrap_or(0),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0),
        _ => 0,
    }
}

fn json_f64(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_id_from_tiktok_filename() {
        let id = id_from_filename(Path::new("001_7325519850367913224.mp4"));
        assert_eq!(id.as_deref(), Some("7325519850367913224"));
    }

    #[test]
    fn extracts_id_without_prefix() {
        let id = id_from_filename(Path::new("7325519850367913224.mp4"));
        assert_eq!(id.as_deref(), Some("7325519850367913224"));
    }

    #[test]
    fn recognizes_video_and_metadata_files() {
        assert!(is_video_file(Path::new("a/b/clip.mp4")));
        assert!(!is_video_file(Path::new("a/b/clip.txt")));
        assert!(is_metadata_file(Path::new("x/cellow111_metadata.json")));
        assert!(!is_metadata_file(Path::new("x/cellow111.json")));
    }

    #[test]
    fn parses_engagement_ratio() {
        let sample = VideoSample {
            path: PathBuf::new(),
            id: "1".into(),
            username: "u".into(),
            view_count: 100,
            like_count: 25,
            duration: 5.0,
            title: String::new(),
        };
        assert!((sample.engagement_ratio() - 0.25).abs() < 1e-9);
    }
}
