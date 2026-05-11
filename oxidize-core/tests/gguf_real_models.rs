use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use oxidize_core::model_loader::{GgufModelLoader, ModelLoader, load_gguf_llama_cpp_baseline};

const SKIP_MODEL_HINT: &str = "set OXIDIZE_TEST_GGUF_MODELS, or place .gguf files under /run/media/<user>/<mount>/AI (removable drives on Linux)";

fn model_paths_from_env() -> Option<Vec<PathBuf>> {
    let raw = env::var("OXIDIZE_TEST_GGUF_MODELS").ok()?;
    let normalized = raw.replace([';', ':'], "\n");
    let paths = normalized
        .lines()
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty() { None } else { Some(paths) }
}

fn discover_gguf_files_in_dir(dir: &Path) -> Option<Vec<PathBuf>> {
    if !dir.is_dir() {
        return None;
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let is_gguf = path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
            is_gguf.then_some(path)
        })
        .collect();
    if paths.is_empty() {
        return None;
    }
    paths.sort();
    Some(paths)
}

fn discover_gguf_under_run_media_ai() -> Option<Vec<PathBuf>> {
    let media = Path::new("/run/media");
    if !media.is_dir() {
        return None;
    }
    let mut paths = Vec::new();
    for user_entry in fs::read_dir(media).ok()?.filter_map(Result::ok) {
        let user_path = user_entry.path();
        if !user_path.is_dir() {
            continue;
        }
        let mount_dirs = match fs::read_dir(&user_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        for mount_entry in mount_dirs.filter_map(Result::ok) {
            let ai = mount_entry.path().join("AI");
            if let Some(mut found) = discover_gguf_files_in_dir(&ai) {
                paths.append(&mut found);
            }
        }
    }
    if paths.is_empty() {
        return None;
    }
    paths.sort();
    paths.dedup();
    Some(paths)
}

fn configured_model_paths() -> Option<Vec<PathBuf>> {
    if let Some(paths) = model_paths_from_env() {
        return Some(paths);
    }
    discover_gguf_under_run_media_ai()
}

fn configured_compatibility_model_count() -> usize {
    env::var("OXIDIZE_TEST_GGUF_COMPAT_MIN_MODELS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(100)
}

#[test]
fn real_gguf_models_load_consistently_across_loaders() {
    let Some(model_paths) = configured_model_paths() else {
        eprintln!("skipping real GGUF integration test: {SKIP_MODEL_HINT}");
        return;
    };

    let loader = GgufModelLoader;
    for path in model_paths {
        assert!(path.exists(), "model path must exist: {}", path.display());
        assert!(
            path.is_file(),
            "model path must be a file: {}",
            path.display()
        );

        let mapped = loader
            .load(&path)
            .unwrap_or_else(|err| panic!("mapped loader failed for {}: {err}", path.display()));
        let baseline = load_gguf_llama_cpp_baseline(&path)
            .unwrap_or_else(|err| panic!("baseline loader failed for {}: {err}", path.display()));

        let parsed = mapped.parsed();
        assert!(parsed.version == 2 || parsed.version == 3);
        assert!(
            parsed.tensor_count > 0,
            "expected tensors in {}",
            path.display()
        );
        assert!(
            !parsed.tensor_infos.is_empty(),
            "expected tensor infos in {}",
            path.display()
        );
        assert_eq!(parsed.tensor_count as usize, parsed.tensor_infos.len());
        assert_eq!(parsed, baseline.parsed());
        assert_eq!(mapped.bytes(), baseline.bytes());

        for tensor in &parsed.tensor_infos {
            assert!(
                tensor.absolute_offset >= parsed.data_section_start,
                "tensor absolute offset should be in data section for {}",
                path.display()
            );
            assert!(
                (tensor.absolute_offset as usize) < mapped.bytes().len(),
                "tensor absolute offset should be in file bounds for {}",
                path.display()
            );
        }
    }
}

#[test]
fn real_gguf_models_compatibility_suite_covers_100_plus_models() {
    let Some(model_paths) = configured_model_paths() else {
        eprintln!("skipping GGUF compatibility suite: {SKIP_MODEL_HINT}");
        return;
    };

    let min_model_count = configured_compatibility_model_count();
    assert!(
        model_paths.len() >= min_model_count,
        "compatibility suite requires at least {min_model_count} models, got {}",
        model_paths.len()
    );

    let loader = GgufModelLoader;
    for path in model_paths {
        assert!(path.exists(), "model path must exist: {}", path.display());
        assert!(
            path.is_file(),
            "model path must be a file: {}",
            path.display()
        );

        loader.load(&path).unwrap_or_else(|err| {
            panic!("compatibility loader failed for {}: {err}", path.display())
        });
    }
}

#[test]
fn real_gguf_models_emit_monotonic_progress_events() {
    let Some(model_paths) = configured_model_paths() else {
        eprintln!("skipping real GGUF integration progress test: {SKIP_MODEL_HINT}");
        return;
    };

    let loader = GgufModelLoader;
    for path in model_paths {
        assert!(path.exists(), "model path must exist: {}", path.display());
        assert!(
            path.is_file(),
            "model path must be a file: {}",
            path.display()
        );

        let mut events = Vec::new();
        let model = loader
            .load_with_progress(&path, |progress| events.push(progress))
            .unwrap_or_else(|err| {
                panic!("loader with progress failed for {}: {err}", path.display())
            });

        assert!(!events.is_empty(), "expected progress events");
        assert_eq!(events.first().map(|event| event.stage), Some("starting"));
        assert_eq!(events.first().map(|event| event.percent), Some(0));
        assert_eq!(events.last().map(|event| event.stage), Some("complete"));
        assert_eq!(events.last().map(|event| event.percent), Some(100));
        assert!(
            events
                .windows(2)
                .all(|pair| pair[0].percent <= pair[1].percent)
        );

        let expected_total = model.bytes().len() as u64;
        assert_eq!(
            events.last().and_then(|event| event.total_bytes),
            Some(expected_total)
        );
        assert_eq!(
            events.last().and_then(|event| event.bytes_processed),
            Some(expected_total)
        );
    }
}
