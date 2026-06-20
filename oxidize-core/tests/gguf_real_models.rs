use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use oxidize_core::model_loader::{GgufModelLoader, ModelLoader, load_gguf_llama_cpp_baseline};

const SKIP_MODEL_HINT: &str = "set OXIDIZE_TEST_GGUF_MODELS, or place .gguf files under /run/media/<user>/<mount>/AI (searched recursively, e.g. AI/models/...)";

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

fn collect_gguf_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    if !dir.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files_recursive(&path, out);
        } else if path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
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
            collect_gguf_files_recursive(&ai, &mut paths);
        }
    }
    if paths.is_empty() {
        return None;
    }
    paths.sort();
    paths.dedup();
    Some(paths)
}

fn discovery_max_bytes() -> u64 {
    env::var("OXIDIZE_TEST_GGUF_MAX_BYTES")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(12 * 1024 * 1024 * 1024)
}

fn passes_auto_discovery_sanity(path: &Path, max_bytes: u64) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    if meta.len() > max_bytes {
        return false;
    }
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic).is_ok() && magic == *b"GGUF"
}

fn model_paths_from_discovery_filtered() -> Option<Vec<PathBuf>> {
    let candidates = discover_gguf_under_run_media_ai()?;
    let max_bytes = discovery_max_bytes();
    let filtered: Vec<PathBuf> = candidates
        .into_iter()
        .filter(|path| passes_auto_discovery_sanity(path, max_bytes))
        .collect();
    if filtered.is_empty() {
        None
    } else {
        Some(filtered)
    }
}

fn configured_model_paths_for_loader_tests() -> Option<Vec<PathBuf>> {
    model_paths_from_env().or_else(model_paths_from_discovery_filtered)
}

fn configured_compatibility_model_count() -> usize {
    env::var("OXIDIZE_TEST_GGUF_COMPAT_MIN_MODELS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(100)
}

#[test]
fn real_gguf_models_load_consistently_across_loaders() {
    let Some(model_paths) = configured_model_paths_for_loader_tests() else {
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
    let Some(model_paths) = model_paths_from_env() else {
        eprintln!(
            "skipping GGUF compatibility suite: set OXIDIZE_TEST_GGUF_MODELS with {}+ model paths (not driven by /run/media auto-discovery)",
            configured_compatibility_model_count()
        );
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
    let Some(model_paths) = configured_model_paths_for_loader_tests() else {
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

/// Speculative rejection must roll the KV cache back to the last accepted token.
/// This re-forwards an identical probe token after `rewind_to` and asserts the
/// logits match the first forward. The KV state must be restored exactly for
/// this to hold: on a CUDA build with `OX_GPU_ATTN=1` it covers the
/// device-resident F16 cache via `gpu_kv_rewind` (the Phase-2 fix); on a CPU
/// build it covers the host `KvCache::rewind_to`. A missed device rollback would
/// make the replay attend over a stale/duplicated KV row and the logits would
/// diverge. Env-gated (needs a real GGUF); run on Modal H100 with
/// `--features cuda` + `OX_GPU_ATTN=1`.
#[test]
#[ignore = "needs OXIDIZE_TEST_GGUF_MODELS; run with --features cuda + OX_GPU_ATTN=1 for the device path"]
fn rewind_to_round_trip_is_deterministic_on_real_model() {
    use oxidize_core::inference::{InferenceConfig, InferenceModel};
    use oxidize_core::model::{Model, Session};

    let Some(model_paths) = model_paths_from_env() else {
        eprintln!("skipping rewind determinism test: {SKIP_MODEL_HINT}");
        return;
    };
    let path = &model_paths[0];
    let loader = GgufModelLoader;
    let mapped = loader
        .load(path)
        .unwrap_or_else(|err| panic!("load failed for {}: {err}", path.display()));
    let config = InferenceConfig::from_gguf(&mapped);
    let mut model = InferenceModel::load_from_gguf(&mapped, config, true)
        .unwrap_or_else(|err| panic!("model load failed for {}: {err}", path.display()));

    // A short prompt of low token ids (valid in any non-trivial vocab) plus a
    // probe token we forward, roll back, and re-forward.
    let prompt: [u32; 6] = [1, 2, 3, 4, 5, 6];
    let probe: u32 = 7;

    let mut session = Session::new();
    model
        .forward_many(&prompt, &mut session)
        .expect("prompt forward should succeed");
    let consumed = session.consumed_tokens();
    assert_eq!(consumed, prompt.len());

    let logits1 = model
        .forward(&[probe], &mut session)
        .expect("first probe forward should succeed");
    assert_eq!(session.consumed_tokens(), consumed + 1);

    // Roll model KV cache + session back to before the probe, then re-forward it.
    model.rewind_to(consumed).expect("rewind_to should succeed");
    session.rewind_to(consumed);
    assert_eq!(session.consumed_tokens(), consumed);

    let logits2 = model
        .forward(&[probe], &mut session)
        .expect("replayed probe forward should succeed");
    assert_eq!(logits1.len(), logits2.len(), "logit vector lengths differ");

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    };
    assert_eq!(
        argmax(&logits1),
        argmax(&logits2),
        "argmax diverged after rewind -> KV cache was not restored"
    );
    let max_abs = logits1
        .iter()
        .zip(&logits2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs < 1e-3,
        "logits diverged after rewind (max|delta|={max_abs}); KV rollback incomplete"
    );
}
