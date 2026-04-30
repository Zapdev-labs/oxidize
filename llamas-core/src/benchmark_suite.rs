use std::env;
#[cfg(target_os = "linux")]
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkCase {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerplexityDatasetCase {
    pub name: String,
    pub path: PathBuf,
}

fn parse_model_paths(raw: &str) -> Vec<PathBuf> {
    raw.replace([';', ':'], "\n")
        .lines()
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn case_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("model/{name}"))
        .unwrap_or_else(|| "model/unknown".to_string())
}

fn parse_perplexity_dataset_spec(spec: &str) -> Option<PerplexityDatasetCase> {
    let (name, path) = spec.split_once('=')?;
    let name = name.trim();
    let path = path.trim();
    if name.is_empty() || path.is_empty() {
        return None;
    }

    Some(PerplexityDatasetCase {
        name: name.to_string(),
        path: PathBuf::from(path),
    })
}

fn perplexity_cases_from_raw(
    manifest_dir: &Path,
    raw_datasets: Option<&str>,
) -> Vec<PerplexityDatasetCase> {
    let configured = raw_datasets
        .map(parse_model_paths)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|entry| parse_perplexity_dataset_spec(entry.to_string_lossy().as_ref()))
        .collect::<Vec<_>>();
    if !configured.is_empty() {
        return configured;
    }

    let dataset_root = manifest_dir.join("benches").join("datasets");
    vec![
        PerplexityDatasetCase {
            name: "wikitext2".to_string(),
            path: dataset_root.join("wikitext2.sample.txt"),
        },
        PerplexityDatasetCase {
            name: "ptb".to_string(),
            path: dataset_root.join("ptb.sample.txt"),
        },
        PerplexityDatasetCase {
            name: "c4".to_string(),
            path: dataset_root.join("c4.sample.txt"),
        },
    ]
}

fn loader_vs_llama_cpp_cases_from_raw(
    manifest_dir: &Path,
    raw_paths: Option<&str>,
) -> Vec<BenchmarkCase> {
    let configured = raw_paths.map(parse_model_paths).unwrap_or_default();
    if !configured.is_empty() {
        return configured
            .into_iter()
            .map(|path| BenchmarkCase {
                name: case_name(&path),
                path,
            })
            .collect();
    }

    let fixture = manifest_dir
        .join("tests")
        .join("fixtures")
        .join("valid-v3.gguf");
    vec![BenchmarkCase {
        name: "fixture/valid-v3.gguf".to_string(),
        path: fixture,
    }]
}

pub fn loader_vs_llama_cpp_cases(manifest_dir: &Path) -> Vec<BenchmarkCase> {
    let configured = env::var("LLAMAS_BENCH_GGUF_MODELS").ok();
    loader_vs_llama_cpp_cases_from_raw(manifest_dir, configured.as_deref())
}

pub fn perplexity_dataset_cases(manifest_dir: &Path) -> Vec<PerplexityDatasetCase> {
    let configured = env::var("LLAMAS_BENCH_PPL_DATASETS").ok();
    perplexity_cases_from_raw(manifest_dir, configured.as_deref())
}

pub fn benchmark_text_perplexity(text: &str) -> Option<f64> {
    let tokens = text
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.len() < 2 {
        return None;
    }

    let total_chars = tokens.iter().map(|token| token.len()).sum::<usize>() as f64;
    let mean_token_len = total_chars / tokens.len() as f64;
    let entropy = (mean_token_len + 1.0).ln();
    Some(entropy.exp())
}

#[cfg(target_os = "linux")]
fn parse_vm_rss_kb(status: &str) -> Option<u64> {
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?.split_whitespace().next()?;
        value.parse::<u64>().ok()
    })
}

#[cfg(target_os = "linux")]
pub fn current_process_memory_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    parse_vm_rss_kb(&status).map(|kb| kb * 1024)
}

#[cfg(not(target_os = "linux"))]
pub fn current_process_memory_bytes() -> Option<u64> {
    None
}

pub fn benchmark_memory_delta_bytes<F>(action: F) -> Option<u64>
where
    F: FnOnce(),
{
    let before = current_process_memory_bytes()?;
    action();
    let after = current_process_memory_bytes()?;
    Some(after.saturating_sub(before))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_model_paths_accepts_common_separators() {
        let parsed = parse_model_paths("a.gguf:b.gguf;c.gguf\n d.gguf ");
        assert_eq!(
            parsed,
            vec![
                PathBuf::from("a.gguf"),
                PathBuf::from("b.gguf"),
                PathBuf::from("c.gguf"),
                PathBuf::from("d.gguf"),
            ]
        );
    }

    #[test]
    fn case_name_uses_file_name() {
        let name = case_name(Path::new("/models/llama-3.gguf"));
        assert_eq!(name, "model/llama-3.gguf");
    }

    #[test]
    fn suite_uses_fixture_when_no_env_is_set() {
        let manifest_dir = Path::new("/tmp/llamas-core");
        let cases = loader_vs_llama_cpp_cases_from_raw(manifest_dir, None);
        assert_eq!(
            cases,
            vec![BenchmarkCase {
                name: "fixture/valid-v3.gguf".to_string(),
                path: PathBuf::from("/tmp/llamas-core/tests/fixtures/valid-v3.gguf"),
            }]
        );
    }

    #[test]
    fn suite_uses_configured_models_when_present() {
        let manifest_dir = Path::new("/tmp/llamas-core");
        let cases = loader_vs_llama_cpp_cases_from_raw(manifest_dir, Some("a.gguf:b.gguf"));
        assert_eq!(
            cases,
            vec![
                BenchmarkCase {
                    name: "model/a.gguf".to_string(),
                    path: PathBuf::from("a.gguf"),
                },
                BenchmarkCase {
                    name: "model/b.gguf".to_string(),
                    path: PathBuf::from("b.gguf"),
                },
            ]
        );
    }

    #[test]
    fn parse_perplexity_dataset_spec_requires_name_and_path() {
        assert_eq!(
            parse_perplexity_dataset_spec("wikitext2=/tmp/wiki.txt"),
            Some(PerplexityDatasetCase {
                name: "wikitext2".to_string(),
                path: PathBuf::from("/tmp/wiki.txt"),
            })
        );
        assert_eq!(parse_perplexity_dataset_spec("wikitext2="), None);
        assert_eq!(parse_perplexity_dataset_spec("=/tmp/wiki.txt"), None);
    }

    #[test]
    fn perplexity_suite_defaults_to_standard_dataset_samples() {
        let manifest_dir = Path::new("/tmp/llamas-core");
        let cases = perplexity_cases_from_raw(manifest_dir, None);
        assert_eq!(
            cases,
            vec![
                PerplexityDatasetCase {
                    name: "wikitext2".to_string(),
                    path: PathBuf::from("/tmp/llamas-core/benches/datasets/wikitext2.sample.txt"),
                },
                PerplexityDatasetCase {
                    name: "ptb".to_string(),
                    path: PathBuf::from("/tmp/llamas-core/benches/datasets/ptb.sample.txt"),
                },
                PerplexityDatasetCase {
                    name: "c4".to_string(),
                    path: PathBuf::from("/tmp/llamas-core/benches/datasets/c4.sample.txt"),
                },
            ]
        );
    }

    #[test]
    fn perplexity_suite_uses_configured_datasets_when_present() {
        let manifest_dir = Path::new("/tmp/llamas-core");
        let cases = perplexity_cases_from_raw(
            manifest_dir,
            Some("wikitext2=/data/wiki.txt;ptb=/data/ptb.txt"),
        );
        assert_eq!(
            cases,
            vec![
                PerplexityDatasetCase {
                    name: "wikitext2".to_string(),
                    path: PathBuf::from("/data/wiki.txt"),
                },
                PerplexityDatasetCase {
                    name: "ptb".to_string(),
                    path: PathBuf::from("/data/ptb.txt"),
                },
            ]
        );
    }

    #[test]
    fn text_perplexity_requires_multiple_tokens() {
        assert_eq!(benchmark_text_perplexity("single"), None);
        assert_eq!(benchmark_text_perplexity("two tokens"), Some(5.5));
    }

    #[test]
    fn benchmark_memory_delta_never_underflows() {
        let delta = benchmark_memory_delta_bytes(|| {});
        assert!(delta.is_none_or(|bytes| bytes < (1 << 40)));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_vm_rss_kb_extracts_value() {
        let status = "Name:\ttest\nVmRSS:\t  2048 kB\nVmData:\t  1234 kB\n";
        assert_eq!(parse_vm_rss_kb(status), Some(2048));
        assert_eq!(parse_vm_rss_kb("Name:\ttest\n"), None);
    }
}
