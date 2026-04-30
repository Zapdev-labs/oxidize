use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkCase {
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
}
