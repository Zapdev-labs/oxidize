use std::fs;
use std::path::PathBuf;

fn workspace_readme_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("llamas-core should have workspace root parent")
        .join("README.md")
}

#[test]
fn readme_includes_architecture_section() {
    let path = workspace_readme_path();
    let readme = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));

    for required in [
        "## Architecture",
        "llamas-core",
        "llamas-cli",
        "llamas-server",
        "llamas-py",
        "llamas-quantize",
        "input prompt -> interface crate -> `llamas-core`",
    ] {
        assert!(
            readme.contains(required),
            "README architecture docs missing required text: {required}"
        );
    }
}

#[test]
fn readme_includes_quantization_guide() {
    let path = workspace_readme_path();
    let readme = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));

    for required in [
        "#### Quantization guide",
        "Use `F16` for a low-risk size reduction",
        "Run inference/perplexity checks on representative prompts",
        "--target Q4_0",
    ] {
        assert!(
            readme.contains(required),
            "README quantization guide missing required text: {required}"
        );
    }
}
