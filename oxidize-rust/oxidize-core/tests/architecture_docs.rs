use std::fs;
use std::path::PathBuf;

fn workspace_readme_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("oxidize-core should have workspace root parent")
        .join("README.md")
}

fn workspace_contributing_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("oxidize-core should have workspace root parent")
        .join("CONTRIBUTING.md")
}

#[test]
fn readme_includes_architecture_section() {
    let path = workspace_readme_path();
    let readme = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));

    for required in [
        "## Architecture",
        "oxidize-core",
        "oxidize-cli",
        "oxidize-server",
        "oxidize-py",
        "oxidize-quantize",
        "input prompt -> interface crate -> `oxidize-core`",
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

#[test]
fn readme_includes_examples_section() {
    let path = workspace_readme_path();
    let readme = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));

    for required in [
        "## Examples",
        "### Basic inference",
        "### Chat completion",
        "### Streaming generation",
        "### Batch processing",
        "### Custom sampling",
        "### Embedding extraction",
    ] {
        assert!(
            readme.contains(required),
            "README examples docs missing required text: {required}"
        );
    }
}

#[test]
fn readme_includes_troubleshooting_guide() {
    let path = workspace_readme_path();
    let readme = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));

    for required in [
        "## Troubleshooting guide",
        "Model path errors",
        "Slow or no GPU acceleration",
        "Server auth failures (`401`)",
        "WASM build failures",
        "Unexpected output quality after quantization",
    ] {
        assert!(
            readme.contains(required),
            "README troubleshooting guide missing required text: {required}"
        );
    }
}

#[test]
fn readme_includes_release_announcement() {
    let path = workspace_readme_path();
    let readme = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));

    for required in [
        "## Release announcement: oxidize 0.1.0",
        "first stable workspace release for local-first LLM workflows in Rust",
        "What this means for early users:",
        "Thank you to everyone testing early builds and sharing feedback.",
    ] {
        assert!(
            readme.contains(required),
            "README release announcement missing required text: {required}"
        );
    }
}

#[test]
fn contributing_guide_includes_required_sections() {
    let path = workspace_contributing_path();
    let guide = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));

    for required in [
        "# Contributing to oxidize",
        "## Development setup",
        "## Workflow",
        "## Quality checks",
        "## Pull requests",
        "make test",
        "make lint",
    ] {
        assert!(
            guide.contains(required),
            "CONTRIBUTING guide missing required text: {required}"
        );
    }
}
