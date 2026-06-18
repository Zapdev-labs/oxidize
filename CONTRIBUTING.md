# Contributing to oxidize

Thanks for contributing.

## Development setup

1. Install a recent Rust toolchain with `rustup`.
2. Clone the repository and move into it.
3. Build once to verify your environment:

```bash
make build
```

## Workflow

1. Create a focused branch from `master`.
2. Keep each change scoped to one logical task.
3. Prefer small pull requests with clear commit messages.
4. Update docs and tests when behavior changes.

## Quality checks

All checks must pass before a pull request is ready for review. CI will reject changes that fail tests, lint, formatting, or benchmark regression checks.

Run the full local gate:

```bash
make ci
```

At minimum, run these before opening a pull request:

```bash
make test
make lint
make fmt
make udeps   # unused Cargo dependencies (needs nightly + cargo-udeps)
```

For quick validation while iterating:

```bash
make check
```

## Commit guidelines

- Use descriptive, imperative commit messages.
- Avoid mixing unrelated changes in one commit.
- Remove dead code instead of leaving unused paths.

## Pull requests

Every pull request needs a clear, honest markdown description. Write for reviewers who do not have your local context.

### Description requirements

Use proper markdown structure (headings, lists, code blocks). At minimum, include:

- **Summary** — what changed and why, in plain language.
- **Motivation** — the problem this solves or the behavior it improves.
- **Testing** — how you verified the change (commands run, scenarios covered).
- **Follow-ups** — anything intentionally left out of scope.

Be accurate and ethical: do not overstate performance gains, hide known regressions, or omit breaking changes. If something is experimental or incomplete, say so explicitly.

### Tests

- All workspace tests must pass (`make test` / `cargo test --workspace --all-targets`).
- Add or update tests when behavior changes.
- Do not disable, skip, or weaken tests to make CI green without reviewer agreement.

### Benchmarks

Performance-sensitive changes (kernels, quantization, attention, memory layout, backends, generation loops) must include benchmark evidence.

Run criterion benchmarks locally before opening the PR:

```bash
cargo bench -p oxidize-core --bench criterion -- --noplot
python3 scripts/ci_benchmark_regression.py
```

For end-to-end throughput comparisons against llama.cpp:

```bash
scripts/bench_vs_llamacpp.sh --model /path/to/model.gguf --llamacpp-bin /path/to/llama-cli
```

In the PR description, include:

- Which benchmarks you ran and on what hardware.
- Before/after numbers or a short summary of the impact.
- Any expected regressions and why they are acceptable.

CI runs the benchmark job on every pull request; avoid shipping changes that fail regression detection without discussion.
