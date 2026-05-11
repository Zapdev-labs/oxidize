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

Run these before opening a pull request:

```bash
make test
make lint
```

For quick local validation:

```bash
make fmt
make check
```

## Commit guidelines

- Use descriptive, imperative commit messages.
- Avoid mixing unrelated changes in one commit.
- Remove dead code instead of leaving unused paths.

## Pull requests

- Describe what changed and why.
- Include test coverage for the change.
- Note any follow-up work explicitly.
