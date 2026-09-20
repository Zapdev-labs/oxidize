# Contributing to oxidize

The product is **C11** (`oxidize-c`) plus a **TypeScript TUI** (`oxidize-tui`). Do not introduce Rust, Go, Python, or C++ as product languages.

```bash
make build
make test
make tui-test
```

C style: C11, `-Wall -Wextra -Werror`, tests co-located under `oxidize-c/tests/`. TUI: `bun test` and `bun run typecheck`. One focused branch and PR per change.
