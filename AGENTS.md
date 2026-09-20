# PROJECT KNOWLEDGE BASE

Workspace: C11 LLM inference (`oxidize-c`) + TypeScript TUI (`oxidize-tui`).

## Structure

```
oxidize-c/     # engine, CLI, HTTP API
oxidize-tui/   # OpenTUI/Bun client
```

Default build is `make build` → `oxidize-c/oxidize-c`. TUI drives `oxidize-c serve`. Do not restore Rust/Go/Python/C++ ports as the product.
