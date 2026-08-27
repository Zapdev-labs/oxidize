# oxidize-tui

**Domain:** Terminal UI for the oxidize inference engine (OpenTUI + React, run by Bun)

## OVERVIEW
`oxidize-tui` is a full-screen TUI front end. It does **not** link the engine — it drives
`oxidize serve` as a child process and talks to it over the OpenAI-compatible HTTP API, so
streaming, cancellation, `/metrics` and `/v1/models` all come from the server that already
exists. The only thing it reimplements is a **pure-TypeScript GGUF header reader**, so the
model browser can show arch/quant/params for a 300 GB file in milliseconds and still works
when no engine binary has been built yet.

Four views: `chat`, `models`, `monitor`, `logs`. One command palette (`ctrl+k`).

## STRUCTURE
```
oxidize-tui/
├── package.json      # bin: oxidize-tui; scripts: start, dev, typecheck, test, build
├── tsconfig.json     # jsxImportSource: @opentui/react
└── src/
    ├── index.tsx     # arg parsing, renderer bootstrap, signal handling
    ├── App.tsx       # layout + ALL keyboard routing (single useKeyboard)
    ├── theme.ts      # colours + glyphs — the whole visual system
    ├── engine/
    │   ├── binary.ts   # locate the oxidize binary (env → target/ → PATH → cargo run)
    │   ├── models.ts   # scan model dirs, collapse split shards, enrich headers
    │   ├── gguf.ts     # GGUF v2/v3 header reader, incl. the AL quant family
    │   ├── server.ts   # spawn `oxidize serve`, pump output, poll /readyz
    │   ├── api.ts      # /v1/chat/completions SSE streaming
    │   └── metrics.ts  # Prometheus /metrics scrape
    ├── state/
    │   ├── store.ts    # immutable snapshot + useSyncExternalStore
    │   └── actions.ts  # the ONLY module that touches the engine
    ├── components/     # Chrome, ChatView, ModelsView, MonitorView, LogsView,
    │                   # Palette, Prompt, Markdown, primitives
    └── util/format.ts  # bytes/params/ms/fuzzy
```

## INVARIANTS
- **`actions.ts` owns all side effects.** Components read the store and call actions; they
  never spawn processes or fetch.
- **Store selectors must be referentially stable.** `useStore` memoises per snapshot; a
  selector returning a fresh object without that memo would spin `useSyncExternalStore`.
- **Token deltas are coalesced to one store write per 40 ms.** A fast model otherwise
  re-renders the transcript hundreds of times a second.
- **`ctrl+*` is global, bare keys are view-local.** The chat input is always focused, so
  single-key bindings are gated on `view !== "chat"`.
- **Rows must not wrap.** Table/status rows are width-budgeted and drop columns or hints
  right-to-left; `src/ui.test.tsx` asserts no frame line exceeds the terminal width.
- **Large GGUF arrays are walked, not retained** (`> 512` elements are summarised) —
  otherwise a tokenizer vocab would sit in memory for every scanned model.

## COMMANDS
```bash
cd oxidize-tui
bun install
bun run start                     # attach/scan and launch
bun run start -- --api http://127.0.0.1:8080   # attach to a running server
bun run start -- model.gguf --backend cuda --ctx-size 8192
bun run typecheck
bun test                          # engine unit tests + rendered-frame tests
bun run build                     # single-file binary → dist/oxidize-tui
```

## TESTING
`src/ui.test.tsx` renders the real `App` through `@opentui/react/test-utils` and asserts on
`captureCharFrame()` output — chrome, transcript, table columns, palette filtering, key
routing, and responsive behaviour at 76 cols. That harness is also the fastest way to *see*
a layout change: render a view and `console.log` the frame.

## GOTCHAS
- `<ascii-font>` takes `color`, not `fg`.
- `InputProps["onSubmit"]` is typed `(v: string | SubmitEvent)`; coerce via the ref.
- Do not pass `value` to `<input>` — the renderable owns its buffer; use `onInput`.
- The engine's `serve` subcommand is rewritten internally to `--serve-api --api-only`;
  pass-through flags are listed in `oxidize-cli/src/main/command_rewrite.rs`.
- Model load has **no timeout** — a 300 GB GGUF takes minutes; the user aborts with `esc`.
