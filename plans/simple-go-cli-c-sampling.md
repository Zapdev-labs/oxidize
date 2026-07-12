# Simple Go CLI and C Sampling Parity

## Objective

Make the existing Go command-line interface easier to discover and use interactively, while adding a small, cohesive set of `oxidize-core` sampling features to the standalone C runtime.

## Key Decisions

- Keep the CLI in Go and retain Cobra. OpenTUI is excluded because it would add a second runtime and packaging path for no inference benefit.
- Keep interaction line-oriented instead of adopting a full-screen TUI. The CLI remains friendly over SSH and compatible with redirected input/output.
- Port sampling behavior, not model architecture code: min-p and repetition controls are isolated, user-visible, and testable without a large GGUF.
- Defaults remain behavior-compatible: min-p disabled, penalties zero, penalty window 256.

## Scope

### In

- A clearer Go interactive-chat header, prompt roles, command help, `/clear`, `/settings`, and `/set` for generation controls.
- Terminal styling only when stdout is an interactive terminal; plain deterministic text for tests, pipes, and logs.
- C sampler support for min-p, frequency penalty, presence penalty, and penalty-last-n.
- C CLI flags and help text for those sampler options.
- Focused Go and C tests plus build and real CLI smoke verification.

### Out

- OpenTUI or any TypeScript package.
- Full-screen alternate-buffer UI, mouse handling, or terminal database dependencies.
- Changes to the C HTTP API or WebSocket schema.
- New model architectures, quantization formats, GPU kernels, or speculative algorithms.
- Refactoring existing oversized C translation units.

## Implementation Tasks

## TODOs

- [x] Implement and verify the Go interactive chat contract and UX.
- [x] Implement and verify C sampling parity controls.
- [x] Expose and verify the C CLI sampling flags.

### 1. Lock the Go interactive contract with tests

Files:
- `oxidize-golang/internal/cli/chat_test.go` (new)
- `oxidize-golang/internal/cli/chat.go`

Work:
- Extract the REPL's input reader as an `io.Reader` parameter so tests do not depend on process stdin.
- Add table-driven tests using `strings.NewReader` for `/help`, `/settings`, `/set temperature 0.7`, `/clear`, invalid settings, and `/bye`.
- Assert observable output and updated configuration, not ANSI byte sequences or internal helper calls.
- Ensure EOF exits cleanly and unknown slash commands do not invoke generation.

Acceptance:
- Tests initially fail against the current REPL for missing commands/injection seam.
- Tests run without a GGUF and without an interactive terminal.
- Existing `run`, `chat`, and Cobra tests remain unchanged and passing.

QA:
- Happy path: enter `/settings`, change temperature, inspect settings, then exit.
- Failure path: enter `/set temperature nope`; receive a concise error and remain in the session.

### 2. Implement the simple Go-native chat experience

Files:
- `oxidize-golang/internal/cli/chat.go`
- `oxidize-golang/internal/cli/cli.go`

Work:
- Render a compact startup block containing model name and the most important generation settings.
- Use stable `you>` and `assistant>` roles; keep generation output streaming through the current runtime.
- Implement `/help`, `/settings`, `/set temperature VALUE`, `/set top-p VALUE`, `/set top-k VALUE`, `/set max-tokens VALUE`, `/clear`, and `/bye`.
- `/clear` emits the standard ANSI clear-screen sequence only for an interactive terminal; otherwise it emits a visible separator so redirected output is meaningful.
- Parse settings at the command boundary with range checks matching existing flag semantics. Failed parsing must not mutate the active config.
- Do not add Bubble Tea, Lip Gloss, or other dependencies.

Acceptance:
- A user can discover and adjust core generation settings without restarting the process.
- Plain `oxidize run MODEL PROMPT` output and piped usage do not change.
- `oxidize chat --help` documents the interactive commands.

QA:
- Happy path: build the Go binary, open chat with a fixture or available GGUF, run `/help`, `/settings`, one valid `/set`, then `/bye`.
- Failure path: run `oxidize chat` without a model and confirm the existing concise model-required error remains.

### 3. Lock the C sampling behavior with unit tests

Files:
- `oxidize-c/test_oc.c`
- `oxidize-c/gen.h`

Work:
- Expose a narrow sampler function suitable for deterministic unit tests, or place tests behind a test-only declaration if the public API should remain small.
- Add Given/When/Then cases proving min-p removes tokens below `max_probability * min_p`.
- Add deterministic cases proving frequency penalty scales with occurrence count and presence penalty applies once to seen tokens.
- Add a bounded-window case proving occurrences older than `penalty_last_n` do not affect logits.
- Add zero/default cases proving existing temperature/top-k/top-p behavior remains unchanged.

Acceptance:
- New tests fail against the current sampler for the intended missing behavior.
- Tests use small synthetic logits and fixed RNG seeds; no model file is required.

QA:
- Happy path: a repeated high-logit token loses to an unseen token when configured penalties warrant it.
- Failure boundary: invalid min-p or negative penalties are rejected by CLI parsing before sampling.

### 4. Port the sampling controls into `oxidize-c`

Files:
- `oxidize-c/gen.h`
- `oxidize-c/gen.c`

Work:
- Extend `oc_gen` with `min_p`, `frequency_penalty`, `presence_penalty`, and `penalty_last_n`.
- Apply penalties to a temporary logit view before temperature and rank filtering. Count only the last `penalty_last_n` history tokens; zero means no history penalty.
- Implement min-p after softmax probabilities are available and before top-p truncation, matching the `oxidize-core` threshold definition.
- Route every sampling site, including speculative verification and bonus-token selection, through the same configured sampler.
- Preserve greedy fast behavior when temperature is non-positive and all new controls are disabled.

Acceptance:
- Synthetic sampler tests pass.
- Existing speculative-generation and sequence-scheduler tests pass.
- No new allocation is added to the disabled greedy fast path.

QA:
- Happy path: seeded sampling produces the expected token under combined min-p and repetition controls.
- Boundary: penalty history shorter than the configured window is handled without out-of-bounds access.

### 5. Expose and validate the C CLI flags

Files:
- `oxidize-c/main.c`

Work:
- Add `--min-p`, `--frequency-penalty`, `--presence-penalty`, and `--penalty-last-n` to usage and parsing.
- Validate min-p in `[0,1]`, penalties as finite non-negative floats, and the window as an integer.
- Populate the new `oc_gen` fields for one-shot generation.
- Keep defaults equivalent to current behavior.

Acceptance:
- `oxidize-c --help` or invalid invocation displays the new options.
- Invalid numeric values fail before model loading with the option name in the error.
- Existing command lines behave identically when the new flags are omitted.

QA:
- Happy path: invoke the binary with all new flags and a real model when available.
- Failure path: invoke with `--min-p 1.5` and verify immediate non-zero failure without loading a model.

## Final Verification Wave

- [x] Run Go formatting, package/full tests, build, and real CLI QA.
- [x] Run C tests, build, invalid-input QA, optional model QA, and scoped diff review.

## Guardrails

- Preserve the user's dirty worktree and edit only the scoped source/test files.
- Do not modify Rust sources; `oxidize-core` is the behavioral reference only.
- Do not change server request schemas in this slice.
- Avoid broad formatting of large existing C files.
- Do not claim model-backed inference verification unless a real runnable GGUF is exercised.
