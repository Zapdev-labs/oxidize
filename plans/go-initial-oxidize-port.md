# Initial Go Oxidize Port

## TL;DR
> Summary:      Add an isolated `golang/` module that provides the first Go foundation for Oxidize: deterministic CLI behavior, OpenAI-compatible HTTP placeholder endpoints, auth parity, and a GGUF metadata parser foundation. Keep the Rust workspace intact, verify with TDD plus CLI/server QA evidence, then push a branch and open a draft PR.
> Deliverables:
> - `golang/` Go module at `github.com/Zapdev-labs/oxidize/golang`
> - `cmd/oxidize` Go CLI with `run`, `list`, `serve`, and legacy `--prompt`
> - Go HTTP server for health, OpenAPI, metrics, models, chat completions, completions, and embeddings
> - Deterministic no-model generation contract matching Rust placeholder behavior
> - API-key auth compatible with `OXIDIZE_API_KEY`, `x-api-key`, and `Authorization: Bearer`
> - GGUF metadata parser foundation using existing Rust fixtures
> - Go CI workflow, manual QA evidence, pushed branch, and draft PR
> Effort:       Large
> Risk:         Medium - broad new language surface plus release workflow, bounded by no real tensor inference.

## Scope
### Must have
- Create branch `go/initial-oxidize-port` from `master` before implementation.
- Add all Go product code under `golang/`, with module path `github.com/Zapdev-labs/oxidize/golang`.
- Preserve the Rust workspace in [Cargo.toml](../Cargo.toml) and the current Rust CI behavior in [.github/workflows/ci.yml](../.github/workflows/ci.yml).
- Mirror the key Rust CLI surfaces from [oxidize-cli/src/main.rs](../oxidize-cli/src/main.rs): `run`, `serve`, `list`, and the existing no-model `--prompt` test behavior from [oxidize-cli/tests/cli_binary.rs](../oxidize-cli/tests/cli_binary.rs).
- Mirror the key server routes from [oxidize-server/src/app.rs](../oxidize-server/src/app.rs): `/healthz`, `/livez`, `/readyz`, `/metrics`, `/openapi.json`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`.
- Mirror request fields from [oxidize-server/src/schema.rs](../oxidize-server/src/schema.rs), including structured-output fields and `n`/`best_of`.
- Mirror Rust no-model placeholder response rules from [oxidize-server/src/routes/chat.rs](../oxidize-server/src/routes/chat.rs) and [oxidize-server/src/routes/completions.rs](../oxidize-server/src/routes/completions.rs).
- Mirror auth behavior from [oxidize-server/src/auth.rs](../oxidize-server/src/auth.rs): only `/v1/*` is gated, empty/missing env disables auth, accepted headers are `x-api-key` and `Authorization: Bearer`.
- Use Go standard library first: `flag`, `net/http`, `encoding/json`, `httptest`, `testing`, `log/slog`, `crypto/subtle`, `context`, and `os/signal`.
- Add a new Go CI workflow that runs `gofmt`, `go vet`, `go test`, and `go test -race` in `golang/`.
- Capture TDD red/green evidence and manual CLI/server QA under `evidence/`.
- Commit logical changes, push `go/initial-oxidize-port`, and open a draft PR against `master`.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Must not edit existing Rust source, Rust manifests, Rust tests, Dockerfiles, Python bindings, quantizer, training crate, or benchmark scripts.
- Must not add `golang/` to the Cargo workspace.
- Must not claim real Rust feature parity or production inference.
- Must not implement tensor math, tokenizer algorithms, CUDA/Metal/Vulkan/MLX/WebGPU backends, LoRA, DFlash, mesh, paged attention, Hugging Face downloads, SafeTensors parsing, Python bindings, quantization conversion, or training.
- Must not add third-party Go dependencies unless a task explicitly proves the standard library cannot cover it. This plan expects zero third-party Go dependencies.
- Must not commit generated binaries, server logs, local model files, temporary fixtures outside testdata/evidence, or secrets.
- Must not force-push or overwrite an existing remote branch without explicit user approval.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + Go `testing`, `httptest`, subprocess tests, `go vet`, `go test -race`, and bash/curl manual QA
- QA policy: every task has agent-executed scenarios
- Evidence: `evidence/task-<N>-<slug>.<ext>`

## Execution strategy
### Parallel execution waves
> Target 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks to maximize parallelism.

Wave 1 (no dependencies):
- Task 1: Create branch and Go module skeleton
- Task 2: Add OpenAI API schema and response contracts
- Task 3: Add API-key auth package
- Task 4: Add deterministic generation package
- Task 5: Add GGUF metadata parser foundation
- Task 6: Add static OpenAPI and metrics contract package

Wave 2 (after Wave 1):
- Task 7: depends [1, 4, 5]
- Task 8: depends [2, 3, 4, 6]
- Task 9: depends [7, 8]
- Task 10: depends [7, 8, 9]
- Task 11: depends [1, 7, 8, 9]
- Task 12: depends [1, 7, 8, 9, 10, 11]

Wave 3 (after Wave 2):
- Task 13: depends [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

Critical path: Task 1 -> Task 4 -> Task 7 -> Task 9 -> Task 10 -> Task 12 -> Task 13

### Dependency matrix
| Task | Depends on | Blocks | Can parallelize with |
|------|------------|--------|----------------------|
| 1    | none       | 7, 11, 12, 13 | 2, 3, 4, 5, 6 |
| 2    | none       | 8, 13 | 1, 3, 4, 5, 6 |
| 3    | none       | 8, 13 | 1, 2, 4, 5, 6 |
| 4    | none       | 7, 8, 13 | 1, 2, 3, 5, 6 |
| 5    | none       | 7, 13 | 1, 2, 3, 4, 6 |
| 6    | none       | 8, 13 | 1, 2, 3, 4, 5 |
| 7    | 1, 4, 5   | 9, 10, 11, 12, 13 | 8 |
| 8    | 2, 3, 4, 6 | 9, 10, 11, 12, 13 | 7 |
| 9    | 7, 8      | 10, 11, 12, 13 | none |
| 10   | 7, 8, 9   | 12, 13 | 11 |
| 11   | 1, 7, 8, 9 | 12, 13 | 10 |
| 12   | 1, 7, 8, 9, 10, 11 | 13 | none |
| 13   | 1-12      | none | none |

## Todos
> Implementation + Test = ONE task. Never separate.
> Every task MUST have: References + Acceptance Criteria + QA Scenarios + Commit.

- [ ] 1. Create branch and Go module skeleton

  What to do: Start from `master`, ensure the worktree contains only the plan file as an expected planner artifact or is otherwise clean, create branch `go/initial-oxidize-port`, and add the Go module skeleton. Use module path `github.com/Zapdev-labs/oxidize/golang`. Create `golang/go.mod`, `golang/README.md`, `golang/internal/buildinfo/buildinfo.go`, and `golang/internal/buildinfo/buildinfo_test.go`. The buildinfo package should expose name/version/module constants only; no product behavior yet. Include the plan file in the first commit if it is uncommitted.
  Must NOT do: Do not touch `Cargo.toml`, Rust crates, Dockerfiles, existing CI, or add third-party dependencies.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [7, 11, 12, 13] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  [Cargo.toml](/home/dih/oxidize-3/oxidize/Cargo.toml:1) - Rust workspace must remain unchanged.
  - Pattern:  [Makefile](/home/dih/oxidize-3/oxidize/Makefile:16) - existing local validation commands are Rust-only and must keep working.
  - Pattern:  [.github/workflows/ci.yml](/home/dih/oxidize-3/oxidize/.github/workflows/ci.yml:33) - existing CI gates Rust fmt/lint/audit/test/build.
  - External: `https://go.dev/doc/modules/layout` - Go module/package layout guidance.
  - External: `https://go.dev/doc/modules/managing-dependencies` - Go module dependency management.

  Acceptance criteria (agent-executable only):
  - [ ] `git branch --show-current` prints exactly `go/initial-oxidize-port`.
  - [ ] `test -f golang/go.mod && grep -qx 'module github.com/Zapdev-labs/oxidize/golang' golang/go.mod`.
  - [ ] `cd golang && go test ./...` exits 0.
  - [ ] `git diff -- Cargo.toml Makefile oxidize-core oxidize-cli oxidize-server oxidize-quantize oxidize-py oxidize-train` prints no diff.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: module skeleton builds
    Tool:     bash
    Steps:    mkdir -p evidence && cd golang && go test ./... -v | tee ../evidence/task-1-module.txt
    Expected: output contains "PASS" and no package imports outside the standard library
    Evidence: evidence/task-1-module.txt

  Scenario: Rust workspace untouched
    Tool:     bash
    Steps:    git diff -- Cargo.toml Makefile oxidize-core oxidize-cli oxidize-server oxidize-quantize oxidize-py oxidize-train | tee evidence/task-1-rust-untouched.diff && test ! -s evidence/task-1-rust-untouched.diff
    Expected: diff file is empty
    Evidence: evidence/task-1-rust-untouched.diff
  ```

  Commit: YES | Message: `chore(go): initialize Go port module` | Files: [plans/go-initial-oxidize-port.md, golang/go.mod, golang/README.md, golang/internal/buildinfo/buildinfo.go, golang/internal/buildinfo/buildinfo_test.go]

- [ ] 2. Add OpenAI API schema and response contracts

  What to do: Create `golang/internal/api` with request structs, response structs/builders, and tests. Match the Rust schema fields for chat, completions, embeddings, models, stop sequences, response format, guided JSON/schema/regex/choice, `stream`, `max_tokens`, `max_completion_tokens`, sampling knobs, `seed`, `n`, `best_of`, and `echo`. Add response builders for chat completion, chat chunk, text completion, text chunk, models list, embeddings placeholder, unsupported candidate count, model-not-found, invalid API key, and malformed JSON. Write tests first, capture the failing red run, then implement.
  Must NOT do: Do not implement HTTP handlers here. Do not reject unknown optional OpenAI fields unless the Rust server rejects them.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [8, 13] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - API/Type: [oxidize-server/src/schema.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/schema.rs:6) - request field names and optionality.
  - Pattern:  [oxidize-server/src/routes/responses.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/responses.rs:19) - response object shapes and usage fields.
  - Pattern:  [oxidize-server/src/routes/responses.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/responses.rs:229) - unsupported `n`/`best_of` error.
  - Pattern:  [oxidize-server/src/routes/models.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/models.rs:8) - models response shape and default model id.
  - Pattern:  [oxidize-server/src/routes/embeddings.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/embeddings.rs:8) - placeholder embeddings response.
  - External: `https://platform.openai.com/docs/api-reference/chat/create` - OpenAI chat completion API reference.
  - External: `https://platform.openai.com/docs/api-reference/models/list` - OpenAI models list API reference.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-2-api-red.txt && grep -Eq 'FAIL|build failed' evidence/task-2-api-red.txt`.
  - [ ] `cd golang && go test ./internal/api -run 'Test' -v` exits 0.
  - [ ] `cd golang && go test ./internal/api -run 'TestCandidateCountRejectsUnsupportedValues|TestChatResponseFormatJsonObject|TestModelsResponseDefaultID' -v` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: response contract happy path
    Tool:     bash
    Steps:    cd golang && go test ./internal/api -run 'TestChatCompletionResponseShape|TestTextCompletionResponseShape|TestEmbeddingsResponseShape' -v | tee ../evidence/task-2-api.txt
    Expected: all named tests pass and assert OpenAI-compatible JSON keys
    Evidence: evidence/task-2-api.txt

  Scenario: unsupported candidate count
    Tool:     bash
    Steps:    cd golang && go test ./internal/api -run 'TestCandidateCountRejectsUnsupportedValues' -v | tee ../evidence/task-2-api-error.txt
    Expected: test asserts HTTP 400 body contains "oxidize-server currently supports only n=1 and best_of=1"
    Evidence: evidence/task-2-api-error.txt
  ```

  Commit: YES | Message: `feat(go-api): add OpenAI schema contracts` | Files: [golang/internal/api/schema.go, golang/internal/api/responses.go, golang/internal/api/schema_test.go, golang/internal/api/responses_test.go]

- [ ] 3. Add API-key auth package

  What to do: Create `golang/internal/auth` with a small auth config and middleware/helper that reads the expected key from `OXIDIZE_API_KEY`, gates only `/v1/*`, accepts `x-api-key` and `Authorization: Bearer`, and uses `crypto/subtle.ConstantTimeCompare`. Write tests first, capture the failing red run, then implement.
  Must NOT do: Do not gate `/healthz`, `/livez`, `/readyz`, `/metrics`, or `/openapi.json`. Do not log API keys.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [8, 13] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  [oxidize-server/src/auth.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/auth.rs:1) - accepted headers and auth comments.
  - Pattern:  [oxidize-server/src/auth.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/auth.rs:24) - only `/v1/` paths are gated.
  - Pattern:  [oxidize-server/src/auth.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/auth.rs:46) - constant-time comparison intent.
  - External: `https://pkg.go.dev/crypto/subtle#ConstantTimeCompare` - Go constant-time byte comparison.
  - External: `https://pkg.go.dev/net/http` - middleware-compatible handler APIs.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-3-auth-red.txt && grep -Eq 'FAIL|build failed' evidence/task-3-auth-red.txt`.
  - [ ] `cd golang && go test ./internal/auth -v` exits 0.
  - [ ] `cd golang && go test ./internal/auth -run 'TestAuthAcceptsXAPIKey|TestAuthAcceptsBearer|TestAuthSkipsNonV1Routes|TestAuthRejectsMissingKey' -v` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: accepted header forms
    Tool:     bash
    Steps:    cd golang && go test ./internal/auth -run 'TestAuthAcceptsXAPIKey|TestAuthAcceptsBearer' -v | tee ../evidence/task-3-auth.txt
    Expected: both accepted header tests pass
    Evidence: evidence/task-3-auth.txt

  Scenario: missing key rejected only on v1
    Tool:     bash
    Steps:    cd golang && go test ./internal/auth -run 'TestAuthRejectsMissingKey|TestAuthSkipsNonV1Routes' -v | tee ../evidence/task-3-auth-error.txt
    Expected: `/v1/models` without a key is rejected and `/healthz` proceeds
    Evidence: evidence/task-3-auth-error.txt
  ```

  Commit: YES | Message: `feat(go-auth): add API key middleware` | Files: [golang/internal/auth/auth.go, golang/internal/auth/auth_test.go]

- [ ] 4. Add deterministic generation package

  What to do: Create `golang/internal/generate` with deterministic placeholder generation used by both CLI and server. Server text selection must match Rust no-model fallback order: first guided choice, then guided JSON or JSON schema as `{}`, then guided regex literal, then `response_format` output (`json_object` and `json_schema` as `{}`, text as empty string), otherwise empty string. CLI output must expose `oxidize-cli: <prompt>` and two deterministic progress lines to mirror the Rust CLI integration test. Add tests first and capture red evidence.
  Must NOT do: Do not implement sampling, logits, tokenizer algorithms, real inference, random output, or model downloads.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [7, 8, 13] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  [oxidize-server/src/routes/chat.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/chat.rs:205) - no-model chat fallback order.
  - Pattern:  [oxidize-server/src/routes/completions.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/completions.rs:201) - no-model completion fallback order.
  - Pattern:  [oxidize-server/src/schema.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/schema.rs:114) - response format enum behavior.
  - Pattern:  [oxidize-cli/src/main.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/src/main.rs:484) - CLI greeting string.
  - Test:     [oxidize-cli/tests/cli_binary.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/tests/cli_binary.rs:15) - expected no-model CLI output.
  - External: `https://pkg.go.dev/testing` - Go table-driven unit tests.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-4-generate-red.txt && grep -Eq 'FAIL|build failed' evidence/task-4-generate-red.txt`.
  - [ ] `cd golang && go test ./internal/generate -v` exits 0.
  - [ ] `cd golang && go test ./internal/generate -run 'TestServerPlaceholderPriority|TestCLITranscript' -v` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: guided choice wins
    Tool:     bash
    Steps:    cd golang && go test ./internal/generate -run 'TestServerPlaceholderPriority' -v | tee ../evidence/task-4-generate.txt
    Expected: test asserts guided choice returns the first provided choice before JSON/regex fallbacks
    Evidence: evidence/task-4-generate.txt

  Scenario: unsupported real inference rejected by scope
    Tool:     bash
    Steps:    cd golang && go test ./internal/generate -run 'TestNoRealInferenceHooks' -v | tee ../evidence/task-4-generate-error.txt
    Expected: test asserts package has deterministic placeholder behavior only
    Evidence: evidence/task-4-generate-error.txt
  ```

  Commit: YES | Message: `feat(go-generate): add deterministic placeholder generation` | Files: [golang/internal/generate/generate.go, golang/internal/generate/generate_test.go]

- [ ] 5. Add GGUF metadata parser foundation

  What to do: Create `golang/internal/gguf` with a metadata-only parser for GGUF magic/version/counts, metadata key/value entries needed for architecture and file type, tensor directory entries, alignment, and quantization mapping. Tests must read existing fixtures from `oxidize-core/tests/fixtures/valid-v3.gguf`, `invalid-magic.gguf`, `unsupported-version.gguf`, and `invalid-alignment.gguf`. Add a tiny in-package builder only if needed for focused edge cases. Capture red evidence before implementation.
  Must NOT do: Do not parse tensor bytes into Go tensors. Do not memory-map. Do not implement SafeTensors. Do not mutate Rust fixtures.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [7, 13] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - API/Type: [oxidize-core/src/format/gguf.rs](/home/dih/oxidize-3/oxidize/oxidize-core/src/format/gguf.rs:12) - `GgufFile`, tensor info, metadata, alignment, data start.
  - Pattern:  [oxidize-core/src/format/gguf.rs](/home/dih/oxidize-3/oxidize/oxidize-core/src/format/gguf.rs:98) - architecture and quantization helpers.
  - Pattern:  [oxidize-core/src/format/gguf.rs](/home/dih/oxidize-3/oxidize/oxidize-core/src/format/gguf.rs:167) - file type and tensor type quantization mapping.
  - Test:     [oxidize-core/src/model/loader.rs](/home/dih/oxidize-3/oxidize/oxidize-core/src/model/loader.rs:127) - Rust loader tests using `valid-v3.gguf`.
  - Fixture:  `oxidize-core/tests/fixtures/valid-v3.gguf` - valid parser fixture.
  - Fixture:  `oxidize-core/tests/fixtures/invalid-magic.gguf` - invalid magic fixture.
  - External: `https://github.com/ggml-org/ggml/blob/master/docs/gguf.md` - GGUF format specification.
  - External: `https://pkg.go.dev/encoding/binary` - little-endian binary parsing.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-5-gguf-red.txt && grep -Eq 'FAIL|build failed' evidence/task-5-gguf-red.txt`.
  - [ ] `cd golang && go test ./internal/gguf -v` exits 0.
  - [ ] `cd golang && go test ./internal/gguf -run 'TestParseValidV3Fixture|TestRejectInvalidMagic|TestRejectUnsupportedVersion|TestRejectInvalidAlignment' -v` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: valid Rust fixture parses
    Tool:     bash
    Steps:    cd golang && go test ./internal/gguf -run 'TestParseValidV3Fixture' -v | tee ../evidence/task-5-gguf.txt
    Expected: test asserts version 3, tensor count 1, alignment 64, and nonzero data section start
    Evidence: evidence/task-5-gguf.txt

  Scenario: invalid GGUF rejected
    Tool:     bash
    Steps:    cd golang && go test ./internal/gguf -run 'TestRejectInvalidMagic|TestRejectUnsupportedVersion|TestRejectInvalidAlignment' -v | tee ../evidence/task-5-gguf-error.txt
    Expected: tests pass and assert typed parser errors
    Evidence: evidence/task-5-gguf-error.txt
  ```

  Commit: YES | Message: `feat(go-gguf): add metadata parser foundation` | Files: [golang/internal/gguf/gguf.go, golang/internal/gguf/gguf_test.go]

- [ ] 6. Add static OpenAPI and metrics contract package

  What to do: Create `golang/internal/serviceinfo` with static OpenAPI JSON and Prometheus-style metrics text used by the HTTP server. OpenAPI must include `openapi: 3.1.0`, title `oxidize-server API`, routes `/healthz`, `/livez`, `/readyz`, `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, and security schemes `ApiKeyAuth` and `BearerAuth`. Metrics can be static counters/gauges for the first port but must return text/plain and not panic. Write tests first and capture red evidence.
  Must NOT do: Do not add a Prometheus dependency. Do not claim full Rust metrics parity.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [8, 13] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  [oxidize-server/src/openapi.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/openapi.rs:10) - OpenAPI route and auth scheme contract.
  - Pattern:  [oxidize-server/src/app.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/app.rs:51) - `/metrics` and `/openapi.json` route registration.
  - External: `https://spec.openapis.org/oas/v3.1.0` - OpenAPI 3.1 specification.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-6-serviceinfo-red.txt && grep -Eq 'FAIL|build failed' evidence/task-6-serviceinfo-red.txt`.
  - [ ] `cd golang && go test ./internal/serviceinfo -v` exits 0.
  - [ ] `cd golang && go test ./internal/serviceinfo -run 'TestOpenAPIListsPublicRoutes|TestMetricsTextIsStable' -v` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: OpenAPI route list
    Tool:     bash
    Steps:    cd golang && go test ./internal/serviceinfo -run 'TestOpenAPIListsPublicRoutes' -v | tee ../evidence/task-6-openapi.txt
    Expected: test asserts all required paths and both security schemes exist
    Evidence: evidence/task-6-openapi.txt

  Scenario: metrics content type body is stable
    Tool:     bash
    Steps:    cd golang && go test ./internal/serviceinfo -run 'TestMetricsTextIsStable' -v | tee ../evidence/task-6-metrics.txt
    Expected: test asserts metrics text contains `oxidize_requests_total`
    Evidence: evidence/task-6-metrics.txt
  ```

  Commit: YES | Message: `feat(go-server): add static service info contracts` | Files: [golang/internal/serviceinfo/openapi.go, golang/internal/serviceinfo/metrics.go, golang/internal/serviceinfo/serviceinfo_test.go]

- [ ] 7. Add Go CLI `run`, `list`, and legacy `--prompt`

  What to do: Create `golang/cmd/oxidize/main.go` and `golang/internal/cli` for CLI parsing and execution. Support `oxidize --help`, `oxidize --prompt ping`, `oxidize run <model> [prompt]`, `oxidize list`, and errors for missing `run` model. `list` must scan `./models` for `.gguf` files and print a table like the Rust CLI. `run` must parse common flags (`--prompt`, `--model`, `--backend`, `--max-tokens`, `--temperature`, `--chat`) but only use deterministic placeholder output. If the model path exists, optionally parse GGUF metadata through Task 5 and print a concise metadata line; if it does not exist, still run deterministic placeholder output unless the invocation is missing the model argument. Write unit/subprocess tests first and capture red evidence.
  Must NOT do: Do not download Hugging Face models. Do not execute real inference. Do not require a real model for `--prompt` legacy behavior.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [9, 10, 11, 12, 13] | Blocked by: [1, 4, 5]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  [oxidize-cli/src/main.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/src/main.rs:152) - Rust run/serve/list help text and command names.
  - Pattern:  [oxidize-cli/src/main.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/src/main.rs:190) - `.gguf` model listing behavior.
  - Pattern:  [oxidize-cli/src/main.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/src/main.rs:300) - run argument rewrite and required model message.
  - Pattern:  [oxidize-cli/src/main.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/src/main.rs:484) - greeting output.
  - Test:     [oxidize-cli/tests/cli_binary.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/tests/cli_binary.rs:15) - top-level `--prompt ping` expected behavior.
  - External: `https://pkg.go.dev/flag` - standard CLI flag parsing.
  - External: `https://pkg.go.dev/os/exec` - subprocess testing for CLI behavior.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-7-cli-red.txt && grep -Eq 'FAIL|build failed' evidence/task-7-cli-red.txt`.
  - [ ] `cd golang && go test ./internal/cli ./cmd/oxidize -v` exits 0.
  - [ ] `cd golang && go run ./cmd/oxidize --prompt ping` exits 0 and prints `generation progress: 1/2 tokens`, `generation progress: 2/2 tokens`, `oxidize-cli: ping`, and `generation stats: tokens=2 speed=`.
  - [ ] `cd golang && go run ./cmd/oxidize run` exits nonzero and prints `oxidize run requires a model name or local .gguf path` to stderr.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: legacy prompt CLI
    Tool:     bash
    Steps:    cd golang && go run ./cmd/oxidize --prompt ping | tee ../evidence/task-7-cli-run.txt
    Expected: output contains both progress lines, `oxidize-cli: ping`, and `generation stats: tokens=2 speed=`
    Evidence: evidence/task-7-cli-run.txt

  Scenario: run without model fails
    Tool:     bash
    Steps:    cd golang && (go run ./cmd/oxidize run >../evidence/task-7-cli-error.out 2>../evidence/task-7-cli-error.txt; test $? -ne 0) && grep -q 'oxidize run requires a model name or local .gguf path' ../evidence/task-7-cli-error.txt
    Expected: command exits nonzero and stderr contains the exact required-model message
    Evidence: evidence/task-7-cli-error.txt
  ```

  Commit: YES | Message: `feat(go-cli): add run and list commands` | Files: [golang/cmd/oxidize/main.go, golang/internal/cli/cli.go, golang/internal/cli/run.go, golang/internal/cli/list.go, golang/internal/cli/cli_test.go]

- [ ] 8. Add HTTP server route handlers

  What to do: Create `golang/internal/server` using `net/http` and `http.ServeMux`. Add config with host, port, model id default `oxidize-default`, and API key. Register `/healthz`, `/livez`, `/readyz`, `/metrics`, `/openapi.json`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`. Use `httptest` tests for all routes. POST handlers must reject malformed JSON with HTTP 400, unsupported `n`/`best_of` with HTTP 400 and the exact unsupported message, unauthorized `/v1/*` with HTTP 401 and `{"error":"invalid api key"}`, and valid placeholder requests with HTTP 200. Support streaming SSE for `stream: true` with final `[DONE]`.
  Must NOT do: Do not implement `/v1/mesh/chat/completions` in Go. Do not add external routers. Do not add real model state.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [9, 10, 11, 12, 13] | Blocked by: [2, 3, 4, 6]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  [oxidize-server/src/app.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/app.rs:46) - route list and middleware order.
  - Pattern:  [oxidize-server/src/routes/health.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/health.rs:5) - health/liveness/readiness all return 200.
  - Pattern:  [oxidize-server/src/routes/models.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/models.rs:8) - default model list response.
  - Pattern:  [oxidize-server/src/routes/chat.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/chat.rs:222) - placeholder chat streaming chunks and `[DONE]`.
  - Pattern:  [oxidize-server/src/routes/completions.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/completions.rs:218) - placeholder completion streaming chunks and `[DONE]`.
  - Pattern:  [oxidize-server/src/routes/embeddings.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/routes/embeddings.rs:8) - embeddings placeholder.
  - External: `https://pkg.go.dev/net/http` - standard HTTP server package.
  - External: `https://pkg.go.dev/net/http/httptest` - handler tests.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-8-server-red.txt && grep -Eq 'FAIL|build failed' evidence/task-8-server-red.txt`.
  - [ ] `cd golang && go test ./internal/server -v` exits 0.
  - [ ] `cd golang && go test ./internal/server -run 'TestHealthRoutes|TestModelsRoute|TestChatCompletionRoute|TestCompletionsRoute|TestEmbeddingsRoute|TestAuthGate|TestMalformedJSON|TestStreamingDone' -v` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: httptest route contract
    Tool:     bash
    Steps:    cd golang && go test ./internal/server -run 'TestHealthRoutes|TestModelsRoute|TestChatCompletionRoute|TestCompletionsRoute|TestEmbeddingsRoute' -v | tee ../evidence/task-8-server.txt
    Expected: all named route tests pass with HTTP 200 and expected JSON/SSE shapes
    Evidence: evidence/task-8-server.txt

  Scenario: auth and malformed JSON errors
    Tool:     bash
    Steps:    cd golang && go test ./internal/server -run 'TestAuthGate|TestMalformedJSON|TestUnsupportedCandidateCount' -v | tee ../evidence/task-8-server-error.txt
    Expected: tests assert 401 invalid API key, 400 malformed JSON, and 400 unsupported candidate count
    Evidence: evidence/task-8-server-error.txt
  ```

  Commit: YES | Message: `feat(go-server): add OpenAI-compatible placeholder routes` | Files: [golang/internal/server/config.go, golang/internal/server/router.go, golang/internal/server/handlers.go, golang/internal/server/server_test.go]

- [ ] 9. Add CLI `serve` lifecycle

  What to do: Wire `oxidize serve [model] --host --port --model --max-tokens --temperature` to the Go HTTP server. Defaults must be host `127.0.0.1`, port `8080`, model id `oxidize-default`. Use context cancellation and OS signal handling for graceful shutdown. Add subprocess or integration tests that start on `127.0.0.1:0` when testing and verify readiness before shutdown. Capture red evidence first.
  Must NOT do: Do not bind a fixed port in automated tests. Do not leave server processes running. Do not require a model file to serve.

  Parallelization: Can parallel: NO | Wave 2 | Blocks: [10, 11, 12, 13] | Blocked by: [7, 8]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  [oxidize-cli/src/main.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/src/main.rs:164) - `serve` help and examples.
  - Pattern:  [oxidize-cli/src/main.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/src/main.rs:399) - serve argument rewrite and default CPU optimized behavior.
  - API/Type: [oxidize-server/src/cli.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/cli.rs:48) - server host/port/model defaults.
  - Pattern:  [oxidize-server/src/main.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/main.rs:88) - bind listener and serve with shutdown.
  - External: `https://pkg.go.dev/context` - cancellation context.
  - External: `https://pkg.go.dev/os/signal` - signal handling.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-9-serve-red.txt && grep -Eq 'FAIL|build failed' evidence/task-9-serve-red.txt`.
  - [ ] `cd golang && go test ./internal/cli -run 'TestServeCommand' -v` exits 0.
  - [ ] Manual QA command in the happy-path scenario starts `go run ./cmd/oxidize serve --host 127.0.0.1 --port 18080`, observes `/healthz` HTTP 200, posts chat completion, and kills the process.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: real serve command
    Tool:     bash
    Steps:    mkdir -p evidence && (cd golang && go run ./cmd/oxidize serve --host 127.0.0.1 --port 18080 >../evidence/task-9-serve.log 2>&1 & echo $! >../evidence/task-9-serve.pid) && for i in $(seq 1 50); do curl -sf http://127.0.0.1:18080/healthz && break; sleep 0.1; done | tee evidence/task-9-health.txt && curl -sS -X POST http://127.0.0.1:18080/v1/chat/completions -H 'content-type: application/json' -d '{"model":"oxidize-default","messages":[{"role":"user","content":"hi"}],"guided_choice":["hello"]}' | tee evidence/task-9-chat.json && kill "$(cat evidence/task-9-serve.pid)"
    Expected: health curl succeeds, chat JSON contains `"content":"hello"`, and the server process exits after kill
    Evidence: evidence/task-9-chat.json

  Scenario: auth-protected serve command
    Tool:     bash
    Steps:    mkdir -p evidence && (cd golang && OXIDIZE_API_KEY=secret go run ./cmd/oxidize serve --host 127.0.0.1 --port 18081 >../evidence/task-9-auth-serve.log 2>&1 & echo $! >../evidence/task-9-auth-serve.pid) && for i in $(seq 1 50); do curl -sf http://127.0.0.1:18081/healthz && break; sleep 0.1; done >/dev/null && curl -sS -o evidence/task-9-auth-missing.json -w '%{http_code}' http://127.0.0.1:18081/v1/models | tee evidence/task-9-auth-missing.status && curl -sS -H 'Authorization: Bearer secret' http://127.0.0.1:18081/v1/models | tee evidence/task-9-auth-ok.json && kill "$(cat evidence/task-9-auth-serve.pid)"
    Expected: missing-key status file contains `401`, authorized JSON contains `"owned_by":"oxidize"`
    Evidence: evidence/task-9-auth-missing.status
  ```

  Commit: YES | Message: `feat(go-cli): add serve command` | Files: [golang/internal/cli/serve.go, golang/internal/cli/serve_test.go, golang/cmd/oxidize/main.go]

- [ ] 10. Add manual QA evidence script

  What to do: Add `golang/scripts/manual_qa.sh` that runs from repository root, creates `evidence/`, runs CLI prompt QA, CLI list QA using a temporary `models/*.gguf` fixture copy, starts server on `127.0.0.1:18082`, tests health, models, chat, completions, embeddings, streaming `[DONE]`, auth rejection/acceptance, and always cleans up background processes and temporary files via `trap`. Add a shell test or Go test that validates the script has `set -euo pipefail`, uses a trap, and writes all expected evidence filenames. Capture red evidence first.
  Must NOT do: Do not require real models or network access. Do not leave temporary `models/` files behind. Do not overwrite existing `models/` content.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [12, 13] | Blocked by: [7, 8, 9]

  References (executor has NO interview context - be exhaustive):
  - Test:     [oxidize-cli/tests/cli_binary.rs](/home/dih/oxidize-3/oxidize/oxidize-cli/tests/cli_binary.rs:15) - CLI prompt QA expectations.
  - Pattern:  [oxidize-server/src/app.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/app.rs:46) - route list for server QA.
  - Pattern:  [oxidize-server/src/auth.rs](/home/dih/oxidize-3/oxidize/oxidize-server/src/auth.rs:24) - auth QA expectations.
  - Fixture:  `oxidize-core/tests/fixtures/valid-v3.gguf` - safe small fixture for CLI list/model metadata QA.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-10-manual-qa-red.txt && grep -Eq 'FAIL|missing|not found' evidence/task-10-manual-qa-red.txt`.
  - [ ] `bash -n golang/scripts/manual_qa.sh` exits 0.
  - [ ] `bash golang/scripts/manual_qa.sh` exits 0 and creates all expected `evidence/task-10-*` files.
  - [ ] `git status --short -- models` prints no tracked or untracked model fixture residue after the script exits.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: full manual QA script
    Tool:     bash
    Steps:    bash golang/scripts/manual_qa.sh | tee evidence/task-10-manual-qa.log
    Expected: script exits 0, evidence includes CLI, health, models, chat, completions, embeddings, streaming, and auth files
    Evidence: evidence/task-10-manual-qa.log

  Scenario: cleanup after failure path
    Tool:     bash
    Steps:    OXIDIZE_GO_QA_FORCE_FAILURE=1 bash golang/scripts/manual_qa.sh >evidence/task-10-manual-qa-error.log 2>&1; test $? -ne 0; test ! -f evidence/task-10-serve.pid; git status --short -- models | tee evidence/task-10-cleanup.txt
    Expected: script exits nonzero, removes pid file, and leaves no model fixture residue
    Evidence: evidence/task-10-cleanup.txt
  ```

  Commit: YES | Message: `test(go): add manual QA evidence script` | Files: [golang/scripts/manual_qa.sh, golang/internal/cli/manual_qa_script_test.go]

- [ ] 11. Add Go CI workflow and local quality gates

  What to do: Add `.github/workflows/go.yml` with pull_request/push triggers for `master` and path filters covering `golang/**` plus the workflow file. Use `actions/checkout`, `actions/setup-go`, Go version `1.26.x`, working directory `golang`, and run `test -z "$(gofmt -l .)"`, `go vet ./...`, `go test ./...`, and `go test -race ./...`. Add `golang/README.md` command documentation if Task 1 left it skeletal. Capture red evidence by first running at least one gate before the workflow/doc updates.
  Must NOT do: Do not modify the existing Rust CI workflow unless there is no alternative. Do not add Go lint tools outside the standard toolchain.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [12, 13] | Blocked by: [1, 7, 8, 9]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  [.github/workflows/ci.yml](/home/dih/oxidize-3/oxidize/.github/workflows/ci.yml:1) - existing workflow style and target branch `master`.
  - Pattern:  [.github/workflows/ci.yml](/home/dih/oxidize-3/oxidize/.github/workflows/ci.yml:33) - check/build step structure.
  - External: `https://github.com/actions/setup-go` - official Go setup action.
  - External: `https://pkg.go.dev/cmd/gofmt` - Go formatter.
  - External: `https://pkg.go.dev/cmd/vet` - Go vet analyzer.

  Acceptance criteria (agent-executable only):
  - [ ] `test -s evidence/task-11-ci-red.txt`.
  - [ ] `python3 - <<'PY'\nfrom pathlib import Path\ns=Path('.github/workflows/go.yml').read_text()\nfor needle in ['actions/setup-go', 'working-directory: golang', 'go test ./...', 'go test -race ./...', 'go vet ./...']:\n    assert needle in s, needle\nPY` exits 0.
  - [ ] `cd golang && test -z "$(gofmt -l .)" && go vet ./... && go test ./... && go test -race ./...` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: local Go gates
    Tool:     bash
    Steps:    cd golang && test -z "$(gofmt -l .)" && go vet ./... && go test ./... && go test -race ./... | tee ../evidence/task-11-go-gates.txt
    Expected: all Go quality gates exit 0
    Evidence: evidence/task-11-go-gates.txt

  Scenario: workflow contract
    Tool:     bash
    Steps:    python3 - <<'PY' | tee evidence/task-11-workflow.txt
from pathlib import Path
s=Path('.github/workflows/go.yml').read_text()
needles=['actions/checkout', 'actions/setup-go', 'working-directory: golang', 'go vet ./...', 'go test ./...', 'go test -race ./...']
missing=[n for n in needles if n not in s]
assert not missing, missing
print('workflow ok')
PY
    Expected: output is `workflow ok`
    Evidence: evidence/task-11-workflow.txt
  ```

  Commit: YES | Message: `ci(go): add Go port validation` | Files: [.github/workflows/go.yml, golang/README.md]

- [ ] 12. Run full local verification and scope audit

  What to do: Run full Go gates, manual QA script, and Rust preservation smoke checks. Capture evidence for Go tests, manual QA, `cargo fmt --all --check`, `cargo test --workspace --all-targets`, git diff stats, and file list. This task may fix small issues only inside files owned by Tasks 1-11, with tests updated in the same fix. If Rust commands fail for pre-existing environment reasons, capture the exact failure and do not patch Rust.
  Must NOT do: Do not skip manual QA. Do not edit Rust to satisfy this task. Do not stage evidence files unless the repository policy allows evidence artifacts in commits; by default, evidence remains untracked and is summarized in the PR body.

  Parallelization: Can parallel: NO | Wave 2 | Blocks: [13] | Blocked by: [1, 7, 8, 9, 10, 11]

  References (executor has NO interview context - be exhaustive):
  - Command:  [Makefile](/home/dih/oxidize-3/oxidize/Makefile:16) - `cargo fmt --all --check`.
  - Command:  [Makefile](/home/dih/oxidize-3/oxidize/Makefile:25) - `cargo test --workspace --all-targets`.
  - CI:       [.github/workflows/ci.yml](/home/dih/oxidize-3/oxidize/.github/workflows/ci.yml:33) - Rust CI commands.
  - Command:  `golang/scripts/manual_qa.sh` - manual Go CLI/server QA from Task 10.

  Acceptance criteria (agent-executable only):
  - [ ] `cd golang && test -z "$(gofmt -l .)" && go vet ./... && go test ./... && go test -race ./...` exits 0.
  - [ ] `bash golang/scripts/manual_qa.sh` exits 0.
  - [ ] `cargo fmt --all --check` exits 0.
  - [ ] `cargo test --workspace --all-targets` exits 0, or its exact environment failure is captured in `evidence/task-12-rust-test-failure.txt` with no Rust edits.
  - [ ] `git diff --stat master...HEAD | tee evidence/task-12-diff-stat.txt` shows only `plans/`, `golang/`, and `.github/workflows/go.yml`.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: full local verification
    Tool:     bash
    Steps:    mkdir -p evidence && (cd golang && test -z "$(gofmt -l .)" && go vet ./... && go test ./... && go test -race ./...) | tee evidence/task-12-go-full.txt && bash golang/scripts/manual_qa.sh | tee evidence/task-12-manual-qa.txt && cargo fmt --all --check | tee evidence/task-12-rust-fmt.txt && cargo test --workspace --all-targets | tee evidence/task-12-rust-test.txt
    Expected: all commands exit 0
    Evidence: evidence/task-12-go-full.txt

  Scenario: scope audit
    Tool:     bash
    Steps:    git diff --name-only master...HEAD | tee evidence/task-12-files.txt && python3 - <<'PY'
from pathlib import Path
allowed=('plans/', 'golang/', '.github/workflows/go.yml')
bad=[p for p in Path('evidence/task-12-files.txt').read_text().splitlines() if p and not p.startswith(allowed)]
assert not bad, bad
PY
    Expected: no changed files outside `plans/`, `golang/`, and `.github/workflows/go.yml`
    Evidence: evidence/task-12-files.txt
  ```

  Commit: YES | Message: `chore(go): verify initial port foundation` | Files: [only fixes inside files from Tasks 1-11, if any]

- [ ] 13. Push branch and open draft PR

  What to do: Ensure all intended implementation commits are present and no evidence/log/model residue is staged. Push `go/initial-oxidize-port` to `origin`. Create a draft PR against `master` with title `go: initial oxidize port foundation`. PR body must summarize scope, explicitly state no real inference/tensor parity, list verification commands and evidence files, and reference `Plan: plans/go-initial-oxidize-port.md`. If `gh pr create` reports an existing PR for the branch, use `gh pr view` and update the body with `gh pr edit` instead of opening a duplicate. Do not force-push.
  Must NOT do: Do not mark the PR ready for review. Do not include secrets or raw auth tokens in the PR body. Do not push if `git status --short` shows staged/untracked product files outside the planned paths.

  Parallelization: Can parallel: NO | Wave 3 | Blocks: [] | Blocked by: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

  References (executor has NO interview context - be exhaustive):
  - Remote:   `origin` is `https://github.com/Zapdev-labs/oxidize.git`.
  - GitHub:   `gh auth status` was authenticated for account `Jackson57279` with `repo` and `workflow` scopes during planning.
  - Branch:   default branch is `master`.
  - Plan:     `plans/go-initial-oxidize-port.md` - final PR footer reference.

  Acceptance criteria (agent-executable only):
  - [ ] `git branch --show-current` prints `go/initial-oxidize-port`.
  - [ ] `git status --short` shows no staged/untracked product files outside the intended plan, Go, and workflow paths.
  - [ ] `git push -u origin go/initial-oxidize-port` exits 0.
  - [ ] `gh pr view go/initial-oxidize-port --json isDraft,baseRefName,headRefName,title,url` exits 0 and reports `isDraft: true`, `baseRefName: master`, `headRefName: go/initial-oxidize-port`, and title `go: initial oxidize port foundation`.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: draft PR exists
    Tool:     bash
    Steps:    gh pr view go/initial-oxidize-port --json number,isDraft,baseRefName,headRefName,title,url | tee evidence/task-13-pr.json
    Expected: JSON reports draft PR, base `master`, head `go/initial-oxidize-port`, and the planned title
    Evidence: evidence/task-13-pr.json

  Scenario: push/PR residue guard
    Tool:     bash
    Steps:    git status --short | tee evidence/task-13-status.txt && python3 - <<'PY'
from pathlib import Path
allowed=('?? evidence/',)
bad=[]
for line in Path('evidence/task-13-status.txt').read_text().splitlines():
    path=line[3:] if len(line) > 3 else ''
    if path.startswith('evidence/'):
        continue
    if line:
        bad.append(line)
assert not bad, bad
PY
    Expected: only untracked evidence files are allowed after PR creation
    Evidence: evidence/task-13-status.txt
  ```

  Commit: NO | Message: `N/A` | Files: []

## Final verification wave (MANDATORY - after all implementation tasks)
> Runs in PARALLEL. ALL must APPROVE. Surface results to the caller and wait for an explicit "okay" before declaring complete.
- [ ] F1. Plan compliance audit - every task done, every acceptance criterion met
- [ ] F2. Code quality review - diagnostics clean, idioms match, no dead code
- [ ] F3. Real manual QA - every QA scenario executed with evidence captured
- [ ] F4. Scope fidelity - nothing extra shipped beyond Must-Have, nothing Must-NOT-Have introduced

## Commit strategy
- One logical change per commit. Conventional Commits (`<type>(<scope>): <subject>` body + footer).
- Atomic: every commit builds and passes tests on its own.
- No "WIP" / "fix typo squash later" commits on the final branch - clean up before merge.
- Reference the plan file path in the final commit footer: `Plan: plans/go-initial-oxidize-port.md`.

## Success criteria
- All Must-Have shipped; all QA scenarios pass with captured evidence; F1-F4 approved; commit history clean.
