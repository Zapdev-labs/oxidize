# ULW Notepad: GPU K8s Ports Implementation

Objective: implement GPU Kubernetes mesh planning surfaces in Rust, Go, and pure Python ports.

Skills: omo:ulw-loop for evidence-bound execution; omo:programming for Rust/Go/Python edits.

Success criteria:
1. Rust: `OxidizeClusterSpec` planning derives mesh namespace, pod env, GPU tags, degraded status, and rejects malformed specs. Test: `oxidize-core/src/mesh/k8s.rs::tests::*`. Manual QA: tmux runs cargo test + doc-check grep.
2. Go: `core/mesh` exposes equivalent Kubernetes plan/status helpers with parity naming and validation. Test: `core/mesh/k8s_test.go`. Manual QA: tmux runs go test and small `go test -run` surface.
3. Python: `oxidize_python.core.mesh.k8s` exposes equivalent dataclasses/helpers with pytest coverage. Test: `core/mesh/test_k8s.py`. Manual QA: tmux runs pytest and import/print scenario.
4. Regression: existing mesh tests still pass and docs remain aligned.

## Evidence
- RED Rust: .omo/ulw-loop/evidence/gpu-k8s-rust-red.txt
- RED Go: .omo/ulw-loop/evidence/gpu-k8s-go-red.txt
- RED Python: .omo/ulw-loop/evidence/gpu-k8s-python-red.txt
- GREEN Rust targeted: .omo/ulw-loop/evidence/gpu-k8s-rust-green.txt
- GREEN Rust mesh: .omo/ulw-loop/evidence/gpu-k8s-rust-mesh-green.txt
- GREEN Go mesh: .omo/ulw-loop/evidence/gpu-k8s-go-mesh-green.txt
- GREEN Python target/API: .omo/ulw-loop/evidence/gpu-k8s-python-target-green.txt
- Manual Rust tmux: .omo/ulw-loop/evidence/gpu-k8s-rust-tmux.txt | cleanup .omo/ulw-loop/evidence/gpu-k8s-rust-cleanup.txt
- Manual Go tmux: .omo/ulw-loop/evidence/gpu-k8s-go-tmux.txt | cleanup .omo/ulw-loop/evidence/gpu-k8s-go-cleanup.txt
- Manual Python tmux: .omo/ulw-loop/evidence/gpu-k8s-python-tmux.txt | cleanup .omo/ulw-loop/evidence/gpu-k8s-python-cleanup.txt

## Broad Suite Notes
- `sfw cargo test -p oxidize-core` failed in two pre-existing unrelated tests: inference::tests::workspace_buffer_capacities_cover_model_dimensions and sampling::tests::rejects_invalid_sampling_inputs.
- `go test ./...` failed in pre-existing unrelated packages due missing GemvRust, missing cgo symbols, and scripts/manual_qa.sh shape. `go test ./core/mesh` passed.
- `sfw uv run pytest -q` passed: 51 passed, 3 skipped.

## Cleanup Receipts
cleanup: tmux kill-session -t ulw-qa-gpu-k8s-rust; verified session absent
cleanup: tmux kill-session -t ulw-qa-gpu-k8s-go; verified session absent
cleanup: tmux kill-session -t ulw-qa-gpu-k8s-python; verified session absent

## Review Fix Evidence
- Reviewer rejection fixed: Go/Python negative boundary validation, Rust status re-export.
- RED Go review boundary: .omo/ulw-loop/evidence/gpu-k8s-go-review-red.txt
- RED Python review boundary: .omo/ulw-loop/evidence/gpu-k8s-python-review-red.txt
- GREEN Go review: .omo/ulw-loop/evidence/gpu-k8s-go-review-green.txt
- GREEN Python review: .omo/ulw-loop/evidence/gpu-k8s-python-review-green.txt
- GREEN Rust review: .omo/ulw-loop/evidence/gpu-k8s-rust-review-green.txt
- GREEN Rust mesh after review: .omo/ulw-loop/evidence/gpu-k8s-rust-mesh-review-green.txt
- GREEN Go mesh after review: .omo/ulw-loop/evidence/gpu-k8s-go-mesh-review-green.txt
- GREEN Python target after review: .omo/ulw-loop/evidence/gpu-k8s-python-target-review-green.txt
- Final manual Rust: .omo/ulw-loop/evidence/gpu-k8s-rust-tmux-final.txt | cleanup .omo/ulw-loop/evidence/gpu-k8s-rust-cleanup-final.txt
- Final manual Go: .omo/ulw-loop/evidence/gpu-k8s-go-tmux-final.txt | cleanup .omo/ulw-loop/evidence/gpu-k8s-go-cleanup-final.txt
- Final manual Python: .omo/ulw-loop/evidence/gpu-k8s-python-tmux-final.txt | cleanup .omo/ulw-loop/evidence/gpu-k8s-python-cleanup-final.txt
- Unrelated untracked Go files existed before this implementation and remain untouched/untracked.
