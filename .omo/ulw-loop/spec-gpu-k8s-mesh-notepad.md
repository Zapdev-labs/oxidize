# ULW Notepad: spec-gpu-k8s-mesh

Skills surveyed: omo:ulw-loop used for evidence-bound docs delivery. omo:programming not used because no .rs/.go/.py/.ts code edit is planned. Docs-only change is test-exempt for production code but still gets doc acceptance checks.

Success criteria:
1. happy_path: spec has a grounded current-state + target architecture and no vibe-code marker. Test: rg checks before/after. Manual QA: tmux transcript renders key headings.
2. edge_completeness: spec includes Kubernetes CRD, resource model, scheduling, failure modes, security, observability, and rollout. Test: token checks. Manual QA: CLI section extraction.
3. regression_alignment: spec references existing mesh/backends/paged_attention docs and does not claim current NCCL/RDMA/PlanarQuant/IsoQuant implementation. Test: forbidden/required phrase checks. Manual QA: git diff/stat transcript.


## Final Evidence
- RED: .omo/ulw-loop/evidence/spec-gpu-k8s-red.txt
- GREEN: .omo/ulw-loop/evidence/spec-gpu-k8s-green-final.txt
- Mesh tests: .omo/ulw-loop/evidence/spec-gpu-k8s-mesh-tests-final.txt
- Manual QA: .omo/ulw-loop/evidence/spec-gpu-k8s-tmux-final-clean.txt
- Cleanup: .omo/ulw-loop/evidence/spec-gpu-k8s-cleanup-final-clean.txt
- Reviewer: APPROVE after validate_shard_plan wording fix.

## Findings
- Documentation-only production change; no code test file added. RED/GREEN doc contract checks were used instead.
- Current validate_shard_plan only rejects empty assignments; capability-aware validation remains future work.
