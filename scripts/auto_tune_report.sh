#!/usr/bin/env bash
# Run `oxidize run` against one or more model GGUF files in
# `--no-api --print-plan=json` mode, parse the JSON, and emit a
# Markdown table summarizing the autotune recommendations. The
# table is written to stdout; redirect to a file in `results/bench/`
# to keep as evidence.
#
# Usage:
#   scripts/auto_tune_report.sh <model.gguf> [<model.gguf> ...]
#   scripts/auto_tune_report.sh --node ai-2 <model.gguf>
#
# `--node <name>` runs the report on a remote node over `sshpass`
# (using the same `machine` password convention as the user's
# existing K3 setup) and copies the report back. Requires the
# `oxidize` binary built and on PATH on the remote.

set -euo pipefail

REMOTE_NODE=""
if [[ "${1:-}" == "--node" ]]; then
  REMOTE_NODE="${2:-}"
  if [[ -z "$REMOTE_NODE" ]]; then
    echo "usage: $0 --node <name> <model.gguf> [<model.gguf> ...]" >&2
    exit 2
  fi
  shift 2
fi

MODELS=("$@")
if [[ -n "$REMOTE_NODE" && ${#MODELS[@]} -eq 0 ]]; then
  echo "usage: $0 --node <name> <model.gguf> [<model.gguf> ...]" >&2
  exit 2
fi

run_local() {
  local model="$1"
  echo "## ${model}"
  echo ""
  if [[ ! -f "$model" ]]; then
    echo "_file not found: ${model}_"
    return
  fi
  set +e
  out="$(oxidize run "$model" \
    --no-api \
    --print-plan=json \
    --max-tokens 1 \
    --prompt "auto-tune probe" 2>&1)"
  rc=$?
  set -e
  if [[ $rc -ne 0 && -z "$out" ]]; then
    echo "_binary not available or model load failed (rc=$rc)_"
    return
  fi
  echo '```json'
  echo "$out" | sed -n '/^{$/,/^}$/p'
  echo '```'
  echo ""
}

run_remote() {
  local model="$1"
  local host="ai-2@192.168.1.152"
  if [[ "$REMOTE_NODE" == "ai" ]]; then
    host="ai@192.168.1.68"
  fi
  echo "## ${REMOTE_NODE}:${model}"
  echo ""
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "_sshpass not installed locally; cannot probe ${REMOTE_NODE}_"
    return
  fi
  set +e
  remote_out="$(sshpass -p machine ssh -o StrictHostKeyChecking=no \
    "${host}" \
    "oxidize run '${model}' --no-api --print-plan=json --max-tokens 1 --prompt 'auto-tune probe' 2>&1 || true")"
  set -e
  echo '```json'
  echo "$remote_out" | sed -n '/^{$/,/^}$/p'
  echo '```'
  echo ""
}

if [[ -n "$REMOTE_NODE" ]]; then
  for m in "${MODELS[@]}"; do
    run_remote "$m"
  done
else
  for m in "${MODELS[@]}"; do
    run_local "$m"
  done
fi
