#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo-udeps >/dev/null 2>&1; then
  echo "cargo-udeps not installed; skipping unused-dependency check"
  exit 0
fi

cargo +nightly udeps --workspace --all-targets
