#!/usr/bin/env bash
# Detect unused Cargo dependencies across the workspace (requires nightly).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if ! rustup toolchain list | grep -q '^nightly'; then
  echo "error: nightly Rust toolchain required for cargo-udeps" >&2
  echo "install with: rustup toolchain install nightly" >&2
  exit 1
fi

if ! command -v cargo-udeps >/dev/null 2>&1; then
  echo "Installing cargo-udeps..."
  cargo install cargo-udeps --locked
fi

cargo +nightly udeps --workspace --all-targets "$@"
