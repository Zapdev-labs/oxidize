#!/usr/bin/env bash
set -euo pipefail

targets=(help fmt lint test build check ci)

for target in "${targets[@]}"; do
  make -n "$target" >/dev/null
done

echo "Makefile targets validated."
