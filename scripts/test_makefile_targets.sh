#!/usr/bin/env bash
set -euo pipefail

targets=(help build test lint tui tui-test check ci clean)

for target in "${targets[@]}"; do
  make -n "$target" >/dev/null
done

echo "Makefile targets validated."
