#!/usr/bin/env bash
set -euo pipefail

targets=(help fmt lint audit test build check ci c-build c-test)

for target in "${targets[@]}"; do
  make -n "$target" >/dev/null
done

echo "Makefile targets validated."
