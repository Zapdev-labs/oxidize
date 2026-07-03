#!/usr/bin/env bash
set -euo pipefail

script="scripts/bench-ai-box.sh"
test -x "$script"
bash -n "$script"

require_rg() {
  local pattern="${1:?pattern required}"
  local message="${2:?message required}"
  if ! rg -q "$pattern" "$script"; then
    echo "$message" >&2
    exit 1
  fi
}

require_rg 'small\)' "bench-ai-box missing small model command"
require_rg 'medium\)' "bench-ai-box missing medium model command"
require_rg 'big\)' "bench-ai-box missing big model command"
require_rg 'Meta-Llama-3\.1-8B-Instruct-Q8_0\.gguf' "bench-ai-box missing Llama 3.1 8B path"
require_rg 'Mixtral-8x22B-Instruct-v0\.1\.Q4_K_M-00001-of-00002\.gguf' "bench-ai-box missing Mixtral 8x22B path"
require_rg 'k2-merged-oxidize-Q4_K_M\.gguf' "bench-ai-box missing Kimi/K2 path"
require_rg 'usage: .*small.*medium.*big' "bench-ai-box usage does not advertise full matrix"
require_rg 'rm -rf build' "bench-ai-box should rebuild cleanly on remote"
require_rg 'OXIDIZE_SMALL_TOKENS' "bench-ai-box missing small token override"
require_rg 'OXIDIZE_MEDIUM_TOKENS' "bench-ai-box missing medium token override"

if rg -q '0\.5B|qwen2\.5-0\.5b|qwen05b|qwen-auto|glm\)' "$script"; then
  echo "bench-ai-box still contains stale 0.5B/GLM-only benchmark entries" >&2
  exit 1
fi

echo "bench-ai-box model matrix validated."
