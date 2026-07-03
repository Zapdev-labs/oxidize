#!/usr/bin/env bash
set -euo pipefail

source_file="oxidize-cpp/src/model_llama.cpp"

if sed -n '/Logits LlamaModel::forward_batched/,/Logits LlamaModel::final_head/p' "$source_file" |
   rg -q 'cfg\.context_size'; then
  echo "forward_batched must use kv_context_ for KV cache layer stride" >&2
  exit 1
fi

sed -n '/Logits LlamaModel::forward_batched/,/Logits LlamaModel::final_head/p' "$source_file" |
  rg -q 'kv_context_'

echo "forward_batched KV stride validated."
