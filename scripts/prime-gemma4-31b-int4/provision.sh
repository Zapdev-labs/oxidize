#!/usr/bin/env bash
# Provision 2× H100 on Prime Intellect for Gemma 4 31B INT4 serving.
# Docs: https://docs.primeintellect.ai/cli-reference/provision-gpu
set -euo pipefail

NAME="${PRIME_POD_NAME:-gemma4-31b-int4}"
GPU_TYPE="${PRIME_GPU_TYPE:-H100_80GB}"
GPU_COUNT="${PRIME_GPU_COUNT:-2}"
DISK_GB="${PRIME_DISK_GB:-500}"

if ! command -v prime >/dev/null 2>&1; then
  echo "Install Prime CLI: uv tool install -U prime && prime login"
  exit 1
fi

echo "==> GPU availability (filter: ${GPU_TYPE} x${GPU_COUNT})"
prime availability list --gpu-type "$GPU_TYPE" --gpu-count "$GPU_COUNT" || true

echo ""
echo "==> Creating pod '${NAME}' (${GPU_COUNT}× ${GPU_TYPE}, ${DISK_GB}GB disk)"
echo "    Attach a persistent disk for HF cache:"
echo "      prime disks list"
echo "      prime pods create --gpu-type ${GPU_TYPE} --gpu-count ${GPU_COUNT} --disks <disk-id>"

prime pods create \
  --name "$NAME" \
  --gpu-type "$GPU_TYPE" \
  --gpu-count "$GPU_COUNT" \
  --disk-size "$DISK_GB" \
  --image ubuntu_22_cuda_12

echo ""
echo "After provision:"
echo "  prime pods status <pod-id>"
echo "  prime pods ssh <pod-id>"
echo "  On pod: bash scripts/prime-gemma4-31b-int4/setup-node.sh"
