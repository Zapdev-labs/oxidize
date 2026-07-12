#!/usr/bin/env bash
# Provision cheap A100 40GB spot on Prime Intellect for Qwen3.5 9B self-train SFT.
set -euo pipefail

NAME="${PRIME_POD_NAME:-qwen35-self-train}"
GPU_ID="${PRIME_GPU_ID:-724407}"   # A100_40GB spot ~$0.79/hr (datacrunch FIN-02)
DISK_GB="${PRIME_DISK_GB:-250}"
VCPUS="${PRIME_VCPUS:-22}"
MEM_GB="${PRIME_MEM_GB:-120}"

if ! command -v prime >/dev/null 2>&1; then
  echo "Install: uv tool install -U prime && prime login"
  exit 1
fi

echo "==> availability (A100_40GB spot)"
prime availability list --gpu-type A100_40GB --plain 2>&1 | head -12 || true

echo ""
echo "==> creating pod '${NAME}' (id=${GPU_ID}, ${DISK_GB}GB disk, ${VCPUS} vCPU, ${MEM_GB}GB RAM)"
prime pods create \
  --id "$GPU_ID" \
  --name "$NAME" \
  --disk-size "$DISK_GB" \
  --vcpus "$VCPUS" \
  --memory "$MEM_GB" \
  --image ubuntu_22_cuda_12 \
  --yes \
  --plain

echo ""
echo "Next:"
echo "  prime pods list --plain"
echo "  prime pods ssh <pod-id>"
echo "  On pod: git clone https://github.com/Zapdev-labs/oxidize.git ~/oxidize"
echo "          bash ~/oxidize/scripts/prime-qwen35-self-train/setup-node.sh"
