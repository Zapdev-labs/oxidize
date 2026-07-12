#!/usr/bin/env bash
# Dual-NUMA oxidize-c inference for Qwen3.5 9B self-train on ai@192.168.1.122.
# node0:8092 + node1:8093 + round-robin LB on :11434
set -euo pipefail

HOST="${OXIDIZE_AI_HOST:-ai@192.168.1.122}"
PASS="${OXIDIZE_AI_PASS:-machine}"
BIN=~/oxidize-c-fast/oxidize-c
MODEL=~/models/gguf/qwen35-9b-self-train-merged-Q4_0.gguf

run_remote() {
  sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$HOST" "$@"
}

case "${1:-start}" in
  start)
    run_remote "bash -s" <<'REMOTE'
set -e
BIN=/home/ai/oxidize-c-fast/oxidize-c
MODEL=/home/ai/models/gguf/qwen35-9b-self-train-merged-Q4_0.gguf
export OMP_DYNAMIC=false OXIDIZE_API_KEY=localdev

pkill -f 'oxidize_lb_11434.py' 2>/dev/null || true
pkill -f 'oxidize-c.*--serve' 2>/dev/null || true
sleep 2

setsid env OMP_NUM_THREADS=24 OMP_DYNAMIC=false OXIDIZE_API_KEY=localdev \
  numactl --cpunodebind=0 --membind=0 \
  "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port 8092 \
  --threads 26 --numa single --ctx 8192 --spec off --draft 0 --temperature 0 \
  >/tmp/qwen35-node0.log 2>&1 < /dev/null &

setsid env OMP_NUM_THREADS=24 OMP_DYNAMIC=false OXIDIZE_API_KEY=localdev \
  numactl --cpunodebind=1 --membind=1 \
  "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port 8093 \
  --threads 26 --numa single --ctx 8192 --spec off --draft 0 --temperature 0 \
  >/tmp/qwen35-node1.log 2>&1 < /dev/null &

if [ ! -f /tmp/oxidize_lb_11434.py ]; then
  cp /home/ai/oxidize_lb.py /tmp/oxidize_lb_11434.py
  sed -i 's/8091/11434/' /tmp/oxidize_lb_11434.py
fi
setsid python3 /tmp/oxidize_lb_11434.py >/tmp/oxidize_lb_11434.log 2>&1 < /dev/null &
echo "started: LB http://0.0.0.0:11434 -> node0:8092 node1:8093"
REMOTE
    ;;
  stop)
    run_remote "pkill -f oxidize_lb_11434.py; pkill -f 'oxidize-c.*--serve'; echo stopped"
    ;;
  status)
    run_remote "ss -lntp | grep -E '11434|8092|8093' || true; pgrep -af 'oxidize-c.*serve|oxidize_lb_11434' || true"
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac
