#!/usr/bin/env bash
# Start oxidize-server (OpenAI-compatible) and run the codeneedle benchmark
# against LFM2.5-8B-A1B. Small sample because lfm2 prefill is per-token (~20 t/s)
# and each ~11K-token query takes ~9 min.
set +e
cd /home/dih/oxidize
pkill -9 -f oxidize-server >/dev/null 2>&1
sleep 1

./target/release/oxidize-server \
  --host 127.0.0.1 --port 9100 \
  --model ./models/LFM2.5-8B-A1B-Q4_K_M.gguf \
  --model-id LFM2.5-8B-A1B-Q4_K_M \
  --temperature 0 --max-tokens 2048 > /home/dih/oxidize/serve.log 2>&1 &
SRV=$!

ready=0
for i in $(seq 1 40); do
  curl -s -m 2 http://127.0.0.1:9100/v1/models >/dev/null 2>&1 && ready=1 && break
  sleep 1
done
echo "server_ready=$ready"
if [ "$ready" != "1" ]; then tail -20 /home/dih/oxidize/serve.log; kill -9 $SRV 2>/dev/null; exit 1; fi

cd /home/dih/codeneedle
K="${1:-2}"
echo "=== codeneedle run: corpus=http_server model=LFM2.5-8B-A1B-Q4_K_M k=$K ==="
.venv/bin/python bench.py run \
  --corpus http_server \
  --model LFM2.5-8B-A1B-Q4_K_M \
  --base-url http://127.0.0.1:9100 \
  --temperature 0 \
  --max-tokens 1024 \
  --timeout 1800 \
  -k "$K" \
  --skip-preflight \
  --no-fail-fast 2>&1
RC=$?
echo "bench_rc=$RC"

echo "=== results json ==="
cat results/http_server__LFM2.5-8B-A1B-Q4_K_M*.json 2>/dev/null | tail -60

kill -9 $SRV 2>/dev/null
echo "ALL_DONE"
