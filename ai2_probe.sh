#!/usr/bin/env bash
set -u
sshpass -e ssh -vvv -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/oxidize_ai2_known_hosts -o ConnectTimeout=10 ai-2@192.168.1.152 'hostname; whoami; df -h /data 2>/dev/null || df -h .; free -h; python3 --version; command -v cargo || true; command -v hf || true; command -v git || true' > /tmp/ai2_probe.out 2> /tmp/ai2_probe.err
status=$?
echo "$status" > /tmp/ai2_probe.status
exit "$status"
