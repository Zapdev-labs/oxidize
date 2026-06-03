#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVIDENCE_DIR="$ROOT_DIR/evidence"
TMP_DIR="$(mktemp -d)"
PORT=""
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

pick_port() {
  for port in $(seq 18080 18120); do
    local output
    if ! output="$(ss -ltn "( sport = :${port} )" 2>/dev/null)"; then
      printf 'ss failed while probing port %s\n' "${port}" >&2
      return 1
    fi
    line_count="$(printf '%s\n' "${output}" | wc -l | tr -d '[:space:]')"
    if [[ "${line_count}" -le 1 ]]; then
      printf '%s' "${port}"
      return 0
    fi
  done
  return 1
}

mkdir -p "${EVIDENCE_DIR}" "${TMP_DIR}/models"
cp "${ROOT_DIR}/oxidize-core/tests/fixtures/valid-v3.gguf" "${TMP_DIR}/models/valid-v3.gguf"
PORT="$(pick_port)"

(
  cd "${ROOT_DIR}/oxidize-golang"
  go run ./cmd/oxidize --prompt "hello"
) | tee "${EVIDENCE_DIR}/task-10-cli-prompt.txt"

(
  cd "${ROOT_DIR}/oxidize-golang"
  go run ./cmd/oxidize list --models-dir "${TMP_DIR}/models"
) | tee "${EVIDENCE_DIR}/task-10-cli-list.txt"

(
  cd "${ROOT_DIR}/oxidize-golang"
  OXIDIZE_API_KEY=secret go run ./cmd/oxidize serve --host 127.0.0.1 --port "${PORT}" --models-dir "${TMP_DIR}/models"
) >"${EVIDENCE_DIR}/task-10-server.log" 2>&1 &
SERVER_PID="$!"

for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

curl -fsS "http://127.0.0.1:${PORT}/healthz" | tee "${EVIDENCE_DIR}/task-10-health.txt"
curl -fsS -H "x-api-key: secret" "http://127.0.0.1:${PORT}/v1/models" | tee "${EVIDENCE_DIR}/task-10-models.txt"
curl -fsS -H "x-api-key: secret" -H "content-type: application/json" -d '{"model":"valid-v3","messages":[{"role":"user","content":"hi"}]}' "http://127.0.0.1:${PORT}/v1/chat/completions" | tee "${EVIDENCE_DIR}/task-10-chat.txt"
curl -fsS -H "x-api-key: secret" -H "content-type: application/json" -d '{"model":"valid-v3","prompt":"hi","guided_choice":["hello"]}' "http://127.0.0.1:${PORT}/v1/completions" | tee "${EVIDENCE_DIR}/task-10-completions.txt"
curl -fsS -H "x-api-key: secret" -H "content-type: application/json" -d '{"model":"valid-v3","input":"hi"}' "http://127.0.0.1:${PORT}/v1/embeddings" | tee "${EVIDENCE_DIR}/task-10-embeddings.txt"
curl -fsS -N -H "x-api-key: secret" -H "content-type: application/json" -d '{"model":"valid-v3","messages":[{"role":"user","content":"hi"}],"stream":true}' "http://127.0.0.1:${PORT}/v1/chat/completions" | tee "${EVIDENCE_DIR}/task-10-stream.txt"
grep -F "[DONE]" "${EVIDENCE_DIR}/task-10-stream.txt" > "${EVIDENCE_DIR}/task-10-stream-done.txt"
curl -sS -o "${EVIDENCE_DIR}/task-10-auth.txt" -w "%{http_code}\n" "http://127.0.0.1:${PORT}/v1/models" > "${EVIDENCE_DIR}/task-10-auth-status.txt"
auth_status="$(<"${EVIDENCE_DIR}/task-10-auth-status.txt")"
if [[ "${auth_status}" != "401" ]]; then
  printf 'expected 401 from unauthenticated request, got %s\n' "${auth_status}" >&2
  exit 1
fi
