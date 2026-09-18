#!/usr/bin/env bash
# Stream a Qwen3.6-35B-A3B GGUF the Edge0 way on dih@192.168.1.15:
# mmap experts, prerouter-as-routing, unmerged Recover-LoRA.
set -euo pipefail

HOST="${EDGE0_HOST:-dih@192.168.1.15}"
REPO="${EDGE0_HF_REPO:-Edge0/Edge0-35B-A3B-preview}"
CACHE="${EDGE0_CACHE:-$HOME/.cache/oxidize/edge0}"
PROMPT="${EDGE0_PROMPT:-Write a short haiku about streaming experts from SSD.}"
N_PREDICT="${EDGE0_N_PREDICT:-32}"
THREADS="${EDGE0_THREADS:-16}"
CACHE_MB="${EDGE0_EXPERT_CACHE_MB:-3072}"
CTX="${EDGE0_CTX:-2048}"
DRY_RUN=0
LOCAL=0

usage() {
    cat <<'EOF'
Usage: scripts/edge0_dih.sh [options]

Options:
  --host USER@HOST     SSH destination (default: dih@192.168.1.15)
  --model PATH         GGUF on the remote host
  --cache DIR          Where to store Edge0 prerouter + LoRA
  --local              Run on this machine (skip SSH)
  --dry-run            Print the remote command and exit
  --prompt TEXT        Generation prompt
  --n-predict N        Tokens to generate (default 32)
  --threads N          CPU threads (default 16)
  --expert-cache-mb N  Expert working-set budget (default 3072)
EOF
}

MODEL="${EDGE0_MODEL:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST=${2:?}; shift 2 ;;
        --model) MODEL=${2:?}; shift 2 ;;
        --cache) CACHE=${2:?}; shift 2 ;;
        --prompt) PROMPT=${2:?}; shift 2 ;;
        --n-predict) N_PREDICT=${2:?}; shift 2 ;;
        --threads) THREADS=${2:?}; shift 2 ;;
        --expert-cache-mb) CACHE_MB=${2:?}; shift 2 ;;
        --local) LOCAL=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done

find_model() {
    local candidates=(
        "${MODEL:-}"
        "${EDGE0_MODEL:-}"
        /home/ai/models/qwen36-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
        /home/dih/models/qwen36-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
        /home/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
    )
    local p
    for p in "${candidates[@]}"; do
        [[ -n "$p" && -f "$p" ]] && { printf '%s\n' "$p"; return 0; }
    done
    p=$(find /run/media/dih /home/dih/models /home/ai/models /mnt -name '*Qwen3.6-35B-A3B*.gguf' -type f 2>/dev/null | head -n 1 || true)
    [[ -n "$p" && -f "$p" ]] && { printf '%s\n' "$p"; return 0; }
    return 1
}

download_adapters() {
    mkdir -p "$CACHE"
    local file
    for file in prerouter_edge0_35b.safetensors lora_edge0_35b.safetensors; do
        if [[ -s "$CACHE/$file" ]]; then
            continue
        fi
        printf 'downloading %s/%s\n' "$REPO" "$file" >&2
        if command -v huggingface-cli >/dev/null 2>&1; then
            huggingface-cli download "$REPO" "$file" --local-dir "$CACHE"
        else
            curl -fL --retry 5 --retry-delay 2 \
                -o "$CACHE/$file" \
                "https://huggingface.co/${REPO}/resolve/main/${file}"
        fi
        [[ -s "$CACHE/$file" ]] || {
            printf 'error: failed to download %s\n' "$file" >&2
            return 1
        }
    done
}

build_and_run() {
    local src root model bin
    root=$(cd "$(dirname "$0")/.." && pwd)
    if [[ -f "$root/oxidize-c/Makefile" ]]; then
        src="$root/oxidize-c"
    elif [[ -f "$PWD/oxidize-c/Makefile" ]]; then
        src="$PWD/oxidize-c"
    else
        printf 'error: oxidize-c sources not found\n' >&2
        return 1
    fi
    download_adapters
    model=$(find_model) || {
        printf 'error: no Qwen3.6-35B-A3B GGUF found. Pass --model PATH\n' >&2
        return 1
    }
    printf 'model=%s\n' "$model" >&2
    make -C "$src" -j"$(nproc 2>/dev/null || echo 8)" oxidize-c \
        CFLAGS='-std=c11 -O3 -march=native -DNDEBUG'
    bin="$src/oxidize-c"
    /usr/bin/time -v "$bin" prompt \
        --model "$model" \
        --prompt "$PROMPT" \
        --n-predict "$N_PREDICT" \
        --ctx "$CTX" \
        --threads "$THREADS" \
        --no-auto \
        --stream-experts \
        --expert-cache-mb "$CACHE_MB" \
        --prerouter "$CACHE/prerouter_edge0_35b.safetensors" \
        --lora "$CACHE/lora_edge0_35b.safetensors" \
        --experts-per-tok 4 \
        --verbose
}

remote_body() {
    cat <<'REMOTE'
set -euo pipefail
CACHE=${EDGE0_CACHE:-$HOME/.cache/oxidize/edge0}
REPO=${EDGE0_HF_REPO:-Edge0/Edge0-35B-A3B-preview}
PROMPT=${EDGE0_PROMPT:-Write a short haiku about streaming experts from SSD.}
N_PREDICT=${EDGE0_N_PREDICT:-32}
THREADS=${EDGE0_THREADS:-16}
CACHE_MB=${EDGE0_EXPERT_CACHE_MB:-3072}
CTX=${EDGE0_CTX:-2048}
MODEL=${EDGE0_MODEL:-}
SRC=${EDGE0_SRC:-}

find_model() {
    local candidates=(
        "${MODEL:-}"
        /home/ai/models/qwen36-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
        /home/dih/models/qwen36-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
        /home/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
    )
    local p
    for p in "${candidates[@]}"; do
        [[ -n "$p" && -f "$p" ]] && { printf '%s\n' "$p"; return 0; }
    done
    p=$(find /run/media/dih /home/dih/models /home/ai/models /mnt -name '*Qwen3.6-35B-A3B*.gguf' -type f 2>/dev/null | head -n 1 || true)
    [[ -n "$p" && -f "$p" ]] && { printf '%s\n' "$p"; return 0; }
    return 1
}

mkdir -p "$CACHE"
for file in prerouter_edge0_35b.safetensors lora_edge0_35b.safetensors; do
    if [[ ! -s "$CACHE/$file" ]]; then
        echo "downloading $REPO/$file"
        if command -v huggingface-cli >/dev/null 2>&1; then
            huggingface-cli download "$REPO" "$file" --local-dir "$CACHE"
        else
            curl -fL --retry 5 --retry-delay 2 \
                -o "$CACHE/$file" \
                "https://huggingface.co/${REPO}/resolve/main/${file}"
        fi
    fi
done

if [[ -z "$SRC" ]]; then
    if [[ -d "$HOME/oxidize/oxidize-c" ]]; then SRC="$HOME/oxidize/oxidize-c"
    elif [[ -d /home/ai/oxidize/oxidize-c ]]; then SRC=/home/ai/oxidize/oxidize-c
    else
        echo "error: set EDGE0_SRC to the oxidize-c directory" >&2
        exit 1
    fi
fi

model=$(find_model) || {
    echo "error: no Qwen3.6-35B-A3B GGUF found. Set EDGE0_MODEL" >&2
    exit 1
}
echo "model=$model"
make -C "$SRC" -j"$(nproc)" oxidize-c CFLAGS='-std=c11 -O3 -march=native -DNDEBUG'
/usr/bin/time -v "$SRC/oxidize-c" prompt \
    --model "$model" \
    --prompt "$PROMPT" \
    --n-predict "$N_PREDICT" \
    --ctx "$CTX" \
    --threads "$THREADS" \
    --no-auto \
    --stream-experts \
    --expert-cache-mb "$CACHE_MB" \
    --prerouter "$CACHE/prerouter_edge0_35b.safetensors" \
    --lora "$CACHE/lora_edge0_35b.safetensors" \
    --experts-per-tok 4 \
    --verbose
REMOTE
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'host=%s local=%s model=%s cache=%s\n' "$HOST" "$LOCAL" "${MODEL:-auto}" "$CACHE"
    remote_body
    exit 0
fi

if [[ "$LOCAL" -eq 1 ]]; then
    build_and_run
    exit 0
fi

if ! ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST" true; then
    printf 'error: cannot ssh to %s — run this script on that host with --local\n' "$HOST" >&2
    printf 'recipe:\n' >&2
    remote_body >&2
    exit 1
fi

ssh "$HOST" env \
    EDGE0_MODEL="${MODEL:-}" \
    EDGE0_CACHE="$CACHE" \
    EDGE0_PROMPT="$PROMPT" \
    EDGE0_N_PREDICT="$N_PREDICT" \
    EDGE0_THREADS="$THREADS" \
    EDGE0_EXPERT_CACHE_MB="$CACHE_MB" \
    EDGE0_CTX="$CTX" \
    EDGE0_SRC="${EDGE0_SRC:-}" \
    bash -s <<EOF
$(remote_body)
EOF
