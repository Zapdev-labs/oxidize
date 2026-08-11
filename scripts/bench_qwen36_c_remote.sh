#!/usr/bin/env bash
set -euo pipefail

readonly CPU_LIST='0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30'
readonly NUMA_NODE=0
readonly PINNED_LLAMA_COMMIT='c588c4f'
readonly DEFAULT_MODEL='/home/ai/models/qwen36-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
readonly DEFAULT_REVISION='6e67975'

usage() {
    cat <<'USAGE'
Usage: scripts/bench_qwen36_c_remote.sh [options]

Options:
  --model PATH          Canonical GGUF on the remote host
  --revision REV        Commit to build in an isolated remote checkout
  --oxidize-binary PATH Use an existing remote oxidize-c executable
  --host HOST           SSH destination (default: ai@192.168.1.121)
  --threads N           Must be 16 (the fixed benchmark contract)
  --warmup N            Must be 2 (the fixed benchmark contract)
  --repetitions N       Must be 5 (the fixed benchmark contract)
  --label LABEL         JSONL run label (default: unlabelled)
  --dry-run             Print the remote contract without connecting
  --self-test           Validate the local contract without connecting
USAGE
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

readonly REMOTE_DIR_TEMPLATE='/tmp/oxidize-qwen36-${USER}.XXXXXXXX'

emit_contract() {
    local remote_dir=$1
    local model=$2
    local revision=$3
    local warmup=$4
    local repetitions=$5
    local label=$6
    local binary=${7:-build-from-revision}
    local case_name prompt_tokens decode_tokens

    printf 'remote_dir=%s\n' "$remote_dir"
    printf 'revision=%s oxidize_binary=%s model=%s affinity=taskset -c %s numa=numactl --membind=%s threads=16 mmap=enabled warmup=%s repetitions=%s label=%s\n' \
        "$revision" "$binary" "$model" "$CPU_LIST" "$NUMA_NODE" "$warmup" "$repetitions" "$label"
    for case_name in pp64/tg32 pp512/tg32; do
        prompt_tokens=${case_name#pp}
        prompt_tokens=${prompt_tokens%/tg32}
        decode_tokens=${case_name##*/tg}
        printf 'oxidize case=%s command=taskset -c %s numactl --membind=%s ./oxidize-c bench --model %s --threads 16 --no-auto --bench-prompt-tokens %s --bench-decode-tokens %s --bench-no-eos --bench-warmup %s --bench-iters %s --json\n' \
            "$case_name" "$CPU_LIST" "$NUMA_NODE" "$model" "$prompt_tokens" "$decode_tokens" "$warmup" "$repetitions"
        printf 'llama case=%s command=taskset -c %s numactl --membind=%s <isolated-pinned-build>/llama-bench -m %s -p %s -n %s -t 16 --no-warmup -r 1 -mmp 1 -o jsonl (external warmup=%s repetitions=%s)\n' \
            "$case_name" "$CPU_LIST" "$NUMA_NODE" "$model" "$prompt_tokens" "$decode_tokens" "$warmup" "$repetitions"
    done
}

self_test() {
    local remote_dir=$REMOTE_DIR_TEMPLATE
    [[ "$remote_dir" == '/tmp/oxidize-qwen36-${USER}.XXXXXXXX' ]]
    [[ "$CPU_LIST" == '0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30' ]]
    [[ "$NUMA_NODE" == 0 ]]

    local schema
    schema='{"schema":"qwen36-cpu-benchmark-v1","command":"taskset -c 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 numactl --membind=0 engine","engine":"oxidize-c","case":"pp64/tg32","phase":"measure","run":1,"model_path":"/model.gguf","model_sha256":"sha","model_size_bytes":1,"revision":"6e67975","llama_commit":"c588c4f","affinity":"0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30","numa_node":0,"threads":16,"mmap":true,"load_before":1.0,"load_after":1.0,"rss_kb":1,"prefill_tok_per_s":2.0,"decode_tok_per_s":1.0,"timing":{"elapsed_s":1.0}}'
    python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); required={"schema","command","engine","case","phase","run","model_path","model_sha256","model_size_bytes","revision","llama_commit","affinity","numa_node","threads","mmap","load_before","load_after","rss_kb","prefill_tok_per_s","decode_tok_per_s","timing"}; assert required <= d.keys(); assert d["case"] == "pp64/tg32"; assert d["timing"]["elapsed_s"] > 0; assert d["prefill_tok_per_s"] > 0; assert d["decode_tok_per_s"] > 0' <<<"$schema"
    emit_contract "$remote_dir" "$DEFAULT_MODEL" "$DEFAULT_REVISION" 2 5 self-test
    printf '%s\n' "$schema"
}

host='ai@192.168.1.121'
model=$DEFAULT_MODEL
revision=$DEFAULT_REVISION
oxidize_binary=''
threads=16
warmup=2
repetitions=5
label='unlabelled'
dry_run=false
selftest=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) model=${2:?missing value for --model}; shift 2 ;;
        --revision) revision=${2:?missing value for --revision}; shift 2 ;;
        --oxidize-binary) oxidize_binary=${2:?missing value for --oxidize-binary}; shift 2 ;;
        --host) host=${2:?missing value for --host}; shift 2 ;;
        --threads) threads=${2:?missing value for --threads}; shift 2 ;;
        --warmup) warmup=${2:?missing value for --warmup}; shift 2 ;;
        --repetitions) repetitions=${2:?missing value for --repetitions}; shift 2 ;;
        --label) label=${2:?missing value for --label}; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --self-test) selftest=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

if "$selftest"; then
    "$dry_run" && die '--self-test and --dry-run are mutually exclusive'
    self_test
    exit 0
fi

is_uint "$threads" && is_uint "$warmup" && is_uint "$repetitions" || die 'threads, warmup, and repetitions must be unsigned integers'
[[ "$threads" == 16 ]] || die 'the Qwen3.6 contract requires --threads 16'
[[ "$warmup" == 2 ]] || die 'the Qwen3.6 contract requires --warmup 2'
[[ "$repetitions" == 5 ]] || die 'the Qwen3.6 contract requires --repetitions 5'

remote_dir=$REMOTE_DIR_TEMPLATE
if "$dry_run"; then
    emit_contract "$remote_dir" "$model" "$revision" "$warmup" "$repetitions" "$label" "$oxidize_binary"
    exit 0
fi

if ! host_state=$(ssh "$host" bash -s <<'GATE'
set -euo pipefail
awk '{print $1}' /proc/loadavg
ps -eo args=
GATE
); then
    printf 'error: unable to inspect shared host\n' >&2
    exit 1
fi
load_before=${host_state%%$'\n'*}
processes=${host_state#*$'\n'}
if awk -v one_minute_load="$load_before" 'BEGIN { exit !(one_minute_load >= 96.0) }' \
    || grep -E '[l]lama-(bench|cli)|[o]xidize-c([^[:alnum:]_-]|$)|[[:alnum:]_-](train|finetune)([^[:alnum:]_-]|$)' <<<"$processes" >/dev/null; then
    printf 'benchmark deferred: shared host busy\n' >&2
    exit 75
fi

ssh "$host" bash -s -- "$model" "$revision" "$label" "$CPU_LIST" "$NUMA_NODE" "$threads" "$warmup" "$repetitions" "$PINNED_LLAMA_COMMIT" "$oxidize_binary" <<'REMOTE'
set -euo pipefail

model=$1
revision=$2
label=$3
cpus=$4
numa_node=$5
threads=$6
warmup=$7
repetitions=$8
llama_commit=$9
oxidize_binary=${10}
source_repo=${QWEN36_SOURCE_REPO:-/home/ai/oxidize}
llama_repo=${QWEN36_LLAMA_REPO:-/home/ai/llama.cpp}

load_before=$(awk '{print $1}' /proc/loadavg)
busy=false
awk -v one_minute_load="$load_before" 'BEGIN { exit !(one_minute_load >= 96.0) }' && busy=true
if ps -eo args= | grep -E '[l]lama-(bench|cli)|[o]xidize-c([^[:alnum:]_-]|$)|[[:alnum:]_-](train|finetune)([^[:alnum:]_-]|$)' >/dev/null; then
    busy=true
fi
if "$busy"; then
    printf 'benchmark deferred: shared host busy\n' >&2
    exit 75
fi

[[ "$(git -C "$llama_repo" rev-parse --verify HEAD)" == "$llama_commit" ]] || {
    printf 'error: pinned llama.cpp commit %s is required\n' "$llama_commit" >&2
    exit 1
}
[[ -f "$model" ]] || { printf 'error: model not found: %s\n' "$model" >&2; exit 1; }
run_dir=$(mktemp -d "/tmp/oxidize-qwen36-${USER}.XXXXXXXX")
[[ -d "$run_dir" && ! -L "$run_dir" && -O "$run_dir" ]] || {
    printf 'error: could not create a private run directory\n' >&2
    exit 1
}
repo="$run_dir/perf/qwen36-cpu"
llama_build="$run_dir/llama-build"
out="$run_dir/results.jsonl"
manifest="$run_dir/manifest.json"

cmake -S "$llama_repo" -B "$llama_build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build "$llama_build" --target llama-bench -j"$threads"
llama_bin="$llama_build/bin/llama-bench"

if [[ -n "$oxidize_binary" ]]; then
    [[ -x "$oxidize_binary" ]] || {
        printf 'error: oxidize binary is not executable: %s\n' "$oxidize_binary" >&2
        exit 1
    }
    oxidize_bin=$oxidize_binary
else
    git clone --no-hardlinks --no-checkout "$source_repo" "$repo"
    git -C "$repo" checkout --detach "$revision"
    make -C "$repo/oxidize-c" clean oxidize-c CFLAGS='-std=c11 -O3 -march=native -DNDEBUG'
    oxidize_bin="$repo/oxidize-c/oxidize-c"
fi

model_sha=$(sha256sum "$model" | awk '{print $1}')
model_size=$(stat -c '%s' "$model")
oxidize_sha=$(sha256sum "$oxidize_bin" | awk '{print $1}')
printf '{"schema":"qwen36-cpu-benchmark-v1","run_dir":"%s","revision":"%s","oxidize_binary":"%s","oxidize_sha256":"%s","llama_commit":"%s","model_path":"%s","model_sha256":"%s","model_size_bytes":%s,"affinity":"%s","numa_node":%s,"threads":%s,"warmup":%s,"repetitions":%s,"label":"%s"}\n' \
    "$run_dir" "$revision" "$oxidize_bin" "$oxidize_sha" "$llama_commit" "$model" "$model_sha" "$model_size" "$cpus" "$numa_node" "$threads" "$warmup" "$repetitions" "$label" >"$manifest"

record() {
    local engine=$1 case_name=$2 phase=$3 run=$4 command=$5 timing=$6 stdout=$7
    local load_after rss
    load_after=$(awk '{print $1}' /proc/loadavg)
    rss=$(awk '/Maximum resident set size/ {print $6}' "$timing")
    python3 - "$engine" "$case_name" "$phase" "$run" "$command" "$timing" "$stdout" "$load_before" "$load_after" "$rss" "$model" "$model_sha" "$model_size" "$revision" "$llama_commit" "$cpus" "$numa_node" "$threads" "$label" >>"$out" <<'PY'
import json, pathlib, sys
engine, case_name, phase, run, command, timing, stdout, load_before, load_after, rss, model, sha, size, revision, llama, cpus, numa, threads, label = sys.argv[1:]

def clock_seconds(value):
    parts = [float(part) for part in value.split(':')]
    total = 0.0
    for part in parts:
        total = total * 60.0 + part
    return total

elapsed_text = next(line.rsplit(': ', 1)[1].strip()
                    for line in pathlib.Path(timing).read_text().splitlines()
                    if 'Elapsed (wall clock) time' in line)
records = []
for line in pathlib.Path(stdout).read_text().splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict):
        records.append(value)

if engine == 'oxidize-c':
    result = records[-1]['results'][0]
    prefill = float(result['prefill_tok_per_s'])
    decode = float(result['decode_tok_per_s'])
else:
    prompt = next(value for value in records
                  if int(value.get('n_prompt', 0)) > 0 and int(value.get('n_gen', 0)) == 0)
    generation = next(value for value in records if int(value.get('n_gen', 0)) > 0)
    prefill = float(prompt['avg_ts'])
    decode = float(generation['avg_ts'])

print(json.dumps({'schema':'qwen36-cpu-benchmark-v1','command':command,'engine':engine,'case':case_name,'phase':phase,'run':int(run),'model_path':model,'model_sha256':sha,'model_size_bytes':int(size),'revision':revision,'llama_commit':llama,'affinity':cpus,'numa_node':int(numa),'threads':int(threads),'mmap':True,'load_before':float(load_before),'load_after':float(load_after),'rss_kb':int(rss or 0),'prefill_tok_per_s':prefill,'decode_tok_per_s':decode,'timing':{'elapsed_s':clock_seconds(elapsed_text)},'label':label}, sort_keys=True))
PY
}

run_case() {
    local case_name=$1 prompt_tokens=$2 decode_tokens=$3 engine=$4 phase=$5 run=$6
    local timing="$run_dir/${engine}-${case_name//\//-}-${phase}-${run}.time"
    local -a command
    if [[ "$engine" == oxidize-c ]]; then
        command=(taskset -c "$cpus" numactl --membind="$numa_node" "$oxidize_bin" bench --model "$model" --threads "$threads" --no-auto --bench-prompt-tokens "$prompt_tokens" --bench-decode-tokens "$decode_tokens" --bench-no-eos --bench-warmup 0 --bench-iters 1 --json)
    else
        command=(taskset -c "$cpus" numactl --membind="$numa_node" "$llama_bin" -m "$model" -p "$prompt_tokens" -n "$decode_tokens" -t "$threads" --no-warmup -r 1 -mmp 1 -o jsonl)
    fi
    local stdout="$run_dir/${engine}-${case_name//\//-}-${phase}-${run}.stdout"
    /usr/bin/time -v -o "$timing" "${command[@]}" >"$stdout"
    record "$engine" "$case_name" "$phase" "$run" "$(printf '%q ' "${command[@]}")" "$timing" "$stdout"
}

for spec in 'pp64/tg32 64 32' 'pp512/tg32 512 32'; do
    read -r case_name prompt_tokens decode_tokens <<<"$spec"
    for engine in oxidize-c llama; do
        for ((run = 1; run <= warmup; run++)); do run_case "$case_name" "$prompt_tokens" "$decode_tokens" "$engine" warmup "$run"; done
    done
    for ((run = 1; run <= repetitions; run++)); do
        for engine in oxidize-c llama; do run_case "$case_name" "$prompt_tokens" "$decode_tokens" "$engine" measure "$run"; done
    done
done
cat "$out"
printf 'artifacts=%s\n' "$run_dir"
REMOTE
