#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
runner="$root/scripts/bench_qwen36_c_remote.sh"
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
assert_contains() { grep -F -- "$2" "$1" >/dev/null || fail "missing: $2"; }

# Given the checked-in runner, when its local contract is checked, then no
# remote command is needed and the machine-readable schema validates.
bash -n "$runner"
"$runner" --self-test >"$tmp/self-test.out"
assert_contains "$tmp/self-test.out" 'remote_dir=/tmp/oxidize-qwen36-${USER}.XXXXXXXX'
assert_contains "$tmp/self-test.out" '"schema":"qwen36-cpu-benchmark-v1"'

# Given a dry run, when both engines are rendered, then they have identical
# CPU affinity, NUMA binding, thread count, case sizes, warmups, and repeats.
"$runner" --dry-run --model /home/ai/models/qwen36-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --oxidize-binary /tmp/oxidize-c-green2 --threads 16 >"$tmp/dry-run.out"
for token in '0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30' '--membind=0' 'threads=16' 'pp64/tg32' 'pp512/tg32' '--bench-prompt-tokens 64 --bench-decode-tokens 32' '--bench-prompt-tokens 512 --bench-decode-tokens 32' '-p 64 -n 32' '-p 512 -n 32' 'warmup=2 repetitions=5'; do
    assert_contains "$tmp/dry-run.out" "$token"
done
assert_contains "$tmp/dry-run.out" 'oxidize_binary=/tmp/oxidize-c-green2'
[[ $(grep -c '^oxidize case=' "$tmp/dry-run.out") -eq 2 ]] || fail 'expected two oxidize cases'
[[ $(grep -c '^llama case=' "$tmp/dry-run.out") -eq 2 ]] || fail 'expected two llama cases'

cc -std=c11 -Wall -Wextra -Werror -O2 -o "$tmp/qwen36_record" "$root/oxidize-c/tools/qwen36_record.c"
printf 'Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\nMaximum resident set size (kbytes): 3\n' >"$tmp/time.txt"
printf '%s\n' '{"results":[{"prefill_tok_per_s":3.25,"decode_tok_per_s":1.75}]}' >"$tmp/oxidize.out"
ox_row=$("$tmp/qwen36_record" row --engine oxidize-c --case pp64/tg32 --phase measure --run 1 --command cmd --timing "$tmp/time.txt" --stdout "$tmp/oxidize.out" --load-before 1 --load-after 2 --rss 3 --model /model.gguf --sha sha --size 4 --revision rev --llama commit --affinity cpus --numa 0 --threads 16 --label lab)
[[ "$ox_row" == *'"prefill_tok_per_s":3.25'* ]] || fail "oxidize prefill: $ox_row"
[[ "$ox_row" == *'"decode_tok_per_s":1.75'* ]] || fail "oxidize decode: $ox_row"
[[ "$ox_row" == *'"elapsed_s":62.5'* ]] || fail "elapsed: $ox_row"
printf '%s\n' '{"n_prompt":64,"n_gen":0,"avg_ts":20.5}' '{"n_prompt":0,"n_gen":32,"avg_ts":7.25}' >"$tmp/llama.out"
llama_row=$("$tmp/qwen36_record" row --engine llama --case pp64/tg32 --phase measure --run 1 --command cmd --timing "$tmp/time.txt" --stdout "$tmp/llama.out" --load-before 1 --load-after 2 --rss 3 --model /model.gguf --sha sha --size 4 --revision rev --llama commit --affinity cpus --numa 0 --threads 16 --label lab)
[[ "$llama_row" == *'"prefill_tok_per_s":20.5'* ]] || fail "llama prefill: $llama_row"
[[ "$llama_row" == *'"decode_tok_per_s":7.25'* ]] || fail "llama decode: $llama_row"
! grep -F python3 "$runner" >/dev/null || fail 'bench script still invokes python3'

# Given a fake shared host reporting load 250, when the runner is invoked,
# then it refuses with EX_TEMPFAIL before any benchmark/build command executes.
mkdir -p "$tmp/bin"
cat >"$tmp/bin/ssh" <<'SSH'
#!/usr/bin/env bash
printf 'ssh %q load=250.00 process=conflict\n' "$*" >>"$FAKE_SSH_LOG"
printf '250.00\nllama-bench\n'
SSH
chmod +x "$tmp/bin/ssh"
set +e
PATH="$tmp/bin:$PATH" FAKE_SSH_LOG="$tmp/ssh.log" "$runner" --host fake@host --model /model.gguf >"$tmp/busy.out" 2>&1
status=$?
set -e
[[ $status -eq 75 ]] || fail "busy host exit was $status, expected 75"
assert_contains "$tmp/ssh.log" 'ssh '
assert_contains "$tmp/busy.out" 'benchmark deferred: shared host busy'
assert_contains "$tmp/ssh.log" 'load=250.00'
! grep -E 'make|llama-bench|oxidize-c bench|git clone' "$tmp/ssh.log" >/dev/null || fail 'benchmark or build was invoked while busy'
[[ $(wc -l <"$tmp/ssh.log") -eq 1 ]] || fail 'busy host performed more than one SSH request'

if [[ -n "${BENCH_QWEN36_TEST_EVIDENCE_DIR:-}" ]]; then
    mkdir -p "$BENCH_QWEN36_TEST_EVIDENCE_DIR"
    cp "$tmp/dry-run.out" "$BENCH_QWEN36_TEST_EVIDENCE_DIR/task-1-benchmark-contract.txt"
    cp "$tmp/busy.out" "$BENCH_QWEN36_TEST_EVIDENCE_DIR/task-1-benchmark-contract-error.txt"
fi

printf 'PASS: qwen36 remote benchmark contract\n'
