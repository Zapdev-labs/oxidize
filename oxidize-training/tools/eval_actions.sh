#!/bin/sh
# Fail unless sample() after ACTION= emits the side that matches the ticket.
# Usage: tools/eval_actions.sh [bin] [ckpt] [vocab]
set -eu
BIN=${1:-./bin/oxidize-training}
CKPT=${2:-out/trader.bin}
VOCAB=${3:-data/vocab.bin}
need=8
ok=0
n=0
check() {
    prompt=$1
    want=$2
    n=$((n + 1))
    out=$("$BIN" sample --ckpt "$CKPT" --vocab "$VOCAB" --prompt "$prompt" --tokens 40 --temp 0.2 --seed "$n")
    gen=$(printf '%s\n' "$out" | awk 'f{print} /^---$/{f=1}')
    printf 'WANT %s\nPROMPT %s\n%s\n' "$want" "$prompt" "$gen"
    got=$(printf '%s\n' "$gen" | grep -oE 'ACTION=(BUY|SELL|HOLD)' | head -1 | sed 's/ACTION=//')
    if [ "$got" = "$want" ]; then
        ok=$((ok + 1))
        echo "OK $got $ok/$n"
    else
        echo "MISS got=${got:-none} want=$want $ok/$n"
    fi
}
check "AAPL 1h close 228 RSI 32 ATR 9. Setup: opening range breakout. ACTION=" BUY
check "NVDA 15m close 110 RSI 78 ATR 4. Setup: exhaustion at session highs. ACTION=" SELL
check "SPY 1d close 510 RSI 51 ATR 6. Setup: balanced auction. ACTION=" HOLD
check "TSLA 5m close 240 RSI 30 ATR 8. Setup: failed breakdown reclaim. ACTION=" BUY
check "MSFT 4h close 420 RSI 70 ATR 5. Setup: range fade at value area high. ACTION=" SELL
check "AMD 1h close 155 RSI 40 ATR 3. Setup: trend pullback to the 20 EMA. ACTION=" BUY
check "QQQ 15m close 480 RSI 62 ATR 7. Setup: failed to take prior day high. ACTION=" SELL
check "JPM 1d close 198 RSI 48 ATR 2. Setup: inside day. ACTION=" HOLD
check "META 1h close 510 RSI 35 ATR 11. Setup: liquidity sweep then reverse. ACTION=" BUY
check "XOM 4h close 118 RSI 80 ATR 3. Setup: lower high on weak breadth. ACTION=" SELL
echo "hit $ok/$n need $need"
test "$ok" -ge "$need"
