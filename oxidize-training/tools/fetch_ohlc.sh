#!/bin/sh
# Daily OHLC. Prefers Yahoo (Stooq is JS-walled).
set -eu
dir=$(dirname "$0")
exec python3 "$dir/fetch_ohlc.py" "$@"
