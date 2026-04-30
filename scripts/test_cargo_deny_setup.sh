#!/usr/bin/env bash
set -euo pipefail

test -f deny.toml
rg -q '^\[advisories\]$' deny.toml
rg -q '^\[licenses\]$' deny.toml
rg -q '^\[sources\]$' deny.toml
make -n audit >/dev/null

echo "cargo-deny setup validated."
