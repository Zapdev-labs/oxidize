#!/usr/bin/env bash
# Diff vendored crates against their crates.io base release (see vendor/UPSTREAM.md).
# Fails if a file other than the documented local patches differs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# crate version allowed-modified-files...
CRATES=(
  "libp2p-dns 0.44.0 Cargo.toml"
  "libp2p-mdns 0.48.0 Cargo.toml"
  "libp2p-yamux 0.47.0 Cargo.toml src/lib.rs"
)

status=0
for entry in "${CRATES[@]}"; do
  read -r name version allowed <<<"$entry"
  curl --proto '=https' --tlsv1.2 -fsSL "https://static.crates.io/crates/$name/$name-$version.crate" |
    tar -xz -C "$TMP"
  base="$TMP/$name-$version"
  while IFS= read -r line; do
    case "$line" in
      "Only in $base: Cargo.lock") continue ;;
      Files\ *\ differ)
        rel="${line#Files "$base"/}"
        rel="${rel%% and *}"
        if [[ " $allowed " == *" $rel "* ]]; then continue; fi
        ;;
    esac
    echo "$name: unexpected difference: $line" >&2
    status=1
  done < <(diff -rq "$base" "$ROOT/vendor/$name" || true)
done
[[ $status -eq 0 ]] && echo "vendor provenance ok"
exit $status
