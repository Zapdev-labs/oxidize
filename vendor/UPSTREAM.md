# Vendored crate provenance

These crates are patched in via `[patch.crates-io]` in the workspace `Cargo.toml`
(added in b0932075) to pick up yamux >= 0.13.10 and hickory 0.26.1 security fixes
until libp2p 0.57 ships on crates.io. Remove each entry once upstream releases cover it.

| Crate | Base release | Upstream source (`.cargo_vcs_info.json`) | Locally modified files |
|---|---|---|---|
| `libp2p-dns` | crates.io `0.44.0` | `libp2p/rust-libp2p@d92dabcbb87c4796d46e3d312b7fa042af10279a` `transports/dns` | `Cargo.toml` (`hickory-resolver` 0.25.2 → 0.26.1) |
| `libp2p-mdns` | crates.io `0.48.0` | `libp2p/rust-libp2p@c9bd92ba2d4727d79b0c2623c20c998a4e60995a` `protocols/mdns` | `Cargo.toml` (`hickory-proto` 0.25.2 → 0.26.1) |
| `libp2p-yamux` | crates.io `0.47.0` | `libp2p/rust-libp2p@9736aacf814eb7b9df0372c5f9adcffba8a4b212` `muxers/yamux` | `Cargo.toml`, `src/lib.rs` (drop the `yamux012` compat path, single `yamux` 0.13.10 dependency) |

Every other file is byte-identical to the published `.crate` (the published
`Cargo.lock` is omitted). `scripts/check_vendor_provenance.sh` re-downloads each
base release from crates.io and fails if any file outside the "Locally modified"
column differs; update this table and the script's list together when re-vendoring.
