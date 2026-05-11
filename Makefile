SHELL := /bin/bash

.PHONY: help fmt lint audit test build wasm check ci

help:
	@echo "Common tasks:"
	@echo "  make fmt    - Check Rust formatting"
	@echo "  make lint   - Run clippy with warnings denied"
	@echo "  make audit  - Run cargo-deny license/security audit"
	@echo "  make test   - Run workspace tests"
	@echo "  make build  - Build release binaries for all targets"
	@echo "  make wasm   - Build oxidize-core with wasm-bindgen output"
	@echo "  make check  - Run fmt + lint + test"
	@echo "  make ci     - Run check + build"

fmt:
	cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

audit:
	cargo deny check

test:
	cargo test --workspace --all-targets

build:
	cargo build --workspace --all-targets --release

wasm:
	cargo build -p oxidize-core --target wasm32-unknown-unknown --release --features wasm
	command -v wasm-bindgen >/dev/null || cargo install --locked wasm-bindgen-cli --version 0.2.120
	wasm-bindgen --target web --out-dir dist/wasm target/wasm32-unknown-unknown/release/oxidize_core.wasm

check: fmt lint audit test

ci: check build
