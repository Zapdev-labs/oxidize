SHELL := /bin/bash

.DEFAULT_GOAL := build

.PHONY: help build test clean lint \
        rust-fmt rust-lint rust-audit rust-udeps rust-test rust-build \
        rust-wasm rust-check rust-ci fmt lint-rust audit check ci

help:
	@echo "Primary engine is oxidize-c:"
	@echo "  make            - Release build of oxidize-c"
	@echo "  make test       - oxidize-c tests (ASan when the compiler provides it)"
	@echo "  make clean      - Remove oxidize-c build outputs"
	@echo "  make lint       - clang-tidy on oxidize-c, if clang-tidy is installed"
	@echo "Rust workspace (crates live in oxidize-rust/):"
	@echo "  make rust-build - cargo build --workspace --release"
	@echo "  make rust-test  - cargo test --workspace --all-targets"
	@echo "  make rust-fmt   - cargo fmt --all --check"
	@echo "  make rust-lint  - clippy -D warnings"
	@echo "  make rust-check - fmt, lint, audit, and rust tests"
	@echo "  make rust-ci    - rust-check, udeps, and rust-build"

build:
	$(MAKE) -C oxidize-c build

test:
	$(MAKE) -C oxidize-c test

clean:
	$(MAKE) -C oxidize-c clean

lint:
	$(MAKE) -C oxidize-c lint

rust-fmt fmt:
	cargo fmt --all --check

rust-lint lint-rust:
	cargo clippy --workspace --all-targets -- -D warnings

rust-audit:
	cargo deny check

rust-udeps:
	./scripts/check-udeps.sh

rust-test:
	cargo test --workspace --all-targets

rust-build:
	cargo build --workspace --release

rust-wasm:
	cargo build -p oxidize-core --target wasm32-unknown-unknown --release --features wasm
	command -v wasm-bindgen >/dev/null || cargo install --locked wasm-bindgen-cli --version 0.2.120
	wasm-bindgen --target web --out-dir dist/wasm target/wasm32-unknown-unknown/release/oxidize_core.wasm

rust-check check: rust-fmt rust-lint rust-audit rust-test

rust-ci ci: rust-check rust-udeps rust-build

audit: rust-audit
