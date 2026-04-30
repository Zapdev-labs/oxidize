SHELL := /bin/bash

.PHONY: help fmt lint test build check ci

help:
	@echo "Common tasks:"
	@echo "  make fmt    - Check Rust formatting"
	@echo "  make lint   - Run clippy with warnings denied"
	@echo "  make test   - Run workspace tests"
	@echo "  make build  - Build release binaries for all targets"
	@echo "  make check  - Run fmt + lint + test"
	@echo "  make ci     - Run check + build"

fmt:
	cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace --all-targets

build:
	cargo build --workspace --all-targets --release

check: fmt lint test

ci: check build
