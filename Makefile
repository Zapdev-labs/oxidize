SHELL := /bin/bash

.PHONY: help build test lint tui tui-test check ci clean

help:
	@echo "oxidize — C runtime + TypeScript TUI"
	@echo "  make build     - Build oxidize-c (CPU)"
	@echo "  make test      - C tests (ASan/UBSan) + TUI tests"
	@echo "  make lint      - clang-tidy on oxidize-c (best-effort)"
	@echo "  make tui       - bun install + typecheck the TUI"
	@echo "  make tui-test  - bun test in oxidize-tui"
	@echo "  make check     - build + test"
	@echo "  make ci        - check + tui"
	@echo "  make clean     - clean oxidize-c artifacts"

build:
	$(MAKE) -C oxidize-c build

test: build
	$(MAKE) -C oxidize-c test

lint:
	$(MAKE) -C oxidize-c lint

tui:
	cd oxidize-tui && bun install && bun run typecheck

tui-test:
	cd oxidize-tui && bun install && bun test

check: test tui-test

ci: check lint

clean:
	$(MAKE) -C oxidize-c clean
