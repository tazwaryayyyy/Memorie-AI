# ─────────────────────────────────────────────────────────────────────────────
#  Memoire — Makefile
#  Usage: make help
# ─────────────────────────────────────────────────────────────────────────────

CARGO        := cargo
PYTHON       := python3
RUST_LOG     ?= info
RELEASE_DIR  := target/release
LIB_NAME     := memoire
DB_DEFAULT   := ./memoire.db

# Detect OS for shared-library extension
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
  LIB_EXT := dylib
  LIB_FILE := lib$(LIB_NAME).$(LIB_EXT)
else ifeq ($(OS), Windows_NT)
  LIB_EXT := dll
  LIB_FILE := $(LIB_NAME).$(LIB_EXT)
else
  LIB_EXT := so
  LIB_FILE := lib$(LIB_NAME).$(LIB_EXT)
endif

# ─── Phony targets ───────────────────────────────────────────────────────────
.PHONY: all build release check test test-unit test-integration test-python \
        bench fmt lint doc clean help install-dev demo-rust demo-python \
        demo-node symbols model-info

all: release

# ─── Build ───────────────────────────────────────────────────────────────────

build:
	@echo "→ Building debug..."
	$(CARGO) build --all-targets
	@echo "✓ Debug build complete."

release:
	@echo "→ Building release..."
	$(CARGO) build --release
	@echo "✓ Release build complete: $(RELEASE_DIR)/$(LIB_FILE)"
	@ls -lh $(RELEASE_DIR)/$(LIB_FILE)

# ─── Check / Lint ────────────────────────────────────────────────────────────

check:
	$(CARGO) check --all-targets

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

lint:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

# ─── Tests ───────────────────────────────────────────────────────────────────

test: test-unit test-integration

test-unit:
	@echo "→ Running unit tests (fast, in-memory)..."
	RUST_LOG=$(RUST_LOG) $(CARGO) test --lib -- --test-threads=4

test-integration:
	@echo "→ Running integration tests (downloads model on first run)..."
	RUST_LOG=$(RUST_LOG) $(CARGO) test --test integration_test -- --nocapture

test-all:
	RUST_LOG=$(RUST_LOG) $(CARGO) test -- --nocapture

test-python: release
	@echo "→ Running Python test suite..."
	cd bindings/python && \
	  MEMOIRE_LIB=../../$(RELEASE_DIR)/$(LIB_FILE) \
	  $(PYTHON) -m pytest tests/ -v --tb=short

# ─── Benchmarks ──────────────────────────────────────────────────────────────

bench: release
	@echo "→ Running benchmarks..."
	$(CARGO) bench --bench core_bench
	@echo "✓ Results saved to target/criterion/"

bench-quick: release
	$(CARGO) bench --bench core_bench -- --warm-up-time 1 --measurement-time 3

# ─── Docs ────────────────────────────────────────────────────────────────────

doc:
	$(CARGO) doc --no-deps --open

doc-build:
	$(CARGO) doc --no-deps

# ─── Demos ───────────────────────────────────────────────────────────────────

demo-rust: release
	@echo "→ Running Rust example..."
	RUST_LOG=$(RUST_LOG) $(CARGO) run --release --example basic_usage

demo-python: release
	@echo "→ Running Python demo..."
	MEMOIRE_LIB=$(RELEASE_DIR)/$(LIB_FILE) $(PYTHON) examples/agent_demo.py

demo-node: release
	@echo "→ Running Node.js demo..."
	cd bindings/node && node demo.js

# ─── CLI ─────────────────────────────────────────────────────────────────────

cli: release
	@echo "Binary: $(RELEASE_DIR)/$(LIB_NAME)"
	@echo "Usage:  ./$(RELEASE_DIR)/$(LIB_NAME) --help"

# ─── Debug helpers ───────────────────────────────────────────────────────────

symbols: release
	@echo "→ Exported FFI symbols:"
	@nm -D $(RELEASE_DIR)/$(LIB_FILE) 2>/dev/null | grep ' T memoire_' \
	  || objdump -t $(RELEASE_DIR)/$(LIB_FILE) 2>/dev/null | grep memoire_ \
	  || echo "(nm/objdump not available)"

model-info:
	@echo "→ Checking for cached embedding model..."
	@$(PYTHON) -c " \
import os, pathlib; \
hf = pathlib.Path(os.environ.get('HF_HOME', pathlib.Path.home() / '.cache' / 'huggingface')); \
models = list(hf.glob('hub/models--*MiniLM*')); \
print('Model cache:', hf / 'hub'); \
print('Found:', models[0].name if models else 'NOT CACHED — will download on first run (~23 MB)'); \
"

# ─── Install helpers ─────────────────────────────────────────────────────────

install-dev:
	@echo "→ Installing Python dev dependencies..."
	$(PYTHON) -m pip install pytest
	@echo "→ Checking Node.js dependencies..."
	@command -v node >/dev/null 2>&1 && echo "  Node.js: OK" || echo "  Node.js: not found"
	@command -v go   >/dev/null 2>&1 && echo "  Go:      OK" || echo "  Go:      not found"

# ─── Clean ───────────────────────────────────────────────────────────────────

clean:
	$(CARGO) clean
	find . -name "*.db" -not -path "./.git/*" -delete
	find . -name "*.db-shm" -delete
	find . -name "*.db-wal" -delete
	@echo "✓ Cleaned."

clean-all: clean
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "✓ Deep clean done."

# ─── Help ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  Memoire — build system"
	@echo "  ──────────────────────────────────────────────────────"
	@echo ""
	@echo "  Build"
	@echo "    make build            Debug build (all targets)"
	@echo "    make release          Release build — produces $(LIB_FILE)"
	@echo "    make check            Cargo check (no codegen)"
	@echo ""
	@echo "  Quality"
	@echo "    make fmt              Auto-format all Rust source"
	@echo "    make lint             Clippy (hard mode — -D warnings)"
	@echo "    make doc              Build + open Rust docs"
	@echo ""
	@echo "  Tests"
	@echo "    make test             Unit + integration tests"
	@echo "    make test-unit        Fast in-memory unit tests only"
	@echo "    make test-integration Full integration tests (needs model)"
	@echo "    make test-python      Python binding test suite"
	@echo "    make test-all         Everything"
	@echo ""
	@echo "  Benchmarks"
	@echo "    make bench            Full Criterion benchmark suite"
	@echo "    make bench-quick      Faster bench run (shorter measurement)"
	@echo ""
	@echo "  Demos"
	@echo "    make demo-rust        Run the Rust basic_usage example"
	@echo "    make demo-python      Run the Python agent demo"
	@echo "    make demo-node        Run the Node.js demo"
	@echo ""
	@echo "  Debug"
	@echo "    make symbols          List exported C symbols in the library"
	@echo "    make model-info       Check if the embedding model is cached"
	@echo ""
	@echo "  Misc"
	@echo "    make install-dev      Install Python/Node dev deps"
	@echo "    make clean            Remove build artifacts and .db files"
	@echo "    make clean-all        Deep clean including __pycache__"
	@echo ""
