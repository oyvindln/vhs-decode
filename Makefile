.PHONY: all install install-dev build-rust build-tools test test-decode \
       test-tools bench clean clean-rust clean-tools clean-all lint help

PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip
VENV        ?= .venv
CMAKE_BUILD ?= build
CMAKE_JOBS  ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
QWT         ?= ON

# ─── Default ──────────────────────────────────────────────────────────────────

all: install-dev build-tools  ## Build everything (decode + ld-tools)

# ─── Python / Rust (decode) ──────────────────────────────────────────────────

install:  ## Install vhs-decode into current environment
	$(PIP) install .

install-dev:  ## Install vhs-decode in editable/dev mode (builds Rust extension)
	$(PIP) install -e .

build-rust:  ## Build only the Rust extension module
	$(PYTHON) -m setuptools_rust build

# ─── C++ tools (ld-tools suite) ──────────────────────────────────────────────

build-tools: $(CMAKE_BUILD)/Makefile  ## Build ld-tools (ld-analyse, ld-chroma-decoder, etc.)
	cmake --build $(CMAKE_BUILD) -j$(CMAKE_JOBS)

$(CMAKE_BUILD)/Makefile:
	cmake -S . -B $(CMAKE_BUILD) \
		-DCMAKE_BUILD_TYPE=Release \
		-DUSE_QWT=$(QWT) \
		-DBUILD_TESTING=ON

install-tools: build-tools  ## Install ld-tools to system
	cmake --install $(CMAKE_BUILD)

# ─── Testing ─────────────────────────────────────────────────────────────────

test: test-decode  ## Run all available tests

test-decode:  ## Run Python decode unit tests
	$(PYTHON) -m unittest tests.DemodTest -v

test-tools: build-tools  ## Run C++ ld-tools test suite
	cd $(CMAKE_BUILD) && ctest --output-on-failure

bench:  ## Benchmark decode pipeline using fixture data
	$(PYTHON) bench.py

# ─── Virtual environment ─────────────────────────────────────────────────────

venv:  ## Create a virtual environment and install in dev mode
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .
	@echo ""
	@echo "Activate with:  source $(VENV)/bin/activate"

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:  ## Remove Python build artefacts
	rm -rf build/ dist/ *.egg-info __pycache__
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.so' -path './vhsdecode/*' -delete
	find . -name '*.c' -path './vhsdecode/*.c' ! -name '__init__*' -delete

clean-rust:  ## Remove Rust build artefacts
	cargo clean

clean-tools:  ## Remove C++ build directory
	rm -rf $(CMAKE_BUILD)

clean-all: clean clean-rust clean-tools  ## Remove all build artefacts

# ─── Lint ────────────────────────────────────────────────────────────────────

lint:  ## Check Rust code
	cargo clippy -- -D warnings
	cargo fmt --check

# ─── Help ────────────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
