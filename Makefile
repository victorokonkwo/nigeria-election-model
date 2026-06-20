.PHONY: data train forecast test lint fmt clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Data ──────────────────────────────────────
data:  ## Pull raw data + build processed panel
	python scripts/pull_data.py
	python scripts/build_panel.py

# ── Train ─────────────────────────────────────
train:  ## Fit all model layers
	python scripts/fit_models.py

# ── Forecast ──────────────────────────────────
forecast:  ## Run dated Monte Carlo forecast
	python scripts/run_forecast.py

# ── Tests ─────────────────────────────────────
test:  ## Run full test suite
	pytest tests/ -x -q --tb=short

test-unit:  ## Run unit tests only
	pytest tests/unit/ -x -q --tb=short

test-integration:  ## Run integration tests
	pytest tests/integration/ -m integration -q --tb=short

# ── Code quality ──────────────────────────────
lint:  ## Lint with ruff
	ruff check src/ tests/ scripts/

fmt:  ## Format with ruff + black
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/

typecheck:  ## Static type checking
	mypy src/ngforecast/

# ── Cleanup ───────────────────────────────────
clean:  ## Remove generated artifacts (not raw data)
	rm -rf outputs/models/* outputs/forecasts/* outputs/figures/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ── Pipeline (Snakemake) ─────────────────────
pipeline:  ## Run full Snakemake pipeline
	snakemake --cores 4 all

pipeline-dry:  ## Dry-run Snakemake DAG
	snakemake --cores 1 -n all
