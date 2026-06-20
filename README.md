# ngforecast — Nigeria Election Forecasting Model

A production-grade, spatial-panel election forecasting system for Nigerian presidential elections. Combines structural fundamentals, game-theoretic scenario modelling, multilevel regression with poststratification (MRP), and Monte Carlo simulation under the constitutional win/runoff rule (§134).

## Quickstart

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create environment & install
uv sync

# 3. Pull raw data (requires API keys in .env)
make data

# 4. Train models
make train

# 5. Produce a dated forecast
make forecast

# 6. Run tests
make test
```

## Project Layout

| Directory | Purpose |
|-----------|---------|
| `config/` | All tuneable parameters — code reads config, never hardcodes |
| `src/ngforecast/` | Installable Python package — all production logic |
| `R/` | Isolated R estimators (spatial panel, GMM) called via pipeline |
| `scripts/` | Thin CLI wrappers (one job each, call into `src/`) |
| `tests/` | Unit, integration, and fixture-based test suites |
| `notebooks/` | Exploration only — never imported by `src/` |
| `data/` | Gitignored, DVC-tracked raw → interim → processed pipeline |
| `outputs/` | Gitignored, DVC-tracked model artifacts and forecasts |
| `docs/` | Methodology, data dictionary, ADRs, preregistration |

## Pipeline

The full pipeline DAG is defined in `Snakefile` and can also be invoked via `make`:

```
ingest → features → fit → simulate → report
```

Each stage is independently reproducible via DVC and version-locked dependencies.

## Key Design Choices

- **Spatial panel models** (SAR/SEM via R's `splm`) capture cross-state contagion.
- **Game-theoretic scenarios** model coalition dynamics, candidate entry/exit, and voter migration via QRE.
- **Monte Carlo engine** draws correlated errors at national, zonal, and state levels, then applies the §134 constitutional rule (majority + 25%-in-24-states) to determine win vs. runoff.
- **Forecast log** is append-only and timestamped for public auditability.

## Requirements

- Python ≥ 3.11
- R ≥ 4.3 (for spatial panel and GMM estimators)
- `uv` for Python dependency management
- DVC + remote storage (S3/GCS) for data versioning

## License

See [LICENSE](LICENSE).
