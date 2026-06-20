# Pre-Registration Protocol

## Forecast: 2027 Nigerian Presidential Election

### Version
1.0 — Frozen on [DATE]

### Model Specification
As documented in `docs/methodology.md`. The model will not be modified after this date except:
- Bug fixes (logged in the forecast log)
- Data updates (new vintages of existing sources)

### Data Sources
- INEC historical results (1999–2023)
- NBS poverty/unemployment/CPI
- CBN inflation/exchange rate
- Oil prices (EIA/OPEC)
- ACLED political violence
- Pre-election polls (if ≥ 3 available)

### Key Parameters (Frozen)
- Spatial weight matrix: zone-block, row-standardized
- Ensemble method: Bayesian stacking
- Monte Carlo simulations: 20,000
- Error structure: 3-level (national + zonal + state)
- Constitutional rule: §134 as implemented in `src/ngforecast/simulation/constitution.py`

### Evaluation Criteria
The model will be evaluated against the actual 2027 result on:
1. **Brier score** — for win probabilities
2. **RMSE** — for state-level vote share predictions
3. **Coverage** — 90% prediction intervals
4. **Calibration** — reliability diagram

### Forecast Log
All forecasts are appended to `outputs/forecasts/forecast_log.jsonl` with timestamps. This log is public and append-only.

### Commitments
- No post-hoc model selection after seeing results
- All code and data versioned via Git + DVC
- Backtest results (2015/2019/2023) published before election day
