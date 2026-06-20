# Data Dictionary

## Panel Variables (`data/processed/panel.parquet`)

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `year` | int | INEC | Election year (1999, 2003, ..., 2023) |
| `state` | str | Constitution | Two-letter state code (37 units incl. FC) |
| `party` | str | INEC | Political party acronym |
| `votes` | int | INEC | Raw vote count for party in state |
| `vote_share` | float | Derived | Party votes / total valid votes in state |
| `turnout` | float | Derived | Total votes cast / registered voters in state |
| `registered_voters` | int | INEC | Registered voters in state for that election |
| `incumbent_party` | int | Derived | 1 if party holds presidency, else 0 |
| `vote_share_lag1` | float | Derived | Party vote share in previous election |
| `voter_turnout_lag1` | float | Derived | State turnout in previous election |
| `poverty_rate` | float | NBS | Headcount poverty ratio by state (%) |
| `unemployment_rate` | float | NBS | Unemployment rate by state (%) |
| `cpi_change` | float | CBN | Year-on-year CPI change (national) |
| `oil_price_change` | float | EIA/OPEC | YoY Brent crude price change (%) |
| `violence_index` | float | ACLED | log(1+events) + log(1+fatalities) by state-year |
| `ethnic_frac` | float | Census | Ethnic fractionalization index |
| `urbanization` | float | Census/NBS | Urban population share by state |

## Weight Matrix (`data/processed/W.npz`)

- **Format**: Sparse CSR matrix (scipy), 37 × 37
- **Construction**: Zone-block (default) — W[i,j] = 1 iff same geopolitical zone, i ≠ j. Row-standardized.
- **Alternatives**: Contiguity (shared border), hybrid (0.5 × contiguity + 0.5 × zone-block)

## Strata Frame (`data/processed/strata.parquet`)

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `state` | str | Census | State code |
| `age_group` | str | Census | Age bracket (e.g., 18-24, 25-34, ...) |
| `gender` | str | Census | Male / Female |
| `urban_rural` | str | Census | Urban / Rural |
| `population` | int | Census | Population count for cell |
| `cell_weight` | float | Derived | Cell proportion within state |

## External Reference Data (`data/external/`)

| File | Description |
|------|-------------|
| `state_adjacency.csv` | Pairwise adjacency list (state_a, state_b) |
| `zone_map.csv` | State → geopolitical zone mapping |
| `census_marginals.csv` | Demographic cell counts for MRP poststratification |

## Data Sources & Vintages

See `config/data_sources.yaml` for full API endpoints and vintage timestamps.
