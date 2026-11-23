# Methodology

## Overview

This model forecasts the Nigerian presidential election by combining:

1. **Structural fundamentals** — a spatial autoregressive (SAR) panel model estimated on state × election data (1999–2023), capturing economic conditions, incumbency, and cross-state contagion.

2. **Game-theoretic scenarios** — cooperative and non-cooperative models of candidate coalitions, entry/exit, and voter migration via Quantal Response Equilibrium (QRE).

3. **Multilevel regression + poststratification (MRP)** — when polling data is available, a multilevel model adjusts for demographic composition using census strata.

4. **Monte Carlo simulation** — correlated error draws at national, zonal, and state levels, filtered through the §134 constitutional win/runoff rule.

## Spatial Panel Model

The core specification is a spatial autoregressive (SAR) panel:

$$y_{it} = \rho W y_{it} + X_{it} \beta + \alpha_i + \gamma_t + \varepsilon_{it}$$

where:
- $y_{it}$ = vote share of the leading party in state $i$, election $t$
- $W$ = zone-block spatial weight matrix (row-standardized)
- $X_{it}$ = covariates (poverty, unemployment, CPI change, oil price, violence index, lagged vote share, incumbency)
- $\alpha_i$ = state fixed effects
- $\gamma_t$ = election fixed effects
- $\rho$ = spatial autoregressive parameter

Estimated via ML using R's `splm` package.

## Game-Theoretic Layer

### Coalition Formation
- Cooperative game theory (Shapley values, core stability) determines credible coalitions.
- Coalition value = combined predicted vote share under the spatial model.

### Voter Migration (QRE)
When candidates withdraw or form coalitions, voter reallocation follows a Quantal Response Equilibrium:

$$P(j|i) = \frac{\exp(\lambda \cdot U_{ij})}{\sum_k \exp(\lambda \cdot U_{ik})}$$

where $\lambda$ controls rationality (higher = more strategic).

### Turnout (Tullock Contest)
Mobilization effort modelled as a Tullock contest → turnout multipliers by state reflecting competitiveness.

## Ensemble

Component predictions are combined via Bayesian stacking, with weights optimized on leave-future-out backtest predictions (2015, 2019, 2023).

## Simulation Engine

For each of $S = 20{,}000$ simulations:
1. Sample a scenario $j$ with probability $\pi_j$
2. Draw correlated errors: $\varepsilon_s = \eta_s + \zeta_{s,z(i)} + \nu_{s,i}$
3. Compute state-level vote shares and turnout
4. Apply §134: check plurality + 25%-in-24 spread
5. If no winner → simulate runoff

## Constitutional Rule (§134)

**Outright win, §134(2).** A candidate wins if:
- **(a)** they have the highest number of votes cast nationally (plurality), AND
- **(b)** they have at least one-quarter of the votes cast in each of at least two-thirds of all the States and the FCT.

Following *Obi v INEC* and *Atiku v INEC* (Supreme Court, 26 Oct 2023), the FCT is treated as the 37th unit (§299) and is not a separate requirement. Two-thirds of 37 is $24\tfrac{2}{3}$; following *Awolowo v Shagari* (SC.62/1979), the fraction is met by **24 units at ≥ 25% plus a 25th unit at ≥ $\tfrac{2}{3} \times 25\% \approx 16.67\%$**.

Alternative readings are configurable in `config/constitution.yaml` for sensitivity analysis: `fct_interpretation` ∈ {`fct_as_state`, `states_only`, `states_plus_fct`} and `fractional_rule` ∈ {`awolowo_v_shagari`, `ceil`, `floor`}. See ADR-003.

**Second election, §134(3)–(4).** If nobody wins, INEC holds a second election within seven days between (a) the national vote leader and (b) the remaining candidate who led in the most units (ties go to the higher national total). This is not necessarily the national runner-up. The winner needs a majority of votes cast plus the same spread as (b).

**Third election, §134(5).** If the second election produces no winner, a third election between the same two candidates is held within seven days and decided by a simple majority.

The engine reports `runoff_probability` (a second election is needed) and `third_election_probability`.
