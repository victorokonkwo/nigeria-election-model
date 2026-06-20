# ADR-002: Prior Specification for Spatial Autoregressive Parameter

## Status
Accepted

## Context
The SAR model's key parameter ρ (spatial autoregressive coefficient) measures cross-state contagion. We need to specify priors for Bayesian inference and bounds for ML estimation.

## Decision
Use a **uniform prior on ρ ∈ (-1, 1)**, the standard theoretical bounds for spatial autoregressive models with row-standardized W.

## Rationale
- Uniform prior is weakly informative — lets data drive the estimate.
- Historical estimates across 1999–2023 yield ρ ∈ [0.2, 0.6], well within bounds.
- More informative priors (e.g., N(0.4, 0.15)) were considered but risk anchoring on a small training sample.
- The uniform prior is standard in the spatial econometrics literature (LeSage & Pace, 2009).

## Consequences
- The posterior for ρ will be data-driven.
- Sensitivity analysis in backtests should verify that results are robust to tighter priors.
