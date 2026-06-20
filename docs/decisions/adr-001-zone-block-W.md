# ADR-001: Zone-Block Spatial Weight Matrix

## Status
Accepted

## Context
The spatial panel model requires a weight matrix W defining "neighbourhood" relationships among the 37 states + FCT. Options considered:

1. **Contiguity (queen)**: W[i,j] = 1 if states share a border. Reflects physical proximity.
2. **Zone-block**: W[i,j] = 1 if states are in the same geopolitical zone. Reflects political/ethnic clustering.
3. **Hybrid**: Average of contiguity and zone-block.

## Decision
We use **zone-block** as the default W specification.

## Rationale
- Nigerian voting behaviour is heavily clustered by geopolitical zone (North-West, South-West, etc.), more so than by physical adjacency.
- Zone-block W captures the dominant spatial correlation structure in backtests (lower Brier score on 2019/2023 holdouts vs. contiguity).
- The hybrid was competitive but added complexity without statistically significant improvement.
- Contiguity W is retained as a robustness check in the GMM cross-check layer.

## Consequences
- The fundamentals model's ρ estimate reflects zone-level contagion, not border effects.
- States like Lagos (South-West) are linked to Ogun, Oyo, Osun, Ondo, Ekiti — all in the same zone — rather than to only its geographic neighbours.
- If future elections show a break from zonal voting patterns, W should be revisited.
