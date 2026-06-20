"""Monte Carlo simulation engine — posterior × scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from ngforecast.simulation.constitution import check_win_rule
from ngforecast.simulation.errors import draw_correlated_errors
from ngforecast.simulation.runoff import simulate_runoff


def run_simulations(
    predicted_shares: np.ndarray,
    turnout: np.ndarray,
    registered_voters: np.ndarray,
    zone_assignments: np.ndarray,
    candidates: list[str],
    n_sims: int = 20_000,
    seed: int = 2027,
    config_path: Path = Path("config/constitution.yaml"),
) -> dict:
    """Run full Monte Carlo simulation over scenarios.

    Parameters
    ----------
    predicted_shares : (n_states, n_candidates) baseline predicted vote shares
    turnout : (n_states,) predicted turnout rates
    registered_voters : (n_states,) registered voter counts
    zone_assignments : (n_states,) zone index per state
    candidates : list of candidate/party IDs
    n_sims : number of Monte Carlo draws
    seed : random seed

    Returns
    -------
    results dict with win probabilities, runoff probability, state-level summaries
    """
    rng = np.random.default_rng(seed)
    n_states, n_cand = predicted_shares.shape

    with open(config_path) as f:
        const = yaml.safe_load(f)

    # Draw correlated errors
    errors = draw_correlated_errors(
        n_sims=n_sims,
        n_states=n_states,
        zone_assignments=zone_assignments,
        rng=rng,
    )

    wins = {c: 0 for c in candidates}
    runoff_count = 0
    state_shares_sum = np.zeros((n_states, n_cand))

    for s in range(n_sims):
        # Perturb shares
        sim_shares = predicted_shares + errors[s, :, np.newaxis]
        sim_shares = np.maximum(sim_shares, 0.001)
        sim_shares /= sim_shares.sum(axis=1, keepdims=True)

        # Compute votes
        sim_turnout = np.clip(turnout + errors[s] * 0.5, 0.1, 0.99)
        votes_by_state = sim_shares * (registered_voters[:, np.newaxis] * sim_turnout[:, np.newaxis])

        # National totals
        national_votes = votes_by_state.sum(axis=0)

        # Check constitutional win rule
        winner = check_win_rule(
            votes_by_state=votes_by_state,
            national_votes=national_votes,
            candidates=candidates,
            constitution=const,
        )

        if winner is not None:
            wins[winner] += 1
        else:
            runoff_count += 1
            # Simulate runoff between top 2
            runoff_winner = simulate_runoff(
                votes_by_state=votes_by_state,
                candidates=candidates,
                rng=rng,
            )
            if runoff_winner:
                wins[runoff_winner] += 1

        state_shares_sum += sim_shares

    # Compile results
    win_probs = {c: wins[c] / n_sims for c in candidates}
    avg_state_shares = state_shares_sum / n_sims

    results = {
        "n_sims": n_sims,
        "seed": seed,
        "win_probabilities": win_probs,
        "runoff_probability": runoff_count / n_sims,
        "avg_state_shares": avg_state_shares.tolist(),
        "candidates": candidates,
    }

    logger.info("Simulation complete: {} sims, runoff prob = {:.3f}", n_sims, results["runoff_probability"])
    for c, p in win_probs.items():
        logger.info("  {} win prob: {:.3f}", c, p)

    return results
