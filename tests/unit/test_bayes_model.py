"""Bayesian national-swing + SAR state model: reconciliation and parameter recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ngforecast.data.geography import build_zone_block_matrix, load_constitution_units, row_standardize
from ngforecast.features.panel import logit
from ngforecast.models.fundamentals.bayes import SamplerConfig, _reconcile, fit, predict

STATES = sorted(u["code"] for u in load_constitution_units())


def _W():
    codes = [u["code"] for u in load_constitution_units()]
    W = row_standardize(build_zone_block_matrix())
    order = [codes.index(s) for s in STATES]
    return W[np.ix_(order, order)]


def simulate_panel(rho=0.4, phi=0.7, beta=(0.6, -0.3), sigma=0.15, years=range(2003, 2031, 4), seed=0):
    """Panel drawn from the model itself, with two covariates."""
    rng = np.random.default_rng(seed)
    W = _W()
    n = len(STATES)
    A_inv = np.linalg.inv(np.eye(n) - rho * W)
    weight = rng.dirichlet(np.full(n, 20))
    d_prev = rng.normal(0, 0.5, n)
    nat_prev = 0.55
    rows = []
    for y in years:
        x1 = (rng.random(n) < 0.2).astype(float)
        x2 = rng.normal(0, 1, n)
        d = A_inv @ (phi * d_prev + beta[0] * x1 + beta[1] * x2 + rng.normal(0, sigma, n))
        nat = expit(logit(nat_prev) + rng.normal(-0.05, 0.2))
        shares = _reconcile((logit(nat) + d)[None, :], weight, np.array([nat]))[0]
        dev = logit(shares) - logit(nat)
        for i, s in enumerate(STATES):
            rows.append(
                {
                    "year": y,
                    "state": s,
                    "x1": x1[i],
                    "x2": x2[i],
                    "dev": dev[i],
                    "dev_lag": d_prev[i],
                    "inc_share": shares[i],
                    "nat_share": nat,
                    "nat_share_lag": nat_prev,
                    "incumbent_running": int(rng.random() < 0.5),
                    "weight": weight[i],
                }
            )
        d_prev, nat_prev = dev, nat
    return pd.DataFrame(rows), W


def test_reconcile_matches_national_target():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, (50, 37))
    w = rng.dirichlet(np.ones(37))
    target = rng.uniform(0.3, 0.7, 50)
    p = _reconcile(logits, w, target)
    assert np.allclose(p @ (w / w.sum()), target, atol=1e-8)
    # order of states within a draw is preserved
    assert np.all(np.argsort(p[0]) == np.argsort(logits[0]))


@pytest.mark.slow
def test_parameter_recovery_and_prediction():
    panel, W = simulate_panel()
    train, test = panel[panel.year < 2027], panel[panel.year == 2027]
    cfg = SamplerConfig(draws=400, tune=400, chains=2, seed=3)
    model = fit(train, W, STATES, covariates=["x1", "x2"], cfg=cfg)

    assert model.diagnostics["state"]["max_rhat"] < 1.05
    assert np.mean(model.state["rho"]) == pytest.approx(0.4, abs=0.2)
    assert np.mean(model.state["phi"]) == pytest.approx(0.7, abs=0.15)
    assert np.mean(model.state["beta"][:, 0]) == pytest.approx(0.6, abs=0.2)
    assert np.mean(model.state["beta"][:, 1]) == pytest.approx(-0.3, abs=0.15)

    pred = predict(model, test, W, national_override=float(test.nat_share.iloc[0]))
    lo, hi = np.percentile(pred["state"], [5, 95], axis=0)
    covered = np.mean((test.sort_values("state").inc_share.to_numpy() >= lo) & (test.sort_values("state").inc_share.to_numpy() <= hi))
    assert covered >= 0.75
