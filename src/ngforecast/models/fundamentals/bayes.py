"""Bayesian national-swing + spatial state-deviation model (PyMC).

National:  logit N_t - logit N_{t-1} = mu + gamma * incumbent_running_t + e_t
State:     d_t = rho W d_t + phi d_{t-1} + X_t beta + alpha + u_state + eps_t
           (SAR in the state deviations d = logit y - logit N; W row-standardised)

Prediction draws national swing and state deviations from the posterior,
then shifts each draw's state logits by a common constant so that the
vote-weighted mean of state shares equals the national draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from loguru import logger
from scipy.special import expit

from ngforecast.features.panel import logit

DEFAULT_COVARIATES = [
    "home_state",
    "home_zone",
    "opp_home_zone",
    "home_state_lag",
    "home_zone_lag",
    "opp_home_zone_lag",
    "violence_z",
]


@dataclass
class SamplerConfig:
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.99  # 0.95 left divergences in 2-observation national fits (protocol §7)
    seed: int = 2027


@dataclass
class FittedModel:
    national: dict  # posterior draws: mu, gamma, sigma
    state: dict  # posterior draws: rho, phi, beta, alpha, u, sigma
    covariates: list[str]
    states: list[str]
    spatial: bool
    phi_fixed: float | None = None
    diagnostics: dict = field(default_factory=dict)


# ── National swing ───────────────────────────


def fit_national(train: pd.DataFrame, cfg: SamplerConfig) -> tuple[dict, dict]:
    nat = train.groupby("year").first()
    swing = (logit(nat["nat_share"]) - logit(nat["nat_share_lag"])).to_numpy()
    running = nat["incumbent_running"].to_numpy(dtype=float)

    with pm.Model():
        mu = pm.Normal("mu", 0.0, 0.5)
        gamma = pm.Normal("gamma", 0.0, 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)
        pm.Normal("swing", mu + gamma * running, sigma, observed=swing)
        idata = pm.sample(
            cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.seed,
            progressbar=False,
        )
    return _flatten(idata, ["mu", "gamma", "sigma"]), _diagnostics(idata)


# ── State deviations (SAR) ───────────────────


def fit_state(
    train: pd.DataFrame,
    W: np.ndarray,
    states: list[str],
    covariates: list[str],
    cfg: SamplerConfig,
    spatial: bool = True,
    phi_fixed: float | None = None,
    phi_prior: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict, dict]:
    """Fit the state-deviation model.

    ``phi_fixed=1.0`` gives the swing formulation: the change in a state's
    deviation, d_t - d_{t-1}, is modelled (spatially, if ``spatial``), so the
    model nests uniform swing (beta = alpha = u = 0, sigma -> 0).
    """
    train = train.sort_values(["year", "state"])
    years = sorted(train["year"].unique())
    n = len(states)
    eig = np.real(np.linalg.eigvals(W))

    d = np.stack([train.loc[train.year == y, "dev"].to_numpy() for y in years])  # T x n
    d_lag = np.stack([train.loc[train.year == y, "dev_lag"].to_numpy() for y in years])
    X = np.stack([train.loc[train.year == y, covariates].to_numpy(float) for y in years])

    # Levels: (I - rho W) d_t = phi d_{t-1} + X beta + ...
    # Swing:  (I - rho W)(d_t - phi d_{t-1}) = X beta + ...   (phi fixed)
    y = d if phi_fixed is None else d - phi_fixed * d_lag
    Wy = y @ W.T  # (W y_t) for each year, as rows

    with pm.Model():
        rho = pm.Uniform("rho", -1.0, 1.0) if spatial else 0.0
        phi = pm.Normal("phi", *phi_prior) if phi_fixed is None else 0.0
        k = len(covariates)
        beta = pm.Normal("beta", 0.0, 0.5, shape=k) if k else None
        alpha = pm.Normal("alpha", 0.0, 0.5)
        tau = pm.HalfNormal("tau", 0.3)
        u_raw = pm.Normal("u_raw", 0.0, 1.0, shape=n)
        u = pm.Deterministic("u", tau * u_raw)
        sigma = pm.HalfNormal("sigma", 0.5)

        x_term = pt.tensordot(X, beta, axes=[[2], [0]]) if k else 0.0
        mean = alpha + phi * d_lag + x_term + u[None, :]
        resid = y - rho * Wy - mean
        T = len(years)
        logdet = pt.sum(pt.log1p(-rho * eig)) if spatial else 0.0
        pm.Potential(
            "sar_loglik",
            T * logdet
            - T * n * pt.log(sigma)
            - 0.5 * pt.sum(resid**2) / sigma**2,
        )
        idata = pm.sample(
            cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.seed,
            progressbar=False,
        )
    names = (["beta"] if k else []) + ["alpha", "u", "sigma", "tau"] + (["rho"] if spatial else [])
    names += ["phi"] if phi_fixed is None else []
    draws = _flatten(idata, names)
    if not k:
        draws["beta"] = np.zeros((len(draws["sigma"]), 0))
    if phi_fixed is not None:
        draws["phi"] = np.full(len(draws["sigma"]), float(phi_fixed))
    return draws, _diagnostics(idata)


def fit(
    train: pd.DataFrame,
    W: np.ndarray,
    states: list[str],
    covariates: list[str] | None = None,
    cfg: SamplerConfig | None = None,
    spatial: bool = True,
    phi_fixed: float | None = None,
    phi_prior: tuple[float, float] = (0.0, 1.0),
) -> FittedModel:
    cfg = cfg or SamplerConfig()
    covariates = DEFAULT_COVARIATES if covariates is None else covariates
    national, nat_diag = fit_national(train, cfg)
    state, st_diag = fit_state(
        train, W, states, covariates, cfg, spatial=spatial, phi_fixed=phi_fixed, phi_prior=phi_prior
    )
    logger.info(
        "Fitted on {}: rho={:.2f}, phi={:.2f}, max r_hat={:.3f}",
        sorted(train.year.unique()),
        float(np.mean(state.get("rho", 0.0))),
        float(np.mean(state["phi"])),
        max(nat_diag["max_rhat"], st_diag["max_rhat"]),
    )
    return FittedModel(
        national=national,
        state=state,
        covariates=covariates,
        states=states,
        spatial=spatial,
        phi_fixed=phi_fixed,
        diagnostics={"national": nat_diag, "state": st_diag},
    )


# ── Prediction ───────────────────────────────


def predict(
    model: FittedModel,
    test: pd.DataFrame,
    W: np.ndarray,
    seed: int = 0,
    national_override: float | None = None,
) -> dict[str, np.ndarray]:
    """Posterior predictive draws for one election.

    Returns {"state": (S, n) incumbent-party shares, "national": (S,)}.
    `national_override` fixes the national share (for "oracle national"
    evaluation of the state model alone).
    """
    rng = np.random.default_rng(seed)
    test = test.sort_values("state")
    assert list(test.state) == model.states, "test rows must cover all states in order"
    n = len(model.states)
    st, nt = model.state, model.national
    S = len(st["phi"])

    # National
    lag_nat = float(test["nat_share_lag"].iloc[0])
    running = float(test["incumbent_running"].iloc[0])
    if national_override is None:
        swing = rng.normal(nt["mu"] + nt["gamma"] * running, nt["sigma"])
        nat = expit(logit(lag_nat) + swing)
    else:
        nat = np.full(S, national_override)

    # State deviations
    X = test[model.covariates].to_numpy(float).reshape(len(test), len(model.covariates))
    d_lag = test["dev_lag"].to_numpy()
    swing = model.phi_fixed is not None
    lag_term = st["phi"][:, None] * d_lag[None, :]
    mean = st["alpha"][:, None] + st["beta"] @ X.T + st["u"] + (0.0 if swing else lag_term)
    eps = rng.normal(0.0, st["sigma"][:, None], size=(S, n))
    if model.spatial:
        filtered = np.empty((S, n))
        eye = np.eye(n)
        for s in range(S):
            filtered[s] = np.linalg.solve(eye - st["rho"][s] * W, mean[s] + eps[s])
    else:
        filtered = mean + eps
    # Swing: the spatial filter applies to the change only, then add the lag back
    dev = filtered + lag_term if swing else filtered

    w = test["weight"].to_numpy()
    shares = _reconcile(logit(nat)[:, None] + dev, w, nat)
    return {"state": shares, "national": nat}


def _reconcile(logits: np.ndarray, w: np.ndarray, target: np.ndarray, iters: int = 30):
    """Shift each row of logits by c so that sum_i w_i expit(logit_i + c) = target."""
    w = w / w.sum()
    c = np.zeros(logits.shape[0])
    for _ in range(iters):
        p = expit(logits + c[:, None])
        f = p @ w - target
        fp = (p * (1 - p)) @ w
        c -= f / np.maximum(fp, 1e-9)
    return expit(logits + c[:, None])


# ── Helpers ──────────────────────────────────


def _flatten(idata, names: list[str]) -> dict[str, np.ndarray]:
    post = idata.posterior
    out = {}
    for name in names:
        arr = post[name].to_numpy()
        out[name] = arr.reshape(-1, *arr.shape[2:])
    return out


def _diagnostics(idata) -> dict:
    import arviz as az

    summ = az.summary(idata, kind="diagnostics")
    return {
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }
