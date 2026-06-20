"""Versioned forecast schema — pydantic models for forecast artifacts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CandidateForecast(BaseModel):
    """Forecast for a single candidate."""

    candidate_id: str
    party: str
    win_probability: float = Field(ge=0, le=1)
    national_vote_share_mean: float = Field(ge=0, le=1)
    national_vote_share_ci_lower: float = Field(ge=0, le=1)
    national_vote_share_ci_upper: float = Field(ge=0, le=1)
    states_above_25pct_mean: float = Field(ge=0)
    state_vote_shares: dict[str, float] = Field(default_factory=dict)


class ForecastArtifact(BaseModel):
    """Top-level dated forecast object — the deliverable."""

    schema_version: str = "1.0.0"
    forecast_date: datetime
    election_date: str
    n_simulations: int
    seed: int
    scenario_weights: dict[str, float] = Field(default_factory=dict)
    runoff_probability: float = Field(ge=0, le=1)
    candidates: list[CandidateForecast]
    ensemble_weights: dict[str, float] = Field(default_factory=dict)
    backtest_scores: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class ForecastLogEntry(BaseModel):
    """Single entry in the append-only forecast log."""

    forecast_date: datetime
    win_probabilities: dict[str, float]
    runoff_probability: float
    model_version: str
    data_vintage: str
    notes: str = ""
