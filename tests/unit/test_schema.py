"""Unit tests for forecast schema validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ngforecast.forecast.schema import CandidateForecast, ForecastArtifact, ForecastLogEntry


class TestForecastSchema:
    def test_valid_artifact(self):
        artifact = ForecastArtifact(
            forecast_date=datetime.now(timezone.utc),
            election_date="2027-02-27",
            n_simulations=20000,
            seed=2027,
            runoff_probability=0.15,
            candidates=[
                CandidateForecast(
                    candidate_id="APC",
                    party="APC",
                    win_probability=0.45,
                    national_vote_share_mean=0.38,
                    national_vote_share_ci_lower=0.32,
                    national_vote_share_ci_upper=0.44,
                    states_above_25pct_mean=28.5,
                ),
            ],
        )
        assert artifact.schema_version == "1.0.0"
        assert len(artifact.candidates) == 1

    def test_invalid_probability_rejected(self):
        with pytest.raises(Exception):
            CandidateForecast(
                candidate_id="X",
                party="X",
                win_probability=1.5,  # > 1
                national_vote_share_mean=0.5,
                national_vote_share_ci_lower=0.4,
                national_vote_share_ci_upper=0.6,
                states_above_25pct_mean=25,
            )

    def test_log_entry_roundtrip(self):
        entry = ForecastLogEntry(
            forecast_date=datetime.now(timezone.utc),
            win_probabilities={"APC": 0.45, "PDP": 0.35, "LP": 0.15, "NNPP": 0.05},
            runoff_probability=0.12,
            model_version="0.1.0",
            data_vintage="2025-Q2",
        )
        json_str = entry.model_dump_json()
        roundtrip = ForecastLogEntry.model_validate_json(json_str)
        assert roundtrip.win_probabilities == entry.win_probabilities
