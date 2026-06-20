"""Unit tests for scoring metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ngforecast.calibration.scores import brier_score, coverage, log_score, mae, rmse


class TestBrierScore:
    def test_perfect_prediction(self):
        pred = np.array([1.0, 0.0, 1.0])
        actual = np.array([1.0, 0.0, 1.0])
        assert brier_score(pred, actual) == pytest.approx(0.0)

    def test_worst_prediction(self):
        pred = np.array([0.0, 1.0])
        actual = np.array([1.0, 0.0])
        assert brier_score(pred, actual) == pytest.approx(1.0)


class TestRMSE:
    def test_perfect(self):
        x = np.array([0.5, 0.3, 0.7])
        assert rmse(x, x) == pytest.approx(0.0)

    def test_known_value(self):
        pred = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 4.0])
        assert rmse(pred, actual) == pytest.approx(np.sqrt(1 / 3))


class TestCoverage:
    def test_full_coverage(self):
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        actual = np.array([0.5, 0.5])
        assert coverage(lower, upper, actual) == pytest.approx(1.0)

    def test_no_coverage(self):
        lower = np.array([0.6, 0.6])
        upper = np.array([0.7, 0.7])
        actual = np.array([0.1, 0.1])
        assert coverage(lower, upper, actual) == pytest.approx(0.0)
