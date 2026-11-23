"""Unit tests for geography module."""

from __future__ import annotations

import numpy as np
import pytest

from ngforecast.data.geography import (
    build_zone_block_matrix,
    build_zone_map,
    load_constitution_units,
    row_standardize,
    state_code,
)


class TestConstitutionUnits:
    def test_37_units(self):
        units = load_constitution_units()
        assert len(units) == 37

    def test_all_have_code_name_zone(self):
        units = load_constitution_units()
        for u in units:
            assert "code" in u
            assert "name" in u
            assert "zone" in u

    def test_fct_included(self):
        units = load_constitution_units()
        codes = [u["code"] for u in units]
        assert "FC" in codes

    def test_six_zones(self):
        zone_map = build_zone_map()
        zones = set(zone_map.values())
        assert len(zones) == 6


class TestZoneBlockMatrix:
    def test_shape(self):
        W = build_zone_block_matrix()
        assert W.shape == (37, 37)

    def test_diagonal_zero(self):
        W = build_zone_block_matrix()
        assert np.all(np.diag(W) == 0)

    def test_symmetric(self):
        W = build_zone_block_matrix()
        np.testing.assert_array_equal(W, W.T)

    def test_same_zone_connected(self):
        W = build_zone_block_matrix()
        units = load_constitution_units()
        # Find two states in the same zone
        zone_states = {}
        for i, u in enumerate(units):
            zone_states.setdefault(u["zone"], []).append(i)
        for zone, indices in zone_states.items():
            if len(indices) >= 2:
                assert W[indices[0], indices[1]] == 1.0


class TestRowStandardize:
    def test_rows_sum_to_one(self):
        W = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        Ws = row_standardize(W)
        np.testing.assert_allclose(Ws.sum(axis=1), 1.0)

    def test_isolate_handled(self):
        W = np.array([[0, 0], [1, 0]], dtype=float)
        Ws = row_standardize(W)
        assert Ws[0].sum() == 0.0  # isolate stays zero
        assert Ws[1].sum() == pytest.approx(1.0)


class TestStateCodes:
    def test_codes_are_strings(self):
        """YAML parses bare ON/NO/YES as booleans; every code must stay a string."""
        for u in load_constitution_units():
            assert isinstance(u["code"], str) and len(u["code"]) == 2, u

    @pytest.mark.parametrize(
        "name,code",
        [
            ("Ondo", "ON"),
            ("FCT", "FC"),
            ("Federal Capital Territory", "FC"),
            ("FCT Abuja", "FC"),
            ("Nassarawa", "NA"),
            ("Nasarawa", "NA"),
            ("AKWA IBOM", "AK"),
            ("Cross-River", "CR"),
            ("Lagos State", "LA"),
        ],
    )
    def test_name_variants(self, name, code):
        assert state_code(name) == code

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            state_code("Atlantis")
