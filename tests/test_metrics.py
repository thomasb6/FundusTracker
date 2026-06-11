"""Tests des métriques cliniques de plus haut niveau :
- soustraction des zones d'exclusion (logique « Boolean-like », manuscrit l.140)
- test de permutation (comparaison statistique globale, manuscrit l.182-184)
"""
import numpy as np
import pytest

import main

EN = "en"


# ── Zones d'exclusion ──────────────────────────────────────────────────────────
class TestEffectiveAreas:
    def _poly(self, coords, label="Lesion"):
        path = "M " + " L ".join(f"{x},{y}" for x, y in coords) + " Z"
        return {"type": "path", "path": path, "customdata": label}

    def test_no_exclusion_keeps_raw_area(self):
        shape = self._poly([(0, 0), (10, 0), (10, 10), (0, 10)])
        results = main.calculate_effective_areas([shape], EN)
        assert results[0]["raw_area"] == 100.0
        assert results[0]["effective_area"] == 100.0

    def test_exclusion_inside_lesion_is_subtracted(self):
        lesion = self._poly([(0, 0), (20, 0), (20, 20), (0, 20)])           # 400
        hole = self._poly([(5, 5), (10, 5), (10, 10), (5, 10)], "Exclusion")  # 25
        results = main.calculate_effective_areas([lesion, hole], EN)
        lesion_res = next(r for r in results if r["original_index"] == 0)
        assert lesion_res["raw_area"] == 400.0
        assert lesion_res["effective_area"] == 375.0  # 400 - 25

    def test_effective_area_never_negative(self):
        small = self._poly([(0, 0), (4, 0), (4, 4), (0, 4)])                # 16
        big_hole = self._poly([(0, 0), (10, 0), (10, 10), (0, 10)], "Exclusion")
        results = main.calculate_effective_areas([small, big_hole], EN)
        small_res = next(r for r in results if r["original_index"] == 0)
        assert small_res["effective_area"] >= 0  # max(0, ...) du manuscrit


# ── Test de permutation ────────────────────────────────────────────────────────
class TestPermutationTest:
    def test_p_value_in_unit_interval(self):
        rng = np.random.RandomState(0)
        np.random.seed(0)
        a = rng.normal(100, 10, 8)
        b = rng.normal(105, 10, 8)
        p = main.calculate_permutation_test(a, b, num_permutations=2000)
        assert 0.0 <= p <= 1.0

    def test_identical_distributions_not_significant(self):
        np.random.seed(42)
        areas = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        p = main.calculate_permutation_test(areas, areas.copy(), num_permutations=2000)
        # Différence observée nulle → quasiment toutes les permutations sont
        # « au moins aussi extrêmes » → p proche de 1.
        assert p > 0.9

    def test_large_separation_is_significant(self):
        np.random.seed(7)
        low = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        high = np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
        p = main.calculate_permutation_test(low, high, num_permutations=5000)
        assert p < 0.05
