"""The relative sigma of L = (e_x e_y e_z)^(1/3) against Monte Carlo draws of the extents."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.physics.geometry import characteristic_lengths
from physverify.physics.length_sigma import DEFAULT_AXIS_ERRORS, length_rel_sigma

EXTENTS = np.array([0.2, 0.3, 0.25])
DRAWS = 400_000


def simulated(common: float, per_axis: float) -> float:
    """Relative std of L when every extent shares one factor and each also draws its own."""
    rng = np.random.default_rng(0)
    shared = rng.normal(0.0, common, (DRAWS, 1))
    own = rng.normal(0.0, per_axis, (DRAWS, 3))
    lengths = characteristic_lengths(EXTENTS * np.exp(shared + own))
    return float(np.std(np.log(lengths)))


def test_the_default_is_correlated() -> None:
    assert DEFAULT_AXIS_ERRORS == "correlated"


def test_equal_axis_errors_differ_by_root_three() -> None:
    rel = np.full(3, 0.1)
    correlated = float(length_rel_sigma(rel, "correlated"))
    independent = float(length_rel_sigma(rel, "independent"))
    assert correlated == pytest.approx(0.1, abs=1e-15)
    assert correlated == pytest.approx(math.sqrt(3.0) * independent, rel=1e-12)


def test_unequal_axes_give_the_mean_and_the_quadrature() -> None:
    rel = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    assert length_rel_sigma(rel, "correlated") == pytest.approx([0.2, 0.0], abs=1e-15)
    expected = math.sqrt(0.01 + 0.04 + 0.09) / 3.0
    assert length_rel_sigma(rel, "independent") == pytest.approx([expected, 0.0], abs=1e-15)


@pytest.mark.parametrize("axis_errors", ["correlated", "independent"])
def test_nan_propagates(axis_errors: str) -> None:
    rel = np.array([0.1, math.nan, 0.1])
    assert math.isnan(float(length_rel_sigma(rel, axis_errors)))  # type: ignore[arg-type]


def test_a_common_factor_matches_the_correlated_formula() -> None:
    formula = float(length_rel_sigma(np.full(3, 0.1), "correlated"))
    assert formula == pytest.approx(simulated(0.1, 0.0), rel=0.01)


def test_independent_axes_match_the_independent_formula() -> None:
    formula = float(length_rel_sigma(np.full(3, 0.1), "independent"))
    assert formula == pytest.approx(simulated(0.0, 0.1), rel=0.01)


def test_a_common_and_a_per_axis_term_are_bounded_by_the_mean() -> None:
    """r_j = sqrt(c^2 + q^2): exact sqrt(c^2 + q^2 / 3); the mean is about 1.3 % high here."""
    common, per_axis = 0.40, 0.08
    rel = np.full(3, math.hypot(common, per_axis))
    bound = float(length_rel_sigma(rel, "correlated"))
    exact = math.sqrt(common**2 + per_axis**2 / 3.0)
    assert simulated(common, per_axis) == pytest.approx(exact, rel=0.01)
    assert 1.0 < bound / exact < 1.02
