"""Exact binomial intervals."""

from __future__ import annotations

import json
import math

import pytest
from scipy.stats import binomtest

from physverify.stats.intervals import Interval, clopper_pearson, clopper_pearson_lower


@pytest.mark.parametrize(("k", "n"), [(0, 20), (3, 20), (20, 20), (1, 100), (7, 13)])
def test_clopper_pearson_is_the_exact_binomial_interval(k: int, n: int) -> None:
    expected = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    found = clopper_pearson(k, n, 0.95)
    assert found.low == pytest.approx(expected.low, abs=1e-12)
    assert found.high == pytest.approx(expected.high, abs=1e-12)


def test_zero_out_of_twenty_has_upper_bound_0_168() -> None:
    found = clopper_pearson(0, 20, 0.95)
    assert found.low == 0.0
    assert found.high == pytest.approx(0.168, abs=5e-4)
    assert clopper_pearson(0, 0, 0.95).low is None


@pytest.mark.parametrize(("k", "n", "low"), [(5, 5, 0.05 ** (1 / 5)), (20, 20, 0.05 ** (1 / 20))])
def test_the_one_sided_lower_bound_of_perfect_rates(k: int, n: int, low: float) -> None:
    assert clopper_pearson_lower(k, n, 0.95) == pytest.approx(low)
    assert clopper_pearson_lower(0, 0, 0.95) == 0.0


def test_an_interval_without_finite_bounds_reads_back_as_nan() -> None:
    written = Interval(low=math.nan, high=math.nan).to_json()
    assert json.loads(written) == {"high": None, "low": None}
    back = Interval.model_validate_json(written)
    assert math.isnan(back.low) and math.isnan(back.high)
    assert Interval.model_validate_json(Interval(low=0.1, high=0.4).to_json()) == Interval(
        low=0.1, high=0.4
    )
