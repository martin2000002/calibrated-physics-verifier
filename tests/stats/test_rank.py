"""Kendall tau between injected levels and measured magnitudes."""

from __future__ import annotations

import math

import pytest

from physverify.stats.rank import MIN_RANKED_LEVELS, kendall


def test_below_the_minimum_number_of_levels_tau_is_nan() -> None:
    assert MIN_RANKED_LEVELS == 2
    assert math.isnan(kendall([]))
    assert math.isnan(kendall([(0.1, 1.0)]))


@pytest.mark.parametrize(
    ("pairs", "tau"),
    [
        ([(0.1, 1.0), (0.2, 2.0), (0.4, 3.0)], 1.0),
        ([(0.1, 3.0), (0.2, 2.0), (0.4, 1.0)], -1.0),
        ([(0.1, 1.0), (0.2, 3.0), (0.4, 2.0)], 1.0 / 3.0),
    ],
)
def test_tau_counts_concordant_minus_discordant_pairs(
    pairs: list[tuple[float, float]], tau: float
) -> None:
    assert kendall(pairs) == pytest.approx(tau)
