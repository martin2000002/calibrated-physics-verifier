"""The declared decision rules: Part 1, the Part 2 conjunction and the three states of Part 3."""

from __future__ import annotations

import pytest

from physverify.baselines.rules import (
    CONFIRMED,
    NO_DIFFERENCE,
    NOT_EVALUABLE,
    REFUTED,
    Verdict,
    p1,
    part2,
    part3,
)
from physverify.baselines.stats import Estimate
from physverify.stats.intervals import Ci, Rate


def estimate(value: float, low: float, high: float) -> Estimate:
    return Estimate(value, Ci(low=low, high=high), 100)


def quiet(value: float = 0.05, high: float = 0.15) -> Rate:
    return Rate(value=value, ci=Ci(low=0.0, high=high), k=1, n=20)


@pytest.mark.parametrize(
    ("delta", "unread", "expected"),
    [
        (estimate(0.2, 0.05, 0.3), 0.0, CONFIRMED),
        (estimate(0.2, -0.01, 0.3), 0.0, REFUTED),
        (estimate(0.2, 0.05, 0.3), 0.11, NOT_EVALUABLE),
        (estimate(0.2, 0.05, 0.3), 0.10, CONFIRMED),
        (None, 0.0, NOT_EVALUABLE),
    ],
)
def test_part1_rule(delta: Estimate | None, unread: float, expected: Verdict) -> None:
    assert p1(delta, unread, 0.10) == expected


DETECTS = estimate(0.3, 0.1, 0.5)


@pytest.mark.parametrize(
    ("real", "detected", "expected"),
    [
        ({"a": quiet(), "b": quiet()}, DETECTS, CONFIRMED),
        ({"a": quiet(), "b": quiet(0.12)}, DETECTS, REFUTED),
        ({"a": quiet(), "b": quiet(high=0.25)}, DETECTS, REFUTED),
        ({"a": quiet()}, estimate(0.1, 0.0, 0.2), REFUTED),
        ({}, DETECTS, NOT_EVALUABLE),
        ({"a": quiet()}, None, NOT_EVALUABLE),
        ({"a": Rate(value=None, ci=Ci(), k=0, n=0)}, DETECTS, NOT_EVALUABLE),
    ],
)
def test_part2_is_a_conjunction(
    real: dict[str, Rate], detected: Estimate | None, expected: Verdict
) -> None:
    assert part2(real, detected, 0.10, 0.20) == expected


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-0.3, -0.01, CONFIRMED),
        (-0.1, 0.1, NO_DIFFERENCE),
        (0.01, 0.2, REFUTED),
        (None, 0.2, NOT_EVALUABLE),
    ],
)
def test_part3_has_three_states(low: float | None, high: float | None, expected: Verdict) -> None:
    assert part3(low, high) == expected
