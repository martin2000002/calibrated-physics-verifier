"""Check weights from the informedness lower bound (AC9) and the monotone balanced Platt fit."""

from __future__ import annotations

import numpy as np
import pytest
from calibration_cases import settings

from physverify.calibration.artifact import Weight
from physverify.calibration.check_weights import (
    Flag,
    Labelled,
    fit_probability,
    normalised,
    raw_weight,
)
from physverify.stats.intervals import Ci


def flags(rng: np.random.Generator, tpr: float, fpr: float, n: int = 20) -> list[Flag]:
    pos = [Flag(f"p{i}", True, bool(rng.random() < tpr)) for i in range(n)]
    neg = [Flag(f"n{i}", False, bool(rng.random() < fpr)) for i in range(n)]
    return pos + neg


def test_an_uninformative_check_gets_weight_zero() -> None:
    """AC9: TPR = FPR, weight 0 in >= 95 % of 100 replicates."""
    rng = np.random.default_rng(0)
    config = settings(bootstrap=100)
    zeros = [raw_weight(flags(rng, 0.3, 0.3), config).value == 0.0 for _ in range(100)]
    assert float(np.mean(zeros)) >= 0.95


def test_an_informative_check_gets_a_positive_weight() -> None:
    found = raw_weight(flags(np.random.default_rng(1), 0.9, 0.05), settings())
    assert found.value > 0.0
    assert found.informedness is not None and found.informedness > found.value
    assert (found.n_positive, found.n_negative) == (20, 20)


def test_weights_are_normalised_by_the_largest() -> None:
    def weight(value: float) -> Weight:
        return Weight(
            value=value, informedness=None, informedness_ci=Ci(), n_positive=0, n_negative=0
        )

    found = normalised({"a": weight(0.4), "b": weight(0.2)})
    assert (found["a"].value, found["b"].value) == (1.0, 0.5)
    assert normalised({"a": weight(0.0)})["a"].value == 0.0


def labelled(rng: np.random.Generator, n_pos: int, n_neg: int) -> list[Labelled]:
    pos = [Labelled(f"p{i}", float(m), True) for i, m in enumerate(rng.normal(1.0, 0.3, n_pos))]
    neg = [Labelled(f"n{i}", float(m), False) for i, m in enumerate(rng.normal(0.0, 0.3, n_neg))]
    return pos + neg


def predict(fit_a: float, fit_b: float, scale: float, tau: float, m: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(fit_a + fit_b * (m - tau) / scale))))


def test_platt_is_monotone_and_cross_fitted() -> None:
    found = fit_probability(labelled(np.random.default_rng(2), 40, 40), 0.5, settings())
    assert found.a is not None and found.b is not None and found.b > 0.0
    grid = [predict(found.a, found.b, found.scale, 0.5, m) for m in np.linspace(-1, 2, 7)]
    assert grid == sorted(grid)
    assert found.ece_cv is not None and 0.0 <= found.ece_cv <= 0.5
    assert found.estimated


def test_a_reversed_check_cannot_get_a_negative_slope() -> None:
    data = [
        Labelled(d.scene_id, -d.magnitude, d.positive)
        for d in labelled(np.random.default_rng(3), 30, 30)
    ]
    found = fit_probability(data, 0.0, settings())
    assert found.b == 0.0


def test_the_design_prevalence_does_not_move_p() -> None:
    """Class-balanced: repeating every negative four times leaves p nearly unchanged (only the
    fixed weak prior weighs less against the larger sample)."""
    data = labelled(np.random.default_rng(4), 20, 20)
    heavy = data + [d for d in data if not d.positive] * 3
    one, four = fit_probability(data, 0.5, settings()), fit_probability(heavy, 0.5, settings())
    assert None not in (one.a, one.b, four.a, four.b)
    for m in np.linspace(-0.5, 1.5, 9):
        p_one = predict(one.a, one.b, one.scale, 0.5, m)  # type: ignore[arg-type]
        p_four = predict(four.a, four.b, four.scale, 0.5, m)  # type: ignore[arg-type]
        assert p_one == pytest.approx(p_four, abs=0.05)


def test_without_tau_there_is_no_probability() -> None:
    found = fit_probability(labelled(np.random.default_rng(5), 5, 5), None, settings())
    assert (found.a, found.b, found.ece_cv) == (None, None, None)
