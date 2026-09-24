"""Factorial arithmetic (ADR 0007 §5.4, AC6): Yates, Shapley, budget and the clip bootstrap."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from physverify.diagnosis.factorial import (
    ALL_PERCEIVED,
    ALL_TRUTH,
    MEAN,
    bootstrap,
    budget,
    effect_name,
    effect_terms,
    shapley,
    yates_effects,
)
from physverify.diagnosis.plans import CELLS, FACTORS, Cell
from physverify.diagnosis.report import Interval
from physverify.schema.stages import Stage

TOL = 1e-12
INTERCEPT = 1.0
COEFFICIENTS = {
    "track": 0.30,
    "depth": -0.20,
    "scale": 0.05,
    "track:depth": 0.10,
    "track:scale": 0.0,
    "depth:scale": -0.07,
    "track:depth:scale": 0.02,
}


def linear_means(coefficients: Mapping[str, float]) -> dict[Cell, float]:
    """y_c = a + sum_S b_S prod_{f in S} x_f(c), x = +1 perceived, -1 ground truth."""

    def coded(cell: Cell, term: Sequence[int]) -> float:
        return math.prod(1.0 if cell[i] else -1.0 for i in term)

    return {
        cell: INTERCEPT
        + sum(coefficients.get(effect_name(t), 0.0) * coded(cell, t) for t in effect_terms())
        for cell in CELLS
    }


def test_effect_names_follow_the_factor_order() -> None:
    assert [effect_name(t) for t in effect_terms()] == list(COEFFICIENTS)


def test_yates_recovers_a_known_linear_model() -> None:
    effects = yates_effects(linear_means(COEFFICIENTS))
    assert set(effects) == {MEAN, *COEFFICIENTS}
    assert effects[MEAN] == pytest.approx(INTERCEPT, abs=TOL)
    for name, b in COEFFICIENTS.items():
        assert abs(effects[name] - 2.0 * b) <= TOL, name


def test_main_effect_is_the_difference_of_half_means() -> None:
    means = linear_means(COEFFICIENTS)
    effects = yates_effects(means)
    for i, stage in enumerate(FACTORS):
        high = np.mean([means[c] for c in CELLS if c[i]])
        low = np.mean([means[c] for c in CELLS if not c[i]])
        assert abs(effects[stage.value] - (high - low)) <= TOL


def test_shapley_adds_up_to_all_perceived_minus_all_truth() -> None:
    means = linear_means(COEFFICIENTS)
    phi = shapley(means)
    assert tuple(phi) == FACTORS
    assert abs(sum(phi.values()) - (means[ALL_PERCEIVED] - means[ALL_TRUTH])) <= TOL


def test_shapley_of_an_additive_model_is_twice_each_coefficient() -> None:
    additive = {"track": 0.3, "depth": -0.2, "scale": 0.05}
    phi = shapley(linear_means(additive))
    for stage in FACTORS:
        assert abs(phi[stage] - 2.0 * additive[stage.value]) <= TOL


def test_shapley_splits_a_pure_interaction_evenly() -> None:
    means = {cell: 1.0 if cell[0] and cell[1] else 0.0 for cell in CELLS}
    phi = shapley(means)
    assert phi[Stage.TRACK] == pytest.approx(0.5, abs=TOL)
    assert phi[Stage.DEPTH] == pytest.approx(0.5, abs=TOL)
    assert phi[Stage.SCALE] == pytest.approx(0.0, abs=TOL)


def test_budget_rows_add_up_to_v0() -> None:
    means = linear_means(COEFFICIENTS)
    v0 = means[ALL_PERCEIVED] + 0.4
    rows = budget(means, v0)
    assert rows.floor == means[ALL_TRUTH]
    assert abs(rows.fixed - 0.4) <= TOL
    labelled = rows.rows("detect+camera")
    assert [name for name, _ in labelled] == ["floor", "track", "depth", "scale", "detect+camera"]
    assert abs(math.fsum(value for _, value in labelled) - v0) <= TOL
    assert rows.total == v0


def test_missing_cells_are_rejected() -> None:
    means = linear_means(COEFFICIENTS)
    del means[ALL_TRUTH]
    with pytest.raises(ValueError, match="missing"):
        yates_effects(means)
    with pytest.raises(ValueError, match="missing"):
        shapley(means)


@dataclass(frozen=True)
class Item:
    clip_id: str
    value: float


def mean_value(items: Sequence[Item]) -> dict[str, float]:
    return {"mean": float(np.mean([i.value for i in items])), "nan": math.nan}


ITEMS = tuple(Item(f"clip{k % 4}", float(k)) for k in range(12))


def test_bootstrap_is_deterministic_with_the_seed() -> None:
    first = bootstrap(ITEMS, mean_value, n=200, seed=0, ci=0.9)
    again = bootstrap(ITEMS, mean_value, n=200, seed=0, ci=0.9)
    other = bootstrap(ITEMS, mean_value, n=200, seed=1, ci=0.9)
    assert first == again
    assert first.intervals["mean"] != other.intervals["mean"]
    interval = first.intervals["mean"]
    assert interval.low < float(np.mean([i.value for i in ITEMS])) < interval.high
    assert (first.replicas, first.dropped) == (200, 0)


def test_bootstrap_resamples_whole_clips() -> None:
    result = bootstrap(ITEMS[:3], mean_value, n=50, seed=3, ci=0.95, clusters=["clip0"])
    only = ITEMS[0].value
    assert result.intervals["mean"] == Interval(low=only, high=only)


def test_bootstrap_skips_non_finite_values() -> None:
    result = bootstrap(ITEMS, mean_value, n=20, seed=0, ci=0.95)
    assert math.isnan(result.intervals["nan"].low)
    assert math.isnan(result.intervals["nan"].high)


def test_bootstrap_drops_and_counts_replicas_without_blocks() -> None:
    n, seed = 300, 7
    items = (Item("a", 1.0), Item("a", 3.0))
    result = bootstrap(items, mean_value, n=n, seed=seed, ci=0.95, clusters=["a", "b"])
    rng = np.random.default_rng(seed)
    expected = sum(bool(np.all(rng.integers(0, 2, size=2) == 1)) for _ in range(n))
    assert result.dropped == expected > 0
    assert result.replicas == n - expected
    assert result.intervals["mean"] == Interval(low=2.0, high=2.0)


def test_bootstrap_without_replicas_or_clusters_is_empty() -> None:
    assert bootstrap(ITEMS, mean_value, n=0, seed=0, ci=0.95).intervals == {}
    empty = bootstrap((), mean_value, n=10, seed=0, ci=0.95)
    assert (empty.intervals, empty.replicas, empty.dropped) == ({}, 0, 0)
