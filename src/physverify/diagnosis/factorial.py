"""Arithmetic of the 2^3 ground-truth substitution factorial (ADR 0007 §5.4). Pure, no I/O.

A cell says, for each factor in ``FACTORS`` (TRACK, DEPTH, SCALE), whether it runs perceived
(True, coded +1) or ground truth (False, coded -1). ``means`` maps every cell to its mean response.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from physverify.diagnosis.plans import CELLS, FACTORS, Cell
from physverify.diagnosis.report import Interval
from physverify.schema.stages import Stage

Means = Mapping[Cell, float]

MEAN = "mean"
FLOOR = "floor"
INTERACTION = ":"
ALL_TRUTH: Cell = CELLS[0]
ALL_PERCEIVED: Cell = CELLS[-1]


def _require_cells(means: Means) -> None:
    missing = [cell for cell in CELLS if cell not in means]
    if missing:
        raise ValueError(f"the factorial needs all {len(CELLS)} cells; missing {missing}")


def effect_terms() -> tuple[tuple[int, ...], ...]:
    """Factor index sets of the main effects and interactions, by order then factor order."""
    indices = range(len(FACTORS))
    return tuple(
        term
        for order in range(1, len(FACTORS) + 1)
        for term in itertools.combinations(indices, order)
    )


def effect_name(term: Sequence[int]) -> str:
    return INTERACTION.join(FACTORS[i].value for i in term)


def _coded(cell: Cell, term: Sequence[int]) -> float:
    return math.prod(1.0 if cell[i] else -1.0 for i in term)


def yates_effects(means: Means) -> dict[str, float]:
    """``mean`` = (1/8) sum y_c and, per term S, E_S = (1/4) sum_c prod_{f in S} x_f(c) y_c."""
    _require_cells(means)
    half = len(CELLS) / 2
    effects = {MEAN: math.fsum(means[cell] for cell in CELLS) / len(CELLS)}
    for term in effect_terms():
        effects[effect_name(term)] = math.fsum(_coded(c, term) * means[c] for c in CELLS) / half
    return effects


def _perceive(cell: Cell, index: int) -> Cell:
    track, depth, scale = cell
    return (track or index == 0, depth or index == 1, scale or index == 2)


def _shapley_weight(coalition: int, players: int) -> float:
    """|A|! (n - |A| - 1)! / n!: 1/3, 1/6, 1/3 for n = 3."""
    numerator = math.factorial(coalition) * math.factorial(players - coalition - 1)
    return numerator / math.factorial(players)


def shapley(means: Means) -> dict[Stage, float]:
    """phi_f = sum_{A not containing f} w(|A|) [v(A + f) - v(A)], v(A) = mean of the cell whose
    perceived set is A. The values add up to v(all perceived) - v(all ground truth)."""
    _require_cells(means)
    n = len(FACTORS)
    return {
        stage: math.fsum(
            _shapley_weight(sum(cell), n) * (means[_perceive(cell, i)] - means[cell])
            for cell in CELLS
            if not cell[i]
        )
        for i, stage in enumerate(FACTORS)
    }


@dataclass(frozen=True)
class BudgetRows:
    """``floor`` + sum(``stages``) + ``fixed`` = ``total`` (the V0 mean error)."""

    floor: float
    stages: Mapping[Stage, float]
    fixed: float
    total: float

    def rows(self, fixed_label: str) -> tuple[tuple[str, float], ...]:
        stage_rows = tuple((stage.value, value) for stage, value in self.stages.items())
        return ((FLOOR, self.floor), *stage_rows, (fixed_label, self.fixed))


def budget(means: Means, v0: float) -> BudgetRows:
    """Floor = all-truth cell (light stages only), Shapley per factor, and the fixed stages
    (DETECT, CAMERA) as V0 minus the all-perceived cell."""
    return BudgetRows(
        floor=means[ALL_TRUTH],
        stages=shapley(means),
        fixed=v0 - means[ALL_PERCEIVED],
        total=v0,
    )


class Clustered(Protocol):
    @property
    def clip_id(self) -> str: ...


@dataclass(frozen=True)
class BootstrapResult:
    """``non_finite`` counts, per key, the kept replicas whose value was NaN or infinite (for
    example kappa on a one-class resample); they are left out of that key's interval."""

    intervals: Mapping[str, Interval]
    replicas: int
    dropped: int
    non_finite: Mapping[str, int] = field(default_factory=dict)


def _interval(values: Sequence[float], ci: float) -> Interval:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return Interval(low=math.nan, high=math.nan)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(finite, (alpha, 1.0 - alpha))
    return Interval(low=float(low), high=float(high))


def bootstrap[B: Clustered](
    blocks: Sequence[B],
    statistic: Callable[[Sequence[B]], Mapping[str, float]],
    n: int,
    seed: int,
    ci: float,
    clusters: Sequence[str] | None = None,
) -> BootstrapResult:
    """Percentile intervals from ``n`` resamples of whole clips (blocks of a clip are correlated).

    ``clusters`` are the clip ids to resample (default: the clips of ``blocks``); a replica that
    draws only clips without blocks is dropped and counted. Non-finite values of a key are left
    out of that key's interval and counted in ``non_finite``.
    """
    ids = tuple(sorted(set(clusters) if clusters is not None else {b.clip_id for b in blocks}))
    if n <= 0 or not ids:
        return BootstrapResult(intervals={}, replicas=0, dropped=0)
    by_clip = {clip_id: [b for b in blocks if b.clip_id == clip_id] for clip_id in ids}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    dropped = 0
    for _ in range(n):
        sample = [
            b for index in rng.integers(0, len(ids), size=len(ids)) for b in by_clip[ids[index]]
        ]
        if not sample:
            dropped += 1
            continue
        for key, value in statistic(sample).items():
            draws.setdefault(key, []).append(value)
    intervals = {key: _interval(values, ci) for key, values in sorted(draws.items())}
    non_finite = {
        key: int(np.sum(~np.isfinite(np.asarray(values, dtype=np.float64))))
        for key, values in sorted(draws.items())
    }
    return BootstrapResult(intervals, n - dropped, dropped, non_finite)
