"""The white-noise null of a per-stencil chi_3 statistic, tested on the stencils away from its
peak: under the null each stencil exceeds a level m with the chi_3 tail probability, so a track
whose other stencils exceed it far more often has heavier tails than its sigma-hat allows.
Adjacent stencils share positions and are correlated, so exceedances cluster and the binomial
p-value is a heuristic gate, not an exact one."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom

from physverify.physics.stencil_tail import chi3_log_sf


@dataclass(frozen=True)
class TailTest:
    """Stencils over the level, stencils tested, and the binomial p-value of that many or more
    under the null."""

    exceed: int
    stencils: int
    p_value: float


def off_peak(per_row: ArrayLike, row: int, guard: int) -> NDArray[np.float64]:
    """The finite values of ``per_row`` more than ``guard`` rows from ``row``: one step of the
    positions moves the third differences of three adjacent stencils, so guard 2 keeps a jump and
    a one-row excursion out of the test of the null they would reject."""
    values = np.asarray(per_row, dtype=np.float64).reshape(-1)
    far = np.abs(np.arange(values.size) - row) > guard
    return values[far & np.isfinite(values)]


def tail_test(values: ArrayLike, level: float) -> TailTest:
    """Binomial test of the count of ``values`` over ``level`` against the chi_3 tail
    P(chi_3 > level); p-value 1 with no values."""
    found = np.asarray(values, dtype=np.float64).reshape(-1)
    n, exceed = int(found.size), int(np.count_nonzero(found > level))
    if not n:
        return TailTest(0, 0, 1.0)
    tail = math.exp(chi3_log_sf(level))
    return TailTest(exceed, n, float(binom.sf(exceed - 1, n, tail)))
