"""Rank correlation between injected levels and measured magnitudes."""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.stats import kendalltau

MIN_RANKED_LEVELS = 2


def kendall(pairs: Sequence[tuple[float, float]]) -> float:
    """Kendall tau between injected level and measured magnitude; NaN below two levels."""
    if len(pairs) < MIN_RANKED_LEVELS:
        return math.nan
    x, y = zip(*pairs, strict=True)
    return float(kendalltau(x, y).statistic)
