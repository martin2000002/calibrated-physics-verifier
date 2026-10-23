"""Margins from the published AutoEval scores (ADR 0013 D5), derived before the freeze.

margin = trunc2((max - min) / 2) of the per-generator mean of one AutoEval column, over the
generators with at least ``min_n`` videos: giving up less than half of the spread between the
worst and the best modern generator, as AutoEval itself sees them. The table is untrusted data:
read with the csv module, every non-numeric cell is skipped.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from physverify.core.errors import UsageError
from physverify.core.models import FrozenModel
from physverify.datasets._videophy_tables import to_float

DECIMALS = 2


@dataclass(frozen=True)
class GeneratorMean:
    mean: float
    sd: float
    n: int


class MarginDerivation(FrozenModel):
    column: str
    min_n: int
    n_generators: int
    lo_generator: str
    lo: float
    hi_generator: str
    hi: float
    spread: float
    se: float
    value: float


def generator_means(
    rows: Iterable[Mapping[str, str]], column: str, generator_column: str
) -> dict[str, GeneratorMean]:
    values: dict[str, list[float]] = {}
    for row in rows:
        value = to_float(row.get(column))
        if math.isfinite(value):
            values.setdefault(row.get(generator_column, "") or "", []).append(value)
    return {
        name: GeneratorMean(
            float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)
        )
        for name, v in sorted(values.items())
    }


def truncate(value: float, decimals: int = DECIMALS) -> float:
    scale = 10.0**decimals
    return math.floor(round(value * scale, 9)) / scale


def margin(
    rows: Iterable[Mapping[str, str]], column: str, generator_column: str, min_n: int
) -> MarginDerivation:
    means = {
        name: m
        for name, m in generator_means(rows, column, generator_column).items()
        if m.n >= min_n
    }
    if len(means) < 2:
        raise UsageError(f"{column}: fewer than two generators with n >= {min_n}")
    lo_name = min(means, key=lambda name: means[name].mean)
    hi_name = max(means, key=lambda name: means[name].mean)
    lo, hi = means[lo_name], means[hi_name]
    spread = hi.mean - lo.mean
    return MarginDerivation(
        column=column,
        min_n=min_n,
        n_generators=len(means),
        lo_generator=lo_name,
        lo=lo.mean,
        hi_generator=hi_name,
        hi=hi.mean,
        spread=spread,
        se=math.sqrt(lo.sd**2 / lo.n + hi.sd**2 / hi.n),
        value=truncate(spread / 2.0),
    )
