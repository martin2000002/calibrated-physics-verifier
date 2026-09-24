"""Stage error entries as numbers and draws, and Monte Carlo propagation (ADR 0002, 0005 4.4)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.schema.error_profile import ErrorEntry

FloatArray = NDArray[np.float64]


def entry_rms(entry: ErrorEntry | None) -> float:
    """sqrt(bias^2 + sigma^2): NaN when the error is unknown, 0 for an exact stage."""
    if entry is None:
        return math.nan
    return math.hypot(entry.bias, entry.sigma)


def entry_draws(
    entry: ErrorEntry | None, rng: np.random.Generator, shape: tuple[int, ...]
) -> FloatArray | None:
    """Error samples: bootstrap of the residuals when present, else Normal(bias, sigma).

    None when the error is unknown (no entry or a NaN sigma without residuals)."""
    if entry is None:
        return None
    if entry.residuals:
        return rng.choice(np.asarray(entry.residuals, dtype=np.float64), size=shape)
    if not (math.isfinite(entry.bias) and math.isfinite(entry.sigma)):
        return None
    if entry.sigma == 0.0:
        return np.full(shape, entry.bias)
    return rng.normal(entry.bias, entry.sigma, size=shape)


@dataclass(frozen=True)
class MonteCarloResult:
    mean: FloatArray
    std: FloatArray
    n_draws: int


def monte_carlo(
    model: Callable[[Mapping[str, FloatArray]], FloatArray],
    draws: Mapping[str, FloatArray | None],
) -> MonteCarloResult:
    """Mean and sample std of ``model`` over draws stacked on axis 0.

    An unknown draw (None) is replaced by zeros of shape (n_draws,) and makes every std NaN."""
    known = [value for value in draws.values() if value is not None]
    if not known:
        raise ValueError("monte_carlo needs at least one known draw")
    n_draws = int(known[0].shape[0])
    if any(value.shape[0] != n_draws for value in known):
        raise ValueError("every draw must have the same length on axis 0")
    filled = {k: v if v is not None else np.zeros(n_draws) for k, v in draws.items()}
    values = np.asarray(model(filled), dtype=np.float64)
    std = values.std(axis=0, ddof=1) if n_draws > 1 else np.full(values.shape[1:], np.nan)
    if len(known) != len(draws):
        std = np.full(values.shape[1:], np.nan)
    return MonteCarloResult(mean=values.mean(axis=0), std=np.asarray(std), n_draws=n_draws)
