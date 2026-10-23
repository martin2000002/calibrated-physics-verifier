"""Per-check randomness and Monte Carlo sigma for the checks."""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Mapping

import numpy as np

from physverify.physics.uncertainty import FloatArray, monte_carlo


def check_rng(seed: int, clip_id: str, check_id: str) -> np.random.Generator:
    """Independent, reproducible stream per (seed, clip, check)."""
    return np.random.default_rng(
        [seed, zlib.crc32(clip_id.encode()), zlib.crc32(check_id.encode())]
    )


def mc_sigma(
    model: Callable[[Mapping[str, FloatArray]], FloatArray],
    sigmas: Mapping[str, FloatArray],
    draws: int,
    rng: np.random.Generator,
) -> float:
    """Std of ``model`` under N(0, sigma) noise on each input.

    ``model`` receives one (draws, *shape) noise array per key and returns (draws,) magnitudes.
    Exact inputs give 0; any NaN sigma makes the result NaN.
    """
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in sigmas.items()}
    if all(bool(np.all(value == 0.0)) for value in arrays.values()):
        return 0.0
    noise = {key: _noise(value, draws, rng) for key, value in arrays.items()}
    if all(value is None for value in noise.values()):
        return math.nan
    std = np.asarray(monte_carlo(model, noise).std, dtype=np.float64)
    if std.size != 1:
        raise ValueError(f"model must return one magnitude per draw, got std shape {std.shape}")
    return float(std.reshape(()))


def _noise(sigma: FloatArray, draws: int, rng: np.random.Generator) -> FloatArray | None:
    if not bool(np.all(np.isfinite(sigma))):
        return None
    return rng.normal(0.0, 1.0, size=(draws, *sigma.shape)) * sigma
