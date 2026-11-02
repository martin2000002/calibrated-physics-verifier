"""Timing jitter injected into a perceived state table: every row read at instant r + delta_r.

A diagnosis input, like the chain variants: it alters what a check reads to measure how the check
answers to frames that are not evenly spaced in time (repeated or irregular frames under uniform
timestamps), and is not a perturbation of the calibration set. Pure, no I/O.

Known artifact: a fractional delta interpolates linearly between two perceived rows, which
averages their position noise (variance (1 - a)^2 + a^2 < 1 of one row's), so such rows come out
somewhat smoother than a real frame taken at that instant.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JitterSpec:
    """delta_r = g_r + e_r in rows: g_r ~ N(0, ``pervasive_sigma``^2) and, with probability
    ``event_rate``, an event e_r of -1 (a repeated frame, with probability ``duplicate_share``)
    or +-U(``half_range``). The defaults are estimates from the timing diagnosis of real
    generated video, not from the calibration set: events at 0.15 of the rows, 0.59 of them
    repeats (372 of the 635 stalls and half steps over tau), and a pervasive jitter whose step
    ratio s/m - 1 ~ delta_k - delta_{k-1} has a robust scale of 0.12, so 0.12 / sqrt(2) per row.
    ``event_rate`` 0 and ``pervasive_sigma`` 0 leave a table unchanged; ``pervasive_sigma`` 0
    alone keeps the events only."""

    seed: int
    event_rate: float = 0.15
    duplicate_share: float = 0.59
    half_range: tuple[float, float] = (0.3, 0.7)
    pervasive_sigma: float = 0.085


def clip_rng(spec: JitterSpec, clip_id: str) -> np.random.Generator:
    """The generator of one clip: the same offsets for a clip under every arm that reads it."""
    return np.random.default_rng([spec.seed, zlib.crc32(clip_id.encode("utf-8"))])


def draw_offsets(n_rows: int, spec: JitterSpec, rng: np.random.Generator) -> FloatArray:
    """delta per row; 0 on the first and last rows, and r + delta kept within the rows."""
    pervasive = rng.normal(0.0, spec.pervasive_sigma, n_rows)
    event = rng.random(n_rows) < spec.event_rate
    repeated = rng.random(n_rows) < spec.duplicate_share
    half = rng.uniform(*spec.half_range, n_rows) * rng.choice((-1.0, 1.0), n_rows)
    offsets = pervasive + np.where(event, np.where(repeated, -1.0, half), 0.0)
    if n_rows:
        offsets[0] = offsets[-1] = 0.0
    rows = np.arange(n_rows, dtype=np.float64)
    return np.asarray(np.clip(rows + offsets, 0.0, max(n_rows - 1, 0)) - rows)


def apply_offsets(states: ObjectStateTable, offsets: FloatArray) -> ObjectStateTable:
    """A new table whose row r of every object holds its position and size at r + delta_r: a copy
    of row r + delta_r when that is a row (delta = -1 copies row r - 1), else the linear
    interpolation of the two rows around it. Per object and field, a row that is not known or
    whose neighbours are not all known stays as it was; the other fields and ``states`` are not
    changed."""
    delta = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if delta.shape[0] != states.n_frames:
        raise ValueError(f"{delta.shape[0]} offsets for {states.n_frames} rows")
    target = np.arange(delta.shape[0], dtype=np.float64) + delta
    low = np.floor(target).astype(np.int64)
    high = np.minimum(low + 1, max(states.n_frames - 1, 0))
    weight = target - low
    return replace(
        states,
        position=_shifted(states.position, low, high, weight),
        size=_shifted(states.size, low, high, weight),
    )


def _shifted(
    values: FloatArray, low: NDArray[np.int64], high: NDArray[np.int64], weight: FloatArray
) -> FloatArray:
    a = weight.reshape(-1, *(1,) * (values.ndim - 1))
    exact = (a == 0.0) | (low == high).reshape(a.shape)
    near = np.where(exact, values[low], (1.0 - a) * values[low] + a * values[high])
    known = np.isfinite(values) & np.isfinite(values[low]) & (exact | np.isfinite(values[high]))
    whole = np.all(known, axis=-1, keepdims=True)
    return np.asarray(np.where(whole, near, values), dtype=np.float64)
