"""Temporal stabilisation of LIFT's per-object summaries (LIFT version 3; ADR 0005 4.2, it2).

Along each visible run of an object, a running median replaces the surface depth z and the image
extent q = (e_x, e_y) / z (pixels / f, independent of depth). The centre is rescaled along its
viewing ray by z_new / z, so its pixel stays put; e_x, e_y = q_new z_new and e_z follows from
them by the LIFT rule. A median of w = 2h + 1 rows removes impulses of up to h rows and keeps
steps, monotone runs and plateaus of h + 1 or more rows; a real change shorter than that is lost.

Ends of a run: ``"keep"`` (LIFT's rule) leaves the first and last h rows as measured.
``"shifted"`` moves the window inside the run (row 0 uses rows 0..w - 1): an impulse on those rows
is removed too, but a monotone run also loses its slope there, since such an impulse is itself
monotone. It fails the survival test J5 (tests/verification/test_it2_temporal_survival.py) and
exists only for the descriptive run of the J6 gate (plan-it2, judgement 2 a).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from physverify.extraction.providers._backprojection import ObjectSummary, full_extents
from physverify.physics.kinematics import visible_runs

FloatArray = NDArray[np.float64]
EdgeMode = Literal["shifted", "keep"]


def running_median(x: FloatArray, window: int, edges: EdgeMode) -> FloatArray:
    """Median over ``window`` rows around each row; the first and last ``window // 2`` rows use a
    window shifted inside the series or keep their value (``edges``). Non-finite rows are skipped
    as samples and left as they are; a series shorter than the window comes back unchanged."""
    values = np.asarray(x, dtype=np.float64)
    out = values.copy()
    n = values.size
    if n < window:
        return out
    half = window // 2
    for row in range(n):
        inside = half <= row < n - half
        if not math.isfinite(values[row]) or (edges == "keep" and not inside):
            continue
        first = min(max(row - half, 0), n - window)
        span = values[first : first + window]
        out[row] = float(np.median(span[np.isfinite(span)]))
    return out


def stabilize_run(
    centres: FloatArray, extents: FloatArray, window: int, edges: EdgeMode
) -> tuple[FloatArray, FloatArray]:
    """(n, 3) centres and extents of one visible run with the median depth and image extent."""
    z = centres[:, 2]
    image_extent = extents[:, :2] / z[:, None]
    z_new = running_median(z, window, edges)
    q_new = np.column_stack([running_median(image_extent[:, k], window, edges) for k in range(2)])
    ratio = z_new / z
    usable = np.isfinite(ratio) & np.isfinite(q_new).all(axis=1)
    new_centres = np.where(usable[:, None], centres * ratio[:, None], centres)
    new_extents = np.where(usable[:, None], full_extents(q_new * z_new[:, None]), extents)
    return new_centres, new_extents


def stabilize_summaries(
    column: Sequence[ObjectSummary | None], window: int, edges: EdgeMode
) -> list[ObjectSummary | None]:
    """One object's summaries over time, each visible run stabilised (``None`` = not visible)."""
    out = list(column)
    present = np.array([summary is not None for summary in column], dtype=bool)
    for run in visible_runs(present):
        if run.stop - run.start < window:
            continue
        summaries = [s for s in out[run] if s is not None]
        centres = np.stack([s.centre for s in summaries])
        extents = np.stack([s.extents for s in summaries])
        new_centres, new_extents = stabilize_run(centres, extents, window, edges)
        out[run] = [
            replace(s, centre=c, extents=e)
            for s, c, e in zip(summaries, new_centres, new_extents, strict=True)
        ]
    return out
