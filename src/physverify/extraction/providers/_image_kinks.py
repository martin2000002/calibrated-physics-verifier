"""Velocity kinks of the objects' mask-centroid pixel tracks; not a plugin.

Each object's centroid track (``_image_motion.py``) is searched as in ``_impacts.py`` with the
known scale sigma_px of its median r_eq and no Bonferroni bound (shape selection and local maxima
only); significance is left to the attribution, over the m kinks of the object. The kink fit
p(u) = c0 + c1 u + c2 u^2 + c3 (u - s)_+ (u in row steps) gives the change dv = c3 / step and the
velocities just before and after s, (c1 + 2 c2 s) / step and that plus dv, with their sigmas from
the fit covariance sigma_px^2 (D^T D)^-1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.extraction.providers._image_motion import ObjectImage, PixelError
from physverify.extraction.providers._impacts import (
    Bases,
    Impact,
    ImpactRows,
    boundaries,
    fit_kink,
    track_impacts,
)
from physverify.physics.tracks import NoiseScale

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImageKink:
    """A kink of object ``index`` between rows ``row`` and ``row + 1`` at ``time``: its change
    ``delta_v`` (px/s) with the sigma of each component, the velocities ``before`` and ``after``
    it, the sigma of ``after``, its F ratio and the number of kinks of its object."""

    index: int
    row: int
    time: float
    delta_v: FloatArray
    sigma: float
    before: FloatArray
    after: FloatArray
    sigma_after: float
    score: float
    tests: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.index, self.row)


def clip_kinks(
    images: Sequence[ObjectImage | None],
    t: FloatArray,
    rows: ImpactRows,
    error: PixelError,
    alpha: float,
) -> list[ImageKink]:
    """The kinks of every object with a mask; the window bases are shared across the clip."""
    bases: Bases = {}
    found: list[ImageKink] = []
    for index, image in enumerate(images):
        if image is not None and image.valid.any():
            found.extend(object_kinks(index, image, t, rows, error, alpha, bases))
    return found


def object_kinks(
    index: int,
    image: ObjectImage,
    t: FloatArray,
    rows: ImpactRows,
    error: PixelError,
    alpha: float,
    bases: Bases,
) -> list[ImageKink]:
    sigma = float(error.sigma(np.asarray(np.median(image.radius[image.valid]))))
    if not sigma > 0.0:
        return []
    noise = NoiseScale(np.array([sigma, sigma]), known=True)
    impacts = track_impacts(t, image.centroid, noise, rows, alpha, bases, bonferroni=False)
    windows = {row: (lo, hi) for row, lo, hi in boundaries(image.centroid, rows)}
    return [
        kink_of(index, impact, t, image.centroid, windows[impact.row], sigma, len(impacts))
        for impact in impacts
    ]


def kink_of(
    index: int,
    impact: Impact,
    t: FloatArray,
    position: FloatArray,
    window: tuple[int, int],
    sigma: float,
    tests: int,
) -> ImageKink:
    lo, hi = window
    step = float(t[impact.row + 1] - t[impact.row])
    u = (t[lo:hi] - t[impact.row]) / step
    s = (impact.time - t[impact.row]) / step
    design, coef = fit_kink(u, s, position[lo:hi])
    covariance = np.linalg.inv(design.T @ design)
    before, after = np.array([0.0, 1.0, 2.0 * s, 0.0]), np.array([0.0, 1.0, 2.0 * s, 1.0])
    scale = sigma / step
    return ImageKink(
        index=index,
        row=impact.row,
        time=impact.time,
        delta_v=coef[3] / step,
        sigma=math.sqrt(float(covariance[3, 3])) * scale,
        before=before @ coef / step,
        after=after @ coef / step,
        sigma_after=math.sqrt(float(after @ covariance @ after)) * scale,
        score=impact.score,
        tests=tests,
    )
