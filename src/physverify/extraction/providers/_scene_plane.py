"""The clip's scene plane, normal up, judged without the body it judges; not a plugin.

Each object m with evidence gives a height H_m: the median, over its rest rows on the
environment and its upward impacts, of its lowest point along up. The plane for object n sits at
the lowest H_m of the other objects (the lower envelope: stacked or flying objects stay above
it), so no body is ever judged against its own bottom. One object alone has no plane.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.extraction.providers._proximity import PlacedBody
from physverify.physics.geometry import Plane, Pose, Solid
from physverify.physics.signed_distance import signed_distance
from physverify.physics.uncertainty import entry_rms
from physverify.schema.error_profile import ErrorEntry

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HeightError:
    """Relative centroid and size errors of the lift: sigma_H = L hypot(e_c, e_s / 2)."""

    centroid: float
    size: float

    @classmethod
    def of(cls, centroid: ErrorEntry | None, size: ErrorEntry | None) -> HeightError:
        return cls(entry_rms(centroid), entry_rms(size))

    def sigma(self, length: float) -> float:
        return length * math.hypot(self.centroid, self.size / 2.0)


@dataclass(frozen=True)
class SupportPlane:
    """Plane under one object, its height sigma and the object that set it."""

    plane: Plane
    sigma: float
    source: int

    @property
    def height(self) -> float:
        return self.plane.offset


def up_of(gravity: FloatArray | None) -> FloatArray | None:
    """-g / ||g||; None when gravity is unknown."""
    if gravity is None:
        return None
    g = np.asarray(gravity, dtype=np.float64)
    norm = float(np.linalg.norm(g))
    return -g / norm if bool(np.all(np.isfinite(g))) and norm > 0.0 else None


def lowest(body: Solid, up: FloatArray) -> float:
    """Height along up of the body's lowest point."""
    return signed_distance(body, Plane.ground(0.0, up)).distance


def moved(body: Solid, position: FloatArray) -> Solid:
    return Solid(body.primitive, Pose(np.asarray(position, dtype=np.float64), body.pose.rotation))


def solid(placed: PlacedBody | None) -> Solid | None:
    return placed.body if placed is not None and isinstance(placed.body, Solid) else None


def height(placed: PlacedBody | None, up: FloatArray, position: FloatArray | None = None) -> float:
    """Lowest height along up of a placed body (centre moved to ``position``); NaN unplaced."""
    body = solid(placed)
    if body is None:
        return math.nan
    return lowest(body if position is None else moved(body, position), up)


def support_gap(placed: PlacedBody | None, support: SupportPlane) -> float:
    """Signed distance from the body to the plane (negative = penetration); NaN unplaced."""
    body = solid(placed)
    return math.nan if body is None else signed_distance(body, support.plane).distance


def median_positions(position: FloatArray, visible: NDArray[np.bool_]) -> FloatArray:
    """(N, 3) median position of each object over its visible rows (NaN when never seen)."""
    shown = np.where(visible[..., None], position, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median: FloatArray = np.nanmedian(shown, axis=0)
    return median


@dataclass(frozen=True)
class ScenePlane:
    """Per-object evidence heights H_m and what the plane sigma needs."""

    up: FloatArray
    heights: Mapping[int, float]
    centres: FloatArray
    height_sigmas: FloatArray
    gravity_sigma: float

    @classmethod
    def of(
        cls,
        up: FloatArray,
        evidence: Mapping[int, Sequence[float]],
        centres: FloatArray,
        height_sigmas: FloatArray,
        gravity_sigma: float,
    ) -> ScenePlane:
        heights = {m: float(np.median(h)) for m, h in evidence.items() if len(h)}
        return cls(up, heights, centres, height_sigmas, gravity_sigma)

    def for_object(self, n: int) -> SupportPlane | None:
        """P_n at c_n = min over m != n of H_m; sigma_c = hypot(sigma_H(m*), r_n sigma_g) with
        r_n the distance across up between the two objects (small-angle tilt error)."""
        others = {m: h for m, h in self.heights.items() if m != n and math.isfinite(h)}
        if not others:
            return None
        source = min(others, key=lambda m: (others[m], m))
        offset = self.centres[n] - self.centres[source]
        across = float(np.linalg.norm(offset - (offset @ self.up) * self.up))
        sigma = math.hypot(float(self.height_sigmas[source]), across * self.gravity_sigma)
        return SupportPlane(Plane.ground(others[source], self.up), sigma, source)
