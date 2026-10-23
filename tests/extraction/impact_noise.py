"""Monte Carlo rates of the kinematic impact detector on noisy tracks (it3 T5, judgement J4).

Each replicate adds noise to the camera-frame track of one analytic scene, measures the track
noise as ``contacts/kinematic`` does (``physics.tracks.track_noise``, floored by one pixel of the
test camera) and runs the detector. White noise is asserted by test_impacts_noise.py; the
correlated cases (AR(1) noise, two-row depth impulses) are a description (J4), printed by

    PYTHONPATH=src:tests/fixtures:tests/extraction python tests/extraction/impact_noise.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import contact_scenes as cs
import numpy as np
from numpy.typing import NDArray

from physverify.extraction.providers._impact_attribution import object_sigma
from physverify.extraction.providers._impacts import Bases, track_impacts
from physverify.extraction.providers.contacts_kinematic import Params

sc = cs.sc
FloatArray = NDArray[np.float64]
REPLICATES = 400
SEED = 20261021
DEFAULTS = Params()
Noise = Callable[[np.random.Generator, tuple[int, ...]], FloatArray]


def white(sigma: float) -> Noise:
    def draw(rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
        return rng.normal(0.0, sigma, shape)

    return draw


def ar1(sigma: float, rho: float) -> Noise:
    """Stationary AR(1) along the rows with marginal ``sigma`` per axis."""

    def draw(rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
        innovations = rng.normal(0.0, sigma * math.sqrt(1.0 - rho**2), shape)
        out = np.empty(shape)
        out[0] = rng.normal(0.0, sigma, shape[1:])
        for row in range(1, shape[0]):
            out[row] = rho * out[row - 1] + innovations[row]
        return out

    return draw


def depth_impulse(sigma: float, size: float) -> Noise:
    """White ``sigma`` plus ``size`` sigma on the camera depth axis of two consecutive rows at
    a random interior place (a mask flicker or a depth glitch of VDA)."""

    def draw(rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
        out = rng.normal(0.0, sigma, shape)
        row = int(rng.integers(1, shape[0] - 2))
        out[row : row + 2, 2] += size * sigma
        return out

    return draw


def rate(case: sc.Case, noise: Noise, replicates: int = REPLICATES, seed: int = SEED) -> float:
    """Share of replicates with at least one impact on object 0."""
    rng = np.random.default_rng(seed)
    states = case.states
    bases: Bases = {}
    focal = float(cs.scene_camera(case).K[0, 0])
    hits = 0
    for _ in range(replicates):
        all_positions = states.position.copy()
        all_positions[:, 0] += noise(rng, all_positions[:, 0].shape)
        position = all_positions[:, 0]
        noisy = replace(states, position=all_positions)
        sigma = object_sigma(noisy, 0, position[:, 2], focal)
        assert sigma is not None
        found = track_impacts(states.t, position, sigma, DEFAULTS.side_rows, DEFAULTS.alpha, bases)
        hits += bool(found)
    return hits / replicates


def clean_parabola() -> sc.Case:
    return sc.parabola(pitch_deg=25.0)


def description(replicates: int = REPLICATES) -> dict[str, float]:
    """J4 description: false impacts on a clean parabola under correlated noise."""
    sigma = 0.01
    out = {
        f"ar1 rho={rho}": rate(clean_parabola(), ar1(sigma, rho), replicates) for rho in (0.5, 0.8)
    }
    for size in (1, 2, 3, 4, 5):
        out[f"two-row depth impulse {size} sigma"] = rate(
            clean_parabola(), depth_impulse(sigma, size), replicates
        )
    return out


if __name__ == "__main__":
    for label, value in description().items():
        print(f"{label}: {value:.4f}")
