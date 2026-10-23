"""Parameter models every check inherits, and the tie tolerance of per-row maxima."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from physverify.core.models import FrozenModel
from physverify.physics.tracks import MIN_NOISE_STENCILS, OCCLUDED_AT

# Relative slack under which two per-row values count as the same maximum, so ties (a teleport
# inflates two consecutive rows equally) resolve to the earliest row at any row step.
TIE_RTOL = 1e-9


class CommonParams(FrozenModel):
    """Fields every check's ``Params`` inherits."""

    mc_draws: int = Field(default=200, ge=2)
    seed: int = Field(default=0, ge=0)


class OcclusionParams(CommonParams):
    """Occlusion fraction from which an object counts as hidden (checks that read occlusion)."""

    occluded_at: float = Field(default=OCCLUDED_AT, gt=0.0, le=1.0)


class FallParams(CommonParams):
    """Flight search and fall filter shared by R5a and R5b.

    A flight is a fall when the observed travel max_k |p_k - p_first| exceeds both
    ``min_travel_lengths`` body lengths and ``travel_sigmas`` times the noise of a position
    difference. The fitted acceleration is deliberately not used: R5b tests that acceleration,
    and filtering on it would drop weak-gravity flights.

    ``travel_noise`` = "declared": the noise is sqrt(2) |median declared position sigma| (an
    absolute sigma, before 2026-10-20). "track": sqrt(2) |sigma_w| measured on the object's own
    third differences (``physics.tracks.track_noise``), in this order: exact positions (at least
    one known sigma and every known one 0) keep the length floor only; otherwise, with
    ``min_noise_stencils`` usable stencils, the track noise, also when no sigma is known; otherwise
    the declared rule (the length floor alone when no sigma is known).
    """

    rest_speed: float = Field(default=0.1, ge=0.0)
    jump_gate: float = Field(default=3.0, gt=1.0)
    min_travel_lengths: float = Field(default=0.5, ge=0.0)
    travel_sigmas: float = Field(default=3.0, ge=0.0)
    travel_noise: Literal["declared", "track"] = "track"
    min_noise_stencils: int = Field(default=MIN_NOISE_STENCILS, ge=4)
