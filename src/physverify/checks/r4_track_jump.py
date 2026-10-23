"""R4_jump v2: spatio-temporal continuity in units of the object's own position noise."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._params import CommonParams
from physverify.checks._peaks import first_max
from physverify.physics.geometry import characteristic_length
from physverify.physics.measures.object_integrity import jump_ratio
from physverify.physics.measures.track_jump import jump_over_noise, noise_jumps
from physverify.physics.tracks import (
    MIN_NOISE_STENCILS,
    camera_depth,
    impact_stencil_mask,
    median_extents,
    native_span,
    noise_stencils,
    pixel_floor,
    track_noise,
)
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.contacts import ContactTable

FloatArray = NDArray[np.float64]
NoiseSource = Literal["exact", "track", "declared"]


class NoiseJumpParams(CommonParams):
    """``few_stencils``: with fewer than ``min_noise_stencils`` usable stencils and positions that
    are not exact, "declared" takes the median declared position sigma per axis (the order of
    ``checks._falls.find_falls``) and "abstain" gives up (the earlier rule, kept to report the
    survival of the positives with and without that branch)."""

    min_noise_stencils: int = Field(default=MIN_NOISE_STENCILS, ge=4)
    few_stencils: Literal["declared", "abstain"] = "declared"


@dataclass(frozen=True)
class TrackNoise:
    """sigma-hat per axis = max(noise, pixel floor), where the noise came from, and how many axes
    the floor set."""

    sigma: FloatArray
    source: NoiseSource
    floored_axes: int


def floored(noise: FloatArray, floor: float, source: NoiseSource) -> TrackNoise:
    measured = np.asarray(noise, dtype=np.float64)
    return TrackNoise(np.maximum(measured, floor), source, int(np.count_nonzero(measured <= floor)))


def object_noise(
    inputs: CheckInputs, contacts: ContactTable, index: int, floor: float, params: NoiseJumpParams
) -> TrackNoise | None:
    """sigma-hat of object ``index``: the floor alone for exact positions (a known sigma, all 0);
    else its track noise; else, under "declared", its median declared sigma; else None."""
    declared = inputs.states.position_sigma[:, index]
    known = declared[np.all(np.isfinite(declared), axis=-1)]
    if known.shape[0] and bool(np.all(known == 0.0)):
        return TrackNoise(np.full(3, floor), "exact", 3)
    track = track_noise(inputs.states, contacts, index, params.min_noise_stencils)
    if track is not None:
        return floored(track, floor, "track")
    if params.few_stencils == "declared" and known.shape[0]:
        return floored(np.median(known, axis=0), floor, "declared")
    return None


class NoiseContinuityCheck:
    """R4_jump [jump_over_noise]: max_k ||d3_k / (sqrt(20) sigma-hat)|| over the rows whose
    stencil spans no impact (impacts converted to native frames), with sigma-hat per axis the
    larger of the object's own noise (``track_noise``: robust scale of its third differences /
    sqrt(20)) and one pixel at its median depth, z / (f sqrt(12)) with f the focal length of the
    camera LIFT used.

    A test statistic, not a physical size: the jump it stands for is in ``details``
    (``jump_over_size_estimated``, an estimate at the peak row). ``sigma`` is 1 in its own unit
    (the per-row sigma at the peak, as in v1) and 0 for exact positions; the error of sigma-hat
    itself (about 30 % relative with 8 to 20 stencils) is not propagated, a known limitation. The
    perturbations it answers to are injected in ``jump_over_size`` (``perturbation_unit``); the
    sweep compares that estimate with the injected jump (``perturbation_estimate``)."""

    check_id: ClassVar[str] = "R4_jump"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts", "camera"})
    time_dependent: ClassVar[bool] = True
    magnitude_unit: ClassVar[str] = "jump_over_noise"
    perturbation_unit: ClassVar[str] = "jump_over_size"
    perturbation_estimate: ClassVar[str] = "jump_over_size_estimated"
    Params: ClassVar[type[BaseModel]] = NoiseJumpParams

    def __init__(self, params: BaseModel) -> None:
        self.params = NoiseJumpParams.model_validate(params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        if inputs.contacts is None or inputs.camera is None:
            raise ValueError(f"{self.check_id} requires contacts and camera")
        contacts, focal = inputs.contacts, float(inputs.camera.K[0, 0])
        return [self._object(inputs, contacts, focal, i) for i in range(inputs.states.n_objects)]

    def _object(
        self, inputs: CheckInputs, contacts: ContactTable, focal: float, index: int
    ) -> CheckResult:
        states = inputs.states
        object_id = str(states.object_ids[index])
        exclude = impact_stencil_mask(object_id, contacts, states.frame_index)
        usable = int(noise_stencils(states, contacts, index)[1].sum())
        details: dict[str, object] = {
            "excluded_rows": int(np.count_nonzero(exclude)),
            "usable_stencils": usable,
        }
        floor = pixel_floor(camera_depth(states, index), focal)
        if usable and not math.isfinite(floor):
            reason = AbstainReason.UNKNOWN_UNCERTAINTY
            return abstain(type(self), inputs, (object_id,), reason, None, details)
        noise = object_noise(inputs, contacts, index, floor, self.params) if usable else None
        if noise is None:
            reason = AbstainReason.TOO_FEW_FRAMES
            return abstain(type(self), inputs, (object_id,), reason, None, details)
        per_row = noise_jumps(states.position[:, index], noise.sigma, exclude)
        k = first_max(per_row)
        magnitude = jump_over_noise(states.position[:, index], noise.sigma, exclude)
        sigma = 0.0 if noise.source == "exact" else 1.0
        span = native_span(states.frame_index, k - 1, k + 2)
        details |= {"row": k, **self._physical(inputs, index, k, noise)}
        return unscored(type(self), inputs, (object_id,), magnitude, sigma, span, details)

    @staticmethod
    def _physical(
        inputs: CheckInputs, index: int, row: int, noise: TrackNoise
    ) -> dict[str, object]:
        """What the statistic stands for, labelled as estimates: the jump at the peak row in body
        lengths and sigma-hat per axis in body lengths (None without a finite size)."""
        extents = median_extents(inputs.states, index)
        length = characteristic_length(extents) if bool(np.all(np.isfinite(extents))) else math.nan
        window = inputs.states.position[row - 1 : row + 3, index]
        jump = jump_ratio(window, extents) if math.isfinite(length) else math.nan
        return {
            "noise_source": noise.source,
            "floored_axes": noise.floored_axes,
            "jump_over_size_estimated": jump if math.isfinite(jump) else None,
            "sigma_over_length": (noise.sigma / length).tolist() if math.isfinite(length) else None,
        }


PLUGINS = (NoiseContinuityCheck,)
