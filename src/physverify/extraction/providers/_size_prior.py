"""Scale from the known typical size of labelled objects.

Focal convention shared by every SCALE provider: meters per unit goes as f^k, k = 1
here (lift extents are z / f), k = 0 for a metric depth and, for free fall, the lateral share of
the fitted acceleration (``scale_free_fall``). The clip's relative focal sigma phi
is one error common to every object, so it enters the relative scale sigma once, as (k phi)^2,
after the objects are combined; the per-object terms leave it out."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.config.models import SIZE_PRIORS_SECTION, SizePrior, SizePriorTable
from physverify.core.models import absent_when_none
from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.backprojection import camera_focal_phi
from physverify.extraction.providers._common import expect_params
from physverify.physics.geometry import characteristic_lengths
from physverify.schema.perception import CameraModel
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing.priors import SizePriorParams, lookup_label, prior_keys

AxisErrors = Literal["correlated"]
"""How the relative errors of the three extents combine into the error of L. ``correlated``: one
error shared by every axis (LIFT scales all extents by one depth / focal factor and e_z is the mean
of the lateral extents). None keeps the independent-axis formula, and so the original cache keys."""


class Params(SizePriorParams):
    axis_errors: AxisErrors | None = absent_when_none()


@dataclass(frozen=True)
class ObjectScale:
    """Scale implied by one object and its squared relative sigma rho^2."""

    log_scale: float
    rel_var: float


class SizePriorScale:
    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = "size_prior"
    version: ClassVar[str] = "2"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.CAMERA, Stage.LIFT)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params
    settings_sections: ClassVar[Mapping[str, tuple[str, type[BaseModel]]]] = {
        "priors": (SIZE_PRIORS_SECTION, SizePriorTable)
    }

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ScaleEstimate:
        lift = ctx.require(Stage.LIFT, ObjectStateTable)
        phi = clip_focal_phi(ctx)
        estimates = [
            estimate
            for n, label in enumerate(lift.labels.tolist())
            if (prior := self._prior(label)) is not None
            and (estimate := object_scale(lift, n, prior, phi, self.params.axis_errors)) is not None
        ]
        if not estimates:
            return ScaleEstimate(None, math.nan, ScaleSource.NONE, lift.source_key, "no_prior")
        scale, sigma = combine(estimates)
        return ScaleEstimate(
            scale, with_focal(scale, sigma, phi), ScaleSource.SIZE_PRIOR, lift.source_key
        )

    def _prior(self, label: str) -> SizePrior | None:
        priors = self.params.priors.root
        key = lookup_label(label, prior_keys(priors, self.params.label_to_prior))
        return None if key is None else priors[key]


def clip_focal_phi(ctx: StageContext) -> float:
    return camera_focal_phi(ctx.require(Stage.CAMERA, CameraModel))


def object_scale(
    lift: ObjectStateTable,
    n: int,
    prior: SizePrior,
    phi: float,
    axis_errors: AxisErrors | None = None,
) -> ObjectScale | None:
    """s_i = mean_m / median_t L and rho_i^2 = rel_spread^2 + median_t(sigma_L / L)^2, with
    sigma_L free of the focal error ``phi`` (``without_focal``)."""
    size = lift.size[:, n]
    known = lift.visible[:, n] & np.all(np.isfinite(size), axis=1) & np.all(size > 0, axis=1)
    if not bool(known.any()):
        return None
    lengths = characteristic_lengths(size[known])
    rel_extent = without_focal(lift.size_sigma[:, n][known] / size[known], phi)
    rel_sigma = length_rel_sigma(rel_extent, axis_errors)
    length = float(np.median(lengths))
    return ObjectScale(
        log_scale=math.log(prior.mean_m / length),
        rel_var=prior.rel_spread**2 + float(np.median(rel_sigma)) ** 2,
    )


def length_rel_sigma(
    rel_extent: NDArray[np.float64], axis_errors: AxisErrors | None
) -> NDArray[np.float64]:
    """sigma_L / L for L = (e_x e_y e_z)^(1/3) from the relative extent sigmas (..., 3):
    d ln L = (1/3) sum_j d ln e_j, so (1/3) sum_j r_j for fully correlated axis errors and
    (1/3) sqrt(sum_j r_j^2) for independent ones."""
    if axis_errors == "correlated":
        mean: NDArray[np.float64] = np.mean(rel_extent, axis=-1)
        return mean
    quadrature: NDArray[np.float64] = np.sqrt(np.sum(rel_extent**2, axis=-1)) / 3.0
    return quadrature


def without_focal(rel_sigma: NDArray[np.float64], phi: float) -> NDArray[np.float64]:
    """Relative extent sigmas without the focal term LIFT adds in quadrature to every extent
    (``closed_form_sigma``); NaN when phi is unknown."""
    removed: NDArray[np.float64] = np.sqrt(np.clip(rel_sigma**2 - phi**2, 0.0, None))
    return removed


def with_focal(scale: float, sigma: float, phi: float) -> float:
    """sigma of s = mean_m / L with the clip's focal error added once: s goes as f."""
    return scale * math.hypot(sigma / scale, phi)


def combine(estimates: list[ObjectScale]) -> tuple[float, float]:
    """Inverse-variance mean in log space; sigma/s = 1/sqrt(sum w) for independent objects.

    Objects with zero variance outweigh the rest: their mean is kept and, when several of them
    disagree, sigma/s is the standard error of their log scales instead of zero."""
    logs = np.array([e.log_scale for e in estimates])
    rel_var = np.array([e.rel_var for e in estimates])
    if not bool(np.all(np.isfinite(rel_var))):
        return math.exp(float(logs.mean())), math.nan
    exact = rel_var == 0
    if bool(exact.any()):
        return exact_combine(logs[exact])
    weights = 1.0 / rel_var
    scale = math.exp(float(np.sum(weights * logs) / np.sum(weights)))
    return scale, scale / math.sqrt(float(np.sum(weights)))


def exact_combine(logs: NDArray[np.float64]) -> tuple[float, float]:
    scale = math.exp(float(logs.mean()))
    if logs.size < 2:
        return scale, 0.0
    return scale, scale * float(logs.std(ddof=1)) / math.sqrt(logs.size)
