"""Scale from the known typical size of labelled objects (ADR 0005, 4.6)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.config.models import SIZE_PRIORS_SECTION, SizePrior, SizePriorTable
from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.physics.geometry import characteristic_lengths
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing.priors import SizePriorParams, lookup_label

Params = SizePriorParams


@dataclass(frozen=True)
class ObjectScale:
    """Scale implied by one object and its squared relative sigma rho^2."""

    log_scale: float
    rel_var: float


class SizePriorScale:
    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = "size_prior"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.LIFT,)
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
        estimates = [
            estimate
            for n, label in enumerate(lift.labels.tolist())
            if (prior := self._prior(label)) is not None
            and (estimate := object_scale(lift, n, prior)) is not None
        ]
        if not estimates:
            return ScaleEstimate(None, math.nan, ScaleSource.NONE, lift.source_key, "no_prior")
        scale, sigma = combine(estimates)
        return ScaleEstimate(scale, sigma, ScaleSource.SIZE_PRIOR, lift.source_key)

    def _prior(self, label: str) -> SizePrior | None:
        priors = self.params.priors.root
        direct = lookup_label(label, priors)
        if direct is not None:
            return direct
        alias = lookup_label(label, self.params.label_to_prior)
        return priors.get(alias) if alias is not None else None


def object_scale(lift: ObjectStateTable, n: int, prior: SizePrior) -> ObjectScale | None:
    """s_i = mean_m / median_t L and rho_i^2 = rel_spread^2 + median_t(sigma_L / L)^2."""
    size = lift.size[:, n]
    known = lift.visible[:, n] & np.all(np.isfinite(size), axis=1) & np.all(size > 0, axis=1)
    if not bool(known.any()):
        return None
    lengths = characteristic_lengths(size[known])
    # sigma_L / L = (1/3) sqrt(sum_j (sigma_e_j / e_j)^2) for L = (e_x e_y e_z)^(1/3).
    rel_sigma = np.sqrt(np.sum((lift.size_sigma[:, n][known] / size[known]) ** 2, axis=1)) / 3.0
    length = float(np.median(lengths))
    return ObjectScale(
        log_scale=math.log(prior.mean_m / length),
        rel_var=prior.rel_spread**2 + float(np.median(rel_sigma)) ** 2,
    )


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


PLUGINS = (SizePriorScale,)
