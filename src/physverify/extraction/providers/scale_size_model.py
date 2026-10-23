"""Scale from object sizes learned on Objectron, falling back to the frozen priors.

Objects that resolve to the same class share the prior error with correlation ``same_class_rho``
(compound symmetry): within a class group only the measurement term averages down, and the
shared term rho * p^2 is added back to the group before the groups are combined."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar

from pydantic import BaseModel, Field

from physverify.config.models import SIZE_PRIORS_SECTION, SizePrior, SizePriorTable
from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.extraction.providers.scale_size_prior import (
    ObjectScale,
    combine,
    object_scale,
)
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing.model import SECTION as SIZE_MODEL_SECTION
from physverify.sizing.model import SizeModelTable, model_hash, prior_aliases, resolve
from physverify.sizing.priors import lookup_label
from physverify.sizing.settings import SECTION as SIZING_SECTION
from physverify.sizing.settings import SizeLookup

DETAIL_PREFIX = "size_model"
SIGMA_SCOPE = "sigma=size_term_only"
"""The sigma carries the object-size error alone: the camera intrinsics (an assumed FOV on real
video) are not propagated, so on real video it undercovers."""
FALLBACK_GROUP = "prior:"


class Params(ProviderParams):
    label_to_prior: dict[str, str] = Field(default_factory=prior_aliases)
    same_class_rho: float = Field(default=1.0, ge=0.0, le=1.0)
    priors: SizePriorTable = SizePriorTable({})
    model: SizeModelTable = SizeModelTable()
    lookup: SizeLookup = SizeLookup()


class SizeModelScale:
    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = "size_model"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.LIFT,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params
    settings_sections: ClassVar[Mapping[str, tuple[str, type[BaseModel]]]] = {
        "priors": (SIZE_PRIORS_SECTION, SizePriorTable),
        "model": (SIZE_MODEL_SECTION, SizeModelTable),
        "lookup": (SIZING_SECTION, SizeLookup),
    }

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ScaleEstimate:
        lift = ctx.require(Stage.LIFT, ObjectStateTable)
        groups = self._groups(lift)
        if not groups:
            return ScaleEstimate(None, math.nan, ScaleSource.NONE, lift.source_key, "no_prior")
        if self.params.same_class_rho == 0.0:
            scale, sigma = combine([e for _, members in groups.values() for e in members])
        else:
            scale, sigma = combine([self._group(p, m) for p, m in groups.values()])
        detail = (
            f"{DETAIL_PREFIX}:{model_hash(self.params.model, self.params.lookup)}:{SIGMA_SCOPE}"
        )
        return ScaleEstimate(scale, sigma, ScaleSource.SIZE_PRIOR, lift.source_key, detail)

    def _groups(self, lift: ObjectStateTable) -> dict[str, tuple[SizePrior, list[ObjectScale]]]:
        """Per resolved class: its prior and each object's scale with only the unshared part
        (1 - rho) p^2 of the prior variance."""
        keep = math.sqrt(1.0 - self.params.same_class_rho)
        groups: dict[str, tuple[SizePrior, list[ObjectScale]]] = {}
        for n, label in enumerate(lift.labels.tolist()):
            resolved = self._resolve(label)
            if resolved is None:
                continue
            key, prior = resolved
            unshared = SizePrior(mean_m=prior.mean_m, rel_spread=keep * prior.rel_spread)
            estimate = object_scale(lift, n, unshared)
            if estimate is not None:
                groups.setdefault(key, (prior, []))[1].append(estimate)
        return groups

    def _group(self, prior: SizePrior, members: list[ObjectScale]) -> ObjectScale:
        scale, sigma = combine(members)
        shared = self.params.same_class_rho * prior.rel_spread**2
        return ObjectScale(math.log(scale), shared + (sigma / scale) ** 2)

    def _resolve(self, label: str) -> tuple[str, SizePrior] | None:
        """The learned or fixed size first, else the frozen prior (directly or by alias)."""
        learned = resolve(label, self.params.model, self.params.lookup)
        if learned is not None:
            return learned
        priors = self.params.priors.root
        key = lookup_label(label, {name: name for name in priors})
        if key is None:
            alias = lookup_label(label, self.params.label_to_prior)
            key = alias if alias in priors else None
        return None if key is None else (FALLBACK_GROUP + key, priors[key])


PLUGINS = (SizeModelScale,)
