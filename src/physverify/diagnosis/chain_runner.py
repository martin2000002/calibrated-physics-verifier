"""Runs chain variants through the extraction pipeline and loads their cached outputs."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain_variants import Variant, apply_overrides
from physverify.diagnosis.report import Failure
from physverify.extraction._base import StageProvider
from physverify.extraction.pipeline import (
    ClipFailure,
    ExtractionPipeline,
    ExtractionReport,
    PipelineDeps,
)
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.schema.clip import ClipManifest
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.verification.check_set import occlusion_thresholds


@dataclass(frozen=True)
class VariantOutputs:
    variant: Variant
    plan: ResolvedPlan
    outputs: Mapping[Stage, StageOutput]


class ChainRunner:
    """One pipeline run per variant (``warm``), then cache-only reloads, one clip at a time."""

    def __init__(
        self,
        settings: Settings,
        providers: Mapping[str, type[StageProvider]],
        checks: Mapping[str, type[Check]],
        deps: PipelineDeps,
        dataset: DatasetSource,
    ) -> None:
        self._settings = settings
        self._providers = providers
        self._checks = checks
        self._deps = deps
        self._dataset = dataset

    @property
    def dataset(self) -> DatasetSource:
        return self._dataset

    @property
    def error_profile(self) -> ErrorProfile | None:
        return self._deps.error_profile

    def with_error_profile(self, profile: ErrorProfile | None) -> ChainRunner:
        """Same chain reading another error profile (calib runs without one, eval with it)."""
        deps = dataclasses.replace(self._deps, error_profile=profile)
        return ChainRunner(self._settings, self._providers, self._checks, deps, self._dataset)

    def plan(self, variant: Variant) -> ResolvedPlan:
        settings = apply_overrides(self._settings, variant)
        return resolve_plan(
            settings,
            self._providers,
            dataset=self._dataset.name,
            occlusion=occlusion_thresholds(settings, self._checks),
            gt_stages=variant.gt_order,
        )

    def _run(self, variant: Variant, clips: Sequence[ClipManifest]) -> ExtractionReport:
        return ExtractionPipeline(self.plan(variant), self._deps).run(self._dataset, clips)

    def warm(
        self, clips: Sequence[ClipManifest], variants: Sequence[Variant]
    ) -> tuple[ClipFailure, ...]:
        """Fill the cache variant by variant, so each heavy model loads once per variant."""
        failures: list[ClipFailure] = []
        for variant in variants:
            failures.extend(self._run(variant, clips).failures)
        return tuple(failures)

    def outputs(self, clip: ClipManifest, variant: Variant) -> VariantOutputs | None:
        """Every stage output of ``clip`` under ``variant``, or None if any stage failed."""
        plan = self.plan(variant)
        report = ExtractionPipeline(plan, self._deps).run(self._dataset, [clip])
        loaded: dict[Stage, StageOutput] = {}
        for step in plan.stages:
            key = report.keys.get((clip.clip_id, step.stage))
            output = None if key is None else self._load(clip, step.stage, step.name, key)
            if output is None:
                return None
            loaded[step.stage] = output
        return VariantOutputs(variant, plan, loaded)

    def _load(
        self, clip: ClipManifest, stage: Stage, provider: str, key: str
    ) -> StageOutput | None:
        return self._deps.cache.load(self._dataset.name, clip.clip_id, stage, provider, key)


def warm_variants(
    chain: ChainRunner,
    clips: Sequence[ClipManifest],
    variants: Sequence[Variant],
    *,
    skip_failed: bool = True,
) -> tuple[Failure, ...]:
    """Fill the cache variant by variant, naming the variant of each failure.

    With ``skip_failed`` a clip that failed once is not run again (levels 2 and 3 need every
    variant of a clip); level 1 keeps it, since a later variant can still measure other stages.
    """
    failures: list[Failure] = []
    for variant in variants:
        failed = {f.clip_id for f in failures} if skip_failed else set()
        pending = [clip for clip in clips if clip.clip_id not in failed]
        failures += [
            Failure(clip_id=f.clip_id, error=f.error, stage=f.stage, variant=variant.name)
            for f in chain.warm(pending, (variant,))
        ]
    return tuple(failures)
