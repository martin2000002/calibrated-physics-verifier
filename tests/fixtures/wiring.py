"""Test conveniences over the injected factories: each reads the plugin registries at call time
(``physverify.app.plugins.build``), so a test that patches ``PLUGIN_KINDS`` is honoured."""

from __future__ import annotations

from collections.abc import Mapping

from physverify.app import plugins
from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.extraction._base import StageProvider
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.plan import resolve_plan as resolve_injected_plan
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scoring import build_scorer as build_injected_scorer
from physverify.schema.stages import Stage
from physverify.verification import check_set


def checks() -> Mapping[str, type[Check]]:
    registry: Mapping[str, type[Check]] = plugins.build("checks")
    return registry


def build_scorer(name: str, settings: Settings) -> AggregateScorer:
    return build_injected_scorer(name, settings, scorers=plugins.build("scorers"), checks=checks())


def active_checks_hash(settings: Settings) -> str:
    return check_set.active_checks_hash(settings, checks())


def resolve_plan(
    settings: Settings,
    providers: Mapping[str, type[StageProvider]],
    *,
    dataset: str,
    gt_stages: tuple[Stage, ...] = (),
    fps: float | None = None,
    max_side: int | None = None,
) -> ResolvedPlan:
    return resolve_injected_plan(
        settings,
        providers,
        dataset=dataset,
        occlusion=check_set.occlusion_thresholds(settings, checks()),
        gt_stages=gt_stages,
        fps=fps,
        max_side=max_side,
    )
