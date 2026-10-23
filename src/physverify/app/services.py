"""Services wired from an application context: what several commands build the same way."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from physverify.config.plugin_context import PluginContext
from physverify.core.errors import ConfigError
from physverify.core.registry import satisfies
from physverify.datasets._base import DatasetSource, dataset_from
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.verification_adapter import VerificationCheckSuite
from physverify.extraction._base import StageObserver
from physverify.extraction.pipeline import PipelineDeps
from physverify.extraction.pipeline_resources import ProcRssProbe
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.finetune.candidate_scoring import build_scorer
from physverify.finetune.scorers._base import AggregateScorer
from physverify.generator_comparison.judges._base import ExternalJudge, Judge
from physverify.generator_comparison.settings import ComparisonSettings
from physverify.schema.stages import Stage
from physverify.verification.check_set import active_checks_hash, occlusion_thresholds


class LockingContext(PluginContext, Protocol):
    """A plugin context that also hands out the machine's heavy lock."""

    def heavy_lock(self) -> AbstractContextManager[object]: ...


def dataset_for(context: PluginContext, name: str) -> DatasetSource:
    """Dataset ``name`` of the context's registry, built from its settings."""
    return dataset_from(context.registry("datasets"), name, context.settings)


def pipeline_deps(
    context: LockingContext,
    observer: StageObserver | None,
    heavy_lock: Callable[[], AbstractContextManager[object]] | None = None,
) -> PipelineDeps:
    """``heavy_lock`` replaces the machine lock, e.g. with one that refuses in commands that must
    never run a heavy stage."""
    settings = context.settings
    return PipelineDeps.build(
        settings.paths,
        settings.paths.cache_dir,
        heavy_lock or context.heavy_lock,
        device=settings.runtime.device,
        threads=settings.runtime.threads,
        observer=observer,
        rss=ProcRssProbe(),
    )


def plan_for(
    context: PluginContext,
    *,
    dataset: str,
    gt_stages: tuple[Stage, ...] = (),
    fps: float | None = None,
    max_side: int | None = None,
) -> ResolvedPlan:
    """The extraction plan of the context's providers, checked against its checks' occlusion."""
    return resolve_plan(
        context.settings,
        context.registry("providers"),
        dataset=dataset,
        occlusion=occlusion_thresholds(context.settings, context.registry("checks")),
        gt_stages=gt_stages,
        fps=fps,
        max_side=max_side,
    )


def chain_runner(context: PluginContext, deps: PipelineDeps, dataset: DatasetSource) -> ChainRunner:
    """Chain variants of ``dataset`` over the context's providers and checks."""
    return ChainRunner(
        context.settings, context.registry("providers"), context.registry("checks"), deps, dataset
    )


def checks_hash_for(context: PluginContext) -> str:
    """The ``checks_hash`` of the context's check registry under its ``[checks]`` overrides."""
    return active_checks_hash(context.settings, context.registry("checks"))


def check_suite(context: PluginContext) -> VerificationCheckSuite:
    """The diagnosis adapter over the context's checks, policy and thresholds."""
    return VerificationCheckSuite.build(
        context.settings, context.registry("checks"), context.registry("policies")
    )


def scorer_for(context: PluginContext, name: str) -> AggregateScorer:
    """Scorer ``name`` of the context's registry, over its checks, inside ``[check_scope]``."""
    return build_scorer(
        name,
        context.settings,
        scorers=context.registry("scorers"),
        checks=context.registry("checks"),
    )


def external_judge(context: PluginContext, cfg: ComparisonSettings, workdir: Path) -> Judge:
    """The judge named by ``[h3] external_judge``, built from its own configuration."""
    registry = context.registry("judges")
    judge = registry.get(cfg.external_judge)
    if judge is None:
        raise ConfigError(
            f"[h3] external_judge {cfg.external_judge!r} is unknown; known: {sorted(registry)}"
        )
    if not satisfies(judge, ExternalJudge):
        raise ConfigError(f"[h3] external_judge {cfg.external_judge!r} cannot be built from config")
    built: Judge = judge.from_config(cfg, workdir)
    return built
