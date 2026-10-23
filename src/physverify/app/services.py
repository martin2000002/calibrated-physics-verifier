"""Services wired from an application context: what several commands build the same way."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, TypeGuard

from physverify.checks._base import CHECKS
from physverify.config.plugin_context import PluginContext, registry_of
from physverify.core.errors import ConfigError
from physverify.core.registry import satisfies
from physverify.datasets._base import DATASETS, DatasetSource, HumanLabelSource, dataset_from
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.verification_adapter import VerificationCheckSuite
from physverify.extraction._base import PROVIDERS, StageObserver
from physverify.extraction.pipeline import PipelineDeps
from physverify.extraction.pipeline_resources import ProcRssProbe
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.finetune.candidate_scoring import build_scorer
from physverify.finetune.scorers._base import SCORERS, AggregateScorer
from physverify.generator_comparison.judges._base import JUDGES, ExternalJudge, Judge
from physverify.generator_comparison.settings import ComparisonSettings
from physverify.schema.stages import Stage
from physverify.verification._base import POLICIES
from physverify.verification.check_set import active_checks_hash, occlusion_thresholds


class LockingContext(PluginContext, Protocol):
    """A plugin context that also hands out the machine's heavy lock."""

    def heavy_lock(self) -> AbstractContextManager[object]: ...


def dataset_for(context: PluginContext, name: str) -> DatasetSource:
    """Dataset ``name`` of the context's registry, built from its settings."""
    return dataset_from(registry_of(context, DATASETS), name, context.settings)


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
        registry_of(context, PROVIDERS),
        dataset=dataset,
        occlusion=occlusion_thresholds(context.settings, registry_of(context, CHECKS)),
        gt_stages=gt_stages,
        fps=fps,
        max_side=max_side,
    )


def chain_runner(context: PluginContext, deps: PipelineDeps, dataset: DatasetSource) -> ChainRunner:
    """Chain variants of ``dataset`` over the context's providers and checks."""
    return ChainRunner(
        context.settings,
        registry_of(context, PROVIDERS),
        registry_of(context, CHECKS),
        deps,
        dataset,
    )


def checks_hash_for(context: PluginContext) -> str:
    """The ``checks_hash`` of the context's check registry under its ``[checks]`` overrides."""
    return active_checks_hash(context.settings, registry_of(context, CHECKS))


def check_suite(context: PluginContext) -> VerificationCheckSuite:
    """The diagnosis adapter over the context's checks, policy and thresholds."""
    return VerificationCheckSuite.build(
        context.settings, registry_of(context, CHECKS), registry_of(context, POLICIES)
    )


def scorer_for(context: PluginContext, name: str) -> AggregateScorer:
    """Scorer ``name`` of the context's registry, over its checks, inside ``[check_scope]``."""
    return build_scorer(
        name,
        context.settings,
        scorers=registry_of(context, SCORERS),
        checks=registry_of(context, CHECKS),
    )


def external_judge(context: PluginContext, cfg: ComparisonSettings, workdir: Path) -> Judge:
    """The judge named by ``[h3] external_judge``, built from its own configuration."""
    registry = registry_of(context, JUDGES)
    judge = registry.get(cfg.external_judge)
    if judge is None:
        raise ConfigError(
            f"[h3] external_judge {cfg.external_judge!r} is unknown; known: {sorted(registry)}"
        )
    if not _is_external(judge):
        raise ConfigError(f"[h3] external_judge {cfg.external_judge!r} cannot be built from config")
    return judge.from_config(cfg, workdir)


def label_source_class(context: PluginContext, key: str, name: str) -> type[HumanLabelSource]:
    """Dataset ``name`` (configured as ``key``) of the context's registry, which must publish
    human labels."""
    registry = registry_of(context, DATASETS)
    source = registry.get(name)
    if source is None:
        raise ConfigError(f"{key} {name!r} is unknown; known: {sorted(registry)}")
    if not _publishes_labels(source):
        raise ConfigError(f"{key} {name!r} publishes no human labels")
    return source


def _is_external(judge: type[Judge]) -> TypeGuard[type[ExternalJudge]]:
    return satisfies(judge, ExternalJudge)


def _publishes_labels(source: type[DatasetSource]) -> TypeGuard[type[HumanLabelSource]]:
    return satisfies(source, HumanLabelSource)
