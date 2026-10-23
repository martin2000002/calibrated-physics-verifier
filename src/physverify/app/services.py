"""Services wired from an ``AppContext``: what several commands build the same way."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from physverify.app.context import AppContext
from physverify.core.errors import ConfigError, UsageError
from physverify.core.registry import satisfies
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain import ChainRunner
from physverify.diagnosis.checks_port import A7CheckSuite
from physverify.evaluation.judges._base import ExternalJudge, Judge
from physverify.evaluation.settings import H3Settings
from physverify.extraction._base import StageObserver
from physverify.extraction.cache import StageCache
from physverify.extraction.error_profile import load_error_profile
from physverify.extraction.pipeline import LazyLock, PipelineDeps, ProcRssProbe
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scoring import build_scorer
from physverify.schema.stages import Stage
from physverify.verification.check_set import active_checks_hash, occlusion_thresholds


def resolve_dataset(context: AppContext, name: str) -> DatasetSource:
    registry = context.registry("datasets")
    if name not in registry:
        raise UsageError(f"unknown dataset {name!r}; known: {sorted(registry)}")
    source: DatasetSource = registry[name](context.settings)
    return source


def pipeline_deps(
    context: AppContext,
    observer: StageObserver | None,
    heavy_lock: Callable[[], AbstractContextManager[object]] | None = None,
) -> PipelineDeps:
    """``heavy_lock`` replaces the machine lock, e.g. with one that refuses in commands that must
    never run a heavy stage."""
    settings = context.settings
    return PipelineDeps(
        cache=StageCache(settings.paths.cache_dir),
        lock=LazyLock(heavy_lock or context.heavy_lock),
        paths=settings.paths,
        device=settings.runtime.device,
        threads=settings.runtime.threads,
        error_profile=load_error_profile(settings.paths.error_profile),
        observer=observer,
        rss=ProcRssProbe(),
    )


def plan_for(
    context: AppContext,
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


def chain_runner(context: AppContext, deps: PipelineDeps, dataset: DatasetSource) -> ChainRunner:
    """Chain variants of ``dataset`` over the context's providers and checks."""
    return ChainRunner(
        context.settings, context.registry("providers"), context.registry("checks"), deps, dataset
    )


def context_checks_hash(context: AppContext) -> str:
    """The ``checks_hash`` of the context's check registry under its ``[checks]`` overrides."""
    return active_checks_hash(context.settings, context.registry("checks"))


def check_suite(context: AppContext) -> A7CheckSuite:
    """The diagnosis adapter over the context's checks, policy and thresholds."""
    return A7CheckSuite.build(
        context.settings, context.registry("checks"), context.registry("policies")
    )


def context_scorer(context: AppContext, name: str) -> AggregateScorer:
    """Scorer ``name`` of the context's registry, over its checks, inside ``[check_scope]``."""
    return build_scorer(
        name,
        context.settings,
        scorers=context.registry("scorers"),
        checks=context.registry("checks"),
    )


def external_judge(context: AppContext, cfg: H3Settings, workdir: Path) -> Judge:
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
