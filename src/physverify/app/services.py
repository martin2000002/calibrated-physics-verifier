"""Services wired from an ``AppContext``: what several commands build the same way."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

from physverify.app.context import AppContext
from physverify.calibration.settings import active_checks_hash
from physverify.core.errors import UsageError
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.checks_port import A7CheckSuite
from physverify.extraction._base import StageObserver
from physverify.extraction.cache import StageCache
from physverify.extraction.error_profile import load_error_profile
from physverify.extraction.pipeline import LazyLock, PipelineDeps, ProcRssProbe


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


def context_checks_hash(context: AppContext) -> str:
    """The ``checks_hash`` of the context's check registry under its ``[checks]`` overrides."""
    return active_checks_hash(context.settings, context.registry("checks"))


def check_suite(context: AppContext) -> A7CheckSuite:
    """The diagnosis adapter over the context's checks, policy and thresholds."""
    return A7CheckSuite.build(
        context.settings, context.registry("checks"), context.registry("policies")
    )
