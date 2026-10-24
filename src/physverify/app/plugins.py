"""Every plugin kind and how its registry is discovered.

``PLUGIN_KINDS`` is the single seam: readers look it up through this module at call time.
The CLI discovers its own commands.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType, ModuleType
from typing import Any

from physverify.checks._base import CHECKS, Check, validate_check
from physverify.config.plugin_context import PluginKind
from physverify.core.errors import InvalidPluginError, UsageError
from physverify.core.registry import Registry, discover, discover_modules
from physverify.datasets._base import DATASETS, DatasetSource
from physverify.diagnosis._base import METRICS, StageMetric, metric_key
from physverify.extraction._base import PROVIDERS, StageProvider, provider_key, validate_provider
from physverify.finetune.scorers._base import SCORERS, AggregateScorer
from physverify.generator_comparison.judges._base import JUDGES, Judge
from physverify.kubric_worker import TASKS_PACKAGE
from physverify.synth.families._base import FAMILIES, SceneFamily
from physverify.synth.perturbations._base import (
    PERTURBATIONS,
    ScenePerturbation,
    TrajectoryPerturbation,
    validate_perturbation,
)
from physverify.synth.truth_measures._base import TRUTH_MEASURES, TruthMeasure
from physverify.timing._base import TIMING_TARGETS, TimingTarget
from physverify.verification._base import POLICIES, VerdictPolicy

CONTAINER_TASKS = PluginKind[ModuleType]("container_tasks")


def check_task(module: ModuleType) -> None:
    if not callable(getattr(module, "run", None)):
        raise InvalidPluginError(f"{module.__name__}: a container task defines run(job)")


def module_name(module: ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


PLUGIN_PACKAGES: Mapping[str, str] = MappingProxyType(
    {
        CHECKS.name: "physverify.checks",
        PERTURBATIONS.name: "physverify.synth.perturbations",
        FAMILIES.name: "physverify.synth.families",
        PROVIDERS.name: "physverify.extraction.providers",
        DATASETS.name: "physverify.datasets",
        CONTAINER_TASKS.name: TASKS_PACKAGE,
        METRICS.name: "physverify.diagnosis.metrics",
        POLICIES.name: "physverify.verification.policies",
        TIMING_TARGETS.name: "physverify.timing.targets",
        JUDGES.name: "physverify.generator_comparison.judges",
        SCORERS.name: "physverify.finetune.scorers",
        TRUTH_MEASURES.name: "physverify.synth.truth_measures",
    }
)
"""The package each plugin kind is discovered from."""

PLUGIN_KINDS: Mapping[str, Callable[[], Registry[Any]]] = MappingProxyType(
    {
        CHECKS.name: lambda: discover(
            PLUGIN_PACKAGES[CHECKS.name], (Check,), "check_id", validate_check
        ),
        PERTURBATIONS.name: lambda: discover(
            PLUGIN_PACKAGES[PERTURBATIONS.name],
            (ScenePerturbation, TrajectoryPerturbation),
            "name",
            validate_perturbation,
        ),
        FAMILIES.name: lambda: discover(PLUGIN_PACKAGES[FAMILIES.name], (SceneFamily,), "name"),
        PROVIDERS.name: lambda: discover(
            PLUGIN_PACKAGES[PROVIDERS.name], (StageProvider,), provider_key, validate_provider
        ),
        DATASETS.name: lambda: discover(PLUGIN_PACKAGES[DATASETS.name], (DatasetSource,), "name"),
        CONTAINER_TASKS.name: lambda: discover_modules(
            PLUGIN_PACKAGES[CONTAINER_TASKS.name], check_task, module_name
        ),
        METRICS.name: lambda: discover(PLUGIN_PACKAGES[METRICS.name], (StageMetric,), metric_key),
        POLICIES.name: lambda: discover(PLUGIN_PACKAGES[POLICIES.name], (VerdictPolicy,), "name"),
        TIMING_TARGETS.name: lambda: discover(
            PLUGIN_PACKAGES[TIMING_TARGETS.name], (TimingTarget,), "name"
        ),
        JUDGES.name: lambda: discover(PLUGIN_PACKAGES[JUDGES.name], (Judge,), "name"),
        SCORERS.name: lambda: discover(PLUGIN_PACKAGES[SCORERS.name], (AggregateScorer,), "name"),
        TRUTH_MEASURES.name: lambda: discover(
            PLUGIN_PACKAGES[TRUTH_MEASURES.name], (TruthMeasure,), "check_id"
        ),
    }
)
"""Perceived and ground-truth providers share ``providers`` (keys ``<stage>/<name>`` and
``<stage>/ground_truth``)."""


def build(kind: str) -> Registry[Any]:
    """Discover the registry of one plugin kind."""
    kinds = PLUGIN_KINDS
    if kind not in kinds:
        raise UsageError(f"unknown plugin kind {kind!r}; known: {sorted(kinds)}")
    return kinds[kind]()
