"""Every plugin kind and how its registry is discovered.

``PLUGIN_KINDS`` is the single seam: readers look it up through this module at call time.
The CLI discovers its own commands.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType, ModuleType
from typing import Any

from physverify.checks._base import CHECKS, Check, validate_check
from physverify.core.errors import InvalidPluginError, UsageError
from physverify.core.registry import Registry, discover, discover_modules
from physverify.datasets._base import DatasetSource
from physverify.diagnosis._base import StageMetric, metric_key
from physverify.extraction._base import StageProvider, provider_key, validate_provider
from physverify.finetune.scorers._base import SCORERS, AggregateScorer
from physverify.generator_comparison.judges._base import Judge
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


def check_task(module: ModuleType) -> None:
    if not callable(getattr(module, "run", None)):
        raise InvalidPluginError(f"{module.__name__}: a container task defines run(job)")


def module_name(module: ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


PLUGIN_KINDS: Mapping[str, Callable[[], Registry[Any]]] = MappingProxyType(
    {
        CHECKS.name: lambda: discover("physverify.checks", (Check,), "check_id", validate_check),
        PERTURBATIONS.name: lambda: discover(
            "physverify.synth.perturbations",
            (ScenePerturbation, TrajectoryPerturbation),
            "name",
            validate_perturbation,
        ),
        FAMILIES.name: lambda: discover("physverify.synth.families", (SceneFamily,), "name"),
        "providers": lambda: discover(
            "physverify.extraction.providers", (StageProvider,), provider_key, validate_provider
        ),
        "datasets": lambda: discover("physverify.datasets", (DatasetSource,), "name"),
        "container_tasks": lambda: discover_modules(TASKS_PACKAGE, check_task, module_name),
        "metrics": lambda: discover("physverify.diagnosis.metrics", (StageMetric,), metric_key),
        POLICIES.name: lambda: discover(
            "physverify.verification.policies", (VerdictPolicy,), "name"
        ),
        TIMING_TARGETS.name: lambda: discover("physverify.timing.targets", (TimingTarget,), "name"),
        "judges": lambda: discover("physverify.generator_comparison.judges", (Judge,), "name"),
        SCORERS.name: lambda: discover("physverify.finetune.scorers", (AggregateScorer,), "name"),
        TRUTH_MEASURES.name: lambda: discover(
            "physverify.synth.truth_measures", (TruthMeasure,), "check_id"
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
