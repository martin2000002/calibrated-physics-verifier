"""Every plugin kind and how its registry is discovered.

``PLUGIN_KINDS`` is the single seam: readers look it up through this module at call time.
The CLI discovers its own commands.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType, ModuleType
from typing import Any

from physverify.checks._base import Check
from physverify.core.errors import InvalidPluginError, UsageError
from physverify.core.registry import Registry, discover, discover_modules
from physverify.datasets._base import DatasetSource
from physverify.diagnosis._base import StageMetric, metric_key
from physverify.extraction._base import StageProvider, provider_key, validate_provider
from physverify.finetune.scorers._base import AggregateScorer
from physverify.synth.families._base import SceneFamily
from physverify.synth.perturbations._base import (
    ScenePerturbation,
    TrajectoryPerturbation,
    validate_perturbation,
)
from physverify.timing._base import TimingTarget
from physverify.verification._base import VerdictPolicy

TASKS_PACKAGE = "physverify.kubric_worker.tasks"


def check_task(module: ModuleType) -> None:
    if not callable(getattr(module, "run", None)):
        raise InvalidPluginError(f"{module.__name__}: a container task defines run(job)")


def module_name(module: ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


PLUGIN_KINDS: Mapping[str, Callable[[], Registry[Any]]] = MappingProxyType(
    {
        "checks": lambda: discover("physverify.checks", (Check,), "check_id"),
        "perturbations": lambda: discover(
            "physverify.synth.perturbations",
            (ScenePerturbation, TrajectoryPerturbation),
            "name",
            validate_perturbation,
        ),
        "families": lambda: discover("physverify.synth.families", (SceneFamily,), "name"),
        "providers": lambda: discover(
            "physverify.extraction.providers", (StageProvider,), provider_key, validate_provider
        ),
        "datasets": lambda: discover("physverify.datasets", (DatasetSource,), "name"),
        "container_tasks": lambda: discover_modules(TASKS_PACKAGE, check_task, module_name),
        "metrics": lambda: discover("physverify.diagnosis.metrics", (StageMetric,), metric_key),
        "policies": lambda: discover("physverify.verification.policies", (VerdictPolicy,), "name"),
        "timing_targets": lambda: discover("physverify.timing.targets", (TimingTarget,), "name"),
        "scorers": lambda: discover("physverify.finetune.scorers", (AggregateScorer,), "name"),
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
