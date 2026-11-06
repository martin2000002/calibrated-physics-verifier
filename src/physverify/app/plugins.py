"""Every plugin kind and how its registry is discovered.

``PLUGIN_KINDS`` is the single seam: readers look it up through this module at call time.
The CLI discovers its own commands.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType, ModuleType
from typing import Any

from physverify.checks._base import CHECKS, Check, validate_check
from physverify.checks.variants._base import (
    CHECK_VARIANTS,
    CheckVariant,
    validate_variant,
    variant_key,
)
from physverify.checks.variants.event_gates._base import (
    EVENT_GATES,
    EVENT_GATES_PACKAGE,
    event_gates,
)
from physverify.config.models import Settings
from physverify.config.plugin_context import PluginKind
from physverify.core.errors import InvalidPluginError, UsageError
from physverify.core.registry import Registry, discover, discover_modules
from physverify.datasets._base import DATASETS, DatasetSource
from physverify.diagnosis._base import METRICS, StageMetric, metric_key
from physverify.extraction._base import PROVIDERS, StageProvider, provider_key, validate_provider
from physverify.extraction.centres._base import (
    CENTRE_STRATEGIES,
    CENTRES_PACKAGE,
    centre_strategies,
)
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
from physverify.verification.check_set import (
    CHECK_VARIANTS_SECTION,
    CHECKS_SECTION,
    ChecksSettings,
    CheckVariantsSettings,
    select_variants,
    validate_overrides,
)

CONTAINER_TASKS = PluginKind[ModuleType]("container_tasks")


def check_task(module: ModuleType) -> None:
    if not callable(getattr(module, "run", None)):
        raise InvalidPluginError(f"{module.__name__}: a container task defines run(job)")


def module_name(module: ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


PLUGIN_PACKAGES: Mapping[str, str] = MappingProxyType(
    {
        CHECKS.name: "physverify.checks",
        CHECK_VARIANTS.name: "physverify.checks.variants",
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
        CENTRE_STRATEGIES.name: CENTRES_PACKAGE,
        EVENT_GATES.name: EVENT_GATES_PACKAGE,
    }
)
"""The package each plugin kind is discovered from."""

PLUGIN_KINDS: Mapping[str, Callable[[], Registry[Any]]] = MappingProxyType(
    {
        CHECKS.name: lambda: discover(
            PLUGIN_PACKAGES[CHECKS.name], (Check,), "check_id", validate_check
        ),
        CHECK_VARIANTS.name: lambda: discover(
            PLUGIN_PACKAGES[CHECK_VARIANTS.name], (CheckVariant,), variant_key, validate_variant
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
        CENTRE_STRATEGIES.name: centre_strategies,
        EVENT_GATES.name: event_gates,
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


def _configured_checks(settings: Settings) -> Registry[Any]:
    checks, variants = build(CHECKS.name), build(CHECK_VARIANTS.name)
    validate_overrides(checks, variants, settings.section(CHECKS_SECTION, ChecksSettings))
    selection = settings.section(CHECK_VARIANTS_SECTION, CheckVariantsSettings)
    return select_variants(checks, variants, selection)


CONFIGURED: Mapping[str, Callable[[Settings], Registry[Any]]] = MappingProxyType(
    {CHECKS.name: _configured_checks}
)
"""Kinds whose active registry depends on the settings."""


def configured(kind: str, settings: Settings) -> Registry[Any]:
    """The registry of one plugin kind as ``settings`` selects it ([check_variants] for checks)."""
    select = CONFIGURED.get(kind)
    return build(kind) if select is None else select(settings)
