"""Wiring of a learned variant's training data and training from the application context.

Samples come from the pipeline cache: each clip runs under the substitution arm it draws
(``learned.arms``) and its inputs are routed as ``verify`` routes them (``variant_inputs``), so
training sees exactly what evaluation will. A sealed dataset is refused by ``guard_sealed``
before any clip is read, and a set of evaluated manifests that shares a perturbation seed with
the training set is refused before training.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import chain_runner, dataset_for, pipeline_deps
from physverify.checks.variants._base import CHECK_VARIANTS
from physverify.checks.variants._learned import LearnedMixin, dense_layout, require_disjoint
from physverify.config.plugin_context import registry_of
from physverify.core.errors import ConfigError
from physverify.core.feature_table import FeatureTable
from physverify.core.output import Echo
from physverify.diagnosis.verification_adapter import variant_inputs
from physverify.learned.arms import arm_level, arm_name, arm_variant, drawn_arm
from physverify.learned.samples import Sample, SampleSet, subject_samples
from physverify.learned.train import Trained, TrainSettings, train
from physverify.schema.clip import ClipManifest
from physverify.verification.check_set import (
    CHECKS_SECTION,
    ChecksSettings,
    build_checks,
)

COMMAND = "learned-samples"
VARIANT = "learned"


@dataclass(frozen=True)
class Collected:
    samples: SampleSet
    failed: tuple[str, ...]


def learned_check(app: AppContext, check_id: str) -> LearnedMixin:
    """``<check_id>/learned`` under the context's ``[checks]`` tables (no artifact is read)."""
    key = f"{check_id}/{VARIANT}"
    variants = registry_of(app, CHECK_VARIANTS)
    if key not in variants:
        raise ConfigError(f"no check variant {key}")
    overrides = app.settings.section(CHECKS_SECTION, ChecksSettings)
    (check,) = build_checks({check_id: variants[key]}, overrides)
    if not isinstance(check, LearnedMixin):
        raise ConfigError(f"{key} is not a learned variant")
    return check


def collect(
    app: AppContext,
    selection: tuple[str, str | None],
    check_id: str,
    images: tuple[FeatureTable, tuple[str, ...], tuple[str, ...]],
    arm_draw: tuple[int, float],
    *,
    err: Echo,
) -> Collected:
    """``selection``: (dataset, subset); ``images``: the pair image table and the dense and token
    columns read from it; ``arm_draw``: (seed, ground-truth share per stage)."""
    name, subset = selection
    dataset = dataset_for(app, name)
    clips = tuple(dataset.clips(subset))
    guard_sealed(app, COMMAND, [Selection(name, subset, clips)], err=err)
    check = learned_check(app, check_id)
    runner = chain_runner(app, pipeline_deps(app, None), dataset)
    table, dense_columns, token_columns = images
    samples: list[Sample] = []
    failed: list[str] = []
    for clip in clips:
        arm = drawn_arm(clip.clip_id, *arm_draw)
        loaded = runner.outputs(clip, arm_variant(arm))
        if loaded is None:
            failed.append(clip.clip_id)
            continue
        routed = variant_inputs(runner, clip, loaded, arm_level(arm))
        inputs = routed.by_check.get(check_id, routed).inputs
        if inputs is None:
            failed.append(clip.clip_id)
            continue
        samples += subject_samples(
            check, inputs, table, (dense_columns, token_columns), arm_name(arm)
        )
    layout = dense_layout(type(check).TRACK_COLUMNS, dense_columns)
    return Collected(SampleSet(tuple(samples), layout, token_columns), tuple(failed))


def training_seeds(clips: Iterable[ClipManifest]) -> frozenset[int]:
    return frozenset(int(c.perturbation.seed) for c in clips if c.perturbation is not None)


def fit(
    samples: SampleSet,
    identity: tuple[str, str, str],
    training: frozenset[int],
    evaluated: Sequence[Iterable[int]],
    settings: TrainSettings,
) -> Trained:
    """Train after refusing any evaluated seed set that meets the training seeds."""
    for seeds in evaluated:
        require_disjoint(training, seeds)
    return train(samples, settings, identity, training)
