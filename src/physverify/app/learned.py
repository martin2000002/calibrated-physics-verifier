"""Wiring of a learned variant's training data and training from the application context.

``warm`` fills the pipeline cache, each clip under the arm it draws; it is the only step that may
run a heavy stage. Image rows and samples are read from that cache only (a heavy stage that is
not cached fails its clip, as in ``verify``). Each clip runs under the substitution arm it draws
(``learned.arms``) and its inputs are routed as ``verify`` routes them (``variant_inputs``), so
training sees exactly what evaluation will. A selection that looks at a sealed set is refused
before any clip is read (never unsealed: training reads its data many times), and a set of
evaluated manifests that shares a perturbation seed with the training set is refused before
training.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial

from physverify.app.context import AppContext
from physverify.app.sealing import Selection, sealed_looks
from physverify.app.services import chain_runner, dataset_for, pipeline_deps
from physverify.calibration.sealed_ledger import SealedSetError
from physverify.checks.variants._base import CHECK_VARIANTS
from physverify.checks.variants._learned import LearnedMixin, dense_layout, require_disjoint
from physverify.config.plugin_context import registry_of
from physverify.core.errors import ConfigError
from physverify.core.feature_table import FeatureTable
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain_runner import ChainRunner, warm_variants
from physverify.diagnosis.report import Failure
from physverify.diagnosis.verification_adapter import variant_inputs
from physverify.extraction.contact_crops import CropSettings
from physverify.extraction.crop_encoder import CropEncoder
from physverify.learned.arms import Arm, arm_groups, arm_level, arm_name, arm_variant, drawn_arm
from physverify.learned.image_table import (
    ClipImages,
    clip_images,
    clip_rows,
    image_table,
    object_pairs,
)
from physverify.learned.samples import Sample, SampleSet, subject_samples
from physverify.learned.train import Trained, TrainSettings, train
from physverify.schema.clip import ClipManifest
from physverify.schema.perception import DepthMap, MaskTrack
from physverify.schema.stages import Stage
from physverify.verification.check_set import (
    CHECKS_SECTION,
    ChecksSettings,
    build_checks,
)
from physverify.verification.inputs import refuse_heavy

COMMAND = "learned-samples"
IMAGES_COMMAND = "learned-images"
WARM_COMMAND = "learned-warm"
VARIANT = "learned"


@dataclass(frozen=True)
class Collected:
    samples: SampleSet
    failed: tuple[str, ...]


def cached_runner(app: AppContext, dataset: DatasetSource) -> ChainRunner:
    """A chain that reads the cache and refuses to run a heavy stage."""
    return chain_runner(app, pipeline_deps(app, None, partial(refuse_heavy, dataset.name)), dataset)


def _selected(
    app: AppContext, selection: tuple[str, str | None], command: str, err: Echo
) -> tuple[DatasetSource, tuple[ClipManifest, ...]]:
    name, subset = selection
    dataset = dataset_for(app, name)
    clips = tuple(dataset.clips(subset))
    found = sealed_looks(app, [Selection(name, subset, clips)], err=err)
    if found:
        raise SealedSetError(
            f"{command}: {name} {subset or ''} looks at sealed sets {list(found)}; training data "
            "never comes from a sealed set and nothing was read"
        )
    return dataset, clips


def warm(
    app: AppContext, selection: tuple[str, str | None], arm_draw: tuple[int, float], *, err: Echo
) -> tuple[Failure, ...]:
    """Run every clip's chain under the arm it draws, one variant at a time."""
    dataset, clips = _selected(app, selection, WARM_COMMAND, err)
    by_id = {clip.clip_id: clip for clip in clips}
    runner = chain_runner(app, pipeline_deps(app, None), dataset)
    failures: list[Failure] = []
    for arm, ids in arm_groups(by_id, *arm_draw).items():
        failures += warm_variants(runner, [by_id[i] for i in ids], (arm_variant(arm),))
    return tuple(failures)


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
    dataset, clips = _selected(app, selection, COMMAND, err)
    check = learned_check(app, check_id)
    runner = cached_runner(app, dataset)
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


@dataclass(frozen=True)
class CollectedImages:
    table: FeatureTable
    failed: tuple[str, ...]


@dataclass(frozen=True)
class ImagePlan:
    """What ``collect_images`` would read, without the encoder: clips, pairs and failures."""

    clips: int
    pairs: int
    arms: dict[str, int]
    failed: tuple[str, ...]


def read_images(runner: ChainRunner, clip: ClipManifest, arm: Arm) -> ClipImages | None:
    """The masks, depth and frames of ``clip`` under ``arm``; None when a stage failed."""
    loaded = runner.outputs(clip, arm_variant(arm))
    if loaded is None:
        return None
    track = loaded.outputs.get(Stage.TRACK)
    depth = loaded.outputs.get(Stage.DEPTH)
    if not isinstance(track, MaskTrack) or not isinstance(depth, DepthMap):
        return None
    video = runner.dataset.video(clip)
    return clip_images(clip.clip_id, loaded.plan.names, track, depth, video)


def _clip_images(
    app: AppContext,
    selection: tuple[str, str | None],
    arm_draw: tuple[int, float],
    *,
    err: Echo,
) -> Iterator[tuple[ClipManifest, Arm, ClipImages | None]]:
    dataset, clips = _selected(app, selection, IMAGES_COMMAND, err)
    runner = cached_runner(app, dataset)
    for clip in clips:
        arm = drawn_arm(clip.clip_id, *arm_draw)
        yield clip, arm, read_images(runner, clip, arm)


def collect_images(
    app: AppContext,
    selection: tuple[str, str | None],
    arm_draw: tuple[int, float],
    encoder: tuple[CropEncoder, CropSettings],
    *,
    err: Echo,
) -> CollectedImages:
    """The pair image table of every clip under the arm it draws (``arm_draw`` as ``collect``),
    so the rows carry the providers ``collect`` and ``verify`` look them up by."""
    rows = []
    failed: list[str] = []
    for clip, _, images in _clip_images(app, selection, arm_draw, err=err):
        if images is None:
            failed.append(clip.clip_id)
            continue
        rows.append(clip_rows(images, *encoder))
    return CollectedImages(image_table(rows, *encoder), tuple(failed))


def plan_images(
    app: AppContext, selection: tuple[str, str | None], arm_draw: tuple[int, float], *, err: Echo
) -> ImagePlan:
    """``collect_images`` up to the encoder: reads every clip's inputs and counts its pairs."""
    pairs = 0
    arms: dict[str, int] = {}
    failed: list[str] = []
    clips = 0
    for clip, arm, images in _clip_images(app, selection, arm_draw, err=err):
        clips += 1
        if images is None:
            failed.append(clip.clip_id)
            continue
        pairs += len(object_pairs(images.track))
        arms[arm_name(arm)] = arms.get(arm_name(arm), 0) + 1
    return ImagePlan(clips, pairs, arms, tuple(failed))


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
