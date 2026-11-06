"""Wiring of a learned variant's gates and controls from the application context.

The training selection must be split=train and off every seal; only its internal validation
clips are scored. C-hdri reads the same validation clips, split by whether a training scene used
each scene's background; a background selection, when given, must be the training selection
itself. Every sample file must hold the arm its gate reads (ground truth for G-L0, perception for
the rest), rows read with the artifact's encoder from the frames its gate reads (the whole clip,
or the middle frame for the single-frame control), and name only clips of its selection. Each
refusal comes before anything is read.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from physverify.app.context import AppContext
from physverify.app.learned import selected_clips, training_clips
from physverify.checks.variants._learned import LearnedArtifact
from physverify.config.plugin_context import registry_of
from physverify.core.digests import file_sha256
from physverify.core.errors import MissingInputError, UsageError
from physverify.core.feature_table import FeatureTable
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource
from physverify.extraction.pair_features import ZONE_MAX
from physverify.learned.arms import PERCEIVED_ARM, TRUTH_ARM
from physverify.learned.controls import ControlResult, ControlSettings, Status, clip_scores
from physverify.learned.gates import ArmScores, GateReport, run_gates
from physverify.learned.image_table import ALL_FRAMES, MIDDLE_FRAME
from physverify.learned.populations import (
    ClipFacts,
    aimed_checks,
    populations,
    validation_facts,
)
from physverify.learned.samples import SampleSet, TableSource
from physverify.learned.train import TrainSettings
from physverify.learned.visibility import PositiveEvidence
from physverify.schema.clip import ClipManifest
from physverify.schema.scene import PerturbationRecord, SceneSpec
from physverify.synth.perturb import seed_stream
from physverify.synth.perturbations._base import PERTURBATIONS

GATES_COMMAND = "learned-gates"
UNKNOWN_BACKGROUND = "unknown"


class SampleFileError(UsageError):
    """A sample file holds another arm, another check's artifact or clips outside its selection."""


@dataclass(frozen=True)
class GateFiles:
    """The artifact and the sample files of each arm."""

    artifact: Path
    oracle: Path
    perceived: Path
    single_frame: Path


@dataclass(frozen=True)
class GateSelections:
    """(dataset, subset) of the training set and, optionally, of the background control: only
    the training set itself is allowed."""

    training: tuple[str, str | None]
    hdri: tuple[str, str | None] | None = None


class BackgroundSelectionError(UsageError):
    """A background control selection other than the training selection."""


def require_internal_hdri(selections: GateSelections) -> None:
    if selections.hdri is not None and selections.hdri != selections.training:
        raise BackgroundSelectionError(
            f"{GATES_COMMAND}: C-hdri reads the internal validation of {selections.training} "
            f"only, not {selections.hdri}"
        )


def aimed(app: AppContext) -> dict[str, tuple[str, ...]]:
    plugins = registry_of(app, PERTURBATIONS)
    return aimed_checks(
        {name: tuple(p.target_checks) for name, p in plugins.items()},
        {name: seed_stream(p) for name, p in plugins.items()},
    )


@runtime_checkable
class RenderedBackground(Protocol):
    """A dataset that reads the background each clip's render recorded."""

    def rendered_background(self, clip: ClipManifest) -> str | None: ...


def _scene(dataset: DatasetSource, clip: ClipManifest) -> SceneSpec | None:
    try:
        store = dataset.ground_truth(clip)
    except MissingInputError:
        return None
    return None if store is None else store.scene()


def scene_background(dataset: object, clip: ClipManifest) -> str:
    """The background the clip's render recorded (its HDRI id or its kind), or unknown when the
    render recorded none; never recomputed from the current configuration."""
    if not isinstance(dataset, RenderedBackground):
        return UNKNOWN_BACKGROUND
    found = dataset.rendered_background(clip)
    return UNKNOWN_BACKGROUND if found is None else found


def target_zones(oracle: SampleSet | None) -> dict[str, float]:
    """The largest ground-truth zone share over each clip's target subjects (label > 0)."""
    if oracle is None or ZONE_MAX not in oracle.dense_columns:
        return {}
    index = oracle.dense_columns.index(ZONE_MAX)
    zones: dict[str, float] = {}
    for s in oracle.samples:
        if s.label > 0.0:
            value = float(s.dense[index])
            zones[s.clip_id] = max(zones.get(s.clip_id, value), value)
    return zones


def gate_facts(
    app: AppContext,
    dataset: DatasetSource,
    clips: Sequence[ClipManifest],
    check_id: str,
    oracle: SampleSet | None,
    share: float,
) -> list[ClipFacts]:
    """``share``: the validation share the artifact was trained with."""
    zones = target_zones(oracle)

    def evidence(clip: ClipManifest, record: PerturbationRecord) -> PositiveEvidence:
        scene = _scene(dataset, clip)
        camera = None if scene is None else scene.camera
        return PositiveEvidence(record, camera, zones.get(clip.clip_id))

    def background(clip: ClipManifest) -> str:
        return scene_background(dataset, clip)

    return validation_facts(clips, check_id, share, aimed(app), (evidence, background))


def _training(
    app: AppContext, selections: GateSelections, err: Echo
) -> tuple[DatasetSource, tuple[ClipManifest, ...]]:
    require_internal_hdri(selections)
    dataset, clips = selected_clips(app, selections.training, GATES_COMMAND, err)
    return dataset, training_clips(clips)


def plan_gates(
    app: AppContext, selections: GateSelections, check_id: str, *, err: Echo
) -> dict[str, object]:
    """The populations of every gate and control, from the manifests alone: no sample, no score.
    The visibility of a positive that needs ground-truth image rows stays undetermined; with no
    artifact yet, the validation split is the one training will hold out."""
    dataset, clips = _training(app, selections, err)
    share = TrainSettings().validation_share
    found = populations(gate_facts(app, dataset, clips, check_id, None, share))
    return {"check_id": check_id, "populations": found}


def read_samples(path: Path, arm: str, allowed: frozenset[str], source: TableSource) -> SampleSet:
    """``source``: the frames and encoder the rows must have been read with."""
    samples = SampleSet.from_table(FeatureTable.load(path))
    if samples.source != source:
        raise SampleFileError(f"{path}: rows read from {samples.source}, this gate reads {source}")
    arms = sorted({s.arm for s in samples.samples} - {arm})
    if arms:
        raise SampleFileError(f"{path}: samples of arm(s) {arms}, this gate reads {arm} only")
    outside = sorted({s.clip_id for s in samples.samples} - allowed)
    if outside:
        raise SampleFileError(f"{path}: {len(outside)} clips outside the selection: {outside[:5]}")
    return samples


def arm_scores(artifact: LearnedArtifact, samples: SampleSet) -> dict[str, float]:
    """Each clip's largest estimate over its subjects."""
    if not samples.samples:
        return {}
    dense, tokens, _ = samples.arrays()
    keys = [(s.clip_id, s.scene_id, s.role) for s in samples.samples]
    return {
        c.clip_id: c.score for c in clip_scores(keys, artifact.estimate(dense, tokens).tolist())
    }


def _artifact(path: Path, check_id: str) -> LearnedArtifact:
    artifact = LearnedArtifact.load(path)
    if artifact.check_id != check_id:
        raise SampleFileError(f"{path}: an artifact of {artifact.check_id}, not {check_id}")
    return artifact


def gate_report(
    app: AppContext, selections: GateSelections, check_id: str, files: GateFiles, *, err: Echo
) -> dict[str, object]:
    """Every gate and control of ``check_id`` as one JSON-ready mapping."""
    dataset, clips = _training(app, selections, err)
    allowed = frozenset(c.clip_id for c in clips)
    artifact = _artifact(files.artifact, check_id)
    clip_rows = TableSource(ALL_FRAMES, artifact.encoder_sha256)
    oracle = read_samples(files.oracle, TRUTH_ARM, allowed, clip_rows)
    perceived = read_samples(files.perceived, PERCEIVED_ARM, allowed, clip_rows)
    single = read_samples(
        files.single_frame, PERCEIVED_ARM, allowed, replace(clip_rows, frames=MIDDLE_FRAME)
    )
    facts = gate_facts(app, dataset, clips, check_id, oracle, artifact.validation_share)
    scores = ArmScores(
        arm_scores(artifact, oracle),
        arm_scores(artifact, perceived),
        arm_scores(artifact, single),
    )
    settings = ControlSettings()
    report = run_gates(facts, scores, settings)
    return {
        "check_id": check_id,
        "inputs": _inputs(files),
        "settings": asdict(settings),
        "validation_share": artifact.validation_share,
        "populations": populations(facts),
        "unscored": report.unscored,
        **report_json(report),
    }


def _inputs(files: GateFiles) -> dict[str, str]:
    return {name: f"{path} sha256={file_sha256(path)}" for name, path in asdict(files).items()}


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def result_json(result: ControlResult) -> dict[str, object]:
    ci = result.ci
    return {
        "status": result.status.value,
        "value": _finite(result.value),
        "ci": None if ci is None else [_finite(ci.low), _finite(ci.high)],
        "detail": result.detail,
    }


def report_json(report: GateReport) -> dict[str, object]:
    return {
        "gates": {name: result_json(r) for name, r in report.results.items()},
        "decoy_control_by_origin": {
            origin: result_json(r) for origin, r in report.decoys_by_origin.items()
        },
        "verdict": {
            "G-L0": report.oracle_passed,
            "G-L1_with_controls": report.perceived_passed,
            "C-hdri_flagged": report.results["C-hdri"].status is Status.FAIL,
        },
    }
