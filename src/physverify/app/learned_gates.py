"""Wiring of a learned variant's gates and controls from the application context.

The training selection must be split=train and off every seal; only its internal validation
clips are scored. The unseen-background selection (C-hdri), when given, must be off every seal
and share no perturbation seed with the artifact's training set. Every sample file must hold the
arm its gate reads (ground truth for G-L0, perception for the rest), rows read with the
artifact's encoder from the frames its gate reads (the whole clip, or the middle frame for the
single-frame control), and name only clips of its selection. Each refusal comes before any score
is computed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.learned import selected_clips, training_clips, training_seeds
from physverify.checks.variants._learned import LearnedArtifact, require_disjoint
from physverify.config.plugin_context import registry_of
from physverify.core.digests import file_sha256
from physverify.core.errors import MissingInputError, UsageError
from physverify.core.feature_table import FeatureTable
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource
from physverify.extraction.pair_features import ZONE_MAX
from physverify.learned.arms import PERCEIVED_ARM, TRUTH_ARM
from physverify.learned.controls import (
    ClipScore,
    ControlResult,
    ControlSettings,
    Status,
    clip_scores,
)
from physverify.learned.gates import ArmScores, GateReport, run_gates
from physverify.learned.image_table import ALL_FRAMES, MIDDLE_FRAME
from physverify.learned.populations import (
    ClipFacts,
    aimed_checks,
    populations,
    validation_facts,
)
from physverify.learned.samples import Role, SampleSet, TableSource, clip_role
from physverify.learned.train import TrainSettings
from physverify.learned.visibility import PositiveEvidence
from physverify.schema.clip import ClipManifest
from physverify.schema.scene import CameraSpec, PerturbationRecord
from physverify.synth.perturb import seed_stream
from physverify.synth.perturbations._base import PERTURBATIONS

GATES_COMMAND = "learned-gates"
HDRI_COMMAND = "learned-gates-hdri"


class SampleFileError(UsageError):
    """A sample file holds another arm, another check's artifact or clips outside its selection."""


@dataclass(frozen=True)
class GateFiles:
    """The artifact and the sample files of each arm; ``hdri`` with ``GateSelections.hdri``."""

    artifact: Path
    oracle: Path
    perceived: Path
    single_frame: Path
    hdri: Path | None = None


@dataclass(frozen=True)
class GateSelections:
    """(dataset, subset) of the training set and, optionally, of the unseen-background set."""

    training: tuple[str, str | None]
    hdri: tuple[str, str | None] | None = None


def aimed(app: AppContext) -> dict[str, tuple[str, ...]]:
    plugins = registry_of(app, PERTURBATIONS)
    return aimed_checks(
        {name: tuple(p.target_checks) for name, p in plugins.items()},
        {name: seed_stream(p) for name, p in plugins.items()},
    )


def _camera(dataset: DatasetSource, clip: ClipManifest) -> CameraSpec | None:
    try:
        store = dataset.ground_truth(clip)
    except MissingInputError:
        return None
    scene = None if store is None else store.scene()
    return None if scene is None else scene.camera


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
        return PositiveEvidence(record, _camera(dataset, clip), zones.get(clip.clip_id))

    return validation_facts(clips, check_id, share, aimed(app), evidence)


def _training(
    app: AppContext, selection: tuple[str, str | None], err: Echo
) -> tuple[DatasetSource, tuple[ClipManifest, ...]]:
    dataset, clips = selected_clips(app, selection, GATES_COMMAND, err)
    return dataset, training_clips(clips)


def _hdri_clips(
    app: AppContext, selection: tuple[str, str | None] | None, err: Echo
) -> tuple[ClipManifest, ...] | None:
    if selection is None:
        return None
    return tuple(selected_clips(app, selection, HDRI_COMMAND, err)[1])


def plan_gates(
    app: AppContext, selections: GateSelections, check_id: str, *, err: Echo
) -> dict[str, object]:
    """The populations of every gate and control, from the manifests alone: no sample, no score.
    The visibility of a positive that needs ground-truth image rows stays undetermined; with no
    artifact yet, the validation split is the one training will hold out."""
    dataset, clips = _training(app, selections.training, err)
    hdri = _hdri_clips(app, selections.hdri, err)
    share = TrainSettings().validation_share
    found = populations(gate_facts(app, dataset, clips, check_id, None, share))
    if hdri is not None:
        found["C-hdri"] = _hdri_population(hdri, check_id)
    return {"check_id": check_id, "populations": found}


def _hdri_population(clips: Sequence[ClipManifest], check_id: str) -> dict[str, int]:
    roles = [clip_role(c.perturbation, check_id) for c in clips]
    return {r.value: roles.count(r) for r in (Role.POSITIVE, Role.CLEAN)}


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


def _hdri_scores(
    artifact: LearnedArtifact,
    clips: Sequence[ClipManifest] | None,
    path: Path | None,
    check_id: str,
) -> list[ClipScore] | None:
    if clips is None or path is None:
        if (clips is None) != (path is None):
            raise UsageError("the unseen-background set needs both its selection and samples")
        return None
    require_disjoint(artifact.training_seeds, training_seeds(clips))
    source = TableSource(ALL_FRAMES, artifact.encoder_sha256)
    samples = read_samples(path, PERCEIVED_ARM, frozenset(c.clip_id for c in clips), source)
    scores = arm_scores(artifact, samples)
    by_id = {c.clip_id: c for c in clips}
    found = []
    for clip_id, score in scores.items():
        clip = by_id[clip_id]
        role = clip_role(clip.perturbation, check_id)
        if role is not None:
            found.append(ClipScore(clip_id, clip.scene_id or clip_id, role, score))
    return found


def gate_report(
    app: AppContext, selections: GateSelections, check_id: str, files: GateFiles, *, err: Echo
) -> dict[str, object]:
    """Every gate and control of ``check_id`` as one JSON-ready mapping."""
    dataset, clips = _training(app, selections.training, err)
    hdri = _hdri_clips(app, selections.hdri, err)
    allowed = frozenset(c.clip_id for c in clips)
    artifact = _artifact(files.artifact, check_id)
    clip_rows = TableSource(ALL_FRAMES, artifact.encoder_sha256)
    oracle = read_samples(files.oracle, TRUTH_ARM, allowed, clip_rows)
    perceived = read_samples(files.perceived, PERCEIVED_ARM, allowed, clip_rows)
    single = read_samples(
        files.single_frame, PERCEIVED_ARM, allowed, replace(clip_rows, frames=MIDDLE_FRAME)
    )
    hdri_scores = _hdri_scores(artifact, hdri, files.hdri, check_id)
    facts = gate_facts(app, dataset, clips, check_id, oracle, artifact.validation_share)
    scores = ArmScores(
        arm_scores(artifact, oracle),
        arm_scores(artifact, perceived),
        arm_scores(artifact, single),
        hdri_scores,
    )
    settings = ControlSettings()
    report = run_gates(facts, scores, settings)
    found = populations(facts)
    if hdri is not None:
        found["C-hdri"] = _hdri_population(hdri, check_id)
    return {
        "check_id": check_id,
        "inputs": _inputs(files),
        "settings": asdict(settings),
        "populations": found,
        "unscored": _unscored(facts, scores),
        **report_json(report),
    }


def _inputs(files: GateFiles) -> dict[str, str | None]:
    return {
        name: None if path is None else f"{path} sha256={file_sha256(path)}"
        for name, path in asdict(files).items()
    }


def _unscored(facts: Sequence[ClipFacts], scores: ArmScores) -> dict[str, int]:
    arms: Mapping[str, Mapping[str, float]] = {
        "oracle": scores.oracle,
        "perceived": scores.perceived,
        "single_frame": scores.single_frame,
    }
    return {arm: sum(f.clip_id not in s for f in facts) for arm, s in arms.items()}


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
