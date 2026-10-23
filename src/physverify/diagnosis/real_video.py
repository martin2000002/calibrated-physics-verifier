"""Real-video diagnosis (ADR 0007 §5.5): end-to-end only, since real clips have no per-stage truth.

Physics-IQ clips are scored against their pixel-change masks, PISA drops by a free-fall fit, and
every clip goes through the check suite to count verdicts (a FAIL on real video is a false alarm).
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from physverify.core.download import MANIFEST_FILE, read_manifest
from physverify.core.errors import MissingInputError, PhysverifyError, UsageError
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.cases import stage_output
from physverify.diagnosis.chain import ChainRunner, VariantOutputs, warm_variants
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.end_to_end import decided, track_of, variant_outputs, variant_results
from physverify.diagnosis.plans import perceived_variant
from physverify.diagnosis.real_metrics import TimeBases, fall_diagnosis, motion_scores
from physverify.diagnosis.report import (
    Failure,
    FallRow,
    MotionRow,
    RealVideoReport,
    VerdictDistribution,
    build_provenance,
)
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, FrameSampling
from physverify.schema.scale import ScaleEstimate, effective_scale
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

ALL_CLIPS = "(all)"
NO_SUITE = "A7's check suite is unavailable: verdict distributions were not computed"
ESTIMATED_FALL = ("g_est", "real_fps")
ESTIMATED_GRAVITY_DIR = "gravity_dir"
PROVISIONAL_VERDICTS = "fail_rate"


@runtime_checkable
class MotionMaskSource(Protocol):
    """A dataset with pixel-change masks; ``motion_masks`` returns grey levels (T, h, w)."""

    def motion_masks(self, clip: ClipManifest, sampling: FrameSampling) -> NDArray[np.uint8]: ...

    def is_static(self, clip: ClipManifest) -> bool: ...


@dataclass
class _Collected:
    motion: list[MotionRow] = field(default_factory=list)
    falls: list[FallRow] = field(default_factory=list)
    verdicts: list[VerdictDistribution] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    clips: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)


def select_clips(
    ctx: DiagnosisContext, datasets: Sequence[str], subset: str, clip_ids: Iterable[str] = ()
) -> list[tuple[DatasetSource, tuple[ClipManifest, ...]]]:
    """The clips of each dataset, restricted to ``clip_ids`` when given (sorted by id)."""
    wanted = set(clip_ids)
    selections: list[tuple[DatasetSource, tuple[ClipManifest, ...]]] = []
    for name in datasets:
        dataset = ctx.dataset(name)
        clips = sorted(dataset.clips(subset), key=lambda c: c.clip_id)
        chosen = tuple(c for c in clips if not wanted or c.clip_id in wanted)
        if chosen:
            selections.append((dataset, chosen))
    unknown = wanted - {c.clip_id for _, clips in selections for c in clips}
    if unknown:
        raise UsageError(f"unknown clip ids {sorted(unknown)} in {', '.join(datasets)}")
    return selections


def motion_row(
    ctx: DiagnosisContext, source: MotionMaskSource, clip: ClipManifest, outputs: VariantOutputs
) -> MotionRow:
    track = track_of(outputs)
    static = source.is_static(clip)
    row = MotionRow(
        dataset=clip.dataset,
        clip_id=clip.clip_id,
        scenario=clip.scene_id or clip.clip_id,
        dynamic=not static,
        n_frames=0,
    )
    step = track.sampling.step
    if step != 1:
        reason = f"sampling step {step} != 1: the mask rate differs from extraction.target_fps"
        return row.model_copy(update={"unavailable": reason})
    real = ctx.diag.real
    grey = source.motion_masks(clip, track.sampling)
    scores = motion_scores(track, grey > real.motion_threshold, real)
    return row.model_copy(
        update={
            "n_frames": scores.n_frames,
            "motion_iou": scores.motion_iou,
            "precision": scores.precision,
            "recall": scores.recall,
            "false_motion_rate": scores.false_motion_rate if static else None,
        }
    )


def fall_row(ctx: DiagnosisContext, clip: ClipManifest, outputs: VariantOutputs) -> FallRow:
    states = stage_output(outputs.outputs, Stage.KINEMATICS, ObjectStateTable)
    if states is None:
        raise MissingInputError(f"{clip.clip_id}: no {Stage.KINEMATICS.value} output")
    estimate = stage_output(outputs.outputs, Stage.SCALE, ScaleEstimate)
    scale = effective_scale(states, estimate)
    fps = TimeBases(declared_fps=clip.real_fps, container_fps=clip.container_fps)
    diagnosis = fall_diagnosis(states, scale, ctx.diag.real, fps)
    return FallRow(dataset=clip.dataset, clip_id=clip.clip_id, **dataclasses.asdict(diagnosis))


def verdict_distributions(
    dataset: str, results: Sequence[CheckResult]
) -> tuple[VerdictDistribution, ...]:
    """Per check: verdict counts, ``fail_rate`` = #FAIL / #(PASS + FAIL) and abstentions."""
    by_check: defaultdict[str, list[CheckResult]] = defaultdict(list)
    for result in results:
        by_check[result.check_id].append(result)
    rows: list[VerdictDistribution] = []
    for check_id in sorted(by_check):
        part = by_check[check_id]
        counts = {v: sum(r.verdict is v for r in part) for v in Verdict}
        n_decided = counts[Verdict.PASS] + counts[Verdict.FAIL]
        reasons = Counter(r.abstain_reason for r in part if r.abstain_reason is not None)
        rows.append(
            VerdictDistribution(
                dataset=dataset,
                check_id=check_id,
                counts=counts,
                fail_rate=counts[Verdict.FAIL] / n_decided if n_decided else None,
                abstentions=dict(sorted(reasons.items(), key=lambda item: item[0].value)),
            )
        )
    return tuple(rows)


def _attempt[T](clip: ClipManifest, failures: list[Failure], action: Callable[[], T]) -> T | None:
    try:
        return action()
    except (PhysverifyError, ValueError) as error:
        failures.append(Failure.from_error(clip.clip_id, error, perceived_variant().name))
        return None


def _run_dataset(
    ctx: DiagnosisContext, dataset: DatasetSource, clips: Sequence[ClipManifest], out: _Collected
) -> None:
    chain = ctx.chain(dataset)
    warm = warm_variants(chain, clips, (perceived_variant(),))
    out.failures += warm
    failed = {f.clip_id for f in warm}
    results: list[CheckResult] = []
    for clip in (c for c in clips if c.clip_id not in failed):
        results += _clip_results(ctx, chain, dataset, clip, out)
    out.verdicts += verdict_distributions(dataset.name, results)
    out.clips += [f"{dataset.name}/{c.clip_id}" for c in clips]
    manifest = read_manifest(ctx.settings.paths.raw_dir / dataset.name / MANIFEST_FILE)
    if manifest is not None:
        out.versions[dataset.name] = manifest.revision


def _clip_results(
    ctx: DiagnosisContext,
    chain: ChainRunner,
    dataset: DatasetSource,
    clip: ClipManifest,
    out: _Collected,
) -> list[CheckResult]:
    """Motion or fall row of one clip (into ``out``) and its decided check results."""
    variant = perceived_variant()
    outputs = _attempt(clip, out.failures, lambda: variant_outputs(chain, clip, variant))
    if outputs is None:
        return []
    if isinstance(dataset, MotionMaskSource):
        motion = _attempt(clip, out.failures, lambda: motion_row(ctx, dataset, clip, outputs))
        out.motion += [] if motion is None else [motion]
    else:
        fall = _attempt(clip, out.failures, lambda: fall_row(ctx, clip, outputs))
        out.falls += [] if fall is None else [fall]
    suite = ctx.suite
    if suite is None:
        return []
    level = InputLevel.PERCEIVED
    results = _attempt(
        clip, out.failures, lambda: decided(suite, variant_results(suite, clip, outputs, level))
    )
    return list(results or ())


def run_real(
    ctx: DiagnosisContext, datasets: Sequence[str], subset: str, clips: Iterable[str] = ()
) -> RealVideoReport:
    """Motion, fall and verdict rows of the perceived chain (V0) over the real datasets."""
    out = _Collected()
    for dataset, selected in select_clips(ctx, datasets, subset, clips):
        _run_dataset(ctx, dataset, selected, out)
    if ctx.suite is None:
        out.failures.append(Failure(clip_id=ALL_CLIPS, error=NO_SUITE))
    provisional = ctx.suite is None or ctx.suite.provisional
    estimated = [*ESTIMATED_FALL] if out.falls else []
    if any(row.gravity_dir_estimated for row in out.falls):
        estimated.append(ESTIMATED_GRAVITY_DIR)
    provenance = build_provenance(
        ctx.run,
        dataset=",".join(datasets),
        dataset_versions=out.versions,
        subset=subset,
        error_profile=ctx.deps.error_profile,
        clips=out.clips,
        estimated=estimated,
        provisional=(PROVISIONAL_VERDICTS,) if out.verdicts and provisional else (),
    )
    return RealVideoReport(
        provenance=provenance,
        motion=tuple(out.motion),
        falls=tuple(out.falls),
        verdicts=tuple(out.verdicts),
        thresholds_provisional=provisional,
        failures=tuple(out.failures),
    )
