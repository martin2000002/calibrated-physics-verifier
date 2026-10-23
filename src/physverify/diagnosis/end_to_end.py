"""Level 2 of ADR 0002: check magnitudes of the perceived chain against the oracle (ADR 0007 §5.3).

Two comparisons are always reported side by side: the perceived chain (V0) against V8, the all
ground-truth chain on the sampled frames, and V8 against the simulator truth on native frames
(``OracleSource``). The second one is the cost of frame sampling alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from physverify.core.errors import MissingInputError, PhysverifyError
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain_runner import ChainRunner, VariantOutputs, warm_variants
from physverify.diagnosis.chain_variants import Variant, oracle_variant, perceived_variant
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.measurement_cases import FloatArray, stage_output
from physverify.diagnosis.pairing import Paired, ResultPair, common_unit, merge, pair_results
from physverify.diagnosis.pairing import subject_map as match_subjects
from physverify.diagnosis.report import (
    CheckAgreementRow,
    ClipCheckRow,
    Comparison,
    EndToEndReport,
    Failure,
    FlipStats,
    build_provenance,
    dataset_versions,
    z_coverage,
)
from physverify.diagnosis.verification_adapter import CheckSuite
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage

PERCEIVED_VS_V8 = "perceived_vs_v8"
V8_VS_ORACLE = "v8_vs_oracle"
ORACLE_SOURCE = "oracle"
NO_POLICY = "no verdict policy or thresholds"
FLIPS_TAG = "flips"
DECIDED = frozenset({Verdict.PASS, Verdict.FAIL})


@dataclass(frozen=True)
class Decision:
    """Whether a verdict policy set PASS/FAIL, and whether its thresholds are provisional."""

    available: bool
    provisional: bool = False
    reason: str | None = None


# --- Agreement measures -----------------------------------------------------------------------


def _mean(values: FloatArray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def flip_stats(pairs: Sequence[ResultPair], decision: Decision) -> FlipStats:
    """4x4 ``confusion[reference][other]`` and the flip rate among pairs decided on both sides."""
    if not decision.available:
        return FlipStats(available=False, reason=decision.reason or NO_POLICY)
    confusion = {ref: dict.fromkeys(Verdict, 0) for ref in Verdict}
    for pair in pairs:
        confusion[pair.oracle.verdict][pair.other.verdict] += 1
    decided = [p for p in pairs if p.oracle.verdict in DECIDED and p.other.verdict in DECIDED]
    flips = sum(p.oracle.verdict is not p.other.verdict for p in decided)
    return FlipStats(
        available=True,
        provisional=decision.provisional,
        rate=flips / len(decided) if decided else None,
        n_decided=len(decided),
        n_flips=flips,
        confusion=confusion,
    )


def _abstain_rates(
    reference: Sequence[CheckResult], other: Sequence[CheckResult]
) -> tuple[dict[AbstainReason, float], dict[AbstainReason, float]]:
    """Share of each abstention reason per side, over the reasons seen on either side."""
    reasons = {r.abstain_reason for r in (*reference, *other) if r.abstain_reason is not None}

    def rates(side: Sequence[CheckResult]) -> dict[AbstainReason, float]:
        total = len(side)
        return {
            reason: sum(r.abstain_reason is reason for r in side) / total if total else math.nan
            for reason in sorted(reasons, key=lambda item: item.value)
        }

    return rates(reference), rates(other)


def _check_row(
    check_id: str, part: Paired, z_levels: Sequence[float], decision: Decision
) -> CheckAgreementRow:
    reference = [p.oracle for p in part.pairs] + list(part.missing)
    other = [p.other for p in part.pairs] + list(part.spurious)
    unit = common_unit(
        check_id, {"reference": [r.unit for r in reference], "other": [r.unit for r in other]}
    )
    measured = [p for p in part.pairs if p.delta is not None]
    delta = np.array([p.delta for p in measured], dtype=np.float64)
    truth = np.array([p.oracle.magnitude for p in measured], dtype=np.float64)
    sigma = np.array([p.other.sigma for p in measured], dtype=np.float64)
    relative = np.abs(delta[truth != 0]) / np.abs(truth[truth != 0])
    has_sigma = np.isfinite(sigma) & (sigma > 0)
    z = delta[has_sigma] / sigma[has_sigma]
    abstain_reference, abstain_other = _abstain_rates(reference, other)
    return CheckAgreementRow(
        check_id=check_id,
        unit=unit,
        n_pairs=len(part.pairs),
        n_missing=len(part.missing),
        n_spurious=len(part.spurious),
        bias=_mean(delta),
        mae=_mean(np.abs(delta)),
        rms=math.sqrt(_mean(delta**2)),
        rel_error=_mean(relative),
        n_z=int(z.size),
        coverage=z_coverage(z, z_levels),
        abstain_reference=abstain_reference,
        abstain_other=abstain_other,
        flips=flip_stats(part.pairs, decision),
    )


def agreement(
    paired: Paired, z_levels: Sequence[float], decision: Decision
) -> tuple[CheckAgreementRow, ...]:
    """Per ``check_id``: counts, delta = m_other - m_reference statistics, z coverage, flips."""
    return tuple(
        _check_row(check_id, paired.of_check(check_id), z_levels, decision)
        for check_id in paired.check_ids
    )


def _clip_row(
    comparison: str,
    present: CheckResult,
    ref: CheckResult | None,
    other: CheckResult | None,
    subject: tuple[str, ...],
) -> ClipCheckRow:
    m_ref = None if ref is None else ref.magnitude
    m_other = None if other is None else other.magnitude
    return ClipCheckRow(
        comparison=comparison,
        clip_id=present.clip_id,
        check_id=present.check_id,
        subject=subject,
        m_reference=m_ref,
        m_other=m_other,
        delta=None if m_ref is None or m_other is None else m_other - m_ref,
        verdict_reference=None if ref is None else ref.verdict,
        verdict_other=None if other is None else other.verdict,
    )


def clip_rows(comparison: str, paired: Paired) -> list[ClipCheckRow]:
    """One row per pair, missing reference result and spurious other result."""
    rows = [_clip_row(comparison, p.oracle, p.oracle, p.other, p.subject) for p in paired.pairs]
    rows += [_clip_row(comparison, r, r, None, r.subject) for r in paired.missing]
    rows += [_clip_row(comparison, r, None, r, r.subject) for r in paired.spurious]
    return rows


# --- Running variants through the check suite --------------------------------------------------


def require_suite(ctx: DiagnosisContext) -> CheckSuite:
    if ctx.suite is None:
        raise MissingInputError(
            "end-to-end and attribution diagnosis need A7's check suite (verification/)"
        )
    return ctx.suite


def report_clips(ctx: DiagnosisContext, clips: Sequence[ClipManifest]) -> tuple[ClipManifest, ...]:
    """The clips of ``diagnosis.report_split``, sorted by id."""
    split = ctx.diag.report_split
    selected = tuple(sorted((c for c in clips if c.split is split), key=lambda c: c.clip_id))
    if not selected:
        raise MissingInputError(
            f"none of the {len(clips)} selected clips is in split {split.value}"
        )
    return selected


def variant_outputs(chain: ChainRunner, clip: ClipManifest, variant: Variant) -> VariantOutputs:
    outputs = chain.outputs(clip, variant)
    if outputs is None:
        raise MissingInputError(f"{clip.clip_id}: variant {variant.name} has a failed stage")
    return outputs


def variant_results(
    suite: CheckSuite, clip: ClipManifest, outputs: VariantOutputs, level: InputLevel
) -> tuple[CheckResult, ...]:
    """Check results of one variant, built through the suite's single inputs builder."""
    inputs = suite.inputs(clip, outputs.outputs, outputs.plan.names, level)
    return tuple(suite.evaluate(inputs))


def decided(suite: CheckSuite, results: Sequence[CheckResult]) -> tuple[CheckResult, ...]:
    """Results with the policy's verdicts, or unchanged when there is no policy."""
    verdicts = suite.decide(results)
    return tuple(results if verdicts is None else verdicts)


def track_of(outputs: VariantOutputs) -> MaskTrack:
    track = stage_output(outputs.outputs, Stage.TRACK, MaskTrack)
    if track is None:
        raise MissingInputError(f"variant {outputs.variant.name} has no {Stage.TRACK.value} output")
    return track


def _decision(suite: CheckSuite) -> Decision:
    """``decide`` returns None exactly when no verdict policy or thresholds are loaded."""
    if suite.decide(()) is None:
        return Decision(available=False, reason=NO_POLICY)
    return Decision(available=True, provisional=suite.provisional)


# --- Level 2 ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipEndToEnd:
    perceived: Paired
    sampling: Paired | None


@dataclass(frozen=True)
class _Run:
    suite: CheckSuite
    chain: ChainRunner
    dataset: DatasetSource
    min_span_iou: float


def _clip_end_to_end(run: _Run, clip: ClipManifest) -> ClipEndToEnd:
    suite = run.suite
    v8 = variant_outputs(run.chain, clip, oracle_variant())
    v0 = variant_outputs(run.chain, clip, perceived_variant())
    oracle = decided(suite, variant_results(suite, clip, v8, InputLevel.ORACLE))
    perceived = decided(suite, variant_results(suite, clip, v0, InputLevel.PERCEIVED))
    subjects = match_subjects(track_of(v0), track_of(v8))
    perceived_pairs = pair_results(oracle, perceived, subjects, run.min_span_iou)
    truth_inputs = suite.truth_inputs(run.dataset, clip)
    if truth_inputs is None:
        return ClipEndToEnd(perceived_pairs, None)
    truth = decided(suite, tuple(suite.evaluate(truth_inputs)))
    return ClipEndToEnd(perceived_pairs, pair_results(truth, oracle, None, run.min_span_iou))


def run_end_to_end(ctx: DiagnosisContext, clips: Sequence[ClipManifest]) -> EndToEndReport:
    """Level 2 over the ``report_split`` clips: perceived vs V8 and V8 vs ``OracleSource``."""
    suite = require_suite(ctx)
    selected = report_clips(ctx, clips)
    dataset = ctx.dataset()
    run = _Run(suite, ctx.chain(dataset), dataset, ctx.diag.end_to_end.min_span_iou)
    failures = list(warm_variants(run.chain, selected, (oracle_variant(), perceived_variant())))
    failed = {f.clip_id for f in failures}
    perceived: list[Paired] = []
    sampling: list[Paired] = []
    evaluated: list[ClipManifest] = []
    for clip in (c for c in selected if c.clip_id not in failed):
        try:
            outcome = _clip_end_to_end(run, clip)
        except (PhysverifyError, ValueError) as error:
            failures.append(Failure.from_error(clip.clip_id, error))
            continue
        evaluated.append(clip)
        perceived.append(outcome.perceived)
        if outcome.sampling is None:
            failures.append(
                Failure(clip_id=clip.clip_id, error="no OracleSource inputs", variant=ORACLE_SOURCE)
            )
        else:
            sampling.append(outcome.sampling)
    return _report(ctx, run, selected, evaluated, perceived, sampling, failures)


def _report(
    ctx: DiagnosisContext,
    run: _Run,
    selected: Sequence[ClipManifest],
    evaluated: Sequence[ClipManifest],
    perceived: Sequence[Paired],
    sampling: Sequence[Paired],
    failures: Sequence[Failure],
) -> EndToEndReport:
    decision = _decision(run.suite)
    levels = ctx.diag.end_to_end.z_levels
    v0, v8 = perceived_variant().name, oracle_variant().name
    perceived_all, sampling_all = merge(perceived), merge(sampling)
    dataset = run.dataset
    provenance = build_provenance(
        ctx.run,
        dataset=dataset.name,
        dataset_versions=dataset_versions(dataset, evaluated),
        subset=ctx.diag.subset,
        error_profile=ctx.deps.error_profile,
        clips=(c.clip_id for c in selected),
        provisional=(FLIPS_TAG,) if decision.available and decision.provisional else (),
    )
    return EndToEndReport(
        provenance=provenance,
        report_split=ctx.diag.report_split,
        comparisons=(
            Comparison(
                name=PERCEIVED_VS_V8,
                reference=v8,
                other=v0,
                checks=agreement(perceived_all, levels, decision),
            ),
            Comparison(
                name=V8_VS_ORACLE,
                reference=ORACLE_SOURCE,
                other=v8,
                checks=agreement(sampling_all, levels, decision),
            ),
        ),
        clips=(
            *(row for part in perceived for row in clip_rows(PERCEIVED_VS_V8, part)),
            *(row for part in sampling for row in clip_rows(V8_VS_ORACLE, part)),
        ),
        failures=tuple(failures),
    )
