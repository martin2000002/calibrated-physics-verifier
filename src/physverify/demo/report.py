"""Per-video verifier report: what the demo shows next to the overlay.

The check rows are the verifier's own table rows (``CandidateScore.per_check``) plus, per check,
the abstain reasons, threshold and frame span of its results and whether the active rule decided
it. The clip block is the active rule's score S with its sigma and coverage. A rule that is not
calibrated is labelled as such everywhere: its S is not a calibrated verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from pydantic import Field

from physverify.core.atomic import write_text_atomic
from physverify.core.markdown import SIG3_FINITE, table
from physverify.core.models import FrozenModel
from physverify.finetune.records import CheckSummary, PromptRecord, finite_or_none
from physverify.finetune.scorers._base import AggregateScore, AggregateScorer
from physverify.schema.checks import CheckResult
from physverify.verification.decisive import decisive
from physverify.verification.runner import CALIBRATED, PROVISIONAL

REPORT_JSON: Final = "report.json"
REPORT_MD: Final = "report.md"
PROVISIONAL_BANNER: Final = "PROVISIONAL RULE: not a calibrated verdict"
NO_VERDICT: Final = "NO_EVENT"
CHECK_COLUMNS: Final = (
    "check",
    "verdict",
    "status",
    "magnitude",
    "sigma",
    "threshold",
    "abstain reasons",
    "frames",
    "decided by rule",
    "rule term",
)
TERM_MEANING: Final = {
    True: "p_c, estimated violation probability",
    False: "s_j = 1 / (1 + m/tau), provisional factor",
}


class RuleInfo(FrozenModel):
    """The rule behind S. ``note`` says why a provisional rule stands in for the calibrated one."""

    scorer: str
    calibrated: bool
    label: str
    note: str | None = None

    @property
    def status(self) -> str:
        return CALIBRATED if self.calibrated else PROVISIONAL


class CheckRow(FrozenModel):
    check_id: str
    status: str
    verdict: str
    magnitude: float | None
    sigma: float | None
    unit: str
    threshold: float | None
    abstain_reasons: list[str]
    frame_span: tuple[int, int] | None
    n_subjects: int
    rule_decided: bool
    rule_term: float | None

    def summary(self) -> CheckSummary:
        verdict = None if self.verdict == NO_VERDICT else self.verdict
        return CheckSummary(
            status=self.status,
            magnitude=self.magnitude,
            sigma=self.sigma,
            verdict=verdict,
            unit=self.unit,
        )


class ClipBlock(FrozenModel):
    score: float | None
    sigma: float | None
    coverage: float
    abstained: bool
    reason: str | None
    n_decided: int
    n_failed: int
    motion: float | None
    objects_ok: bool | None


class VideoReport(FrozenModel):
    prompt: PromptRecord
    model_tag: str
    candidate_id: str
    video_path: str
    results_path: str
    rule: RuleInfo
    clip: ClipBlock
    checks: list[CheckRow]
    provenance: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)


class CandidateRef(FrozenModel):
    """Where a report's candidate lives; ``motion`` and ``objects_ok`` are rule-independent."""

    model_tag: str
    candidate_id: str
    video_path: str
    results_path: str
    motion: float | None = None
    objects_ok: bool | None = None


def rule_info(scorer: AggregateScorer, note: str | None = None) -> RuleInfo:
    return RuleInfo(
        scorer=scorer.name, calibrated=scorer.calibrated, label=scorer.thresholds(), note=note
    )


def _worst(results: Sequence[CheckResult]) -> CheckResult | None:
    return decisive([r for r in results if r.magnitude is not None])


def check_row(
    check_id: str,
    summary: CheckSummary,
    results: Sequence[CheckResult],
    aggregate: AggregateScore,
) -> CheckRow:
    worst = _worst(results)
    reasons = sorted({r.abstain_reason.value for r in results if r.abstain_reason is not None})
    return CheckRow(
        check_id=check_id,
        status=summary.status,
        verdict=summary.verdict or NO_VERDICT,
        magnitude=summary.magnitude,
        sigma=summary.sigma,
        unit=summary.unit,
        threshold=None if worst is None else worst.threshold,
        abstain_reasons=reasons,
        frame_span=None if worst is None else worst.frame_span,
        n_subjects=len(results),
        rule_decided=check_id in aggregate.decided,
        rule_term=aggregate.per_check.get(check_id),
    )


def build_report(
    prompt: PromptRecord,
    candidate: CandidateRef,
    summaries: Mapping[str, CheckSummary],
    results: Sequence[CheckResult],
    aggregate: AggregateScore,
    rule: RuleInfo,
    provenance: Mapping[str, str] | None = None,
    timings: Mapping[str, float] | None = None,
) -> VideoReport:
    by_check: dict[str, list[CheckResult]] = {}
    for result in results:
        by_check.setdefault(result.check_id, []).append(result)
    return VideoReport(
        prompt=prompt,
        model_tag=candidate.model_tag,
        candidate_id=candidate.candidate_id,
        video_path=candidate.video_path,
        results_path=candidate.results_path,
        rule=rule,
        clip=ClipBlock(
            score=finite_or_none(aggregate.score),
            sigma=finite_or_none(aggregate.sigma),
            coverage=aggregate.coverage,
            abstained=aggregate.abstained or aggregate.score is None,
            reason=aggregate.reason,
            n_decided=aggregate.n_decided,
            n_failed=aggregate.n_failed,
            motion=candidate.motion,
            objects_ok=candidate.objects_ok,
        ),
        checks=[
            check_row(check_id, summaries[check_id], by_check.get(check_id, []), aggregate)
            for check_id in sorted(summaries)
        ],
        provenance=dict(provenance or {}),
        timings=dict(timings or {}),
    )


def candidate_of(report: VideoReport) -> CandidateRef:
    return CandidateRef(
        model_tag=report.model_tag,
        candidate_id=report.candidate_id,
        video_path=report.video_path,
        results_path=report.results_path,
        motion=report.clip.motion,
        objects_ok=report.clip.objects_ok,
    )


def summaries_of(report: VideoReport) -> dict[str, CheckSummary]:
    return {row.check_id: row.summary() for row in report.checks}


_num = SIG3_FINITE


def score_text(clip: ClipBlock) -> str:
    if clip.score is None:
        return f"S = abstained ({clip.reason or 'no score'})"
    sigma = "unknown" if clip.sigma is None else _num(clip.sigma)
    return f"S = {_num(clip.score)} +/- {sigma}"


def _span(span: tuple[int, int] | None) -> str:
    return "-" if span is None else f"{span[0]}-{span[1]}"


def _row_cells(row: CheckRow) -> list[str]:
    reasons = ", ".join(row.abstain_reasons) or "-"
    decided = "yes" if row.rule_decided else "no"
    return [
        row.check_id,
        row.verdict,
        row.status,
        f"{_num(row.magnitude)} {row.unit}".strip(),
        _num(row.sigma),
        _num(row.threshold),
        reasons,
        _span(row.frame_span),
        decided,
        _num(row.rule_term),
    ]


def _header(report: VideoReport) -> list[str]:
    lines = [f"# Verifier report: {report.candidate_id} ({report.model_tag})", ""]
    if not report.rule.calibrated:
        lines += [f"> **{PROVISIONAL_BANNER}.**", ""]
        if report.rule.note:
            lines += [f"> {report.rule.note}", ""]
    return lines


def _clip_lines(report: VideoReport) -> list[str]:
    clip = report.clip
    rule = report.rule
    return [
        (
            f"- Prompt ({report.prompt.prompt_id}, family {report.prompt.family}): "
            f"{report.prompt.text}"
        ),
        f"- Video: `{report.video_path}`",
        f"- Rule: {rule.scorer}, status **{rule.status}**, label `{rule.label}`",
        (
            f"- {score_text(clip)}; coverage {clip.coverage:.2f}; "
            f"{clip.n_decided} checks decided, {clip.n_failed} failed"
        ),
        (
            f"- Motion index {_num(clip.motion)}; expected objects found: "
            f"{'-' if clip.objects_ok is None else clip.objects_ok}"
        ),
        (
            "- Magnitudes from perceived inputs are estimates; sigma is their propagated error "
            "('-' when not yet measured)."
        ),
        "",
    ]


def _table(report: VideoReport) -> list[str]:
    return [
        "## Checks",
        "",
        *table(CHECK_COLUMNS, [_row_cells(row) for row in report.checks]),
        "",
        f"Rule term: {TERM_MEANING[report.rule.calibrated]}.",
        "",
    ]


def _provenance(report: VideoReport) -> list[str]:
    items = {**report.provenance, **{k: f"{v:.1f} s" for k, v in report.timings.items()}}
    if not items:
        return []
    return ["## Provenance", "", *(f"- {k}: {v}" for k, v in sorted(items.items())), ""]


def render_markdown(report: VideoReport) -> str:
    lines = [*_header(report), *_clip_lines(report), *_table(report), *_provenance(report)]
    return "\n".join(lines)


def write_report(directory: Path, report: VideoReport) -> None:
    write_text_atomic(directory / REPORT_JSON, report.model_dump_json(indent=2))
    write_text_atomic(directory / REPORT_MD, render_markdown(report))


def read_report(path: Path) -> VideoReport:
    file = path / REPORT_JSON if path.is_dir() else path
    return VideoReport.model_validate_json(file.read_text(encoding="utf-8"))
