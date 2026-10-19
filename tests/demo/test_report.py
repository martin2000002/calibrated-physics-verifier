from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from physverify.config.models import Settings
from physverify.datasets.local_video import QueryEntry
from physverify.demo.report import (
    NO_VERDICT,
    PROVISIONAL_BANNER,
    REPORT_JSON,
    REPORT_MD,
    CandidateRef,
    RuleInfo,
    VideoReport,
    build_report,
    check_row,
    read_report,
    render_markdown,
    rule_info,
    score_text,
    summaries_of,
    write_report,
)
from physverify.finetune.records import CheckSummary, PromptRecord
from physverify.finetune.scorers._base import AggregateScore
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.schema.checks import CheckResult, Verdict

CID = "p__k00"
Results = Callable[[str], list[CheckResult]]


def _summaries() -> dict[str, CheckSummary]:
    def summary(status: str, verdict: str | None, magnitude: float | None) -> CheckSummary:
        return CheckSummary(
            status=status, magnitude=magnitude, sigma=None, verdict=verdict, unit=""
        )

    return {
        "R1": summary("scored", "PASS", 0.2),
        "R2": summary("scored", "FAIL", 3.0),
        "R3a": summary("abstained", "ABSTAIN", None),
        "R4": summary("no_event", None, None),
    }


def _prompt() -> PromptRecord:
    return PromptRecord(
        prompt_id="p",
        text="a ball falls",
        queries=[QueryEntry(phrase="ball", expected_count=1)],
        family="fall",
        template_id="fixed",
        target_checks=["R2"],
        split="eval",
        tags=["adhoc"],
    )


def make_report(
    settings: Settings, clip_results: Results, rule: RuleInfo | None = None
) -> VideoReport:
    scorer = ProvisionalScorer(settings)
    results = clip_results(CID)
    candidate = CandidateRef(
        model_tag="base", candidate_id=CID, video_path="v.mp4", results_path="r.jsonl"
    )
    return build_report(
        _prompt(),
        candidate,
        _summaries(),
        results,
        scorer.score(results),
        rule or rule_info(scorer),
        {"commit": "abc"},
        {"score_s": 1.5},
    )


def test_rows_join_the_table_the_results_and_the_rule(
    settings: Settings, clip_results: Results
) -> None:
    report = make_report(settings, clip_results)
    rows = {row.check_id: row for row in report.checks}
    assert list(rows) == ["R1", "R2", "R3a", "R4"]
    assert rows["R2"].verdict == "FAIL" and rows["R2"].frame_span == (2, 4)
    assert rows["R2"].threshold == 2.0 and rows["R2"].rule_decided
    assert rows["R3a"].abstain_reasons == ["MISSING_INPUT"] and not rows["R3a"].rule_decided
    assert rows["R4"].verdict == NO_VERDICT and rows["R4"].n_subjects == 0
    assert rows["R1"].rule_term == pytest.approx(1 / 1.2)
    assert report.clip.score == pytest.approx(1 / (1.2 * 2.5) ** 0.5)
    assert report.clip.coverage == pytest.approx(2 / 3)
    assert summaries_of(report) == _summaries()


def test_small_sigmas_keep_three_significant_figures(
    settings: Settings, clip_results: Results
) -> None:
    report = make_report(settings, clip_results)
    clip = report.clip.model_copy(update={"score": 0.61234, "sigma": 0.0043})
    assert score_text(clip) == "S = 0.612 +/- 0.0043"
    tiny = report.model_copy(update={"clip": clip})
    assert "S = 0.612 +/- 0.0043" in render_markdown(tiny)


def test_markdown_shows_rule_score_and_every_verdict(
    settings: Settings, clip_results: Results
) -> None:
    report = make_report(settings, clip_results)
    text = render_markdown(report)
    assert PROVISIONAL_BANNER in text
    assert f"label `{report.rule.label}`" in text and "status **provisional**" in text
    assert f"S = {report.clip.score:.3g} +/- unknown" in text
    assert "coverage 0.67" in text
    assert "| R2 | FAIL |" in text and "| 2-4 |" in text
    assert "| R3a | ABSTAIN | abstained |" in text and "MISSING_INPUT" in text
    assert "| R4 | NO_EVENT | no_event |" in text
    assert "- commit: abc" in text and "- score_s: 1.5 s" in text


def test_a_calibrated_rule_has_no_banner(settings: Settings, clip_results: Results) -> None:
    rule = RuleInfo(scorer="calibrated", calibrated=True, label="cal-1")
    text = render_markdown(make_report(settings, clip_results, rule))
    assert PROVISIONAL_BANNER not in text
    assert "status **calibrated**" in text


def test_reports_round_trip(tmp_path: Path, settings: Settings, clip_results: Results) -> None:
    report = make_report(settings, clip_results)
    write_report(tmp_path / "r", report)
    assert (tmp_path / "r" / REPORT_MD).is_file()
    assert read_report(tmp_path / "r") == report
    assert read_report(tmp_path / "r" / REPORT_JSON) == report


def test_a_nan_subject_listed_first_does_not_pick_the_row_span(
    make_result: Callable[..., CheckResult],
) -> None:
    results = [
        make_result("R1", Verdict.PASS, float("nan"), 1.0, frame_span=(0, 1)),
        make_result("R1", Verdict.PASS, 0.2, 1.0, frame_span=(5, 9)),
    ]
    summary = CheckSummary(status="scored", magnitude=0.2, sigma=None, verdict="PASS", unit="")
    aggregate = AggregateScore(None, None, True, None, 0, 0, 0.0)
    assert check_row("R1", summary, results, aggregate).frame_span == (5, 9)
