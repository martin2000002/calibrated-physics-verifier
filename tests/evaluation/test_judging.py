from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from physverify.app import plugins
from physverify.app.services import external_judge
from physverify.core.errors import ConfigError, UsageError
from physverify.datasets._videophy_tables import VideophyRow
from physverify.finetune.records import CandidateRecord, CandidateScore
from physverify.generator_comparison.gate import gate_rows, gate_task, task_usability, usability
from physverify.generator_comparison.judges._base import JudgeInput, JudgeScore
from physverify.generator_comparison.judges.verifier import VerifierJudge
from physverify.generator_comparison.judges.videophy2_autoeval import Videophy2AutoevalJudge
from physverify.generator_comparison.judging import (
    candidate_items,
    gate_items,
    gate_problem,
    gate_reference,
    gate_report,
    judge_output,
    reusable,
)
from physverify.generator_comparison.settings import (
    AutoevalSettings,
    GateSettings,
    H3Settings,
    UsabilitySettings,
)
from physverify.schema.clip import Split

SCALE = (1.0, 5.0)
GATE = GateSettings(n=3, min_exact=0.5, max_mae=0.5, max_parse_fail=1)


def row(clip_id: str, reference: float = 3.0) -> VideophyRow:
    return VideophyRow(
        part="v2_test",
        version="v2",
        split=Split.EVAL,
        clip_id=clip_id,
        video_url=f"https://x/{clip_id}.mp4",
        caption=f"caption {clip_id}",
        generator="g",
        human_raw=4.0,
        human_cut=4.0,
        reference_raw=reference,
        reference_cut=4.0,
        reference_name="pc_pred",
        violated_rules="",
    )


def scores(values: Mapping[str, float | None], stamp: str = "s") -> dict[str, JudgeScore]:
    return {
        k: JudgeScore(v, SCALE, stamp, None if v else "parse_failure") for k, v in values.items()
    }


class FakeJudge:
    name: ClassVar[str] = "fake"

    def __init__(self, values: Mapping[str, float | None]) -> None:
        self.values = values

    def unavailable(self) -> str | None:
        return None

    def stamp(self, task: str) -> str:
        return f"fake:{task}"

    def judge(self, items: Sequence[JudgeInput], task: str) -> Mapping[str, JudgeScore]:
        return scores({i.video_id: self.values.get(i.video_id) for i in items}, self.stamp(task))


def test_the_judge_registry_finds_both_judges() -> None:
    registry = plugins.build("judges")
    assert set(registry) == {"verifier", "videophy2_autoeval"}


class RegistryContext:
    def registry(self, kind: str) -> Any:
        return plugins.build(kind)


def test_the_external_judge_is_selected_by_name(tmp_path: Path) -> None:
    cfg = H3Settings(autoeval=AutoevalSettings(hf_revision="0" * 12, code_commit="0" * 12))
    context: Any = RegistryContext()
    assert isinstance(external_judge(context, cfg, tmp_path), Videophy2AutoevalJudge)
    for name in ("verifier", "missing"):
        with pytest.raises(ConfigError, match=name):
            external_judge(context, cfg.model_copy(update={"external_judge": name}), tmp_path)


def test_gate_rows_need_a_reference_and_a_video() -> None:
    rows = [row("c"), row("a"), row("b", math.nan), row("d")]
    kept = gate_rows(rows, lambda r: r.clip_id != "d", n=5)
    assert [r.clip_id for r in kept] == ["a", "c"]
    assert [r.clip_id for r in gate_rows(rows, lambda r: True, n=1)] == ["a"]


def test_gate_task_counts_missing_values_as_mismatches() -> None:
    reference = {"a": 3.0, "b": 4.0, "c": 2.0}
    task = gate_task("pc", "s", scores({"a": 3.0, "b": 4.0, "c": None}), reference, GATE)
    assert (task.n, task.exact, task.parse_failures, task.mae) == (3, 2, 1, 0.0)
    assert task.passed
    worse = gate_task("pc", "s", scores({"a": 1.0, "b": None, "c": None}), reference, GATE)
    assert not worse.passed
    assert not gate_task("pc", "s", {}, {}, GATE).passed


def test_usability_needs_parsed_and_varied_values() -> None:
    cfg = UsabilitySettings(max_parse_fail_share=0.25, max_modal_share=0.9)
    assert task_usability([3.0, 4.0, 2.0, None], cfg).usable
    assert not task_usability([3.0, 3.0, 3.0, 3.0], cfg).usable
    assert not task_usability([3.0, None, None, 4.0], cfg).usable
    assert not task_usability([], cfg).usable
    both = usability({"pc": [3.0, 4.0], "sa": [2.0, 2.0]}, cfg)
    assert both.tasks["pc"].usable and not both.usable


def test_candidate_items_caption_with_the_prompt(campaign: Any) -> None:
    prompts = {p.prompt_id: p for p in campaign.write_prompts()}
    record = campaign.write_candidate("base", prompts["e0"], 0)
    other: CandidateRecord = campaign.write_candidate("base", prompts["d0"], 0)
    found = candidate_items([record, other], {"e0": prompts["e0"]})
    assert found == [JudgeInput("e0__k00", Path(record.video_path), prompts["e0"].text)]


def test_judge_output_records_where_and_the_freeze() -> None:
    items = [JudgeInput("a", Path("a.mp4"), "c"), JudgeInput("b", Path("b.mp4"), "c")]
    output = judge_output(FakeJudge({"a": 4.0}), items, ("eval", "base", "pc"), "sha")
    assert (output.split, output.model_tag, output.task, output.stamp) == (
        "eval",
        "base",
        "pc",
        "fake:pc",
    )
    assert output.rows["a"].value == 4.0 and output.rows["b"].value is None
    assert reusable(output, "fake:pc", "sha", {"a", "b"})
    assert not reusable(output, "fake:pc", "other", {"a", "b"})
    assert not reusable(output, "fake:sa", "sha", {"a", "b"})
    assert not reusable(output, "fake:pc", "sha", {"a"})
    assert not reusable(None, "fake:pc", "sha", set())
    with pytest.raises(UsageError):
        judge_output(FakeJudge({}), [], ("eval", "base", "pc"), "sha")


def test_gate_report_and_its_use() -> None:
    rows = [row("a"), row("b"), row("c")]
    table = {
        "https://x/a.mp4": {"pc_pred": "3"},
        "https://x/b.mp4": {"pc_pred": "4"},
        "https://x/c.mp4": {"pc_pred": "n/a"},
    }
    reference = gate_reference(rows, table, "pc_pred")
    assert reference == {"a": 3.0, "b": 4.0}
    items = gate_items(rows, {r.clip_id: Path(f"{r.clip_id}.mp4") for r in rows})
    assert [i.caption for i in items] == ["caption a", "caption b", "caption c"]
    judged = {"pc": scores({"a": 3.0, "b": 4.0}, "fake:pc")}
    report = gate_report(judged, {"pc": reference}, {"pc": "fake:pc"}, ("ref", "frz"), GATE)
    assert report.passed and report.clip_ids == ["a", "b"]
    assert (report.reference_sha256, report.freeze_sha256) == ("ref", "frz")
    assert gate_problem(report, {"pc": "fake:pc"}) is None
    assert "stamp" in (gate_problem(report, {"pc": "other"}) or "")
    assert "stamp" in (gate_problem(report, {"sa": "fake:sa"}) or "")
    assert "not run" in (gate_problem(None, {}) or "")
    failed = report.model_copy(update={"passed": False})
    assert "failed" in (gate_problem(failed, {"pc": "fake:pc"}) or "")


def test_the_verifier_judge_reads_stored_scores(make_score: Any) -> None:
    stored: dict[str, CandidateScore] = {
        "a": make_score("a", 0.8),
        "b": make_score("b", None, reason="no_checks"),
    }
    judge = VerifierJudge(stored, "provisional")
    items = [JudgeInput(v, Path(f"{v}.mp4"), "c") for v in ("a", "b", "c")]
    found = judge.judge(items, "verifier")
    assert found["a"].value == 0.8 and found["a"].scale == (0.0, 1.0)
    assert (found["b"].value, found["b"].reason) == (None, "no_checks")
    assert (found["c"].value, found["c"].reason) == (None, "missing_score")
    assert VerifierJudge({}, "s").unavailable() is not None
