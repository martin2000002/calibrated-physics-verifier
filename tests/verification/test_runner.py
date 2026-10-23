"""AC12: the runner with check doubles (9 rows per clip, aggregation, row_step, outputs)."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import check_scenarios as sc
import pytest

from physverify.checks._base import Check, CheckInputs
from physverify.checks._params import CommonParams
from physverify.checks._results import abstain, unscored
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.io import read_jsonl
from physverify.verification.inputs import ClipInputs
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import (
    CHECKS_FILE,
    TABLE_FILE,
    RowStatus,
    TableRow,
    VerificationReport,
    VerificationRunner,
    aggregate,
    write_report,
)

Behavior = Callable[[type[Check], CheckInputs], Sequence[CheckResult]]
REASONS = AbstainReason


def make_check(check_id: str, behavior: Behavior, requires: frozenset[str] = frozenset()) -> Any:
    def evaluate(self: Check, inputs: CheckInputs) -> Sequence[CheckResult]:
        return behavior(type(self), inputs)

    def init(self: Check, params: CommonParams | None = None) -> None:
        return None

    namespace = {
        "check_id": check_id,
        "requires": requires,
        "magnitude_unit": f"{check_id}_unit",
        "Params": CommonParams,
        "__init__": init,
        "evaluate": evaluate,
    }
    return type(f"Fake{check_id}", (), namespace)


def scored(*values: tuple[str, float, float]) -> Behavior:
    def run(check: type[Check], inputs: CheckInputs) -> Sequence[CheckResult]:
        return [unscored(check, inputs, (s,), m, sd, (0, 3), {}) for s, m, sd in values]

    return run


def abstaining(*reasons: AbstainReason) -> Behavior:
    def run(check: type[Check], inputs: CheckInputs) -> Sequence[CheckResult]:
        return [abstain(check, inputs, (f"o{i}",), r, None, {}) for i, r in enumerate(reasons)]

    return run


def no_event(check: type[Check], inputs: CheckInputs) -> Sequence[CheckResult]:
    return []


def boom(check: type[Check], inputs: CheckInputs) -> Sequence[CheckResult]:
    raise RuntimeError("boom")


NINE = (
    make_check("F1", scored(("a", 0.2, 0.01), ("b", 0.4, 0.02))),
    make_check("F2", no_event),
    make_check("F3", abstaining(REASONS.OCCLUDED, REASONS.TOO_FEW_FRAMES, REASONS.OCCLUDED)),
    make_check("F4", boom),
    make_check("F5", scored(("a", 0.1, 0.0)), requires=frozenset({"contacts"})),
    make_check("F6", scored(("a", 0.1, 0.0)), requires=frozenset({"scene"})),
    make_check("F7", scored(("a", 5.0, 0.0))),
    make_check("F8", scored(("a", 0.5, math.nan))),
    make_check("F9", scored(("a", 0.1, 0.0), ("b", 0.9, 0.0))),
)
THRESHOLDS = {
    "status": "provisional",
    "source": "test",
    "checks": {cid: {"tau": 1.0} for cid in ("F1", "F5", "F7", "F8")},
}


class ListSource:
    def __init__(self, items: Sequence[ClipInputs]) -> None:
        self._items = {item.clip.clip_id: item for item in items}

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            yield self._items[clip.clip_id]


def prepared(case: sc.Case) -> ClipInputs:
    return ClipInputs(case.clip, case.inputs(), None)


def run(items: Sequence[ClipInputs], checks: Sequence[Any] = NINE) -> VerificationReport:
    runner = VerificationRunner([c() for c in checks], ThresholdPolicy(), THRESHOLDS)
    return runner.run(ListSource(items), [item.clip for item in items])


def rows_of(report: VerificationReport, clip_id: str) -> dict[str, TableRow]:
    return {row.check_id: row for row in report.table if row.clip_id == clip_id}


def test_nine_rows_per_clip_whatever_the_checks_do() -> None:
    a, b = sc.parabola(clip_id="a"), sc.overhang(0.0, clip_id="b")
    report = run([prepared(a), prepared(b)])
    assert len(report.table) == 9 * 2
    rows = rows_of(report, "a")
    assert rows["F1"].status is RowStatus.SCORED
    assert rows["F2"].status is RowStatus.NO_EVENT
    assert rows["F2"].verdict is None and rows["F2"].magnitude is None
    assert rows["F3"].status is RowStatus.ABSTAINED
    assert rows["F4"].status is RowStatus.ERROR
    assert rows["F5"].status is RowStatus.SCORED
    assert rows["F6"].status is RowStatus.ABSTAINED
    assert rows["F6"].abstain_reason is REASONS.MISSING_INPUT
    assert [c.clip_id for c in report.clips] == ["a", "b"]


def test_aggregation_takes_the_worst_subject_and_its_sigma() -> None:
    rows = rows_of(run([prepared(sc.parabola(clip_id="a"))]), "a")
    assert (rows["F1"].magnitude, rows["F1"].sigma) == (0.4, 0.02)
    assert (rows["F1"].n_subjects, rows["F1"].verdict) == (2, Verdict.PASS)
    assert (rows["F3"].abstain_reason, rows["F3"].n_abstained) == (REASONS.OCCLUDED, 3)
    assert rows["F7"].verdict is Verdict.FAIL
    assert rows["F8"].verdict is Verdict.ABSTAIN
    assert rows["F8"].abstain_reason is REASONS.UNKNOWN_UNCERTAINTY
    assert rows["F9"].verdict is Verdict.UNSCORED
    assert rows["F1"].unit == "F1_unit"


def test_any_failing_subject_fails_the_row() -> None:
    check = NINE[0]()
    inputs = sc.parabola().inputs()
    results = [
        replace(unscored(type(check), inputs, ("a",), 0.1, 0.0, None, {}), verdict=Verdict.PASS),
        replace(unscored(type(check), inputs, ("b",), 3.0, 0.0, None, {}), verdict=Verdict.FAIL),
    ]
    assert aggregate("clip", check, results).verdict is Verdict.FAIL
    assert aggregate("clip", check, results[:1]).verdict is Verdict.PASS
    assert aggregate("clip", check, []).status is RowStatus.NO_EVENT


def test_the_raw_policy_keeps_rows_unscored() -> None:
    items = [prepared(sc.parabola(clip_id="a"))]
    runner = VerificationRunner([c() for c in NINE], RawPolicy(), THRESHOLDS)
    report = runner.run(ListSource(items), [items[0].clip])
    assert rows_of(report, "a")["F7"].verdict is Verdict.UNSCORED
    assert all(r.threshold is None for r in report.results)


def test_row_step_is_the_native_stride_of_the_table() -> None:
    case = sc.strided(sc.parabola(clip_id="s"), 2, quantize=False)
    report = run([prepared(case)])
    assert {row.row_step for row in report.table} == {2}


def test_a_failing_check_gives_an_error_row_and_the_others_run() -> None:
    report = run([prepared(sc.parabola(clip_id="a"))])
    errors = [r for r in report.results if r.check_id == "F4"]
    assert len(errors) == 1
    assert errors[0].verdict is Verdict.ABSTAIN
    assert errors[0].abstain_reason is REASONS.MISSING_INPUT
    assert errors[0].details["error"] == "RuntimeError: boom"
    assert rows_of(report, "a")["F7"].status is RowStatus.SCORED


def test_missing_required_inputs_abstain_naming_the_field() -> None:
    report = run([prepared(sc.parabola(clip_id="a"))])
    (missing,) = [r for r in report.results if r.check_id == "F6"]
    assert missing.details["missing"] == ["scene"]


def test_a_clip_without_inputs_abstains_on_every_check() -> None:
    clip = sc.parabola(clip_id="bad").clip
    item = ClipInputs(clip, None, "time_axis: not uniform", InputLevel.PERCEIVED)
    report = run([item])
    assert [r.status for r in report.table] == [RowStatus.ABSTAINED] * 9
    assert all(r.abstain_reason is REASONS.MISSING_INPUT for r in report.results)
    assert all(r.details["error"].startswith("time_axis") for r in report.results)
    assert {r.input_level for r in report.results} == {InputLevel.PERCEIVED}


def test_results_carry_the_threshold_origin() -> None:
    report = run([prepared(sc.parabola(clip_id="a"))])
    assert {r.provenance["thresholds"] for r in report.results} == {"provisional"}


def test_outputs_round_trip(tmp_path: Path) -> None:
    report = run([prepared(sc.parabola(clip_id="a")), prepared(sc.overhang(0.0, clip_id="b"))])
    write_report(tmp_path, report)
    assert read_jsonl(CheckResult, tmp_path / CHECKS_FILE) == list(report.results)
    first = json.loads((tmp_path / CHECKS_FILE).read_text(encoding="utf-8").splitlines()[0])
    assert CheckResult.from_dict(first["data"]) == report.results[0]
    with (tmp_path / TABLE_FILE).open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 18
    assert {row["status"] for row in table} <= {s.value for s in RowStatus}
    assert {row["row_step"] for row in table} == {"1"}


@pytest.mark.parametrize("status", list(RowStatus))
def test_row_status_values_are_the_declared_ones(status: RowStatus) -> None:
    assert status.value in {"scored", "abstained", "no_event", "error"}
