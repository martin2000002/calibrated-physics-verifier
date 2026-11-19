"""The comparison table's values on hand-built rows: pair accuracy, its difference, B1's false
alarms and detection at u_B1, and the shape of every interval."""

from __future__ import annotations

import re

from comparison_cases import KUBRIC, REAL, U_B1, config

from physverify.baselines.rules import CONFIRMED, NOT_EVALUABLE, REFUTED
from physverify.reporting.final_comparison import Line, p1_lines, part2_lines

CI = re.compile(r"^\[(-?\d\.\d{3}), (-?\d\.\d{3})\]$")


def by_id(lines: list[Line]) -> dict[str, Line]:
    return {line.id: line for line in lines}


def bounds(line: Line, value: float) -> tuple[float, float]:
    """The interval's bounds, checked to hold ``value`` (the point estimate)."""
    found = CI.match(line.ci)
    assert found, line.ci
    low, high = float(found[1]), float(found[2])
    assert low <= value <= high
    return low, high


def test_pair_accuracy_and_its_difference() -> None:
    lines = by_id(p1_lines(KUBRIC, config(), "kubric.json"))
    pa_b1, pa_judge, delta = lines["part1.pa_b1"], lines["part1.pa_judge"], lines["part1.delta"]
    assert (pa_b1.value, pa_judge.value, delta.value) == ("0.667", "0.333", "0.333")
    assert all(x.note.startswith("n 3") for x in (pa_b1, pa_judge, delta))
    assert "judge unread 0.167 (limit 0.5)" in delta.note
    for line in (pa_b1, pa_judge):
        low, high = bounds(line, float(line.value))
        assert low >= 0.0 and high <= 1.0
    low, high = bounds(delta, 1 / 3)
    assert low >= -0.5 and high <= 1.0
    assert delta.verdict == (CONFIRMED if low > 0 else REFUTED)


def test_a_judge_that_reads_too_few_clips_leaves_the_difference_unjudged() -> None:
    delta = by_id(p1_lines(KUBRIC, config(max_uninterpretable=0.1), ""))["part1.delta"]
    assert delta.value == "0.333" and delta.verdict == NOT_EVALUABLE


def test_false_alarms_and_detection_at_u_b1() -> None:
    lines = by_id(part2_lines(KUBRIC, REAL, config(), "kubric.json", "real.json"))
    fpr = lines["part2.fpr.real_set"]
    assert fpr.value == "0.333 (1/3)" and fpr.note == "Clopper-Pearson"
    assert bounds(fpr, 1 / 3)[0] >= 0.0
    detection = lines["part2.detection"]
    assert detection.value == "0.333" and detection.note == "n 6"
    low, high = bounds(detection, 1 / 3)
    assert low >= -1.0 and high <= 1.0
    assert lines["part2"].value == f"u_B1 {U_B1:.3f}"
