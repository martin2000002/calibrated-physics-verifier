from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from physverify.core.video import Mp4VideoSource
from physverify.demo.captions import arm_name, badge_text, bar_text, shown_rows
from physverify.demo.overlay import (
    AMBER_COLOR,
    BAR_COLOR,
    DIVIDER_COLOR,
    FAIL_COLOR,
    FAIL_DIM_COLOR,
    FONT,
    NO_EVENT_COLOR,
    PASS_COLOR,
    UNDECIDED_COLOR,
    badge_color,
    fit_lines,
    overlay_frames,
    probability_color,
    render_overlay,
    side_by_side,
    truncate,
)
from physverify.demo.report import NO_VERDICT, CheckRow, RuleInfo, VideoReport
from physverify.demo.settings import OverlayStyle

STYLE = OverlayStyle()
OPAQUE = OverlayStyle(badge_alpha=1.0)
BUDGET = 10**9
HEIGHT = 480
WIDTH = 832
CALIBRATED = RuleInfo(scorer="graded", calibrated=True, label="calibrated:" + "x" * 120)


def _row(
    check_id: str,
    verdict: str,
    span: tuple[int, int] | None = None,
    *,
    decided: bool = True,
    term: float | None = None,
    reasons: tuple[str, ...] = (),
    measured: tuple[float, float | None] | None = None,
) -> CheckRow:
    return CheckRow(
        check_id=check_id,
        status="scored",
        verdict=verdict,
        magnitude=None if measured is None else measured[0],
        sigma=None if measured is None else measured[1],
        unit="",
        threshold=None,
        abstain_reasons=list(reasons),
        frame_span=span,
        n_subjects=1,
        rule_decided=decided,
        rule_term=term,
    )


def _report(video: Path, model_tag: str = "base") -> VideoReport:
    payload = {
        "prompt": {
            "prompt_id": "p",
            "text": "t",
            "queries": ["ball"],
            "family": "fall",
            "template_id": "fixed",
            "target_checks": ["R2"],
            "split": "eval",
        },
        "model_tag": model_tag,
        "candidate_id": "p__k00",
        "video_path": str(video),
        "results_path": "r.jsonl",
        "rule": {"scorer": "provisional", "calibrated": False, "label": "thr-1"},
        "clip": {
            "score": 0.5,
            "sigma": None,
            "coverage": 1.0,
            "abstained": False,
            "reason": None,
            "n_decided": 2,
            "n_failed": 1,
            "motion": None,
            "objects_ok": None,
        },
        "checks": [_row("R2", "FAIL", (2, 4)).model_dump(), _row("R1", "PASS").model_dump()],
    }
    return VideoReport.model_validate(payload)


def _with(report: VideoReport, *rows: CheckRow, rule: RuleInfo | None = None) -> VideoReport:
    return report.model_copy(update={"checks": list(rows), "rule": rule or report.rule})


def _badge_pixel(frames: np.ndarray, slot: int = 0) -> tuple[int, ...]:
    """A pixel at the right end of a badge, clear of its text, on HEIGHT x WIDTH frames with a
    one-line bar."""
    top = STYLE.bar_height + STYLE.margin + slot * (STYLE.badge_height + STYLE.margin // 2)
    y, x = top + 2, STYLE.margin + STYLE.badge_width - 2
    return tuple(int(v) for v in frames[y, x])


def _blank(width: int = WIDTH, shade: int = 0) -> np.ndarray:
    return np.full((1, HEIGHT, width, 3), shade, dtype=np.uint8)


def test_badge_colors() -> None:
    fail = _row("R2", "FAIL", (2, 4))
    assert badge_color(fail, 3) == FAIL_COLOR
    assert badge_color(fail, 5) == FAIL_DIM_COLOR
    assert badge_color(_row("R2", "FAIL"), 7) == FAIL_COLOR
    assert badge_color(_row("R1", "PASS"), 0) == PASS_COLOR
    assert badge_color(_row("R3a", "ABSTAIN"), 0) == UNDECIDED_COLOR
    assert badge_color(_row("R4", NO_VERDICT), 0) == NO_EVENT_COLOR


def test_a_calibrated_decision_is_coloured_by_its_violation_probability() -> None:
    assert probability_color(0.0) == PASS_COLOR
    assert probability_color(0.5) == AMBER_COLOR
    assert probability_color(1.0) == FAIL_COLOR
    assert probability_color(7.0) == FAIL_COLOR
    graded = _row("R4_size", "ABSTAIN", term=0.25)
    assert badge_color(graded, 0, calibrated=True) == probability_color(0.25)
    assert badge_color(graded, 0) == UNDECIDED_COLOR


def test_a_check_that_fed_s_never_reads_abstain() -> None:
    reason = ("INTERVAL_CROSSES_THRESHOLD",)
    graded = _row("R4_size", "ABSTAIN", term=0.991, reasons=reason, measured=(0.25, 0.5))
    assert badge_text(graded, calibrated=True) == "R4_size P(violation) 0.99"
    idle = _row("R1", "ABSTAIN", decided=False, reasons=reason)
    assert badge_text(idle, calibrated=True) == "R1 ABSTAIN (too close to call)"
    assert badge_text(_row("R2", "FAIL", measured=(3.0, 0.5))) == "R2 FAIL m = 3 +/- 0.5"
    assert badge_text(_row("R2", "PASS")) == "R2 PASS m = -"
    assert badge_text(_row("R2", "PASS", measured=(3.0, None))) == "R2 PASS m = 3 +/- unknown"
    assert badge_text(_row("R4_jump", "FAIL", decided=False)).endswith(", not in S")
    assert badge_text(_row("R5a", NO_VERDICT, decided=False)) == "R5a no event"
    occluded = _row("R3a", "ABSTAIN", decided=False, reasons=("OCCLUDED",))
    assert badge_text(occluded) == "R3a ABSTAIN (occluded)"


def test_only_targeted_decided_or_failing_checks_get_a_badge(
    write_video: Callable[..., Path],
) -> None:
    rows = (
        _row("R1", "PASS", decided=False),
        _row("R2", NO_VERDICT, decided=False),
        _row("R3a", "FAIL", decided=False),
        _row("R4_size", "ABSTAIN", term=0.4),
        _row("R5a", NO_VERDICT, decided=False),
    )
    report = _with(_report(write_video()), *rows)
    assert [r.check_id for r in shown_rows(report)] == ["R2", "R3a", "R4_size"]
    assert len(shown_rows(report, all_checks=True)) == len(rows)


def test_overlay_marks_the_failure_only_inside_its_span(write_video: Callable[..., Path]) -> None:
    video = write_video(height=HEIGHT, width=WIDTH)
    report = _report(video)
    frames = np.zeros((6, HEIGHT, WIDTH, 3), dtype=np.uint8)
    out = overlay_frames(frames, report, OPAQUE)
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert not frames.any()
    assert _badge_pixel(out[3]) == FAIL_COLOR
    assert _badge_pixel(out[5]) == FAIL_DIM_COLOR
    assert _badge_pixel(out[0], slot=1) == PASS_COLOR


def test_badges_let_the_scene_show_through(write_video: Callable[..., Path]) -> None:
    report = _report(write_video())
    out = overlay_frames(_blank(shade=200), report, STYLE)
    blended = [round(STYLE.badge_alpha * c + (1 - STYLE.badge_alpha) * 200) for c in PASS_COLOR]
    assert _badge_pixel(out[0], slot=1) == pytest.approx(tuple(blended), abs=1)


def test_the_badge_panel_widens_to_fit_its_longest_text(write_video: Callable[..., Path]) -> None:
    row = _row("R4_missing", "ABSTAIN", reasons=("CONTACT_GEOMETRY_UNKNOWN",))
    report = _with(_report(write_video()), row)
    out = overlay_frames(_blank(), report, OPAQUE)
    top = STYLE.bar_height + STYLE.margin
    lit = np.flatnonzero((out[0, top + 2] == UNDECIDED_COLOR).all(axis=1))
    assert lit.max() > STYLE.margin + STYLE.badge_width + 50


def test_the_bar_names_the_arm_and_the_rule_not_their_ids(
    write_video: Callable[..., Path],
) -> None:
    video = write_video()
    assert arm_name("base") == "base"
    assert arm_name("lora-0123456789ab") == "fine-tuned"
    assert arm_name("video-0123456789ab") == "given video"
    assert arm_name("other") == "other"
    provisional = bar_text(_report(video, "lora-0123456789ab"))
    assert "fine-tuned" in provisional and "PROVISIONAL" in provisional
    assert "lora-" not in provisional and "thr-1" not in provisional
    calibrated = bar_text(_with(_report(video), rule=CALIBRATED))
    assert "calibrated rule (graded)" in calibrated and "PROVISIONAL" not in calibrated
    abstained = _report(video).clip.model_copy(update={"score": None, "reason": "no_graded_check"})
    assert "(no graded check)" in bar_text(_report(video).model_copy(update={"clip": abstained}))


def _width(text: str, scale: float) -> int:
    (width, _), _ = cv2.getTextSize(text, FONT, scale, max(1, round(scale)))
    return int(width)


def test_the_bar_wraps_then_shrinks_and_never_overflows() -> None:
    segments = ["fine-tuned", "S = 0.584 +/- 0.0847", "coverage 1.00", "calibrated rule (graded)"]
    lines, scale = fit_lines(segments, 2000, 0.5, 0.35)
    assert (len(lines), scale) == (1, 0.5)
    lines, scale = fit_lines(segments, 260, 0.5, 0.35)
    assert len(lines) == 2 and 0.35 <= scale <= 0.5
    assert all(_width(line, scale) <= 260 for line in lines)
    lines, scale = fit_lines(["x" * 400, "y" * 400, "z" * 400], 200, 0.5, 0.35)
    assert scale == 0.35 and len(lines) == 2
    assert all(_width(line, scale) <= 200 and line.endswith("...") for line in lines)
    assert truncate("short", 200, 0.5) == "short"


def test_a_narrow_frame_wraps_the_bar_inside_it(write_video: Callable[..., Path]) -> None:
    report = _with(_report(write_video(), "video-0123456789ab"), rule=CALIBRATED)
    out = overlay_frames(_blank(width=200, shade=255), report, STYLE)[0]
    edge = (out[:, 0] == BAR_COLOR).all(axis=1)
    bar_height = int(np.argmin(edge))
    assert STYLE.bar_height < bar_height <= 2 * STYLE.bar_height
    assert (out[:bar_height, -3:] == BAR_COLOR).all()


def test_side_by_side_matches_heights() -> None:
    left = np.zeros((5, 96, 128, 3), dtype=np.uint8)
    right = np.zeros((4, 48, 32, 3), dtype=np.uint8)
    both = side_by_side(left, right, ("base", "ft"), STYLE)
    assert both.shape == (4, 48, 64 + 32, 3)


def test_side_by_side_labels_sit_on_a_box_beside_a_divider() -> None:
    clip = np.full((2, HEIGHT, 320, 3), 100, dtype=np.uint8)
    both = side_by_side(clip, clip, ("base", "fine-tuned"), STYLE)
    for x in (STYLE.margin + 1, 320 + STYLE.margin + 1):
        assert tuple(int(v) for v in both[0, HEIGHT - STYLE.margin - 1, x]) == BAR_COLOR
    assert tuple(int(v) for v in both[0, HEIGHT // 2, 320]) == DIVIDER_COLOR
    assert tuple(int(v) for v in both[0, HEIGHT // 2, 300]) == (100, 100, 100)


def test_render_overlay_round_trips(tmp_path: Path, write_video: Callable[..., Path]) -> None:
    video = write_video()
    path = render_overlay(_report(video), tmp_path / "o" / "overlay.mp4", STYLE, BUDGET)
    source = Mp4VideoSource(path, BUDGET)
    assert source.n_frames == Mp4VideoSource(video, BUDGET).n_frames
    assert source.native_size == (128, 96)
    assert source.real_fps == pytest.approx(8.0)
