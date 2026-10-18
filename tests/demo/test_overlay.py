from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from physverify.core.video import Mp4VideoSource
from physverify.demo.overlay import (
    FAIL_COLOR,
    FAIL_DIM_COLOR,
    NO_EVENT_COLOR,
    PASS_COLOR,
    UNDECIDED_COLOR,
    badge_color,
    bar_text,
    overlay_frames,
    render_overlay,
    side_by_side,
)
from physverify.demo.report import NO_VERDICT, CheckRow, VideoReport
from physverify.demo.settings import OverlayStyle

STYLE = OverlayStyle()
BUDGET = 10**9
HEIGHT = 480


def _row(check_id: str, verdict: str, span: tuple[int, int] | None = None) -> CheckRow:
    return CheckRow(
        check_id=check_id,
        status="scored",
        verdict=verdict,
        magnitude=None,
        sigma=None,
        unit="",
        threshold=None,
        abstain_reasons=[],
        frame_span=span,
        n_subjects=1,
        rule_decided=True,
        rule_term=None,
    )


def _report(video: Path) -> VideoReport:
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
        "model_tag": "base",
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


def _badge_pixel(frames: np.ndarray, slot: int = 0) -> tuple[int, ...]:
    """A pixel at the right end of a badge, clear of its text (HEIGHT-high frames)."""
    top = STYLE.bar_height + STYLE.margin + slot * (STYLE.badge_height + STYLE.margin // 2)
    y, x = top + 2, STYLE.margin + STYLE.badge_width - 2
    return tuple(int(v) for v in frames[y, x])


def test_badge_colors() -> None:
    fail = _row("R2", "FAIL", (2, 4))
    assert badge_color(fail, 3) == FAIL_COLOR
    assert badge_color(fail, 5) == FAIL_DIM_COLOR
    assert badge_color(_row("R2", "FAIL"), 7) == FAIL_COLOR
    assert badge_color(_row("R1", "PASS"), 0) == PASS_COLOR
    assert badge_color(_row("R3a", "ABSTAIN"), 0) == UNDECIDED_COLOR
    assert badge_color(_row("R4", NO_VERDICT), 0) == NO_EVENT_COLOR


def test_overlay_marks_the_failure_only_inside_its_span(write_video: Callable[..., Path]) -> None:
    video = write_video(height=HEIGHT, width=320)
    report = _report(video)
    frames = np.zeros((6, HEIGHT, 320, 3), dtype=np.uint8)
    out = overlay_frames(frames, report, STYLE)
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert not frames.any()
    assert _badge_pixel(out[3]) == FAIL_COLOR
    assert _badge_pixel(out[5]) == FAIL_DIM_COLOR
    assert _badge_pixel(out[0], slot=1) == PASS_COLOR
    assert "PROVISIONAL" in bar_text(report) and "thr-1" in bar_text(report)


def test_side_by_side_matches_heights() -> None:
    left = np.zeros((5, 96, 128, 3), dtype=np.uint8)
    right = np.zeros((4, 48, 32, 3), dtype=np.uint8)
    both = side_by_side(left, right, ("base", "ft"), STYLE)
    assert both.shape == (4, 48, 64 + 32, 3)


def test_render_overlay_round_trips(tmp_path: Path, write_video: Callable[..., Path]) -> None:
    video = write_video()
    path = render_overlay(_report(video), tmp_path / "o" / "overlay.mp4", STYLE, BUDGET)
    source = Mp4VideoSource(path, BUDGET)
    assert source.n_frames == Mp4VideoSource(video, BUDGET).n_frames
    assert source.native_size == (128, 96)
    assert source.real_fps == pytest.approx(8.0)
