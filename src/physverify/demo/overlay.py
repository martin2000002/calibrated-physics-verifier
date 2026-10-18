"""Overlay video of a verifier report: a top bar with S and the rule, one badge per check.

A FAIL badge is bright red only inside the failing result's frame span (native frames, inclusive)
and dim red elsewhere; a FAIL without a span stays red on every frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from physverify.core.resample import resize_frame
from physverify.core.video import Frames, Mp4VideoSource, write_mp4
from physverify.demo.report import NO_VERDICT, CheckRow, VideoReport, score_text
from physverify.demo.settings import OverlayStyle
from physverify.schema.checks import Verdict

Color = tuple[int, int, int]
PASS_COLOR: Final[Color] = (40, 170, 60)
FAIL_COLOR: Final[Color] = (220, 40, 40)
FAIL_DIM_COLOR: Final[Color] = (110, 30, 30)
UNDECIDED_COLOR: Final[Color] = (128, 128, 128)
NO_EVENT_COLOR: Final[Color] = (70, 70, 70)
BAR_COLOR: Final[Color] = (0, 0, 0)
TEXT_COLOR: Final[Color] = (255, 255, 255)
FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_FILE: Final = "overlay.mp4"
SIDE_BY_SIDE_FILE: Final = "side_by_side.mp4"


def badge_color(row: CheckRow, frame: int) -> Color:
    if row.verdict == Verdict.PASS.value:
        return PASS_COLOR
    if row.verdict == Verdict.FAIL.value:
        span = row.frame_span
        inside = span is None or span[0] <= frame <= span[1]
        return FAIL_COLOR if inside else FAIL_DIM_COLOR
    if row.verdict == NO_VERDICT:
        return NO_EVENT_COLOR
    return UNDECIDED_COLOR


def bar_text(report: VideoReport) -> str:
    rule = report.rule
    text = f"{score_text(report.clip)} cov {report.clip.coverage:.2f} | {report.model_tag}"
    text += f" | {rule.label}"
    return text if rule.calibrated else f"{text} | PROVISIONAL"


def badge_text(row: CheckRow) -> str:
    reason = f" ({row.abstain_reasons[0]})" if row.abstain_reasons else ""
    return f"{row.check_id} {row.verdict}{reason}"


def _scale(frame: Frames, style: OverlayStyle) -> float:
    return float(frame.shape[0]) / style.reference_height


def _put_text(frame: Frames, text: str, origin: tuple[int, int], scale: float) -> None:
    thickness = max(1, round(scale))
    cv2.putText(frame, text, origin, FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_bar(frame: Frames, text: str, style: OverlayStyle) -> None:
    k = _scale(frame, style)
    height = max(1, round(style.bar_height * k))
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, height), BAR_COLOR, -1)
    _put_text(frame, text, (round(style.margin * k), round(height * 0.7)), style.font_scale * k)


def _draw_badges(frame: Frames, rows: Sequence[CheckRow], index: int, style: OverlayStyle) -> None:
    k = _scale(frame, style)
    height, width, margin = (
        round(v * k) for v in (style.badge_height, style.badge_width, style.margin)
    )
    top = round(style.bar_height * k) + margin
    for row in rows:
        if top + height > frame.shape[0]:
            return
        corner = (margin + width, top + height)
        cv2.rectangle(frame, (margin, top), corner, badge_color(row, index), -1)
        origin = (margin + max(1, margin // 2), top + round(height * 0.72))
        _put_text(frame, badge_text(row), origin, style.font_scale * 0.9 * k)
        top += height + max(1, margin // 2)


def overlay_frames(frames: Frames, report: VideoReport, style: OverlayStyle) -> Frames:
    """Annotated RGB copies of ``frames`` (T x H x W x 3, uint8); frame i is native frame i."""
    out = np.ascontiguousarray(frames.copy())
    text = bar_text(report)
    for index, frame in enumerate(out):
        _draw_bar(frame, text, style)
        _draw_badges(frame, report.checks, index, style)
    return out


def _to_height(frames: Frames, height: int) -> Frames:
    if frames.shape[1] == height:
        return frames
    width = max(1, round(frames.shape[2] * height / frames.shape[1]))
    return np.stack([resize_frame(frame, (width, height)) for frame in frames])


def _label(frames: Frames, label: str, style: OverlayStyle) -> Frames:
    out = np.ascontiguousarray(frames.copy())
    k = _scale(out[0], style)
    y = out.shape[1] - round(style.margin * k)
    for frame in out:
        _put_text(frame, label, (round(style.margin * k), y), style.font_scale * k)
    return out


def side_by_side(
    left: Frames, right: Frames, labels: tuple[str, str], style: OverlayStyle
) -> Frames:
    """Both clips at the smaller height, truncated to the shorter one, left beside right."""
    height = min(left.shape[1], right.shape[1])
    length = min(len(left), len(right))
    parts = [
        _label(_to_height(clip[:length], height), label, style)
        for clip, label in zip((left, right), labels, strict=True)
    ]
    return np.concatenate(parts, axis=2)


def read_video(path: Path, max_frame_bytes: int) -> tuple[Frames, float]:
    source = Mp4VideoSource(path, max_frame_bytes)
    return source.frames(0, source.n_frames), source.real_fps


def save_mp4(frames: Frames, path: Path, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(frames, path, fps)
    return path


def render_overlay(
    report: VideoReport, path: Path, style: OverlayStyle, max_frame_bytes: int
) -> Path:
    frames, fps = read_video(Path(report.video_path), max_frame_bytes)
    return save_mp4(overlay_frames(frames, report, style), path, fps)
