"""Overlay video of a verifier report: a top bar and check badges, captioned by ``captions``.

The bar wraps at its separators and shrinks its font, down to ``min_font_scale``, to stay inside
the frame; a line that still overflows at that floor is cut with an ellipsis. Badge boxes are
blended into the frame with opacity ``badge_alpha``. A check the calibrated rule decided gets a box
from green (p_c = 0) through amber to red (p_c = 1). A FAIL box is bright red only inside the
failing result's frame span (native frames, inclusive) and dim red elsewhere; a FAIL without a span
stays red on every frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from physverify.core.resample import resize_frame
from physverify.core.video import Frames, Mp4VideoSource, write_mp4
from physverify.demo.captions import SEPARATOR, badge_text, bar_segments, shown_rows, violation
from physverify.demo.report import NO_VERDICT, CheckRow, VideoReport
from physverify.demo.settings import OverlayStyle
from physverify.schema.checks import Verdict

Color = tuple[int, int, int]
PASS_COLOR: Final[Color] = (40, 170, 60)
AMBER_COLOR: Final[Color] = (225, 150, 20)
FAIL_COLOR: Final[Color] = (220, 40, 40)
FAIL_DIM_COLOR: Final[Color] = (110, 30, 30)
UNDECIDED_COLOR: Final[Color] = (128, 128, 128)
NO_EVENT_COLOR: Final[Color] = (70, 70, 70)
BAR_COLOR: Final[Color] = (0, 0, 0)
TEXT_COLOR: Final[Color] = (255, 255, 255)
OUTLINE_COLOR: Final[Color] = (0, 0, 0)
DIVIDER_COLOR: Final[Color] = (255, 255, 255)
FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_FILE: Final = "overlay.mp4"
SIDE_BY_SIDE_FILE: Final = "side_by_side.mp4"
ELLIPSIS: Final = "..."
MAX_BAR_LINES: Final = 2
SHRINK_STEP: Final = 0.9
BADGE_FONT: Final = 0.9
HALO_OFFSETS: Final = ((-1, -1), (1, -1), (-1, 1), (1, 1))


def _mix(low: Color, high: Color, t: float) -> Color:
    r, g, b = (round(a + (z - a) * t) for a, z in zip(low, high, strict=True))
    return r, g, b


def probability_color(p: float) -> Color:
    """Green at p = 0, amber at 0.5, red at 1."""
    p = min(1.0, max(0.0, p))
    return (
        _mix(PASS_COLOR, AMBER_COLOR, 2 * p)
        if p <= 0.5
        else _mix(AMBER_COLOR, FAIL_COLOR, 2 * p - 1)
    )


def badge_color(row: CheckRow, frame: int, calibrated: bool = False) -> Color:
    """``calibrated``: the active rule's term is a violation probability."""
    p = violation(row, calibrated)
    if p is not None:
        return probability_color(p)
    if row.verdict == Verdict.PASS.value:
        return PASS_COLOR
    if row.verdict == Verdict.FAIL.value:
        span = row.frame_span
        inside = span is None or span[0] <= frame <= span[1]
        return FAIL_COLOR if inside else FAIL_DIM_COLOR
    if row.verdict == NO_VERDICT:
        return NO_EVENT_COLOR
    return UNDECIDED_COLOR


def _thickness(scale: float) -> int:
    return max(1, round(scale))


def _metrics(text: str, scale: float) -> tuple[int, int, int]:
    """Width, ascent and descent of ``text`` in pixels."""
    (width, ascent), descent = cv2.getTextSize(text, FONT, scale, _thickness(scale))
    return int(width), int(ascent), int(descent)


def _text_width(text: str, scale: float) -> int:
    return _metrics(text, scale)[0]


def truncate(text: str, width: int, scale: float) -> str:
    """``text``, or its longest prefix with an ellipsis that is at most ``width`` pixels wide."""
    if _text_width(text, scale) <= width:
        return text
    for end in range(len(text) - 1, 0, -1):
        cut = text[:end].rstrip() + ELLIPSIS
        if _text_width(cut, scale) <= width:
            return cut
    return ELLIPSIS


def _wrap(segments: Sequence[str], width: int, scale: float) -> list[str]:
    lines: list[str] = []
    for segment in segments:
        joined = f"{lines[-1]}{SEPARATOR}{segment}" if lines else segment
        if lines and _text_width(joined, scale) <= width:
            lines[-1] = joined
        else:
            lines.append(segment)
    return lines


def _fits(lines: Sequence[str], width: int, scale: float) -> bool:
    return len(lines) <= MAX_BAR_LINES and all(_text_width(x, scale) <= width for x in lines)


def fit_lines(
    segments: Sequence[str], width: int, scale: float, min_scale: float
) -> tuple[list[str], float]:
    """At most ``MAX_BAR_LINES`` lines of ``segments`` no wider than ``width`` pixels, at the
    largest scale from ``scale`` down to ``min_scale``; at ``min_scale`` the overflow is cut."""
    while scale > min_scale:
        lines = _wrap(segments, width, scale)
        if _fits(lines, width, scale):
            return lines, scale
        scale = max(min_scale, scale * SHRINK_STEP)
    lines = _wrap(segments, width, scale)
    kept = lines[: MAX_BAR_LINES - 1]
    rest = SEPARATOR.join(lines[MAX_BAR_LINES - 1 :])
    return [truncate(line, width, scale) for line in [*kept, rest] if line], scale


@dataclass(frozen=True)
class _Layout:
    """Everything an overlay draws that does not change from frame to frame."""

    margin: int
    pad: int
    bar_lines: tuple[str, ...]
    bar_scale: float
    line_height: int
    rows: tuple[CheckRow, ...]
    texts: tuple[str, ...]
    badge_scale: float
    badge_height: int
    panel_width: int
    alpha: float
    calibrated: bool

    @property
    def bar_height(self) -> int:
        return self.line_height * len(self.bar_lines)


def _scale(frame: Frames, style: OverlayStyle) -> float:
    return float(frame.shape[0]) / style.reference_height


def _layout(frame: Frames, report: VideoReport, style: OverlayStyle) -> _Layout:
    k = _scale(frame, style)
    margin = round(style.margin * k)
    pad = max(1, margin // 2)
    room = max(1, frame.shape[1] - 2 * margin)
    lines, bar_scale = fit_lines(
        bar_segments(report),
        room,
        style.font_scale * k,
        min(style.min_font_scale, style.font_scale) * k,
    )
    calibrated = report.rule.calibrated
    rows = tuple(shown_rows(report, style.all_checks))
    badge_scale = style.font_scale * BADGE_FONT * k
    texts = tuple(truncate(badge_text(r, calibrated), room - 2 * pad, badge_scale) for r in rows)
    widest = max((_text_width(t, badge_scale) + 2 * pad for t in texts), default=0)
    return _Layout(
        margin=margin,
        pad=pad,
        bar_lines=tuple(lines),
        bar_scale=bar_scale,
        line_height=max(1, round(style.bar_height * bar_scale / style.font_scale)),
        rows=rows,
        texts=texts,
        badge_scale=badge_scale,
        badge_height=max(1, round(style.badge_height * k)),
        panel_width=min(room, max(round(style.badge_width * k), widest)),
        alpha=style.badge_alpha,
        calibrated=calibrated,
    )


def _put_text(frame: Frames, text: str, origin: tuple[int, int], scale: float) -> None:
    """White text on a dark halo, legible on any background. The halo is the same text shifted,
    not a thicker stroke: OpenCV 5 draws a thicker stroke wider."""
    thickness = _thickness(scale)
    x, y = origin
    for dx, dy in HALO_OFFSETS:
        shifted = (x + dx * thickness, y + dy * thickness)
        cv2.putText(frame, text, shifted, FONT, scale, OUTLINE_COLOR, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def _blend_box(
    frame: Frames, corner: tuple[int, int], size: tuple[int, int], color: Color, alpha: float
) -> None:
    x, y = corner
    region = frame[y : y + size[1], x : x + size[0]]
    if region.size == 0:
        return
    tint = np.empty_like(region)
    tint[:] = color
    region[:] = cv2.addWeighted(tint, alpha, region, 1.0 - alpha, 0.0)


def _draw_bar(frame: Frames, layout: _Layout) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, layout.bar_height), BAR_COLOR, -1)
    for index, line in enumerate(layout.bar_lines):
        baseline = index * layout.line_height + round(layout.line_height * 0.7)
        _put_text(frame, line, (layout.margin, baseline), layout.bar_scale)


def _draw_badges(frame: Frames, layout: _Layout, index: int) -> None:
    top = layout.bar_height + layout.margin
    height = layout.badge_height
    for row, text in zip(layout.rows, layout.texts, strict=True):
        if top + height > frame.shape[0]:
            return
        color = badge_color(row, index, layout.calibrated)
        _blend_box(frame, (layout.margin, top), (layout.panel_width, height), color, layout.alpha)
        origin = (layout.margin + layout.pad, top + round(height * 0.72))
        _put_text(frame, text, origin, layout.badge_scale)
        top += height + layout.pad


def overlay_frames(frames: Frames, report: VideoReport, style: OverlayStyle) -> Frames:
    """Annotated RGB copies of ``frames`` (T x H x W x 3, uint8); frame i is native frame i."""
    out = np.ascontiguousarray(frames.copy())
    if len(out) == 0:
        return out
    layout = _layout(out[0], report, style)
    for index, frame in enumerate(out):
        _draw_bar(frame, layout)
        _draw_badges(frame, layout, index)
    return out


def _to_height(frames: Frames, height: int) -> Frames:
    if frames.shape[1] == height:
        return frames
    width = max(1, round(frames.shape[2] * height / frames.shape[1]))
    return np.stack([resize_frame(frame, (width, height)) for frame in frames])


def _label(frames: Frames, label: str, style: OverlayStyle) -> Frames:
    """``label`` on an opaque box at the bottom left of every frame."""
    out = np.ascontiguousarray(frames.copy())
    k = _scale(out[0], style)
    margin, scale = round(style.margin * k), style.label_scale * k
    pad = max(1, margin // 2)
    text = truncate(label, out.shape[2] - 2 * (margin + pad), scale)
    width, ascent, descent = _metrics(text, scale)
    bottom = out.shape[1] - margin
    top = max(0, bottom - ascent - descent - 2 * pad)
    for frame in out:
        cv2.rectangle(frame, (margin, top), (margin + width + 2 * pad, bottom), BAR_COLOR, -1)
        _put_text(frame, text, (margin + pad, bottom - pad - descent), scale)
    return out


def side_by_side(
    left: Frames, right: Frames, labels: tuple[str, str], style: OverlayStyle
) -> Frames:
    """Both clips at the smaller height, truncated to the shorter one, left beside right, with a
    divider on the seam."""
    height = min(left.shape[1], right.shape[1])
    length = min(len(left), len(right))
    parts = [
        _label(_to_height(clip[:length], height), label, style)
        for clip, label in zip((left, right), labels, strict=True)
    ]
    both = np.concatenate(parts, axis=2)
    seam, half = parts[0].shape[2], max(1, round(_scale(both[0], style)))
    both[:, :, max(0, seam - half) : seam + half] = DIVIDER_COLOR
    return both


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
