"""End-to-end check of the perceived scale on Objectron videos (ADR 0010, section 2.3(2)).

The truth scale of a clip is the median over frames of ARKit's metric depth of the box center
divided by the lifted depth of the matched object (camera z, relative units). The match is the
perceived object whose mask bounding box best overlaps the projected 3D box; clips whose best
median IoU is below ``min_box_iou`` are excluded and counted. ARKit's own depth error is the
limit of this truth."""

from __future__ import annotations

import json
import math
import shutil
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physverify.sizing.evaluate import Errors, grouped_intervals, metrics, subset, with_intervals
from physverify.sizing.objectron import CENTER_KEYPOINT, FrameRecord, VideoRef

NATIVE_FPS = 30.0
TIMEOUT_S = 60.0
Box = tuple[float, float, float, float]
"""Normalized (x0, y0, x1, y1), x right and y down in the upright (portrait) frame."""


@dataclass(frozen=True)
class TruthFrame:
    index: int
    """Index of the decoded frame (the native frame is ``index * step``)."""
    depth_m: float
    box: Box


@dataclass(frozen=True)
class ClipTruth:
    clip_id: str
    video_id: str
    label: str
    step: int
    frames: tuple[TruthFrame, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> ClipTruth:
        raw = json.loads(text)
        frames = tuple(
            TruthFrame(int(f["index"]), float(f["depth_m"]), tuple(f["box"])) for f in raw["frames"]
        )
        return cls(raw["clip_id"], raw["video_id"], raw["label"], int(raw["step"]), frames)


def clip_id_of(ref: VideoRef) -> str:
    return ref.video_id.replace("/", "_")


def step_of(fps: float) -> int:
    return max(1, round(NATIVE_FPS / fps))


def keypoint_box(points_2d: NDArray[np.float64]) -> Box:
    low, high = np.clip(points_2d.min(axis=0), 0, 1), np.clip(points_2d.max(axis=0), 0, 1)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def clip_truth(
    ref: VideoRef, records: Sequence[FrameRecord], *, step: int, n_frames: int
) -> ClipTruth:
    """The first annotated object of each decoded frame; frames without it are left out."""
    by_id = {r.frame_id: r for r in records}
    frames = []
    for index in range(n_frames):
        record = by_id.get(index * step)
        if record is None or not record.objects:
            continue
        obj = record.objects[0]
        depth = float(obj.depth_m[CENTER_KEYPOINT])
        if math.isfinite(depth) and depth > 0:
            frames.append(TruthFrame(index, depth, keypoint_box(obj.points_2d)))
    return ClipTruth(clip_id_of(ref), ref.video_id, ref.label, step, tuple(frames))


def box_iou(a: Box, b: Box) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_box(mask: NDArray[np.bool_]) -> Box | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def match_object(
    truth: ClipTruth, masks: NDArray[np.bool_], frame_indices: NDArray[np.int64]
) -> tuple[int, float] | None:
    """The object (column of ``masks``, shape (T, N, h, w)) with the best median IoU."""
    row_of = {int(f): t for t, f in enumerate(frame_indices)}
    best: tuple[int, float] | None = None
    for n in range(masks.shape[1]):
        ious = [
            box_iou(tf.box, box) if (box := mask_box(masks[row_of[tf.index], n])) else 0.0
            for tf in truth.frames
            if tf.index in row_of
        ]
        if ious and (best is None or float(np.median(ious)) > best[1]):
            best = (n, float(np.median(ious)))
    return best


def true_scale(
    truth: ClipTruth, lifted_z: NDArray[np.float64], frame_indices: NDArray[np.int64]
) -> float:
    """Median of ARKit depth / lifted depth over the frames where both exist (meters per unit)."""
    row_of = {int(f): t for t, f in enumerate(frame_indices)}
    ratios = [
        tf.depth_m / z
        for tf in truth.frames
        if tf.index in row_of and math.isfinite(z := float(lifted_z[row_of[tf.index]])) and z > 0
    ]
    return float(np.median(ratios)) if ratios else math.nan


@dataclass(frozen=True)
class ScaleRow:
    clip_id: str
    method: str
    s_true: float
    s_hat: float
    sigma: float

    @property
    def log_error(self) -> float:
        return math.log(self.s_hat / self.s_true)


def summarize(
    rows: Sequence[ScaleRow], *, n_boot: int, seed: int, ci: float
) -> dict[str, dict[str, Any]]:
    """Per method: the size metrics of ``sizing.evaluate`` on the scale, intervals by clip."""
    out: dict[str, dict[str, Any]] = {}
    for method in sorted({r.method for r in rows}):
        mine = [r for r in rows if r.method == method]
        errors = Errors(
            np.array([r.log_error for r in mine]), np.array([r.sigma / r.s_hat for r in mine])
        )
        bounds = grouped_intervals(
            [r.clip_id for r in mine], partial(_metrics_of, errors), n_boot=n_boot, seed=seed, ci=ci
        )
        out[method] = {"n": len(mine), **with_intervals(metrics(errors), bounds)}
    return out


def _metrics_of(errors: Errors, index: NDArray[np.intp]) -> dict[str, float]:
    return metrics(subset(errors, index))


def download(url: str, target: Path, timeout_s: float = TIMEOUT_S) -> None:
    pending = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout_s) as response, pending.open("wb") as out:
        shutil.copyfileobj(response, out)
    pending.replace(target)


def remote_size(url: str, timeout_s: float = TIMEOUT_S) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return int(response.headers.get("Content-Length", "0"))


def decode_frames(video: Path, target: Path, *, step: int, n_frames: int, max_side: int) -> int:
    """Upright frames ``0, step, 2 step, ...`` as JPEG files; returns how many were written."""
    import cv2

    target.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    written = 0
    try:
        for native in range(n_frames * step):
            ok, image = capture.read()
            if not ok:
                break
            if native % step:
                continue
            h, w = image.shape[:2]
            factor = max_side / max(h, w)
            if factor < 1:
                image = cv2.resize(image, (round(w * factor), round(h * factor)))
            cv2.imwrite(str(target / f"{written:05d}.jpg"), image)
            written += 1
    finally:
        capture.release()
    return written
