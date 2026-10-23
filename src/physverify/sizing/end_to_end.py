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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physverify.config.models import Settings
from physverify.core.atomic import write_text_atomic
from physverify.schema.clip import ClipManifest
from physverify.schema.perception import MaskTrack
from physverify.schema.scale import ScaleEstimate
from physverify.schema.states import ObjectStateTable
from physverify.sizing import objectron
from physverify.sizing.evaluate import (
    Errors,
    grouped_intervals,
    interval_text,
    metrics,
    subset,
    with_intervals,
)
from physverify.sizing.objectron import CENTER_KEYPOINT, VIDEO_FILE, FrameRecord, VideoRef
from physverify.sizing.objectron_draws import EndToEndDraw, e2e_draw, objectron_dir
from physverify.sizing.settings import EndToEndSettings, SizingSettings

NATIVE_FPS = 30.0
TIMEOUT_S = 60.0
Box = tuple[float, float, float, float]
"""Normalized (x0, y0, x1, y1), x right and y down in the upright (portrait) frame."""
ScaleOutputs = Callable[
    [ClipManifest], tuple[MaskTrack, ObjectStateTable, dict[str, ScaleEstimate]] | None
]
"""Track, lift and the scale of each method of a clip from the perception chain; None if any
stage failed."""


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


def truth_file(settings: Settings, clip_id: str) -> Path:
    return objectron_dir(settings) / "e2e" / f"{clip_id}.json"


def prepare_clip(
    settings: Settings,
    sizing: SizingSettings,
    plan: EndToEndSettings,
    frames_root: Path,
    ref: VideoRef,
) -> bool:
    """Frames and truth of one video; the MOV is deleted once decoded. False if too large."""
    clip_id = clip_id_of(ref)
    truth_path = truth_file(settings, clip_id)
    if truth_path.is_file():
        return True
    video_url = f"{sizing.bucket_url}/{objectron.video_prefix(ref)}{VIDEO_FILE}"
    if remote_size(video_url) > plan.max_video_mb * 1e6:
        return False
    work = objectron_dir(settings) / "e2e_download" / clip_id
    work.mkdir(parents=True, exist_ok=True)
    annotation = work / "annotation.pbdata"
    download(objectron.annotation_url(sizing.bucket_url, ref), annotation)
    download(video_url, work / VIDEO_FILE)
    step = step_of(plan.fps)
    written = decode_frames(
        work / VIDEO_FILE,
        frames_root / clip_id,
        step=step,
        n_frames=round(plan.seconds * plan.fps),
        max_side=plan.max_side,
    )
    buf = annotation.read_bytes()
    parsed = objectron.sequence_objects(buf, complete=True)
    if parsed is None:
        raise objectron.ObjectronError(f"{ref.video_id}: annotation has no objects")
    truth = clip_truth(ref, objectron.frame_records(buf[parsed[1] :]), step=step, n_frames=written)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(truth_path, truth.to_json())
    shutil.rmtree(work)
    return True


def manifest_of(refs: Sequence[VideoRef], plan: EndToEndSettings) -> dict[str, Any]:
    return {
        "clips": [
            {
                "video": f"{clip_id_of(ref)}/",
                "real_fps": plan.fps,
                "queries": [{"phrase": ref.label.replace("_", " "), "expected_count": 1}],
            }
            for ref in refs
        ]
    }


def clip_record(
    settings: Settings, outputs: ScaleOutputs, clip: ClipManifest, min_box_iou: float
) -> tuple[dict[str, Any], list[ScaleRow]]:
    record: dict[str, Any] = {"clip": clip.clip_id}
    path = truth_file(settings, clip.clip_id)
    if not path.is_file():
        return {**record, "excluded": "no_truth"}, []
    truth = ClipTruth.from_json(path.read_text(encoding="utf-8"))
    loaded = outputs(clip)
    if loaded is None:
        return {**record, "excluded": "chain_failed"}, []
    track, lift, estimates = loaded
    match = match_object(truth, track.masks, track.sampling.frame_indices)
    if match is None or match[1] < min_box_iou:
        iou = None if match is None else match[1]
        return {**record, "excluded": "low_box_iou", "iou": iou}, []
    object_id = str(track.object_ids[match[0]])
    n = lift.object_ids.tolist().index(object_id)
    s_true = true_scale(truth, lift.position[:, n, 2], lift.frame_index)
    record |= {"label": truth.label, "iou": match[1], "s_true": s_true}
    rows = [
        ScaleRow(clip.clip_id, method, s_true, est.meters_per_unit, est.sigma)
        for method, est in estimates.items()
        if est.meters_per_unit is not None and math.isfinite(s_true)
    ]
    record["rows"] = [asdict(r) for r in rows]
    return record, rows


def prepare(
    settings: Settings,
    sizing: SizingSettings,
    plan: EndToEndSettings,
    frames_root: Path,
    reader: objectron.PrefixReader,
) -> tuple[EndToEndDraw, Path]:
    """Prepare the drawn clips under ``frames_root`` and write their ``manifest.json``."""
    draw = e2e_draw(
        settings, sizing, plan, reader, partial(prepare_clip, settings, sizing, plan, frames_root)
    )
    manifest = frames_root / "manifest.json"
    write_text_atomic(manifest, json.dumps(manifest_of(draw.taken, plan), indent=1))
    return draw, manifest


def evaluate_clips(
    settings: Settings, sizing: SizingSettings, clips: Iterable[ClipManifest], outputs: ScaleOutputs
) -> dict[str, Any]:
    rows: list[ScaleRow] = []
    per_clip: list[dict[str, Any]] = []
    for clip in clips:
        record, clip_rows = clip_record(settings, outputs, clip, sizing.e2e.min_box_iou)
        per_clip.append(record)
        rows += clip_rows
    excluded: dict[str, int] = {}
    for record in per_clip:
        if "excluded" in record:
            excluded[record["excluded"]] = excluded.get(record["excluded"], 0) + 1
    total = summarize(rows, n_boot=sizing.bootstrap, seed=sizing.seed, ci=sizing.ci)
    return {
        "estimated": True,
        "n_clips": len(per_clip),
        "excluded": excluded,
        "min_box_iou": sizing.e2e.min_box_iou,
        "total": total,
        "per_clip": per_clip,
        "camera": "assumed FOV (the true-FOV attribution is not run)",
        "truth_limit": "ARKit depth of the box center; its own error is not documented",
    }


def e2e_lines(report: Mapping[str, Any]) -> list[str]:
    lines = [
        f"{method}: n {m['n']}  median|s/s-1| {interval_text(m['median_abs_rel'])}  "
        f"rms log {interval_text(m['rms_log'])}  cover 1s {interval_text(m['cover_1s'])}"
        for method, m in report["total"].items()
    ]
    return [*lines, f"excluded: {report['excluded']}"]
