"""Identity matching between perceived and ground-truth objects, and truth re-expressed in the
perceived ids, frames and length unit (ADR 0005, 3.6 and 4.5)."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from physverify.core.errors import MissingInputError
from physverify.datasets._base import GroundTruthStore
from physverify.schema.clip import FrameSampling
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.matching import IdMap, MatchMethod
from physverify.schema.perception import Detections, MaskTrack
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable, PrimitiveKind

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

DEFAULT_MIN_IOU = 0.3
DEFAULT_MATCH_FRAMES: int | None = None
"""Sampled frames ``match_masks`` reads; None is every frame, so an object that enters late or
starts occluded is matched on the frames where it is visible."""
PROBE_FRAMES, PROBE_OBJECTS = 5, 7
"""Axis sizes of the layout probe: distinct from each other and from the trailing 3 and 4."""


def box_iou_matrix(a: FloatArray, b: FloatArray) -> FloatArray:
    """(N, 4) x (M, 4) xyxy boxes -> (N, M) intersection over union."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(hi - lo, 0.0, None), axis=-1)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0.0, None), axis=-1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0.0, None), axis=-1)
    union = area_a[:, None] + area_b[None, :] - inter
    iou: FloatArray = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return iou


def mask_iou_matrix(pred: BoolArray, gt: BoolArray, rows: Sequence[int]) -> FloatArray:
    """(T, N, h, w) x (T, M, h, w) -> (N, M): per-frame IoU averaged, for each ground-truth
    object, over the ``rows`` where it is visible (0 when it is visible in none).

    A per-frame mean, not pooled pixel counts, so a large object on a few frames cannot outweigh
    a small one on the rest; accumulated one frame at a time, so memory does not grow with T."""
    n, m = pred.shape[1], gt.shape[1]
    total = np.zeros((n, m), dtype=np.float64)
    visible = np.zeros(m, dtype=np.float64)
    for row in rows:
        p = pred[row].reshape(n, -1).astype(np.float64)
        g = gt[row].reshape(m, -1).astype(np.float64)
        inter = p @ g.T
        union = p.sum(axis=1)[:, None] + g.sum(axis=1)[None, :] - inter
        shown = g.sum(axis=1) > 0
        total += np.divide(inter, union, out=np.zeros_like(inter), where=union > 0) * shown
        visible += shown
    iou: FloatArray = np.divide(
        total, visible[None, :], out=np.zeros_like(total), where=visible[None, :] > 0
    )
    return iou


def hungarian(iou: FloatArray, min_iou: float) -> tuple[tuple[int, int], ...]:
    """Maximum-IoU one-to-one assignment, keeping only pairs with IoU >= ``min_iou``."""
    if iou.size == 0:
        return ()
    rows, cols = linear_sum_assignment(iou, maximize=True)
    return tuple((int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou)


def _id_map(
    iou: FloatArray,
    pred_ids: Sequence[str],
    gt_ids: Sequence[str],
    *,
    method: MatchMethod,
    frames_used: tuple[int, ...],
    min_iou: float,
) -> IdMap:
    pairs = hungarian(iou, min_iou)
    matched_pred = {i for i, _ in pairs}
    matched_gt = {j for _, j in pairs}
    return IdMap(
        pairs=tuple((str(pred_ids[i]), str(gt_ids[j]), float(iou[i, j])) for i, j in pairs),
        unmatched_pred=tuple(str(p) for i, p in enumerate(pred_ids) if i not in matched_pred),
        unmatched_gt=tuple(str(g) for j, g in enumerate(gt_ids) if j not in matched_gt),
        method=method,
        frames_used=frames_used,
        min_score=float(min_iou),
    )


def match_boxes(pred: Detections, gt: Detections, min_iou: float = DEFAULT_MIN_IOU) -> IdMap:
    return _id_map(
        box_iou_matrix(pred.boxes_xyxy, gt.boxes_xyxy),
        pred.object_ids.tolist(),
        gt.object_ids.tolist(),
        method=MatchMethod.BOX_IOU,
        frames_used=(int(pred.frame_index),),
        min_iou=min_iou,
    )


def gt_detections_at(gt: GroundTruthStore, sampling: FrameSampling, frame: int) -> Detections:
    """Ground-truth boxes of one native frame, at the processing size of ``sampling``."""
    single = replace(sampling, frame_indices=np.array([frame], dtype=np.int64))
    return gt.detections(single)


def match_masks(
    pred: MaskTrack,
    gt: MaskTrack,
    frames: int | None = DEFAULT_MATCH_FRAMES,
    min_iou: float = DEFAULT_MIN_IOU,
) -> IdMap:
    """One-to-one identities by mask IoU over the first ``frames`` sampled frames (all: None)."""
    shared = min(pred.masks.shape[0], gt.masks.shape[0])
    rows = list(range(shared if frames is None else min(frames, shared)))
    return _id_map(
        mask_iou_matrix(pred.masks, gt.masks, rows),
        pred.object_ids.tolist(),
        gt.object_ids.tolist(),
        method=MatchMethod.MASK_IOU,
        frames_used=tuple(int(pred.sampling.frame_indices[r]) for r in rows),
        min_iou=min_iou,
    )


@functools.cache
def _state_array_shapes() -> dict[str, tuple[int, ...]]:
    """Array field shapes of a blank probe table, read through the public API only."""
    ids = [f"probe_{n}" for n in range(PROBE_OBJECTS)]
    probe = ObjectStateTable.blank(
        clip_id="probe",
        frame=CoordFrame.CAMERA_OPENCV,
        unit=LengthUnit.RELATIVE,
        source_key="probe",
        t=np.arange(PROBE_FRAMES, dtype=np.float64),
        frame_index=np.arange(PROBE_FRAMES, dtype=np.int64),
        object_ids=ids,
        labels=ids,
        primitive=[PrimitiveKind.UNKNOWN.value] * PROBE_OBJECTS,
    )
    return {
        name: value.shape
        for name, value in probe.field_values().items()
        if isinstance(value, np.ndarray)
    }


def _per_frame_fields() -> tuple[str, ...]:
    shapes = _state_array_shapes()
    return tuple(name for name, shape in shapes.items() if shape[:1] == (PROBE_FRAMES,))


def _per_object_frame_fields() -> tuple[str, ...]:
    wanted = (PROBE_FRAMES, PROBE_OBJECTS)
    return tuple(name for name, shape in _state_array_shapes().items() if shape[:2] == wanted)


def take_frames(states: ObjectStateTable, frame_indices: IntArray) -> ObjectStateTable:
    """Rows of ``states`` for the native ``frame_indices`` (every one must exist)."""
    lookup = {int(f): row for row, f in enumerate(states.frame_index.tolist())}
    wanted = [int(f) for f in np.asarray(frame_indices).tolist()]
    missing = [f for f in wanted if f not in lookup]
    if missing:
        raise MissingInputError(f"{states.clip_id}: ground-truth states lack frames {missing[:5]}")
    rows = np.array([lookup[f] for f in wanted], dtype=np.int64)
    values = {name: getattr(states, name)[rows] for name in _per_frame_fields()}
    return replace(states, **values)


def _take_objects(values: NDArray[Any], source: IntArray) -> NDArray[Any]:
    fill: Any = False if values.dtype == np.bool_ else np.nan
    out = np.full((values.shape[0], source.size, *values.shape[2:]), fill, dtype=values.dtype)
    matched = source >= 0
    out[:, matched] = values[:, source[matched]]
    return out


def remap_states(
    gt: ObjectStateTable, ids: IdMap, pred_ids: Sequence[str], pred_labels: Sequence[str]
) -> ObjectStateTable:
    """Truth rows in the order of ``pred_ids``; an unmatched id is NaN and never visible."""
    gt_of = ids.pred_to_gt()
    source = np.array(
        [gt.object_index(gt_of[p]) if p in gt_of else -1 for p in pred_ids], dtype=np.int64
    )
    values: dict[str, Any] = {
        name: _take_objects(getattr(gt, name), source) for name in _per_object_frame_fields()
    }
    primitive = [str(gt.primitive[i]) if i >= 0 else PrimitiveKind.UNKNOWN.value for i in source]
    return replace(
        gt,
        object_ids=np.array([str(p) for p in pred_ids], dtype=str),
        labels=np.array([str(label) for label in pred_labels], dtype=str),
        primitive=np.array(primitive, dtype=str),
        **values,
    )


def _remap_record(
    record: ContactRecord, pred_of: dict[str, str], length_scale: float
) -> ContactRecord | None:
    names = [ENV_ID if o == ENV_ID else pred_of.get(o) for o in (record.object_a, record.object_b)]
    a, b = names
    if a is None or b is None:
        return None
    sign = 1.0
    if b != ENV_ID and (a == ENV_ID or b < a):
        a, b, sign = b, a, -1.0
    return replace(
        record,
        object_a=a,
        object_b=b,
        normal=sign * record.normal,
        points=record.points / length_scale,
        point_normals=None if record.point_normals is None else sign * record.point_normals,
        point_distances=None
        if record.point_distances is None
        else record.point_distances / length_scale,
        max_penetration=None
        if record.max_penetration is None
        else record.max_penetration / length_scale,
    )


def _clip_to_window(
    records: tuple[ContactRecord, ...], window: tuple[float, float]
) -> list[ContactRecord]:
    lo, hi = window
    return [
        replace(r, t_start=max(r.t_start, lo), t_end=min(r.t_end, hi))
        for r in records
        if r.t_end >= lo and r.t_start <= hi
    ]


def remap_contacts(
    table: ContactTable,
    ids: IdMap,
    *,
    source_key: str,
    unit: LengthUnit,
    length_scale: float,
    window: tuple[float, float] | None = None,
) -> ContactTable:
    """Contacts in perceived ids and table units (lengths divided by meters per unit).

    With ``window`` (first and last sampled native frame), records outside it are dropped and the
    rest are clipped to it, so no contact refers to frames without states.
    """
    pred_of = ids.gt_to_pred()
    in_window = table.records if window is None else _clip_to_window(table.records, window)
    records = [_remap_record(r, pred_of, length_scale) for r in in_window]
    return replace(
        table,
        source_key=source_key,
        unit=unit,
        records=tuple(r for r in records if r is not None),
    )


@dataclass(frozen=True)
class ScaleFit:
    meters_per_unit: float
    sigma: float
    n_objects: int
    n_samples: int


def _paired_samples(
    lift: ObjectStateTable, gt: ObjectStateTable
) -> list[tuple[FloatArray, FloatArray]]:
    """Per matched object: (T_k, 3) lift and truth positions where both are visible and finite."""
    common, lift_rows, gt_rows = np.intersect1d(
        lift.frame_index, gt.frame_index, return_indices=True
    )
    pairs = []
    for i, object_id in enumerate(lift.object_ids.tolist()):
        if object_id not in gt.object_ids.tolist() or common.size == 0:
            continue
        j = gt.object_index(object_id)
        p_l = lift.position[lift_rows, i]
        p_g = gt.position[gt_rows, j]
        ok = (
            lift.visible[lift_rows, i]
            & gt.visible[gt_rows, j]
            & np.isfinite(p_l).all(axis=1)
            & np.isfinite(p_g).all(axis=1)
        )
        if bool(ok.any()):
            pairs.append((p_l[ok], p_g[ok]))
    return pairs


def _ratio(p_l: FloatArray, p_g: FloatArray) -> float:
    return float(np.sum(p_l * p_g) / np.sum(p_l * p_l))


def conditional_scale_fit(lift: ObjectStateTable, gt: ObjectStateTable) -> ScaleFit | None:
    """Least-squares s with p_gt ~ s p_lift over every matched visible sample (ADR 0005, 4.5).

    sigma is the standard error of the per-object ratios (per-sample with one object) and NaN
    when a single ratio exists: one sample cannot show its own spread."""
    samples = _paired_samples(lift, gt)
    if not samples:
        return None
    p_l = np.concatenate([s[0] for s in samples])
    p_g = np.concatenate([s[1] for s in samples])
    scale = _ratio(p_l, p_g)
    if len(samples) > 1:
        ratios = np.array([_ratio(a, b) for a, b in samples])
    else:
        a, b = samples[0]
        ratios = np.sum(a * b, axis=1) / np.sum(a * a, axis=1)
    sigma = float(np.std(ratios, ddof=1) / np.sqrt(ratios.size)) if ratios.size > 1 else math.nan
    return ScaleFit(
        meters_per_unit=scale,
        sigma=sigma,
        n_objects=len(samples),
        n_samples=int(p_l.shape[0]),
    )


@dataclass(frozen=True)
class AlignedTruth:
    ids: IdMap
    states: ObjectStateTable


def align_ground_truth(
    gt: GroundTruthStore, track: MaskTrack, sampling: FrameSampling
) -> AlignedTruth:
    """Truth in camera_opencv, on the sampled frames, in TRACK ids and order, in meters."""
    ids = match_masks(track, gt.masks(sampling))
    states = take_frames(gt.states(CoordFrame.CAMERA_OPENCV), sampling.frame_indices)
    return AlignedTruth(
        ids=ids,
        states=remap_states(states, ids, track.object_ids.tolist(), track.labels.tolist()),
    )
