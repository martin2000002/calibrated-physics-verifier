"""Pure metrics of the real-video diagnosis (ADR 0007 §5.5).

Physics-IQ masks mark pixel change, so the motion scores are agreement, not accuracy. The fall
diagnosis fits a parabola to the falling object; g is an estimate whenever the scale is.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from physverify.diagnosis.cases import BoolArray, FloatArray
from physverify.diagnosis.masks import mask_iou
from physverify.diagnosis.settings import RealSettings
from physverify.extraction.providers._gravity_pose import LEVEL_CAMERA_GRAVITY
from physverify.physics.measures.gravity_fit import (
    fall_fit,
    fall_residual_ratio,
    gravity_rel_error,
    lateral_accel_ratio,
)
from physverify.schema.checks import AbstainReason
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.perception import MaskTrack
from physverify.schema.scale import EffectiveScale, ScaleSource
from physverify.schema.states import ObjectStateTable

TimeBase = Literal["declared", "container"]

SCALE_ABSTAIN = {
    ScaleSource.FREE_FALL: AbstainReason.SCALE_FROM_GRAVITY,
    ScaleSource.NONE: AbstainReason.SCALE_UNAVAILABLE,
}


# --- Physics-IQ motion ------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionScores:
    n_frames: int
    motion_iou: float | None
    precision: float | None
    recall: float | None
    false_motion_rate: float | None


def moving_objects(track: MaskTrack, moving_iou: float) -> BoolArray:
    """(T, N): object i moves at f when IoU(M_i,f, M_i,f-1) < ``moving_iou`` (never at f = 0)."""
    masks = track.masks
    moving = np.zeros(masks.shape[:2], dtype=bool)
    if masks.shape[0] > 1:
        moving[1:] = mask_iou(masks[1:], masks[:-1]) < moving_iou
    return moving


def predicted_motion(track: MaskTrack, moving: BoolArray) -> BoolArray:
    """(T, h, w): P_f = union over moving i of (M_i,f | M_i,f-1)."""
    masks = track.masks
    swept = masks.copy()
    swept[1:] |= masks[:-1]
    region: BoolArray = np.any(swept & moving[:, :, None, None], axis=1)
    return region


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def motion_scores(track: MaskTrack, gt_motion: BoolArray, s: RealSettings) -> MotionScores:
    """Agreement of the tracked motion with G_f; frame 0 has no previous frame and is left out."""
    predicted = predicted_motion(track, moving_objects(track, s.moving_iou))
    if predicted.shape != gt_motion.shape:
        raise ValueError(f"motion masks {gt_motion.shape} do not match the track {predicted.shape}")
    p_region, g_region = predicted[1:], gt_motion[1:]
    n_frames = p_region.shape[0]
    iou = mask_iou(p_region, g_region)
    scored = iou[np.isfinite(iou)]
    a_min = s.motion_min_frac * track.sampling.width * track.sampling.height
    p = np.sum(p_region, axis=(1, 2)) >= a_min
    g = np.sum(g_region, axis=(1, 2)) >= a_min
    hits = float(np.sum(p & g))
    return MotionScores(
        n_frames=n_frames,
        motion_iou=float(np.mean(scored)) if scored.size else None,
        precision=_ratio(hits, float(np.sum(p))),
        recall=_ratio(hits, float(np.sum(g))),
        false_motion_rate=_ratio(float(np.sum(p)), n_frames),
    )


# --- PISA free fall ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeBases:
    """Declared capture rate (the time axis of the states) and the container rate."""

    declared_fps: float
    container_fps: float


@dataclass(frozen=True)
class FallWindow:
    object_index: int
    start: int
    stop: int

    @property
    def n_frames(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FallDiagnosis:
    """Fields of ``FallRow`` without dataset and clip; every g is an estimate."""

    object_id: str | None
    n_frames: int
    residual_ratio: float | None = None
    lateral_ratio: float | None = None
    g_est: float | None = None
    g_rel_error: float | None = None
    g_rel_sigma: float | None = None
    g_declared: float | None = None
    g_container: float | None = None
    closer_time_base: TimeBase | None = None
    gravity_dir_estimated: bool = False
    abstain_reason: AbstainReason | None = None


def gravity_direction(states: ObjectStateTable) -> tuple[FloatArray, bool]:
    """``states.gravity_dir``, else the level-camera +y flagged as estimated."""
    g_dir = np.asarray(states.gravity_dir, dtype=np.float64)
    if bool(np.all(np.isfinite(g_dir))):
        return g_dir, False
    return LEVEL_CAMERA_GRAVITY.copy(), True


def _descent(heights: FloatArray, visible: BoolArray) -> FloatArray:
    """Per object: sum of max(0, dh) over consecutive visible frames (h along gravity)."""
    step = np.diff(heights, axis=0)
    both = visible[1:] & visible[:-1] & np.isfinite(step)
    total: FloatArray = np.sum(np.where(both, np.maximum(step, 0.0), 0.0), axis=0)
    return total


def _window_stop(speed: FloatArray, visible: BoolArray, start: int, drop: float) -> int:
    """First frame after ``start`` that is invisible or slower than ``drop`` x the running max."""
    peak = -math.inf
    for f in range(start, speed.shape[0]):
        if not visible[f]:
            return f
        peak = max(peak, float(speed[f]))
        if peak > 0 and speed[f] < drop * peak:
            return f
    return int(speed.shape[0])


def fall_window(states: ObjectStateTable, g_dir: FloatArray, s: RealSettings) -> FallWindow | None:
    """The object with the longest descent, from its first visible frame to the impact."""
    heights = states.position @ g_dir
    visible = states.visible & np.isfinite(heights)
    if heights.shape[1] == 0 or not bool(visible.any()):
        return None
    descent = _descent(heights, visible)
    index = int(np.argmax(descent))
    if descent[index] <= 0:
        return None
    start = int(np.flatnonzero(visible[:, index])[0])
    speed = states.velocity[:, index] @ g_dir
    stop = _window_stop(speed, visible[:, index], start, s.fall_speed_drop)
    return FallWindow(index, start, stop)


def closer_time_base(g_declared: float, g_container: float) -> TimeBase:
    """The time base whose g is closer to g0 in |log(g / g0)|."""
    declared = abs(math.log(g_declared / STANDARD_GRAVITY))
    container = abs(math.log(g_container / STANDARD_GRAVITY))
    return "declared" if declared <= container else "container"


def with_gravity(
    row: FallDiagnosis, accel_norm: float, scale: EffectiveScale | None, fps: TimeBases
) -> FallDiagnosis:
    """g on both time bases; only a scale not derived from gravity itself gives one."""
    source = ScaleSource.NONE if scale is None else scale.source
    if scale is None or source in SCALE_ABSTAIN:
        return dataclasses.replace(row, abstain_reason=SCALE_ABSTAIN[source])
    g_declared = accel_norm * scale.meters_per_unit
    # A time axis stretched by k = declared/container scales accelerations by 1/k^2.
    g_container = g_declared * (fps.container_fps / fps.declared_fps) ** 2
    return dataclasses.replace(
        row,
        g_est=g_declared,
        g_rel_error=gravity_rel_error(accel_norm, scale.meters_per_unit),
        g_rel_sigma=scale.rel_sigma if math.isfinite(scale.rel_sigma) else None,
        g_declared=g_declared,
        g_container=g_container,
        closer_time_base=closer_time_base(g_declared, g_container),
    )


def fall_diagnosis(
    states: ObjectStateTable, scale: EffectiveScale | None, s: RealSettings, fps: TimeBases
) -> FallDiagnosis:
    g_dir, estimated = gravity_direction(states)
    window = fall_window(states, g_dir, s)
    if window is None:
        empty = states.n_objects == 0
        reason = AbstainReason.MISSING_INPUT if empty else AbstainReason.OUT_OF_SCOPE
        return FallDiagnosis(None, 0, gravity_dir_estimated=estimated, abstain_reason=reason)
    object_id = str(states.object_ids[window.object_index])
    frames = slice(window.start, window.stop)
    base = FallDiagnosis(object_id, window.n_frames, gravity_dir_estimated=estimated)
    if window.n_frames < s.min_fall_frames:
        return dataclasses.replace(base, abstain_reason=AbstainReason.TOO_FEW_FRAMES)
    t = states.t[frames]
    try:
        fit = fall_fit(t, states.position[frames, window.object_index])
    except ValueError:
        return dataclasses.replace(base, abstain_reason=AbstainReason.ILL_CONDITIONED)
    accel_norm = float(np.linalg.norm(fit.a))
    if not accel_norm > 0.0:
        return dataclasses.replace(base, abstain_reason=AbstainReason.ILL_CONDITIONED)
    shaped = dataclasses.replace(
        base,
        residual_ratio=fall_residual_ratio(fit, float(t[-1] - t[0])),
        lateral_ratio=lateral_accel_ratio(fit.a, g_dir),
    )
    return with_gravity(shaped, accel_norm, scale, fps)
