"""Real-video metrics (ADR 0007 §5.5): motion agreement and the free-fall diagnosis."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_clip import build_clip

from physverify.config.models import RuntimeSettings
from physverify.diagnosis.real_video_metrics import (
    TimeBases,
    fall_diagnosis,
    fall_window,
    gravity_direction,
    motion_scores,
    moving_objects,
)
from physverify.diagnosis.settings import RealSettings
from physverify.schema.checks import AbstainReason
from physverify.schema.clip import FrameSampling
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.perception import MaskTrack
from physverify.schema.scale import EffectiveScale, ScaleSource
from physverify.schema.states import CoordFrame, ObjectStateTable

WIDTH, HEIGHT, SIDE = 40, 20, 10
SAME_BASE = TimeBases(declared_fps=24.0, container_fps=24.0)
TRUTH_SCALE = EffectiveScale(meters_per_unit=1.0, rel_sigma=0.0, source=ScaleSource.GROUND_TRUTH)


@pytest.fixture
def real() -> RealSettings:
    return RealSettings(
        datasets=("physics_iq", "pisa_real"),
        motion_threshold=20,
        motion_min_frac=0.01,
        moving_iou=0.9,
        fall_speed_drop=0.5,
        min_fall_frames=4,
    )


# --- Motion -----------------------------------------------------------------------------------


def square(left: int) -> NDArray[np.bool_]:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[0:SIDE, left : left + SIDE] = True
    return mask


def columns(start: int, stop: int) -> NDArray[np.bool_]:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[0:SIDE, start:stop] = True
    return mask


def track(*objects: list[NDArray[np.bool_]]) -> MaskTrack:
    """Objects given as per-frame masks; every object has the same number of frames."""
    masks = np.stack([np.stack(frames) for frames in objects], axis=1)
    n_t, n_obj = masks.shape[:2]
    sampling = FrameSampling(
        frame_indices=np.arange(n_t, dtype=np.int64),
        width=WIDTH,
        height=HEIGHT,
        native_width=WIDTH,
        native_height=HEIGHT,
        seconds_per_frame=1.0 / 16,
        step=1,
        max_side=None,
    )
    return MaskTrack(
        sampling=sampling,
        object_ids=np.array([f"o{i}" for i in range(n_obj)]),
        labels=np.array(["ball"] * n_obj),
        masks=masks,
        scores=np.ones((n_t, n_obj)),
    )


def still(n_frames: int) -> list[NDArray[np.bool_]]:
    """A second object parked at the right edge."""
    return [square(WIDTH - SIDE)] * n_frames


def test_moving_objects_flags_the_shifted_square_only() -> None:
    moving = moving_objects(track([square(0), square(5), square(10)], still(3)), 0.9)
    assert moving.tolist() == [[False, False], [True, False], [True, False]]


def test_motion_scores_are_analytic(real: RealSettings) -> None:
    masks = track([square(0), square(5), square(10)], still(3))
    gt = np.stack([columns(0, 0), columns(0, 15), columns(10, 20)])
    scores = motion_scores(masks, gt, real)
    assert scores.n_frames == 2
    # P_1 = cols 0-14 matches G_1; P_2 = cols 5-19 against G_2 = cols 10-19 gives 2/3.
    assert scores.motion_iou == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert scores.precision == pytest.approx(1.0)
    assert scores.recall == pytest.approx(1.0)
    assert scores.false_motion_rate == pytest.approx(1.0)


def test_static_clip_without_motion_scores_zero_false_motion(real: RealSettings) -> None:
    gt = np.zeros((4, HEIGHT, WIDTH), dtype=bool)
    scores = motion_scores(track([square(3)] * 4), gt, real)
    assert scores.false_motion_rate == 0.0
    assert scores.motion_iou is None
    assert scores.precision is None
    assert scores.recall is None


def test_jitter_on_a_static_clip_is_false_motion(real: RealSettings) -> None:
    gt = np.zeros((3, HEIGHT, WIDTH), dtype=bool)
    # A 1 px shift of a 10 px square has IoU 9/11 < 0.9: it counts as moving.
    scores = motion_scores(track([square(3), square(4), square(3)]), gt, real)
    assert scores.false_motion_rate == pytest.approx(1.0)
    assert scores.precision == 0.0
    assert scores.recall is None
    assert scores.motion_iou == 0.0


def test_motion_scores_reject_a_mask_of_another_size(real: RealSettings) -> None:
    with pytest.raises(ValueError, match="do not match"):
        motion_scores(track([square(0)] * 2), np.zeros((2, HEIGHT, WIDTH + 1), bool), real)


# --- Free fall --------------------------------------------------------------------------------


def fixture_states(frame: CoordFrame = CoordFrame.WORLD_ZUP) -> ObjectStateTable:
    """The analytic clip: a ball in free fall (g0, 24 fps) next to a resting box."""
    return build_clip(RuntimeSettings()).ground_truth.states(frame)


def test_fixture_fall_recovers_g0_with_the_truth_scale(real: RealSettings) -> None:
    states = fixture_states()
    result = fall_diagnosis(states, TRUTH_SCALE, real, SAME_BASE)
    assert result.abstain_reason is None
    assert result.object_id == "ball"
    assert result.n_frames == states.n_frames
    assert result.residual_ratio is not None and result.residual_ratio <= 1e-9
    assert result.lateral_ratio is not None and result.lateral_ratio <= 1e-9
    assert result.g_est is not None
    assert abs(result.g_est - STANDARD_GRAVITY) / STANDARD_GRAVITY <= 1e-6
    assert result.g_rel_error == pytest.approx(0.0, abs=1e-6)
    assert result.g_rel_sigma == 0.0
    assert result.closer_time_base == "declared"
    assert not result.gravity_dir_estimated


@pytest.mark.parametrize(
    ("scale", "reason"),
    [
        (
            EffectiveScale(meters_per_unit=1.0, rel_sigma=0.0, source=ScaleSource.FREE_FALL),
            AbstainReason.SCALE_FROM_GRAVITY,
        ),
        (None, AbstainReason.SCALE_UNAVAILABLE),
    ],
)
def test_g_abstains_without_an_independent_scale(
    real: RealSettings, scale: EffectiveScale | None, reason: AbstainReason
) -> None:
    result = fall_diagnosis(fixture_states(), scale, real, SAME_BASE)
    assert result.abstain_reason is reason
    assert result.g_est is None and result.g_declared is None and result.closer_time_base is None
    assert result.residual_ratio is not None and result.residual_ratio <= 1e-9


def test_scaled_units_and_unknown_sigma(real: RealSettings) -> None:
    states = fixture_states()
    halved = dataclasses.replace(
        states, position=states.position * 0.5, velocity=states.velocity * 0.5
    )
    scale = EffectiveScale(meters_per_unit=2.0, rel_sigma=math.nan, source=ScaleSource.SIZE_PRIOR)
    result = fall_diagnosis(halved, scale, real, SAME_BASE)
    assert result.g_est == pytest.approx(STANDARD_GRAVITY, rel=1e-6)
    assert result.g_rel_sigma is None


def test_time_base_check_compares_declared_and_container_rates(real: RealSettings) -> None:
    fps = TimeBases(declared_fps=120.0, container_fps=29.97)
    result = fall_diagnosis(fixture_states(), TRUTH_SCALE, real, fps)
    assert result.g_declared == pytest.approx(STANDARD_GRAVITY, rel=1e-6)
    assert result.g_container == pytest.approx(STANDARD_GRAVITY * (29.97 / 120.0) ** 2, rel=1e-6)
    assert result.closer_time_base == "declared"


def test_container_rate_wins_when_the_declared_rate_is_wrong(real: RealSettings) -> None:
    # A time axis 4x too long divides g by 16; the container rate (4x the declared) restores it.
    states = fixture_states()
    stretched = dataclasses.replace(states, t=states.t * 4.0)
    fps = TimeBases(declared_fps=6.0, container_fps=24.0)
    result = fall_diagnosis(stretched, TRUTH_SCALE, real, fps)
    assert result.g_declared == pytest.approx(STANDARD_GRAVITY / 16.0, rel=1e-6)
    assert result.g_container == pytest.approx(STANDARD_GRAVITY, rel=1e-6)
    assert result.closer_time_base == "container"


def test_window_stops_at_the_impact(real: RealSettings) -> None:
    states = fixture_states()
    impact = 8
    position, velocity = states.position.copy(), states.velocity.copy()
    position[impact:, 0] = position[impact - 1, 0]
    velocity[impact:, 0] = 0.0
    hit = dataclasses.replace(states, position=position, velocity=velocity)
    g_dir, _ = gravity_direction(hit)
    window = fall_window(hit, g_dir, real)
    assert window is not None
    assert (window.object_index, window.start, window.stop) == (0, 0, impact)
    result = fall_diagnosis(hit, TRUTH_SCALE, real, SAME_BASE)
    assert result.n_frames == impact
    assert result.g_est == pytest.approx(STANDARD_GRAVITY, rel=1e-6)


def test_window_starts_at_the_first_visible_frame(real: RealSettings) -> None:
    states = fixture_states()
    visible = states.visible.copy()
    visible[:3, 0] = False
    window = fall_window(
        dataclasses.replace(states, visible=visible), gravity_direction(states)[0], real
    )
    assert window is not None and window.start == 3


def test_short_window_and_no_fall_abstain(real: RealSettings) -> None:
    strict = real.model_copy(update={"min_fall_frames": 50})
    short = fall_diagnosis(fixture_states(), TRUTH_SCALE, strict, SAME_BASE)
    assert short.abstain_reason is AbstainReason.TOO_FEW_FRAMES
    assert short.object_id == "ball"
    states = fixture_states()
    parked = states.position.copy()
    parked[:, 0] = parked[0, 0]
    resting = dataclasses.replace(states, position=parked)
    result = fall_diagnosis(resting, TRUTH_SCALE, real, SAME_BASE)
    assert result.abstain_reason is AbstainReason.OUT_OF_SCOPE
    assert result.object_id is None


def test_missing_gravity_direction_falls_back_to_camera_down(real: RealSettings) -> None:
    states = fixture_states(CoordFrame.CAMERA_OPENCV)
    unknown = dataclasses.replace(states, gravity_dir=np.full(3, np.nan))
    result = fall_diagnosis(unknown, TRUTH_SCALE, real, SAME_BASE)
    assert result.gravity_dir_estimated
    assert result.g_est == pytest.approx(STANDARD_GRAVITY, rel=1e-6)
