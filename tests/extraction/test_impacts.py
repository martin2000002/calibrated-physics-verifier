"""it3 change A: kinematic impacts (T2-T4, T6-T8, T10, T13; judgement J3, J6, J7).

Analytic scenes (tests/checks/check_scenarios.py) through ``contacts/kinematic`` with a pinhole
camera of FOCAL_PX pixels, so the pixel floor of every object is about 2 mm at 4 m.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import contact_scenes as cs
import numpy as np
import pytest
from synthetic_clip import SyntheticClip
from test_cache import UPSTREAM as CACHE_UPSTREAM
from test_cache import key_of, versioned

from physverify.extraction.providers._attribution import SOURCE
from physverify.extraction.providers._impacts import (
    NoiseScale,
    noise_scale,
    pixel_floor,
    track_impacts,
)
from physverify.extraction.providers.contacts_kinematic import KinematicContacts
from physverify.physics.tracks import MIN_NOISE_STENCILS
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, PrimitiveKind

sc = cs.sc
SIDE_ROWS, ALPHA = 4, 0.05
TEST_SIGMA = 4.0 / (cs.FOCAL_PX * math.sqrt(12.0))
UPSTREAM = {**CACHE_UPSTREAM, Stage.KINEMATICS: "k-kinematics"}


def kinematic(table: ContactTable) -> list[ContactRecord]:
    return [r for r in table.records if r.source == SOURCE]


def impacts_of(case: sc.Case, index: int = 0) -> list[Any]:
    """Detector alone on one track with the test pixel floor as its noise."""
    states = case.states
    noise = NoiseScale(np.full(3, TEST_SIGMA), known=True)
    return track_impacts(states.t, states.position[:, index], noise, SIDE_ROWS, ALPHA)


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(u @ v, -1.0, 1.0))))


# --- T2: a bounce between rows ---------------------------------------------------------------


@pytest.mark.parametrize("pitch", [0.0, 25.0])
@pytest.mark.parametrize("beta", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("e", [0.5, 0.8, 1.2, 1.5])
def test_t2_one_environment_impact_at_the_right_time(e: float, beta: float, pitch: float) -> None:
    case = sc.bounce(e=e, impact_frame=15.0 + beta, pitch_deg=pitch)
    records = kinematic(cs.kinematic_table(case))
    assert len(records) == 1
    record = records[0]
    assert (record.object_a, record.object_b, record.kind) == ("ball", ENV_ID, ContactKind.IMPACT)
    assert record.t_start == record.t_end
    assert abs(record.t_start - (15.0 + beta)) <= 1e-3
    up = -case.states.gravity_dir / np.linalg.norm(case.states.gravity_dir)
    np.testing.assert_allclose(record.normal, up, atol=1e-12)


def test_t2_the_record_is_one_point_on_the_surfaces_with_unknown_confidence() -> None:
    """J6: point_distances is an explicit zero, never None; the point is the ball's lowest
    point at the impact, i.e. the true contact point."""
    case = sc.bounce(impact_frame=15.3, pitch_deg=25.0)
    (record,) = kinematic(cs.kinematic_table(case))
    assert record.manifold is ManifoldKind.POINT and record.points.shape == (1, 3)
    assert record.point_distances is not None
    np.testing.assert_array_equal(record.point_distances, np.zeros(1))
    assert math.isnan(record.confidence)
    assert record.max_penetration is None
    (truth,) = case.contacts.records if case.contacts else ()
    np.testing.assert_allclose(record.points, truth.points, atol=1e-9)


# --- T3: a collision in the air --------------------------------------------------------------


@pytest.mark.parametrize("gain", [0.0, 0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("mass_ratio", [1.0, 3.0])
def test_t3_a_collision_gives_one_pair_record_and_no_environment(
    mass_ratio: float, gain: float
) -> None:
    case = sc.collision(mass_ratio=mass_ratio, gain=gain, impact_frame=15.3, pitch_deg=25.0)
    records = kinematic(cs.kinematic_table(case))
    assert [(r.object_a, r.object_b) for r in records] == [("ball_i", "ball_j")]
    (truth,) = case.contacts.records if case.contacts else ()
    assert angle_deg(records[0].normal, truth.normal) <= 5.0
    assert abs(records[0].t_start - 15.3) <= 1e-3


# --- T4: no impact without a velocity kink ---------------------------------------------------


def no_kinematic_records(case: sc.Case) -> bool:
    return not kinematic(cs.kinematic_table(case))


@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_t4_parabola_rest_and_rolling_have_no_impact(pitch: float) -> None:
    assert no_kinematic_records(sc.parabola(pitch_deg=pitch))
    assert no_kinematic_records(sc.rolling_ball(pitch_deg=pitch))
    assert no_kinematic_records(sc.stack(pitch_deg=pitch))
    assert no_kinematic_records(sc.floating_box(pitch_deg=pitch))


@pytest.mark.parametrize("change_frame", [12.3, 15.5, 16.0])
@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_t4_gravity_changes_and_scales_have_no_impact(pitch: float, change_frame: float) -> None:
    """r5a_gravity_switch (g changes in flight), r5b_gravity_scale and r5a_object_gravity (a
    whole flight at another g), levels 0.05..0.4 both ways."""
    g = sc.STANDARD_GRAVITY
    for level in (0.05, 0.1, 0.2, 0.4):
        for sign in (1.0, -1.0):
            factor = 1.0 + sign * level
            switch = sc.parabola(g_after=g * factor, change_frame=change_frame, pitch_deg=pitch)
            assert no_kinematic_records(switch), (level, sign)
            assert no_kinematic_records(sc.parabola(g=g * factor, pitch_deg=pitch))


@pytest.mark.parametrize("flight", [False, True])
@pytest.mark.parametrize("level", [0.25, 0.5, 1.0, 2.0])
def test_t4_a_teleport_at_any_boundary_has_no_impact(level: float, flight: bool) -> None:
    for pitch in (0.0, 25.0):
        for frame in range(1, sc.FRAMES):
            delta = level * 2 * sc.BALL_RADIUS
            case = sc.teleport(delta=delta, frame=frame, flight=flight, pitch_deg=pitch)
            assert not impacts_of(case), (level, flight, frame, pitch)
    assert no_kinematic_records(sc.teleport(delta=level * 0.2, flight=flight, pitch_deg=25.0))


@pytest.mark.parametrize("lengths", [1, 2, 3, 4, 5])
def test_t4_an_isolated_error_on_one_row_has_no_impact(lengths: int) -> None:
    clean = sc.parabola(pitch_deg=25.0)
    shift = clean.view.vector([0.0, lengths * 2 * sc.BALL_RADIUS, 0.0])
    for frame in range(sc.FRAMES):
        moved = clean.states.position[frame, 0] + shift
        assert not impacts_of(sc.with_values(clean, "position", "ball", [frame], moved)), frame


# --- T6 and J3: the pixel floor ----------------------------------------------------------------


def micro_kink(px: float = 1e-3) -> sc.Case:
    """A still ball whose position bends by ``px`` pixels per frame from frame 15.5: exact
    data, so its track noise sigma_w is 0."""
    case = sc.floating_box(pitch_deg=25.0)
    depth = float(case.states.position[0, 0, 2])
    drift = np.maximum(0.0, sc.times() - 15.5 * sc.DT) / sc.DT * px * depth / cs.FOCAL_PX
    moved = case.states.position[:, 0] + drift[:, None] * np.array([1.0, 0.0, 0.0])
    return sc.with_values(case, "position", "box", range(sc.FRAMES), moved)


def test_t6_a_kink_of_a_thousandth_of_a_pixel_is_below_the_floor() -> None:
    case = micro_kink()
    tiny = NoiseScale(np.full(3, 1e-12), known=True)
    unfloored = track_impacts(case.states.t, case.states.position[:, 0], tiny, 4, 0.05)
    assert unfloored, "without a floor the exact micro kink would be an impact"
    assert no_kinematic_records(case)


def test_t6_the_floor_is_nan_without_a_focal_or_a_depth() -> None:
    assert pixel_floor(4.0, 500.0) == pytest.approx(4.0 / (500.0 * math.sqrt(12.0)))
    for depth, focal in ((4.0, math.nan), (math.nan, 500.0), (-1.0, 500.0), (4.0, 0.0)):
        assert math.isnan(pixel_floor(depth, focal))
    assert noise_scale(np.zeros(3), math.nan) is None
    assert noise_scale(None, 1.0) is None
    floored = noise_scale(np.zeros(3), 1.0)
    assert floored is not None and floored.known
    measured = noise_scale(np.array([0.0, 2.0, 0.5]), 1.0)
    assert measured is not None and not measured.known
    np.testing.assert_array_equal(measured.sigma, [1.0, 2.0, 1.0])


def in_world(case: sc.Case) -> tuple[sc.Case, np.ndarray]:
    """The case's states moved to world_zup, and the camera pose that brings them back."""
    view = case.view
    states = case.states
    world = states.position @ view.rotation + view.origin
    moved = replace(
        states,
        frame=CoordFrame.WORLD_ZUP,
        position=world,
        velocity=states.velocity @ view.rotation,
        acceleration=states.acceleration @ view.rotation,
        gravity_dir=np.array([0.0, 0.0, -1.0]),
    )
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = view.rotation.T, view.origin
    return replace(case, states=moved), pose


def test_j3_world_states_use_the_camera_pose_or_get_no_floor() -> None:
    case, pose = in_world(sc.bounce(impact_frame=15.3, pitch_deg=25.0))
    with_pose = kinematic(cs.kinematic_table(case, camera=cs.scene_camera(case, pose=pose)))
    assert [(r.object_b, round(r.t_start, 6)) for r in with_pose] == [(ENV_ID, 15.3)]
    np.testing.assert_allclose(with_pose[0].normal, [0.0, 0.0, 1.0], atol=1e-12)
    assert not kinematic(cs.kinematic_table(case))


# --- T7: rows, frames and stencils ------------------------------------------------------------


@pytest.mark.parametrize("step", [2, 3])
def test_t7_a_strided_table_keeps_the_native_frame(step: int) -> None:
    case = sc.strided(sc.bounce(impact_frame=15.4, pitch_deg=25.0), step, quantize=False)
    records = kinematic(cs.kinematic_table(case))
    assert len(records) == 1 and abs(records[0].t_start - 15.4) <= 1e-3


@pytest.mark.parametrize("fps", [10.0, 60.0])
def test_t7_the_record_frame_is_the_time_over_the_frame_period(fps: float) -> None:
    case = sc.bounce(impact_frame=15.4, pitch_deg=25.0)
    states = replace(case.states, t=case.states.frame_index / fps)
    records = kinematic(cs.kinematic_table(case, states=states, seconds_per_frame=1.0 / fps))
    assert len(records) == 1 and abs(records[0].t_start - 15.4) <= 1e-3


def test_t7_fewer_than_eight_stencils_give_no_impact() -> None:
    rows = MIN_NOISE_STENCILS + 2  # third differences: rows - 3 = 7 stencils
    case = sc.bounce(impact_frame=5.4, frames=rows, pitch_deg=25.0)
    assert impacts_of(case), "the detector alone sees the kink"
    assert no_kinematic_records(case)
    longer = sc.bounce(impact_frame=5.4, frames=rows + 1, pitch_deg=25.0)
    assert not no_kinematic_records(longer)


# --- T8 and J7: attribution -------------------------------------------------------------------


def two_balls(
    dv_i: Any, dv_j: Any, gap_x: float = 0.0, speed: float = 1.0, frame: float = 15.3
) -> sc.Case:
    """Two balls at 1 m height, no acceleration, approaching at ``speed`` and touching at
    ``frame``, whose velocities change by ``dv_i`` and ``dv_j`` there; ``gap_x`` moves ball_j
    away along x."""
    zero, t, t_c = np.zeros(3), sc.times(), frame * sc.DT
    v_i, v_j = np.array([speed, 0.0, 0.0]), np.array([-speed, 0.0, 0.0])
    objects = []
    for name, v, dv, x in (("ball_i", v_i, dv_i, -0.1), ("ball_j", v_j, dv_j, 0.1 + gap_x)):
        p, vel, a = sc.piecewise_ballistic(t, t_c, (x, 0.0, 1.0), (v, zero), (v + dv, zero))
        kind = PrimitiveKind.SPHERE
        objects.append(sc.SceneObject(name, kind, sc.cube(0.2), p, vel, a, label="sphere"))
    return sc.build("two_balls", objects, [], 25.0)


def test_t8_a_downward_kick_in_the_air_has_no_surface() -> None:
    g = sc.STANDARD_GRAVITY * sc.DOWN
    t, t_c = sc.times(), 15.3 * sc.DT
    v_in, v_out = np.array([0.5, 0.0, -2.0]), np.array([0.5, 0.0, -5.0])
    p, v, a = sc.piecewise_ballistic(t, t_c, (0.0, 0.0, 2.0), (v_in, g), (v_out, g))
    ball = sc.SceneObject("ball", PrimitiveKind.SPHERE, sc.cube(0.2), p, v, a, label="sphere")
    case = sc.build("kick", [ball], [], 25.0)
    assert impacts_of(case), "the kink is found"
    assert no_kinematic_records(case)


def test_t8_impulses_of_the_same_sign_are_not_a_pair() -> None:
    """Still balls 5 cm apart (beyond the proximity tolerance, inside each other's image box)
    both kicked along +x and a little down: no pair, and no environment either."""
    case = two_balls([1.0, 0.0, -0.1], [1.0, 0.0, -0.1], gap_x=0.05, speed=0.0)
    assert impacts_of(case, 0) and impacts_of(case, 1)
    assert not [r for r in cs.kinematic_table(case, impacts="none").records if r.object_b != ENV_ID]
    assert no_kinematic_records(case)


def test_t8_boxes_apart_in_the_image_are_not_a_pair() -> None:
    case = two_balls([-2.0, 0.0, -0.2], [2.0, 0.0, -0.2], gap_x=3.0)
    assert no_kinematic_records(case)
    touching = two_balls([-2.0, 0.0, -0.2], [2.0, 0.0, -0.2])
    assert [(r.object_a, r.object_b) for r in kinematic(cs.kinematic_table(touching))] == [
        ("ball_i", "ball_j")
    ]


def test_t8_a_kinematic_impact_replaces_the_proximity_impact_of_the_pair() -> None:
    case = cs.collision_over_floor(gain=0.5, impact_frame=15.0)
    table = cs.kinematic_table(case)
    pair = [r for r in table.records if {r.object_a, r.object_b} == {"ball_i", "ball_j"}]
    proximity = [
        r
        for r in cs.kinematic_table(case, impacts="none").records
        if {r.object_a, r.object_b} == {"ball_i", "ball_j"}
    ]
    assert [r.kind for r in proximity] == [ContactKind.IMPACT], "proximity sees the touch"
    assert [(r.source, r.kind) for r in pair] == [(SOURCE, ContactKind.IMPACT)]


def test_j7_a_pair_consumes_both_impacts_so_no_environment_record() -> None:
    """ball_j alone would go to the environment (its kick points mostly up); the pair found
    from ball_i consumes it."""
    dv_i, dv_j = np.array([-2.0, 0.0, 0.5]), np.array([1.0, 0.0, 2.0])
    case = two_balls(dv_i, dv_j)
    up = -case.states.gravity_dir
    to_j = case.view.vector(dv_j) / np.linalg.norm(dv_j)
    normal_j = case.view.vector(dv_j - dv_i) / np.linalg.norm(dv_j - dv_i)
    assert float(to_j @ up) > float(to_j @ normal_j), "ball_j prefers the environment alone"
    records = kinematic(cs.kinematic_table(case))
    assert [(r.object_a, r.object_b) for r in records] == [("ball_i", "ball_j")]


# --- T10 (finding 10) and B --------------------------------------------------------------------


def test_t10_a_rocking_box_rests_once_unless_the_point_count_splits() -> None:
    case = cs.rocking_case()
    by_normal = cs.kinematic_table(case).records
    by_count = cs.kinematic_table(case, impacts="none", resting_split="normal_and_count").records
    assert [(r.kind, r.t_start, r.t_end) for r in by_normal] == [(ContactKind.RESTING, 0.0, 31.0)]
    assert len(by_count) == 16 and all(r.kind is ContactKind.IMPACT for r in by_count)


# --- T13: the cache key ------------------------------------------------------------------------


def test_t13_every_parameter_and_the_version_enter_the_key(fixture_clip: SyntheticClip) -> None:
    def key(provider: Any = KinematicContacts, **params: Any) -> str:
        return key_of(fixture_clip, provider, params=params, upstream=UPSTREAM)

    base = key()
    changed = [
        key(impacts="none"),
        key(resting_split="normal_and_count"),
        key(side_rows=5),
        key(alpha=0.01),
        key(versioned(KinematicContacts, "2")),
    ]
    assert base not in changed and len(set(changed)) == len(changed)


def test_t13_the_provider_declares_camera_and_kinematics_and_no_profile() -> None:
    assert KinematicContacts.requires == (Stage.CAMERA, Stage.KINEMATICS)
    assert KinematicContacts.error_profile_reads == () and not KinematicContacts.heavy
    assert KinematicContacts.version == "1"
