import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.physics.geometry import signed_distance
from physverify.physics.tracks import (
    body_length,
    check_time_axis,
    contact_rows,
    flight_mask,
    impact_stencil_mask,
    line_fit,
    native_span,
    place_body,
    rest_mask,
    row_step,
    segments,
    split_at_jumps,
    support_plane,
)
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.states import CoordFrame, LengthUnit
from physverify.synth.perturbations._windows import impact_frames

NATIVE = np.arange(12, dtype=np.int64)
EVEN = np.arange(0, 24, 2, dtype=np.int64)
YAW_AXIS = np.array([1.0, 0.0, 0.0])


def record(t_start: float, t_end: float, kind: ContactKind = ContactKind.IMPACT) -> ContactRecord:
    return ContactRecord(
        object_a="ball",
        object_b=ENV_ID,
        t_start=t_start,
        t_end=t_end,
        kind=kind,
        normal=np.array([0.0, 0.0, 1.0]),
        points=np.zeros((1, 3)),
        manifold=ManifoldKind.POINT,
    )


def table(*records: ContactRecord) -> ContactTable:
    return ContactTable(
        clip_id="c",
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.METER,
        source_key="gt:c",
        records=records,
    )


def frames_of(mask: np.ndarray, frame_index: np.ndarray) -> set[int]:
    return {int(f) for f in frame_index[mask]}


# --- steps and time axis ---------------------------------------------------------------------


def test_row_step() -> None:
    assert row_step(NATIVE) == 1
    assert row_step(EVEN) == 2
    assert row_step(np.array([7], dtype=np.int64)) == 1
    for bad in ([0, 2, 3], [4, 4], [3, 1]):
        with pytest.raises(ValueError):
            row_step(np.array(bad, dtype=np.int64))


def test_time_axis_uses_native_frames() -> None:
    check_time_axis(sc.times(24)[::2], EVEN, sc.DT)
    with pytest.raises(ValueError):
        check_time_axis(sc.times(EVEN.size), EVEN, sc.DT)


# --- contact windows -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frame_index", "bounds", "margin", "expected"),
    [
        (NATIVE, (5.4, 5.4), 0, {5, 6}),
        (NATIVE, (5.0, 7.0), 1, {4, 5, 6, 7, 8}),
        (EVEN, (6.0, 6.0), 0, {6}),
        (EVEN, (6.0, 6.0), 1, {4, 6, 8}),
    ],
)
def test_contact_rows(
    frame_index: np.ndarray, bounds: tuple[float, float], margin: int, expected: set[int]
) -> None:
    rows = contact_rows(record(*bounds), frame_index, margin)
    assert frames_of(rows, frame_index) == expected


@pytest.mark.parametrize(
    ("frame_index", "bounds", "expected"),
    [
        (NATIVE, (5.4, 5.4), {4, 5, 6}),
        (NATIVE, (5.0, 5.0), {3, 4, 5, 6}),
        (EVEN, (6.0, 6.0), {2, 4, 6, 8}),
    ],
)
def test_impact_stencil_mask(
    frame_index: np.ndarray, bounds: tuple[float, float], expected: set[int]
) -> None:
    contacts = table(record(*bounds), record(0.0, 11.0, ContactKind.RESTING))
    mask = impact_stencil_mask("ball", contacts, frame_index)
    assert frames_of(mask, frame_index) == expected


@pytest.mark.parametrize("n_rows", [1, 2, 3])
def test_stencil_mask_virtual_neighbours_reach_two_steps_past_the_end(n_rows: int) -> None:
    frame_index = np.arange(5, 5 + n_rows, dtype=np.int64)
    last = int(frame_index[-1])
    reached = impact_stencil_mask("ball", table(record(last + 2.0, last + 2.0)), frame_index)
    missed = impact_stencil_mask("ball", table(record(last + 2.5, last + 2.5)), frame_index)
    assert bool(reached[-1])
    assert not bool(np.any(missed))


def a4_flight(records: list[ContactRecord], n: int, margin: int) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    for r in records:
        lo, hi = max(0, math.floor(r.t_start) - margin), min(n - 1, math.ceil(r.t_end) + margin)
        if lo <= hi:
            mask[lo : hi + 1] = False
    return mask


def a4_impact_frames(records: list[ContactRecord], n: int) -> np.ndarray:
    """A4's rows floor(t_start) - 1 <= k <= ceil(t_end) of the 4-row stencil, one row wider on
    each bound that falls on a frame (quantized)."""
    mask = np.zeros(n, dtype=bool)
    for r in records:
        lo = math.floor(r.t_start) - 1 - int(r.t_start.is_integer())
        hi = math.ceil(r.t_end) + int(r.t_end.is_integer())
        lo, hi = max(0, lo), min(n - 1, hi)
        if r.kind is ContactKind.IMPACT and lo <= hi:
            mask[lo : hi + 1] = True
    return mask


def random_records(rng: np.random.Generator, n: int, count: int) -> list[ContactRecord]:
    out = []
    for _ in range(count):
        start = float(rng.uniform(-2.0, n + 1.0))
        if rng.random() < 0.3:
            start = float(round(start))
        length = float(rng.choice([0.0, rng.uniform(0.0, 4.0), float(rng.integers(0, 4))]))
        kind = ContactKind.IMPACT if rng.random() < 0.6 else ContactKind.RESTING
        out.append(record(start, start + length, kind))
    return out


def test_native_windows_match_a4_formulas() -> None:
    rng = np.random.default_rng(0)
    n = 40
    frame_index = np.arange(n, dtype=np.int64)
    records = random_records(rng, n, 50)
    for margin in (0, 1, 2):
        for r in records:
            single = table(r)
            np.testing.assert_array_equal(
                flight_mask("ball", single, frame_index, margin), a4_flight([r], n, margin)
            )
        np.testing.assert_array_equal(
            flight_mask("ball", table(*records), frame_index, margin),
            a4_flight(records, n, margin),
        )
    for r in records:
        np.testing.assert_array_equal(
            impact_stencil_mask("ball", table(r), frame_index), a4_impact_frames([r], n)
        )
    np.testing.assert_array_equal(
        impact_stencil_mask("ball", table(*records), frame_index), a4_impact_frames(records, n)
    )


def test_native_stencil_mask_is_a4_impact_frames_on_fractional_bounds() -> None:
    rng = np.random.default_rng(1)
    n = 40
    frame_index = np.arange(n, dtype=np.int64)
    records = [
        r
        for r in random_records(rng, n, 80)
        if not (r.t_start.is_integer() or r.t_end.is_integer())
    ]
    assert len(records) > 20
    for r in records:
        np.testing.assert_array_equal(
            impact_stencil_mask("ball", table(r), frame_index), impact_frames("ball", table(r), n)
        )


def test_flight_mask_ignores_other_objects() -> None:
    other = ContactRecord(
        object_a="box",
        object_b=ENV_ID,
        t_start=3.0,
        t_end=4.0,
        kind=ContactKind.RESTING,
        normal=np.array([0.0, 0.0, 1.0]),
        points=np.zeros((1, 3)),
        manifold=ManifoldKind.POINT,
    )
    assert flight_mask("ball", table(other), NATIVE).all()


def test_segments_and_native_span() -> None:
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert segments(mask, 1) == [(0, 2), (3, 6), (7, 8)]
    assert segments(mask, 3) == [(3, 6)]
    assert native_span(EVEN, 1, 4) == (2, 8)


# --- per-object helpers ----------------------------------------------------------------------


@pytest.mark.parametrize(("step", "expected"), [(1, [(0, 16), (16, 32)]), (2, [(0, 8), (8, 16)])])
def test_split_at_jumps_cuts_at_the_teleport(step: int, expected: list[tuple[int, int]]) -> None:
    case = sc.strided(sc.teleport(delta=0.3, frame=16, flight=True), step, quantize=False)
    rows = case.states.frame_index.size
    parts = split_at_jumps(case.states, 0, (0, rows), gate=3.0)
    assert parts == expected
    assert int(case.states.frame_index[parts[1][0]]) == 16


def test_split_at_jumps_keeps_a_clean_parabola() -> None:
    states = sc.parabola().states
    assert split_at_jumps(states, 0, (0, 32), gate=3.0) == [(0, 32)]
    assert split_at_jumps(states, 0, (4, 6), gate=3.0) == [(4, 6)]


def test_rest_mask_and_body_length() -> None:
    rolling = sc.rolling_ball(speed=0.5).states
    assert body_length(rolling, 0) == pytest.approx(0.2)
    assert not rest_mask(rolling, 0, rest_speed=0.1).any()
    assert rest_mask(rolling, 0, rest_speed=3.0).all()
    assert rest_mask(sc.stack().states, 1, rest_speed=0.1).all()


def test_place_body_levels_of_knowledge() -> None:
    case = sc.stack(pitch_deg=30.0)
    placed = place_body(case.states, 0, 1, YAW_AXIS)
    assert placed is not None and not placed.assumed and not placed.point_only
    base = place_body(case.states, 0, 0, YAW_AXIS)
    assert base is not None
    assert signed_distance(placed.solid, base.solid).distance == pytest.approx(0.0, abs=1e-9)

    no_pose = sc.with_values(case, "quaternion", "top", range(32), np.nan)
    upright = place_body(no_pose.states, 0, 1, YAW_AXIS)
    assert upright is not None and upright.assumed and not upright.point_only

    blind = replace(no_pose.states, gravity_dir=np.full(3, np.nan))
    no_gravity = replace(no_pose, states=blind)
    point = place_body(no_gravity.states, 0, 1, YAW_AXIS)
    assert point is not None and point.point_only

    hidden = sc.with_values(case, "visible", "top", [0], False)
    assert place_body(hidden.states, 0, 1, YAW_AXIS) is None


def test_support_plane_is_the_ground_in_camera_axes() -> None:
    case = sc.sinking_box(depth=0.05, pitch_deg=30.0)
    assert case.contacts is not None
    plane = support_plane("box", case.contacts)
    assert plane is not None
    np.testing.assert_allclose(plane.normal, -case.states.gravity_dir, atol=1e-12)
    assert plane.offset == pytest.approx(float(plane.normal @ case.view.point(np.zeros(3))))
    box = place_body(case.states, 0, 0, YAW_AXIS)
    assert box is not None
    assert signed_distance(box.solid, plane).distance == pytest.approx(-0.05)
    spheres = sc.overlapping_spheres()
    assert spheres.contacts is not None
    assert support_plane("ball_a", spheres.contacts) is None


def test_line_fit_recovers_slope_on_spaced_times() -> None:
    t = sc.times(30)[::3]
    y = np.asarray(np.array([1.0, -2.0, 0.5]) + np.outer(t - 0.3, [3.0, 0.25, -9.81]))
    fit = line_fit(t, y, np.full(y.shape, 0.01))
    np.testing.assert_allclose(fit.slope, [3.0, 0.25, -9.81], atol=1e-12)
    value, sigma = fit.at(0.3)
    np.testing.assert_allclose(value, [1.0, -2.0, 0.5], atol=1e-12)
    spread = float(np.sum((t - t.mean()) ** 2))
    assert fit.cov is not None
    np.testing.assert_allclose(fit.cov[:, 1, 1], 0.01**2 / spread)
    assert np.all(sigma > 0)


def test_line_fit_sigma_edge_cases() -> None:
    t = sc.times(5)
    y = np.asarray(2.0 - 4.0 * t, dtype=np.float64)
    exact = line_fit(t, y, np.zeros(5))
    assert exact.at(1.0)[1] == 0.0
    unknown = line_fit(t, y, np.full(5, np.nan))
    assert unknown.cov is None
    assert float(unknown.slope) == pytest.approx(-4.0)
    assert math.isnan(float(unknown.at(0.0)[1]))
    with pytest.raises(ValueError):
        line_fit(t[:1], y[:1], np.zeros(1))


# --- scenarios -------------------------------------------------------------------------------


@pytest.mark.parametrize("step", [1, 2, 3])
def test_scenarios_keep_t_on_native_frames(step: int) -> None:
    for build in (sc.stack, sc.bounce, sc.collision, sc.parabola, sc.disappearance):
        case = sc.strided(build(), step, quantize=True)
        assert row_step(case.states.frame_index) == step
        check_time_axis(case.states.t, case.states.frame_index, case.clip.seconds_per_frame)


def test_quantized_contacts_snap_to_sampled_frames() -> None:
    native = sc.bounce(impact_frame=15.0)
    snapped = sc.strided(native, 2, quantize=True)
    kept = sc.strided(native, 2, quantize=False)
    assert snapped.contacts is not None and kept.contacts is not None
    (r,) = snapped.contacts.records
    assert (r.t_start, r.t_end) == (14.0, 16.0)
    assert kept.contacts.records[0].t_start == 15.0


def test_bounce_and_collision_are_consistent() -> None:
    case = sc.bounce(e=0.8, pitch_deg=30.0)
    up = -case.states.gravity_dir
    v = case.states.velocity[:, 0] @ up
    np.testing.assert_allclose(v[15] - 9.81 * 0.4 * sc.DT, -5.0)
    np.testing.assert_allclose(v[16] + 9.81 * 0.6 * sc.DT, 4.0)
    assert np.all(
        case.states.position[:, 0] @ up - up @ case.view.point(np.zeros(3)) >= 0.1 - 1e-12
    )

    r = 3.0
    hit = sc.collision(mass_ratio=r, gain=0.2)
    before, after = hit.states.velocity[0], hit.states.velocity[-1]
    np.testing.assert_allclose(r * before[0] + before[1], r * after[0] + after[1], atol=1e-12)
    energy = [r * v[0] @ v[0] + v[1] @ v[1] for v in (before, after)]
    assert energy[1] / energy[0] - 1 == pytest.approx(0.2)


@pytest.mark.parametrize("yaw_deg", [0.0, 30.0, 45.0])
def test_overhang_stays_in_contact_when_turned(yaw_deg: float) -> None:
    case = sc.overhang(0.25, yaw_deg=yaw_deg)
    assert case.contacts is not None
    top_on_base = case.contacts.records[1]
    assert (top_on_base.object_a, top_on_base.object_b) == ("top", "base")
    assert top_on_base.points.shape[0] >= 3


def test_integrity_scenes() -> None:
    gone = sc.disappearance(first=4, last=7)
    assert set(np.flatnonzero(~gone.states.visible[:, 0])) == {4, 5, 6, 7}
    grown = sc.size_change(factor=1.3, frame=16)
    np.testing.assert_allclose(grown.states.size[[0, 31], 0, 0], [0.2, 0.26])
    jump = sc.teleport(delta=0.3, frame=16)
    step = np.linalg.norm(np.diff(jump.states.position[:, 0], axis=0), axis=-1)
    assert int(np.argmax(step)) == 15
