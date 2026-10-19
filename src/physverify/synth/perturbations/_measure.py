"""Ground-truth magnitudes of every check (R1..R5b) on a clip, built only from the frozen
measures in ``physics/measures``. Plugins reuse the per-object helpers for their ``measure``."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from physverify.physics.geometry import (
    Manifold,
    Plane,
    Solid,
    characteristic_length,
    contact_manifold,
    signed_distance,
    support_polygon,
)
from physverify.physics.measures.r1_penetration import penetration_ratio
from physverify.physics.measures.r2_support import overhang_fraction
from physverify.physics.measures.r3_collision import kinetic_energy_gain, restitution
from physverify.physics.measures.r4_integrity import jump_ratio, missing_frames, size_change_ratio
from physverify.physics.measures.r5_gravity import (
    FallFit,
    accel_inconsistency,
    fall_fit,
    gravity_rel_error,
)
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.scene import PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.perturbations._ballistics import impact_velocity, time_to_height
from physverify.synth.perturbations._windows import (
    UP,
    flight_mask,
    impact_frames,
    on_top_pairs,
    records_of,
    rest_mask,
    segments,
    solid_at,
)

Vec = NDArray[np.float64]
R5A_MIN_FRAMES = 6
R5B_MIN_FRAMES = 4
# A flight cut at a known gravity switch keeps pieces of at least this many frames on each side.
SIDE_FRAMES = 4
SWITCH_FRAME = "switch_frame"
MEASURE_KEYS = ("R1", "R2", "R3a", "R3b", "R4_missing", "R4_size", "R4_jump", "R5a", "R5b")


def measure_all(
    scene: SceneSpec,
    states: ObjectStateTable,
    contacts: ContactTable,
    tol: float,
    cuts: Sequence[int] = (),
) -> dict[str, float]:
    """Every check on the clip; ``cuts`` are frames where gravity is known to change, so R5 fits
    each side of them separately (the windowing of ``r5a_gravity_switch.measure``)."""
    g = np.asarray(scene.gravity, dtype=np.float64)
    masses = {obj.object_id: obj.mass for obj in scene.objects}
    return {
        "R1": max_penetration(states),
        "R2": max_overhang(states, contacts, tol),
        "R3a": max_restitution_excess(states, contacts, g),
        "R3b": max_energy_gain(states, contacts, masses, g),
        "R4_missing": _nanmax([float(_missing(states, i)) for i in range(states.n_objects)]),
        "R4_size": _nanmax([size_change_ratio(states.size[:, i]) for i in range(states.n_objects)]),
        "R4_jump": _nanmax([object_jump(states, contacts, i) for i in range(states.n_objects)]),
        "R5a": gravity_inconsistency(states, contacts, cuts),
        "R5b": gravity_scale_error(states, contacts, cuts),
    }


def record_cuts(record: PerturbationRecord) -> tuple[int, ...]:
    """Gravity-switch frames a perturbation record declares (none for most plugins)."""
    return (int(record.params[SWITCH_FRAME]),) if SWITCH_FRAME in record.params else ()


def _nanmax(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else math.nan


def nominal_extents(states: ObjectStateTable, index: int) -> Vec:
    out: Vec = np.nanmedian(states.size[:, index], axis=0)
    return out


def bounding_radius(solid: Solid) -> float:
    return float(np.linalg.norm(solid.primitive.extents)) / 2.0


def penetration_between(a: Solid, b: Solid) -> float:
    centres = float(np.linalg.norm(a.pose.position - b.pose.position))
    if centres > bounding_radius(a) + bounding_radius(b):
        return 0.0
    return penetration_ratio(
        signed_distance(a, b).distance, a.primitive.extents, b.primitive.extents
    )


def floor_penetration(solid: Solid) -> float:
    floor = Plane.ground()
    if float(floor.normal @ solid.pose.position) - floor.offset > bounding_radius(solid):
        return 0.0
    ext = solid.primitive.extents
    return penetration_ratio(signed_distance(solid, floor).distance, ext, ext)


def pair_penetration(states: ObjectStateTable, i: int, j: int) -> float:
    return max(
        penetration_between(solid_at(states, i, t), solid_at(states, j, t))
        for t in range(states.n_frames)
    )


def max_penetration(states: ObjectStateTable) -> float:
    """Largest penetration over object pairs and object-floor, divided by the smaller L."""
    values: list[float] = []
    for t in range(states.n_frames):
        solids = [solid_at(states, i, t) for i in range(states.n_objects)]
        for i, solid in enumerate(solids):
            values.append(floor_penetration(solid))
            values.extend(penetration_between(solid, other) for other in solids[i + 1 :])
    return _nanmax(values)


def support_manifolds(upper: Solid, lowers: Sequence[Solid | None], tol: float) -> list[Manifold]:
    """Manifolds of ``upper`` against each support (``None`` is the floor)."""
    manifolds = [contact_manifold(upper, Plane.ground() if s is None else s, tol) for s in lowers]
    return [m for m in manifolds if m is not None]


def overhang_on(upper: Solid, lowers: Sequence[Solid | None], tol: float) -> float:
    """``overhang_fraction`` of ``upper`` over the polygon of its manifolds (NaN if none)."""
    manifolds = support_manifolds(upper, lowers, tol)
    if not manifolds:
        return math.nan
    polygon = support_polygon(manifolds, -UP)
    return overhang_fraction(upper.pose.position, polygon, -UP, upper.primitive.extents)


def max_overhang(states: ObjectStateTable, contacts: ContactTable, tol: float) -> float:
    """R2 over bodies at rest in the middle frame that stand on at least one movable body."""
    mid = states.n_frames // 2
    ids = states.object_ids.astype(str).tolist()
    pairs = on_top_pairs(contacts)
    values: list[float] = []
    for i, object_id in enumerate(ids):
        lowers = sorted({low for up, low in pairs if up == object_id})
        if not lowers or all(low == ENV_ID for low in lowers) or not rest_mask(states, i)[mid]:
            continue
        solids = [
            None if low == ENV_ID else solid_at(states, ids.index(low), mid) for low in lowers
        ]
        values.append(overhang_on(solid_at(states, i, mid), solids, tol))
    return _nanmax(values)


def bounce_rows(states: ObjectStateTable, index: int, t_guess: float) -> list[int]:
    """Rows k near native frame ``t_guess`` where the sphere turns from falling (k) to rising
    (k + 1), in the order ``floor_bounce`` tries them."""
    vz = states.velocity[:, index, 2]
    base = math.floor(t_guess)
    return [
        k
        for k in (base, base - 1, base + 1)
        if 0 <= k < states.n_frames - 1 and vz[k] < 0.0 < vz[k + 1]
    ]


def floor_crossing(
    states: ObjectStateTable, index: int, k: int, radius: float, gz: float
) -> tuple[int, float] | None:
    """(source row, t_c in seconds) where the incoming arc reaches z = radius inside
    [t_src, t_{src+1}]: from row k, or from k - 1 when row k is already at or below the floor
    (its arc has no forward root). None when that arc does not cross there."""
    src = k - 1 if float(states.position[k, index, 2]) <= radius else k
    if src < 0:
        return None
    t_src = float(states.t[src])
    z, vz = float(states.position[src, index, 2]), float(states.velocity[src, index, 2])
    tau = time_to_height(z, vz, gz, radius)
    if not math.isfinite(tau) or t_src + tau > float(states.t[src + 1]) + 1e-12:
        return None
    return src, t_src + tau


def bounce_crossing(
    states: ObjectStateTable, index: int, t_guess: float, g: Vec
) -> tuple[int, int, float] | None:
    """(k, source row, t_c) of the first row of ``bounce_rows`` whose incoming arc crosses the
    floor (``floor_crossing``)."""
    radius = float(nominal_extents(states, index)[0]) / 2.0
    for k in bounce_rows(states, index, t_guess):
        crossing = floor_crossing(states, index, k, radius, float(g[2]))
        if crossing is not None:
            return k, crossing[0], crossing[1]
    return None


def floor_bounce(
    states: ObjectStateTable, index: int, t_guess: float, g: Vec
) -> tuple[float, Vec, Vec] | None:
    """(t_c in seconds, v-, v+) of a sphere's floor bounce near native frame ``t_guess``.

    t_c is where the incoming ballistic arc crosses z = radius (the convention of check R3a), so
    it does not depend on the event timing; v- is extrapolated to t_c from the arc's source row
    and v+ from row k + 1 (ADR 0004, WP-H amendment)."""
    found = bounce_crossing(states, index, t_guess, g)
    if found is None:
        return None
    k, src, t_c = found
    v_minus = impact_velocity(states.velocity[src, index], float(states.t[src]), t_c, g)
    v_plus = impact_velocity(states.velocity[k + 1, index], float(states.t[k + 1]), t_c, g)
    return t_c, v_minus, v_plus


def bounce_restitution(states: ObjectStateTable, index: int, t_guess: float, g: Vec) -> float:
    bounce = floor_bounce(states, index, t_guess, g)
    return math.nan if bounce is None else restitution(bounce[1], bounce[2], UP)


def max_restitution_excess(states: ObjectStateTable, contacts: ContactTable, g: Vec) -> float:
    ids = states.object_ids.astype(str).tolist()
    values = [
        bounce_restitution(states, ids.index(r.object_a), r.t_start, g) - 1.0
        for r in contacts.records
        if r.kind is ContactKind.IMPACT
        and r.object_b == ENV_ID
        and str(states.primitive[ids.index(r.object_a)]) == PrimitiveKind.SPHERE.value
    ]
    return _nanmax(values)


def grounded(object_id: str, contacts: ContactTable, frame: int) -> bool:
    return any(
        r.object_b == ENV_ID and math.floor(r.t_start) - 1 <= frame <= math.ceil(r.t_end) + 1
        for r in records_of(object_id, contacts)
    )


def collision_velocities(
    states: ObjectStateTable, contacts: ContactTable, index: int, t_c: float, g: Vec
) -> tuple[Vec, Vec]:
    """v- and v+ at t_c (s): ballistic extrapolation in flight, the frame value on the floor."""
    fps = 1.0 / float(states.t[1] - states.t[0])
    k = min(max(0, math.floor(t_c * fps - 1e-9)), states.n_frames - 2)
    object_id = str(states.object_ids[index])
    out: list[Vec] = []
    for frame in (k, k + 1):
        v = states.velocity[frame, index]
        airborne = not grounded(object_id, contacts, frame)
        out.append(impact_velocity(v, float(states.t[frame]), t_c, g) if airborne else v.copy())
    return out[0], out[1]


def pair_energy_gain(
    states: ObjectStateTable,
    contacts: ContactTable,
    pair: tuple[int, int],
    mass_ratio: float,
    t_c: float,
    g: Vec,
) -> float:
    va_minus, va_plus = collision_velocities(states, contacts, pair[0], t_c, g)
    vb_minus, vb_plus = collision_velocities(states, contacts, pair[1], t_c, g)
    return kinetic_energy_gain(mass_ratio, va_minus, vb_minus, va_plus, vb_plus)


def max_energy_gain(
    states: ObjectStateTable, contacts: ContactTable, masses: dict[str, float], g: Vec
) -> float:
    ids = states.object_ids.astype(str).tolist()
    fps = 1.0 / float(states.t[1] - states.t[0])
    values = [
        pair_energy_gain(
            states,
            contacts,
            (ids.index(r.object_a), ids.index(r.object_b)),
            masses[r.object_a] / masses[r.object_b],
            r.t_start / fps,
            g,
        )
        for r in contacts.records
        if r.kind is ContactKind.IMPACT and r.object_b != ENV_ID
    ]
    return _nanmax(values)


def _missing(states: ObjectStateTable, index: int) -> int:
    return missing_frames(
        states.visible[:, index], states.in_frame[:, index], states.occlusion[:, index]
    )


def object_missing(states: ObjectStateTable, index: int) -> float:
    return float(_missing(states, index))


def object_jump(states: ObjectStateTable, contacts: ContactTable, index: int) -> float:
    exclude = impact_frames(str(states.object_ids[index]), contacts, states.n_frames)
    return jump_ratio(states.position[:, index], nominal_extents(states, index), exclude)


def fit_frames(states: ObjectStateTable, index: int, frames: NDArray[np.int64]) -> FallFit:
    return fall_fit(states.t[frames], states.position[frames, index])


def flight_fits(
    states: ObjectStateTable,
    contacts: ContactTable,
    index: int,
    min_len: int,
    cuts: Sequence[int] = (),
) -> list[FallFit]:
    mask = flight_mask(str(states.object_ids[index]), contacts, states.n_frames)
    mask &= ~rest_mask(states, index)
    return [
        fit_frames(states, index, np.arange(lo, hi))
        for segment in segments(mask, 1)
        for lo, hi in cut_segment(segment, min_len, cuts)
    ]


def cut_segment(
    segment: tuple[int, int], min_len: int, cuts: Sequence[int]
) -> list[tuple[int, int]]:
    """``segment`` split at the cuts inside it; uncut segments need ``min_len`` frames and the
    pieces of a cut one ``SIDE_FRAMES``."""
    lo, hi = segment
    inner = sorted(c for c in set(cuts) if lo < c < hi)
    if not inner:
        return [segment] if hi - lo >= min_len else []
    bounds = [lo, *inner, hi]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b - a >= SIDE_FRAMES]


def inconsistency(fits: Sequence[FallFit]) -> float:
    return accel_inconsistency([f.a for f in fits]) if len(fits) >= 2 else math.nan


def gravity_inconsistency(
    states: ObjectStateTable, contacts: ContactTable, cuts: Sequence[int] = ()
) -> float:
    fits = [
        fit
        for i in range(states.n_objects)
        for fit in flight_fits(states, contacts, i, R5A_MIN_FRAMES, cuts)
    ]
    return inconsistency(fits)


def gravity_scale_error(
    states: ObjectStateTable, contacts: ContactTable, cuts: Sequence[int] = ()
) -> float:
    """Median over sphere flights of | |a_fit| - g | / g in metres (R5b)."""
    errors = [
        gravity_rel_error(float(np.linalg.norm(fit.a)), 1.0)
        for i in range(states.n_objects)
        if str(states.primitive[i]) == PrimitiveKind.SPHERE.value
        for fit in flight_fits(states, contacts, i, R5B_MIN_FRAMES, cuts)
    ]
    return float(np.median(errors)) if errors else math.nan


def length_of(states: ObjectStateTable, index: int) -> float:
    return characteristic_length(nominal_extents(states, index))
