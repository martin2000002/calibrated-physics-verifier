"""Image-plane impulse events between two tracked bodies, and the true object-object impacts read
as events, for R3b image_relspeed.

The relative image position d(t) = u_a - u_b (u = (X/Z, Y/Z) of the lifted centre, which lies on
its pixel ray, so no object depth enters) is close to a line in free flight: the shared gravity
cancels in the difference. A collision is a kink of d: a line fitted on each side of a candidate
row k, the jump of the slopes significant against the pooled residual scatter, the bodies
approaching before and separating after along the jump, the lines crossing within ``cross_rows``
rows of k, and the centres within ``kappa`` summed apparent radii there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.physics.tracks import contact_window, row_step
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.states import CoordFrame, ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

EVENT_SOURCE = "image_impulse"


@dataclass(frozen=True)
class Gates:
    """Side lengths in rows and the detector's significance and geometry gates; ``radius_rows``
    is the half window of ``apparent_radius``."""

    side_min: int
    side_max: int
    z_min: float
    speed_z_min: float
    kappa: float
    cross_rows: float
    merge_rows: int
    radius_rows: int


@dataclass(frozen=True)
class Event:
    """An object-object impact at fractional native frame ``t``, pair sorted; ``normal`` (b to a,
    camera frame) and ``point`` only for true impacts."""

    a: str
    b: str
    t: float
    z: float = math.inf
    normal: FloatArray | None = None
    point: FloatArray | None = None


@dataclass(frozen=True)
class LineFit:
    intercept: FloatArray
    slope: FloatArray
    rss: float
    sxx: float
    n: int


@dataclass(frozen=True)
class _Candidate:
    row: int
    t: float
    z: float


def image_track(states: ObjectStateTable, index: int) -> FloatArray:
    """(rows, 2) normalized image coordinates; NaN where the centre is unknown or behind."""
    if states.frame is not CoordFrame.CAMERA_OPENCV:
        raise ValueError(f"image track needs camera-frame states, got {states.frame}")
    p = np.asarray(states.position[:, index], dtype=np.float64)
    out = np.full((p.shape[0], 2), np.nan)
    ok = np.all(np.isfinite(p), axis=1) & (p[:, 2] > 0.0)
    out[ok] = p[ok, :2] / p[ok, 2:3]
    return out


def env_records(contacts: ContactTable | None) -> list[ContactRecord]:
    return [] if contacts is None else [r for r in contacts.records if r.object_b == ENV_ID]


def free_rows(
    frame_index: IntArray, records: Sequence[ContactRecord], object_id: str, margin: int
) -> BoolArray:
    """Rows outside every window (``contact_window`` with ``margin``) of the object's records."""
    fi = np.asarray(frame_index, dtype=np.int64)
    step = row_step(fi)
    busy = np.zeros(fi.shape, dtype=np.bool_)
    for r in records:
        if object_id in (r.object_a, r.object_b):
            lo, hi = contact_window(r, step, margin)
            busy |= (fi >= lo) & (fi <= hi)
    return np.asarray(~busy, dtype=np.bool_)


def walk(ok: BoolArray, start: int, step: int, limit: int) -> list[int]:
    """Rows from ``start`` outwards while ``ok``, at most ``limit``, ascending."""
    rows: list[int] = []
    row = start
    while 0 <= row < ok.size and ok[row] and len(rows) < limit:
        rows.append(row)
        row += step
    return sorted(rows)


def line_fit(t: FloatArray, y: FloatArray, t0: float) -> LineFit:
    """Least-squares line y = intercept + slope (t - t0), y of shape (n, 2).

    Not ``physics.tracks.poly_fit`` of degree 1: that one solves the normal equations about
    mean(t) and its slopes differ from these closed-form ones in the last bits (most random
    fits), which would move the probe's outputs; it also gives no rss or sxx."""
    x = np.asarray(t, dtype=np.float64) - t0
    xm = x.mean()
    sxx = float(((x - xm) ** 2).sum())
    slope = ((x - xm)[:, None] * (y - y.mean(axis=0))).sum(axis=0) / sxx
    intercept = y.mean(axis=0) - slope * xm
    resid = y - (intercept + slope * x[:, None])
    return LineFit(intercept, slope, float((resid**2).sum()), sxx, int(x.size))


def apparent_radius(states: ObjectStateTable, index: int, row: int, half: int) -> float:
    """Median over rows row-half..row+half of a quarter of (size_x + size_y) / Z (a learned size
    on a perceived lift: only a proximity gate)."""
    lo, hi = max(0, row - half), min(states.n_frames, row + half + 1)
    size = np.asarray(states.size[lo:hi, index], dtype=np.float64)
    z = np.asarray(states.position[lo:hi, index, 2], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 0.25 * (size[:, 0] + size[:, 1]) / z
    r = r[np.isfinite(r) & (z > 0)]
    return float(np.median(r)) if r.size else math.nan


def _candidate(
    states: ObjectStateTable,
    d: FloatArray,
    ok: BoolArray,
    k: int,
    pair: tuple[int, int],
    gates: Gates,
) -> _Candidate | None:
    fi = np.asarray(states.frame_index, dtype=np.float64)
    left, right = walk(ok, k - 1, -1, gates.side_max), walk(ok, k + 1, 1, gates.side_max)
    if len(left) < gates.side_min or len(right) < gates.side_min:
        return None
    t0 = float(fi[k])
    fl, fr = line_fit(fi[left], d[left], t0), line_fit(fi[right], d[right], t0)
    s2 = (fl.rss + fr.rss) / (2 * (fl.n - 2) + 2 * (fr.n - 2))
    jump = fr.slope - fl.slope
    size = float(np.linalg.norm(jump))
    if size == 0.0:
        return None
    n = jump / size
    se_l, se_r = math.sqrt(s2 / fl.sxx), math.sqrt(s2 / fr.sxx)
    se = math.hypot(se_l, se_r)
    z = math.inf if se == 0.0 else size / se
    approach, separation = -float(fl.slope @ n), float(fr.slope @ n)
    if z < gates.z_min or approach < gates.speed_z_min * se_l:
        return None
    if separation < gates.speed_z_min * se_r or approach <= 0.0 or separation <= 0.0:
        return None
    tau = float((fl.intercept - fr.intercept) @ n) / size
    if abs(tau) > gates.cross_rows * row_step(states.frame_index):
        return None
    de = 0.5 * (fl.intercept + fl.slope * tau + fr.intercept + fr.slope * tau)
    if not float(de @ fl.slope) < 0.0 < float(de @ fr.slope):
        return None
    half = gates.radius_rows
    radii = apparent_radius(states, pair[0], k, half) + apparent_radius(states, pair[1], k, half)
    if not math.isfinite(radii) or float(np.linalg.norm(de)) > gates.kappa * radii:
        return None
    return _Candidate(k, t0 + tau, z)


def _merged(found: Sequence[_Candidate], merge_rows: int) -> list[_Candidate]:
    """One candidate per run of rows with gaps <= ``merge_rows``: the largest z."""
    runs: list[list[_Candidate]] = []
    for c in sorted(found, key=lambda c: c.row):
        if runs and c.row - runs[-1][-1].row <= merge_rows:
            runs[-1].append(c)
        else:
            runs.append([c])
    return [max(run, key=lambda c: c.z) for run in runs]


def detect(states: ObjectStateTable, contacts: ContactTable | None, gates: Gates) -> list[Event]:
    """Image impulse events of every pair of tracked bodies, outside their ENV records."""
    env = env_records(contacts)
    ids = [str(o) for o in states.object_ids]
    tracks = [image_track(states, i) for i in range(states.n_objects)]
    free = [free_rows(states.frame_index, env, o, 0) for o in ids]
    out: list[Event] = []
    for i in range(states.n_objects):
        for j in range(i + 1, states.n_objects):
            d = tracks[i] - tracks[j]
            ok = np.all(np.isfinite(d), axis=1) & free[i] & free[j]
            found = [
                c
                for k in np.flatnonzero(ok)
                if (c := _candidate(states, d, ok, int(k), (i, j), gates)) is not None
            ]
            a, b = sorted((ids[i], ids[j]))
            out += [Event(a, b, c.t, c.z) for c in _merged(found, gates.merge_rows)]
    return sorted(out, key=lambda e: (e.t, e.a, e.b))


def truth_events(contacts: ContactTable | None) -> list[Event]:
    """Object-object IMPACT records as events (t = middle of the record's interval); the normal
    is flipped with the pair so it always points from b to a."""
    if contacts is None:
        return []
    out = []
    for r in contacts.records:
        if r.kind is not ContactKind.IMPACT or r.object_b == ENV_ID:
            continue
        a, b = sorted((r.object_a, r.object_b))
        sign = 1.0 if a == r.object_a else -1.0
        normal = sign * np.asarray(r.normal, dtype=np.float64)
        point = np.asarray(r.points, dtype=np.float64).mean(axis=0)
        out.append(Event(a, b, 0.5 * (r.t_start + r.t_end), math.inf, normal, point))
    return sorted(out, key=lambda e: (e.t, e.a, e.b))


def as_record(e: Event) -> ContactRecord:
    """An event as an IMPACT record, so the windows and flight masks of the checks read it."""
    normal = np.array([1.0, 0.0, 0.0]) if e.normal is None else np.asarray(e.normal)
    point = np.zeros((1, 3)) if e.point is None else np.asarray(e.point).reshape(1, 3)
    return ContactRecord(
        object_a=e.a,
        object_b=e.b,
        t_start=e.t,
        t_end=e.t,
        kind=ContactKind.IMPACT,
        normal=normal,
        points=point,
        manifold=ManifoldKind.POINT,
        confidence=1.0,
        source=EVENT_SOURCE,
    )
