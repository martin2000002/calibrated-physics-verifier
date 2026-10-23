"""Kinematic impacts of one track; not a plugin.

A contact impulse changes a body's velocity at one instant and never its position, and between
impacts the only force is constant (gravity, a constant friction), so a track is a parabola with
slope breaks. Around every boundary between two consecutive rows a parabola is compared with the
parabola plus one column: a velocity kink inside that boundary (an impact), or, as alternatives,
a position step, an isolated outlier or an acceleration change anywhere in the window. The
boundary holds an impact when the kink explains the window best (a kink in another interval of
the window is one more alternative: it localizes the impulse) and its gain over the parabola
exceeds the Bonferroni quantile at 1 - alpha / n_boundaries: chi-square(3) when the noise scale is
the pixel floor (known), an F(3, residual dof) ratio over the window's residual when it was
measured on the track (``NoiseScale``). The bound is nominal: the kink time is optimized and the
best model selected, so under the null the statistic does not follow that law (Davies); the real
false-alarm rate is measured by the tests, never asserted from alpha.

Every model is fitted on positions whitened per axis by the track's noise. Adding a column c to
the parabola P lowers the residual sum of squares by gain(c) = sum_k (r_k . c~)^2 / |c~|^2, with
r_k the parabola residual of axis k and c~ the part of c orthogonal to P; for the kink column
(u - s)_+ inside the boundary the best s solves a quadratic in closed form.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from scipy.stats import f as f_dist

from physverify.physics.kinematics import visible_runs
from physverify.physics.tracks import NoiseScale

FloatArray = NDArray[np.float64]
EDGE_SIDE_ROWS = 3
"""Fewest rows on one side of a boundary next to the end of a finite run."""
LOCAL_ROWS = 3
"""One impulse per neighbourhood: an accepted boundary must hold the largest gain this close."""
DEGREES_OF_FREEDOM = 3
"""One extra coefficient per axis."""
KINK_COEFFICIENTS = 4
"""Parabola plus kink, per axis."""
TIE_RTOL = 1e-9
"""Relative slack under which two kink gains on the same row count as equal."""
GRID = 9
"""Points per interval of the coarse search for an acceleration change, refined at the best."""


@dataclass(frozen=True)
class Impact:
    """A velocity kink of one track between rows ``row`` and ``row + 1`` at ``time`` seconds."""

    row: int
    time: float
    delta_v: FloatArray
    position: FloatArray
    score: float


@dataclass(frozen=True)
class Window:
    """Rows ``lo..hi-1`` of a track around the boundary after row ``row``, whitened per axis.

    ``u`` is the time in row steps from that row (the boundary is the interval [0, 1])."""

    row: int
    lo: int
    u: FloatArray
    y: FloatArray
    step: float
    origin: float


def parabola(u: FloatArray) -> FloatArray:
    return np.column_stack([np.ones_like(u), u, u * u])


def residual_maker(u: FloatArray) -> FloatArray:
    """I - P (P^T P)^-1 P^T for the parabola P over the times ``u``."""
    basis = parabola(u)
    projector = basis @ np.linalg.solve(basis.T @ basis, basis.T)
    return np.eye(u.size) - projector


def accel_column(u: FloatArray, s: float) -> FloatArray:
    return np.asarray(0.5 * np.maximum(0.0, u - s) ** 2, dtype=np.float64)


@dataclass(frozen=True)
class Columns:
    """Alternative columns made orthogonal to the parabola, with their squared norms."""

    orthogonal: FloatArray
    norms: FloatArray

    @classmethod
    def of(cls, columns: FloatArray, maker: FloatArray) -> Columns:
        orthogonal = columns @ maker
        return cls(orthogonal, np.einsum("km,km->k", orthogonal, orthogonal))

    def gains(self, residual: FloatArray) -> FloatArray:
        """RSS drop of the parabola when each column is added (0 for a degenerate column)."""
        projections = self.orthogonal @ residual
        usable = self.norms > 1e-12 * self.orthogonal.shape[1]
        safe = np.where(usable, self.norms, 1.0)
        return np.where(usable, np.einsum("ka,ka->k", projections, projections) / safe, 0.0)


@dataclass(frozen=True)
class Basis:
    """What a window's models depend on besides the positions: only its row times ``u``.

    ``rivals`` are the position steps, outliers and the grid of acceleration changes over the
    interior intervals (in the first and last intervals a kink or an acceleration change is the
    parabola plus an outlier, already a rival); ``kinks[i]`` = (a, b) of interval i."""

    u: FloatArray
    maker: FloatArray
    rivals: Columns
    accel_grid: FloatArray
    kinks: dict[int, tuple[FloatArray, FloatArray]]

    @classmethod
    def of(cls, u: FloatArray) -> Basis:
        m, maker = u.size, residual_maker(u)
        steps = (np.arange(m)[None, :] > np.arange(m - 1)[:, None]).astype(np.float64)
        fractions = np.linspace(0.0, 1.0, GRID)
        grid = (u[1:-2, None] + fractions[None, :] * np.diff(u)[1:-1, None]).ravel()
        accel = np.stack([accel_column(u, s) for s in grid]) if grid.size else np.zeros((0, m))
        rivals = Columns.of(np.vstack([steps, np.eye(m), accel]), maker)
        kinks = {i: kink_columns(u, maker, i) for i in range(1, m - 2)}
        return cls(u, maker, rivals, grid, kinks)

    def accel_gain(self, residual: FloatArray, grid_gains: FloatArray) -> float:
        """Best acceleration change: the best grid point refined between its neighbours."""
        if not self.accel_grid.size:
            return 0.0
        best = int(np.argmax(grid_gains))
        spacing = float(np.diff(self.u).max()) / (GRID - 1)
        lo = max(float(self.u[1]), float(self.accel_grid[best]) - spacing)
        hi = min(float(self.u[-2]), float(self.accel_grid[best]) + spacing)

        def loss(s: float) -> float:
            column = Columns.of(accel_column(self.u, s)[None, :], self.maker)
            return -float(column.gains(residual)[0])

        refined = minimize_scalar(loss, bounds=(lo, hi), method="bounded")
        return max(float(grid_gains[best]), -float(refined.fun))


Bases = dict[tuple[float, ...], Basis]


def kink_columns(u: FloatArray, maker: FloatArray, interval: int) -> tuple[FloatArray, FloatArray]:
    """On [u_i, u_i+1] the kink column (u - s)_+ is a - s b with a = u R and b = R (R: the rows
    after the interval); both made orthogonal to the parabola."""
    right = (np.arange(u.size) > interval).astype(np.float64)
    return maker @ (u * right), maker @ right


def kink_times(basis: Basis, residuals: FloatArray, interval: int) -> tuple[FloatArray, FloatArray]:
    """(s, gain) of the best kink with s in [u_i, u_i+1], i = ``interval``, for a stack of
    window residuals (k, m, 3) that share ``basis``.

    gain(s) = N(s) / D(s) with N = sum_k (alpha_k - s beta_k)^2 and D = |a - s b|^2, two
    quadratics, whose stationary points solve
    (A F - B E) + (C E - A G) s + (B G - C F) s^2 = 0."""
    a, b = basis.kinks[interval]
    alpha = np.einsum("kma,m->ka", residuals, a)
    beta = np.einsum("kma,m->ka", residuals, b)
    big_a, big_b, big_c = (alpha**2).sum(1), (alpha * beta).sum(1), (beta**2).sum(1)
    big_e, big_f, big_g = float(a @ a), float(a @ b), float(b @ b)
    lo, hi = float(basis.u[interval]), float(basis.u[interval + 1])
    roots = real_roots(
        big_b * big_g - big_c * big_f, big_c * big_e - big_a * big_g, big_a * big_f - big_b * big_e
    )
    inside = np.where((roots > lo) & (roots < hi), roots, lo)
    s = np.column_stack([np.full(len(residuals), lo), np.full(len(residuals), hi), inside])
    numerator = ((alpha[:, None, :] - s[:, :, None] * beta[:, None, :]) ** 2).sum(2)
    denominator = big_e - 2.0 * s * big_f + s**2 * big_g
    values = np.where(denominator > 1e-12, numerator / np.maximum(denominator, 1e-300), 0.0)
    best = np.argmax(values, axis=1)
    rows = np.arange(len(residuals))
    return s[rows, best], values[rows, best]


def kink_time(basis: Basis, residual: FloatArray, interval: int) -> tuple[float, float]:
    """``kink_times`` of one window."""
    s, gain = kink_times(basis, residual[None], interval)
    return float(s[0]), float(gain[0])


def real_roots(c2: FloatArray, c1: FloatArray, c0: FloatArray) -> FloatArray:
    """(k, 2) real roots of c2 s^2 + c1 s + c0, NaN where there is none (a vanishing c2 leaves
    the linear root, if any)."""
    scale = np.maximum(np.maximum(np.abs(c2), np.abs(c1)), np.abs(c0))
    tiny = 1e-12 * scale
    disc = c1 * c1 - 4.0 * c2 * c0
    quadratic = (np.abs(c2) > tiny) & (disc >= 0.0)
    linear = (np.abs(c2) <= tiny) & (np.abs(c1) > tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt(np.where(quadratic, disc, 0.0))
        first = np.where(quadratic, (-c1 - root) / (2.0 * c2), np.where(linear, -c0 / c1, np.nan))
        second = np.where(quadratic, (-c1 + root) / (2.0 * c2), np.nan)
    return np.column_stack([first, second])


def explains_best(basis: Basis, residual: FloatArray, interval: int, gain: float) -> bool:
    """Whether the kink in ``interval`` with ``gain`` beats every alternative: a position step,
    an outlier, an acceleration change, or a kink in another interior interval. A kink on the
    row shared with a neighbouring interval ties with that interval's and is kept."""
    m = basis.u.size
    values = basis.rivals.gains(residual)
    simple = 2 * m - 1
    rival = float(values[:simple].max())
    rival = max(rival, basis.accel_gain(residual, values[simple:]))
    kinks = [kink_time(basis, residual, i)[1] for i in basis.kinks if i != interval]
    return gain > rival and all(gain >= other * (1.0 - TIE_RTOL) for other in kinks)


def residual_dof(rows: int) -> int:
    """Residual degrees of freedom of the kink fit: three axes, four coefficients each."""
    return DEGREES_OF_FREEDOM * (rows - KINK_COEFFICIENTS)


def critical_ratio(alpha: float, boundaries: int, rows: int, known: bool) -> float:
    """Quantile at 1 - alpha / boundaries (Bonferroni over the track) of chi2_3 / 3 for a known
    scale, of F(3, residual dof) otherwise."""
    tail = alpha / max(boundaries, 1)
    if known:
        return float(chi2.isf(tail, DEGREES_OF_FREEDOM)) / DEGREES_OF_FREEDOM
    return float(f_dist.isf(tail, DEGREES_OF_FREEDOM, residual_dof(rows)))


def kink_ratios(gains: FloatArray, residuals: FloatArray, known: bool) -> FloatArray:
    """The kinks' gain per coefficient over the noise variance, for a stack of windows (k, m, 3):
    1 (whitened) for a known scale; otherwise the residual variance of each kink fit in its
    window, never below 1. The window's residual keeps the size of the test when sigma-hat,
    measured on a few stencils, comes out low (with a plain chi-square bound 23 % of clean
    tracks had a false impact in the tests)."""
    if known:
        return gains / DEGREES_OF_FREEDOM
    rss = (residuals**2).sum(axis=(1, 2)) - gains
    variance: FloatArray = np.maximum(rss / residual_dof(residuals.shape[1]), 1.0)
    return gains / (DEGREES_OF_FREEDOM * variance)


def finite_runs(position: FloatArray) -> tuple[slice, ...]:
    return visible_runs(np.all(np.isfinite(position), axis=-1))


def boundaries(position: FloatArray, side_rows: int) -> Iterator[tuple[int, int, int]]:
    """(row, lo, hi): boundary after ``row`` with window rows lo..hi-1, at most ``side_rows`` per
    side and at least EDGE_SIDE_ROWS next to the end of a finite run."""
    for run in finite_runs(position):
        for row in range(run.start, run.stop - 1):
            lo, hi = max(run.start, row + 1 - side_rows), min(run.stop, row + 1 + side_rows)
            if row + 1 - lo >= EDGE_SIDE_ROWS and hi - row - 1 >= EDGE_SIDE_ROWS:
                yield row, lo, hi


def make_window(
    t: FloatArray, position: FloatArray, sigma: FloatArray, row: int, lo: int, hi: int
) -> Window:
    """Times in row steps are rounded to 1e-9 so that windows of one track share a Basis."""
    step = float(t[row + 1] - t[row])
    u = np.round((t[lo:hi] - t[row]) / step, 9)
    return Window(row, lo, u, position[lo:hi] / sigma, step, float(t[row]))


def judge_windows(
    windows: Sequence[Window], basis: Basis, critical: float, known: bool
) -> list[tuple[Window, float, float]]:
    """(window, s, ratio) of the windows sharing ``basis`` whose kink inside the boundary beats
    the parabola beyond the Bonferroni quantile ``critical`` and explains the window better than
    every alternative (the alternatives only for the few that pass the quantile)."""
    residuals = np.einsum("ij,kja->kia", basis.maker, np.stack([w.y for w in windows]))
    interval = windows[0].row - windows[0].lo
    s, gains = kink_times(basis, residuals, interval)
    ratios = kink_ratios(gains, residuals, known)
    return [
        (windows[i], float(s[i]), float(ratios[i]))
        for i in np.flatnonzero(ratios > critical).tolist()
        if explains_best(basis, residuals[i], interval, float(gains[i]))
    ]


def kink_fit(window: Window, s: float, sigma: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Velocity change (per second) and the position at the kink, in the track's units."""
    design = np.column_stack([parabola(window.u), np.maximum(0.0, window.u - s)])
    coef, *_ = np.linalg.lstsq(design, window.y, rcond=None)
    at_kink = parabola(np.array([s]))[0] @ coef[:3]
    return coef[3] * sigma / window.step, at_kink * sigma


def track_impacts(
    t: FloatArray,
    position: FloatArray,
    noise: NoiseScale,
    side_rows: int,
    alpha: float,
    bases: Bases | None = None,
) -> list[Impact]:
    """Impacts of one track (rows with a non-finite position split it into runs). ``bases``
    caches the window bases by row times; tracks of one clip can share it."""
    sigma = noise.sigma
    candidates = list(boundaries(position, side_rows))
    cache: Bases = {} if bases is None else bases
    groups: dict[tuple[float, ...], list[Window]] = {}
    for row, lo, hi in candidates:
        window = make_window(t, position, sigma, row, lo, hi)
        groups.setdefault(tuple(window.u.tolist()), []).append(window)
    accepted: list[Impact] = []
    for key, windows in groups.items():
        basis = cache.get(key)
        if basis is None:
            basis = cache[key] = Basis.of(windows[0].u)
        critical = critical_ratio(alpha, len(candidates), len(key), noise.known)
        for window, s, ratio in judge_windows(windows, basis, critical, noise.known):
            delta_v, at_kink = kink_fit(window, s, sigma)
            accepted.append(
                Impact(window.row, window.origin + s * window.step, delta_v, at_kink, ratio)
            )
    return local_maxima(sorted(accepted, key=lambda impact: impact.row))


def local_maxima(impacts: list[Impact]) -> list[Impact]:
    """Keep an impact only when no other one within LOCAL_ROWS rows has a larger F ratio (ties go
    to the earlier row)."""

    def beaten(impact: Impact) -> bool:
        return any(
            abs(other.row - impact.row) <= LOCAL_ROWS
            and (
                other.score > impact.score
                or (other.score == impact.score and other.row < impact.row)
            )
            for other in impacts
            if other is not impact
        )

    return [impact for impact in impacts if not beaten(impact)]
