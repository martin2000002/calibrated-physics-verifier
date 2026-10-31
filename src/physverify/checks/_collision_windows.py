"""Impact windows shared by R3a and R3b. Velocities on each side of an impact come from a
constant-acceleration fit over the nearest flight rows, evaluated at the impact time: by default
a parabola on the raw positions, which, unlike the smoothed velocities, do not mix samples from
both sides of the impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from physverify.checks._base import CheckInputs, abstain
from physverify.checks._params import CommonParams
from physverify.physics.row_time import reference_seconds, rows_in
from physverify.physics.tracks import (
    MOTION_DEGREE,
    FitSource,
    MotionFit,
    contact_rows,
    flight_mask,
    motion_fit,
    native_span,
)
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class CollisionParams(CommonParams):
    impact_margin_s: float = Field(default=0.0, ge=0.0)
    # Time per side (4 / 6 / 8 rows at 24 fps give sigma(e) 0.034 / 0.015 / 0.009 on the noisy
    # unit bounce and the same oracle sweep); 6 rows keep each side's arc short against drag and
    # depth drift.
    fit_s: float = Field(default=reference_seconds(6), gt=0.0)
    min_side_s: float = Field(default=reference_seconds(3), gt=0.0)
    # "velocity" is the line fit on the kinematics velocities, smoothed across the impact on
    # perceived clips. The oracle sweep uses it because the synthetic ground-truth labels are
    # defined on simulator velocities, which lead PyBullet positions by g*h/2.
    fit_source: FitSource = "position"

    @model_validator(mode="after")
    def _check_rows(self) -> CollisionParams:
        """At the reference rate; slower clips are floored to the fit's rows in ``rows``."""
        needed = self.needed_rows
        if rows_in(self.min_side_s, reference_seconds(1)) < needed or self.min_side_s > self.fit_s:
            raise ValueError(
                f"need {needed} reference rows <= min_side_s <= fit_s for a {self.fit_source} fit"
            )
        return self

    @property
    def needed_rows(self) -> int:
        return MOTION_DEGREE[self.fit_source] + 1

    def rows(self, inputs: CheckInputs) -> CollisionRows:
        min_side = rows_in(self.min_side_s, inputs.row_seconds, self.needed_rows)
        return CollisionRows(
            margin=rows_in(self.impact_margin_s, inputs.clip.seconds_per_frame, minimum=0),
            fit=rows_in(self.fit_s, inputs.row_seconds, min_side),
            min_side=min_side,
        )


@dataclass(frozen=True)
class CollisionRows:
    """The impact windows of one table: ``margin`` native frames, ``fit`` rows per side at most
    and ``min_side`` at least."""

    margin: int
    fit: int
    min_side: int


@dataclass(frozen=True, eq=False)
class Window:
    """Rows of one body around an impact: in contact, and the fit rows before and after."""

    contact: IntArray
    before: IntArray
    after: IntArray

    def enough(self, min_rows: int) -> bool:
        return min(self.before.size, self.after.size) >= min_rows

    def counts(self) -> dict[str, int]:
        return {"rows_before": int(self.before.size), "rows_after": int(self.after.size)}


@dataclass(frozen=True, eq=False)
class SideData:
    """Times, fitted quantity (positions or velocities) and its sigma on one side's rows."""

    t: FloatArray
    y: FloatArray
    sigma: FloatArray

    def fit(self, source: FitSource) -> MotionFit:
        return motion_fit(self.t, self.y, self.sigma, source)

    def fit_draws(self, source: FitSource, noise: FloatArray) -> MotionFit:
        """One fit per draw of (draws, rows, 3) ``noise``, batched along a draw axis."""
        y = self.y[:, None] + noise.swapaxes(0, 1)
        return motion_fit(self.t, y, self.sigma[:, None], source)


def fit_window(
    states: ObjectStateTable,
    contacts: ContactTable,
    record: ContactRecord,
    index: int,
    rows: CollisionRows,
    source: FitSource,
) -> Window:
    """Nearest ``rows.fit`` rows with a finite fitted quantity in the flight runs around the
    record (an unknown sigma does not drop a row; it makes the fit's sigma unknown)."""
    fi = states.frame_index
    margin = rows.margin
    contact = np.flatnonzero(contact_rows(record, fi, margin))
    if contact.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Window(contact, empty, empty.copy())
    flight = flight_mask(str(states.object_ids[index]), contacts, fi, margin)
    finite = np.all(np.isfinite(getattr(states, source)[:, index]), axis=-1)
    before = _side(flight, finite, range(int(contact[0]) - 1, -1, -1), rows.fit)
    after = _side(flight, finite, range(int(contact[-1]) + 1, fi.shape[0]), rows.fit)
    return Window(contact, before, after)


def _side(
    flight: NDArray[np.bool_], finite: NDArray[np.bool_], walk: range, count: int
) -> IntArray:
    rows: list[int] = []
    for row in walk:
        if not flight[row] or len(rows) == count:
            break
        if finite[row]:
            rows.append(row)
    return np.array(sorted(rows), dtype=np.int64)


def side_data(states: ObjectStateTable, index: int, rows: IntArray, source: FitSource) -> SideData:
    values, sigma = getattr(states, source), getattr(states, f"{source}_sigma")
    return SideData(states.t[rows], values[rows, index], sigma[rows, index])


def contact_span(states: ObjectStateTable, window: Window) -> tuple[int, int] | None:
    if window.contact.size == 0:
        return None
    return native_span(states.frame_index, int(window.contact[0]), int(window.contact[-1]))


def no_contacts(check: type[Any], inputs: CheckInputs) -> list[CheckResult]:
    subject = tuple(sorted(str(i) for i in inputs.states.object_ids))
    details = {"missing": "contacts"}
    return [abstain(check, inputs, subject, AbstainReason.MISSING_INPUT, None, details)]
