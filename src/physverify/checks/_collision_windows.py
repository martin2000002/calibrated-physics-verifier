"""Impact windows shared by R3a and R3b. Velocities on each side of an impact come from a
constant-acceleration fit over the nearest flight rows, evaluated at the impact time: by default
a parabola on the raw positions, which, unlike the smoothed velocities, do not mix samples from
both sides of the impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator

from physverify.checks._base import CheckInputs
from physverify.checks._params import CommonParams
from physverify.checks._results import abstain
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
    impact_margin_frames: int = Field(default=0, ge=0)
    # Rows per side (4 / 6 / 8 give sigma(e) 0.034 / 0.015 / 0.009 on the noisy unit bounce and
    # the same oracle sweep); 6 keeps each side's arc short against drag and depth drift.
    fit_frames: int = Field(default=6, ge=2)
    min_side_frames: int = Field(default=3, ge=2)
    # "velocity" is the line fit on the kinematics velocities, smoothed across the impact on
    # perceived clips. The oracle sweep uses it because the synthetic ground-truth labels are
    # defined on simulator velocities, which lead PyBullet positions by g*h/2.
    fit_source: FitSource = "position"

    @model_validator(mode="after")
    def _check_rows(self) -> CollisionParams:
        needed = MOTION_DEGREE[self.fit_source] + 1
        if not needed <= self.min_side_frames <= self.fit_frames:
            raise ValueError(
                f"need {needed} <= min_side_frames <= fit_frames for a {self.fit_source} fit"
            )
        return self


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
    states: ObjectStateTable, contacts: ContactTable, record: ContactRecord, index: int, p: Any
) -> Window:
    """Nearest ``fit_frames`` rows with a finite fitted quantity in the flight runs around the
    record (an unknown sigma does not drop a row; it makes the fit's sigma unknown)."""
    fi = states.frame_index
    margin = p.impact_margin_frames
    contact = np.flatnonzero(contact_rows(record, fi, margin))
    if contact.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Window(contact, empty, empty.copy())
    flight = flight_mask(str(states.object_ids[index]), contacts, fi, margin)
    finite = np.all(np.isfinite(getattr(states, p.fit_source)[:, index]), axis=-1)
    before = _side(flight, finite, range(int(contact[0]) - 1, -1, -1), p.fit_frames)
    after = _side(flight, finite, range(int(contact[-1]) + 1, fi.shape[0]), p.fit_frames)
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


def as_params(cls: type[BaseModel], params: BaseModel) -> Any:
    return (
        params
        if isinstance(params, cls)
        else cls.model_validate(params.model_dump(exclude_unset=True))
    )
