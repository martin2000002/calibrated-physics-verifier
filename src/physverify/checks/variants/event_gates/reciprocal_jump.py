"""Keep an event only when the two bodies' image velocity jumps point in opposite directions.

An impulse changes the velocities by J / m_a and -J / m_b; projected, the jumps of the image
centres u = (X/Z, Y/Z) stay opposite, while a kink of one centre (a bounce beside a resting body, a
track glitch) leaves the other body's jump to noise, of either sign."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from physverify.checks.variants._image_impulse import (
    Candidate,
    FloatArray,
    Gates,
    image_track,
    line_fit,
)
from physverify.schema.states import ObjectStateTable


class ReciprocalJump:
    name: ClassVar[str] = "reciprocal_jump"

    def accepts(self, states: ObjectStateTable, event: Candidate, gates: Gates) -> bool:
        a, b = (self._jump(states, event, index) for index in event.pair)
        return float(a @ b) < 0.0

    @staticmethod
    def _jump(states: ObjectStateTable, event: Candidate, index: int) -> FloatArray:
        """Right minus left slope of the body's image centre, on the rows ``detect`` fitted."""
        fi = np.asarray(states.frame_index, dtype=np.float64)
        u, t0 = image_track(states, index), float(fi[event.row])
        left, right = list(event.left), list(event.right)
        return line_fit(fi[right], u[right], t0).slope - line_fit(fi[left], u[left], t0).slope


PLUGINS = (ReciprocalJump,)
