"""R2: the top body of a stack shifted along a face normal of its support until its centre of mass
lies outside the support polygon."""

from __future__ import annotations

from typing import ClassVar

from physverify.schema.states import ObjectStateTable
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._r2_stack import (
    StackParams,
    StackShift,
    overhang_after,
    shift_span,
)
from physverify.synth.perturbations._state_edit import Vec
from physverify.synth.placement import solve_offset
from physverify.synth.settings import PerturbationCommon

NAME = "r2_unsupported_rest"


class Params(StackParams):
    pass


class UnsupportedRest(StackShift):
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R2",)
    Params: ClassVar[type[PerturbationCommon]] = Params

    def _shift(
        self, states: ObjectStateTable, top: int, support: int, direction: Vec, magnitude: float
    ) -> float:
        tol = self.params.contact_tol_m
        _, reach = shift_span(states, top, support, direction, tol)
        if reach <= 0.0:
            raise PerturbationNotApplicableError(
                f"{NAME}: the top body is already at the edge of its support"
            )
        overhang = overhang_after(states, top, support, direction, tol)
        return solve_offset(overhang, magnitude, 0.0, reach, tol=1e-6)


PLUGINS = (UnsupportedRest,)
