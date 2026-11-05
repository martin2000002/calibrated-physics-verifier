"""R2 decoy: the top body ``r2_unsupported_rest`` would push off its support, shifted along the
same face normal but stopped with its centre of mass ``magnitude`` x L inside the support edge.
A clean clip for training labels."""

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

NAME = "r2_unsupported_rest_decoy"
MIRRORS = "r2_unsupported_rest"


class Params(StackParams):
    pass


class UnsupportedRestDecoy(StackShift):
    name: ClassVar[str] = NAME
    seed_stream: ClassVar[str] = MIRRORS
    target_checks: ClassVar[tuple[str, ...]] = ()
    Params: ClassVar[type[PerturbationCommon]] = Params

    def _shift(
        self, states: ObjectStateTable, top: int, support: int, direction: Vec, magnitude: float
    ) -> float:
        tol = self.params.contact_tol_m
        centre, reach = shift_span(states, top, support, direction, tol)
        if reach <= centre:
            raise PerturbationNotApplicableError(f"{NAME}: no room on the support along the normal")
        # The signed overhang rises from the support centre to its edge, so -magnitude is
        # bracketed there; a support too narrow for it refuses the attempt.
        overhang = overhang_after(states, top, support, direction, tol)
        return solve_offset(overhang, -magnitude, centre, reach, tol=1e-6)


PLUGINS = (UnsupportedRestDecoy,)
