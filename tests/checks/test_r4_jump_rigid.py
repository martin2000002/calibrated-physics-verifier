"""R4_jump rigid2: the stencils where the object's size steps are left out."""

from __future__ import annotations

from typing import Any

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import CheckInputs
from physverify.checks.r4_track_jump import NoiseContinuityCheck, NoiseJumpParams
from physverify.checks.variants.r4_jump_rigid import RigidJumpCheck, RigidJumpParams
from physverify.physics.size_steps import size_step_stencil_mask, size_steps
from physverify.physics.tracks import impact_stencil_mask
from physverify.schema.checks import CheckResult

FRAME = 16


def rigid(inputs: CheckInputs, **params: Any) -> CheckResult:
    (result,) = RigidJumpCheck(RigidJumpParams(**params)).evaluate(inputs)
    return result


def v2(inputs: CheckInputs) -> CheckResult:
    (result,) = NoiseContinuityCheck(NoiseJumpParams()).evaluate(inputs)
    return result


def swapped_mask(factor: float) -> sc.Case:
    """A teleport whose apparent size steps with it, as a mask moving to another region does."""
    case = sc.teleport(delta=0.3, frame=FRAME)
    grown = case.states.size[FRAME:, 0] * factor
    return sc.with_values(case, "size", "ball", range(FRAME, sc.FRAMES), grown)


def test_a_step_flags_the_stencils_that_span_it() -> None:
    extents = np.ones((10, 3))
    extents[6:] *= 1.2
    extents[8:] *= 1.05
    assert size_steps(extents)[5] == pytest.approx(0.2)
    flagged = size_step_stencil_mask(extents, 0.10)
    assert np.flatnonzero(flagged).tolist() == [4, 5, 6]


def test_an_unknown_size_flags_nothing() -> None:
    extents = np.ones((8, 3))
    extents[3] = np.nan
    extents[4:] *= 2.0
    assert not size_step_stencil_mask(extents, 0.10).any()


def test_a_teleport_of_a_rigid_body_survives() -> None:
    inputs = sc.teleport(delta=0.3, frame=FRAME).inputs()
    found, reference = rigid(inputs), v2(inputs)
    assert found.magnitude == reference.magnitude
    assert found.details["excluded_rows"] == reference.details["excluded_rows"]


def test_a_jump_with_a_size_step_is_left_out() -> None:
    inputs = swapped_mask(1.3).inputs()
    found, reference = rigid(inputs), v2(inputs)
    assert reference.magnitude > 10.0
    assert found.magnitude == pytest.approx(0.0, abs=1e-9)
    assert found.details["excluded_rows"] == reference.details["excluded_rows"] + 3


def test_the_size_step_stencils_are_left_out_of_the_noise_estimate() -> None:
    inputs = swapped_mask(1.3).inputs()
    found, reference = rigid(inputs), v2(inputs)
    assert found.details["usable_stencils"] == reference.details["usable_stencils"] - 3


def test_the_size_step_is_configurable() -> None:
    inputs = swapped_mask(1.3).inputs()
    assert rigid(inputs, rigid_size_step=0.5).magnitude == v2(inputs).magnitude


def test_the_params_keep_the_size_step() -> None:
    check = RigidJumpCheck(RigidJumpParams(rigid_size_step=0.2))
    assert isinstance(check.params, RigidJumpParams)
    assert check.params.rigid_size_step == 0.2


@pytest.mark.parametrize("step_frame", [26, 16])
def test_impacts_and_size_steps_are_both_left_out(step_frame: int) -> None:
    bounce = sc.bounce()
    grown = bounce.states.size[step_frame:, 0] * 1.3
    case = sc.with_values(bounce, "size", "ball", range(step_frame, sc.FRAMES), grown)
    states, contacts = case.states, case.contacts
    assert contacts is not None
    impacts = impact_stencil_mask("ball", contacts, states.frame_index)
    steps = size_step_stencil_mask(states.size[:, 0], 0.10)
    assert impacts.any() and steps.any()
    found, reference = rigid(case.inputs()), v2(case.inputs())
    assert reference.details["excluded_rows"] == np.count_nonzero(impacts)
    assert found.details["excluded_rows"] == np.count_nonzero(impacts | steps)
