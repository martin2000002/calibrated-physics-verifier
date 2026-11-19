"""R4_size v3_window_depth: v3 on extents taken to the running-median depth of each object."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.app import plugins
from physverify.checks._base import CheckInputs
from physverify.checks.variants.r4_size_v3 import DetrendedSizeCheck, DetrendedSizeParams
from physverify.checks.variants.r4_size_window_depth import WindowDepthSizeCheck
from physverify.schema.checks import CheckResult
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame
from physverify.verification.check_set import (
    ChecksSettings,
    CheckVariantsSettings,
    checks_hash,
    select_variants,
)

FIRST = 12
NOISE = 1.3


def results(check: type[DetrendedSizeCheck], inputs: CheckInputs) -> dict[str, CheckResult]:
    return {r.subject[0]: r for r in check(DetrendedSizeParams()).evaluate(inputs)}


def same(inputs: CheckInputs) -> bool:
    """Both checks give the same results, field by field."""
    window, v3 = results(WindowDepthSizeCheck, inputs), results(DetrendedSizeCheck, inputs)
    return {k: r.to_dict() for k, r in window.items()} == {k: r.to_dict() for k, r in v3.items()}


def lifted_with_depth_error(case: sc.Case, rows: Iterable[int], factor: float) -> sc.Case:
    """ball_0's depth off by ``factor`` on ``rows`` with the same silhouette, so a lift's
    extents (pixels times depth) are off by it too."""
    states = case.states
    position = np.array(states.position, copy=True)
    size = np.array(states.size, copy=True)
    picked = np.isin(states.frame_index, np.fromiter(rows, dtype=np.int64))
    position[picked, 0, 2] *= factor
    size[picked, 0] *= factor
    return replace(case, states=replace(states, position=position, size=size))


def noisy() -> sc.Case:
    return lifted_with_depth_error(sc.resized_pair(1.0, balls=3), range(FIRST, FIRST + 5), NOISE)


@pytest.mark.parametrize("rows", [3, 5, 8])
def test_a_constant_depth_gives_v3(rows: int) -> None:
    inputs = sc.bare_inputs(sc.resized_pair(1.4, FIRST, FIRST + rows - 1, balls=3))
    assert same(inputs)


def test_a_depth_error_the_median_does_not_follow_is_not_a_size_change() -> None:
    inputs = sc.bare_inputs(noisy())
    assert results(DetrendedSizeCheck, inputs)["ball_0"].magnitude == pytest.approx(NOISE - 1.0)
    assert results(WindowDepthSizeCheck, inputs)["ball_0"].magnitude == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("rows", [3, 5, 8])
def test_a_real_size_change_stays_under_depth_spikes(rows: int) -> None:
    case = sc.resized_pair(1.4, FIRST, FIRST + rows - 1, balls=3)
    spiked = lifted_with_depth_error(case, (4, FIRST + 1, 25), NOISE)
    result = results(WindowDepthSizeCheck, sc.bare_inputs(spiked))["ball_0"]
    assert result.magnitude == pytest.approx(0.4, rel=1e-12)


def test_the_sigma_scales_with_the_extent() -> None:
    inputs = sc.bare_inputs(sc.with_sigma(noisy(), size=0.01))
    check = WindowDepthSizeCheck(DetrendedSizeParams())
    states = check._at_window_depth(inputs).states
    np.testing.assert_allclose(states.size[FIRST, 0], inputs.states.size[0, 0], rtol=1e-12)
    np.testing.assert_allclose(
        states.size_sigma / states.size, inputs.states.size_sigma / inputs.states.size, rtol=1e-12
    )


def test_a_ground_truth_lift_is_left_as_v3_sees_it() -> None:
    inputs = replace(sc.bare_inputs(noisy()), provenance={Stage.LIFT.value: GROUND_TRUTH})
    assert same(inputs)


def test_states_outside_the_camera_frame_are_left_as_v3_sees_them() -> None:
    case = noisy()
    inputs = sc.bare_inputs(replace(case, states=replace(case.states, frame=CoordFrame.WORLD_ZUP)))
    assert same(inputs)


def test_it_takes_the_place_of_r4_size_with_its_own_checks_hash() -> None:
    checks, variants = plugins.build("checks"), plugins.build("check_variants")
    chosen = {
        name: select_variants(
            checks, variants, CheckVariantsSettings.model_validate({"R4_size": name})
        )
        for name in ("v3", "v3_window_depth")
    }
    assert chosen["v3_window_depth"]["R4_size"] is WindowDepthSizeCheck
    overrides = ChecksSettings()
    assert checks_hash(chosen["v3_window_depth"], overrides) != checks_hash(chosen["v3"], overrides)
