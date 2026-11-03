"""R4_size v3_window_depth_unoccluded: v3_window_depth without the rows a nearer object covers."""

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
from physverify.checks.variants.r4_size_window_depth_unoccluded import (
    UnoccludedWindowDepthSizeCheck,
)
from physverify.schema.checks import CheckResult
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.verification.check_set import (
    ChecksSettings,
    CheckVariantsSettings,
    checks_hash,
    select_variants,
)

FIRST, LAST = 12, 17
GROWTH = 1.4


def results(check: type[DetrendedSizeCheck], inputs: CheckInputs) -> dict[str, dict[str, object]]:
    found: list[CheckResult] = check(DetrendedSizeParams()).evaluate(inputs)
    return {r.subject[0]: r.to_dict() for r in found}


def grown() -> sc.Case:
    """ball_0 grows by ``GROWTH`` on rows FIRST..LAST among three still balls."""
    return sc.resized_pair(GROWTH, FIRST, LAST, balls=3)


def covered(case: sc.Case, value: float, rows: Iterable[int] = range(FIRST, LAST + 1)) -> sc.Case:
    """``case`` with ball_0's nearer_cover ``value`` on ``rows`` and 0 elsewhere."""
    cover = np.zeros(case.states.size.shape[:2])
    cover[list(rows), 0] = value
    return replace(case, states=replace(case.states, nearer_cover=cover))


def out_of_frame(case: sc.Case, rows: Iterable[int] = range(FIRST, LAST + 1)) -> sc.Case:
    in_frame = np.array(case.states.in_frame, copy=True)
    in_frame[list(rows), 0] = False
    return replace(case, states=replace(case.states, in_frame=in_frame))


@pytest.mark.parametrize("value", [1e-9, 0.25, 1.0])
def test_a_covered_row_is_left_out_as_an_out_of_frame_one(value: float) -> None:
    mine = results(UnoccludedWindowDepthSizeCheck, sc.bare_inputs(covered(grown(), value)))
    assert mine == results(WindowDepthSizeCheck, sc.bare_inputs(out_of_frame(grown())))
    assert mine["ball_0"]["magnitude"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("value", [0.0, np.nan])
def test_an_uncovered_or_unknown_row_stays(value: float) -> None:
    mine = results(UnoccludedWindowDepthSizeCheck, sc.bare_inputs(covered(grown(), value)))
    assert mine == results(WindowDepthSizeCheck, sc.bare_inputs(grown()))
    assert mine["ball_0"]["magnitude"] == pytest.approx(GROWTH - 1.0, rel=1e-12)


def test_a_table_without_the_cover_is_left_as_window_depth_sees_it() -> None:
    inputs = sc.bare_inputs(grown())
    assert inputs.states.nearer_cover is None
    assert results(UnoccludedWindowDepthSizeCheck, inputs) == results(WindowDepthSizeCheck, inputs)


def test_a_ground_truth_lift_is_left_as_window_depth_sees_it() -> None:
    inputs = replace(
        sc.bare_inputs(covered(grown(), 1.0)), provenance={Stage.LIFT.value: GROUND_TRUTH}
    )
    assert results(UnoccludedWindowDepthSizeCheck, inputs) == results(WindowDepthSizeCheck, inputs)


def test_it_takes_the_place_of_r4_size_with_its_own_checks_hash() -> None:
    checks, variants = plugins.build("checks"), plugins.build("check_variants")
    names = ("v3", "v3_window_depth", "v3_window_depth_unoccluded")
    chosen = {
        name: select_variants(
            checks, variants, CheckVariantsSettings.model_validate({"R4_size": name})
        )
        for name in names
    }
    assert chosen["v3_window_depth_unoccluded"]["R4_size"] is UnoccludedWindowDepthSizeCheck
    overrides = ChecksSettings()
    assert len({checks_hash(chosen[name], overrides) for name in names}) == len(names)
