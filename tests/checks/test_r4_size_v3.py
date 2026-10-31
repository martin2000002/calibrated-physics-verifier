"""R4_size v3: persistent change against the running median, windows in seconds, border rows."""

from __future__ import annotations

from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import CheckInputs
from physverify.checks.variants.r4_size_v3 import DetrendedSizeCheck, DetrendedSizeParams
from physverify.physics.measures.detrended_size import persistent_change
from physverify.physics.row_time import odd_rows_in, rows_in, seconds_per_row
from physverify.schema.checks import AbstainReason, CheckResult, Verdict


def size(inputs: CheckInputs, **params: float) -> dict[str, CheckResult]:
    check = DetrendedSizeCheck(DetrendedSizeParams(**params))  # type: ignore[arg-type]
    return {r.subject[0]: r for r in check.evaluate(inputs)}


def bump(mu: float, rows: int, balls: int = 3) -> sc.Case:
    return sc.resized_pair(factor=1.0 + mu, first=12, last=11 + rows, balls=balls)


@pytest.mark.parametrize("rows", [3, 5, 8])
@pytest.mark.parametrize("mu", [0.05, 0.4])
def test_a_bump_of_three_to_eight_rows_gives_its_level(rows: int, mu: float) -> None:
    result = size(sc.bare_inputs(bump(mu, rows)))["ball_0"]
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude == pytest.approx(mu, rel=1e-12)
    assert result.details["window_rows"] == 17 and result.details["persistence_rows"] == 3
    assert result.frame_span is not None and 12 <= result.frame_span[0] <= result.frame_span[1]


@pytest.mark.parametrize("rows", [1, 2])
def test_a_spike_shorter_than_the_persistence_gives_zero(rows: int) -> None:
    result = size(sc.bare_inputs(bump(0.4, rows)))["ball_0"]
    assert result.magnitude == pytest.approx(0.0, abs=1e-12)


def test_the_persistence_is_read_in_seconds() -> None:
    result = size(sc.bare_inputs(bump(0.4, 2)), persistence_s=2.0 / 24.0)["ball_0"]
    assert result.details["persistence_rows"] == 2
    assert result.magnitude == pytest.approx(0.4, rel=1e-12)


def test_a_single_body_abstains() -> None:
    (result,) = size(sc.bare_inputs(sc.size_change())).values()
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES


@pytest.mark.parametrize(
    ("fps", "window", "persistence"), [(24.0, 17, 3), (30.0, 21, 4), (120.0, 85, 15), (16.0, 11, 2)]
)
def test_windows_follow_the_real_frame_rate(fps: float, window: int, persistence: int) -> None:
    params = DetrendedSizeParams()
    assert odd_rows_in(params.window_s, 1.0 / fps) == window
    assert rows_in(params.persistence_s, 1.0 / fps) == persistence
    case = bump(0.2, 3)
    clip = replace(case.clip, seconds_per_frame=1.0 / fps)
    result = size(replace(sc.bare_inputs(case), clip=clip))["ball_0"]
    assert result.details["window_rows"] == window
    assert result.details["persistence_rows"] == persistence


def test_a_strided_table_counts_seconds_per_row() -> None:
    case = sc.strided(bump(0.2, 8), 2, quantize=False)
    assert seconds_per_row(case.states.frame_index, sc.DT) == pytest.approx(2.0 * sc.DT)
    result = size(sc.bare_inputs(case))["ball_0"]
    assert (result.details["window_rows"], result.details["persistence_rows"]) == (9, 2)
    assert result.magnitude == pytest.approx(0.2, rel=1e-12)


def test_a_run_that_changes_sign_holds_nothing() -> None:
    ratios = np.array([1.0] * 8 + [1.2, 0.8, 1.2] + [1.0] * 8)
    change = persistent_change(ratios, 9, 3)
    assert change is not None and change.magnitude == pytest.approx(0.0, abs=1e-12)
    assert persistent_change(np.array([1.0, np.nan, 1.0]), 3, 2) is None


def with_border_cut(case: sc.Case, rows: slice, ball: int) -> CheckInputs:
    cut = np.zeros(case.states.visible.shape, dtype=np.bool_)
    cut[rows, ball] = True
    return replace(sc.bare_inputs(case), states=replace(case.states, border_cut=cut))


def test_rows_cut_by_the_image_border_are_left_out() -> None:
    case = bump(0.4, 5)
    assert case.states.border_cut is None
    clean = size(sc.bare_inputs(case))["ball_0"]
    assert clean.magnitude == pytest.approx(0.4, rel=1e-12) and clean.details["excluded_rows"] == 0
    cut = size(with_border_cut(case, slice(12, 17), 0))["ball_0"]
    assert cut.details["excluded_rows"] == 5
    assert cut.magnitude == pytest.approx(0.0, abs=1e-12)
    reference = size(with_border_cut(case, slice(None), 2))
    assert reference["ball_2"].details["excluded_rows"] == sc.FRAMES
    assert reference["ball_0"].magnitude == pytest.approx(0.4, rel=1e-12)
