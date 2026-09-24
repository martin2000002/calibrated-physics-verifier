import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import Check
from physverify.checks._common import CommonParams
from physverify.checks.r1_interpenetration import (
    PLUGINS,
    InterpenetrationCheck,
    InterpenetrationParams,
)
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.schema.constants import ENV_ID

ALL_FRAMES = range(sc.FRAMES)


def check() -> InterpenetrationCheck:
    return InterpenetrationCheck(InterpenetrationParams())


def by_subject(case: sc.Case) -> dict[tuple[str, ...], CheckResult]:
    return {r.subject: r for r in check().evaluate(case.inputs())}


def deeper_at(case: sc.Case, frames: range | list[int], extra: float) -> sc.Case:
    """Move ball_b towards ball_a along world x by ``extra`` on the given native frames."""
    index = case.states.object_index("ball_b")
    moved = case.states.position[frames[0], index] - case.view.vector([extra, 0.0, 0.0])
    return sc.with_values(case, "position", "ball_b", frames, moved)


def test_plugin_contract() -> None:
    assert (InterpenetrationCheck,) == PLUGINS
    assert isinstance(check(), Check)
    assert InterpenetrationCheck(CommonParams(seed=3)).params.seed == 3


@pytest.mark.parametrize("pitch", [0.0, 30.0])
def test_overlapping_spheres_give_depth_over_the_smaller_size(pitch: float) -> None:
    result = by_subject(sc.overlapping_spheres(0.05, (0.1, 0.15), pitch_deg=pitch))[
        ("ball_a", "ball_b")
    ]
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude == pytest.approx(0.25, abs=1e-9)
    assert result.sigma == 0.0
    assert result.details["distance"] == pytest.approx(-0.05, abs=1e-9)
    assert result.details["ground"] == "none"


def test_box_and_sphere() -> None:
    results = by_subject(sc.box_sphere(depth=0.05))
    assert set(results) == {("box", "ball"), ("box", ENV_ID)}
    assert results[("box", "ball")].magnitude == pytest.approx(0.25, abs=1e-9)
    assert results[("box", ENV_ID)].magnitude == pytest.approx(0.0, abs=1e-9)


def test_ground_comes_from_the_env_records() -> None:
    result = by_subject(sc.sinking_box(depth=0.05, pitch_deg=20.0))[("box", ENV_ID)]
    assert result.magnitude == pytest.approx(0.25, abs=1e-9)
    assert result.details["ground"] == "records"


def test_no_contacts_still_checks_pairs() -> None:
    inputs = replace(sc.overlapping_spheres().inputs(), contacts=None)
    results = check().evaluate(inputs)
    assert [r.subject for r in results] == [("ball_a", "ball_b")]


def test_missing_positions_abstain_with_missing_input() -> None:
    case = sc.with_values(sc.overlapping_spheres(), "position", "ball_a", ALL_FRAMES, np.nan)
    result = by_subject(case)[("ball_a", "ball_b")]
    assert result.verdict is Verdict.ABSTAIN
    assert result.abstain_reason is AbstainReason.MISSING_INPUT


def test_hidden_and_occluded_abstains_with_occluded() -> None:
    case = sc.with_values(sc.overlapping_spheres(), "visible", "ball_b", ALL_FRAMES, False)
    case = sc.with_values(case, "occlusion", "ball_b", ALL_FRAMES, 0.95)
    result = by_subject(case)[("ball_a", "ball_b")]
    assert result.abstain_reason is AbstainReason.OCCLUDED
    weakly = sc.with_values(case, "occlusion", "ball_b", [3], 0.5)
    assert by_subject(weakly)[("ball_a", "ball_b")].abstain_reason is AbstainReason.MISSING_INPUT


def test_the_worst_row_is_reported_in_native_frames() -> None:
    case = deeper_at(sc.overlapping_spheres(depth=0.01), [10], extra=0.04)
    native = by_subject(case)[("ball_a", "ball_b")]
    assert native.magnitude == pytest.approx(0.25, abs=1e-9)
    assert native.frame_span == (10, 10)
    assert (native.details["frame"], native.details["row"]) == (10, 10)
    strided = by_subject(sc.strided(case, 2, quantize=False))[("ball_a", "ball_b")]
    assert strided.magnitude == pytest.approx(native.magnitude, abs=1e-9)
    assert strided.frame_span == (10, 10)
    assert (strided.details["frame"], strided.details["row"]) == (10, 5)
    assert strided.details["row_step"] == 2


@pytest.mark.parametrize("quantize", [False, True])
def test_strided_ground_contact_keeps_its_magnitude(quantize: bool) -> None:
    case = sc.strided(sc.sinking_box(depth=0.05), 2, quantize=quantize)
    result = by_subject(case)[("box", ENV_ID)]
    assert result.magnitude == pytest.approx(0.25, abs=1e-9)
    assert result.frame_span is not None and result.frame_span[0] % 2 == 0


def test_sigma_from_positions_is_first_order() -> None:
    case = sc.with_sigma(sc.overlapping_spheres(0.05, (0.1, 0.15)), position=0.01)
    result = by_subject(case)[("ball_a", "ball_b")]
    assert result.sigma == pytest.approx(math.sqrt(2) * 0.01 / 0.2, rel=1e-5)


def test_sigma_from_sizes_uses_the_radius_not_the_extent() -> None:
    case = sc.with_sigma(sc.overlapping_spheres(0.05, (0.1, 0.15)), size=0.01)
    result = by_subject(case)[("ball_a", "ball_b")]
    assert result.sigma == pytest.approx(math.sqrt(2) * 0.005 / 0.2, rel=1e-5)


def test_clearly_separated_bodies_have_zero_sigma_and_nan_propagates() -> None:
    apart = deeper_at(sc.overlapping_spheres(depth=0.05), ALL_FRAMES, extra=-0.15)
    result = by_subject(sc.with_sigma(apart, position=0.001))[("ball_a", "ball_b")]
    assert result.magnitude == 0.0
    assert result.sigma == 0.0
    unknown = sc.with_sigma(sc.overlapping_spheres(), position=math.nan)
    assert math.isnan(by_subject(unknown)[("ball_a", "ball_b")].sigma)
