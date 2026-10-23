"""R4_size v2, the size relative to the other bodies of each frame."""

from __future__ import annotations

import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks.r4_size_constancy import SizeConstancyCheck, SizeParams
from physverify.physics.measures.relative_size import relative_lengths, relative_size_change
from physverify.schema.checks import AbstainReason, CheckResult, Verdict


def size(case: sc.Case, reference: str = "others") -> dict[str, CheckResult]:
    check = SizeConstancyCheck(SizeParams(reference=reference))
    return {r.subject[0]: r for r in check.evaluate(case.inputs())}


def common_factor(case: sc.Case, factors: np.ndarray) -> sc.Case:
    """Every object's extents times the row's factor (a depth-scale error common to the row)."""
    sizes = case.states.size * factors[:, None, None]
    return replace(case, states=replace(case.states, size=sizes))


def test_a_common_factor_per_row_cancels() -> None:
    case = sc.resized_pair(factor=1.0)
    factors = 1.0 + 0.2 * np.sin(np.arange(sc.FRAMES) / 3.0)
    scaled = common_factor(case, factors)
    relative = size(scaled)
    alone = size(scaled, "self")
    expected = float(np.max(np.abs(factors / np.median(factors) - 1.0)))
    for name in ("ball_0", "ball_1"):
        assert relative[name].magnitude == pytest.approx(0.0, abs=1e-12)
        assert alone[name].magnitude == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("balls", [2, 3, 4])
@pytest.mark.parametrize("mu", [0.05, 0.4])
def test_a_resized_body_against_a_constant_reference_gives_mu(balls: int, mu: float) -> None:
    results = size(sc.resized_pair(factor=1.0 + mu, balls=balls))
    target = results["ball_0"]
    assert target.magnitude == pytest.approx(mu, rel=1e-12)
    assert target.sigma == 0.0
    assert target.details["reference"] == "others"
    assert target.frame_span == (12, 12)


@pytest.mark.parametrize(("balls", "expected"), [(2, lambda m: m / (1 + m)), (3, None)])
def test_a_clean_partner_also_moves(balls: int, expected: object) -> None:
    """With two bodies the partner's ratio is 1 / (1 + mu): the clip is flagged by the larger,
    the subject stays ambiguous (the human-agreement study makes no per-subject use of it)."""
    mu = 0.4
    partner = size(sc.resized_pair(factor=1.0 + mu, balls=balls))["ball_1"]
    if balls == 2:
        assert partner.magnitude == pytest.approx(mu / (1 + mu), rel=1e-12)
    else:
        assert partner.magnitude == pytest.approx(abs(1 / math.sqrt(1 + mu) - 1), rel=1e-12)
    assert partner.magnitude < mu


def test_a_single_body_abstains_with_no_reference() -> None:
    (result,) = size(sc.size_change()).values()
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES
    assert result.details["reference"] == "none"
    assert result.details["size_rows"] == 0


def test_rows_without_a_reference_in_the_image_do_not_count() -> None:
    case = sc.resized_pair(factor=1.5, first=12, last=17)
    gone = range(0, sc.FRAMES - 3)
    hidden = sc.with_values(case, "in_frame", "ball_1", gone, False)
    result = size(hidden)["ball_0"]
    assert result.verdict is Verdict.UNSCORED
    assert result.details["size_rows"] == 3
    assert result.details["unreferenced_rows"] == len(gone)
    assert result.magnitude == pytest.approx(0.0, abs=1e-12)
    fewer = sc.with_values(case, "in_frame", "ball_1", range(0, sc.FRAMES - 2), False)
    assert size(fewer)["ball_0"].abstain_reason is AbstainReason.TOO_FEW_FRAMES
    occluded = sc.with_values(case, "occlusion", "ball_1", range(0, sc.FRAMES - 2), 0.95)
    assert size(occluded)["ball_0"].details["reference"] == "others"
    assert size(occluded)["ball_0"].abstain_reason is AbstainReason.TOO_FEW_FRAMES


def test_the_sigma_adds_the_reference_in_quadrature() -> None:
    sigma = 0.002
    case = sc.with_sigma(sc.resized_pair(factor=1.3, balls=3), size=sigma)
    result = size(case)["ball_0"]
    length = 2 * sc.BALL_RADIUS
    own = math.sqrt(3 * (sigma / (1.3 * length)) ** 2) / 3
    other = math.sqrt(3 * (sigma / length) ** 2) / 3
    reference = math.sqrt(other**2 / 2)
    assert result.sigma == pytest.approx(1.3 * math.sqrt(2) * math.hypot(own, reference), rel=1e-9)


def test_relative_lengths_use_each_reference_relative_to_its_median() -> None:
    """Column 0 has median 1 and column 1 median 2: row 0 divides by sqrt(1 * 4 / 2) = sqrt(2),
    row 1 by 2 / 2 = 1."""
    own = np.array([1.0, 2.0, math.nan, 4.0])
    reference = np.array([[1.0, 4.0], [math.nan, 2.0], [1.0, 1.0], [math.nan, math.nan]])
    ratios = relative_lengths(own, reference)
    assert ratios[:2] == pytest.approx([1 / math.sqrt(2), 2.0])
    assert np.isnan(ratios[2:]).all()
    scaled = ratios[:2] / np.median(ratios[:2])
    assert relative_size_change(own, reference) == pytest.approx(np.max(np.abs(scaled - 1)))
    assert math.isnan(relative_size_change(own[2:], reference[2:]))


@pytest.mark.parametrize("leave", [12, 16, 20])
def test_a_reference_leaving_the_image_makes_no_step(leave: int) -> None:
    """Audit A1: three rigid bodies of lengths 0.2, 0.2 and 0.1; the small one leaves the image
    from row ``leave`` on. With the plain geometric mean the denominator stepped (0.29 at row 16);
    with each reference relative to its median every body stays at 0."""
    case = sc.resized_pair(factor=1.0, balls=3)
    sizes = case.states.size.copy()
    sizes[:, 2] *= 0.5
    case = replace(case, states=replace(case.states, size=sizes))
    gone = sc.with_values(case, "in_frame", "ball_2", range(leave, sc.FRAMES), False)
    occluded = sc.with_values(case, "occlusion", "ball_2", range(leave, leave + 6), 0.95)
    for scene in (gone, occluded):
        results = size(scene)
        for name in ("ball_0", "ball_1"):
            assert results[name].magnitude == pytest.approx(0.0, abs=1e-12)
        assert results["ball_2"].magnitude == pytest.approx(0.0, abs=1e-12)
