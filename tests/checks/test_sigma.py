"""Closed-form sigmas against Monte Carlo, NaN propagation, exact oracle, scale in quadrature."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import Check, CheckInputs
from physverify.cli.context import PLUGIN_KINDS
from physverify.schema.checks import CheckResult
from physverify.schema.constants import STANDARD_GRAVITY

DRAWS = 4000
RATIO_BAND = (0.9, 1.1)
POSITION_SIGMA = 0.002
VELOCITY_SIGMA = 0.01
REL_SIGMA = 0.05
G_FACTOR = 1.2


def make(check_id: str) -> Check:
    cls = PLUGIN_KINDS["checks"]()[check_id]
    check: Check = cls(cls.Params())
    return check


def scored(check: Check, inputs: CheckInputs) -> CheckResult:
    results = [r for r in check.evaluate(inputs) if r.magnitude is not None]
    assert results, f"{check.check_id} gave no magnitude"
    return results[0]


def monte_carlo_ratio(
    check_id: str, case: sc.Case, noise: Mapping[str, float], weighted: bool
) -> float:
    """Sample std of the magnitude under Gaussian input noise over the reported sigma.

    ``weighted`` keeps the input sigmas in every draw (checks whose fits weight by them); else
    the draws carry zero sigma so the check skips its own propagation.
    """
    check = make(check_id)
    noisy = sc.with_sigma(case, **noise)
    reference = scored(check, noisy.inputs())
    draw_case = noisy if weighted else sc.with_sigma(case, **dict.fromkeys(noise, 0.0))
    base = {name: getattr(case.states, name) for name in noise}
    draw_inputs = draw_case.inputs()
    rng = np.random.default_rng(0)
    values = []
    for _ in range(DRAWS):
        changes = {
            name: base[name] + rng.normal(0.0, sigma, base[name].shape)
            for name, sigma in noise.items()
        }
        states = replace(draw_case.states, **changes)
        values.append(scored(check, replace(draw_inputs, states=states)).magnitude)
    return float(np.std(values, ddof=1)) / reference.sigma


def stronger_gravity(frames: int = sc.FRAMES) -> sc.Case:
    return sc.parabola(g=G_FACTOR * STANDARD_GRAVITY, frames=frames)


def short_bounce() -> sc.Case:
    return sc.bounce(frames=20, impact_frame=9.4)


@pytest.mark.parametrize(
    ("check_id", "case", "noise", "weighted"),
    [
        (
            "R1",
            lambda: sc.overlapping_spheres(frames=1),
            {"position": POSITION_SIGMA, "size": POSITION_SIGMA},
            False,
        ),
        ("R3a", short_bounce, {"velocity": VELOCITY_SIGMA}, True),
        ("R4_jump", lambda: sc.teleport(frames=12, frame=6), {"position": POSITION_SIGMA}, True),
        ("R5b", stronger_gravity, {"position": POSITION_SIGMA}, True),
        (
            "R5b",
            lambda: sc.strided(stronger_gravity(), 2, quantize=True),
            {"position": POSITION_SIGMA},
            True,
        ),
    ],
    ids=["R1", "R3a", "R4_jump", "R5b", "R5b_stride2"],
)
def test_closed_form_sigma_matches_monte_carlo(
    check_id: str, case: Callable[[], sc.Case], noise: Mapping[str, float], weighted: bool
) -> None:
    ratio = monte_carlo_ratio(check_id, case(), noise, weighted)
    assert RATIO_BAND[0] <= ratio <= RATIO_BAND[1]


@pytest.mark.parametrize(
    ("check_id", "case", "unknown"),
    [
        ("R1", sc.overlapping_spheres, ("position",)),
        ("R3a", sc.bounce, ("velocity",)),
        ("R4_jump", sc.teleport, ("position",)),
        ("R5b", stronger_gravity, ("position",)),
    ],
)
def test_unknown_input_sigma_gives_nan(
    check_id: str, case: Callable[[], sc.Case], unknown: Sequence[str]
) -> None:
    inputs = sc.with_sigma(case(), **dict.fromkeys(unknown, math.nan)).inputs()
    assert math.isnan(scored(make(check_id), inputs).sigma)


ORACLE_CASES: Mapping[str, Callable[[], sc.Case]] = {
    "R1": sc.overlapping_spheres,
    "R2": sc.stack,
    "R3a": sc.bounce,
    "R3b": sc.collision,
    "R4_missing": sc.disappearance,
    "R4_size": sc.size_change,
    "R4_jump": sc.teleport,
    "R5a": sc.parabola,
    "R5b": stronger_gravity,
}


@pytest.mark.parametrize("check_id", sorted(ORACLE_CASES))
def test_the_oracle_has_zero_sigma(check_id: str) -> None:
    results = make(check_id).evaluate(ORACLE_CASES[check_id]().inputs())
    scored_results = [r for r in results if r.magnitude is not None]
    assert scored_results
    assert all(r.sigma == 0.0 for r in scored_results)


def test_the_scale_sigma_adds_in_quadrature() -> None:
    check = make("R5b")
    scale = sc.oracle_scale(rel_sigma=REL_SIGMA)
    noisy = sc.with_sigma(stronger_gravity(), position=POSITION_SIGMA)
    position_only = scored(check, noisy.inputs())
    scale_only = scored(check, replace(stronger_gravity(), scale=scale).inputs())
    both = scored(check, replace(noisy, scale=scale).inputs())
    assert position_only.magnitude == pytest.approx(G_FACTOR - 1.0, abs=1e-9)
    assert scale_only.sigma == pytest.approx(G_FACTOR * REL_SIGMA, rel=1e-9)
    assert both.sigma == pytest.approx(math.hypot(position_only.sigma, scale_only.sigma), rel=1e-9)
