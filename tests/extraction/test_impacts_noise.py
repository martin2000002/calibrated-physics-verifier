"""The impact detector on tracks with white noise (seed fixed, 400 replicates).

A chi-square rule (the track noise as a known sigma) gives 23 % of clean parabolas
a false impact, because the MAD of 29 third differences often comes out low; the F ratio over the
window's own residual (floored at that noise) brings it to 1.25 % and costs power: a bounce with
dv dt = 20 sigma is found in 85.5 % of the replicates, 30 sigma in 99.5 % (the measured figures the
tests print). The bars: clean false impacts <= 8 %, 20 sigma >= 80 %, 30 sigma >= 95 %,
teleport 0.25 L <= 8 %, and the gravity perturbations (R5a switch, R5a object gravity, R5b scale)
<= 8 % false impacts per level. The correlated-noise description (AR(1), two-row depth impulses)
is printed by impact_noise.py; its test only checks that it runs.
"""

from __future__ import annotations

import impact_noise as mc
import pytest

from physverify.synth.perturbations.r5a_gravity_switch import Params as SwitchLevels
from physverify.synth.perturbations.r5a_object_gravity import Params as ObjectLevels
from physverify.synth.perturbations.r5b_gravity_scale import Params as ScaleLevels

sc = mc.sc
FALSE_MAX = 0.08
SIGMA = 0.01
BOUNCE = {"e": 0.8, "speed": 5.0, "impact_frame": 15.3}
JUMP = (1.0 + BOUNCE["e"]) * BOUNCE["speed"] * sc.DT


def test_clean_parabola_false_rate() -> None:
    value = mc.rate(mc.clean_parabola(), mc.white(SIGMA))
    print(f"clean parabola: {value:.4f}")
    assert value <= FALSE_MAX


@pytest.mark.parametrize(("sigmas", "least"), [(20.0, 0.80), (30.0, 0.95)])
def test_a_bounce_is_found(sigmas: float, least: float) -> None:
    value = mc.rate(sc.bounce(pitch_deg=25.0, **BOUNCE), mc.white(JUMP / sigmas))
    print(f"bounce dv dt = {sigmas} sigma: {value:.4f}")
    assert value >= least


def test_a_small_teleport_is_not_an_impact() -> None:
    length = 2 * sc.BALL_RADIUS
    case = sc.teleport(delta=0.25 * length, flight=True, pitch_deg=25.0)
    value = mc.rate(case, mc.white(0.02 * length))
    print(f"teleport 0.25 L, sigma 0.02 L: {value:.4f}")
    assert value <= FALSE_MAX


def gravity_cases(level: float) -> dict[str, sc.Case]:
    g = sc.STANDARD_GRAVITY
    return {
        "r5a_gravity_switch": sc.parabola(
            g_after=g * (1.0 + level), change_frame=15.5, pitch_deg=25.0
        ),
        "r5a_object_gravity": sc.parabola(g=g * (1.0 + level), pitch_deg=25.0),
        "r5b_gravity_scale": sc.parabola(g=g * (1.0 - level), pitch_deg=25.0),
    }


@pytest.mark.parametrize("kind", ["r5a_gravity_switch", "r5a_object_gravity", "r5b_gravity_scale"])
def test_gravity_perturbations_are_not_impacts(kind: str) -> None:
    levels = {
        "r5a_gravity_switch": SwitchLevels().levels,
        "r5a_object_gravity": ObjectLevels().levels,
        "r5b_gravity_scale": ScaleLevels().levels,
    }[kind]
    for level in levels:
        value = mc.rate(gravity_cases(level)[kind], mc.white(SIGMA))
        print(f"{kind} L={level}: {value:.4f}")
        assert value <= FALSE_MAX, (kind, level)


def test_the_correlated_noise_description_runs() -> None:
    """The description itself (AR(1), two-row depth impulses) is printed by impact_noise.py
    with 400 replicates; here only that it runs."""
    values = mc.description(replicates=5)
    assert len(values) == 7 and all(0.0 <= v <= 1.0 for v in values.values())
