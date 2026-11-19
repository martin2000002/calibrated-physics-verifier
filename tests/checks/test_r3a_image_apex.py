import math
from dataclasses import dataclass, replace
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.checks._collision_windows import IntArray, side_data
from physverify.checks.r3a_bounce_restitution import Bounce, RestitutionCheck, RestitutionParams
from physverify.checks.variants import r3a_image_apex as apex_module
from physverify.checks.variants.r3a_image_apex import (
    ApexRestitution,
    ImageApexRestitutionCheck,
    apex_restitution,
    run_apex,
)
from physverify.core.errors import ConfigError
from physverify.physics.tracks import poly_fit
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, ObjectStateTable
from physverify.verification.check_set import CheckVariantsSettings, select_variants
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.thresholding import ThresholdTable

SIGMA = 0.002
# e of sc.bounce(e, pitch_deg=30) read on the image plane; the camera is 4 m away and the apexes
# about 1.3 m high, so perspective moves e by up to 0.053 from the true value.
PITCHED = {0.6: 0.5699750202734197, 0.8: 0.7755790169549508, 1.2: 1.25252419140884}
GT_LIFT = {Stage.LIFT.value: GROUND_TRUTH}


@dataclass(frozen=True)
class Impact:
    """The arguments the check passes to ``apex_restitution`` for the case's single impact."""

    camera_k: Any
    states: ObjectStateTable
    index: int
    bounce: Bounce
    runs: tuple[IntArray, IntArray]
    t_c: float

    def bounce_on(self, states: ObjectStateTable) -> Bounce:
        """The bounce with its fit rows read from ``states`` (same rows)."""
        sides = [
            side_data(states, self.index, np.searchsorted(states.t, side.t), "position")
            for side in (self.bounce.before, self.bounce.after)
        ]
        return replace(self.bounce, before=sides[0], after=sides[1])

    def restitution(self, states: ObjectStateTable | None = None) -> ApexRestitution:
        """On ``states`` (same rows), with t_c and the runs held fixed."""
        table = self.states if states is None else states
        bounce = self.bounce_on(table)
        return apex_restitution(self.camera_k, table, self.index, bounce, self.runs, self.t_c)


def variant(case: sc.Case, provenance: dict[str, str] | None = None) -> CheckResult:
    inputs = case.inputs(InputLevel.PERCEIVED, provenance)
    (result,) = ImageApexRestitutionCheck(RestitutionParams()).evaluate(inputs)
    return result


def impact(case: sc.Case, monkeypatch: pytest.MonkeyPatch) -> Impact:
    seen: list[Impact] = []

    def spy(*args: Any) -> ApexRestitution:
        seen.append(Impact(*args))
        return apex_restitution(*args)

    monkeypatch.setattr(apex_module, "apex_restitution", spy)
    variant(case)
    monkeypatch.undo()
    return seen[0]


def vertical(case: sc.Case) -> sc.Case:
    """``case`` without its drift along the camera x axis (world x at every pitch)."""
    position, velocity = case.states.position.copy(), case.states.velocity.copy()
    position[..., 0], velocity[..., 0] = 0.0, 0.0
    return replace(case, states=replace(case.states, position=position, velocity=velocity))


def noisy(case: sc.Case, seed: int) -> sc.Case:
    return sc.with_sigma(sc.jittered(case, SIGMA, seed), position=SIGMA)


@pytest.mark.parametrize("e", [0.6, 0.8, 1.0, 1.2])
def test_an_affine_view_reads_the_true_restitution(e: float) -> None:
    result = variant(sc.bounce(e=e))
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude is not None and abs(result.magnitude + 1.0 - e) < 2e-3
    assert result.sigma == 0.0
    assert result.details["apex_interior_before"] and result.details["apex_interior_after"]


@pytest.mark.parametrize("e", sorted(PITCHED))
def test_a_pitched_view_is_pinned_with_its_perspective_error(e: float) -> None:
    result = variant(sc.bounce(e=e, pitch_deg=30.0))
    assert result.magnitude is not None
    assert result.magnitude + 1.0 == pytest.approx(PITCHED[e], abs=1e-12)
    assert abs(PITCHED[e] - e) < 0.06


def test_the_apex_of_a_parabola_is_its_vertex_and_the_weights_sum_to_one() -> None:
    t = np.arange(7) * sc.DT
    vertex = 2.6 * sc.DT
    apex = run_apex(t, 1.5 - 4.0 * (t - vertex) ** 2)
    assert apex.t == pytest.approx(vertex, abs=1e-12)
    assert apex.height == pytest.approx(1.5, abs=1e-12)
    assert apex.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.count_nonzero(apex.weights) == 3 and apex.interior


@pytest.mark.parametrize(
    ("s", "row", "interior"),
    [([0.0, 1.0, 2.0, 3.0], 3, False), ([2.0, 1.0, 1.5, 1.6, 0.0], 0, False)],
)
def test_an_apex_on_the_run_edge_takes_the_highest_row(
    s: list[float], row: int, interior: bool
) -> None:
    apex = run_apex(np.arange(len(s)) * sc.DT, np.array(s))
    assert apex.height == s[row] and apex.interior is interior
    assert apex.weights.tolist() == [float(i == row) for i in range(len(s))]


def test_a_tie_of_two_rows_puts_the_apex_halfway() -> None:
    """The first highest row is above its left neighbour, so its parabola is always concave
    with the vertex within one row: the probe's flat and clipped cases never occur."""
    apex = run_apex(np.arange(4) * sc.DT, np.array([0.0, 1.0, 1.0, 0.0]))
    assert apex.t == pytest.approx(1.5 * sc.DT, abs=1e-15)
    assert apex.height == pytest.approx(1.125, abs=1e-12)
    assert apex.weights == pytest.approx([-0.125, 0.75, 0.375, 0.0], abs=1e-12)


@pytest.mark.parametrize("per_row", [False, True])
def test_e_and_sigma_ignore_the_depth_scale(monkeypatch: pytest.MonkeyPatch, per_row: bool) -> None:
    found = impact(noisy(sc.bounce(pitch_deg=30.0), seed=3), monkeypatch)
    rng = np.random.default_rng(0)
    rows = found.states.n_frames
    k = rng.uniform(0.5, 2.0, (rows, 1, 1)) if per_row else np.full((rows, 1, 1), 1.7)
    scaled = replace(
        found.states,
        position=found.states.position * k,
        position_sigma=found.states.position_sigma * k,
    )
    base, moved = found.restitution(), found.restitution(scaled)
    assert moved.e == pytest.approx(base.e, rel=1e-12)
    assert moved.sigma == pytest.approx(base.sigma, rel=1e-12)


def shifted(found: Impact, row: int, step: float) -> tuple[float, float]:
    """e and s_row after moving row ``row`` in 3D so that its pixel moves ``step`` along d."""
    states, d = found.states, found.restitution().direction
    p = states.position[row, found.index]
    jacobian = (found.camera_k[:2] - np.outer(project(found.camera_k, p), found.camera_k[2])) / p[2]
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T, step * d)
    position = states.position.copy()
    position[row, found.index] = p + delta
    moved = found.restitution(replace(states, position=position))
    return moved.e, float(project(found.camera_k, p + delta) @ d)


def project(camera_k: Any, p: Any) -> Any:
    h = camera_k @ p
    return h[:2] / h[2]


def test_the_weights_are_the_gradient_of_e_in_the_heights(monkeypatch: pytest.MonkeyPatch) -> None:
    found = impact(noisy(sc.bounce(), seed=5), monkeypatch)
    weights = found.restitution().weights
    used = np.flatnonzero(np.isfinite(weights) & (np.abs(weights) > 1e-3 * np.abs(weights).max()))
    assert used.size >= 8
    for row in used:
        (e_up, s_up), (e_down, s_down) = shifted(found, row, 1e-4), shifted(found, row, -1e-4)
        assert (e_up - e_down) / (s_up - s_down) == pytest.approx(weights[row], rel=1e-6)


def test_sigma_matches_a_monte_carlo_of_the_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    scale = 1e-4
    case = sc.with_sigma(vertical(sc.bounce(pitch_deg=30.0)), position=scale)
    found = impact(case, monkeypatch)
    rng = np.random.default_rng(11)
    clean = found.states.position
    inputs = case.inputs(InputLevel.PERCEIVED)
    draws = []
    for _ in range(20_000):
        states = replace(found.states, position=clean + rng.normal(0.0, scale, clean.shape))
        bounce = found.bounce_on(states)
        draws.append(probe_e(replace(inputs, states=states), bounce, found.runs, found.t_c))
    assert float(np.std(draws, ddof=1)) == pytest.approx(found.restitution().sigma, rel=0.1)


def test_sigma_is_zero_exact_linear_in_a_common_factor_and_unknown_with_a_nan_row() -> None:
    case = sc.jittered(sc.bounce(pitch_deg=30.0), SIGMA, seed=7)
    assert variant(case).sigma == 0.0
    one, three = (variant(sc.with_sigma(case, position=f * SIGMA)) for f in (1.0, 3.0))
    assert three.sigma == pytest.approx(3.0 * one.sigma, rel=1e-12)
    assert three.magnitude == one.magnitude
    sigma = case.states.position_sigma + SIGMA
    sigma[20, 0, 1] = math.nan
    unknown = variant(replace(case, states=replace(case.states, position_sigma=sigma)))
    assert unknown.verdict is Verdict.UNSCORED and math.isnan(unknown.sigma)
    table = ThresholdTable.model_validate(
        {"status": "provisional", "source": "test", "checks": {"R3a": {"tau": 0.05}}}
    )
    decided = ThresholdPolicy().decide(unknown, table)
    assert decided.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        (sc.bounce(e=-0.3), AbstainReason.ILL_CONDITIONED),
        (sc.bounce(speed=0.05), AbstainReason.ILL_CONDITIONED),
        (sc.bounce(impact_frame=1.4), AbstainReason.TOO_FEW_FRAMES),
    ],
    ids=["no_climb", "grazing", "too_few_frames"],
)
def test_the_variant_abstains_where_e_is_not_defined(case: sc.Case, reason: AbstainReason) -> None:
    assert variant(case).abstain_reason is reason


def test_without_a_camera_frame_the_variant_misses_its_input() -> None:
    case = sc.bounce()
    no_camera = replace(case.inputs(InputLevel.PERCEIVED), camera=None)
    world = replace(case, states=replace(case.states, frame=CoordFrame.WORLD_ZUP))
    check = ImageApexRestitutionCheck(RestitutionParams())
    for inputs in (no_camera, world.inputs(InputLevel.PERCEIVED)):
        (result,) = check.evaluate(inputs)
        assert result.abstain_reason is AbstainReason.MISSING_INPUT


@pytest.mark.parametrize("case", [sc.bounce(pitch_deg=30.0), noisy(sc.bounce(), seed=2)])
def test_a_ground_truth_lift_is_scored_by_the_default_implementation(case: sc.Case) -> None:
    inputs = case.inputs(InputLevel.ORACLE, GT_LIFT)
    (default,) = RestitutionCheck(RestitutionParams()).evaluate(inputs)
    (chosen,) = ImageApexRestitutionCheck(RestitutionParams()).evaluate(inputs)
    assert (chosen.magnitude, chosen.sigma, chosen.abstain_reason) == (
        default.magnitude,
        default.sigma,
        default.abstain_reason,
    )
    assert {k: v for k, v in chosen.details.items() if k != "tangent_bias"} == default.details
    assert math.isfinite(chosen.details["tangent_bias"])


def test_it_is_the_selectable_image_apex_variant_of_r3a(make_settings: SettingsFactory) -> None:
    chosen = plugins.build("check_variants")["R3a/image_apex"]
    assert chosen is ImageApexRestitutionCheck
    assert chosen.requires == RestitutionCheck.requires
    assert chosen.magnitude_unit == RestitutionCheck.magnitude_unit
    selected = plugins.configured("checks", make_settings("check_variants.R3a=image_apex"))
    assert selected["R3a"] is ImageApexRestitutionCheck
    for profile in (None, "fast"):
        assert plugins.configured("checks", make_settings(profile=profile))["R3a"] is (
            RestitutionCheck
        )
    with pytest.raises(ConfigError, match="check_variants"):
        select_variants(
            plugins.build("checks"),
            plugins.build("check_variants"),
            CheckVariantsSettings.model_validate({"R4_size": "image_apex"}),
        )


def probe_e(inputs: Any, bounce: Bounce, runs: tuple[IntArray, IntArray], t_c: float) -> float:
    """The h estimator of the r3a-imgplane probe (checks/variants/r3a_image_plane.py there),
    copied as an oracle."""
    camera_k = np.asarray(inputs.camera.K, dtype=np.float64)

    def project_rows(positions: Any) -> Any:
        homogeneous = positions @ camera_k.T
        return np.asarray(homogeneous[:, :2] / homogeneous[:, 2:3], dtype=np.float64)

    def apex(t: Any, s: Any) -> tuple[float, float]:
        i = int(np.argmax(s))
        if 0 < i < s.size - 1:
            a, b, c = np.polyfit(t[i - 1 : i + 2] - t[i], s[i - 1 : i + 2], 2)
            if a < 0.0:
                dt = min(max(-b / (2.0 * a), t[i - 1] - t[i]), t[i + 1] - t[i])
                return float(t[i] + dt), float(c + b * dt + a * dt * dt)
        return float(t[i]), float(s[i])

    ones = np.ones(1)
    fits = [
        poly_fit(side.t, project_rows(side.y), ones, 2) for side in (bounce.before, bounce.after)
    ]
    pixel = fits[0].value_at(t_c)[0]
    direction = (camera_k[:2] - np.outer(pixel, camera_k[2])) @ bounce.normal
    d = direction / np.linalg.norm(direction)
    s_c = float(np.mean([fit.value_at(t_c)[0] @ d for fit in fits]))
    states = inputs.states
    heights = [apex(states.t[rows], project_rows(states.position[rows, 0]) @ d)[1] for rows in runs]
    drop, climb = heights[0] - s_c, heights[1] - s_c
    return math.sqrt(climb / drop) if drop > 0.0 and climb > 0.0 else math.nan


@pytest.mark.parametrize("seed", [1, 4])
def test_e_is_bit_identical_to_the_probe_estimator(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    case = noisy(sc.bounce(e=0.9, pitch_deg=30.0), seed)
    found = impact(case, monkeypatch)
    inputs = case.inputs(InputLevel.PERCEIVED)
    expected = probe_e(inputs, found.bounce, found.runs, found.t_c)
    assert variant(case).details["e"] == expected
