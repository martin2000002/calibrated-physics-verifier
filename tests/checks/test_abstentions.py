"""ADR 0006 §3.5: every (check, reason) pair is reachable and no check emits another reason."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks import r2_equilibrium
from physverify.checks._base import Check, CheckInputs
from physverify.cli.context import PLUGIN_KINDS
from physverify.physics.equilibrium import EquilibriumSolution
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ManifoldKind
from physverify.schema.scale import ScaleSource
from physverify.schema.states import PrimitiveKind
from physverify.verification.inputs import ClipInputs
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import VerificationRunner

R = AbstainReason
RUNNER = frozenset({R.MISSING_INPUT})
ALLOWED: dict[str, frozenset[AbstainReason]] = {
    "R1": frozenset({R.OCCLUDED, R.MISSING_INPUT}),
    "R2": frozenset(
        {R.MISSING_INPUT, R.CONTACT_GEOMETRY_UNKNOWN, R.OUT_OF_SCOPE, R.ILL_CONDITIONED}
    ),
    "R3a": frozenset({R.TOO_FEW_FRAMES, R.ILL_CONDITIONED, R.OUT_OF_SCOPE}),
    "R3b": frozenset({R.ILL_CONDITIONED, R.OUT_OF_SCOPE, R.TOO_FEW_FRAMES}),
    "R4_missing": frozenset({R.OCCLUDED}),
    "R4_size": frozenset({R.TOO_FEW_FRAMES}),
    "R4_jump": frozenset({R.TOO_FEW_FRAMES, R.UNKNOWN_UNCERTAINTY}),
    "R5a": frozenset({R.TOO_FEW_FRAMES}),
    "R5b": frozenset({R.SCALE_UNAVAILABLE, R.SCALE_FROM_GRAVITY, R.TOO_FEW_FRAMES}),
}
ALL_FRAMES = range(sc.FRAMES)


def make(check_id: str) -> Check:
    cls = PLUGIN_KINDS["checks"]()[check_id]
    check: Check = cls(cls.Params())
    return check


def with_states(case: sc.Case, **changes: Any) -> sc.Case:
    return replace(case, states=replace(case.states, **changes))


def behind_the_camera(case: sc.Case) -> sc.Case:
    """Depth z < 0: no pixel floor, so R4_jump v2 has no noise unit (UNKNOWN_UNCERTAINTY)."""
    return with_states(case, position=case.states.position * np.array([1.0, 1.0, -1.0]))


def point_contact(case: sc.Case) -> sc.Case:
    assert case.contacts is not None
    records = tuple(
        replace(
            r,
            manifold=ManifoldKind.POINT,
            points=r.points[:1],
            point_normals=None,
            point_distances=None,
        )
        if r.object_a == "top"
        else r
        for r in case.contacts.records
    )
    return replace(case, contacts=replace(case.contacts, records=records))


def ground_record(t_start: float, t_end: float) -> ContactRecord:
    return ContactRecord(
        object_a="ball_i",
        object_b=ENV_ID,
        t_start=t_start,
        t_end=t_end,
        kind=ContactKind.IMPACT,
        normal=np.array([0.0, 0.0, 1.0]),
        points=np.array([[-0.1, 0.0, 0.9]]),
        manifold=ManifoldKind.POINT,
        confidence=1.0,
        source="test",
    )


def hidden_ball_b() -> sc.Case:
    case = sc.with_values(sc.overlapping_spheres(), "visible", "ball_b", ALL_FRAMES, False)
    return sc.with_values(case, "occlusion", "ball_b", ALL_FRAMES, 0.95)


Builder = Callable[[], sc.Case]
REACHABLE: list[tuple[str, AbstainReason, Builder]] = [
    ("R1", R.OCCLUDED, hidden_ball_b),
    (
        "R1",
        R.MISSING_INPUT,
        lambda: sc.with_values(sc.overlapping_spheres(), "position", "ball_a", ALL_FRAMES, np.nan),
    ),
    (
        "R2",
        R.MISSING_INPUT,
        lambda: with_states(sc.stack(), gravity_dir=np.full(3, np.nan), gravity_dir_sigma=np.nan),
    ),
    ("R2", R.CONTACT_GEOMETRY_UNKNOWN, lambda: point_contact(sc.stack())),
    ("R2", R.OUT_OF_SCOPE, lambda: with_states(sc.stack(), labels=np.array(["box", "hand"]))),
    ("R3a", R.TOO_FEW_FRAMES, lambda: sc.bounce(impact_frame=1.4)),
    ("R3a", R.ILL_CONDITIONED, lambda: sc.bounce(speed=0.05)),
    ("R3a", R.OUT_OF_SCOPE, lambda: sc.bounce(kind=PrimitiveKind.BOX)),
    ("R3b", R.ILL_CONDITIONED, lambda: sc.collision(skew_deg=40.0)),
    ("R3b", R.OUT_OF_SCOPE, lambda: sc.with_records(sc.collision(), ground_record(15.0, 15.0))),
    ("R3b", R.TOO_FEW_FRAMES, lambda: sc.collision(impact_frame=1.4)),
    (
        "R4_missing",
        R.OCCLUDED,
        lambda: sc.with_values(sc.disappearance(4, 7), "occlusion", "ball", range(4, 8), np.nan),
    ),
    (
        "R4_size",
        R.TOO_FEW_FRAMES,
        lambda: sc.with_values(sc.rolling_ball(), "size", "ball", range(2, sc.FRAMES), np.nan),
    ),
    (
        "R4_jump",
        R.TOO_FEW_FRAMES,
        lambda: sc.with_values(
            sc.rolling_ball(), "position", "ball", range(1, sc.FRAMES, 2), np.nan
        ),
    ),
    ("R4_jump", R.UNKNOWN_UNCERTAINTY, lambda: behind_the_camera(sc.rolling_ball())),
    ("R5a", R.TOO_FEW_FRAMES, lambda: sc.parabola(frames=5)),
    ("R5b", R.SCALE_UNAVAILABLE, lambda: replace(sc.parabola(), scale=None)),
    (
        "R5b",
        R.SCALE_FROM_GRAVITY,
        lambda: replace(sc.parabola(), scale=sc.oracle_scale(source=ScaleSource.FREE_FALL)),
    ),
    ("R5b", R.TOO_FEW_FRAMES, lambda: sc.parabola(frames=3)),
]
CLEAN: list[Builder] = [
    sc.overlapping_spheres,
    sc.stack,
    sc.floating_box,
    sc.bounce,
    sc.collision,
    sc.rolling_ball,
    sc.teleport,
    sc.parabola,
    lambda: sc.strided(sc.bounce(), 2, quantize=True),
    lambda: sc.strided(sc.parabola(), 2, quantize=False),
]


def reasons(check_id: str, inputs: CheckInputs) -> set[AbstainReason]:
    return {r.abstain_reason for r in make(check_id).evaluate(inputs) if r.abstain_reason}


def ids(entry: tuple[str, AbstainReason, Builder]) -> str:
    return f"{entry[0]}-{entry[1].value}"


def test_the_table_covers_the_nine_checks() -> None:
    assert set(ALLOWED) == set(PLUGIN_KINDS["checks"]())
    listed = {(check_id, reason) for check_id, reason, _ in REACHABLE}
    expected = {(c, r) for c, found in ALLOWED.items() for r in found}
    assert expected - listed == {("R2", R.ILL_CONDITIONED)}


@pytest.mark.parametrize("entry", REACHABLE, ids=[ids(e) for e in REACHABLE])
def test_each_reason_is_reachable(entry: tuple[str, AbstainReason, Builder]) -> None:
    check_id, reason, build = entry
    assert reason in reasons(check_id, build().inputs())


def test_r2_reaches_ill_conditioned_when_the_program_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(*_: object) -> EquilibriumSolution:
        nan = np.full(2, math.nan)
        return EquilibriumSolution(math.nan, nan, nan, 4, "numerical difficulties")

    monkeypatch.setattr(r2_equilibrium, "solve_equilibrium", failing)
    assert R.ILL_CONDITIONED in reasons("R2", sc.stack().inputs())


@pytest.fixture(scope="module")
def corpus() -> list[CheckInputs]:
    return [build().inputs() for build in [*CLEAN, *(b for _, _, b in REACHABLE)]]


@pytest.mark.parametrize("check_id", sorted(ALLOWED))
def test_no_check_emits_a_reason_outside_its_row(corpus: list[CheckInputs], check_id: str) -> None:
    emitted: set[AbstainReason] = set()
    for inputs in corpus:
        emitted |= reasons(check_id, inputs)
    assert emitted <= ALLOWED[check_id] | RUNNER


class OneClip:
    def __init__(self, item: ClipInputs) -> None:
        self._item = item

    def load(self, clips: Any) -> Any:
        return iter([self._item])


def runner_reasons(item: ClipInputs) -> dict[str, set[AbstainReason]]:
    checks = [make(check_id) for check_id in sorted(ALLOWED)]
    report = VerificationRunner(checks, RawPolicy(), {}).run(OneClip(item), [item.clip])
    found: dict[str, set[AbstainReason]] = {}
    for result in report.results:
        if result.abstain_reason is not None:
            found.setdefault(result.check_id, set()).add(result.abstain_reason)
    return found


def test_the_runner_abstains_with_missing_input_for_every_check() -> None:
    case = sc.parabola()
    broken = ClipInputs(case.clip, None, "time_axis: not uniform", InputLevel.ORACLE)
    assert runner_reasons(broken) == {check_id: {R.MISSING_INPUT} for check_id in ALLOWED}
    without_contacts = ClipInputs(case.clip, replace(case.inputs(), contacts=None), None)
    found = runner_reasons(without_contacts)
    needing = {c for c in ALLOWED if "contacts" in make(c).requires}
    assert needing and all(found[c] == {R.MISSING_INPUT} for c in needing)


def test_the_policy_reasons_are_reachable() -> None:
    def result(sigma: float) -> CheckResult:
        return CheckResult(
            check_id="R1",
            clip_id="c",
            subject=("a",),
            frame_span=None,
            magnitude=0.1,
            sigma=sigma,
            unit="pen_over_size",
            verdict=Verdict.UNSCORED,
            input_level=InputLevel.ORACLE,
        )

    thresholds = {"status": "provisional", "source": "test", "checks": {"R1": {"tau": 0.1}}}
    policy = ThresholdPolicy()
    assert policy.decide(result(math.nan), thresholds).abstain_reason is R.UNKNOWN_UNCERTAINTY
    crosses = policy.decide(result(0.01), thresholds).abstain_reason
    assert crosses is R.INTERVAL_CROSSES_THRESHOLD
