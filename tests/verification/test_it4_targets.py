"""it4 judgement J1: r4_teleport stays the target of R4_jump v2 although the check measures
``jump_over_noise`` and the perturbation is labelled ``jump_over_size``.

Roles (calibration, devscore) are the v1 roles for every perturbation; in the sweep the injected
jump is a parameter whose order the check must keep (per scene and edited object), not a value it
must reproduce."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace

import check_scenarios as sc
import pytest
from reference.r4_jump_v1 import ContinuityCheck

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.table import role_of
from physverify.checks._base import Check, CheckInputs
from physverify.checks._results import unscored
from physverify.checks.r4_track_jump import NoiseContinuityCheck
from physverify.schema.checks import CheckResult
from physverify.schema.clip import ClipManifest
from physverify.schema.scene import PerturbationRecord
from physverify.verification.inputs import ClipInputs
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationRunner
from physverify.verification.sweep import (
    CheckUnit,
    Rule,
    SpecificitySettings,
    specificity,
    target_check_ids,
)

COUPLINGS = frozenset({("r5a_gravity_switch", "R5b"), ("r5a_gravity_switch", "R4_jump")})


def v2_units() -> list[CheckUnit]:
    return [CheckUnit.of(cls) for cls in PLUGIN_KINDS["checks"]().values()]


def v1_units() -> list[CheckUnit]:
    return [CheckUnit.of(ContinuityCheck) if c.check_id == "R4_jump" else c for c in v2_units()]


def label(name: str, level: int = 1, objects: tuple[str, ...] = ("a",)) -> PerturbationRecord:
    plugin = PLUGIN_KINDS["perturbations"]()[name]
    return PerturbationRecord(
        name=name,
        target_checks=plugin.target_checks,
        magnitude=0.25 * 2**level,
        magnitude_unit=plugin.magnitude_unit,
        magnitude_kind=plugin.magnitude_kind,
        level=level,
        target_objects=objects,
        frame_span=None,
        seed=0,
    )


def manifest(name: str | None, scene: str = "s0") -> ClipManifest:
    record = None if name is None else label(name)
    clip_id = f"{scene}__{name or 'none'}"
    return replace(sc.parabola().clip, clip_id=clip_id, perturbation=record, scene_id=scene)


NAMES = sorted(name for name in PLUGIN_KINDS["perturbations"]() if name != "none")


@pytest.mark.parametrize("name", [None, *NAMES])
def test_j1_every_role_is_the_v1_role(name: str | None) -> None:
    clip = manifest(name)
    for check in v2_units():
        v1 = role_of(clip, check.check_id, v1_units(), COUPLINGS)
        v2 = role_of(clip, check.check_id, v2_units(), COUPLINGS)
        assert v1 == v2, (name, check.check_id)


def test_j1_the_r4_family_resolves_as_before() -> None:
    targets = {name: target_check_ids(label(name), v2_units()) for name in NAMES}
    assert targets["r4_teleport"] == ("R4_jump",)
    assert targets["r4_resize"] == ("R4_size",)
    assert targets["r4_vanish"] == ("R4_missing",)


class FakeJump:
    """R4_jump v2 as the sweep sees it: a unit of its own answering to jump_over_size."""

    check_id = "R4_jump"
    requires: frozenset[str] = frozenset()
    magnitude_unit = "jump_over_noise"
    perturbation_unit = "jump_over_size"

    def __init__(self, values: dict[str, float]) -> None:
        self.values = values

    def evaluate(self, inputs: CheckInputs) -> Sequence[CheckResult]:
        value = self.values[inputs.clip.clip_id]
        record = inputs.clip.perturbation
        subject = ("a",) if record is None else tuple(record.target_objects)
        return [unscored(type(self), inputs, subject, value, 1.0, None, {})]  # type: ignore[arg-type]


class Items:
    def __init__(self, items: Sequence[ClipInputs]) -> None:
        self.items = list(items)

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        return iter(self.items)


def teleport(level: int, obj: str, scene: str = "s0") -> ClipInputs:
    record = label("r4_teleport", level, (obj,))
    case = sc.parabola(clip_id=f"{scene}_{obj}_L{level}")
    clip = replace(case.clip, perturbation=record, scene_id=scene)
    return ClipInputs(clip, replace(case.inputs(), clip=clip), None)


def sweep(items: list[ClipInputs], values: dict[str, float]) -> list[object]:
    check: Check = FakeJump(values)  # type: ignore[assignment]
    report = VerificationRunner([check], RawPolicy(), {}).run(Items(items), [i.clip for i in items])
    spec = specificity(report, SpecificitySettings(), [CheckUnit.of(check)])
    return [o for o in spec.outcomes if o.check_id == "R4_jump"]


def test_j1_the_sweep_ranks_the_levels_of_each_edited_object() -> None:
    items = [teleport(1, "a"), teleport(3, "a"), teleport(2, "b"), teleport(4, "b")]
    values = {"s0_a_L1": 3.0, "s0_a_L3": 9.0, "s0_b_L2": 1.0, "s0_b_L4": 2.0}
    outcomes = sweep(items, values)
    assert {o.rule for o in outcomes} == {Rule.INJECTED_PARAMETER}  # type: ignore[attr-defined]
    assert len(outcomes) == 2
    assert all(o.passed for o in outcomes)  # type: ignore[attr-defined]


def test_j1_a_wrong_order_within_an_object_fails() -> None:
    items = [teleport(1, "a"), teleport(3, "a")]
    outcomes = sweep(items, {"s0_a_L1": 9.0, "s0_a_L3": 3.0})
    (outcome,) = outcomes
    assert not outcome.passed  # type: ignore[attr-defined]


def test_j1_one_level_per_object_has_no_order_to_keep() -> None:
    (outcome,) = sweep([teleport(2, "a")], {"s0_a_L2": 5.0})
    assert outcome.undefined  # type: ignore[attr-defined]


def real_teleport(mu: float) -> tuple[list[object], list[object]]:
    """R4_jump v2 on the oracle teleport scene (0.3 m = 1.5 body lengths at rest), labelled as
    one r4_teleport level of magnitude ``mu``: (check-unit outcomes, ranking outcomes)."""
    case = sc.teleport(delta=0.3)
    record = replace(label("r4_teleport", 3, ("ball",)), magnitude=mu)
    clip = replace(case.clip, perturbation=record, scene_id="s0")
    item = ClipInputs(clip, replace(case.inputs(), clip=clip), None)
    check: Check = NoiseContinuityCheck(NoiseContinuityCheck.Params())
    report = VerificationRunner([check], RawPolicy(), {}).run(Items([item]), [clip])
    spec = specificity(report, SpecificitySettings(), [CheckUnit.of(check)])
    found = [o for o in spec.outcomes if o.check_id == "R4_jump"]
    return (
        [o for o in found if o.rule is Rule.CHECK_UNIT],
        [o for o in found if o.rule is Rule.INJECTED_PARAMETER],
    )


def test_m1_the_estimated_jump_agrees_with_the_injected_one() -> None:
    """Audit M1: with one r4_teleport level per calib scene the order is undefined, so the sweep
    also checks, clip by clip, the estimated jump at the peak against mu (as v1 did)."""
    (agreement,), (ranking,) = real_teleport(1.5)
    assert agreement.passed  # type: ignore[attr-defined]
    assert agreement.observed == pytest.approx(1.5, rel=1e-9)  # type: ignore[attr-defined]
    assert ranking.undefined  # type: ignore[attr-defined]


def test_m1_a_wrong_estimate_fails() -> None:
    (agreement,), _ = real_teleport(1.0)
    assert not agreement.passed  # type: ignore[attr-defined]
    assert "jump_over_size_estimated" in agreement.detail  # type: ignore[attr-defined]


def test_m1_the_identity_carries_the_estimate_key() -> None:
    unit = CheckUnit.of(NoiseContinuityCheck)
    assert unit.perturbation_estimate == "jump_over_size_estimated"
    assert CheckUnit.of(ContinuityCheck).perturbation_estimate is None
