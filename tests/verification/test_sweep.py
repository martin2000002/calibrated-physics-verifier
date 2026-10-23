"""AC13 (doubles): the specificity sweep rules over labelled perturbation clips."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import check_scenarios as sc
import pytest

from physverify.checks._base import Check, CheckInputs
from physverify.checks._common import unscored
from physverify.schema.checks import CheckResult
from physverify.schema.clip import ClipManifest
from physverify.schema.scene import MagnitudeKind, PerturbationRecord
from physverify.stats.rank import kendall
from physverify.verification.inputs import ClipInputs
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationReport, VerificationRunner
from physverify.verification.sweep import (
    MATRIX_FILE,
    SPECIFICITY_FILE,
    CheckUnit,
    Rule,
    RuleOutcome,
    SpecificityReport,
    SpecificitySettings,
    matrix,
    specificity,
    target_check_ids,
    write_sweep,
)

UNITS = {
    "R1": "pen_over_size",
    "R2": "residual_over_weight",
    "R3a": "e_minus_1",
    "R3b": "dE_over_E",
    "R4_missing": "frames",
    "R4_size": "rel_size_change",
    "R4_jump": "jump_over_size",
    "R5a": "accel_rel_diff",
    "R5b": "g_rel_error",
}
Values = Mapping[str, Mapping[str, Sequence[tuple[tuple[str, ...], float]]]]


def fake_checks(values: Values) -> list[Check]:
    def build(check_id: str, unit: str) -> Check:
        def evaluate(self: Check, inputs: CheckInputs) -> Sequence[CheckResult]:
            found = values.get(inputs.clip.clip_id, {}).get(check_id, [])
            return [unscored(type(self), inputs, s, m, 0.0, None, {}) for s, m in found]

        namespace = {
            "check_id": check_id,
            "requires": frozenset(),
            "magnitude_unit": unit,
            "evaluate": evaluate,
        }
        check: Check = type(f"Fake{check_id}", (), namespace)()
        return check

    return [build(check_id, unit) for check_id, unit in UNITS.items()]


def record(
    name: str,
    target: str,
    magnitude: float,
    unit: str,
    kind: MagnitudeKind = MagnitudeKind.CHECK_UNIT,
    level: int = 1,
) -> PerturbationRecord:
    return PerturbationRecord(
        name=name,
        target_checks=(target,),
        magnitude=magnitude,
        magnitude_unit=unit,
        magnitude_kind=kind,
        level=level,
        target_objects=("a",),
        frame_span=None,
        seed=0,
    )


def clip(clip_id: str, perturbation: PerturbationRecord | None, scene: str = "s0") -> ClipInputs:
    case = sc.parabola(clip_id=clip_id)
    manifest = replace(case.clip, perturbation=perturbation, scene_id=scene)
    return ClipInputs(manifest, replace(case.inputs(), clip=manifest), None)


class ListSource:
    def __init__(self, items: Sequence[ClipInputs]) -> None:
        self._items = list(items)

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        return iter(self._items)


def sweep(items: Sequence[ClipInputs], values: Values) -> VerificationReport:
    runner = VerificationRunner(fake_checks(values), RawPolicy(), {})
    return runner.run(ListSource(items), [item.clip for item in items])


def outcome(spec: SpecificityReport, rule: Rule, perturbation: str, check_id: str) -> RuleOutcome:
    (found,) = [
        o
        for o in spec.outcomes
        if (o.rule, o.perturbation, o.check_id) == (rule, perturbation, check_id)
    ]
    return found


PAIR = ("a", "b")
CLEAN: Values = {"none": {"R1": [(PAIR, 0.002)], "R4_jump": [(("a",), 0.01)]}}


def test_the_r4_label_resolves_by_unit() -> None:
    checks = [CheckUnit(check_id, unit) for check_id, unit in UNITS.items()]
    teleport = record("r4_teleport", "R4", 0.5, "jump_over_size")
    assert target_check_ids(teleport, checks) == ("R4_jump",)
    assert target_check_ids(record("r4_resize", "R4", 0.2, "rel_size_change"), checks) == (
        "R4_size",
    )
    assert target_check_ids(record("r1_penetration", "R1", 0.1, "pen_over_size"), checks) == ("R1",)


def test_a_specific_sweep_passes() -> None:
    items = [
        clip("none", None),
        clip("pen", record("r1_penetration", "R1", 0.1, "pen_over_size")),
        clip("tp", record("r4_teleport", "R4", 0.5, "jump_over_size")),
    ]
    values: Values = {
        **CLEAN,
        "pen": {"R1": [(PAIR, 0.1015)]},
        "tp": {"R4_jump": [(("a",), 0.495)], "R1": [(PAIR, 0.0025)]},
    }
    spec = specificity(sweep(items, values), SpecificitySettings())
    assert spec.passed, spec.failures
    assert outcome(spec, Rule.CHECK_UNIT, "r1_penetration", "R1").observed == 0.1015
    assert outcome(spec, Rule.CHECK_UNIT, "r4_teleport", "R4_jump").passed
    assert outcome(spec, Rule.NULL_BAND, "r4_teleport", "R1").passed
    assert spec.null_band["R1"] == pytest.approx(0.003)
    assert spec.null_band["R2"] == pytest.approx(1e-3)
    assert spec.n_levels == {"none": 1, "r1_penetration": 1, "r4_teleport": 1}


def test_a_wrong_magnitude_and_a_moved_bystander_fail() -> None:
    items = [clip("none", None), clip("rs", record("r4_resize", "R4", 0.2, "rel_size_change"))]
    values: Values = {**CLEAN, "rs": {"R4_size": [(("a",), 0.3)], "R1": [(PAIR, 0.5)]}}
    spec = specificity(sweep(items, values), SpecificitySettings())
    assert not spec.passed
    failed = {(o.rule, o.check_id) for o in spec.failures}
    assert failed == {(Rule.CHECK_UNIT, "R4_size"), (Rule.NULL_BAND, "R1")}


def test_a_label_without_a_matching_check_fails() -> None:
    items = [clip("x", record("r9_unknown", "R9", 0.2, "furlongs"))]
    spec = specificity(sweep(items, {}), SpecificitySettings())
    (failure,) = spec.failures
    assert (failure.rule, failure.check_id) == (Rule.CHECK_UNIT, "R9")
    assert "no check matches" in failure.detail


def test_a_unit_mismatch_on_an_exact_label_fails() -> None:
    items = [clip("x", record("r1_penetration", "R1", 0.1, "frames"))]
    spec = specificity(sweep(items, {"x": {"R1": [(PAIR, 0.1)]}}), SpecificitySettings())
    (failure,) = spec.failures
    assert failure.detail.startswith("unit pen_over_size")


def test_only_subjects_covering_the_target_count() -> None:
    items = [clip("pen", record("r1_penetration", "R1", 0.1, "pen_over_size"))]
    values: Values = {"pen": {"R1": [(PAIR, 0.1), (("c", "d"), 0.9)]}}
    spec = specificity(sweep(items, values), SpecificitySettings())
    assert outcome(spec, Rule.CHECK_UNIT, "r1_penetration", "R1").observed == 0.1


def overhang(scene: str, levels: Sequence[float], measured: Sequence[float]) -> Any:
    items, values = [], {}
    for level, (offset, value) in enumerate(zip(levels, measured, strict=True)):
        clip_id = f"{scene}_{level}"
        kind = MagnitudeKind.INJECTED_PARAMETER
        label = record("r2_unsupported_rest", "R2", offset, "overhang_over_size", kind, level)
        items.append(clip(clip_id, label, scene))
        values[clip_id] = {"R2": [(("a",), value)]}
    return items, values


def test_injected_parameters_are_judged_by_ranking_per_scene() -> None:
    good, good_values = overhang("s1", [0.0, 0.1, 0.2, 0.3], [0.0005, 0.2, 0.5, 0.9])
    bad, bad_values = overhang("s2", [0.1, 0.2, 0.3], [0.5, 0.2, 0.9])
    items = [clip("none", None), *good, *bad]
    spec = specificity(sweep(items, {**CLEAN, **good_values, **bad_values}), SpecificitySettings())
    ranked = [o for o in spec.outcomes if o.rule is Rule.INJECTED_PARAMETER]
    assert [(o.clips[0], o.passed) for o in ranked] == [("s1_1", True), ("s2_0", False)]
    assert ranked[0].observed == pytest.approx(1.0)
    assert ranked[1].observed == pytest.approx(1 / 3)
    supported = outcome(spec, Rule.NULL_BAND, "r2_unsupported_rest", "R2")
    assert supported.clips == ("s1_0",) and supported.passed
    assert spec.n_levels["r2_unsupported_rest"] == 4


def test_a_declared_coupling_is_reported_but_never_fails() -> None:
    items = [clip("none", None), clip("g", record("r5b_gravity_scale", "R5b", 0.3, "g_rel_error"))]
    values: Values = {**CLEAN, "g": {"R5b": [(("a",), 0.3)], "R4_jump": [(("a",), 0.8)]}}
    declared = SpecificitySettings(couplings=[("r5b_gravity_scale", "R4_jump")])
    spec = specificity(sweep(items, values), declared)
    coupled = outcome(spec, Rule.COUPLING, "r5b_gravity_scale", "R4_jump")
    assert not coupled.passed
    assert spec.passed
    strict = specificity(sweep(items, values), SpecificitySettings(couplings=[]))
    assert not strict.passed


def test_kendall_below_two_levels_is_undefined() -> None:
    assert math.isnan(kendall([(0.1, 0.2)]))
    assert kendall([(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]) == pytest.approx(1.0)


def test_the_matrix_and_the_files(tmp_path: Path) -> None:
    items = [clip("pen", record("r1_penetration", "R1", 0.1, "pen_over_size")), clip("none", None)]
    report = sweep(items, {**CLEAN, "pen": {"R1": [(PAIR, 0.1)]}})
    cells = matrix(report)
    assert len(cells) == 2 * len(UNITS)
    assert cells[0].perturbation == "none"
    r1 = next(c for c in cells if (c.perturbation, c.check_id) == ("r1_penetration", "R1"))
    assert (r1.median, r1.max, r1.n, r1.n_clips) == (0.1, 0.1, 1, 1)
    r2 = next(c for c in cells if (c.perturbation, c.check_id) == ("none", "R2"))
    assert (r2.median, r2.n) == (None, 0)
    spec = specificity(report, SpecificitySettings())
    write_sweep(tmp_path, cells, spec)
    with (tmp_path / MATRIX_FILE).open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(cells)
    written = json.loads((tmp_path / SPECIFICITY_FILE).read_text(encoding="utf-8"))
    assert written["passed"] is spec.passed
    assert len(written["outcomes"]) == len(spec.outcomes)


def test_the_null_band_ignores_nan_magnitudes() -> None:
    values: Values = {
        "none": {"R1": [(PAIR, math.nan), (("a",), 0.002)], "R2": [(("a",), math.nan)]}
    }
    spec = specificity(sweep([clip("none", None)], values), SpecificitySettings())
    assert spec.null_band["R1"] == pytest.approx(0.003)
    assert spec.null_band["R2"] == pytest.approx(1e-3)


PEN = (Rule.CHECK_UNIT, "r1_penetration", "R1")


def test_a_nan_subject_listed_first_does_not_set_the_observed_value() -> None:
    items = [clip("pen", record("r1_penetration", "R1", 0.1, "pen_over_size"))]
    values: Values = {"pen": {"R1": [(PAIR, math.nan), (PAIR, 0.1015)]}}
    found = outcome(specificity(sweep(items, values), SpecificitySettings()), *PEN)
    assert found.observed == 0.1015 and found.passed


def test_an_all_nan_target_gives_no_magnitude() -> None:
    items = [clip("pen", record("r1_penetration", "R1", 0.1, "pen_over_size"))]
    values: Values = {"pen": {"R1": [(PAIR, math.nan)]}}
    found = outcome(specificity(sweep(items, values), SpecificitySettings()), *PEN)
    assert found.observed is None and not found.passed


def ranked(items: Sequence[ClipInputs], values: Values) -> tuple[SpecificityReport, RuleOutcome]:
    spec = specificity(
        sweep([clip("none", None), *items], {**CLEAN, **values}), SpecificitySettings()
    )
    (found,) = [o for o in spec.outcomes if o.rule is Rule.INJECTED_PARAMETER]
    return spec, found


def test_a_nan_level_fails_the_ranking() -> None:
    spec, found = ranked(*overhang("s1", [0.1, 0.2], [0.2, math.nan]))
    assert not found.passed and not found.undefined and found in spec.failures
    assert "no magnitude on some level" in found.detail


def test_tied_levels_fail_the_ranking() -> None:
    spec, found = ranked(*overhang("s1", [0.1, 0.2], [0.3, 0.3]))
    assert math.isnan(cast(float, found.observed)) and not found.passed
    assert found in spec.failures and "tied" in found.detail


def test_a_single_level_ranking_is_undefined_not_passed_nor_failed() -> None:
    spec, found = ranked(*overhang("s1", [0.1], [0.3]))
    assert found.undefined and not found.passed and found not in spec.failures
    assert spec.passed
