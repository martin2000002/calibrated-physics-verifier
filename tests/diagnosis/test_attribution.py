"""Level 3: blocks, unit coherence, statistics and the fixture run."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from physverify.checks._base import CheckInputs
from physverify.config.models import Settings
from physverify.core.errors import MissingInputError, PhysverifyError
from physverify.diagnosis.attribution import (
    Block,
    budget_stages,
    check_statistics,
    check_units,
    collect_blocks,
    fixed_label,
    lift_unit,
    stage_metric,
)
from physverify.diagnosis.attribution_run import run_attribution
from physverify.diagnosis.chain_variants import CELLS, Cell, cell_name
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.factorial import ALL_PERCEIVED, ALL_TRUTH, shapley, yates_effects
from physverify.diagnosis.pairing import UnitMismatchError, pair_results
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import Split
from physverify.schema.stages import Stage

CHECK = "r5b"
MIN_IOU = 0.1
FIXED = "detect+camera"
TOL = 1e-12
Evaluator = Callable[[CheckInputs], Sequence[CheckResult]]
Suite = Any
"""The conftest's ``FakeCheckSuite`` (a fixture: conftest is not importable)."""


def result(
    subject: str,
    magnitude: float | None,
    *,
    reason: AbstainReason = AbstainReason.MISSING_INPUT,
    unit: str = "m/s^2",
    clip_id: str = "clip",
    check_id: str = CHECK,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        clip_id=clip_id,
        subject=(subject,),
        frame_span=(0, 30),
        magnitude=magnitude,
        sigma=math.nan,
        unit=unit,
        verdict=Verdict.UNSCORED if magnitude is not None else Verdict.ABSTAIN,
        input_level=InputLevel.MIXED,
        abstain_reason=None if magnitude is not None else reason,
    )


F_TPP: Cell = (False, True, True)


def test_incomplete_blocks_are_counted_by_cause() -> None:
    oracle = (
        result("complete", 1.0),
        result("no_scale", None, reason=AbstainReason.SCALE_UNAVAILABLE),
        result("lost", 2.0),
        result("occluded", 3.0),
        result("no_v0", 4.0),
    )

    def cell_results(cell: Cell) -> list[CheckResult]:
        shift = 0.1 * sum(cell)
        results = [result("complete", 1.0 + shift), result("no_scale", 5.0)]
        results.append(result("no_v0", 4.0 - shift))
        if cell != ALL_PERCEIVED:
            results.append(result("lost", 2.0))
        if cell == F_TPP:
            results.append(result("occluded", None, reason=AbstainReason.OCCLUDED))
        elif cell != ALL_PERCEIVED:
            results.append(result("occluded", 3.0))
        return results

    cells = {cell: pair_results(oracle, cell_results(cell), None, MIN_IOU) for cell in CELLS}
    v0 = pair_results(oracle, [result("complete", 1.5)], None, MIN_IOU)

    collected = collect_blocks(oracle, cells, v0)

    assert collected.incomplete == {
        CHECK: {
            "abstain:F_tpp:OCCLUDED": 1,
            "abstain:oracle:SCALE_UNAVAILABLE": 1,
            "no_pair:F_ppp": 2,
            "no_pair:V0": 1,
            "total": 3,
        }
    }
    complete, no_v0 = collected.blocks
    assert (complete.subject, no_v0.subject) == (("complete",), ("no_v0",))
    assert complete.errors[ALL_TRUTH] == 0.0
    assert complete.errors[ALL_PERCEIVED] == pytest.approx(0.3)
    assert complete.v0 == pytest.approx(0.5)
    assert no_v0.v0 is None


def test_blocks_without_v0_pairing_have_no_v0_response() -> None:
    oracle = (result("a", 1.0),)
    cells = {cell: pair_results(oracle, [result("a", 2.0)], None, MIN_IOU) for cell in CELLS}
    collected = collect_blocks(oracle, cells, None)
    (block,) = collected.blocks
    assert block.v0 is None
    assert collected.incomplete == {CHECK: {}}


def test_check_units_agree_over_magnitudes() -> None:
    results = {
        "oracle": [result("a", 1.0), result("b", None, unit="")],
        "F_ttt": [result("a", 1.1), result("x", 2.0, check_id="other", unit="m")],
        "V0": [],
    }
    assert check_units(CHECK, results) == "m/s^2"
    assert check_units("other", results) == "m"
    assert check_units("absent", results) == "n/a"


def test_a_different_check_unit_raises() -> None:
    results = {"oracle": [result("a", 1.0)], "F_ppp": [result("a", 1.0, unit="px/s^2")]}
    with pytest.raises(UnitMismatchError, match="F_ppp") as raised:
        check_units(CHECK, results)
    assert isinstance(raised.value, PhysverifyError)


def test_lift_unit_is_shared_by_every_cell() -> None:
    names = [cell_name(cell) for cell in CELLS]
    assert lift_unit({name: ["relative", "relative"] for name in names}) == "relative"
    assert lift_unit({name: [] for name in names}) == "n/a"
    mixed = {name: ["meter" if name == "F_ttt" else "relative"] for name in names}
    with pytest.raises(UnitMismatchError, match="lift"):
        lift_unit(mixed)


def linear(cell: Cell) -> float:
    track, depth, scale = (1.0 if perceived else -1.0 for perceived in cell)
    return 1.0 + 0.3 * track - 0.2 * depth + 0.05 * scale + 0.1 * track * depth


def blocks(v0: Sequence[float | None]) -> list[Block]:
    """Two blocks per clip whose errors average to ``linear`` in every cell."""
    return [
        Block(
            f"clip{k // 2}",
            CHECK,
            (f"o{k}",),
            (0, 1),
            {c: linear(c) + (0.1 if k % 2 else -0.1) for c in CELLS},
            value,
        )
        for k, value in enumerate(v0)
    ]


def test_statistics_are_exact() -> None:
    stats = check_statistics(blocks([2.0, 3.0, None, None]), FIXED)
    means = {cell: linear(cell) for cell in CELLS}
    for cell in CELLS:
        assert abs(stats[f"cell:{cell_name(cell)}"] - means[cell]) <= TOL
    for name, value in yates_effects(means).items():
        assert abs(stats[f"effect:{name}"] - value) <= TOL
    for stage, value in shapley(means).items():
        assert abs(stats[f"shapley:{stage.value}"] - value) <= TOL
    assert stats["v0"] == 2.5
    budget_rows = {
        k.removeprefix("budget:"): v for k, v in stats.items() if k.startswith("budget:")
    }
    assert list(budget_rows) == ["floor", "track", "depth", "scale", FIXED]
    assert abs(math.fsum(budget_rows.values()) - 2.5) <= TOL
    assert abs(budget_rows[FIXED] - (2.5 - means[ALL_PERCEIVED])) <= TOL


def test_statistics_without_v0_have_no_budget() -> None:
    stats = check_statistics(blocks([None, None]), FIXED)
    assert not any(key.startswith("budget:") for key in stats)
    assert "v0" not in stats


def test_budget_rows_point_at_their_level1_metrics() -> None:
    fixed = (Stage.DETECT, Stage.CAMERA)
    assert fixed_label(reversed(fixed)) == FIXED
    stages = budget_stages(fixed)
    assert stages["floor"] == (Stage.LIFT, Stage.KINEMATICS, Stage.CONTACTS)
    metrics = ("camera/fov", "depth/absrel", "detect/box", "lift/position", "track/mask")
    assert stage_metric(metrics, stages[FIXED]) == "camera/fov,detect/box"
    assert stage_metric(metrics, stages["floor"]) == "lift/position"
    assert stage_metric(metrics, stages["scale"]) is None


@pytest.fixture
def context(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
) -> Callable[..., DiagnosisContext]:
    def build(suite: Suite | None, *overrides: str) -> DiagnosisContext:
        settings = make_settings(
            "diagnosis.dataset=synthetic_fixture", "diagnosis.factorial.bootstrap=40", *overrides
        )
        return make_diagnosis_context(settings, suite=suite, metrics=METRICS)

    return build


METRICS: dict[str, Any] = {"track/mask": object(), "lift/position": object()}


def test_run_attribution_on_the_fixture(
    context: Callable[..., DiagnosisContext], fake_suite: Callable[..., Suite]
) -> None:
    suite = fake_suite()
    ctx = context(suite)
    clips = ctx.dataset().clips()

    report = run_attribution(ctx, clips)

    assert report.failures == ()
    assert report.fixed_ground_truth == (Stage.DETECT, Stage.CAMERA)
    assert report.lift_unit == "relative"
    assert (report.bootstrap, report.seed, report.ci) == (40, 0, 0.95)
    levels = [inputs.input_level for inputs in suite.evaluated]
    assert levels == [InputLevel.ORACLE, *[InputLevel.MIXED] * 8, InputLevel.PERCEIVED]
    (check,) = report.checks
    assert (check.unit, check.n_blocks, check.n_clips) == ("m", 2, 1)
    assert set(check.cells) == {cell_name(cell) for cell in CELLS}
    assert abs(sum(check.shapley.values()) - (check.cells["F_ppp"] - check.cells["F_ttt"])) <= TOL
    assert [row.row for row in check.budget] == ["floor", "track", "depth", "scale", FIXED]
    assert abs(math.fsum(row.value for row in check.budget) - check.v0) <= TOL
    assert sum(row.percent for row in check.budget) == pytest.approx(100.0)
    metric_of = {row.row: row.stage_metric for row in check.budget}
    assert metric_of == {"floor": "lift/position", "track": "track/mask"} | {
        "depth": None,
        "scale": None,
        FIXED: None,
    }
    assert check.bootstrap_dropped == 0
    for name, value in check.cells.items():
        assert check.cells_ci[name].low == pytest.approx(value)
        assert check.cells_ci[name].high == pytest.approx(value)
    assert set(check.shapley_ci) == set(check.shapley)
    assert check.incomplete == {}


def abstain_without_true_scale(evaluate: Evaluator) -> Evaluator:
    """R5b-like: without a ground-truth scale every result abstains with SCALE_UNAVAILABLE."""

    def evaluator(inputs: CheckInputs) -> list[CheckResult]:
        results = list(evaluate(inputs))
        if inputs.provenance.get(Stage.SCALE.value) == "ground_truth":
            return results
        return [
            dataclasses.replace(
                r,
                magnitude=None,
                sigma=math.nan,
                verdict=Verdict.ABSTAIN,
                abstain_reason=AbstainReason.SCALE_UNAVAILABLE,
            )
            for r in results
        ]

    return evaluator


def centimeters_when_perceived(evaluate: Evaluator) -> Evaluator:
    def evaluator(inputs: CheckInputs) -> list[CheckResult]:
        results = list(evaluate(inputs))
        if inputs.input_level is not InputLevel.PERCEIVED:
            return results
        return [dataclasses.replace(r, unit="cm") for r in results]

    return evaluator


def test_run_attribution_counts_abstentions_and_rejects_unit_changes(
    context: Callable[..., DiagnosisContext], fake_suite: Callable[..., Suite]
) -> None:
    """One chain cache, three suites: the second and third runs reload the cached outputs."""
    base = fake_suite().evaluator
    ctx = context(fake_suite(evaluator=abstain_without_true_scale(base)))
    clips = ctx.dataset().clips()

    report = run_attribution(ctx, clips)

    (check,) = report.checks
    assert check.n_blocks == 0
    assert check.cells == {} and check.budget == ()
    assert math.isnan(check.v0)
    assert check.bootstrap_dropped == report.bootstrap
    assert check.incomplete["total"] == 2
    assert check.incomplete["abstain:F_ttp:SCALE_UNAVAILABLE"] == 2
    assert "abstain:F_tpt:SCALE_UNAVAILABLE" not in check.incomplete

    unit_change = dataclasses.replace(
        ctx, suite=fake_suite(evaluator=centimeters_when_perceived(base))
    )
    with pytest.raises(UnitMismatchError, match="V0"):
        run_attribution(unit_change, clips)


def test_attribution_needs_a_check_suite(context: Callable[..., DiagnosisContext]) -> None:
    ctx = context(None)
    with pytest.raises(MissingInputError, match="check suite"):
        run_attribution(ctx, ctx.dataset().clips())


def test_attribution_needs_report_split_clips(
    context: Callable[..., DiagnosisContext], fake_suite: Callable[..., Suite]
) -> None:
    ctx = context(fake_suite())
    calib = [c for c in ctx.dataset().clips() if c.split is Split.CALIB]
    with pytest.raises(MissingInputError, match="eval"):
        run_attribution(ctx, calib)
