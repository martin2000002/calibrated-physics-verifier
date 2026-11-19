"""Toy numbers only: they live in this fixture and are not costs of anything."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from physverify.core.errors import EXIT_USAGE
from physverify.timing.budget import (
    PER_VIDEO_S,
    BudgetError,
    BudgetQuery,
    CostPerVideo,
    answer,
    bound,
    cost_from_results,
)
from physverify.timing.records import (
    EXTRACT_COLD_EXTRA_S,
    EXTRACT_SPAN_S,
    GENERATE_LOAD_S,
    VERIFY_WALL_S,
    MeasureStats,
    ResultsFile,
    Sample,
    TargetResult,
    TargetStatus,
    generate_measure,
    unit_of,
)
from physverify.timing.settings import TargetRunSettings
from physverify.timing.stats import stat_rows

G1, G4 = generate_measure(1, PER_VIDEO_S), generate_measure(4, PER_VIDEO_S)
TOTAL = (G1, EXTRACT_SPAN_S, VERIFY_WALL_S)
TOY = {G1: 10.0, G4: 6.0, EXTRACT_SPAN_S: 4.0, VERIFY_WALL_S: 1.0, EXTRACT_COLD_EXTRA_S: 20.0}


def rows_for(values: Mapping[str, float]) -> list[MeasureStats]:
    samples = [
        Sample("t", clip, rep, False, measure, unit_of(measure), value)
        for measure, value in values.items()
        for clip in ("c0", "c1")
        for rep in range(3)
    ]
    return stat_rows(samples, 0.1)


def results(*, valid: bool = True, setup: Mapping[str, float] | None = None) -> ResultsFile:
    target = TargetResult(
        name="wan21_generate",
        requires_gpu=True,
        status=TargetStatus.MEASURED,
        run=TargetRunSettings(),
        setup=dict(setup or {}),
    )
    return ResultsFile(
        valid_measurement=valid,
        validity_reasons=[] if valid else ["device is 'cpu', not cuda"],
        run={},
        environment={},
        targets=[target],
    )


def toy_cost() -> CostPerVideo:
    return cost_from_results(results(setup={GENERATE_LOAD_S: 30.0}), rows_for(TOY), TOTAL, 1)


def test_cost_takes_generation_and_the_other_total_measures() -> None:
    cost = toy_cost()
    assert cost.generate_s == 10.0
    assert dict(cost.components) == {EXTRACT_SPAN_S: 4.0, VERIFY_WALL_S: 1.0}
    assert (cost.extract_load_s, cost.generator_load_s) == (20.0, 30.0)
    assert cost.sequential(None) == 15.0
    assert cost.sequential(4) == 20.0
    assert cost.generator_only() == 10.0


def test_batch_size_selects_the_generation_measure() -> None:
    cost = cost_from_results(results(), rows_for(TOY), TOTAL, 4)
    assert (cost.generate_s, cost.sequential(None), cost.generator_load_s) == (6.0, 11.0, None)


def test_both_bounds_without_evaluation() -> None:
    result = answer(toy_cost(), BudgetQuery(gpu_hours=10, candidates=8))
    assert result.sequential.available_s == 36000.0
    assert (result.sequential.prompts, result.sequential.pairs_min) == (300, 300)
    assert result.sequential.pairs_max == 300 * 8 * 7 // 2
    assert (result.generator_only.cost_s, result.generator_only.prompts) == (10.0, 450)
    assert result.generator_only.pairs_max == 450 * 28
    assert not result.extract_load_included
    assert result.generator_load_s == 30.0


def test_evaluation_and_fine_tuning_are_taken_out_first() -> None:
    query = BudgetQuery(
        gpu_hours=10, candidates=8, eval_prompts=10, eval_samples=2, finetune_hours=1
    )
    seq = answer(toy_cost(), query).sequential
    assert seq.available_s == 3600.0 * 9 - 10 * (8 + 2) * 15.0
    assert seq.prompts == 257
    assert seq.pairs_max == 257 * 28


def test_extract_batch_adds_the_amortized_load() -> None:
    result = answer(toy_cost(), BudgetQuery(gpu_hours=10, candidates=8, extract_batch=4))
    assert result.sequential.cost_s == 20.0
    assert result.sequential.prompts == 225
    assert result.extract_load_included


def test_one_candidate_has_no_pairs_and_a_deficit_has_no_prompts() -> None:
    one = bound(10.0, BudgetQuery(gpu_hours=1, candidates=1))
    assert (one.prompts, one.pairs_min, one.pairs_max) == (360, 0, 0)
    deficit = bound(10.0, BudgetQuery(gpu_hours=1, candidates=8, eval_prompts=100))
    assert deficit.available_s < 0
    assert (deficit.prompts, deficit.pairs_max) == (0, 0)


def test_an_invalid_run_is_refused_with_a_usage_exit() -> None:
    with pytest.raises(BudgetError, match="valid H200 measurement") as caught:
        cost_from_results(results(valid=False), rows_for(TOY), TOTAL, 1)
    assert caught.value.exit_code == EXIT_USAGE


def test_a_missing_measure_is_refused() -> None:
    partial = {k: v for k, v in TOY.items() if k != VERIFY_WALL_S}
    with pytest.raises(BudgetError, match=VERIFY_WALL_S):
        cost_from_results(results(), rows_for(partial), TOTAL, 1)
    with pytest.raises(BudgetError, match=r"generate\.b2\.per_video_s"):
        cost_from_results(results(), rows_for(TOY), TOTAL, 2)


def test_extract_batch_needs_the_cold_extra_measure() -> None:
    no_cold = {k: v for k, v in TOY.items() if k != EXTRACT_COLD_EXTRA_S}
    cost = cost_from_results(results(), rows_for(no_cold), TOTAL, 1)
    with pytest.raises(BudgetError, match=EXTRACT_COLD_EXTRA_S):
        answer(cost, BudgetQuery(gpu_hours=1, candidates=2, extract_batch=2))


@pytest.mark.parametrize(
    "fields",
    [
        {"gpu_hours": 0},
        {"candidates": 0},
        {"eval_prompts": -1},
        {"finetune_hours": -1.0},
        {"extract_batch": 0},
        {"batch_size": 0},
    ],
)
def test_query_rejects_invalid_values(fields: dict[str, float]) -> None:
    base: dict[str, float] = {"gpu_hours": 1.0, "candidates": 2}
    with pytest.raises(BudgetError):
        BudgetQuery(**{**base, **fields})  # type: ignore[arg-type]
