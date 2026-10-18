from __future__ import annotations

import math
from dataclasses import replace

import pytest

from physverify.finetune.budget import FinetuneBudgetQuery, budget_lines, train_prompts
from physverify.timing.budget import BudgetError, CostPerVideo

QUERY = FinetuneBudgetQuery(
    gpu_hours=100.0,
    k=4,
    k_eval_base=8,
    k_eval_ft=4,
    n_eval=62,
    n_dev=32,
    k_dev=2,
    dev_checkpoints=3,
    epochs=3.0,
    pair_yield=0.8,
    yield_measured=False,
    n_pilot=20,
    n_beta=4,
    beta_fraction=0.25,
)
COST = CostPerVideo(generate_s=100.0, generator_load_s=60.0)


def test_the_budget_matches_a_hand_computation() -> None:
    result = train_prompts(COST, 100.0, 2.0, QUERY)
    available = 360_000 - 62 * 12 * 100 - 32 * 4 * 2 * 100 - 60 - 20 * 4 * 100
    assert result.available_s == pytest.approx(available, abs=1e-9)
    n_train = math.floor(available / (4 * 100 + (1 + 4 * 0.25) * 3 * 0.8 * 2))
    assert result.n_train == n_train == 615
    assert result.finetune_s == pytest.approx(3 * 0.8 * n_train * 2.0, abs=1e-9)
    assert result.beta_grid_s == pytest.approx(result.finetune_s, abs=1e-9)
    assert result.estimated
    assert any(line.endswith("[estimated]") for line in budget_lines(result))


def test_a_measured_yield_is_not_marked_estimated() -> None:
    result = train_prompts(COST, 100.0, 2.0, replace(QUERY, yield_measured=True))
    assert not result.estimated
    assert not any("[estimated]" in line for line in budget_lines(result))


def test_a_budget_smaller_than_the_fixed_terms_gives_zero_prompts() -> None:
    assert train_prompts(COST, 100.0, 2.0, replace(QUERY, gpu_hours=1.0)).n_train == 0


@pytest.mark.parametrize(
    "update",
    [{"gpu_hours": 0.0}, {"k": 1}, {"pair_yield": 3.0}, {"epochs": 0.0}, {"beta_fraction": 1.5}],
)
def test_an_invalid_query_is_a_budget_error(update: dict[str, float]) -> None:
    with pytest.raises(BudgetError):
        replace(QUERY, **update)


def test_invalid_costs_are_budget_errors() -> None:
    with pytest.raises(BudgetError):
        train_prompts(COST, 0.0, 2.0, QUERY)
    with pytest.raises(BudgetError):
        train_prompts(COST, 100.0, 0.0, QUERY)
