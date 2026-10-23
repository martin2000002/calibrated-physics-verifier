"""Training prompts that fit in a GPU budget, tied to the timing harness's measured costs.

With ``c`` the cost per video, ``t_p`` the seconds per pair of ``dpo_step``, ``y`` pairs per prompt,
``E`` epochs, ``L_g`` the generator load, ``f`` the beta-grid run length as a fraction of the
main run (each of the ``n_beta`` runs costs ``f T_ft``):

    S       = 3600 H - n_eval (K_base + K_ft) c - n_dev (n_ck + 1) K_dev c - L_g - n_pilot K c
    n_train = floor(S / (K c + (1 + n_beta f) E y t_p))
    T_ft    = E y n_train t_p,    T_beta = n_beta f T_ft

Every term that depends on ``y`` is an estimate until the pilot measures it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from physverify.config.models import Settings
from physverify.finetune.layout import CampaignStore
from physverify.finetune.pairs_report import pair_yield_of
from physverify.finetune.settings import load_finetune
from physverify.timing.budget import (
    SECONDS_PER_HOUR,
    BudgetError,
    CostPerVideo,
    cost_from_results,
)
from physverify.timing.results_io import load_results, load_samples, timing_directory
from physverify.timing.settings import load_timing
from physverify.timing.stats import across_medians, stat_rows

PAIR_MEASURE: Final = "dpo.pair_s"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BudgetError(message)


@dataclass(frozen=True)
class FinetuneBudgetQuery:
    gpu_hours: float
    k: int
    k_eval_base: int
    k_eval_ft: int
    n_eval: int
    n_dev: int
    k_dev: int
    dev_checkpoints: int
    epochs: float
    pair_yield: float
    yield_measured: bool
    n_pilot: int
    n_beta: int
    beta_fraction: float

    def __post_init__(self) -> None:
        _require(math.isfinite(self.gpu_hours) and self.gpu_hours > 0, "--gpu-hours must be > 0")
        _require(self.k >= 2, "K must be >= 2 to form pairs")
        _require(0 <= self.pair_yield <= self.k // 2, "pair yield must lie in [0, K/2]")
        _require(self.epochs > 0, "epochs must be > 0")
        _require(0 < self.beta_fraction <= 1, "the beta-grid fraction must lie in (0, 1]")


@dataclass(frozen=True)
class BudgetTerm:
    name: str
    seconds: float
    estimated: bool


@dataclass(frozen=True)
class FinetuneBudget:
    cost_s: float
    pair_s: float
    terms: tuple[BudgetTerm, ...]
    available_s: float
    n_train: int
    finetune_s: float
    beta_grid_s: float
    estimated: bool


def train_prompts(
    cost: CostPerVideo, cost_s: float, pair_s: float, q: FinetuneBudgetQuery
) -> FinetuneBudget:
    _require(cost_s > 0, "the cost per video must be > 0")
    _require(pair_s > 0, "the seconds per pair must be > 0")
    load = cost.generator_load_s or 0.0
    terms = (
        BudgetTerm("eval", q.n_eval * (q.k_eval_base + q.k_eval_ft) * cost_s, False),
        BudgetTerm("dev", q.n_dev * (q.dev_checkpoints + 1) * q.k_dev * cost_s, False),
        BudgetTerm("generator_load", load, False),
        BudgetTerm("pilot", q.n_pilot * q.k * cost_s, False),
    )
    available = SECONDS_PER_HOUR * q.gpu_hours - math.fsum(t.seconds for t in terms)
    train_per_prompt = q.epochs * q.pair_yield * pair_s
    per_prompt = q.k * cost_s + (1 + q.n_beta * q.beta_fraction) * train_per_prompt
    n_train = max(0, math.floor(available / per_prompt))
    finetune_s = train_per_prompt * n_train
    return FinetuneBudget(
        cost_s=cost_s,
        pair_s=pair_s,
        terms=terms,
        available_s=available,
        n_train=n_train,
        finetune_s=finetune_s,
        beta_grid_s=q.n_beta * q.beta_fraction * finetune_s,
        estimated=not q.yield_measured,
    )


def budget_lines(result: FinetuneBudget) -> list[str]:
    mark = " [estimated]" if result.estimated else ""
    lines = [f"c = {result.cost_s:.3f} s per video, t_p = {result.pair_s:.3f} s per pair"]
    lines += [f"  - {t.name}: {t.seconds:.1f} s" for t in result.terms]
    lines.append(f"S = {result.available_s:.1f} s")
    lines.append(f"n_train = {result.n_train}{mark}")
    lines.append(f"T_ft = {result.finetune_s:.1f} s{mark}")
    lines.append(f"T_beta = {result.beta_grid_s:.1f} s{mark}")
    return lines


@dataclass(frozen=True)
class BudgetRequest:
    timing_run: Path
    gpu_hours: float
    pair_yield: float | None
    extract_batch: int | None
    generator_only: bool
    pilot_prompts: int
    scorer: str | None


def budget_report(settings: Settings, request: BudgetRequest) -> list[str]:
    """The budget lines of a valid timing run with ``dpo_step``; the pair yield is the measured
    one of ``report.json`` unless given."""
    cfg = load_finetune(settings)
    timing = load_timing(settings)
    directory = timing_directory(request.timing_run)
    rows = stat_rows(load_samples(directory), timing.max_iqr_rel)
    cost = cost_from_results(load_results(directory), rows, timing.total_measures, 1)
    pair_s = across_medians(rows).get(PAIR_MEASURE)
    if pair_s is None:
        raise BudgetError(f"the timing run has no {PAIR_MEASURE}: run the dpo_step target")
    store = CampaignStore.from_settings(settings)
    y, measured = pair_yield_of(store, request.scorer or cfg.scorer, request.pair_yield)
    query = FinetuneBudgetQuery(
        gpu_hours=request.gpu_hours,
        k=cfg.candidates.k_train,
        k_eval_base=cfg.candidates.k_eval,
        k_eval_ft=cfg.candidates.k_eval_finetuned,
        n_eval=cfg.prompts.n_eval + len(cfg.prompts.fixed_eval),
        n_dev=cfg.prompts.n_dev,
        k_dev=cfg.candidates.k_dev,
        dev_checkpoints=cfg.select.checkpoints,
        epochs=cfg.train.epochs,
        pair_yield=y,
        yield_measured=measured,
        n_pilot=request.pilot_prompts,
        n_beta=len(cfg.train.beta_grid),
        beta_fraction=cfg.train.beta_grid_fraction,
    )
    cost_s = (
        cost.generator_only() if request.generator_only else cost.sequential(request.extract_batch)
    )
    return budget_lines(train_prompts(cost, cost_s, pair_s, query))
