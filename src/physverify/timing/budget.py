"""How many training prompts and preference pairs fit in a GPU-hour budget.

Symbols: ``g_B`` generation per video with batch ``B``, ``e`` and ``v`` the other per-clip
components of ``timing.total_measures``, ``L_x`` the first-cell extraction overhead amortized over
``B_x`` clips, ``H`` GPU hours, ``T_ft`` fine-tuning seconds, ``Q`` evaluation prompts, ``K``
candidates per prompt and ``K'`` samples per prompt from the trained model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from physverify.core.errors import EXIT_USAGE, TimingError
from physverify.timing.records import (
    EXTRACT_COLD_EXTRA_S,
    GENERATE_LOAD_S,
    MeasureStats,
    ResultsFile,
    generate_measure,
)
from physverify.timing.stats import across_medians

PER_VIDEO_S: Final = "per_video_s"
SECONDS_PER_HOUR: Final = 3600.0


class BudgetError(TimingError):
    exit_code = EXIT_USAGE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BudgetError(message)


@dataclass(frozen=True)
class BudgetQuery:
    gpu_hours: float
    candidates: int
    eval_prompts: int = 0
    eval_samples: int = 1
    finetune_hours: float = 0.0
    extract_batch: int | None = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        _require(math.isfinite(self.gpu_hours) and self.gpu_hours > 0, "--gpu-hours must be > 0")
        _require(self.candidates >= 1, "--candidates must be >= 1")
        _require(self.eval_prompts >= 0, "--eval-prompts must be >= 0")
        _require(self.eval_samples >= 0, "--eval-samples must be >= 0")
        _require(
            math.isfinite(self.finetune_hours) and self.finetune_hours >= 0,
            "--finetune-hours must be >= 0",
        )
        _require(
            self.extract_batch is None or self.extract_batch >= 1, "--extract-batch must be >= 1"
        )
        _require(self.batch_size >= 1, "--batch-size must be >= 1")


@dataclass(frozen=True)
class CostPerVideo:
    """GPU seconds per candidate video, from the across-clip medians of one valid run."""

    generate_s: float
    components: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    extract_load_s: float | None = None
    generator_load_s: float | None = None

    def sequential(self, extract_batch: int | None) -> float:
        """``c_seq = g_B + e + v + L_x / B_x``; the load term is left out without ``B_x``."""
        cost = self.generate_s + math.fsum(self.components.values())
        if extract_batch is None:
            return cost
        _require(
            self.extract_load_s is not None,
            f"--extract-batch needs {EXTRACT_COLD_EXTRA_S}, which this run did not measure",
        )
        return cost + (self.extract_load_s or 0.0) / extract_batch

    def generator_only(self) -> float:
        """``c_gen = g_B``: extraction and checks run elsewhere, in parallel."""
        return self.generate_s


@dataclass(frozen=True)
class Bound:
    cost_s: float
    available_s: float
    prompts: int
    pairs_min: int
    pairs_max: int


@dataclass(frozen=True)
class BudgetAnswer:
    sequential: Bound
    generator_only: Bound
    extract_load_included: bool
    generator_load_s: float | None  # L_g, once per job; not in c


def bound(cost_s: float, query: BudgetQuery) -> Bound:
    """``S = 3600 H - T_ft - Q (K + K') c``, ``P_max = floor(S / (K c))``,
    ``P_max <= pairs <= P_max K (K - 1) / 2``."""
    _require(cost_s > 0, "the cost per video must be > 0")
    k = query.candidates
    available = SECONDS_PER_HOUR * (query.gpu_hours - query.finetune_hours) - (
        query.eval_prompts * (k + query.eval_samples) * cost_s
    )
    prompts = max(0, math.floor(available / (k * cost_s)))
    return Bound(
        cost_s=cost_s,
        available_s=available,
        prompts=prompts,
        pairs_min=prompts if k >= 2 else 0,
        pairs_max=prompts * k * (k - 1) // 2,
    )


def _setup_value(results: ResultsFile, measure: str) -> float | None:
    return next((t.setup[measure] for t in results.targets if measure in t.setup), None)


def cost_from_results(
    results: ResultsFile,
    rows: Sequence[MeasureStats],
    total_measures: Sequence[str],
    batch_size: int,
) -> CostPerVideo:
    """``g_B`` for ``batch_size`` plus the non-generation measures of ``total_measures``."""
    if not results.valid_measurement:
        reasons = "; ".join(results.validity_reasons)
        raise BudgetError(f"budget needs a valid H200 measurement; this run is not one ({reasons})")
    generate = generate_measure(batch_size, PER_VIDEO_S)
    components = [m for m in total_measures if m != generate_measure(1, PER_VIDEO_S)]
    medians = across_medians(rows)
    missing = [m for m in (generate, *components) if m not in medians]
    _require(not missing, f"the run has no across-clip median of {', '.join(missing)}")
    cost = CostPerVideo(
        generate_s=medians[generate],
        components=MappingProxyType({m: medians[m] for m in components}),
        extract_load_s=medians.get(EXTRACT_COLD_EXTRA_S),
        generator_load_s=_setup_value(results, GENERATE_LOAD_S),
    )
    _require(cost.generate_s > 0, f"the across-clip median of {generate} is 0")
    return cost


def answer(cost: CostPerVideo, query: BudgetQuery) -> BudgetAnswer:
    return BudgetAnswer(
        sequential=bound(cost.sequential(query.extract_batch), query),
        generator_only=bound(cost.generator_only(), query),
        extract_load_included=query.extract_batch is not None,
        generator_load_s=cost.generator_load_s,
    )


def _bound_line(label: str, result: Bound) -> str:
    return (
        f"{label}: {result.cost_s:.4g} s per video; available {result.available_s:.6g} s; "
        f"training prompts {result.prompts}; preference pairs {result.pairs_min}-"
        f"{result.pairs_max} (before abstentions)"
    )


def budget_lines(result: BudgetAnswer, query: BudgetQuery) -> list[str]:
    extract_load = (
        f"included with B_x = {query.extract_batch}"
        if result.extract_load_included
        else "omitted (pass --extract-batch to include it)"
    )
    generator_load = (
        "not measured"
        if result.generator_load_s is None
        else f"{result.generator_load_s:.4g} s once per job, not in the cost per video"
    )
    return [
        (
            f"budget: H = {query.gpu_hours:g} GPU hours, fine-tuning {query.finetune_hours:g} h, "
            f"K = {query.candidates}, Q = {query.eval_prompts}, K' = {query.eval_samples}, "
            f"B = {query.batch_size}"
        ),
        _bound_line("sequential c_seq = g_B + e + v + L_x/B_x", result.sequential),
        _bound_line("generator only c_gen = g_B", result.generator_only),
        f"extraction load L_x/B_x: {extract_load}",
        f"generator load L_g: {generator_load}",
    ]
