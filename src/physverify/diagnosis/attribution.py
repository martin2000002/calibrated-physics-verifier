"""Level 3 of ADR 0002: each check's error split over TRACK, DEPTH and SCALE (ADR 0007 §5.4).

The 8 factorial cells keep ``diagnosis.factorial.fixed_ground_truth`` (DETECT, CAMERA) in ground
truth. A block is one oracle (V8) event whose magnitude is matched by a paired magnitude in all 8
cells; its response in cell c is e_c = |m_c - m_oracle|, in the check's own unit.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from statistics import fmean

from physverify.diagnosis.chain_variants import (
    CELLS,
    FACTORS,
    Cell,
    cell_name,
    perceived_variant,
)
from physverify.diagnosis.factorial import FLOOR, budget, shapley, yates_effects
from physverify.diagnosis.pairing import Paired, Span, Subject, common_unit
from physverify.diagnosis.report import (
    MISSING,
    BudgetRow,
    CheckAttribution,
)
from physverify.diagnosis.settings import FactorialSettings
from physverify.schema.checks import CheckResult
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.stats.bootstrap import BootstrapResult, bootstrap
from physverify.stats.intervals import Interval

ORACLE = "oracle"
TOTAL = "total"
CELL_KEY, EFFECT_KEY, SHAPLEY_KEY, BUDGET_KEY = "cell:", "effect:", "shapley:", "budget:"
V0_KEY = "v0"


@dataclass(frozen=True)
class Block:
    """``errors[c]`` = |m_c - m_oracle|; ``v0`` the same for V0, None when V0 has no magnitude."""

    clip_id: str
    check_id: str
    subject: Subject
    span: Span | None
    errors: Mapping[Cell, float]
    v0: float | None


@dataclass(frozen=True)
class CollectedBlocks:
    """Complete blocks, and per check the causes of the incomplete ones.

    Causes are ``abstain:<source>:<reason>``, ``no_magnitude:<source>`` and ``no_pair:<source>``
    (source ``oracle`` or a cell name), plus ``total`` incomplete blocks. ``no_pair:V0`` and
    ``abstain:V0:<reason>`` count complete blocks left out of the budget, not in ``total``.
    """

    blocks: tuple[Block, ...]
    incomplete: Mapping[str, Mapping[str, int]]


def _no_magnitude(source: str, result: CheckResult) -> str:
    if result.abstain_reason is not None:
        return f"abstain:{source}:{result.abstain_reason.value}"
    return f"no_magnitude:{source}"


def _magnitude(source: str, result: CheckResult | None) -> float | str:
    """The magnitude of ``result``, or the cause (a str) of its absence."""
    if result is None:
        return f"no_pair:{source}"
    return result.magnitude if result.magnitude is not None else _no_magnitude(source, result)


Partners = Mapping[int, CheckResult]


def _partners(paired: Paired) -> dict[int, CheckResult]:
    """Other side by the identity of its oracle result (results are unhashable records)."""
    return {id(pair.oracle): pair.other for pair in paired.pairs}


def _block(
    result: CheckResult,
    cells: Mapping[Cell, Partners],
    v0: Partners | None,
    tally: Counter[str],
) -> Block | None:
    """The block of one oracle result, or None after counting why it is incomplete."""
    reference = result.magnitude
    if reference is None:
        tally.update((_no_magnitude(ORACLE, result), TOTAL))
        return None
    values = {cell: _magnitude(cell_name(cell), cells[cell].get(id(result))) for cell in CELLS}
    causes = {value for value in values.values() if isinstance(value, str)}
    if causes:
        tally.update(causes | {TOTAL})
        return None
    errors = {c: abs(v - reference) for c, v in values.items() if not isinstance(v, str)}
    v0_error = None
    if v0 is not None:
        value = _magnitude(perceived_variant().name, v0.get(id(result)))
        if isinstance(value, str):
            tally[value] += 1
        else:
            v0_error = abs(value - reference)
    return Block(
        result.clip_id, result.check_id, result.subject, result.frame_span, errors, v0_error
    )


def collect_blocks(
    oracle: Sequence[CheckResult], cells: Mapping[Cell, Paired], v0: Paired | None
) -> CollectedBlocks:
    """Blocks of the ``oracle`` results paired (by ``pair_results``) in ``cells`` and ``v0``."""
    partners = {cell: _partners(cells[cell]) for cell in CELLS}
    v0_partners = _partners(v0) if v0 is not None else None
    counts: dict[str, Counter[str]] = {}
    blocks: list[Block] = []
    for result in oracle:
        tally = counts.setdefault(result.check_id, Counter())
        block = _block(result, partners, v0_partners, tally)
        if block is not None:
            blocks.append(block)
    incomplete = {check: dict(sorted(tally.items())) for check, tally in sorted(counts.items())}
    return CollectedBlocks(tuple(blocks), incomplete)


def check_units(check_id: str, results: Mapping[str, Sequence[CheckResult]]) -> str:
    """The unit of ``check_id`` over every source's magnitudes (``MISSING`` when none has one)."""
    units = {
        source: [r.unit for r in items if r.check_id == check_id and r.magnitude is not None]
        for source, items in results.items()
    }
    present = {source: values for source, values in units.items() if values}
    return common_unit(check_id, present) if present else MISSING


def lift_unit(units: Mapping[str, Iterable[str]]) -> str:
    """The one LIFT ``ObjectStateTable.unit`` over the cells (``MISSING`` with no clip)."""
    present = {cell: list(values) for cell, values in units.items()}
    if not any(present.values()):
        return MISSING
    return common_unit(Stage.LIFT.value, present)


# --- Statistics of a set of blocks --------------------------------------------------------------


def cell_means(blocks: Sequence[Block]) -> dict[Cell, float]:
    return {cell: fmean(b.errors[cell] for b in blocks) for cell in CELLS}


def fixed_label(fixed: Iterable[Stage]) -> str:
    """``detect+camera``: the fixed stages in pipeline order."""
    stages = frozenset(fixed)
    return "+".join(stage.value for stage in STAGE_ORDER if stage in stages)


def check_statistics(blocks: Sequence[Block], fixed: str) -> dict[str, float]:
    """Cell means, Yates effects and Shapley values over ``blocks``; the budget (and ``v0``)
    over the blocks that also have a V0 response. ``fixed`` labels the fixed-stage budget row."""
    means = cell_means(blocks)
    stats = {f"{CELL_KEY}{cell_name(cell)}": value for cell, value in means.items()}
    stats |= {f"{EFFECT_KEY}{name}": value for name, value in yates_effects(means).items()}
    stats |= {f"{SHAPLEY_KEY}{stage.value}": value for stage, value in shapley(means).items()}
    budgeted = [b for b in blocks if b.v0 is not None]
    if budgeted:
        v0 = fmean(b.v0 for b in budgeted if b.v0 is not None)
        rows = budget(cell_means(budgeted), v0).rows(fixed)
        stats |= {f"{BUDGET_KEY}{row}": value for row, value in rows}
        stats[V0_KEY] = v0
    return stats


def _section[T](values: Mapping[str, T], prefix: str) -> dict[str, T]:
    return {key.removeprefix(prefix): v for key, v in values.items() if key.startswith(prefix)}


def budget_stages(fixed: Sequence[Stage]) -> dict[str, tuple[Stage, ...]]:
    """Stages behind each budget row: the floor holds what is neither a factor nor fixed."""
    floor = tuple(s for s in STAGE_ORDER if s not in FACTORS and s not in fixed)
    return {
        FLOOR: floor,
        **{stage.value: (stage,) for stage in FACTORS},
        fixed_label(fixed): tuple(fixed),
    }


def stage_metric(metrics: Iterable[str], stages: Iterable[Stage]) -> str | None:
    """The Level 1 metric keys (``<stage>/<name>``) of ``stages``, comma-joined."""
    names = {stage.value for stage in stages}
    keys = sorted(key for key in metrics if key.split("/", 1)[0] in names)
    return ",".join(keys) or None


def _percent(value: float, total: float) -> float:
    return 100.0 * value / total if total != 0 else math.nan


@dataclass(frozen=True)
class Budgeting:
    """Budget rows of a run: the fixed stages and the Level 1 metric keys shown next to them."""

    fixed: tuple[Stage, ...]
    metrics: tuple[str, ...]

    @property
    def label(self) -> str:
        return fixed_label(self.fixed)

    def rows(self, stats: Mapping[str, float], ci: Mapping[str, Interval]) -> tuple[BudgetRow, ...]:
        stages = budget_stages(self.fixed)
        total = stats.get(V0_KEY, math.nan)
        return tuple(
            BudgetRow(
                row=row,
                value=value,
                percent=_percent(value, total),
                ci=ci.get(row),
                stage_metric=stage_metric(self.metrics, stages[row]),
            )
            for row, value in _section(stats, BUDGET_KEY).items()
        )


def attribute(
    check_id: str,
    unit: str,
    collected: CollectedBlocks,
    clusters: Sequence[str],
    factorial: FactorialSettings,
    budgeting: Budgeting,
) -> CheckAttribution:
    """One check's cells, effects, Shapley values and budget, with clip-bootstrap intervals."""
    blocks = [b for b in collected.blocks if b.check_id == check_id]
    stats = check_statistics(blocks, budgeting.label) if blocks else {}
    boot: BootstrapResult = bootstrap(
        blocks,
        partial(check_statistics, fixed=budgeting.label),
        factorial.bootstrap,
        factorial.seed,
        factorial.ci,
        clusters,
    )
    ci = boot.intervals
    return CheckAttribution(
        check_id=check_id,
        unit=unit,
        n_blocks=len(blocks),
        n_clips=len({b.clip_id for b in blocks}),
        cells=_section(stats, CELL_KEY),
        cells_ci=_section(ci, CELL_KEY),
        v0=stats.get(V0_KEY, math.nan),
        effects=_section(stats, EFFECT_KEY),
        effects_ci=_section(ci, EFFECT_KEY),
        shapley={Stage(k): v for k, v in _section(stats, SHAPLEY_KEY).items()},
        shapley_ci={Stage(k): v for k, v in _section(ci, SHAPLEY_KEY).items()},
        budget=budgeting.rows(stats, _section(ci, BUDGET_KEY)),
        incomplete=dict(collected.incomplete.get(check_id, {})),
        bootstrap_dropped=boot.dropped,
    )
