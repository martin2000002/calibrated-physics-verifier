"""Runs the factorial attribution: each clip's cells paired with the oracle, then each check's
error split over the stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from physverify.core.errors import MissingInputError, PhysverifyError
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.attribution import (
    ORACLE,
    Budgeting,
    CollectedBlocks,
    attribute,
    check_units,
    collect_blocks,
    lift_unit,
)
from physverify.diagnosis.chain_runner import ChainRunner, VariantOutputs, warm_variants
from physverify.diagnosis.chain_variants import (
    CELLS,
    Cell,
    Variant,
    cell_name,
    factorial_cells,
    oracle_variant,
    perceived_variant,
)
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.end_to_end import (
    report_clips,
    require_suite,
    track_of,
    variant_outputs,
    variant_results,
)
from physverify.diagnosis.measurement_cases import stage_output
from physverify.diagnosis.pairing import Paired, merge, pair_results
from physverify.diagnosis.pairing import subject_map as match_subjects
from physverify.diagnosis.report import (
    AttributionReport,
    Failure,
    Provenance,
    build_provenance,
    dataset_versions,
)
from physverify.diagnosis.verification_adapter import CheckSuite
from physverify.schema.checks import CheckResult, InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable


@dataclass(frozen=True)
class _Run:
    suite: CheckSuite
    chain: ChainRunner
    min_span_iou: float
    cells: tuple[Variant, ...]


@dataclass(frozen=True)
class _Oracle:
    results: tuple[CheckResult, ...]
    track: MaskTrack


@dataclass(frozen=True)
class ClipCells:
    """One clip's oracle results, the pairings of each cell and of V0, and each cell's LIFT unit."""

    clip_id: str
    oracle: tuple[CheckResult, ...]
    cells: Mapping[Cell, Paired]
    v0: Paired
    lift_units: Mapping[str, str]


def _lift_unit_of(outputs: VariantOutputs) -> str:
    states = stage_output(outputs.outputs, Stage.LIFT, ObjectStateTable)
    if states is None:
        raise MissingInputError(f"variant {outputs.variant.name} has no {Stage.LIFT.value} output")
    return states.unit.value


def _pairs(
    run: _Run, clip: ClipManifest, variant: Variant, level: InputLevel, oracle: _Oracle
) -> tuple[Paired, str]:
    """``variant``'s results paired with the oracle; ids need matching only without true DETECT."""
    outputs = variant_outputs(run.chain, clip, variant)
    results = variant_results(run.suite, clip, outputs, level)
    true_ids = Stage.DETECT in variant.gt_stages
    subjects = None if true_ids else match_subjects(track_of(outputs), oracle.track)
    paired = pair_results(oracle.results, results, subjects, run.min_span_iou)
    return paired, _lift_unit_of(outputs)


def _clip_cells(run: _Run, clip: ClipManifest) -> ClipCells:
    """Variants are loaded one at a time; only the oracle track is kept for subject matching."""
    v8 = variant_outputs(run.chain, clip, oracle_variant())
    oracle = _Oracle(variant_results(run.suite, clip, v8, InputLevel.ORACLE), track_of(v8))
    del v8
    cells: dict[Cell, Paired] = {}
    units: dict[str, str] = {}
    for cell, variant in zip(CELLS, run.cells, strict=True):
        cells[cell], units[variant.name] = _pairs(run, clip, variant, InputLevel.MIXED, oracle)
    v0, _ = _pairs(run, clip, perceived_variant(), InputLevel.PERCEIVED, oracle)
    return ClipCells(clip.clip_id, oracle.results, cells, v0, units)


def _results_by_source(parts: Sequence[ClipCells]) -> dict[str, list[CheckResult]]:
    def others(paired: Paired) -> list[CheckResult]:
        return [p.other for p in paired.pairs] + list(paired.spurious)

    sources = {ORACLE: [r for part in parts for r in part.oracle]}
    for cell in CELLS:
        sources[cell_name(cell)] = [r for part in parts for r in others(part.cells[cell])]
    sources[perceived_variant().name] = [r for part in parts for r in others(part.v0)]
    return sources


def _collect(parts: Sequence[ClipCells]) -> CollectedBlocks:
    oracle = tuple(r for part in parts for r in part.oracle)
    cells = {cell: merge(part.cells[cell] for part in parts) for cell in CELLS}
    return collect_blocks(oracle, cells, merge(part.v0 for part in parts))


def _evaluate(
    run: _Run, clips: Sequence[ClipManifest], failures: list[Failure]
) -> tuple[ClipCells, ...]:
    failed = {f.clip_id for f in failures}
    parts: list[ClipCells] = []
    for clip in (c for c in clips if c.clip_id not in failed):
        try:
            parts.append(_clip_cells(run, clip))
        except (PhysverifyError, ValueError) as error:
            failures.append(Failure.from_error(clip.clip_id, error))
    return tuple(parts)


def _provenance(
    ctx: DiagnosisContext,
    dataset: DatasetSource,
    selected: Sequence[ClipManifest],
    evaluated: frozenset[str],
) -> Provenance:
    return build_provenance(
        ctx.run,
        dataset=dataset.name,
        dataset_versions=dataset_versions(dataset, (c for c in selected if c.clip_id in evaluated)),
        subset=ctx.diag.subset,
        error_profile=ctx.deps.error_profile,
        clips=(c.clip_id for c in selected),
    )


def run_attribution(ctx: DiagnosisContext, clips: Sequence[ClipManifest]) -> AttributionReport:
    """Level 3 over the ``report_split`` clips: 8 cells, V8 as the oracle and V0 for the budget."""
    suite = require_suite(ctx)
    selected = report_clips(ctx, clips)
    factorial = ctx.diag.factorial
    fixed = tuple(s for s in STAGE_ORDER if s in frozenset(factorial.fixed_ground_truth))
    dataset = ctx.dataset()
    cells = factorial_cells(frozenset(fixed))
    run = _Run(suite, ctx.chain(dataset), ctx.diag.end_to_end.min_span_iou, cells)
    failures = list(
        warm_variants(run.chain, selected, (oracle_variant(), *cells, perceived_variant()))
    )
    parts = _evaluate(run, selected, failures)
    clusters = tuple(part.clip_id for part in parts)
    sources = _results_by_source(parts)
    collected = _collect(parts)
    budgeting = Budgeting(fixed, tuple(sorted(ctx.metrics)))
    checks = tuple(
        attribute(
            check_id, check_units(check_id, sources), collected, clusters, factorial, budgeting
        )
        for check_id in sorted({r.check_id for results in sources.values() for r in results})
    )
    units = {name: [part.lift_units[name] for part in parts] for name in map(cell_name, CELLS)}
    return AttributionReport(
        provenance=_provenance(ctx, dataset, selected, frozenset(clusters)),
        report_split=ctx.diag.report_split,
        fixed_ground_truth=fixed,
        lift_unit=lift_unit(units),
        bootstrap=factorial.bootstrap,
        seed=factorial.seed,
        ci=factorial.ci,
        checks=checks,
        failures=tuple(failures),
    )
