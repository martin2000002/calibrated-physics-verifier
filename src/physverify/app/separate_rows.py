"""The ``verify-separate`` service: the rows reported apart from the nine checks.

They run unscored (``raw`` policy, no thresholds). ``fit_separate_row_sigma`` fits R1_dyn's
out-of-fold contact-row sigma on clean calib clips; ``run_separate`` given that artifact runs each
clip with the sigma of its scene's fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from physverify.app.calibration import subject_lookup
from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import dataset_for
from physverify.app.verification import input_source
from physverify.calibration.row_sigma import (
    ROW_SIGMA_FILE,
    RowSigma,
    fit_row_sigma,
    row_residuals,
)
from physverify.calibration.table import is_clean
from physverify.checks._image import FloatArray
from physverify.checks.separate._base import SEPARATE_CHECKS
from physverify.config.plugin_context import registry_of
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import UsageError
from physverify.core.jsonio import read_model
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource, select_clips
from physverify.runtime.runs import new_run
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.stages import Stage, parse_stages
from physverify.verification.inputs import OracleSource
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import (
    RowStatus,
    VerificationReport,
    VerificationRunner,
    write_report,
)
from physverify.verification.summary import parse_level, summary_lines
from physverify.verification.thresholding import ThresholdTable

ROW_SIGMA_CHECK = "R1_dyn"


def run_separate(
    context: AppContext,
    command: str,
    dataset: str,
    level: str | None,
    gt: Sequence[str],
    *,
    subset: str | None,
    clips: Sequence[str],
    echo: Echo,
    err: Echo,
    row_sigma: Path | None = None,
) -> bool:
    """Run the separate rows with their default parameters, R1_dyn with the fold sigma of the
    ``row_sigma`` artifact when given; whether a row errored."""
    input_level = parse_level(level, gt, sweep=False)
    fitted = None if row_sigma is None else _fitted(row_sigma, input_level, parse_stages(gt))
    source = dataset_for(context, dataset)
    selected = select_clips(source.clips(subset), clips)
    inputs = input_source(context, source, input_level, parse_stages(gt))
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    echo(f"run: {run_dir}")
    selection = [Selection(source.name, subset, tuple(selected))]
    guard_sealed(context, command, selection, err=err, run_dir=run_dir)
    parts = [
        _runner(context, sigma).run(inputs, group)
        for sigma, group in _fold_groups(selected, fitted)
    ]
    report = _joined(parts, selected)
    write_report(run_dir, report)
    for line in summary_lines(report):
        echo(line)
    return any(row.status is RowStatus.ERROR for row in report.table)


def fit_separate_row_sigma(
    context: AppContext,
    command: str,
    dataset: str,
    level: str | None,
    gt: Sequence[str],
    *,
    subset: str | None,
    clips: Sequence[str],
    echo: Echo,
    err: Echo,
) -> Path:
    """Fit R1_dyn's contact-row sigma on the clean calib clips of the selection (perceived rows
    against ground-truth rows; a clip without inputs or subjects is listed without residuals);
    the artifact's path."""
    input_level, stages = parse_level(level, gt, sweep=False), parse_stages(gt)
    if input_level is InputLevel.ORACLE:
        raise UsageError("the row sigma is fitted on perceived inputs (it is 0 in the oracle)")
    source = dataset_for(context, dataset)
    selected = [c for c in select_clips(source.clips(subset), clips) if is_clean(c.perturbation)]
    outside = sorted(c.clip_id for c in selected if c.split is not Split.CALIB)
    if outside:
        raise UsageError(f"the row sigma is fitted on calib clips only, not {outside[:5]}")
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    echo(f"run: {run_dir}")
    selection = [Selection(source.name, subset, tuple(selected))]
    guard_sealed(context, command, selection, err=err, run_dir=run_dir)
    fitted = fit_row_sigma(
        _residuals(context, source, input_level, stages, selected),
        {c.clip_id: c.scene_id for c in selected if c.scene_id is not None},
        check_id=ROW_SIGMA_CHECK,
        level=input_level,
        gt_stages=[stage.value for stage in stages],
    )
    path = run_dir / ROW_SIGMA_FILE
    write_text_atomic(path, fitted.model_dump_json(by_alias=True, indent=2) + "\n")
    echo(f"row sigma by fold (px): {', '.join(f'{s:.4g}' for s in fitted.by_fold)}")
    echo(f"residuals per fold: {list(fitted.residuals_per_fold)}; pooled {fitted.pooled:.4g} px")
    echo(f"clips without residuals: {len(fitted.clips_without_residuals)}")
    echo(f"row sigma: {path}")
    return path


def _fitted(path: Path, level: InputLevel, stages: Sequence[Stage]) -> RowSigma:
    fitted = read_model(RowSigma, path)
    wanted = tuple(stage.value for stage in stages)
    if fitted.level is not level or fitted.gt_stages != wanted:
        raise UsageError(
            f"{path} was fitted at level {fitted.level.value} with --gt {list(fitted.gt_stages)},"
            f" not {level.value} with {list(wanted)}"
        )
    return fitted


def _residuals(
    context: AppContext,
    source: DatasetSource,
    level: InputLevel,
    stages: Sequence[Stage],
    clips: Sequence[ClipManifest],
) -> dict[str, FloatArray]:
    perceived = input_source(context, source, level, stages).load(clips)
    truth = OracleSource(source).load(clips)
    lookup = subject_lookup(context, source)
    out: dict[str, FloatArray] = {}
    for seen, true in zip(perceived, truth, strict=True):
        subjects = lookup(seen.clip)
        if seen.inputs is None or true.inputs is None or subjects is None:
            out[seen.clip.clip_id] = np.empty(0)
        else:
            out[seen.clip.clip_id] = row_residuals(seen.inputs, true.inputs, subjects.ids)
    return out


def _fold_groups(
    clips: Sequence[ClipManifest], fitted: RowSigma | None
) -> list[tuple[float | None, list[ClipManifest]]]:
    """Clips grouped by the row sigma they run with (None: the check's own sigma)."""
    if fitted is None:
        return [(None, list(clips))]
    groups: dict[float | None, list[ClipManifest]] = {}
    for clip in clips:
        groups.setdefault(fitted.sigma_of(clip), []).append(clip)
    return list(groups.items())


def _runner(context: AppContext, row_sigma: float | None) -> VerificationRunner:
    checks = [
        cls(cls.Params.model_validate({"row_sigma": row_sigma}))
        if row_sigma is not None and cls.check_id == ROW_SIGMA_CHECK
        else cls(cls.Params())
        for cls in registry_of(context, SEPARATE_CHECKS).values()
    ]
    return VerificationRunner(checks, RawPolicy(), ThresholdTable.provisional({}))


def _joined(
    parts: Sequence[VerificationReport], clips: Sequence[ClipManifest]
) -> VerificationReport:
    """The parts' results and rows in clip order (each part keeps its own order within a clip)."""
    order = {clip.clip_id: i for i, clip in enumerate(clips)}
    results = sorted((r for p in parts for r in p.results), key=lambda r: order[r.clip_id])
    table = sorted((r for p in parts for r in p.table), key=lambda r: order[r.clip_id])
    return VerificationReport(tuple(results), tuple(table), tuple(clips))
