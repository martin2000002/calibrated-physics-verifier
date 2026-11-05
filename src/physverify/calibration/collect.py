"""Raw-policy reports per input level, read back from ``verify`` runs, and the per-check sigma
pairs."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from physverify.calibration.extension import NO_EXTENSION, ExtensionScope, restrict
from physverify.calibration.fit import LevelData
from physverify.calibration.settings import CalibrationConfig
from physverify.calibration.sigma_recalibration import SigmaPair
from physverify.calibration.table import CalibrationDataError, build_rows
from physverify.config.models import is_cuda_device
from physverify.core.errors import ConfigError, UsageError
from physverify.diagnosis.pairing import pair_results
from physverify.runtime.runs import run_record
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.io import read_jsonl
from physverify.schema.matching import IdMap
from physverify.synth.scene_scope import scene_of
from physverify.verification.runner import (
    CHECKS_FILE,
    CHECKS_HASH_FILE,
    TABLE_FILE,
    RowStatus,
    TableRow,
    VerificationReport,
    run_checks_hash,
)
from physverify.verification.sweep import CheckIdentity


@dataclass(frozen=True)
class Subjects:
    """Perceived ids to ground-truth ids of one clip (``ids`` None: the ids already agree)."""

    ids: IdMap | None


SubjectLookup = Callable[[ClipManifest], Subjects | None]
"""None when the clip has no usable subject map (it is skipped and counted)."""


@dataclass(frozen=True)
class SigmaPairs:
    pairs: tuple[SigmaPair, ...]
    unpaired_clips: tuple[str, ...]


def sigma_pairs(
    oracle: VerificationReport,
    perceived: VerificationReport,
    lookup: SubjectLookup,
    min_span_iou: float,
) -> SigmaPairs:
    """One pair per subject event that both levels scored (``diagnosis.pairing``)."""
    found: list[SigmaPair] = []
    unpaired: list[str] = []
    for clip in perceived.clips:
        subjects = lookup(clip)
        if subjects is None:
            unpaired.append(clip.clip_id)
            continue
        found += _clip_pairs(clip, oracle.results, perceived.results, subjects, min_span_iou)
    return SigmaPairs(tuple(found), tuple(unpaired))


def _clip_pairs(
    clip: ClipManifest,
    oracle: Sequence[CheckResult],
    perceived: Sequence[CheckResult],
    subjects: Subjects,
    min_span_iou: float,
) -> list[SigmaPair]:
    mine = [r for r in perceived if r.clip_id == clip.clip_id]
    reference = [r for r in oracle if r.clip_id == clip.clip_id]
    paired = pair_results(reference, mine, subjects.ids, min_span_iou)
    return [
        SigmaPair(
            scene_id=scene_of(clip),
            check_id=p.check_id,
            oracle=float(p.oracle.magnitude or 0.0),
            perceived=float(p.other.magnitude or 0.0),
            sigma=p.other.sigma,
            oracle_unit=p.oracle.unit,
            perceived_unit=p.other.unit,
        )
        for p in paired.pairs
        if p.delta is not None
    ]


def level_data(
    level: InputLevel,
    report: VerificationReport,
    checks: Sequence[CheckIdentity],
    config: CalibrationConfig,
    *,
    split: Split | None,
    pairs: tuple[SigmaPair, ...] | None = (),
    scope: ExtensionScope = NO_EXTENSION,
) -> LevelData:
    rows = build_rows(report, checks, config.couplings, level, split=split)
    data = LevelData(level=level, rows=rows, results=report.results, sigma_pairs=pairs)
    return restrict(data, scope, {clip.clip_id: scene_of(clip) for clip in report.clips})


def read_run(run_dir: Path, clips: Sequence[ClipManifest]) -> VerificationReport:
    """A ``verify`` run directory back as a report; ``clips`` are the dataset's manifests."""
    checks_path, table_path = run_dir / CHECKS_FILE, run_dir / TABLE_FILE
    if not checks_path.is_file() or not table_path.is_file():
        raise CalibrationDataError(f"{run_dir} has no {CHECKS_FILE} and {TABLE_FILE}")
    results = tuple(read_jsonl(CheckResult, checks_path))
    if any(r.verdict in (Verdict.PASS, Verdict.FAIL) for r in results):
        raise CalibrationDataError(f"{run_dir} was decided by a policy: rerun with --policy raw")
    table = tuple(_table(table_path))
    by_id = {clip.clip_id: clip for clip in clips}
    missing = sorted({row.clip_id for row in table} - set(by_id))
    if missing:
        raise CalibrationDataError(f"{run_dir}: clips not in the dataset: {missing[:5]}")
    seen = tuple(by_id[c] for c in dict.fromkeys(row.clip_id for row in table))
    return VerificationReport(results=results, table=table, clips=seen)


def require_same_checks(run_dir: Path, checks_hash: str, device: str) -> None:
    """A ``--from-run`` run was verified with the session's checks and on its kind of device:
    rows of another check variant, or of cuda perception against a cpu session, never mix."""
    recorded = run_checks_hash(run_dir)
    if recorded != checks_hash:
        found = recorded or f"no {CHECKS_HASH_FILE}"
        raise UsageError(
            f"--from-run {run_dir.name} was verified with checks_hash {found}, this session "
            f"has {checks_hash}: rerun verify with the session's [check_variants]"
        )
    record = run_record(run_dir)
    if "device" not in record:
        raise ConfigError(f"--from-run {run_dir.name}: run predates device recording; rerun verify")
    ran_on = str(record["device"])
    if is_cuda_device(ran_on) != is_cuda_device(device):
        raise UsageError(
            f"--from-run {run_dir.name} ran on {ran_on!r}, this session on {device!r}: "
            "pass --device to match or rerun verify"
        )


def run_level_of(report: VerificationReport) -> InputLevel:
    levels = {r.input_level for r in report.results}
    if len(levels) != 1:
        raise CalibrationDataError(f"a run must hold one input level, got {sorted(levels)}")
    return levels.pop()


def _table(path: Path) -> list[TableRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [_row(record) for record in csv.DictReader(handle)]


def _row(cells: dict[str, str]) -> TableRow:
    return TableRow(
        clip_id=cells["clip_id"],
        check_id=cells["check_id"],
        status=RowStatus(cells["status"]),
        magnitude=float(cells["magnitude"]) if cells["magnitude"] else None,
        sigma=float(cells["sigma"]) if cells["sigma"] else math.nan,
        verdict=Verdict(cells["verdict"]) if cells["verdict"] else None,
        abstain_reason=AbstainReason(cells["abstain_reason"]) if cells["abstain_reason"] else None,
        n_subjects=int(cells["n_subjects"]),
        n_abstained=int(cells["n_abstained"]),
        unit=cells["unit"],
        row_step=int(cells["row_step"]),
    )
