"""Raw-policy reports per input level, read back from ``verify`` runs, and the D9 sigma pairs."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from physverify.calibration.fit import LevelData
from physverify.calibration.settings import CalibrationConfig
from physverify.calibration.sigma import SigmaPair
from physverify.calibration.table import CalibrationDataError, build_rows, scene_of
from physverify.diagnosis.pairing import pair_results
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.io import read_jsonl
from physverify.schema.matching import IdMap
from physverify.verification.runner import (
    CHECKS_FILE,
    TABLE_FILE,
    RowStatus,
    TableRow,
    VerificationReport,
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
) -> LevelData:
    rows = build_rows(report, checks, config.couplings, level, split=split)
    return LevelData(level=level, rows=rows, results=report.results, sigma_pairs=pairs)


# --- --from-run ---------------------------------------------------------------------------------


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
