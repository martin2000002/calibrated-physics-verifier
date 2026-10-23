"""The records ``selective-risk fit`` and ``selective-risk curve`` write, and the lines they
print."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from physverify.calibration.table import Role, role_of
from physverify.core.errors import MissingInputError, PhysverifyError
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.io import stable_hash
from physverify.selective_risk.curve import (
    ABSTAINED,
    FAILED,
    PASSED,
    Case,
    curve_point,
    decision_codes,
    injected_reference,
    reference_codes,
    row_codes,
)
from physverify.selective_risk.fit import per_check_rows, point_row, z_scores
from physverify.selective_risk.records import (
    AbstentionRecord,
    ComparedTables,
    Counts,
    CurveRecord,
    FittedK,
    InjectedLabel,
    PointRow,
    RealError,
    RealRows,
    SweptTable,
    TableRef,
    ZSummary,
)
from physverify.selective_risk.settings import ABSTENTION_FILE, AbstentionSettings
from physverify.selective_risk.tables import (
    DEFAULT_K,
    Collected,
    Session,
    collect_real,
    named_ks,
)
from physverify.verification.thresholding import ThresholdTable

_NAMES = {ABSTAINED: "abstain", PASSED: "PASS", FAILED: "FAIL"}


def artifact_of(thresholds: ThresholdTable) -> TableRef:
    return TableRef(
        status=thresholds.status,
        source=thresholds.source,
        artifact=thresholds.artifact,
        table_hash=stable_hash(thresholds.dump()),
    )


def tables_of(session: Session) -> dict[str, TableRef]:
    return {
        InputLevel.ORACLE.value: artifact_of(session.reference_table),
        InputLevel.PERCEIVED.value: artifact_of(session.thresholds),
    }


def _counts(collected: Collected) -> Counts:
    return Counts(
        cases=len(collected.cases),
        clips=len({c.clip_id for c in collected.cases}),
        clips_failed=len(collected.failed),
        events_without_reference=collected.counts.no_reference,
        spurious_perceived=collected.counts.spurious,
        missed_by_perception=sum(c.perceived is None for c in collected.cases),
    )


def fit_record(
    collected: Collected,
    result: SweptTable,
    settings: AbstentionSettings,
    tables: Mapping[str, TableRef],
) -> AbstentionRecord:
    """The content of ``abstention.json``; ``result`` is ``sweep`` on the calib cases."""
    return AbstentionRecord(
        dataset=settings.dataset,
        split=settings.calib_split,
        risk_target=settings.risk_target,
        ci=settings.ci,
        alpha_star=result.alpha_star,
        k_star=result.k_star,
        n=_counts(collected),
        thresholds=dict(tables),
        curve=result.curve,
    )


@dataclass(frozen=True)
class CurveParts:
    """What ``selective-risk curve`` computes beyond the two swept tables."""

    per_check: dict[str, dict[str, PointRow]]
    injected_label: InjectedLabel
    z: ZSummary
    real_video: dict[str, RealRows | RealError]


def curve_parts(session: Session, evaluated: Collected, fitted: float | None) -> CurveParts:
    """Per check, against the injected label, z and real video, all on the perceived table."""
    calibrated, settings = session.thresholds, session.settings
    return CurveParts(
        per_check=per_check_rows(evaluated.cases, calibrated, named_ks(fitted), settings.ci),
        injected_label=_injected(session, evaluated, fitted),
        z=_z_summary(evaluated.cases, calibrated),
        real_video=_real(session, fitted),
    )


def curve_record(
    evaluated: Collected,
    tables: ComparedTables,
    fitted: float | None,
    settings: AbstentionSettings,
    thresholds: Mapping[str, TableRef],
    parts: CurveParts,
) -> CurveRecord:
    """The content of ``abstention_report.json``; ``tables`` is ``compare_tables``."""
    return CurveRecord(
        split=settings.eval_split,
        k_star_from_calib=fitted,
        n=_counts(evaluated),
        thresholds=dict(thresholds),
        tables=tables,
        per_check=parts.per_check,
        injected_label=parts.injected_label,
        z=parts.z,
        real_video=parts.real_video,
    )


def fitted_k(session: Session) -> float | None:
    path = session.calibration_dir / ABSTENTION_FILE
    if not path.is_file():
        raise MissingInputError(f"no {path}; run `physverify selective-risk fit` first")
    return FittedK.model_validate_json(path.read_text(encoding="utf-8")).k_star


def _row_role(session: Session, clip: ClipManifest, check_id: str) -> Role:
    return role_of(clip, check_id, session.checks, session.couplings)


def _injected(session: Session, collected: Collected, k: float | None) -> InjectedLabel:
    """Risk at k (default 2) of the (clip, check) rows against the calibration chain's injected
    label, and the rows where the oracle's point verdict disagrees with that label."""
    cases = collected.cases
    keys = [(c.clip_id, c.check_id) for c in cases]
    labels = {
        key: label
        for key in dict.fromkeys(keys)
        if (label := injected_reference(_row_role(session, collected.clips[key[0]], key[1])))
        is not None
    }
    if not labels:
        return InjectedLabel(rows=0)
    chosen, s = DEFAULT_K if k is None else k, session.settings
    decided = row_codes(keys, decision_codes(cases, [chosen], session.thresholds)[0])
    oracle = row_codes(keys, reference_codes(cases))
    order = sorted(labels)
    label_codes = reference_codes([Case(*key, labels[key], None) for key in order])
    point = curve_point(np.array([decided[key] for key in order]), label_codes)
    return InjectedLabel(
        rows=len(order),
        at=point_row(point, chosen, float(norm.sf(chosen)), s.ci),
        oracle_label_disagreements=tuple(_disagreements(order, oracle, label_codes)),
    )


def _disagreements(
    order: Sequence[tuple[str, str]],
    oracle: dict[tuple[str, str], int],
    labels: NDArray[np.int64],
) -> list[str]:
    return [
        f"{clip}:{check}:oracle={_NAMES[oracle[(clip, check)]]}:label={_NAMES[int(label)]}"
        for (clip, check), label in zip(order, labels, strict=True)
        if oracle[(clip, check)] != label
    ]


def _z_summary(cases: Sequence[Case], thresholds: ThresholdTable) -> ZSummary:
    z = np.array(z_scores(cases, thresholds))
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return ZSummary(n=0)
    return ZSummary(
        n=int(finite.size),
        quantiles={str(q): float(np.quantile(finite, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    )


def _real(session: Session, k: float | None) -> dict[str, RealRows | RealError]:
    """False-fail rate and coverage on real video at k* and k = 2 (every reference PASS)."""
    out: dict[str, RealRows | RealError] = {}
    s = session.settings
    for dataset in s.real_datasets:
        try:
            collected = collect_real(session, dataset)
        except PhysverifyError as error:
            out[dataset] = RealError(error=str(error))
            continue
        out[dataset] = real_rows(collected, session.thresholds, k, s.ci)
    return out


def real_rows(
    collected: Collected, thresholds: ThresholdTable, k: float | None, ci: float
) -> RealRows:
    """One real dataset: counts and the point rows at k* and k = 2."""
    reference = reference_codes(collected.cases)
    rows: dict[str, PointRow] = {}
    for name, value in named_ks(k).items():
        codes = decision_codes(collected.cases, [value], thresholds)[0]
        rows[name] = point_row(curve_point(codes, reference), value, float(norm.sf(value)), ci)
    return RealRows(n=_counts(collected), **rows)


def curve_csv(tables: ComparedTables) -> str:
    fields = ("table", "alpha", "k_sigma", "n", "decided", "coverage", "risk")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow((*fields, "risk_low", "risk_high"))
    for name, swept in (("raw", tables.raw), ("calibrated", tables.calibrated)):
        for row in swept.curve:
            low, high = row.risk_ci
            writer.writerow((name, *(getattr(row, f) for f in fields[1:]), low, high))
    return buffer.getvalue()


def summary_lines(report: CurveRecord) -> list[str]:
    tables = report.tables
    diff = tables.aurc_calibrated_minus_raw
    lines = [f"eval cases: {report.n.model_dump()}", f"k* (calib): {report.k_star_from_calib}"]
    lines.append(f"  raw: k* on eval {tables.raw.k_star}")
    lines.append(f"  calibrated: k* on eval {tables.calibrated.k_star}")
    lines.append(f"  AURC calibrated - raw: {diff.value:.4f} {diff.ci}")
    injected = report.injected_label
    if injected.rows and injected.at is not None:
        at = injected.at
        disagreements = injected.oracle_label_disagreements or ()
        lines.append(
            f"  injected label: risk {at.risk:.3f} coverage {at.coverage:.3f} "
            f"({len(disagreements)} oracle disagreements)"
        )
    return lines


def fit_lines(record: AbstentionRecord, risk_target: float, target: Path) -> list[str]:
    lines = [f"cases: {record.n.model_dump()}"]
    if record.k_star is None:
        lines.append(f"k*: none (no k reaches risk upper bound <= {risk_target})")
    else:
        alpha = record.alpha_star
        lines.append(f"k*: {record.k_star:.4f} (alpha* {alpha:.3g})")
    lines.append(f"written: {target}")
    return lines
