"""Small hand-built records for the figure tests: an evaluation, a selective-risk curve and an
attribution, each holding only what the figures read; and a selective-risk curve built by the
study's own builders."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from physverify.calibration.artifact import (
    OK,
    Auroc,
    DatasetRef,
    EvaluationReport,
    EvaluationStratum,
    HeldoutCheck,
    LevelTpr,
    RealFpr,
)
from physverify.diagnosis.report import AttributionReport, CheckAttribution
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.schema.stages import Stage
from physverify.selective_risk.curve import Case, CaseCounts, CurvePoint
from physverify.selective_risk.fit import per_check_rows, point_row
from physverify.selective_risk.records import (
    ComparedTables,
    CurveRecord,
    InjectedLabel,
    PointRow,
    SweptTable,
    ZSummary,
)
from physverify.selective_risk.report import CurveParts, curve_record
from physverify.selective_risk.settings import AbstentionSettings
from physverify.selective_risk.tables import Collected, compare_tables, named_ks, sweep
from physverify.stats.intervals import Ci, Interval, Rate, rate
from physverify.verification.thresholding import ThresholdTable

LEVEL_CI = 0.95
SHA = "0" * 64
KUBRIC_SET = "kubric_set"
EVENT_CHECKS = ("R1", "R2", "R3a")
TAU, SIGMA, NOISE, MISSED = 1.0, 0.2, 0.4, 0.1
TABLE = ThresholdTable.model_validate(
    {"status": "calibrated", "source": "test", "checks": {c: {"tau": TAU} for c in EVENT_CHECKS}}
)
ABSTENTION = AbstentionSettings(risk_target=0.10, ci=LEVEL_CI, grid_points=10, bootstrap=20)


def share(k: int, n: int) -> Rate:
    return rate(k, n, LEVEL_CI)


def heldout(
    check_id: str, clean: Rate, levels: Sequence[tuple[float | None, Rate]], first_level: int = 1
) -> HeldoutCheck:
    """``levels``: the median injected magnitude and the detection rate of each level, whose ids
    count up from ``first_level``."""
    return HeldoutCheck(
        check_id=check_id,
        status=OK,
        tau=1.0,
        fpr_clean=clean,
        fpr_clean_decided=clean,
        fpr_off_target=share(0, 1),
        tpr_by_level=tuple(
            LevelTpr(level=first_level + i, mu_median=mu, total=found, decided=found)
            for i, (mu, found) in enumerate(levels)
        ),
    )


def auroc(value: float | None, low: float, high: float, *, in_sample: bool = False) -> Auroc:
    return Auroc(
        value=value, ci=Ci(low=low, high=high), n_positive=4, n_negative=4, in_sample=in_sample
    )


def stratum(
    level: InputLevel,
    checks: Sequence[HeldoutCheck] = (),
    real: Sequence[RealFpr] = (),
    *,
    graded: Auroc | None = None,
    plain: Auroc | None = None,
) -> EvaluationStratum:
    return EvaluationStratum(
        level=level,
        profile="default",
        checks=tuple(checks),
        scene_kendall=(),
        real=tuple(real),
        real_clip_flags=(),
        auroc=plain,
        auroc_graded=graded,
    )


def dataset(name: str, n_clips: int) -> DatasetRef:
    return DatasetRef(
        name=name,
        version="1",
        subset="all",
        split="test",
        n_clips=n_clips,
        n_scenes=n_clips,
        clip_ids_hash=SHA,
    )


def evaluation(
    *strata: EvaluationStratum, real_sets: Sequence[DatasetRef] = ()
) -> EvaluationReport:
    return EvaluationReport(
        created_utc="fixed",
        run_id="run",
        artifact_path="calibration.json",
        artifact_sha256=SHA,
        artifact_run_id="run",
        dataset=dataset(KUBRIC_SET, 8),
        real_datasets=tuple(ref.name for ref in real_sets),
        strata=strata,
        real_sets=tuple(real_sets),
    )


def point(
    k: float, coverage: float, risk: float, low: float = math.nan, high: float = math.nan
) -> PointRow:
    return PointRow(
        k_sigma=k,
        alpha=0.05,
        n=10,
        decided=5,
        coverage=coverage,
        risk=risk,
        risk_ci=(low, high),
        false_fail_rate=risk,
        missed_fail_rate=0.0,
    )


def counted(k: float, n: int, decided: int, errors: int) -> PointRow:
    """A point row from its counts, through the study's own row builder."""
    return point_row(CurvePoint(n, decided, errors, errors, decided, 0, 0), k, 0.05, LEVEL_CI)


def swept(*points: PointRow, best: int | None = None) -> SweptTable:
    return SweptTable(curve=points, best=best, alpha_star=None, k_star=None)


def curve(
    calibrated: SweptTable,
    raw: SweptTable,
    *,
    k_star: float | None = None,
    per_check: Mapping[str, Mapping[str, PointRow]] = {},
) -> CurveRecord:
    tables = ComparedTables.model_construct(raw=raw, calibrated=calibrated)
    return CurveRecord.model_construct(
        tables=tables,
        k_star_from_calib=k_star,
        per_check={check: dict(rows) for check, rows in per_check.items()},
    )


def events(n: int, seed: int) -> Collected:
    """Reference events of three checks: the reference is the point verdict of a true magnitude,
    the perceived magnitude adds more noise than its sigma says, and perception misses some."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        check_id, truth = EVENT_CHECKS[i % len(EVENT_CHECKS)], float(rng.uniform(0.0, 2 * TAU))
        seen = truth + float(rng.normal(0.0, NOISE))
        found = None if rng.random() < MISSED else perceived(check_id, f"c{i}", seen)
        cases.append(Case(f"c{i}", check_id, Verdict.FAIL if truth > TAU else Verdict.PASS, found))
    return Collected(cases, CaseCounts(), {}, [])


def perceived(check_id: str, clip_id: str, magnitude: float) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        clip_id=clip_id,
        subject=("a",),
        frame_span=(0, 3),
        magnitude=magnitude,
        sigma=SIGMA,
        unit="",
        verdict=Verdict.UNSCORED,
        input_level=InputLevel.PERCEIVED,
    )


def built_curve(held: Collected, configured: float) -> CurveRecord:
    """``selective-risk fit`` on other events, then ``selective-risk curve`` on ``held``, both
    through the study's own builders."""
    fitted = sweep(events(300, 1), TABLE, ABSTENTION).table.k_star
    parts = CurveParts(
        per_check=per_check_rows(held.cases, TABLE, named_ks(fitted, configured), ABSTENTION.ci),
        injected_label=InjectedLabel(rows=0),
        z=ZSummary(n=0),
        real_video={},
    )
    compared = compare_tables(held, TABLE, TABLE.raw(), ABSTENTION)
    return curve_record(held, compared, fitted, {}, parts)


def attributed(
    check_id: str,
    n_clips: int,
    shapley: Mapping[Stage, float],
    intervals: Mapping[Stage, tuple[float, float]] = {},
) -> CheckAttribution:
    return CheckAttribution(
        check_id=check_id,
        unit="m",
        n_blocks=1,
        n_clips=n_clips,
        cells={},
        v0=0.0,
        effects={},
        shapley=dict(shapley),
        shapley_ci={stage: Interval(low=lo, high=hi) for stage, (lo, hi) in intervals.items()},
        budget=(),
    )


def attribution(*checks: CheckAttribution) -> AttributionReport:
    return AttributionReport.model_construct(checks=checks)
