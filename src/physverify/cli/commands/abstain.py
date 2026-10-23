"""``physverify abstain``: risk-coverage of the threshold policy and the choice of k* (ADR 0010 §3).

Not heavy: the checks read cached stage outputs (V8 oracle and V0 perceived chains) and a stage
missing from the cache fails its clip, which is counted. The reference of an event is the
oracle's point verdict; on real video, where there is no oracle, every scored event is PASS."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from scipy.stats import norm

from physverify.calibration.loader import default_artifact_path, load_calibration
from physverify.calibration.sealed import file_sha
from physverify.calibration.settings import checks_hash
from physverify.calibration.table import Role, role_of
from physverify.cli.commands._runs import new_run_dir, write_json
from physverify.cli.commands._sealed import Selection, guard_sealed
from physverify.cli.commands.extract import pipeline_deps, resolve_dataset
from physverify.cli.context import AppContext, app_context
from physverify.core.errors import MissingInputError, PhysverifyError, UsageError
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.atomic_io import write_text_atomic
from physverify.diagnosis.chain import ChainRunner
from physverify.diagnosis.checks_port import A7CheckSuite
from physverify.diagnosis.end_to_end import track_of, variant_outputs, variant_results
from physverify.diagnosis.pairing import pair_results, subject_map
from physverify.diagnosis.plans import oracle_variant, perceived_variant
from physverify.diagnosis.settings import diagnosis_settings
from physverify.schema.checks import InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.io import stable_hash
from physverify.verification.inputs import refuse_heavy
from physverify.verification.runner import (
    CHECKS_SECTION,
    VERIFICATION_SECTION,
    ChecksSettings,
    VerificationSettings,
    load_thresholds,
)
from physverify.verification.selective_risk import (
    ABSTAINED,
    ABSTENTION_FILE,
    CURVE_FILE,
    FAILED,
    PASSED,
    REPORT_FILE,
    SECTION,
    AbstentionSettings,
    Case,
    CaseCounts,
    alpha_grid,
    choose_k,
    curve_point,
    curve_rows,
    decision_codes,
    injected_reference,
    k_of,
    paired_aurc_difference,
    paired_cases,
    per_check_rows,
    point_row,
    raw_table,
    reference_codes,
    require_taus,
    row_codes,
    z_scores,
)
from physverify.verification.sweep import CheckUnit

NAME = "abstain"
DEFAULT_K = 2.0
_NAMES = {ABSTAINED: "abstain", PASSED: "PASS", FAILED: "FAIL"}

app = typer.Typer(help="Risk-coverage of the verdict policy and the abstention level k*.")

OracleTableOption = Annotated[
    Path | None,
    typer.Option(help="Oracle-level table for the reference (default: A8's oracle stratum)."),
]
PerceivedTableOption = Annotated[
    Path | None,
    typer.Option(help="Perceived-level table for the decisions (default: A8's perceived stratum)."),
]
ProvisionalOption = Annotated[
    bool,
    typer.Option(
        help="Ignore A8's artifact: both levels use verify's fallback table "
        "(paths.thresholds, else the provisional taus)."
    ),
]


@dataclass(frozen=True)
class Collected:
    cases: list[Case]
    counts: CaseCounts
    clips: dict[str, ClipManifest]
    failed: list[str]


@dataclass(frozen=True)
class Session:
    context: AppContext
    settings: AbstentionSettings
    suite: A7CheckSuite
    reference_table: dict[str, Any]
    """Oracle level: the reference is the point verdict against its raw taus."""
    thresholds: dict[str, Any]
    """Perceived level: the table every swept decision uses."""
    min_span_iou: float
    checks: tuple[CheckUnit, ...]
    couplings: frozenset[tuple[str, str]]


def level_tables(
    artifact: Path | None,
    profile: str,
    fallback: Mapping[str, Any],
    overrides: Mapping[InputLevel, Path],
    *,
    stale_hash: str | None = None,
) -> dict[InputLevel, dict[str, Any]]:
    """Per level: an explicit table file, else the projection of A8's stratum (ADR 0009 §7),
    else ``fallback`` (what ``verify`` uses) when there is no artifact."""
    for level, given in overrides.items():
        if not given.is_file():
            raise UsageError(f"no {level.value} threshold table at {given}")
    calibration = (
        load_calibration(artifact, checks_hash=stale_hash)
        if artifact is not None and artifact.is_file()
        else None
    )
    root = None if artifact is None else artifact.parent
    out: dict[InputLevel, dict[str, Any]] = {}
    for level in (InputLevel.ORACLE, InputLevel.PERCEIVED):
        path = overrides.get(level)
        if path is not None:
            out[level] = dict(load_thresholds(path, {}))
        elif calibration is not None:
            stratum = calibration.stratum(level, profile)
            out[level] = stratum.thresholds_projection(root)
        else:
            out[level] = dict(fallback)
    return out


def _session(oracle: Path | None, perceived: Path | None, provisional: bool) -> Session:
    context = app_context()
    settings = context.settings
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    registry = context.registry("checks")
    fallback = load_thresholds(settings.paths.thresholds, verification.thresholds)
    overrides = {
        level: path
        for level, path in ((InputLevel.ORACLE, oracle), (InputLevel.PERCEIVED, perceived))
        if path is not None
    }
    tables = level_tables(
        None if provisional else default_artifact_path(settings),
        settings.profile,
        fallback,
        overrides,
        stale_hash=checks_hash(registry, settings.section(CHECKS_SECTION, ChecksSettings)),
    )
    for level, table in tables.items():
        require_taus(table, level)
    return Session(
        context=context,
        settings=settings.section(SECTION, AbstentionSettings),
        suite=A7CheckSuite.from_context(context),
        reference_table=tables[InputLevel.ORACLE],
        thresholds=tables[InputLevel.PERCEIVED],
        min_span_iou=diagnosis_settings(settings).end_to_end.min_span_iou,
        checks=tuple(CheckUnit.of(cls) for _, cls in sorted(registry.items())),
        couplings=frozenset((p, c) for p, c in verification.specificity.couplings),
    )


def _chain(session: Session, dataset: str) -> tuple[ChainRunner, DatasetSource]:
    context = session.context
    source = resolve_dataset(context, dataset)
    deps = pipeline_deps(context, None, partial(refuse_heavy, dataset))
    return ChainRunner(context.settings, context.registry("providers"), deps, source), source


def collect_synthetic(session: Session, subset: str) -> Collected:
    """Oracle/perceived pairs of every cached clip of ``subset`` as cases."""
    chain, source = _chain(session, session.settings.dataset)
    clips = list(source.clips(subset))
    cases: list[Case] = []
    counts, failed = CaseCounts(), []
    suite = session.suite
    for clip in clips:
        try:
            v8 = variant_outputs(chain, clip, oracle_variant())
            v0 = variant_outputs(chain, clip, perceived_variant())
            oracle = variant_results(suite, clip, v8, InputLevel.ORACLE)
            perceived = variant_results(suite, clip, v0, InputLevel.PERCEIVED)
            subjects = subject_map(track_of(v0), track_of(v8))
        except (PhysverifyError, ValueError):
            failed.append(clip.clip_id)
            continue
        paired = pair_results(oracle, perceived, subjects, session.min_span_iou)
        found, seen = paired_cases(paired, session.reference_table)
        cases += found
        counts += seen
    return Collected(cases, counts, {c.clip_id: c for c in clips}, failed)


def collect_real(session: Session, dataset: str) -> Collected:
    """Perceived events of real video, all with reference PASS (real physics)."""
    chain, source = _chain(session, dataset)
    clips = list(source.clips(None))
    cases: list[Case] = []
    failed = []
    for clip in clips:
        try:
            v0 = variant_outputs(chain, clip, perceived_variant())
            results = variant_results(session.suite, clip, v0, InputLevel.PERCEIVED)
        except (PhysverifyError, ValueError):
            failed.append(clip.clip_id)
            continue
        cases += [
            Case(r.clip_id, r.check_id, Verdict.PASS, r) for r in results if r.magnitude is not None
        ]
    return Collected(cases, CaseCounts(), {c.clip_id: c for c in clips}, failed)


def _grid(settings: AbstentionSettings) -> tuple[np.ndarray, np.ndarray]:
    alphas = alpha_grid(settings)
    return alphas, k_of(alphas)


def _require(collected: Collected, what: str) -> None:
    if not collected.cases and collected.counts.no_reference:
        raise MissingInputError(
            f"{what}: none of {collected.counts.no_reference} oracle events has a reference; "
            "the oracle table has no tau for their checks"
        )
    if not collected.cases:
        raise MissingInputError(
            f"{what}: no case ({len(collected.failed)} clips without cached chains); "
            "run `physverify extract run` first"
        )


def artifact_of(thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": thresholds.get("status"),
        "source": thresholds.get("source"),
        "artifact": thresholds.get("artifact"),
        "table_hash": stable_hash(thresholds),
    }


def tables_of(session: Session) -> dict[str, Any]:
    return {
        InputLevel.ORACLE.value: artifact_of(session.reference_table),
        InputLevel.PERCEIVED.value: artifact_of(session.thresholds),
    }


def sweep(
    collected: Collected, thresholds: dict[str, Any], settings: AbstentionSettings
) -> dict[str, Any]:
    alphas, ks = _grid(settings)
    reference = reference_codes(collected.cases)
    codes = decision_codes(collected.cases, ks, thresholds)
    points = [curve_point(row, reference) for row in codes]
    best = choose_k(points, list(ks), settings.risk_target, settings.ci)
    return {
        "curve": curve_rows(codes, reference, list(ks), list(alphas), settings.ci),
        "best": best,
        "alpha_star": None if best is None else float(alphas[best]),
        "k_star": None if best is None else float(ks[best]),
        "codes": codes,
    }


def _counts(collected: Collected) -> dict[str, Any]:
    return {
        "cases": len(collected.cases),
        "clips": len({c.clip_id for c in collected.cases}),
        "clips_failed": len(collected.failed),
        "events_without_reference": collected.counts.no_reference,
        "spurious_perceived": collected.counts.spurious,
        "missed_by_perception": sum(c.perceived is None for c in collected.cases),
    }


@app.command("fit", help="Choose k* on the calibration split; writes abstention.json.")
def fit(
    oracle_table: OracleTableOption = None,
    perceived_table: PerceivedTableOption = None,
    provisional: ProvisionalOption = False,
) -> None:
    _guard(app_context(), "fit", real=False)
    session = _session(oracle_table, perceived_table, provisional)
    settings = session.settings
    collected = collect_synthetic(session, settings.calib_split)
    _require(collected, settings.calib_split)
    result = sweep(collected, session.thresholds, settings)
    record = fit_record(collected, result, settings, tables_of(session))
    target = session.context.settings.paths.calibration_dir / ABSTENTION_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, record)
    typer.echo(f"cases: {record['n']}")
    if result["k_star"] is None:
        typer.echo(f"k*: none (no k reaches risk upper bound <= {settings.risk_target})")
    else:
        typer.echo(f"k*: {result['k_star']:.4f} (alpha* {result['alpha_star']:.3g})")
    typer.echo(f"written: {target}")


def fit_record(
    collected: Collected,
    result: Mapping[str, Any],
    settings: AbstentionSettings,
    tables: Mapping[str, Any],
) -> dict[str, Any]:
    """The content of ``abstention.json``; ``result`` is ``sweep`` on the calib cases."""
    return {
        "estimated": True,
        "dataset": settings.dataset,
        "split": settings.calib_split,
        "risk_target": settings.risk_target,
        "ci": settings.ci,
        "alpha_star": result["alpha_star"],
        "k_star": result["k_star"],
        "n": _counts(collected),
        "thresholds": tables,
        "curve": result["curve"],
    }


@dataclass(frozen=True)
class CurveParts:
    """What ``abstain curve`` computes beyond the two swept tables."""

    per_check: dict[str, Any]
    injected_label: dict[str, Any]
    z: dict[str, Any]
    real_video: dict[str, Any]


def curve_record(
    evaluated: Collected,
    tables: Mapping[str, Any],
    fitted: float | None,
    settings: AbstentionSettings,
    thresholds: Mapping[str, Any],
    parts: CurveParts,
) -> dict[str, Any]:
    """The content of ``abstention_report.json``; ``tables`` is ``compare_tables``."""
    return {
        "estimated": True,
        "split": settings.eval_split,
        "k_star_from_calib": fitted,
        "n": _counts(evaluated),
        "thresholds": thresholds,
        "tables": tables,
        "per_check": parts.per_check,
        "injected_label": parts.injected_label,
        "z": parts.z,
        "real_video": parts.real_video,
    }


@app.command("curve", help="Held-out curve, raw vs calibrated table, per check, real video.")
def curve(
    oracle_table: OracleTableOption = None,
    perceived_table: PerceivedTableOption = None,
    provisional: ProvisionalOption = False,
) -> None:
    _guard(app_context(), "curve", real=True)
    session = _session(oracle_table, perceived_table, provisional)
    settings = session.settings
    fitted = _fitted_k(session)
    evaluated = collect_synthetic(session, settings.eval_split)
    _require(evaluated, settings.eval_split)
    calibrated, raw = session.thresholds, raw_table(session.thresholds)
    run_dir = new_run_dir(session.context)
    tables = compare_tables(evaluated, calibrated, raw, settings)
    parts = CurveParts(
        per_check=per_check_rows(evaluated.cases, calibrated, named_ks(fitted), settings.ci),
        injected_label=_injected(session, evaluated, fitted),
        z=_z_summary(evaluated.cases, calibrated),
        real_video=_real(session, fitted),
    )
    report = curve_record(evaluated, tables, fitted, settings, tables_of(session), parts)
    write_text_atomic(run_dir / CURVE_FILE, _curve_csv(tables))
    write_json(run_dir / REPORT_FILE, report)
    for line in summary_lines(report):
        typer.echo(line)
    typer.echo(f"run: {run_dir}")


def _guard(context: AppContext, leaf: str, *, real: bool) -> None:
    """The sealed-ledger guard over the clips ``collect_synthetic`` (the eval split with
    ``real``, else calib) and ``collect_real`` read; a dataset that does not resolve is reported
    where it is read."""
    settings = context.settings
    own = settings.section(SECTION, AbstentionSettings)
    wanted: list[tuple[str, str | None]] = [
        (own.dataset, own.eval_split if real else own.calib_split)
    ]
    wanted += [(name, None) for name in own.real_datasets] if real else []
    selections = []
    for name, subset in wanted:
        try:
            source = resolve_dataset(context, name)
        except PhysverifyError:
            continue
        selections.append(Selection(source.name, subset, tuple(source.clips(subset))))
    abstention = file_sha(settings.paths.calibration_dir / ABSTENTION_FILE)
    shas = {} if abstention is None else {"abstention": abstention}
    guard_sealed(context, f"{NAME} {leaf}", selections, artifacts=shas)


def _fitted_k(session: Session) -> float | None:
    path = session.context.settings.paths.calibration_dir / ABSTENTION_FILE
    if not path.is_file():
        raise MissingInputError(f"no {path}; run `physverify abstain fit` first")
    k = json.loads(path.read_text(encoding="utf-8")).get("k_star")
    return None if k is None else float(k)


def named_ks(fitted: float | None) -> dict[str, float]:
    return {"k_2": DEFAULT_K} if fitted is None else {"k_star": fitted, "k_2": DEFAULT_K}


def compare_tables(
    collected: Collected, calibrated: dict[str, Any], raw: dict[str, Any], s: AbstentionSettings
) -> dict[str, Any]:
    """Both tables swept on the same cases; AURC difference with a bootstrap paired by clip."""
    swept = {"raw": sweep(collected, raw, s), "calibrated": sweep(collected, calibrated, s)}
    reference = reference_codes(collected.cases)
    difference = paired_aurc_difference(
        swept["raw"]["codes"],
        swept["calibrated"]["codes"],
        reference,
        [c.clip_id for c in collected.cases],
        n_boot=s.bootstrap,
        seed=s.seed,
        ci=s.ci,
    )
    out: dict[str, Any] = {
        name: {k: v for k, v in result.items() if k != "codes"} for name, result in swept.items()
    }
    out["aurc_calibrated_minus_raw"] = difference
    out["identical_tables"] = calibrated["checks"] == raw["checks"]
    return out


def _row_role(session: Session, clip: ClipManifest, check_id: str) -> Role:
    return role_of(clip, check_id, session.checks, session.couplings)


def _injected(session: Session, collected: Collected, k: float | None) -> dict[str, Any]:
    """Risk at k (default 2) of the (clip, check) rows against A8's injected label, and the rows
    where the oracle's point verdict disagrees with that label."""
    cases = collected.cases
    keys = [(c.clip_id, c.check_id) for c in cases]
    labels = {
        key: label
        for key in dict.fromkeys(keys)
        if (label := injected_reference(_row_role(session, collected.clips[key[0]], key[1])))
        is not None
    }
    if not labels:
        return {"rows": 0}
    chosen, s = DEFAULT_K if k is None else k, session.settings
    decided = row_codes(keys, decision_codes(cases, [chosen], session.thresholds)[0])
    oracle = row_codes(keys, reference_codes(cases))
    order = sorted(labels)
    label_codes = reference_codes([Case(*key, labels[key], None) for key in order])
    point = curve_point(np.array([decided[key] for key in order]), label_codes)
    return {
        "rows": len(order),
        "at": point_row(point, chosen, float(norm.sf(chosen)), s.ci),
        "oracle_label_disagreements": _disagreements(order, oracle, label_codes),
    }


def _disagreements(
    order: Sequence[tuple[str, str]], oracle: dict[tuple[str, str], int], labels: np.ndarray
) -> list[str]:
    return [
        f"{clip}:{check}:oracle={_NAMES[oracle[(clip, check)]]}:label={_NAMES[int(label)]}"
        for (clip, check), label in zip(order, labels, strict=True)
        if oracle[(clip, check)] != label
    ]


def _z_summary(cases: Sequence[Case], thresholds: dict[str, Any]) -> dict[str, Any]:
    z = np.array(z_scores(cases, thresholds))
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "quantiles": {str(q): float(np.quantile(finite, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }


def _real(session: Session, k: float | None) -> dict[str, Any]:
    """False-fail rate and coverage on real video at k* and k = 2 (every reference PASS)."""
    out: dict[str, Any] = {}
    s = session.settings
    for dataset in s.real_datasets:
        try:
            collected = collect_real(session, dataset)
        except PhysverifyError as error:
            out[dataset] = {"error": str(error)}
            continue
        out[dataset] = real_rows(collected, session.thresholds, k, s.ci)
    return out


def real_rows(
    collected: Collected, thresholds: Mapping[str, Any], k: float | None, ci: float
) -> dict[str, Any]:
    """One real dataset: counts and the point rows at k* and k = 2."""
    reference = reference_codes(collected.cases)
    rows = {}
    for name, value in named_ks(k).items():
        codes = decision_codes(collected.cases, [value], thresholds)[0]
        rows[name] = point_row(curve_point(codes, reference), value, float(norm.sf(value)), ci)
    return {"n": _counts(collected), **rows}


def _curve_csv(tables: dict[str, Any]) -> str:
    fields = ("table", "alpha", "k_sigma", "n", "decided", "coverage", "risk")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow((*fields, "risk_low", "risk_high"))
    for name in ("raw", "calibrated"):
        for row in tables[name]["curve"]:
            low, high = row["risk_ci"]
            writer.writerow((name, *(row[f] for f in fields[1:]), low, high))
    return buffer.getvalue()


def summary_lines(report: dict[str, Any]) -> list[str]:
    tables = report["tables"]
    diff = tables["aurc_calibrated_minus_raw"]
    lines = [f"eval cases: {report['n']}", f"k* (calib): {report['k_star_from_calib']}"]
    for name in ("raw", "calibrated"):
        lines.append(f"  {name}: k* on eval {tables[name]['k_star']}")
    lines.append(f"  AURC calibrated - raw: {diff['value']:.4f} {diff['ci']}")
    injected = report["injected_label"]
    if injected.get("rows"):
        at = injected["at"]
        lines.append(
            f"  injected label: risk {at['risk']:.3f} coverage {at['coverage']:.3f} "
            f"({len(injected['oracle_label_disagreements'])} oracle disagreements)"
        )
    return lines
