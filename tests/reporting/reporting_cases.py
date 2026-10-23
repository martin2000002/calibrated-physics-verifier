"""A fake A8 v2 chain directory over the calibration fixtures, for the results collector."""

from __future__ import annotations

import functools
import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from calibration_cases import (
    CHECKS,
    PROFILE,
    config,
    dataset_ref,
    fit_meta,
    synthetic_level,
    synthetic_report,
)

from physverify.app.sealing import ledger_label
from physverify.calibration.apply import score_clip
from physverify.calibration.artifact import EvaluationReport
from physverify.calibration.artifact_io import Calibration, parse_calibration
from physverify.calibration.collect import level_data
from physverify.calibration.evaluate import (
    EvaluationInputs,
    EvaluationMeta,
    PrimaryScope,
    evaluate,
)
from physverify.calibration.fit import FitInputs, fit_calibration
from physverify.calibration.sigma_recalibration import SigmaPair
from physverify.core.digests import text_sha256
from physverify.datasets._videophy_tables import VideophyRow
from physverify.human_agreement.clip_decision import (
    CalibratedRule,
    ClipDecisionRule,
    ProvisionalAnyFail,
)
from physverify.human_agreement.comparison import Comparison, ComparisonInputs, compare
from physverify.human_agreement.report import REPORT_DIR, write_comparison
from physverify.human_agreement.settings import AgreementSettings
from physverify.human_agreement.undecided_policy import AS_PLAUSIBLE
from physverify.human_agreement.verify_run_reader import VerifyRun
from physverify.reporting.chain import ABSTAIN_CURVE
from physverify.runtime.runs import write_json as write_abstain_json
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.schema.clip import Split
from physverify.schema.io import json_ready
from physverify.selective_risk.curve import Case, CaseCounts, raw_table
from physverify.selective_risk.fit import per_check_rows
from physverify.selective_risk.report import CurveParts, curve_record, fit_record, real_rows
from physverify.selective_risk.report import artifact_of as abstain_table
from physverify.selective_risk.settings import ABSTENTION_FILE, REPORT_FILE, AbstentionSettings
from physverify.selective_risk.tables import Collected, compare_tables, named_ks, sweep
from physverify.synth.scene_scope import SceneScope
from physverify.verification.exclusion_record import ExclusionRecord, InputHashes, RuleEvidence
from physverify.verification.runner import TableRow, VerificationReport

PRIMARY = ("s000", "s011-s039")
LAST_STEP = 11
FIXTURE_ROOT = Path("/fixture")
ARTIFACT_NAME = "calibration.json"
SIGMA = 0.005
ABSTENTION = AbstentionSettings(risk_target=0.2, bootstrap=50, grid_points=10)
AGREEMENT = AgreementSettings(
    rule="calibrated",
    primary_policy=AS_PLAUSIBLE,
    alpha=0.05,
    power=0.8,
    bootstrap_n=50,
    bootstrap_seed=1,
    ci=0.95,
    permutation_exact_max=20,
    error_sample_per_class=2,
    error_seed=1,
    dev_subset="h2_dev",
    test_subset="h2_test",
    preregistration="agreement/preregistration.json",
)


@functools.cache
def artifact_text() -> str:
    oracle = synthetic_report(40, prefix="fam-s")
    perceived = synthetic_report(40, prefix="fam-s", level=InputLevel.PERCEIVED, sigma=SIGMA)
    levels = (
        synthetic_level(oracle, InputLevel.ORACLE),
        level_data(
            InputLevel.PERCEIVED, perceived, CHECKS, config(), split=Split.CALIB, pairs=_pairs()
        ),
    )
    return fit_calibration(FitInputs(levels, CHECKS, fit_meta()), config()).to_json()


def _pairs() -> tuple[SigmaPair, ...]:
    """Measured R1 errors, so that the perceived stratum has a sigma factor and decides."""
    rng = np.random.default_rng(5)
    return tuple(
        SigmaPair(f"fam-s{i:03d}", "R1", 0.1, 0.1 + float(rng.normal(0, SIGMA)), SIGMA)
        for i in range(40)
    )


def _joined(*reports: VerificationReport) -> VerificationReport:
    return VerificationReport(
        results=tuple(r for rep in reports for r in rep.results),
        table=tuple(t for rep in reports for t in rep.table),
        clips=tuple(c for rep in reports for c in rep.clips),
    )


def _inputs(level: InputLevel) -> EvaluationInputs:
    sigma = SIGMA if level is InputLevel.PERCEIVED else 0.0
    calib = synthetic_report(40, prefix="fam-s", level=level, sigma=sigma)
    heldout = synthetic_report(
        40, seed=7, split=Split.EVAL, prefix="held-s", level=level, sigma=sigma
    )
    synthetic = level_data(level, _joined(calib, heldout), CHECKS, config(), split=None)
    if level is InputLevel.ORACLE:
        return EvaluationInputs(synthetic, {})
    real = synthetic_report(20, seed=9, split=Split.EVAL, prefix="r", level=level)
    return EvaluationInputs(
        synthetic, {"real_fixture": level_data(level, real, CHECKS, config(), split=None)}
    )


@functools.cache
def evaluation() -> EvaluationReport:
    meta = EvaluationMeta(
        "2026-10-20T00:00:00Z",
        "eval-a",
        "calibration.json",
        dataset_ref(),
        PROFILE,
        primary=PrimaryScope(SceneScope(primary_scenes=PRIMARY), dataset_ref()),
    )
    inputs = tuple(_inputs(level) for level in (InputLevel.ORACLE, InputLevel.PERCEIVED))
    return evaluate(parse_calibration(artifact_text()), inputs, meta, config())


@dataclass(frozen=True)
class Chain:
    dir: Path
    artifact: Path
    evaluation: Path
    abstention: Path
    curve_run: Path
    record: Path
    h2_run: Path


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> Path:
    return write(path, json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def build_chain(root: Path, *, device: str = "cpu", codes: Mapping[int, str] = {}) -> Chain:
    calibration_dir, runs = root / "data" / "calibration", root / "data" / "runs"
    artifact = write(calibration_dir / ARTIFACT_NAME, artifact_text())
    report = evaluation().model_copy(update={"artifact_path": str(artifact)})
    evaluation_path = write(runs / "eval-a" / "evaluation.json", report.to_json())
    curve_run = runs / "abstain-a"
    abstention = _write_abstention(artifact, curve_run)
    record = _record(calibration_dir, artifact, evaluation_path)
    chain = Chain(
        dir=root / "a8v2",
        artifact=artifact,
        evaluation=evaluation_path,
        abstention=abstention,
        curve_run=curve_run,
        record=record,
        h2_run=runs / "h2",
    )
    write_metrics(chain)
    _chain_files(chain, device, codes)
    return chain


def _chain_files(chain: Chain, device: str, codes: Mapping[int, str]) -> None:
    sealed = json.loads(chain.record.read_text(encoding="utf-8"))["sha256"]
    state = {"CHAIN_DEVICE": device, "EVAL_JSON": str(chain.evaluation), "RECORD_SHA": sealed}
    write(chain.dir / "state.env", "".join(f"{k}={shlex.quote(v)}\n" for k, v in state.items()))
    rows = [f"{n}\tstep-{n}\t{codes.get(n, '0')}\t1\t-\n" for n in range(LAST_STEP + 1)]
    write(chain.dir / "steps.tsv", "".join(rows))
    logs = chain.dir / "logs"
    write(logs / "7-abstain-1.log", f"$ pv {ledger_label('selective-risk fit')}\nk*: 2.0000\n")
    write(
        logs / "7-abstain-2.log", f"$ pv {ABSTAIN_CURVE}\neval cases: 3\nrun: {chain.curve_run}\n"
    )


def _calibration() -> Calibration:
    """The artifact as the consumers load it; only its file name and sha reach their output."""
    return parse_calibration(artifact_text(), path=FIXTURE_ROOT / ARTIFACT_NAME)


def _projection(level: InputLevel) -> dict[str, Any]:
    return _calibration().stratum(level, PROFILE).thresholds_projection(FIXTURE_ROOT)


def _tables() -> dict[str, Any]:
    """``tables_of`` of an abstain session on this artifact."""
    return {
        level.value: abstain_table(_projection(level))
        for level in (InputLevel.ORACLE, InputLevel.PERCEIVED)
    }


def _collected(n_scenes: int, seed: int, prefix: str) -> Collected:
    """Perceived R1 events with the injected label as reference (as on real video: PASS)."""
    report = synthetic_report(
        n_scenes, seed, split=Split.EVAL, prefix=prefix, level=InputLevel.PERCEIVED, sigma=SIGMA
    )
    target = {c.clip_id: c.perturbation is not None and "R1" in c.perturbation.target_checks
              for c in report.clips}  # fmt: skip
    cases = [
        Case(r.clip_id, r.check_id, Verdict.FAIL if target[r.clip_id] else Verdict.PASS, r)
        for r in report.results
        if r.check_id == "R1"
    ]
    return Collected(cases, CaseCounts(), {c.clip_id: c for c in report.clips}, [])


@functools.cache
def _abstention_records() -> tuple[dict[str, Any], dict[str, Any]]:
    """``selective-risk fit`` and ``selective-risk curve`` content through their own builders."""
    tables, calibrated = _tables(), _projection(InputLevel.PERCEIVED)
    calib = _collected(30, 11, "ab-s")
    fitted = fit_record(calib, sweep(calib, calibrated, ABSTENTION), ABSTENTION, tables)
    k = fitted["k_star"]
    held = _collected(30, 12, "ah-s")
    parts = CurveParts(
        per_check=per_check_rows(held.cases, calibrated, named_ks(k), ABSTENTION.ci),
        injected_label={"rows": 0},
        z={"n": 0},
        real_video={"physics_iq": real_rows(_collected(10, 13, "pi-s"), calibrated, k, 0.95)},
    )
    compared = compare_tables(held, calibrated, raw_table(calibrated), ABSTENTION)
    return fitted, curve_record(held, compared, k, ABSTENTION, tables, parts)


def _write_abstention(artifact: Path, curve_run: Path) -> Path:
    fitted, report = _abstention_records()
    write_abstain_json(curve_run / REPORT_FILE, report)
    path = artifact.parent / ABSTENTION_FILE
    write_abstain_json(path, fitted)
    return path


def _record(calibration_dir: Path, artifact: Path, evaluation_path: Path) -> Path:
    text = evaluation_path.read_text(encoding="utf-8")
    record = ExclusionRecord(
        rules_version="1",
        consumed_level="perceived",
        consumed_profile=PROFILE,
        checks_hash="checks",
        inputs=InputHashes(
            artifact_sha256=text_sha256(artifact.read_text(encoding="utf-8")),
            artifact_run_id="run-a",
            evaluation_sha256=text_sha256(text),
            evaluation_run_id="eval-a",
            r2_base_run="r2-base",
            r2_base_report_sha256="b",
            r2_tiny_run="r2-tiny",
            r2_tiny_report_sha256="t",
            rules_sha256="rules",
        ),
        evidence=(
            RuleEvidence(rule="1.1", check_id="R1", excluded=False, decided=True, reason="kept"),
            RuleEvidence(
                rule="1.5", check_id="R5b", excluded=False, decided=False, reason="no events"
            ),
        ),
        exclude_checks=(),
        h3_families_without_target=(),
    ).sealed()
    return write(calibration_dir / "exclusions.json", record.to_json())


def _videophy_rows(report: VerificationReport) -> list[VideophyRow]:
    """Human: perturbed clips implausible; the VLM gets every third clip wrong."""
    rows = []
    for i, clip in enumerate(report.clips):
        perturbed = clip.perturbation is not None
        vlm_right = i % 3 != 0
        rows.append(
            VideophyRow(
                part="test",
                version="v1",
                split=Split.EVAL,
                clip_id=clip.clip_id,
                video_url=f"https://example.invalid/{clip.clip_id}.mp4",
                caption=f"caption {i % 6}",
                generator="g",
                human_raw=1.0 if perturbed else 5.0,
                human_cut=3.0,
                reference_raw=(1.0 if perturbed == vlm_right else 5.0),
                reference_cut=3.0,
                reference_name="autoeval",
                violated_rules="",
            )
        )
    return rows


def _verify_run(path: Path, report: VerificationReport) -> VerifyRun:
    rows: dict[str, list[TableRow]] = {}
    results: dict[str, list[CheckResult]] = {}
    for row in report.table:
        rows.setdefault(row.clip_id, []).append(row)
    for result in report.results:
        results.setdefault(result.clip_id, []).append(result)
    return VerifyRun(
        path=path,
        run={"profile": PROFILE, "device": "cpu"},
        rows={k: tuple(v) for k, v in rows.items()},
        results={k: tuple(v) for k, v in results.items()},
    )


def write_metrics(
    chain: Chain, *, level: InputLevel = InputLevel.PERCEIVED, provisional: bool = False
) -> Path:
    """``agreement compare``'s files through ``write_comparison``."""
    write_comparison(chain.h2_run / REPORT_DIR, _comparison(level, provisional))
    return chain.h2_run / REPORT_DIR / "metrics.json"


@functools.cache
def _comparison(level: InputLevel, provisional: bool) -> Comparison:
    """``compare`` with A8's calibrated rule on the stratum ``level`` (or the provisional rule)."""
    report = synthetic_report(8, 21, split=Split.EVAL, prefix="vp-s", level=InputLevel.PERCEIVED)
    rule: ClipDecisionRule = (
        ProvisionalAnyFail(_projection(InputLevel.PERCEIVED))
        if provisional
        else CalibratedRule(_calibration().stratum(level, PROFILE), score_clip)
    )
    return compare(
        ComparisonInputs(
            rows=_videophy_rows(report),
            run=_verify_run(FIXTURE_ROOT / "h2", report),
            rule=rule,
            agreement=AGREEMENT,
            cut=0.5,
            dev_tuned_cut=None,
            dev_majority=0,
            preregistered=not provisional,
        )
    )
