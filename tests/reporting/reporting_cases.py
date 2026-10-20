"""A fake A8 v2 chain directory over the calibration fixtures, for the results collector."""

from __future__ import annotations

import functools
import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibration_cases import (
    CHECKS,
    PROFILE,
    config,
    dataset_ref,
    fit_meta,
    synthetic_level,
    synthetic_report,
)

from physverify.agreement.analysis import h2_verdict
from physverify.calibration.collect import level_data
from physverify.calibration.evaluate import (
    EvaluationInputs,
    EvaluationMeta,
    PrimaryScope,
    evaluate,
)
from physverify.calibration.exclusion_record import ExclusionRecord, InputHashes, RuleEvidence
from physverify.calibration.fit import FitInputs, fit_calibration
from physverify.calibration.loader import load_calibration, parse_calibration, sha256_of
from physverify.calibration.scenes import SceneScope
from physverify.calibration.schema import EvaluationReport
from physverify.cli.commands.abstain import artifact_of as abstain_table
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.schema.io import json_ready
from physverify.verification.runner import VerificationReport

PRIMARY = ("s000", "s011-s039")
POLICY = "abstain_as_plausible"
K_STAR = 2.0
LAST_STEP = 11


@functools.cache
def artifact_text() -> str:
    oracle = synthetic_report(40, prefix="fam-s")
    perceived = synthetic_report(40, prefix="fam-s", level=InputLevel.PERCEIVED)
    levels = (
        synthetic_level(oracle, InputLevel.ORACLE),
        synthetic_level(perceived, InputLevel.PERCEIVED),
    )
    return fit_calibration(FitInputs(levels, CHECKS, fit_meta()), config()).to_json()


def _joined(*reports: VerificationReport) -> VerificationReport:
    return VerificationReport(
        results=tuple(r for rep in reports for r in rep.results),
        table=tuple(t for rep in reports for t in rep.table),
        clips=tuple(c for rep in reports for c in rep.clips),
    )


def _inputs(level: InputLevel) -> EvaluationInputs:
    calib = synthetic_report(40, prefix="fam-s", level=level)
    heldout = synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s", level=level)
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
    artifact = write(calibration_dir / "calibration.json", artifact_text())
    report = evaluation().model_copy(update={"artifact_path": str(artifact)})
    evaluation_path = write(runs / "eval-a" / "evaluation.json", report.to_json())
    curve_run = runs / "abstain-a"
    tables = _tables(artifact)
    abstention = write_json(calibration_dir / "abstention.json", _fit(tables))
    write_json(curve_run / "abstention_report.json", _curve(tables))
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
    write_json(chain.h2_run / "agreement" / "metrics.json", metrics(artifact))
    _chain_files(chain, device, codes)
    return chain


def _chain_files(chain: Chain, device: str, codes: Mapping[int, str]) -> None:
    sealed = json.loads(chain.record.read_text(encoding="utf-8"))["sha256"]
    state = {"CHAIN_DEVICE": device, "EVAL_JSON": str(chain.evaluation), "RECORD_SHA": sealed}
    write(chain.dir / "state.env", "".join(f"{k}={shlex.quote(v)}\n" for k, v in state.items()))
    rows = [f"{n}\tstep-{n}\t{codes.get(n, '0')}\t1\t-\n" for n in range(LAST_STEP + 1)]
    write(chain.dir / "steps.tsv", "".join(rows))
    logs = chain.dir / "logs"
    write(logs / "7-abstain-1.log", "$ pv abstain fit\nk*: 2.0000\n")
    write(logs / "7-abstain-2.log", f"$ pv abstain curve\neval cases: 3\nrun: {chain.curve_run}\n")


def _tables(artifact: Path) -> dict[str, Any]:
    calibration = load_calibration(artifact)
    return {
        level.value: abstain_table(
            calibration.stratum(level, PROFILE).thresholds_projection(artifact.parent)
        )
        for level in (InputLevel.ORACLE, InputLevel.PERCEIVED)
    }


def curve_row(k: float, risk: float) -> dict[str, Any]:
    return {
        "k_sigma": k,
        "alpha": 0.02,
        "n": 50,
        "decided": 40,
        "coverage": 0.8,
        "risk": risk,
        "risk_ci": [risk / 2, risk * 2],
        "false_fail_rate": 0.05,
        "missed_fail_rate": float("nan"),
    }


def _curve_rows() -> list[dict[str, Any]]:
    return [curve_row(k, 0.05 * k) for k in (1.0, K_STAR, 3.0)]


def _fit(tables: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimated": True,
        "risk_target": 0.2,
        "alpha_star": 0.0228,
        "k_star": K_STAR,
        "thresholds": tables,
        "curve": _curve_rows(),
    }


def _curve(tables: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimated": True,
        "k_star_from_calib": K_STAR,
        "thresholds": tables,
        "tables": {
            "raw": {"k_star": 1.0, "curve": _curve_rows()},
            "calibrated": {"k_star": K_STAR, "curve": _curve_rows()},
            "aurc_calibrated_minus_raw": {"value": -0.01, "ci": [-0.03, 0.005]},
        },
        "real_video": {
            "physics_iq": {
                "n": {"cases": 4},
                "k_star": curve_row(K_STAR, 0.1),
                "k_2": curve_row(2.0, 0.1),
            }
        },
    }


def _record(calibration_dir: Path, artifact: Path, evaluation_path: Path) -> Path:
    text = evaluation_path.read_text(encoding="utf-8")
    record = ExclusionRecord(
        rules_version="1",
        consumed_level="perceived",
        consumed_profile=PROFILE,
        checks_hash="checks",
        inputs=InputHashes(
            artifact_sha256=sha256_of(artifact.read_text(encoding="utf-8")),
            artifact_run_id="run-a",
            evaluation_sha256=sha256_of(text),
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


def metrics(
    artifact: Path, *, rule: str | None = None, preregistered: bool = True
) -> dict[str, Any]:
    source = load_calibration(artifact).stratum(InputLevel.PERCEIVED, PROFILE).source
    primary = {
        "policy": POLICY,
        "paired": True,
        "verifier": {"n": 100, "accuracy": 0.62, "kappa": 0.21, "auc": 0.66},
        "vlm": {"n": 100, "accuracy": 0.55, "kappa": 0.1, "auc": float("nan")},
        "constants": {},
        "mcnemar": {"b_verifier_only": 20, "c_vlm_only": 13, "p_value": 0.3},
        "bootstrap": {
            "intervals": {
                "verifier.accuracy": [0.52, 0.71],
                "verifier.kappa": [0.05, 0.36],
                "vlm.accuracy": [0.45, 0.64],
                "vlm.kappa": [-0.02, 0.22],
                "difference.accuracy": [-0.03, 0.17],
                "difference.kappa": [-0.1, 0.3],
            }
        },
    }
    return {
        "rule": rule or f"calibrated:{source}",
        "preregistered": preregistered,
        "h2_group": "all",
        "primary_policy": POLICY,
        "groups": {"all": {POLICY: primary}},
        "h2_verdict": h2_verdict(primary, preregistered, 0.05),
    }
