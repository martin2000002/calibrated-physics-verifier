"""The aggregate AUROC an evaluation stratum reports: the fit's clip score on held-out clips."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from calibration_cases import (
    CHECKS,
    config,
    dataset_ref,
    fit_meta,
    synthetic_level,
    synthetic_report,
)

from physverify.baselines.verifier_values import B1
from physverify.calibration.api import calibration_api
from physverify.calibration.apply import score_clip
from physverify.calibration.artifact import EvaluationReport
from physverify.calibration.artifact_io import Calibration, Stratum, parse_calibration
from physverify.calibration.clip_values import ClipValueOf
from physverify.calibration.collect import level_data
from physverify.calibration.evaluate import (
    EvaluationInputs,
    EvaluationMeta,
    GradedOf,
    evaluate,
    heldout_auroc,
)
from physverify.calibration.fit import FitInputs, LevelData, fit_calibration
from physverify.schema.checks import CheckResult, InputLevel
from physverify.schema.clip import Split
from physverify.synth.plan import family_of
from physverify.verification.runner import VerificationReport


def _data(*reports: VerificationReport) -> LevelData:
    joined = VerificationReport(
        results=tuple(r for report in reports for r in report.results),
        table=tuple(t for report in reports for t in report.table),
        clips=tuple(c for report in reports for c in report.clips),
    )
    return level_data(InputLevel.ORACLE, joined, CHECKS, config(), split=None)


@pytest.fixture(scope="module")
def frozen() -> Calibration:
    level = synthetic_level(synthetic_report(40, prefix="fam-s"), InputLevel.ORACLE)
    artifact = fit_calibration(FitInputs((level,), CHECKS, fit_meta()), config())
    return parse_calibration(artifact.to_json())


def _report(
    calibration: Calibration, data: LevelData, graded: GradedOf | None = None
) -> EvaluationReport:
    meta = EvaluationMeta(
        "2026-10-21T00:00:00Z", "eval-h", "calibration.json", dataset_ref(), "fast"
    )
    inputs = (EvaluationInputs(data, {}),)
    return evaluate(calibration, inputs, meta, config(), family_of, graded)


def _one_minus_s(stratum: Stratum) -> ClipValueOf:
    def value(results: Sequence[CheckResult]) -> float | None:
        score = score_clip(results, stratum).score
        return None if score is None else 1.0 - score

    return value


def test_calib_clips_never_enter_the_heldout_auroc(frozen: Calibration) -> None:
    stratum = frozen.stratum(InputLevel.ORACLE, "fast")
    held = synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s")
    mixed = _data(synthetic_report(40, prefix="fam-s"), held)
    rows = tuple(r for r in mixed.rows if r.split is not Split.CALIB)
    found = heldout_auroc(stratum, mixed, rows, config().calibration)
    assert found == heldout_auroc(stratum, _data(held), _data(held).rows, config().calibration)
    assert found.value is not None and not found.in_sample
    assert found.n_positive > 0 and found.n_negative > 0


def test_the_evaluation_carries_it_and_an_older_report_still_reads(frozen: Calibration) -> None:
    held = synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s")
    report = _report(frozen, _data(synthetic_report(40, prefix="fam-s"), held))
    assert report.strata[0].auroc is not None
    data = json.loads(report.to_json())
    data["version"] = "1.9.0"
    for stratum in data["strata"]:
        del stratum["auroc"]
    assert EvaluationReport.model_validate_json(json.dumps(data)).strata[0].auroc is None


def test_the_graded_auroc_uses_the_clips_bootstrap_and_seed_of_the_s_auroc(
    frozen: Calibration,
) -> None:
    """Injected with 1 - S itself, the graded AUROC is the S AUROC to the last digit: the
    two rows differ only by the clip value."""
    held = synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s")
    data = _data(synthetic_report(40, prefix="fam-s"), held)
    stratum = _report(frozen, data, _one_minus_s).strata[0]
    assert stratum.auroc is not None and stratum.auroc.value is not None
    assert stratum.auroc_graded == stratum.auroc
    assert _report(frozen, data).strata[0].auroc_graded is None


def test_b1_excluding_every_check_leaves_no_clip_to_rank(frozen: Calibration) -> None:
    held = synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s")
    data = _data(synthetic_report(40, prefix="fam-s"), held)
    everything = frozenset(frozen.stratum(InputLevel.ORACLE, "fast").model.checks)
    found = _report(frozen, data, B1(calibration_api(), everything).under).strata[0]
    assert found.auroc_graded is not None and found.auroc_graded.value is None
    assert found.auroc_graded.n_positive == found.auroc_graded.n_negative == 0
