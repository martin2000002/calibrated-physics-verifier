"""The calibrated scorer v2 opens its stratum through the calibration chain's ``open_stratum``, like
the human-agreement study does."""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

CALIBRATION_TESTS = str(Path(__file__).resolve().parent.parent / "calibration")
if CALIBRATION_TESTS not in sys.path:
    sys.path.insert(0, CALIBRATION_TESTS)

from stratum_cases import write_artifact  # noqa: E402
from wiring import (  # noqa: E402
    active_checks_hash,  # noqa: E402
    build_scorer,
    checks,
)

from physverify.app.finetune import write_calibration_status  # noqa: E402
from physverify.calibration.artifact_io import (  # noqa: E402
    StaleCalibrationError,
    default_artifact_path,
    load_calibration,
)
from physverify.config.models import Settings  # noqa: E402
from physverify.core.errors import ConfigError, UsageError  # noqa: E402
from physverify.finetune.layout import CampaignStore  # noqa: E402
from physverify.finetune.records import BASE_TAG  # noqa: E402
from physverify.finetune.scorers.calibrated import CalibratedScorer  # noqa: E402
from physverify.human_agreement.clip_decision import (  # noqa: E402
    StratumRequest,
    calibrated_rule,
    import_calibration,
)
from physverify.schema.checks import CheckResult, Verdict  # noqa: E402
from physverify.verification.runner import UNKNOWN_ROW_STEP  # noqa: E402

Result = Callable[..., CheckResult]
STALE = "f" * 64


def _artifact(settings: Settings, **options: object) -> Path:
    fields = {"profile": settings.profile, "checks_hash": active_checks_hash(settings)}
    return write_artifact(default_artifact_path(settings), **{**fields, **options})  # type: ignore[arg-type]


def _with_row_step(result: CheckResult, step: int) -> CheckResult:
    return dataclasses.replace(result, details={**result.details, "row_step": step})


def test_the_calibrated_scorer_is_version_two() -> None:
    assert CalibratedScorer.version == "2"


def test_an_insufficient_artifact_with_a_usable_stratum_scores(
    settings: Settings, make_result: Result
) -> None:
    _artifact(settings)
    scorer = CalibratedScorer(settings, checks())
    assert scorer.unavailable() is None
    assert scorer.thresholds().startswith("calibrated:")
    status = scorer.calibration_status()
    assert status is not None and status["artifact_status"] == "insufficient"
    assert status["checks"]["R1"]["status"] == "ok"
    score = scorer.score([make_result("R1", Verdict.PASS, 0.1, 1.0, 0.01)])
    assert score.decided == frozenset({"R1"})


def test_require_calibrated_artifact_refuses_an_insufficient_one(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings("calibration.require_calibrated_artifact=true", profile="precise")
    _artifact(settings)
    reason = CalibratedScorer(settings, checks()).unavailable()
    assert reason is not None and "not 'calibrated'" in reason


def test_a_stratum_that_cannot_score_is_unavailable(settings: Settings) -> None:
    _artifact(settings, ready=False)
    scorer = CalibratedScorer(settings, checks())
    reason = scorer.unavailable()
    assert reason is not None and "insufficient" in reason
    assert scorer.thresholds().startswith("calibrated:unusable@checks:")
    assert scorer.calibration_status() == {"problem": reason}


def test_a_clip_of_another_time_base_is_refused(settings: Settings, make_result: Result) -> None:
    _artifact(settings)
    scorer = CalibratedScorer(settings, checks())
    result = make_result("R1", Verdict.PASS, 0.1, 1.0, 0.01)
    assert scorer.score([_with_row_step(result, 1)]).decided == frozenset({"R1"})
    with pytest.raises(UsageError, match="row_step"):
        scorer.score([_with_row_step(result, 2)])
    with pytest.raises(UsageError, match="one time base"):
        scorer.score([_with_row_step(result, 1), _with_row_step(result, 2)])


def test_agreement_and_comparison_give_the_same_stale_message(settings: Settings) -> None:
    path = _artifact(settings, checks_hash=STALE)
    active = active_checks_hash(settings)
    with pytest.raises(StaleCalibrationError) as stale:
        load_calibration(path, checks_hash=active)
    message = str(stale.value)
    assert "other check parameters" in message and "refit" in message
    request = StratumRequest(path, settings.profile, UNKNOWN_ROW_STEP, active)
    with pytest.raises(ConfigError) as agreement_error:
        calibrated_rule(import_calibration(), request, ())
    assert message in str(agreement_error.value)
    assert build_scorer("calibrated", settings).unavailable() == message


def test_scoring_records_the_state_of_each_check(settings: Settings) -> None:
    _artifact(settings)
    store = CampaignStore.from_settings(settings)
    scorer = build_scorer("calibrated", settings)
    write_calibration_status(store, BASE_TAG, scorer)
    path = store.calibration_status_path(BASE_TAG, "calibrated")
    status = json.loads(path.read_text(encoding="utf-8"))
    assert status["checks"]["R1"] == {"status": "ok", "weight": 1.0}
    assert status["checks"]["R5b"]["status"] == "no_negatives"
    assert store.scores(BASE_TAG, "calibrated") == []
