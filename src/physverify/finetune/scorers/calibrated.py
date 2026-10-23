"""Calibrated score: a thin wrapper over the calibration chain's ``score_clip``.

No score formula lives here; the calibration's clip score maps one to one onto
``AggregateScore``. ``k_sigma`` is not passed: ``score_clip`` uses the one the stratum was fitted
with. The chain is reached through its facade (``calibration.api``), loaded on first use. The
stratum is opened through the calibration's one gate (``open_stratum``), the same the
human-agreement study uses, so a missing, stale or unusable artifact is a reason (exit 2) with the
calibration's own message, never a silent fallback to the provisional thresholds. The checks hash
is part of the thresholds stamp, so scores of another check configuration never mix in one set of
pairs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final

from physverify.calibration.api import CalibrationApi, ScoredClip, calibration_api
from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.core.errors import PhysverifyError, UsageError
from physverify.finetune.scorers._base import AggregateScore
from physverify.schema.checks import CheckResult, InputLevel
from physverify.verification.runner import UNKNOWN_ROW_STEP

if TYPE_CHECKING:
    from physverify.calibration.artifact_io import Stratum

HASH_SHOWN: Final = 12
LEVEL: Final = InputLevel.PERCEIVED
SOURCE_PREFIX: Final = "calibrated:"
ROW_STEP_KEY: Final = "row_step"

ApiLoader = Callable[[], "CalibrationApi[Stratum]"]


def to_aggregate(clip_score: ScoredClip) -> AggregateScore:
    decided = frozenset(str(check) for check in clip_score.decided_checks)
    return AggregateScore(
        score=clip_score.score,
        sigma=clip_score.sigma,
        abstained=bool(clip_score.abstained),
        reason=clip_score.reason,
        n_decided=len(decided),
        n_failed=int(clip_score.n_failed),
        coverage=float(clip_score.coverage),
        decided=decided,
        per_check={str(k): float(v) for k, v in clip_score.per_check.items()},
    )


class CalibratedScorer:
    name: ClassVar[str] = "calibrated"
    version: ClassVar[str] = "2"
    calibrated: ClassVar[bool] = True

    def __init__(
        self,
        settings: Settings,
        checks: Mapping[str, type[Check]],
        api_loader: ApiLoader = calibration_api,
    ) -> None:
        self._settings = settings
        self._api_loader = api_loader
        self._checks = checks
        self._api: CalibrationApi[Stratum] | None = None
        self._stratum: Stratum | None = None
        self._label = ""
        self._status_problem: str | None = None

    def unavailable(self) -> str | None:
        try:
            self._load()
        except (PhysverifyError, OSError, ValueError) as exc:
            return f"no usable calibration: {type(exc).__name__}: {exc}"
        return self._status_problem

    def thresholds(self) -> str:
        self._load()
        return self._label

    def calibration_status(self) -> Mapping[str, Any] | None:
        """The stratum's state per check (``scores/<tag>/calibrated/calibration/status.json``)."""
        self._load()
        if self._stratum is None:
            return {"problem": self._status_problem}
        return self._stratum.check_status()

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore:
        api = self._load()
        if self._status_problem is not None or self._stratum is None:
            raise UsageError(self._status_problem or "no calibrated stratum")
        self._require_time_base(api, self._stratum, results)
        return to_aggregate(api.score_clip(results, self._stratum, only=only))

    def _require_time_base(
        self, api: CalibrationApi[Stratum], stratum: Stratum, results: Sequence[CheckResult]
    ) -> None:
        """tau depends on the time base: a clip of another row_step is refused."""
        steps = (r.details.get(ROW_STEP_KEY) for r in results)
        try:
            step = api.unique_row_step(UNKNOWN_ROW_STEP if s is None else int(s) for s in steps)
        except api.error as exc:
            raise UsageError(str(exc)) from exc
        problem = stratum.time_base_problem(self._settings.profile, step)
        if problem is not None:
            raise UsageError(problem)

    def _load(self) -> CalibrationApi[Stratum]:
        if self._api is not None:
            return self._api
        api = self._api_loader()
        path = api.default_artifact_path(self._settings)
        active = api.active_checks_hash(self._settings, self._checks)
        require = api.require_calibrated(self._settings)
        try:
            self._stratum = api.open_stratum(
                path,
                LEVEL,
                self._settings.profile,
                checks_hash=active,
                row_step=UNKNOWN_ROW_STEP,
                require_calibrated=require,
            )
        except api.error as exc:
            self._status_problem = str(exc)
        source = "unusable" if self._stratum is None else self._stratum.source
        self._label = f"{SOURCE_PREFIX}{source}@checks:{active[:HASH_SHOWN]}"
        self._api = api
        return api


PLUGINS = (CalibratedScorer,)
