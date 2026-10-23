"""Calibrated score: a thin wrapper over A8's ``calibration.apply.score_clip`` (ADR 0009, 7).

No score formula lives here; the ``ClipScore`` of A8 maps one to one onto ``AggregateScore``.
``k_sigma`` is not passed: ``score_clip`` uses the one the stratum was fitted with.
A8's modules are imported lazily. The stratum is opened through A8's one gate
(``loader.open_stratum``), the same H2 uses, so a missing, stale or unusable artifact is a reason
(exit 2) with A8's own message, never a silent fallback to the provisional thresholds. The
checks hash is part of the thresholds stamp, so scores of another check configuration never mix
in one set of pairs.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Final

from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.core.errors import PhysverifyError, UsageError
from physverify.finetune.scorers._base import AggregateScore
from physverify.schema.checks import CheckResult, InputLevel
from physverify.verification.runner import UNKNOWN_ROW_STEP

APPLY_MODULE: Final = "physverify.calibration.apply"
LOADER_MODULE: Final = "physverify.calibration.loader"
SETTINGS_MODULE: Final = "physverify.calibration.settings"
CHECK_SET_MODULE: Final = "physverify.verification.check_set"
HASH_SHOWN: Final = 12
LEVEL: Final = InputLevel.PERCEIVED
SOURCE_PREFIX: Final = "calibrated:"
ROW_STEP_KEY: Final = "row_step"

ModuleImporter = Callable[[str], ModuleType]


def to_aggregate(clip_score: Any) -> AggregateScore:
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
        import_module: ModuleImporter = importlib.import_module,
    ) -> None:
        self._settings = settings
        self._import = import_module
        self._checks = checks
        self._stratum: Any = None
        self._loader: Any = None
        self._label = ""
        self._apply: ModuleType | None = None
        self._status_problem: str | None = None

    def unavailable(self) -> str | None:
        try:
            self._load()
        except ImportError as exc:
            return f"{APPLY_MODULE} is not available ({exc}); run A8's calibration first"
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
        status: Mapping[str, Any] = self._stratum.check_status()
        return status

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore:
        apply = self._load()
        if self._status_problem is not None:
            raise UsageError(self._status_problem)
        self._require_time_base(results)
        return to_aggregate(apply.score_clip(results, self._stratum, only=only))

    def _require_time_base(self, results: Sequence[CheckResult]) -> None:
        """tau depends on the time base (ADR 0006 R9): a clip of another row_step is refused."""
        steps = (r.details.get(ROW_STEP_KEY) for r in results)
        try:
            step = self._loader.unique_row_step(UNKNOWN_ROW_STEP if s is None else s for s in steps)
        except self._loader.CalibrationError as exc:
            raise UsageError(str(exc)) from exc
        problem = self._stratum.time_base_problem(self._settings.profile, step)
        if problem is not None:
            raise UsageError(problem)

    def _load(self) -> ModuleType:
        if self._apply is not None:
            return self._apply
        apply = self._import(APPLY_MODULE)
        loader = self._import(LOADER_MODULE)
        calibration = self._import(SETTINGS_MODULE)
        check_set = self._import(CHECK_SET_MODULE)
        path: Path = loader.default_artifact_path(self._settings)
        active: str = check_set.active_checks_hash(self._settings, self._checks)
        require = calibration.calibration_settings(self._settings).require_calibrated_artifact
        try:
            self._stratum = loader.open_stratum(
                path,
                LEVEL,
                self._settings.profile,
                checks_hash=active,
                row_step=UNKNOWN_ROW_STEP,
                require_calibrated=require,
            )
        except loader.CalibrationError as exc:
            self._status_problem = str(exc)
        source = "unusable" if self._stratum is None else self._stratum.source
        self._label = f"{SOURCE_PREFIX}{source}@checks:{active[:HASH_SHOWN]}"
        self._loader, self._apply = loader, apply
        return apply


PLUGINS = (CalibratedScorer,)
