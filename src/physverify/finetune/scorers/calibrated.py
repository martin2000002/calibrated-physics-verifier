"""Calibrated score: a thin wrapper over A8's ``calibration.apply.score_clip`` (ADR 0009, 7).

No score formula lives here; the ``ClipScore`` of A8 maps one to one onto ``AggregateScore``.
``k_sigma`` is not passed: ``score_clip`` uses the one the stratum was fitted with.
A8's modules are imported lazily, so a missing module or artifact, an artifact that is not
``calibrated``, or one fitted with other check parameters than the active registry is a reason
(exit 2), never a silent fallback to the provisional thresholds. The checks hash is part of the
thresholds stamp, so scores of another check configuration never mix in one set of pairs.
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
from physverify.core.registry import discover
from physverify.finetune.scorers._base import AggregateScore
from physverify.schema.checks import CheckResult, InputLevel
from physverify.verification.runner import CHECKS_SECTION, ChecksSettings

APPLY_MODULE: Final = "physverify.calibration.apply"
LOADER_MODULE: Final = "physverify.calibration.loader"
SETTINGS_MODULE: Final = "physverify.calibration.settings"
CHECKS_PACKAGE: Final = "physverify.checks"
HASH_SHOWN: Final = 12
LEVEL: Final = InputLevel.PERCEIVED
CALIBRATED_STATUS: Final = "calibrated"
SOURCE_PREFIX: Final = "calibrated:"

ModuleImporter = Callable[[str], ModuleType]
CheckRegistry = Callable[[], Mapping[str, type[Check]]]


def check_registry() -> Mapping[str, type[Check]]:
    return discover(CHECKS_PACKAGE, (Check,), "check_id")


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
    version: ClassVar[str] = "1"
    calibrated: ClassVar[bool] = True

    def __init__(
        self,
        settings: Settings,
        import_module: ModuleImporter = importlib.import_module,
        checks: CheckRegistry = check_registry,
    ) -> None:
        self._settings = settings
        self._import = import_module
        self._checks = checks
        self._stratum: Any = None
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

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore:
        apply = self._load()
        if self._status_problem is not None:
            raise UsageError(self._status_problem)
        return to_aggregate(apply.score_clip(results, self._stratum, only=only))

    def _load(self) -> ModuleType:
        if self._apply is not None:
            return self._apply
        apply = self._import(APPLY_MODULE)
        loader = self._import(LOADER_MODULE)
        path: Path = loader.default_artifact_path(self._settings)
        calibration = loader.load_calibration(path)
        self._stratum = calibration.stratum(LEVEL, self._settings.profile)
        active = self._active_hash()
        fitted = self._stratum.checks_hash
        if calibration.status != CALIBRATED_STATUS:
            self._status_problem = (
                f"{path} has status {calibration.status!r}, not {CALIBRATED_STATUS!r}: "
                "its thresholds do not meet A8's acceptance; refit before scoring"
            )
        elif fitted != active:
            shown = fitted[:HASH_SHOWN] if fitted else None
            self._status_problem = (
                f"{path} was fitted with other check parameters ({shown} != "
                f"{active[:HASH_SHOWN]}): refit the calibration"
            )
        self._label = f"{SOURCE_PREFIX}{self._stratum.source}@checks:{active[:HASH_SHOWN]}"
        self._apply = apply
        return apply

    def _active_hash(self) -> str:
        overrides = self._settings.section(CHECKS_SECTION, ChecksSettings)
        digest: str = self._import(SETTINGS_MODULE).checks_hash(self._checks(), overrides)
        return digest


PLUGINS = (CalibratedScorer,)
