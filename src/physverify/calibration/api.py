"""The calibration chain's entry points for its consumers (the human-agreement rule and the
calibrated fine-tuning score), behind one typed facade so a test can pass a fake.

``calibration_api`` imports the chain only when a consumer asks for it."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from physverify.schema.checks import CheckResult, InputLevel

if TYPE_CHECKING:
    from physverify.calibration.artifact_io import Stratum
    from physverify.checks._base import Check
    from physverify.config.models import Settings


class CalibrationStratum(Protocol):
    """What the consumers read from a stratum (per-check tau, aggregate cut, time base and the
    artefact#level/profile@run label)."""

    @property
    def source(self) -> str: ...

    def tau(self, check_id: str) -> float | None: ...

    def score_threshold(self) -> float | None: ...

    def check_status(self) -> dict[str, Any]: ...

    def time_base_problem(self, profile: str, row_step: int) -> str | None: ...


class ScoredClip(Protocol):
    """The chain's clip score: S in [0, 1] or None when it abstains."""

    @property
    def score(self) -> float | None: ...

    @property
    def sigma(self) -> float | None: ...

    @property
    def coverage(self) -> float: ...

    @property
    def abstained(self) -> bool: ...

    @property
    def reason(self) -> str | None: ...

    @property
    def n_failed(self) -> int: ...

    @property
    def decided_checks(self) -> Sequence[str]: ...

    @property
    def abstained_checks(self) -> Sequence[str]: ...

    @property
    def per_check(self) -> Mapping[str, float]: ...


class ScoreClip[S: CalibrationStratum](Protocol):
    def __call__(
        self,
        results: Sequence[CheckResult],
        stratum: S,
        k_sigma: float | None = None,
        *,
        only: Collection[str] | None = None,
    ) -> ScoredClip: ...


class OpenStratum[S: CalibrationStratum](Protocol):
    """The chain's ``open_stratum``: the one gate in front of a stratum."""

    def __call__(
        self,
        path: Path,
        level: InputLevel,
        profile: str,
        *,
        checks_hash: str,
        row_step: int,
        require_calibrated: bool = False,
    ) -> S: ...


@dataclass(frozen=True)
class CalibrationApi[S: CalibrationStratum]:
    """``error`` is a configuration problem (missing or stale artifact, another stratum or time
    base); ``unusable_error``, a subclass of it, a stratum that cannot score yet."""

    open_stratum: OpenStratum[S]
    default_artifact_path: Callable[[Settings], Path]
    active_checks_hash: Callable[[Settings, Mapping[str, type[Check]]], str]
    require_calibrated: Callable[[Settings], bool]
    score_clip: ScoreClip[S]
    unique_row_step: Callable[[Iterable[int]], int]
    error: type[Exception]
    unusable_error: type[Exception]


def calibration_api() -> CalibrationApi[Stratum]:
    from physverify.calibration.apply import score_clip
    from physverify.calibration.artifact_io import (
        CalibrationError,
        UnusableStratumError,
        default_artifact_path,
        open_stratum,
        unique_row_step,
    )
    from physverify.calibration.settings import calibration_settings
    from physverify.verification.check_set import active_checks_hash

    return CalibrationApi(
        open_stratum=open_stratum,
        default_artifact_path=default_artifact_path,
        active_checks_hash=active_checks_hash,
        require_calibrated=lambda s: calibration_settings(s).require_calibrated_artifact,
        score_clip=score_clip,
        unique_row_step=unique_row_step,
        error=CalibrationError,
        unusable_error=UnusableStratumError,
    )
