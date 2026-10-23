"""Typed access to ``calibration.json``: the only reader of the artifact."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from physverify.calibration.artifact import (
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    OK,
    CalibrationArtifact,
    CheckCalibration,
)
from physverify.calibration.artifact import Stratum as StratumModel
from physverify.calibration.settings import calibration_settings
from physverify.config.models import Settings
from physverify.core.digests import text_sha256
from physverify.core.errors import PhysverifyError
from physverify.core.models import LENIENT, SCHEMA_KEY
from physverify.schema.checks import InputLevel
from physverify.verification.runner import (
    CALIBRATED,
    UNKNOWN_ROW_STEP,
    ArtifactRef,
    ThresholdEntry,
    ThresholdTable,
)


class CalibrationError(PhysverifyError):
    """Missing, unreadable or incompatible calibration artifact."""


class StaleCalibrationError(CalibrationError):
    """The artifact was fitted with other check parameters than the active registry."""


class UnusableStratumError(CalibrationError):
    """The stratum loads and matches the run but cannot score yet: a state of the fit (readiness,
    or a non-calibrated artifact when one is required), not a configuration error."""


class MixedRowStepError(CalibrationError):
    """Rows of one run were extracted with different time bases."""


class KSigmaMismatchError(CalibrationError):
    """A k_sigma other than the one the stratum was calibrated with."""


def default_artifact_path(settings: Settings) -> Path:
    """``paths.calibration_dir`` / ``[calibration].artifact``."""
    return settings.paths.calibration_dir / calibration_settings(settings).artifact


def projection_name(level: InputLevel, profile: str) -> str:
    return f"thresholds.{level.value}.{profile}.json"


@dataclass(frozen=True)
class Stratum:
    """One (input level, sampling profile) of the artifact. ``checks_hash`` is the hash of the
    check parameters it was fitted with (``verification.check_set.checks_hash`` of the registry):
    a consumer stamps it to tie a score to this calibration."""

    model: StratumModel
    source: str
    artifact: ArtifactRef | None
    checks_hash: str | None = None
    artifact_status: str | None = None

    @property
    def level(self) -> InputLevel:
        return self.model.level

    @property
    def profile(self) -> str:
        return self.model.profile

    @property
    def on_unknown_sigma(self) -> str:
        return self.model.on_unknown_sigma

    def check(self, check_id: str) -> CheckCalibration | None:
        return self.model.checks.get(check_id)

    def _ok(self, check_id: str) -> CheckCalibration | None:
        check = self.check(check_id)
        return check if check is not None and check.status == OK else None

    def tau(self, check_id: str) -> float | None:
        check = self._ok(check_id)
        return None if check is None else check.tau

    def weight(self, check_id: str) -> float:
        check = self._ok(check_id)
        return 0.0 if check is None else check.weight.value

    def max_weight(self) -> float:
        return max((self.weight(c) for c in self.model.checks), default=0.0)

    def sigma_correction(self, check_id: str) -> tuple[float, float]:
        """(bias, factor); factor NaN when the error was not measured."""
        check = self.check(check_id)
        if check is None or check.sigma.factor is None:
            return (0.0 if check is None else check.sigma.bias), math.nan
        return check.sigma.bias, check.sigma.factor

    def entry(self, check_id: str) -> ThresholdEntry | None:
        check = self._ok(check_id)
        if check is None or check.tau is None:
            return None
        return ThresholdEntry(tau=check.tau, bias=check.sigma.bias, sigma_factor=check.sigma.factor)

    def time_base_problem(self, profile: str, row_step: int) -> str | None:
        """Why this stratum does not apply to a run of ``profile`` and ``row_step`` (tau and the
        magnitudes depend on the time base); None when it does. A row_step of
        ``UNKNOWN_ROW_STEP`` on either side (no time axis) is not compared."""
        if self.profile != profile:
            return f"{self.source} was fitted on profile {self.profile!r}, the run uses {profile!r}"
        fitted = self.model.row_step
        if UNKNOWN_ROW_STEP not in (fitted, row_step) and fitted != row_step:
            return f"{self.source} was fitted with row_step {fitted}, the run has {row_step}"
        return None

    def readiness_problem(self) -> str | None:
        """Why this stratum cannot score a clip; None when it can. Checks that are not ``ok``
        have no tau and do not contribute, so one usable check suffices."""
        if not self.model.aggregate.sufficient:
            return f"{self.source}: the aggregate threshold s* is insufficient"
        if self.max_weight() <= 0.0:
            return f"{self.source}: no check is ok with a weight > 0"
        return None

    def check_status(self) -> dict[str, Any]:
        """The stratum's state for reports: its label, hashes and each check's fit status."""
        return {
            "source": self.source,
            "artifact_status": self.artifact_status,
            "checks_hash": self.checks_hash,
            "aggregate_sufficient": self.model.aggregate.sufficient,
            "checks": {
                c: {"status": self.model.checks[c].status, "weight": self.weight(c)}
                for c in sorted(self.model.checks)
            },
        }

    def score_threshold(self) -> float | None:
        """Flag a clip when 1 - S exceeds it."""
        return self.model.aggregate.score_threshold

    def thresholds_projection(self, root: Path | None = None) -> dict[str, Any]:
        """``ThresholdTable`` of the ``ok`` checks; the others stay UNSCORED under the policy.
        The artifact path is written relative to ``root`` (``paths.calibration_dir``) when it
        lies under it, so the table does not pin one machine's absolute path."""
        entries = {c: e for c in sorted(self.model.checks) if (e := self.entry(c)) is not None}
        table = ThresholdTable(
            status=CALIBRATED, source=self.source, checks=entries, artifact=self._ref(root)
        )
        return table.model_dump(mode="json")

    def _ref(self, root: Path | None) -> ArtifactRef | None:
        if self.artifact is None or root is None:
            return self.artifact
        path = Path(self.artifact.path)
        if not path.resolve().is_relative_to(root.resolve()):
            return self.artifact
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        return self.artifact.model_copy(update={"path": relative})


@dataclass(frozen=True)
class Calibration:
    artifact: CalibrationArtifact
    path: Path | None
    sha256: str

    @property
    def version(self) -> str:
        return self.artifact.version

    @property
    def status(self) -> str:
        return self.artifact.status

    @property
    def commit(self) -> str:
        return self.artifact.commit

    @property
    def run_id(self) -> str:
        return self.artifact.run_id

    def strata(self) -> tuple[tuple[InputLevel, str], ...]:
        return tuple((s.level, s.profile) for s in self.artifact.strata)

    def stratum(self, level: InputLevel, profile: str) -> Stratum:
        for model in self.artifact.strata:
            if model.level is level and model.profile == profile:
                return self._wrap(model)
        known = [f"{lv.value}/{p}" for lv, p in self.strata()]
        raise CalibrationError(f"no stratum {level.value}/{profile} in the artifact; has {known}")

    def _wrap(self, model: StratumModel) -> Stratum:
        name = self.path.name if self.path is not None else "calibration"
        source = f"{name}#{model.level.value}/{model.profile}@{self.run_id}"
        ref = None if self.path is None else ArtifactRef(path=str(self.path), sha256=self.sha256)
        return Stratum(
            model=model,
            source=source,
            artifact=ref,
            checks_hash=self.artifact.checks_hash,
            artifact_status=self.artifact.status,
        )


def load_calibration(path: Path, *, checks_hash: str | None = None) -> Calibration:
    """The artifact at ``path``; ``checks_hash`` of the active registry rejects a stale one."""
    if not path.is_file():
        raise CalibrationError(f"no calibration artifact at {path}: run `calibrate fit`")
    calibration = parse_calibration(path.read_text(encoding="utf-8"), path=path)
    if checks_hash is not None and calibration.artifact.checks_hash != checks_hash:
        raise StaleCalibrationError(
            f"{path} was fitted with other check parameters "
            f"({calibration.artifact.checks_hash[:12]} != {checks_hash[:12]}): refit"
        )
    return calibration


def open_stratum(
    path: Path,
    level: InputLevel,
    profile: str,
    *,
    checks_hash: str,
    row_step: int,
    require_calibrated: bool = False,
) -> Stratum:
    """The one gate a consumer (human-agreement study, preference fine-tuning, generator
    comparison, publish) applies before scoring with a stratum.

    A missing or stale artifact, a missing stratum or another time base raise
    ``CalibrationError`` (a configuration error). A stratum that matches but cannot score (its
    aggregate is insufficient, no check is ok with a weight > 0, or the artifact is not
    ``calibrated`` while ``require_calibrated``) raises ``UnusableStratumError``. Only this
    stratum is required by default: the artifact is ``calibrated`` only when every check of
    every stratum is ok."""
    calibration = load_calibration(path, checks_hash=checks_hash)
    stratum = calibration.stratum(level, profile)
    mismatch = stratum.time_base_problem(profile, row_step)
    if mismatch is not None:
        raise CalibrationError(mismatch)
    problem = stratum.readiness_problem()
    if problem is None and require_calibrated and calibration.status != CALIBRATED:
        problem = f"{path.name} has status {calibration.status!r}, not {CALIBRATED!r}"
    if problem is not None:
        raise UnusableStratumError(problem)
    return stratum


def unique_row_step(steps: Iterable[int]) -> int:
    """The single time base of a run; ``UNKNOWN_ROW_STEP`` entries (no time axis) are ignored,
    and a run with none of them known is ``UNKNOWN_ROW_STEP``."""
    known = sorted({s for s in steps if s != UNKNOWN_ROW_STEP})
    if len(known) > 1:
        raise MixedRowStepError(f"rows with different row_step {known}: one time base per run")
    return known[0] if known else UNKNOWN_ROW_STEP


def parse_calibration(text: str, *, path: Path | None = None) -> Calibration:
    label = str(path) if path is not None else "calibration"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get(SCHEMA_KEY) != CALIBRATION_SCHEMA:
        raise CalibrationError(f"{label} is not a {CALIBRATION_SCHEMA} artifact")
    lenient = _compatible(str(data.get("version", "")), label)
    try:
        artifact = CalibrationArtifact.model_validate(data, context={LENIENT: lenient})
    except ValidationError as exc:
        raise CalibrationError(f"{label} is invalid:\n{exc}") from exc
    return Calibration(artifact=artifact, path=path, sha256=text_sha256(text))


def _compatible(version: str, label: str) -> bool:
    """Same major; True when the minor is newer (unknown optional fields are dropped)."""
    ours, theirs = _semver(CALIBRATION_VERSION) or (0, 0, 0), _semver(version)
    if theirs is None or theirs[0] != ours[0]:
        raise CalibrationError(
            f"{label} has version {version!r}; this code reads {ours[0]}.x ({CALIBRATION_VERSION})"
        )
    return theirs[1] > ours[1]


def _semver(version: str) -> tuple[int, int, int] | None:
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])
