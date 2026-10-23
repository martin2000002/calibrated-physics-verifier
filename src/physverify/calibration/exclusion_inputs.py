"""What the exclusion decision reads: the consumed stratum and the two R2 detector runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from physverify.calibration.artifact import EvaluationReport
from physverify.calibration.artifact_io import (
    Calibration,
    CalibrationError,
    Stratum,
    UnusableStratumError,
    open_stratum,
)
from physverify.calibration.exclusion_rules import (
    KEPT,
    ExclusionRules,
)
from physverify.calibration.table import (
    CalibrationRow,
)
from physverify.verification.exclusion_record import (
    EvidenceValue,
    ExclusionInputError,
    RuleEvidence,
)
from physverify.verification.runner import UNKNOWN_ROW_STEP

# --- inputs -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumedStratum:
    """The stratum H2/H3 consume; ``problem`` says why it cannot score (None: usable)."""

    stratum: Stratum
    problem: str | None


def consumed_stratum(
    path: Path,
    calibration: Calibration,
    rules: ExclusionRules,
    *,
    checks_hash: str,
    require_calibrated: bool,
) -> ConsumedStratum:
    """Through the consumers' gate (``open_stratum``); a missing stratum is an input error."""
    level, profile = rules.consumed_level, rules.consumed_profile
    try:
        stratum = open_stratum(
            path,
            level,
            profile,
            checks_hash=checks_hash,
            row_step=UNKNOWN_ROW_STEP,
            require_calibrated=require_calibrated,
        )
    except UnusableStratumError as exc:
        return ConsumedStratum(calibration.stratum(level, profile), str(exc))
    except CalibrationError as exc:
        raise ExclusionInputError(str(exc)) from exc
    return ConsumedStratum(stratum, None)


@dataclass(frozen=True)
class ArmRun:
    """One ``verify --policy raw`` run of the R2 clips; ``detectors`` are the DETECT providers
    its results name, ``report_sha256`` the hash of its checks.jsonl and table.csv,
    ``checks_hash`` that of the check parameters it ran with."""

    run_id: str
    report_sha256: str
    detectors: frozenset[str]
    profile: str
    commit: str
    checks_hash: str
    rows: tuple[CalibrationRow, ...]


@dataclass(frozen=True)
class R2Arms:
    base: ArmRun
    tiny: ArmRun

    def by_label(self) -> dict[str, ArmRun]:
        return {"base": self.base, "tiny": self.tiny}


@dataclass(frozen=True)
class DecisionInputs:
    calibration: Calibration
    consumed: ConsumedStratum
    evaluation: EvaluationReport
    evaluation_sha256: str
    arms: R2Arms
    checks_hash: str


def rule_evidence(
    rule: str, check_id: str, reason: str, inputs: Mapping[str, EvidenceValue], *, decided: bool
) -> RuleEvidence:
    return RuleEvidence(
        rule=rule,
        check_id=check_id,
        excluded=decided and reason != KEPT,
        decided=decided,
        reason=reason,
        inputs=dict(inputs),
    )
