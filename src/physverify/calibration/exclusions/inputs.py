"""What the exclusion decision reads: the consumed stratum and the two R2 detector runs."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from physverify.calibration.artifact import EvaluationReport
from physverify.calibration.artifact_io import (
    Calibration,
    CalibrationError,
    Stratum,
    UnusableStratumError,
    open_stratum,
)
from physverify.calibration.collect import read_run, run_level_of
from physverify.calibration.exclusions.rules import (
    KEPT,
    ExclusionRules,
    R2Rule,
)
from physverify.calibration.table import (
    CalibrationDataError,
    CalibrationRow,
    build_rows,
    in_family,
)
from physverify.core.digests import text_sha256
from physverify.core.errors import UsageError
from physverify.runtime.runs import RUN_FILE
from physverify.schema.clip import ClipManifest
from physverify.schema.io import stable_hash
from physverify.schema.stages import Stage
from physverify.synth.perturbations.none import NAME as NONE
from physverify.synth.scene_scope import scene_of
from physverify.verification.exclusion_record import (
    EvidenceValue,
    ExclusionInputError,
    ExclusionRecord,
    RuleEvidence,
    parse_record,
)
from physverify.verification.runner import (
    CHECKS_FILE,
    CHECKS_HASH_FILE,
    TABLE_FILE,
    UNKNOWN_ROW_STEP,
    VerificationReport,
)
from physverify.verification.sweep import CheckUnit, perturbation_name


@dataclass(frozen=True)
class ConsumedStratum:
    """The stratum the human-agreement study and the generator comparison consume; ``problem`` says
    why it cannot score (None: usable)."""

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


def evaluation_report(path: Path) -> tuple[EvaluationReport, str]:
    if not path.is_file():
        raise UsageError(f"no evaluation at {path}: run `physverify calibrate evaluate`")
    text = path.read_text(encoding="utf-8")
    try:
        return EvaluationReport.model_validate_json(text), text_sha256(text)
    except ValidationError as exc:
        raise UsageError(f"{path} is not an evaluation report:\n{exc}") from exc


def r2_clip_ids(clips: Sequence[ClipManifest], cfg: R2Rule) -> list[str]:
    """The clean and ``cfg.perturbation`` clips of ``cfg.family`` on ``cfg.split``."""
    return sorted(
        clip.clip_id
        for clip in clips
        if clip.split is cfg.split
        and in_family(scene_of(clip), cfg.family)
        and perturbation_name(clip) in (NONE, cfg.perturbation)
    )


def report_sha256(run_dir: Path) -> str:
    """Hash of the run's checks.jsonl and table.csv, the evidence the decision read."""
    return stable_hash(
        {
            name: text_sha256((run_dir / name).read_text(encoding="utf-8"))
            for name in (CHECKS_FILE, TABLE_FILE)
        }
    )


def detectors_of(report: VerificationReport) -> frozenset[str]:
    """The DETECT providers the run's results name (empty: oracle input)."""
    return frozenset(
        r.provenance[Stage.DETECT.value]
        for r in report.results
        if Stage.DETECT.value in r.provenance
    )


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    profile: str
    commit: str
    checks_hash: str


def run_meta(run_dir: Path) -> RunMeta:
    """``run.json`` plus the ``checks_hash`` that ``verify`` writes (empty: an older run)."""
    path = run_dir / RUN_FILE
    if not path.is_file():
        raise UsageError(f"{run_dir} is not a run: no {RUN_FILE}")
    record = json.loads(path.read_text(encoding="utf-8"))
    hashed = run_dir / CHECKS_HASH_FILE
    checks = hashed.read_text(encoding="utf-8").strip() if hashed.is_file() else ""
    return RunMeta(
        run_dir.name, str(record.get("profile", "")), str(record.get("commit", "")), checks
    )


def arm_run(
    run_dir: Path,
    clips: Sequence[ClipManifest],
    units: Sequence[CheckUnit],
    couplings: Collection[tuple[str, str]],
    rules: ExclusionRules,
) -> ArmRun:
    """One R2 arm read back from its ``verify --policy raw`` run at the consumed level."""
    try:
        report = read_run(run_dir, clips)
        level = run_level_of(report)
    except CalibrationDataError as exc:
        raise ExclusionInputError(str(exc)) from exc
    if level is not rules.consumed_level:
        raise ExclusionInputError(
            f"{run_dir.name} is a {level.value} run, not {rules.consumed_level.value}"
        )
    meta = run_meta(run_dir)
    return ArmRun(
        run_id=meta.run_id,
        report_sha256=report_sha256(run_dir),
        detectors=detectors_of(report),
        profile=meta.profile,
        commit=meta.commit,
        checks_hash=meta.checks_hash,
        rows=build_rows(report, units, couplings, level),
    )


def replacement(
    existing: str | None, fresh: ExclusionRecord, *, replace: bool, label: str
) -> ExclusionRecord | None:
    """``fresh`` to write (bound to the record it replaces), or None when it is unchanged. A
    different record is replaced only with ``replace``; an unreadable one is replaced too."""
    if existing is None:
        return fresh
    try:
        old: ExclusionRecord | None = parse_record(existing, label)
    except ExclusionInputError:
        old = None
    if old is not None and old.model_copy(update={"replaces_sha256": None}).sealed() == fresh:
        return None
    if not replace:
        raise UsageError(f"{label} holds another decision: pass --replace to supersede it")
    replaced = None if old is None else old.sha256
    return fresh.model_copy(update={"replaces_sha256": replaced}).sealed()
