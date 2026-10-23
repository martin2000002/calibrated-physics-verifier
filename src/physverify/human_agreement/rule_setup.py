"""The agreement rule from the config and the calibration artifact, and the verify run it
must fit."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from physverify.checks._base import Check
from physverify.core.errors import ConfigError, UsageError
from physverify.datasets._videophy_tables import VideophyRow
from physverify.human_agreement.clip_decision import (
    CALIBRATED_NAME,
    CalibrationApi,
    CalibrationStratum,
    ClipDecisionRule,
    ProvisionalAnyFail,
    StratumRequest,
    UnusableCalibrationError,
    calibrated_rule,
)
from physverify.human_agreement.comparison import (
    RuleScope,
)
from physverify.human_agreement.settings import (
    AgreementSettings,
)
from physverify.human_agreement.verify_run_reader import (
    VerifyRun,
    run_row_step,
)
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.verification.runner import (
    VERIFICATION_SECTION,
    VerificationSettings,
    load_thresholds,
    thresholds_label,
)
from physverify.verification.scope import CheckScope, check_scope

if TYPE_CHECKING:
    from physverify.config.models import Settings

type ApiLoader[S: CalibrationStratum] = Callable[[], CalibrationApi[S]]


@dataclass(frozen=True)
class RuleSetup:
    """``thresholds`` is the label verify must have used (provisional rule, also as the
    fallback); ``gate`` why the calibrated stratum was not usable, and ``calibration`` its state
    per check when it was opened."""

    rule: ClipDecisionRule
    artifact: Path | None
    thresholds: str | None
    scope: RuleScope = RuleScope()
    gate: str | None = None
    calibration: dict[str, object] | None = None
    checks: CheckScope = CheckScope()
    """The ``[check_scope]`` the rule excludes with, its decision record included."""

    @property
    def primary(self) -> ClipDecisionRule:
        return self.rule.excluding(self.scope.primary_excluded)


def rule_scope(
    settings: Settings, checks: Mapping[str, type[Check]], agreement: AgreementSettings
) -> tuple[CheckScope, RuleScope]:
    """``[check_scope]`` for the human-agreement study: the excluded checks and the time-dependent
    ones."""
    scope = check_scope(settings, checks)
    timed = frozenset(scope.time_dependent_checks)
    primary = timed if agreement.exclude_time_dependent else frozenset()
    return scope, RuleScope(timed, primary)


def build_rule[S: CalibrationStratum](
    settings: Settings,
    registry: Mapping[str, type[Check]],
    agreement: AgreementSettings,
    run: VerifyRun,
    api_loader: ApiLoader[S],
) -> RuleSetup:
    checks, scope = rule_scope(settings, registry, agreement)
    excluded = frozenset(checks.exclude_checks)
    if agreement.rule != CALIBRATED_NAME:
        return _provisional(settings, checks, scope, None)
    api = api_loader()
    artifact = api.default_artifact_path(settings)
    request = StratumRequest(
        artifact=artifact,
        profile=run.profile,
        row_step=run_row_step(run),
        checks_hash=api.active_checks_hash(settings, registry),
        require_calibrated=api.require_calibrated(settings),
    )
    try:
        rule = calibrated_rule(api, request, excluded)
    except UnusableCalibrationError as exc:
        if agreement.on_unusable_stratum == "error":
            raise ConfigError(f"calibration {artifact}: {exc}") from exc
        return _provisional(settings, checks, scope, str(exc), artifact)
    return RuleSetup(rule, artifact, None, scope, calibration=dict(rule.status), checks=checks)


def _provisional(
    settings: Settings,
    checks: CheckScope,
    scope: RuleScope,
    gate: str | None,
    artifact: Path | None = None,
) -> RuleSetup:
    """The provisional rule, also the fallback of an unusable stratum (descriptive study), which
    keeps the artifact so the freeze binds the one whose stratum was found unusable."""
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    thresholds = load_thresholds(settings.paths.thresholds, verification.thresholds)
    rule = ProvisionalAnyFail(thresholds, frozenset(checks.exclude_checks))
    return RuleSetup(rule, artifact, thresholds_label(thresholds), scope, gate, checks=checks)


def check_run(
    run: VerifyRun, rows: Sequence[VideophyRow], setup: RuleSetup, split: Split | None
) -> None:
    """The run must be a perceived verify run over this subset (of ``split`` when given), with
    the rule's thresholds."""
    levels = {r.input_level for clip in run.results.values() for r in clip}
    if levels - {InputLevel.PERCEIVED}:
        raise UsageError(f"{run.path} is not a perceived run (levels {sorted(levels)})")
    known = {row.clip_id: row for row in rows}
    foreign = sorted(set(run.rows) - set(known))
    if foreign:
        raise UsageError(f"{run.path} has {len(foreign)} clips outside the subset: {foreign[:3]}")
    wrong = sorted(c for c in run.rows if split is not None and known[c].split is not split)
    if wrong:
        raise UsageError(f"{len(wrong)} clips of the run are not {split}: {wrong[:3]}")
    labels = run.thresholds_labels()
    if setup.thresholds is not None and labels and labels != {setup.thresholds}:
        raise ConfigError(f"verify used thresholds {sorted(labels)}, the rule {setup.thresholds}")
