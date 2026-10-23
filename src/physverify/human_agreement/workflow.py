"""``agreement freeze`` and ``agreement compare`` as functions of injected settings, dataset and
calibration API (ADR 0011 D10): the CLI only parses options and prints."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from physverify.checks._base import Check
from physverify.core.digests import sha256_if_file
from physverify.core.errors import ConfigError, UsageError
from physverify.datasets._queries import derive_queries
from physverify.datasets._videophy_review import REVIEWED, SubsetReview, load_review, template
from physverify.datasets._videophy_tables import SECTION as VIDEOPHY_SECTION
from physverify.datasets._videophy_tables import VideophyRow
from physverify.human_agreement.analysis import SECONDARY_TESTS
from physverify.human_agreement.coverage import dataset_gaps
from physverify.human_agreement.decision import (
    CALIBRATED_NAME,
    CalibrationApi,
    CalibrationStratum,
    ClipDecisionRule,
    ProvisionalAnyFail,
    StratumRequest,
    UnusableCalibrationError,
    calibrated_rule,
)
from physverify.human_agreement.pipeline import (
    Comparison,
    ComparisonInputs,
    RuleScope,
    compare,
    freeze_dev,
)
from physverify.human_agreement.prereg import (
    FreezeAudit,
    LoggedComparison,
    Preregistration,
    compared_under_other,
    earliest_mtime,
    frozen_after,
    frozen_before,
    load_preregistration,
    log_comparison,
    logged_comparisons,
    preregistration_chain,
    require_unchanged,
)
from physverify.human_agreement.run_io import VerifyRun, load_run, run_files, run_row_step
from physverify.human_agreement.settings import (
    SECTION,
    AgreementSettings,
    agreement_settings,
    preregistration_path,
)
from physverify.runtime.runs import current_commit
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.schema.io import stable_hash
from physverify.verification.runner import (
    VERIFICATION_SECTION,
    VerificationSettings,
    load_thresholds,
    thresholds_label,
)
from physverify.verification.scope import SECTION as SCOPE_SECTION
from physverify.verification.scope import CheckScope, check_scope, require_record_for

if TYPE_CHECKING:
    from physverify.config.models import Settings
    from physverify.datasets.videophy import VideophyDataset

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
    """``[check_scope]`` for H2: the excluded checks and the time-dependent ones."""
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
    """The provisional rule, also the fallback of an unusable stratum (descriptive H2), which
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


def frozen_state(
    settings: Settings, dataset: VideophyDataset, setup: RuleSetup, profile: str
) -> dict[str, object]:
    """Current values of the pre-registered fields (``prereg.FROZEN_FIELDS``)."""
    review = load_review(dataset.subset_path)
    subsets = {} if review is None else {p: r.sha256 for p, r in review.subsets.items()}
    sections = (SECTION, VIDEOPHY_SECTION, SCOPE_SECTION)
    config = {name: settings.sections.get(name) for name in sections}
    return {
        "commit": current_commit(settings.repo_root),
        "agreement_config_sha256": stable_hash(config),
        "subset_sha256": subsets,
        "queries_sha256": sha256_if_file(dataset.queries_path),
        "calibration_artifact_sha256": None
        if setup.artifact is None
        else sha256_if_file(setup.artifact),
        "rule": setup.rule.name,
        "rule_label": setup.rule.label,
        "profile": profile,
        "calibration_gate": setup.gate,
        "exclusion_record_sha256": setup.checks.record_sha256,
        "secondary_tests": list(SECONDARY_TESTS),
    }


def freeze[S: CalibrationStratum](
    settings: Settings,
    checks: Mapping[str, type[Check]],
    dataset: VideophyDataset,
    run_dir: Path,
    api_loader: ApiLoader[S],
    now: dt.datetime,
) -> tuple[Path, Preregistration]:
    agreement = agreement_settings(settings)
    run = load_run(run_dir)
    rows = dataset.rows(agreement.dev_subset)
    setup = build_rule(settings, checks, agreement, run, api_loader)
    require_record_for(
        setup.checks,
        calibrated=agreement.rule == CALIBRATED_NAME,
        artifact_sha256=None if setup.artifact is None else sha256_if_file(setup.artifact),
    )
    check_run(run, rows, setup, Split.CALIB)
    dev = freeze_dev(rows, setup.primary, run, agreement.primary_policy)
    if dev.n == 0:
        raise UsageError(f"{run_dir} has no labelled clip of {agreement.dev_subset}")
    state = frozen_state(settings, dataset, setup, run.profile)
    record = Preregistration.model_validate(
        {
            **state,
            "created_utc": now.isoformat(),
            "device": run.device,
            "dev_run": str(run_dir),
            "dev_subset": agreement.dev_subset,
            "test_subset": agreement.test_subset,
            "n_dev": dev.n,
            "dev_majority": dev.majority,
            "dev_cut": dev.cut,
            "dev_tuned_cut": dev.tuned_cut,
            "kubric_threshold": setup.rule.kubric_threshold,
            "dev_icc": _finite(dev.design, "icc"),
            "dev_design_effect": _finite(dev.design, "deff"),
        }
    )
    return preregistration_path(settings, agreement), record


def _finite(design: Mapping[str, float | int] | None, key: str) -> float | None:
    value = None if design is None else float(design[key])
    return value if value is not None and math.isfinite(value) else None


@dataclass(frozen=True)
class CompareRequest:
    run_dir: Path
    subset: str | None
    rule: str | None
    preregistration: Path | None


def _frozen(
    path: Path, required: bool, state: dict[str, object]
) -> tuple[Preregistration | None, bool]:
    """The pre-registration and whether this comparison is pre-registered: any test clip
    requires it unchanged; dev subsets are exploratory and use it only when present."""
    if required:
        frozen = load_preregistration(path)
        require_unchanged(frozen, state)
        return frozen, True
    return (load_preregistration(path) if path.is_file() else None), False


def freeze_audit(
    path: Path,
    frozen: Preregistration,
    run: VerifyRun,
    commit: str,
    logged: Sequence[LoggedComparison],
) -> FreezeAudit:
    """``earliest_eval_utc`` is the oldest file time of the compared verify run. A logged test
    comparison under another pre-registration makes it post hoc even if the flag was edited."""
    chain = preregistration_chain(path, frozen)
    first = chain[-1].created_utc
    earliest = earliest_mtime(run_files(run.path))
    sha256 = sha256_if_file(path) or ""
    return FreezeAudit(
        prereg_sha256=sha256,
        versions=len(chain),
        created_utc=frozen.created_utc,
        replaces_sha256=frozen.replaces_sha256,
        first_created_utc=first,
        earliest_eval_utc=earliest,
        blind=frozen_before(first, earliest),
        frozen_after_eval=frozen_after(frozen.created_utc, earliest),
        post_hoc=frozen.post_hoc or compared_under_other(logged, sha256),
        frozen_commit=frozen.commit,
        report_commit=commit,
        deviations=list(frozen.deviations),
    )


def run_comparison[S: CalibrationStratum](
    settings: Settings,
    checks: Mapping[str, type[Check]],
    dataset: VideophyDataset,
    request: CompareRequest,
    api_loader: ApiLoader[S],
) -> Comparison:
    agreement = agreement_settings(settings)
    if request.rule is not None:
        agreement = AgreementSettings.model_validate(
            {**agreement.model_dump(), "rule": request.rule}
        )
    subset = request.subset or agreement.test_subset
    run = load_run(request.run_dir)
    rows = dataset.rows(subset)
    setup = build_rule(settings, checks, agreement, run, api_loader)
    check_run(run, rows, setup, None)
    default_path = preregistration_path(settings, agreement)
    path = request.preregistration or default_path
    state = frozen_state(settings, dataset, setup, run.profile)
    on_test = any(row.split is Split.EVAL for row in rows)
    frozen, preregistered = _frozen(path, on_test, state)
    audit_ = (
        freeze_audit(path, frozen, run, str(state["commit"]), logged_comparisons(default_path))
        if preregistered and frozen is not None
        else None
    )
    return compare(
        ComparisonInputs(
            rows=rows,
            run=run,
            rule=setup.rule,
            agreement=agreement,
            cut=None if frozen is None else frozen.dev_cut,
            dev_tuned_cut=None if frozen is None else frozen.dev_tuned_cut,
            dev_majority=None if frozen is None else frozen.dev_majority,
            preregistered=preregistered,
            audit=audit_,
            scope=setup.scope,
            gate=setup.gate,
            calibration=setup.calibration,
            dev_design_effect=None if frozen is None else frozen.dev_design_effect,
            gaps=dataset_gaps(dataset, rows),
        )
    )


def log_test_comparison(
    settings: Settings, comparison: Comparison, run_dir: Path, now: dt.datetime
) -> Path | None:
    """Logs a pre-registered comparison next to the file ``agreement freeze`` writes, so a
    later refreeze is known to be post hoc."""
    audit_ = comparison.audit
    if audit_ is None:
        return None
    path = preregistration_path(settings, agreement_settings(settings))
    entry = LoggedComparison(
        prereg_sha256=audit_.prereg_sha256, run=str(run_dir), compared_utc=now.isoformat()
    )
    log_comparison(path, entry)
    return path


def reviewed_candidates(dataset: VideophyDataset, subset: str) -> dict[str, list[VideophyRow]]:
    reviewed = dataset.settings.reviewed_subsets
    return {p: rows for p, rows in dataset.candidates(subset).items() if p in reviewed}


def audit(dataset: VideophyDataset, subset: str) -> SubsetReview:
    """The pending review over the lexical rule's captions (a recorded review is kept as is)."""
    previous = load_review(dataset.subset_path)
    if previous is not None and previous.status == REVIEWED:
        return previous
    captions = {
        p: [r.caption for r in rows] for p, rows in reviewed_candidates(dataset, subset).items()
    }
    return template(captions, dataset.settings.filter_version, previous)


SUBJECT_ENDS = ("is", "are", "was", "were", "gets", "being", "and", "on", "onto", "into", "off")


def draft_queries(
    existing: Mapping[str, object], captions: Sequence[str], rigid_event: Sequence[str]
) -> dict[str, object]:
    """``{caption: [phrase]}``: the words before the first event verb or ``SUBJECT_ENDS`` word,
    a draft for the hand review; entries already written are kept."""
    drafts: dict[str, object] = dict(existing)
    ends = (*rigid_event, *SUBJECT_ENDS)
    for caption in sorted(set(captions)):
        if caption not in drafts:
            drafts[caption] = [q.phrase for q in derive_queries(caption, ends)]
    return dict(sorted(drafts.items()))


def uncovered(queries: Mapping[str, object], labels: frozenset[str]) -> list[str]:
    """Query phrases with no size: neither the phrase nor its head noun (last word) is sized."""
    phrases = {
        p
        for entries in queries.values()
        if isinstance(entries, list)
        for p in entries
        if isinstance(p, str)
    }
    return sorted(p for p in phrases if p not in labels and p.split()[-1] not in labels)


def sized_labels_or_none(settings: Settings) -> frozenset[str] | None:
    """A9's sized labels, or None when A9's model is not installed yet."""
    try:
        from physverify.sizing.model import sized_labels
    except ImportError:
        return None
    return sized_labels(settings)
