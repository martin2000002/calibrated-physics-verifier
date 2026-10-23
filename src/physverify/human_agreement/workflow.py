"""``agreement freeze`` and ``agreement compare`` as functions of injected settings, dataset and
calibration API: the CLI only parses options and prints."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from physverify.checks._base import Check
from physverify.core.digests import sha256_if_file
from physverify.core.errors import UsageError
from physverify.core.log import get_logger
from physverify.core.output import Echo
from physverify.human_agreement.clip_decision import (
    CALIBRATED_NAME,
    CalibrationStratum,
)
from physverify.human_agreement.comparison import (
    Comparison,
    ComparisonInputs,
    compare,
    freeze_dev,
)
from physverify.human_agreement.coverage import dataset_gaps
from physverify.human_agreement.policy_metrics import SECONDARY_TESTS
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
    plan_refreeze,
    preregistration_chain,
    require_unchanged,
    write_preregistration,
)
from physverify.human_agreement.report import REPORT_DIR, write_comparison
from physverify.human_agreement.rule_setup import ApiLoader, RuleSetup, build_rule, check_run
from physverify.human_agreement.settings import (
    SECTION,
    AgreementSettings,
    agreement_settings,
    preregistration_path,
)
from physverify.human_agreement.verify_run_reader import (
    VerifyRun,
    load_run,
    run_files,
)
from physverify.runtime.runs import current_commit
from physverify.schema.clip import Split
from physverify.schema.io import stable_hash
from physverify.verification.scope import SECTION as SCOPE_SECTION
from physverify.verification.scope import require_record_for

if TYPE_CHECKING:
    from physverify.config.models import Settings
    from physverify.datasets._base import HumanLabelSource

# The command's logger name, kept so the log lines of ``agreement freeze`` do not change.
log = get_logger("cli.agreement")


def frozen_state(
    settings: Settings, dataset: HumanLabelSource, setup: RuleSetup, profile: str
) -> dict[str, object]:
    """Current values of the pre-registered fields (``prereg.FROZEN_FIELDS``)."""
    sections = (SECTION, dataset.config_section, SCOPE_SECTION)
    config = {name: settings.sections.get(name) for name in sections}
    return {
        "commit": current_commit(settings.repo_root),
        "agreement_config_sha256": stable_hash(config),
        "subset_sha256": dataset.recorded_subsets(),
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
    dataset: HumanLabelSource,
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
    dataset: HumanLabelSource,
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


def compare_and_report[S: CalibrationStratum](
    settings: Settings,
    checks: Mapping[str, type[Check]],
    dataset: HumanLabelSource,
    request: CompareRequest,
    api_loader: ApiLoader[S],
    clock: Callable[[], dt.datetime],
    *,
    echo: Echo,
) -> None:
    """Compare a verify run, log it if it is a test comparison, write the report and print the
    verdict. Logged before anything is written or shown: no test report may exist without its
    entry."""
    comparison = run_comparison(settings, checks, dataset, request, api_loader)
    log_test_comparison(settings, comparison, request.run_dir, clock())
    out_dir = request.run_dir / REPORT_DIR
    write_comparison(out_dir, comparison)
    lines = verdict_lines(comparison)
    echo(f"agreement: {out_dir}")
    for line in lines:
        echo(line)


def verdict_lines(comparison: Comparison) -> list[str]:
    verdict = comparison.metrics.h2_verdict
    lines = [f"rule: {comparison.metrics.rule}; compared {comparison.counts.compared}"]
    if not verdict.applicable:
        lines.append(f"H2: {verdict.reason}")
    else:
        post_hoc = " (POST HOC)" if verdict.post_hoc else ""
        lines.append(f"H2: {'supported' if verdict.supported else 'not supported'}{post_hoc}")
    return lines


def apply_freeze(
    path: Path,
    current: Preregistration,
    *,
    force: bool,
    deviation: str | None,
    echo: Echo,
    err: Echo,
) -> None:
    """Write the frozen state at ``path`` unless it is unchanged; a replacement is announced."""
    previous = sha256_if_file(path)
    old = None if previous is None else load_preregistration(path)
    record_ = plan_refreeze(
        old,
        previous,
        current,
        force=force,
        deviation=deviation,
        tested=bool(logged_comparisons(path)),
    )
    if record_ is None:
        echo(f"preregistration: {path}")
        echo("unchanged frozen state; the existing pre-registration is kept")
        return
    if previous is not None:
        log.warning("replacing pre-registration %s (sha256 %s)", path, previous)
        err(f"replacing pre-registration {path} (sha256 {previous})")
    if record_.post_hoc:
        err(
            "WARNING: a test comparison already exists; the human-agreement verdict is now post hoc"
        )
    write_preregistration(path, record_)
    echo(f"preregistration: {path}")
    echo(
        f"rule {record_.rule_label}; n_dev {record_.n_dev}; majority {record_.dev_majority}; "
        f"cut {record_.dev_cut}; tuned cut {record_.dev_tuned_cut}"
    )
