"""Wiring of ``exclusions r2-clips``, ``decide`` and ``show`` from the application context."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.services import checks_hash_for, dataset_for
from physverify.calibration.artifact_io import (
    CalibrationError,
    default_artifact_path,
    load_calibration,
)
from physverify.calibration.exclusion_decision import decide, record_markdown
from physverify.calibration.exclusion_inputs import (
    DecisionInputs,
    R2Arms,
    arm_run,
    consumed_stratum,
    evaluation_report,
    r2_clip_ids,
    replacement,
)
from physverify.calibration.exclusion_rules import SECTION, ExclusionRules, exclusion_rules
from physverify.calibration.settings import (
    calibration_config,
    calibration_settings,
)
from physverify.core.atomic import write_text_atomic
from physverify.core.digests import text_sha256
from physverify.core.errors import UsageError
from physverify.finetune.scorer_scope import family_targets
from physverify.finetune.settings import FinetuneSettings, load_finetune
from physverify.runtime.runs import new_run
from physverify.verification.exclusion_record import (
    ExclusionRecord,
    load_record,
    parse_record,
)
from physverify.verification.scope import SECTION as SCOPE_SECTION
from physverify.verification.scope import CheckScope, record_path
from physverify.verification.sweep import CheckUnit

Out = Callable[[str], None]
FamilyGap = Callable[[frozenset[str]], list[str]]
RunPair = tuple[str, str]


def r2_clip_listing(context: AppContext, out: Path | None) -> str:
    """``exclusions r2-clips``: the ids themselves, or where they were written."""
    cfg = _rules(context).r2
    ids = r2_clip_ids(dataset_for(context, cfg.dataset).clips(cfg.subset), cfg)
    if not ids:
        raise UsageError(f"{cfg.dataset}/{cfg.subset} has no {cfg.family} {cfg.split.value} clip")
    text = "\n".join(ids) + "\n"
    if out is None:
        return text
    _write((out,), text)
    return f"{len(ids)} clips: {out}\n"


def decide_exclusions(
    context: AppContext,
    evaluation: Path,
    runs: RunPair,
    artifact: Path | None,
    *,
    dry_run: bool,
    replace: bool,
    out: Out,
) -> None:
    """``exclusions decide``: the record is printed before it is written, so ``out`` streams."""
    settings = context.settings
    rules = _rules(context)
    scope = _scope(context)
    target = record_path(scope, settings.paths.calibration_dir)
    if target is None:
        raise UsageError("[check_scope].decision_record is unset: nowhere to write the record")
    inputs = _inputs(context, rules, artifact or default_artifact_path(settings), evaluation, runs)
    fresh = decide(inputs, rules, families_without_target(load_finetune(settings), scope))
    out(record_markdown(fresh))
    if dry_run:
        return
    existing = target.read_text(encoding="utf-8") if target.is_file() else None
    record = replacement(existing, fresh, replace=replace, label=str(target))
    if record is None:
        out(f"{target}: unchanged\n")
        return
    run = new_run(context.settings, context.invocation.argv)
    text, markdown = record.to_json(), record_markdown(record)
    _write((target, run.run_dir / target.name), text)
    _write((target.with_suffix(".md"), run.run_dir / target.with_suffix(".md").name), markdown)
    out(f"record: {target} sha256 {record.sha256}\n")


def show_exclusions(context: AppContext, *, checks_hash: bool, out: Out) -> None:
    """``exclusions show``: the record, a note on an unvalidated or stale input, its sha256."""
    settings = context.settings
    target = record_path(_scope(context), settings.paths.calibration_dir)
    if target is None:
        raise UsageError("[check_scope].decision_record is unset: there is no record to show")
    if not target.is_file():
        raise UsageError(f"no decision record at {target}: run `physverify exclusions decide`")
    if checks_hash:
        record = load_record(target, checks_hash_for(context))
    else:
        record = parse_record(target.read_text(encoding="utf-8"), str(target))
        out("note: checks_hash not validated against the active check parameters\n")
    out(record_markdown(record))
    _artifact_note(default_artifact_path(settings), record, out)
    out(f"record: {target} sha256 {record.sha256}\n")


def families_without_target(cfg: FinetuneSettings, scope: CheckScope) -> FamilyGap:
    """The H3 families left with no target check by the excluded ids plus the manual
    ``exclude_checks`` (and the time-dependent ones when ``[finetune]`` excludes them)."""
    fixed = frozenset(scope.excluded(time_dependent=cfg.exclude_time_dependent))

    def gap(excluded: frozenset[str]) -> list[str]:
        kept = family_targets(cfg, excluded | fixed)
        return sorted(name for name, targets in kept.items() if not targets)

    return gap


def _rules(context: AppContext) -> ExclusionRules:
    known = context.registry("checks")
    return exclusion_rules(context.settings.sections.get(SECTION, {}), known)


def _scope(context: AppContext) -> CheckScope:
    """``[check_scope]`` as configured, without the record it may already point to."""
    return context.settings.section(SCOPE_SECTION, CheckScope)


def _units(context: AppContext) -> tuple[CheckUnit, ...]:
    registry = context.registry("checks")
    return tuple(CheckUnit.of(registry[c]) for c in sorted(registry))


def _arms(context: AppContext, rules: ExclusionRules, base: str, tiny: str) -> R2Arms:
    settings = context.settings
    clips = dataset_for(context, rules.r2.dataset).clips(None)
    couplings = calibration_config(settings).couplings
    units = _units(context)
    runs_dir = settings.paths.runs_dir
    return R2Arms(
        *(arm_run(runs_dir / run, clips, units, couplings, rules) for run in (base, tiny))
    )


def _inputs(
    context: AppContext, rules: ExclusionRules, artifact: Path, evaluation: Path, runs: RunPair
) -> DecisionInputs:
    """The cheap inputs are checked first; the two runs are read last."""
    settings = context.settings
    report, report_sha = evaluation_report(evaluation)
    checks_hash = checks_hash_for(context)
    try:
        calibration = load_calibration(artifact, checks_hash=checks_hash)
    except CalibrationError as exc:
        raise UsageError(str(exc)) from exc
    consumed = consumed_stratum(
        artifact,
        calibration,
        rules,
        checks_hash=checks_hash,
        require_calibrated=calibration_settings(settings).require_calibrated_artifact,
    )
    arms = _arms(context, rules, *runs)
    return DecisionInputs(calibration, consumed, report, report_sha, arms, checks_hash)


def _write(paths: Sequence[Path], text: str) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, text)


def _artifact_note(artifact: Path, record: ExclusionRecord, out: Out) -> None:
    if not artifact.is_file():
        return
    current = text_sha256(artifact.read_text(encoding="utf-8"))
    if current != record.inputs.artifact_sha256:
        out(
            f"note: {artifact} (sha256 {current[:12]}) is not the artifact the record was "
            f"decided on ({record.inputs.artifact_sha256[:12]})\n"
        )
