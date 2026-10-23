"""``physverify exclusions``: the preregistered exclusion decision (ADR 0015 rules 1.1, 1.2, 1.5).

``r2-clips`` lists the clips both R2 detector arms must verify (``verify --policy raw``);
``decide`` reads the artifact, its evaluation and the two runs and writes the decision record
that ``[check_scope]`` merges; ``show`` prints that record and its sha256. Not heavy: nothing
here runs a model.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from physverify.app.context import AppContext
from physverify.app.services import context_checks_hash, resolve_dataset
from physverify.calibration.artifact import EvaluationReport
from physverify.calibration.artifact_io import (
    CalibrationError,
    default_artifact_path,
    load_calibration,
)
from physverify.calibration.collect import read_run, run_level_of
from physverify.calibration.exclusion_record import (
    ExclusionInputError,
    ExclusionRecord,
    load_record,
    parse_record,
)
from physverify.calibration.exclusion_rules import (
    SECTION,
    ArmRun,
    DecisionInputs,
    ExclusionRules,
    R2Arms,
    R2Rule,
    consumed_stratum,
    decide,
    exclusion_rules,
    record_markdown,
)
from physverify.calibration.settings import (
    calibration_config,
    calibration_settings,
)
from physverify.calibration.table import CalibrationDataError, build_rows, in_family, scene_of
from physverify.cli.context import app_context
from physverify.core.atomic import write_text_atomic
from physverify.core.digests import text_sha256
from physverify.core.errors import UsageError
from physverify.finetune.scope import family_targets
from physverify.finetune.settings import FinetuneSettings, load_finetune
from physverify.runtime.runs import RUN_FILE, new_run
from physverify.schema.clip import ClipManifest
from physverify.schema.io import stable_hash
from physverify.schema.stages import Stage
from physverify.synth.perturbations.none import NAME as NONE
from physverify.verification.runner import (
    CHECKS_FILE,
    CHECKS_HASH_FILE,
    TABLE_FILE,
    VerificationReport,
)
from physverify.verification.scope import SECTION as SCOPE_SECTION
from physverify.verification.scope import CheckScope, record_path
from physverify.verification.sweep import CheckUnit, perturbation_name

FamilyGap = Callable[[frozenset[str]], list[str]]
RunPair = tuple[str, str]

NAME = "exclusions"

app = typer.Typer(
    help="ADR 0015: the preregistered check exclusions H2 and H3 consume.",
    no_args_is_help=True,
)


# --- pure helpers ------------------------------------------------------------------------------


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


def families_without_target(cfg: FinetuneSettings, scope: CheckScope) -> FamilyGap:
    """The H3 families left with no target check by the excluded ids plus the manual
    ``exclude_checks`` (and the time-dependent ones when ``[finetune]`` excludes them)."""
    fixed = frozenset(scope.excluded(time_dependent=cfg.exclude_time_dependent))

    def gap(excluded: frozenset[str]) -> list[str]:
        kept = family_targets(cfg, excluded | fixed)
        return sorted(name for name, targets in kept.items() if not targets)

    return gap


# --- loading -----------------------------------------------------------------------------------


def _rules(context: AppContext) -> ExclusionRules:
    known = context.registry("checks")
    return exclusion_rules(context.settings.sections.get(SECTION, {}), known)


def _scope(context: AppContext) -> CheckScope:
    """``[check_scope]`` as configured, without the record it may already point to."""
    return context.settings.section(SCOPE_SECTION, CheckScope)


def _evaluation(path: Path) -> tuple[EvaluationReport, str]:
    if not path.is_file():
        raise UsageError(f"no evaluation at {path}: run `physverify calibrate evaluate`")
    text = path.read_text(encoding="utf-8")
    try:
        return EvaluationReport.model_validate_json(text), text_sha256(text)
    except ValidationError as exc:
        raise UsageError(f"{path} is not an evaluation report:\n{exc}") from exc


def _units(context: AppContext) -> tuple[CheckUnit, ...]:
    registry = context.registry("checks")
    return tuple(CheckUnit.of(registry[c]) for c in sorted(registry))


def _arms(context: AppContext, rules: ExclusionRules, base: str, tiny: str) -> R2Arms:
    settings = context.settings
    clips = resolve_dataset(context, rules.r2.dataset).clips(None)
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
    report, report_sha = _evaluation(evaluation)
    checks_hash = context_checks_hash(context)
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


# --- commands ----------------------------------------------------------------------------------


@app.command("r2-clips", help="The clip ids both R2 detector arms verify (rule 1.2).")
def r2_clips(
    out: Annotated[Path | None, typer.Option(help="Write the ids here, one per line.")] = None,
) -> None:
    context = app_context()
    cfg = _rules(context).r2
    ids = r2_clip_ids(resolve_dataset(context, cfg.dataset).clips(cfg.subset), cfg)
    if not ids:
        raise UsageError(f"{cfg.dataset}/{cfg.subset} has no {cfg.family} {cfg.split.value} clip")
    text = "\n".join(ids) + "\n"
    if out is None:
        typer.echo(text, nl=False)
        return
    _write((out,), text)
    typer.echo(f"{len(ids)} clips: {out}")


@app.command("decide", help="Apply rules 1.1, 1.2 and 1.5 and write the decision record.")
def decide_command(
    evaluation: Annotated[Path, typer.Option(help="evaluation.json of `calibrate evaluate`.")],
    r2_base_run: Annotated[str, typer.Option(help="Run id of the base-detector R2 arm.")],
    r2_tiny_run: Annotated[str, typer.Option(help="Run id of the tiny-detector R2 arm.")],
    artifact: Annotated[
        Path | None, typer.Option(help="Artifact (default: the calibration artifact).")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the decision, write nothing.")] = False,
    replace: Annotated[bool, typer.Option(help="Supersede a different record.")] = False,
) -> None:
    context = app_context()
    settings = context.settings
    rules = _rules(context)
    scope = _scope(context)
    target = record_path(scope, settings.paths.calibration_dir)
    if target is None:
        raise UsageError("[check_scope].decision_record is unset: nowhere to write the record")
    runs = (r2_base_run, r2_tiny_run)
    inputs = _inputs(context, rules, artifact or default_artifact_path(settings), evaluation, runs)
    fresh = decide(inputs, rules, families_without_target(load_finetune(settings), scope))
    typer.echo(record_markdown(fresh), nl=False)
    if dry_run:
        return
    existing = target.read_text(encoding="utf-8") if target.is_file() else None
    record = replacement(existing, fresh, replace=replace, label=str(target))
    if record is None:
        typer.echo(f"{target}: unchanged")
        return
    run = new_run(context.settings, context.invocation.argv)
    text, markdown = record.to_json(), record_markdown(record)
    _write((target, run.run_dir / target.name), text)
    _write((target.with_suffix(".md"), run.run_dir / target.with_suffix(".md").name), markdown)
    typer.echo(f"record: {target} sha256 {record.sha256}")


@app.command("show", help="Print the decision record and its sha256 (what H2/H3 freeze on).")
def show_command(
    checks_hash: Annotated[
        bool, typer.Option(help="Refuse a record decided with other check parameters.")
    ] = True,
) -> None:
    context = app_context()
    settings = context.settings
    target = record_path(_scope(context), settings.paths.calibration_dir)
    if target is None:
        raise UsageError("[check_scope].decision_record is unset: there is no record to show")
    if not target.is_file():
        raise UsageError(f"no decision record at {target}: run `physverify exclusions decide`")
    if checks_hash:
        record = load_record(target, context_checks_hash(context))
    else:
        record = parse_record(target.read_text(encoding="utf-8"), str(target))
        typer.echo("note: checks_hash not validated against the active check parameters")
    typer.echo(record_markdown(record), nl=False)
    _artifact_note(default_artifact_path(settings), record)
    typer.echo(f"record: {target} sha256 {record.sha256}")


def _artifact_note(artifact: Path, record: ExclusionRecord) -> None:
    if not artifact.is_file():
        return
    current = text_sha256(artifact.read_text(encoding="utf-8"))
    if current != record.inputs.artifact_sha256:
        typer.echo(
            f"note: {artifact} (sha256 {current[:12]}) is not the artifact the record was "
            f"decided on ({record.inputs.artifact_sha256[:12]})"
        )
