"""``physverify h3``: fine-tuned against base against best-of-8 (ADR 0013).

Order: ``gate-autoeval``, ``judge --split dev``, ``freeze``, ``generate``, ``score``,
``judge --split eval``, ``compare``. Every step resumes. The GPU subcommands exit 2 without
``--device cuda`` and a CUDA device, before they write anything; ``score``, ``rescore``,
``judge --split eval`` and ``compare`` refuse when the pre-registered state changed.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import typer

from physverify.agreement.prereg import file_sha256
from physverify.calibration.loader import default_artifact_path
from physverify.calibration.settings import active_checks_hash
from physverify.cli.commands._finetune_deps import (
    candidate_generator,
    require_cuda,
    scoring_deps,
    write_calibration_status,
)
from physverify.cli.commands._stop_rule import (
    StopRuleDeviationOption,
    current_verdict,
    require_stop_rule,
)
from physverify.cli.context import AppContext, app_context, heavy_command
from physverify.core.errors import EXIT_ERROR, ConfigError, MissingInputError, UsageError
from physverify.datasets._videophy_tables import LABELS_DIR, read_table
from physverify.datasets.videophy import FETCH_HINT, VideophyDataset
from physverify.evaluation.arms import EVAL_SPLIT, analysis_prompts, eval_prompts
from physverify.evaluation.campaign import eval_records, eval_scores, has_eval_scores
from physverify.evaluation.compare import CompareInputs, build_compare
from physverify.evaluation.gate import gate_rows
from physverify.evaluation.judges._base import JudgeInput
from physverify.evaluation.judges.videophy2_autoeval import Videophy2AutoevalJudge
from physverify.evaluation.judging import (
    candidate_items,
    gate_items,
    gate_problem,
    gate_reference,
    gate_report,
    judge_output,
    reusable,
)
from physverify.evaluation.layout import H3Layout
from physverify.evaluation.prereg import (
    FreezeRequest,
    H3Preregistration,
    identity,
    load_prereg,
    optional_prereg,
    plan_freeze,
    require_unchanged,
    write_prereg,
)
from physverify.evaluation.records import (
    JUDGE_SPLITS,
    TASKS,
    GateReport,
    JudgeSplit,
    ScoringManifest,
    load_judge_output,
)
from physverify.evaluation.report import write_report
from physverify.evaluation.scoring import frozen_stamp_problem, manifest, rescore, stale
from physverify.evaluation.settings import H3Settings, load_h3
from physverify.evaluation.state import (
    DEV_SPLIT,
    FreezeInputs,
    current_record,
    require_exclusion_record,
)
from physverify.finetune.records import BASE_TAG, PromptRecord, read_record, write_record
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scoring import build_scorer, rescorer, score_candidates, scorer_registry
from physverify.finetune.seeds import parse_shard
from physverify.finetune.selection import Selection
from physverify.finetune.settings import FinetuneSettings, load_finetune
from physverify.finetune.stop_rule import fired_reason, frozen_state
from physverify.finetune.store import SELECTED_FILE, CampaignStore
from physverify.runtime.runs import current_commit

NAME = "h3"
OWNER = "A13"
IMPLEMENTED = True
ARMS: Final = ("base", "ft")
GATE_LINE: Final = "  gate: "
GATE_PASSED: Final = "passed"
"""``scripts/h200/run_h3.sh eval`` reads this status line to skip the judge without a passed
gate."""

app = typer.Typer(no_args_is_help=True, help="H3: fine-tuned vs base vs best-of-8 (ADR 0013).")

ArmOption = Annotated[str, typer.Option(help="base, or ft (the selected checkpoint).")]
ShardOption = Annotated[str | None, typer.Option(help="Shard i/n of the candidates.")]
TaskOption = Annotated[str | None, typer.Option(help="pc or sa (default: both).")]


@dataclass(frozen=True)
class H3Context:
    context: AppContext
    cfg: H3Settings
    finetune: FinetuneSettings
    store: CampaignStore
    layout: H3Layout


def _h3() -> H3Context:
    context = app_context()
    store = CampaignStore.from_settings(context.settings)
    return H3Context(
        context,
        load_h3(context.settings),
        load_finetune(context.settings),
        store,
        H3Layout.of(store.root),
    )


def _prompts(h: H3Context) -> list[PromptRecord]:
    if not h.store.prompts_path.is_file():
        raise UsageError(f"{h.store.prompts_path} does not exist: run 'finetune prompts build'")
    return h.store.prompts()


def _selection(h: H3Context) -> Selection:
    path = h.store.select_dir(h.cfg.selection_run) / SELECTED_FILE
    if not path.is_file():
        raise UsageError(f"{path} does not exist: run 'finetune select'")
    return read_record(path, Selection)


def _arm(h: H3Context, arm: str) -> tuple[str, Path | None]:
    """(model tag, adapter) of an arm; ft is the checkpoint chosen by ``finetune select``."""
    if arm not in ARMS:
        raise UsageError(f"--arm must be one of {list(ARMS)}, got {arm!r}")
    if arm == "base":
        return BASE_TAG, None
    chosen = _selection(h).selected
    if chosen is None:
        raise UsageError("no checkpoint passed selection, so there is no ft arm")
    return chosen.model_tag, Path(chosen.adapter)


def _scorer(h: H3Context) -> AggregateScorer:
    return build_scorer(h.finetune.scorer, h.context.settings)


def _active_scorer(h: H3Context) -> AggregateScorer:
    scorer = _scorer(h)
    reason = scorer.unavailable()
    if reason is not None:
        raise UsageError(f"scorer {scorer.name} is unavailable: {reason}")
    return scorer


def _checks_hash(h: H3Context) -> str:
    return active_checks_hash(h.context.settings, h.context.registry("checks"))


def _judge(h: H3Context) -> Videophy2AutoevalJudge:
    return Videophy2AutoevalJudge(h.cfg.autoeval, h.layout.runs)


def _available_judge(h: H3Context) -> Videophy2AutoevalJudge:
    judge = _judge(h)
    reason = judge.unavailable()
    if reason is not None:
        raise UsageError(reason)
    return judge


def _reference_path(h: H3Context) -> Path:
    dataset = VideophyDataset(h.context.settings)
    part = dataset.settings.parts.get(h.cfg.gate.part)
    if part is None or part.reference is None:
        raise ConfigError(f"[videophy.parts.{h.cfg.gate.part}] has no reference table")
    return (
        h.context.settings.paths.raw_dir / VideophyDataset.name / LABELS_DIR / part.reference.file
    )


def _reference_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise UsageError(f"{path} is missing; run `{FETCH_HINT}`")
    return read_table(path)


def _current(h: H3Context) -> H3Preregistration:
    _prompts(h)
    scorer = _scorer(h)
    source = _reference_path(h)
    settings = h.context.settings
    inputs = FreezeInputs(
        settings=settings,
        store=h.store,
        layout=h.layout,
        cfg=h.cfg,
        finetune=h.finetune,
        scorer=scorer,
        rescore=rescorer(scorer),
        judge=_judge(h),
        selection=_selection(h),
        calibration_path=default_artifact_path(settings),
        margin_rows=_reference_rows(source),
        margin_source=source,
        commit=current_commit(settings.repo_root),
        stop_rule=frozen_state(current_verdict(settings, h.finetune, h.store, scorer)),
    )
    try:
        return current_record(inputs, dt.datetime.now(dt.UTC))
    except MissingInputError as exc:
        raise UsageError(str(exc)) from exc


def _require_frozen(h: H3Context) -> tuple[H3Preregistration, str]:
    """The pre-registration and its content hash, refused when the current state differs."""
    prereg = load_prereg(h.layout.prereg)
    require_unchanged(prereg, identity(_current(h)))
    sha256 = file_sha256(h.layout.prereg)
    if sha256 is None:
        raise UsageError(f"{h.layout.prereg} disappeared while it was read")
    return prereg, sha256


def _tags(prereg: H3Preregistration) -> list[str]:
    return [BASE_TAG] if prereg.adapter_tag is None else [BASE_TAG, prereg.adapter_tag]


@app.command("generate", help="Generate the eval candidates of one arm (resumes).")
@heavy_command
def generate(
    arm: ArmOption = "base",
    shard: ShardOption = None,
    limit: int | None = None,
    stop_rule_deviation: StopRuleDeviationOption = None,
) -> None:
    h = _h3()
    _require_stop_rule(h, stop_rule_deviation)
    require_cuda(h.context.settings)
    tag, adapter = _arm(h, arm)
    candidates = h.finetune.candidates.model_copy(
        update={"store_latents": h.cfg.eval_store_latents}
    )
    generator = candidate_generator(h.context, h.store, adapter, candidates=candidates)
    if generator.model_tag != tag:
        raise UsageError(f"{adapter} now hashes to {generator.model_tag}, selected was {tag}")
    prompts = eval_prompts(_prompts(h))
    analysed = analysis_prompts(prompts, h.cfg.exclude_tags)
    analysed_ids = {p.prompt_id for p in analysed}
    shown = [p for p in prompts if p.prompt_id not in analysed_ids]
    plan = [
        *generator.plan(analysed, parse_shard(shard)),
        *generator.plan(shown, parse_shard(shard), k=1),
    ]
    plan = plan if limit is None else plan[:limit]
    report = generator.run(plan, {p.prompt_id: p for p in prompts})
    typer.echo(f"{tag}: generated {len(report.generated)} eval candidates")


def _require_stop_rule(h: H3Context, deviation: str | None) -> bool:
    """The eval arm runs only when the pilot showed the verifier can rank candidates."""
    return require_stop_rule(h.context.settings, h.finetune, h.store, _scorer(h), deviation).passed


def _eval_data(h: H3Context) -> bool:
    scorers = list(scorer_registry())
    return has_eval_scores(h.store, scorers) or bool(h.layout.judge_outputs(EVAL_SPLIT))


@app.command("freeze", help="Write the pre-registration (before any eval score or judge row).")
def freeze(
    force: Annotated[
        bool, typer.Option(help="Replace a pre-registration whose state changed.")
    ] = False,
    declare_deviation: Annotated[
        str | None,
        typer.Option(
            help="Why a non-recalibration field changed, or any field once eval data exist "
            "(with --force; the latter makes the verdict post hoc)."
        ),
    ] = None,
    stop_rule_deviation: StopRuleDeviationOption = None,
) -> None:
    h = _h3()
    passed = _require_stop_rule(h, stop_rule_deviation)
    current = _current(h).model_copy(
        update={"stop_rule_deviation": None if passed else stop_rule_deviation}
    )
    require_exclusion_record(h.context.settings, current)
    request = FreezeRequest(force, declare_deviation, _eval_data(h), h.layout.gate.is_file())
    old = optional_prereg(h.layout.prereg)
    record = plan_freeze(old, file_sha256(h.layout.prereg), current, request)
    if record is None:
        typer.echo(f"{h.layout.prereg}: unchanged")
        return
    write_prereg(h.layout, record)
    for line in _freeze_lines(record):
        typer.echo(line)
    typer.echo(f"frozen: {h.layout.prereg} sha256 {file_sha256(h.layout.prereg)}")


def _freeze_lines(record: H3Preregistration) -> list[str]:
    usability = "not judged yet" if record.usability is None else str(record.usability.usable)
    fired = fired_reason(record.stop_rule, record.stop_rule_deviation)
    return [
        f"rule {record.rule} ({record.rule_status}, label {record.rule_label}); "
        f"contaminated {record.contaminated}",
        f"ft model {record.adapter_tag or 'none'}; prompts {json.dumps(record.eval_counts)}",
        f"margins {json.dumps(record.margins)}; external judge usable on dev: {usability}",
        f"MDE (estimated on dev): verifier {record.mde.verifier_mde}, PC {record.mde.pc_mde}",
        *[f"deviation: {d}" for d in record.deviations],
        *([] if fired is None else [f"NO VERDICT: {fired}"]),
        *(["POST HOC: refrozen after eval data existed"] if record.post_hoc else []),
    ]


@app.command("score", help="Extract, verify and score the eval candidates of one arm (resumes).")
@heavy_command
def score(arm: ArmOption = "base", shard: ShardOption = None) -> None:
    h = _h3()
    require_cuda(h.context.settings)
    prereg, sha256 = _require_frozen(h)
    scorer = _active_scorer(h)
    problem = frozen_stamp_problem(scorer, prereg)
    if problem is not None:
        raise UsageError(problem)
    tag, _ = _arm(h, arm)
    if stale(eval_scores(h.store, tag, scorer.name), scorer):
        raise UsageError(f"{tag} has eval scores of another stamp: run `physverify h3 rescore`")
    prompts = eval_prompts(_prompts(h))
    deps = scoring_deps(h.context)
    score_candidates(h.store, tag, prompts, scorer, deps, parse_shard(shard))
    write_calibration_status(h.store, tag, scorer)
    record = manifest(h.store, tag, scorer, _checks_hash(h), sha256)
    write_record(h.layout.scoring(tag, scorer.name), record)
    typer.echo(f"{tag}/{scorer.name}: {record.n} eval scores under {record.stamp}")


@app.command("rescore", help="Re-aggregate stored eval check results under the frozen rule (CPU).")
def rescore_command() -> None:
    h = _h3()
    prereg, sha256 = _require_frozen(h)
    scorer = _active_scorer(h)
    problem = frozen_stamp_problem(scorer, prereg)
    if problem is not None:
        raise UsageError(problem)
    active = _checks_hash(h)
    for tag in _tags(prereg):
        path = h.layout.scoring(tag, scorer.name)
        if not path.is_file():
            raise UsageError(f"{path} is missing: run `physverify h3 score`")
        if read_record(path, ScoringManifest).checks_hash != active:
            raise UsageError(f"{tag}: the checks changed since scoring; score again, not rescore")
        changed = rescore(h.store, tag, scorer)
        write_record(path, manifest(h.store, tag, scorer, active, sha256))
        typer.echo(f"{tag}/{scorer.name}: {changed} scores restamped to {scorer.thresholds()}")


def _split(split: str) -> JudgeSplit:
    for known in JUDGE_SPLITS:
        if split == known:
            return known
    raise UsageError(f"--split must be one of {list(JUDGE_SPLITS)}, got {split!r}")


def _tasks(task: str | None) -> tuple[str, ...]:
    if task is None:
        return TASKS
    if task not in TASKS:
        raise UsageError(f"--task must be one of {list(TASKS)}, got {task!r}")
    return (task,)


def _gate(h: H3Context) -> GateReport | None:
    return read_record(h.layout.gate, GateReport) if h.layout.gate.is_file() else None


def _dev_items(h: H3Context) -> dict[str, list[JudgeInput]]:
    """Dev candidates of the base and of the selected checkpoint (the usability test)."""
    selected = _selection(h).selected
    tags = [BASE_TAG] if selected is None else [BASE_TAG, selected.model_tag]
    dev = {p.prompt_id: p for p in _prompts(h) if p.split == DEV_SPLIT}
    return {
        tag: candidate_items([r for r in h.store.candidates(tag) if r.split == DEV_SPLIT], dev)
        for tag in tags
    }


def _eval_items(h: H3Context, prereg: H3Preregistration) -> dict[str, list[JudgeInput]]:
    analysed = analysis_prompts(_prompts(h), h.cfg.exclude_tags)
    return {
        tag: candidate_items(eval_records(h.store, tag), {p.prompt_id: p for p in analysed})
        for tag in _tags(prereg)
    }


def _judge_one(
    h: H3Context,
    judge: Videophy2AutoevalJudge,
    where: tuple[JudgeSplit, str, str],
    items: list[JudgeInput],
    prereg_sha256: str | None,
) -> None:
    path = h.layout.judge_output(*where)
    ids = {i.video_id for i in items}
    if reusable(load_judge_output(path), judge.stamp(where[2]), prereg_sha256, ids):
        typer.echo(f"{path}: already judged")
        return
    try:
        output = judge_output(judge, items, where, prereg_sha256)
    except MissingInputError as exc:
        raise UsageError(str(exc)) from exc
    write_record(path, output)
    failed = sum(row.value is None for row in output.rows.values())
    typer.echo(f"{where[0]}/{where[1]}/{where[2]}: {len(output.rows)} judged, {failed} unread")


@app.command("judge", help="Run VideoPhy-2 AutoEval on the dev or eval candidates (resumes).")
@heavy_command
def judge(
    split: Annotated[str, typer.Option(help="dev (usability) or eval (after the freeze).")],
    task: TaskOption = None,
) -> None:
    h = _h3()
    require_cuda(h.context.settings)
    which = _split(split)
    tasks = _tasks(task)
    autoeval = _available_judge(h)
    problem = gate_problem(_gate(h), {t: autoeval.stamp(t) for t in tasks})
    if problem is not None:
        raise UsageError(problem)
    sha256: str | None = None
    if which == EVAL_SPLIT:
        prereg, sha256 = _require_frozen(h)
        items = _eval_items(h, prereg)
    else:
        items = _dev_items(h)
    for tag, tag_items in items.items():
        for t in tasks:
            _judge_one(h, autoeval, (which, tag, t), tag_items, sha256)


@app.command("gate-autoeval", help="Regression gate of AutoEval on published VideoPhy-2 videos.")
@heavy_command
def gate_autoeval() -> None:
    h = _h3()
    require_cuda(h.context.settings)
    autoeval = _available_judge(h)
    dataset = VideophyDataset(h.context.settings)
    part_rows = [r for r in dataset.rows(h.cfg.gate.subset) if r.part == h.cfg.gate.part]
    rows = gate_rows(part_rows, lambda r: dataset.video_file(r).is_file(), h.cfg.gate.n)
    if not rows:
        raise UsageError(f"no {h.cfg.gate.part} videos on disk: run `{FETCH_HINT}`")
    source = _reference_path(h)
    table = {r["video_url"]: r for r in _reference_rows(source)}
    items = gate_items(rows, {r.clip_id: dataset.video_file(r) for r in rows})
    try:
        judged = {t: autoeval.judge(items, t) for t in TASKS}
    except MissingInputError as exc:
        raise UsageError(str(exc)) from exc
    references = {t: gate_reference(rows, table, h.cfg.gate.columns[t]) for t in TASKS}
    freeze_file = h.cfg.autoeval.path(h.cfg.autoeval.freeze_file)
    shas = (file_sha256(source), None if freeze_file is None else file_sha256(freeze_file))
    stamps = {t: autoeval.stamp(t) for t in TASKS}
    report = gate_report(judged, references, stamps, shas, h.cfg.gate)
    write_record(h.layout.gate, report)
    for t in report.tasks.values():
        typer.echo(
            f"{t.task}: exact {t.exact}/{t.n} ({t.exact_share:.2f}), MAE {t.mae}, "
            f"unread {t.parse_failures}, passed {t.passed}"
        )
    typer.echo(f"gate {'passed' if report.passed else 'FAILED'}: {h.layout.gate}")
    if not report.passed:
        raise typer.Exit(code=EXIT_ERROR)


@app.command("compare", help="The H3 analysis and verdict from the frozen eval data (CPU).")
def compare() -> None:
    h = _h3()
    prereg, sha256 = _require_frozen(h)
    scorer = _active_scorer(h)
    inputs = CompareInputs(
        store=h.store,
        layout=h.layout,
        cfg=h.cfg,
        finetune=h.finetune,
        prereg=prereg,
        prereg_sha256=sha256,
        scorer=scorer.name,
        rescore=rescorer(scorer),
        commit=current_commit(h.context.settings.repo_root),
    )
    report = build_compare(inputs)
    write_report(h.layout, report)
    verdict = report.verdict
    typer.echo(f"verdict: {verdict.verdict} (external {verdict.external_status})")
    for reason in verdict.reasons:
        typer.echo(f"  {reason}")
    typer.echo(f"report: {h.layout.report_md}")


def _tag_lines(h: H3Context, tag: str) -> list[str]:
    records = eval_records(h.store, tag)
    lines = [f"  {tag}: {len(records)} eval candidates"]
    for name in scorer_registry():
        scores = eval_scores(h.store, tag, name)
        if scores:
            abstained = sum(s.score is None for s in scores)
            lines.append(f"    {name}: {len(scores)} scored, {abstained} abstained")
    latents = {r.latents_path is not None for r in records}
    if latents - {h.cfg.eval_store_latents}:
        lines.append(
            "    WARNING: store_latents differs from "
            f"h3.eval_store_latents={h.cfg.eval_store_latents}"
        )
    return lines


def _judge_lines(h: H3Context) -> list[str]:
    lines = []
    for split in JUDGE_SPLITS:
        for path in h.layout.judge_outputs(split):
            output = load_judge_output(path)
            if output is not None:
                unread = sum(r.value is None for r in output.rows.values())
                lines.append(
                    f"  judge {split}/{path.stem}: {len(output.rows)} rows, {unread} unread"
                )
    return lines


def status_lines(h: H3Context) -> list[str]:
    prereg = optional_prereg(h.layout.prereg)
    gate = _gate(h)
    lines = [
        f"campaign {h.store.campaign}: {h.layout.root}",
        "  pre-registration: "
        + (
            "none" if prereg is None else f"{prereg.rule_status}, ft {prereg.adapter_tag or 'none'}"
        ),
        GATE_LINE + ("not run" if gate is None else (GATE_PASSED if gate.passed else "failed")),
    ]
    for tag in h.store.model_tags():
        lines.extend(_tag_lines(h, tag))
    return [*lines, *_judge_lines(h)]


@app.command("status", help="Eval candidates, scores, judge rows, gate and freeze of H3.")
def status() -> None:
    for line in status_lines(_h3()):
        typer.echo(line)
