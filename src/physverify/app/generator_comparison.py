"""The generator comparison steps from the ``AppContext``: the scorer, the external judge and the
frozen state."""

from __future__ import annotations

import datetime as dt

from physverify.app.context import AppContext
from physverify.app.finetune import (
    active_scorer,
    candidate_generator,
    current_verdict,
    require_cuda,
    require_stop_rule,
    scoring_deps,
    write_calibration_status,
)
from physverify.app.services import checks_hash_for, external_judge, scorer_for
from physverify.calibration.artifact_io import default_artifact_path
from physverify.core.choices import choice
from physverify.core.digests import sha256_if_file
from physverify.core.errors import MissingInputError, UsageError
from physverify.core.output import Echo
from physverify.datasets.videophy import FETCH_HINT, VideophyDataset
from physverify.finetune.candidate_scoring import rescorer, score_candidates
from physverify.finetune.prompts import campaign_prompts
from physverify.finetune.records import read_record, write_record
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.seeds import parse_shard
from physverify.finetune.stop_rule import frozen_state
from physverify.generator_comparison.arms import EVAL_SPLIT, analysis_prompts, eval_prompts
from physverify.generator_comparison.candidate_queries import eval_scores, has_eval_scores
from physverify.generator_comparison.compare import CompareInputs, build_compare
from physverify.generator_comparison.context import (
    H3Context,
    arm_of,
    arm_tags,
    reference_path,
    reference_rows,
    selection,
    stored_gate,
)
from physverify.generator_comparison.judge_gate import gate_rows
from physverify.generator_comparison.judges._base import Judge
from physverify.generator_comparison.judging import (
    dev_items,
    eval_items,
    gate_items,
    gate_problem,
    gate_reference,
    gate_report,
    judge_one,
    judge_tasks,
)
from physverify.generator_comparison.prereg import (
    FreezeRequest,
    H3Preregistration,
    identity,
    load_prereg,
    optional_prereg,
    plan_freeze,
    require_unchanged,
    write_prereg,
)
from physverify.generator_comparison.prereg_state import (
    FreezeInputs,
    current_record,
    require_exclusion_record,
)
from physverify.generator_comparison.records import JUDGE_SPLITS, TASKS, ScoringManifest
from physverify.generator_comparison.report import write_report
from physverify.generator_comparison.scoring import frozen_stamp_problem, manifest, rescore, stale
from physverify.generator_comparison.status import freeze_lines, status_lines
from physverify.runtime.runs import current_commit


def campaign_scorer(context: AppContext, h: H3Context) -> AggregateScorer:
    return scorer_for(context, h.finetune.scorer)


def campaign_judge(context: AppContext, h: H3Context) -> Judge:
    return external_judge(context, h.cfg, h.layout.runs)


def available_judge(context: AppContext, h: H3Context) -> Judge:
    judge = campaign_judge(context, h)
    reason = judge.unavailable()
    if reason is not None:
        raise UsageError(reason)
    return judge


def current_prereg(context: AppContext, h: H3Context) -> H3Preregistration:
    campaign_prompts(h.store)
    scorer = campaign_scorer(context, h)
    source = reference_path(h)
    settings = context.settings
    inputs = FreezeInputs(
        settings=settings,
        checks=context.registry("checks"),
        store=h.store,
        layout=h.layout,
        cfg=h.cfg,
        finetune=h.finetune,
        scorer=scorer,
        rescore=rescorer(scorer),
        judge=campaign_judge(context, h),
        selection=selection(h),
        calibration_path=default_artifact_path(settings),
        margin_rows=reference_rows(source),
        margin_source=source,
        commit=current_commit(settings.repo_root),
        stop_rule=frozen_state(current_verdict(context, h.finetune, h.store, scorer)),
    )
    try:
        return current_record(inputs, dt.datetime.now(dt.UTC))
    except MissingInputError as exc:
        raise UsageError(str(exc)) from exc


def require_frozen(context: AppContext, h: H3Context) -> tuple[H3Preregistration, str]:
    """The pre-registration and its content hash, refused when the current state differs."""
    prereg = load_prereg(h.layout.prereg)
    require_unchanged(prereg, identity(current_prereg(context, h)))
    sha256 = sha256_if_file(h.layout.prereg)
    if sha256 is None:
        raise UsageError(f"{h.layout.prereg} disappeared while it was read")
    return prereg, sha256


def require_pilot_stop_rule(
    context: AppContext, h: H3Context, deviation: str | None, *, err: Echo
) -> bool:
    """The eval arm runs only when the pilot showed the verifier can rank candidates."""
    return require_stop_rule(
        context, h.finetune, h.store, campaign_scorer(context, h), deviation, err=err
    ).passed


def eval_data(context: AppContext, h: H3Context) -> bool:
    scorers = list(context.registry("scorers"))
    return has_eval_scores(h.store, scorers) or bool(h.layout.judge_outputs(EVAL_SPLIT))


def generate_arm(
    context: AppContext,
    arm: str,
    shard: str | None,
    limit: int | None,
    deviation: str | None,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    h = H3Context.of(context.settings)
    require_pilot_stop_rule(context, h, deviation, err=err)
    require_cuda(context.settings)
    tag, adapter = arm_of(h, arm)
    candidates = h.finetune.candidates.model_copy(
        update={"store_latents": h.cfg.eval_store_latents}
    )
    generator = candidate_generator(context, h.store, adapter, candidates=candidates)
    if generator.model_tag != tag:
        raise UsageError(f"{adapter} now hashes to {generator.model_tag}, selected was {tag}")
    prompts = eval_prompts(campaign_prompts(h.store))
    analysed = analysis_prompts(prompts, h.cfg.exclude_tags)
    analysed_ids = {p.prompt_id for p in analysed}
    shown = [p for p in prompts if p.prompt_id not in analysed_ids]
    plan = [
        *generator.plan(analysed, parse_shard(shard)),
        *generator.plan(shown, parse_shard(shard), k=1),
    ]
    plan = plan if limit is None else plan[:limit]
    report = generator.run(plan, {p.prompt_id: p for p in prompts})
    echo(f"{tag}: generated {len(report.generated)} eval candidates")


def freeze_prereg(
    context: AppContext,
    force: bool,
    declare_deviation: str | None,
    deviation: str | None,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    h = H3Context.of(context.settings)
    passed = require_pilot_stop_rule(context, h, deviation, err=err)
    current = current_prereg(context, h).model_copy(
        update={"stop_rule_deviation": None if passed else deviation}
    )
    require_exclusion_record(context.settings, current, context.registry("checks"))
    request = FreezeRequest(
        force, declare_deviation, eval_data(context, h), h.layout.gate.is_file()
    )
    old = optional_prereg(h.layout.prereg)
    record = plan_freeze(old, sha256_if_file(h.layout.prereg), current, request)
    if record is None:
        echo(f"{h.layout.prereg}: unchanged")
        return
    write_prereg(h.layout, record)
    for line in freeze_lines(record):
        echo(line)
    echo(f"frozen: {h.layout.prereg} sha256 {sha256_if_file(h.layout.prereg)}")


def _frozen_scorer(
    context: AppContext, h: H3Context
) -> tuple[H3Preregistration, str, AggregateScorer]:
    prereg, sha256 = require_frozen(context, h)
    scorer = active_scorer(context, h.finetune.scorer)
    problem = frozen_stamp_problem(scorer, prereg)
    if problem is not None:
        raise UsageError(problem)
    return prereg, sha256, scorer


def score_arm(context: AppContext, arm: str, shard: str | None, *, echo: Echo) -> None:
    h = H3Context.of(context.settings)
    require_cuda(context.settings)
    _, sha256, scorer = _frozen_scorer(context, h)
    tag, _ = arm_of(h, arm)
    if stale(eval_scores(h.store, tag, scorer.name), scorer):
        raise UsageError(
            f"{tag} has eval scores of another stamp: run `physverify compare-generators rescore`"
        )
    prompts = eval_prompts(campaign_prompts(h.store))
    deps = scoring_deps(context)
    score_candidates(h.store, tag, prompts, scorer, deps, parse_shard(shard))
    write_calibration_status(h.store, tag, scorer)
    record = manifest(h.store, tag, scorer, checks_hash_for(context), sha256)
    write_record(h.layout.scoring(tag, scorer.name), record)
    echo(f"{tag}/{scorer.name}: {record.n} eval scores under {record.stamp}")


def rescore_arms(context: AppContext, *, echo: Echo) -> None:
    h = H3Context.of(context.settings)
    prereg, sha256, scorer = _frozen_scorer(context, h)
    active = checks_hash_for(context)
    for tag in arm_tags(prereg):
        path = h.layout.scoring(tag, scorer.name)
        if not path.is_file():
            raise UsageError(f"{path} is missing: run `physverify compare-generators score`")
        if read_record(path, ScoringManifest).checks_hash != active:
            raise UsageError(f"{tag}: the checks changed since scoring; score again, not rescore")
        changed = rescore(h.store, tag, scorer)
        write_record(path, manifest(h.store, tag, scorer, active, sha256))
        echo(f"{tag}/{scorer.name}: {changed} scores restamped to {scorer.thresholds()}")


def judge_split(context: AppContext, split: str, task: str | None, *, echo: Echo) -> None:
    h = H3Context.of(context.settings)
    require_cuda(context.settings)
    which = choice(split, JUDGE_SPLITS, "--split")
    tasks = judge_tasks(task)
    autoeval = available_judge(context, h)
    problem = gate_problem(stored_gate(h), {t: autoeval.stamp(t) for t in tasks})
    if problem is not None:
        raise UsageError(problem)
    sha256: str | None = None
    if which == EVAL_SPLIT:
        prereg, sha256 = require_frozen(context, h)
        items = eval_items(h, prereg)
    else:
        items = dev_items(h)
    for tag, tag_items in items.items():
        for t in tasks:
            judge_one(h, autoeval, (which, tag, t), tag_items, sha256, echo=echo)


def run_gate(context: AppContext, *, echo: Echo) -> bool:
    """The AutoEval regression gate on published videos; True when it passed."""
    h = H3Context.of(context.settings)
    require_cuda(context.settings)
    autoeval = available_judge(context, h)
    dataset = VideophyDataset(context.settings)
    part_rows = [r for r in dataset.rows(h.cfg.gate.subset) if r.part == h.cfg.gate.part]
    rows = gate_rows(part_rows, lambda r: dataset.video_file(r).is_file(), h.cfg.gate.n)
    if not rows:
        raise UsageError(f"no {h.cfg.gate.part} videos on disk: run `{FETCH_HINT}`")
    source = reference_path(h)
    table = {r["video_url"]: r for r in reference_rows(source)}
    items = gate_items(rows, {r.clip_id: dataset.video_file(r) for r in rows})
    try:
        judged = {t: autoeval.judge(items, t) for t in TASKS}
    except MissingInputError as exc:
        raise UsageError(str(exc)) from exc
    references = {t: gate_reference(rows, table, h.cfg.gate.columns[t]) for t in TASKS}
    freeze_file = h.cfg.autoeval.path(h.cfg.autoeval.freeze_file)
    shas = (sha256_if_file(source), None if freeze_file is None else sha256_if_file(freeze_file))
    stamps = {t: autoeval.stamp(t) for t in TASKS}
    report = gate_report(judged, references, stamps, shas, h.cfg.gate)
    write_record(h.layout.gate, report)
    for t in report.tasks.values():
        echo(
            f"{t.task}: exact {t.exact}/{t.n} ({t.exact_share:.2f}), MAE {t.mae}, "
            f"unread {t.parse_failures}, passed {t.passed}"
        )
    echo(f"gate {'passed' if report.passed else 'FAILED'}: {h.layout.gate}")
    return report.passed


def compare_arms(context: AppContext, *, echo: Echo) -> None:
    h = H3Context.of(context.settings)
    prereg, sha256 = require_frozen(context, h)
    scorer = active_scorer(context, h.finetune.scorer)
    inputs = CompareInputs(
        store=h.store,
        layout=h.layout,
        cfg=h.cfg,
        finetune=h.finetune,
        prereg=prereg,
        prereg_sha256=sha256,
        scorer=scorer.name,
        rescore=rescorer(scorer),
        commit=current_commit(context.settings.repo_root),
    )
    report = build_compare(inputs)
    write_report(h.layout, report)
    verdict = report.verdict
    echo(f"verdict: {verdict.verdict} (external {verdict.external_status})")
    for reason in verdict.reasons:
        echo(f"  {reason}")
    echo(f"report: {h.layout.report_md}")


def campaign_status_lines(context: AppContext) -> list[str]:
    h = H3Context.of(context.settings)
    return status_lines(h, lambda: context.registry("scorers"))
