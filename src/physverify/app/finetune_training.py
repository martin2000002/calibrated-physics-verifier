"""The Flow-DPO runs of a campaign from the ``AppContext``: training, the beta grid and the
checkpoint selection on dev."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.finetune import (
    active_scorer,
    candidate_generator,
    require_cuda,
    require_stop_rule,
    score_prompts,
)
from physverify.app.services import scorer_for
from physverify.config.models import Settings
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import UsageError
from physverify.core.output import Echo
from physverify.finetune.candidate_scoring import rescorer
from physverify.finetune.candidates import generate_candidates
from physverify.finetune.dpo import ShiftedFlowNoise
from physverify.finetune.layout import SELECTED_FILE, CampaignStore
from physverify.finetune.pairs_report import check_current, load_pairs, require_calibrated
from physverify.finetune.prompts import campaign_prompts
from physverify.finetune.records import (
    BASE_TAG,
    CandidateScore,
    PairRecord,
    PromptRecord,
    write_record,
)
from physverify.finetune.schedule import Schedule, checkpoints
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.selection import (
    CheckpointResult,
    Selection,
    beats_base,
    choose,
    evaluate_checkpoint,
    spaced,
)
from physverify.finetune.settings import FinetuneSettings, SelectSettings, load_finetune
from physverify.finetune.stop_rule import pairs_sha256
from physverify.finetune.stored_pairs import StoredPairSource
from physverify.finetune.trainer import (
    ADAPTER_DIR,
    DpoTrainer,
    ValMetrics,
    choose_beta,
    last_val,
)
from physverify.finetune.training_run import (
    adapter_model,
    grid_schedule,
    pair_sources,
    run_beta,
    run_record,
    run_trainer,
    start_run,
    train_schedule,
)
from physverify.generation.settings import GenerationSettings
from physverify.runtime.runs import current_commit
from physverify.timing.settings import load_timing


@dataclass(frozen=True)
class TrainRequest:
    scorer: str | None
    run_id: str
    resume: bool
    force: bool
    max_steps: int | None
    beta: float | None
    allow_provisional: bool
    deviation: str | None


@dataclass(frozen=True)
class _Campaign:
    """The campaign's settings, store and the pairs of one scorer, checked before any run."""

    settings: Settings
    cfg: FinetuneSettings
    store: CampaignStore
    scorer: str
    pairs: list[PairRecord]


def _campaign(
    context: AppContext,
    scorer: str | None,
    allow_provisional: bool,
    deviation: str | None,
    *,
    err: Echo,
) -> _Campaign:
    settings = context.settings
    cfg = load_finetune(settings)
    store = CampaignStore.from_settings(settings)
    name = scorer or cfg.scorer
    pairs_ = load_pairs(store, name)
    require_calibrated(pairs_, allow_provisional)
    require_stop_rule(context, cfg, store, scorer_for(context, name), deviation, err=err)
    return _Campaign(settings, cfg, store, name, pairs_)


def _trainer(
    campaign: _Campaign,
    gen: GenerationSettings,
    sources: tuple[StoredPairSource, StoredPairSource],
    run_dir: Path,
    beta: float,
) -> DpoTrainer:
    settings, cfg = campaign.settings, campaign.cfg
    data, val = sources
    noise = ShiftedFlowNoise(gen.flow_shift, cfg.train.seed)
    model = adapter_model(gen, settings.paths, cfg, settings.runtime.device)
    return DpoTrainer(model, data, cfg.train, noise, run_dir, val=val, beta=beta)


def train_lora(
    context: AppContext,
    request: TrainRequest,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    campaign = _campaign(
        context, request.scorer, request.allow_provisional, request.deviation, err=err
    )
    settings, cfg, pairs_ = campaign.settings, campaign.cfg, campaign.pairs
    plan = train_schedule(cfg, len(pairs_), request.max_steps)
    run_dir = campaign.store.train_dir(request.run_id)
    if not request.resume and not request.force and checkpoints(run_dir):
        raise UsageError(f"{run_dir} holds checkpoints: resume, use a new --run-id or --force")
    require_cuda(settings)
    data, val, _ = pair_sources(campaign.store, campaign.scorer, settings.runtime.device)
    chosen, source = run_beta(campaign.store, cfg, request.beta, pairs_sha256(pairs_))
    gen = load_timing(settings).generation
    commit = current_commit(settings.repo_root)
    record = run_record(
        cfg, gen, commit, request.run_id, pairs_, len(val), chosen, source, plan, request.max_steps
    )
    start_run(run_dir, record, request.resume)
    trainer = _trainer(campaign, gen, (data, val), run_dir, chosen)
    report = run_trainer(trainer, request.resume, plan.total, request.force)
    echo(f"run: {run_dir} (beta {chosen:g} from {source})")
    echo(
        f"steps {report.resumed_from} -> {report.steps} (warmup {plan.warmup}, checkpoint every "
        f"{plan.checkpoint_every}); last checkpoint {report.last_checkpoint}"
    )
    if report.val is not None:
        echo(f"val implicit_acc {report.val.implicit_acc:.3f} kl_proxy {report.val.kl_proxy:.4g}")


def _grid_run(
    campaign: _Campaign,
    gen: GenerationSettings,
    sources: tuple[StoredPairSource, StoredPairSource],
    plan: Schedule,
    beta: float,
) -> ValMetrics:
    """One short run of the grid at ``beta``, resumed when it already started."""
    settings, cfg = campaign.settings, campaign.cfg
    run_id = f"beta-{beta:g}"
    run_dir = campaign.store.train_dir(run_id)
    commit = current_commit(settings.repo_root)
    val_pairs = len(sources[1])
    record = run_record(
        cfg, gen, commit, run_id, campaign.pairs, val_pairs, beta, "beta-grid", plan, plan.total
    )
    start_run(run_dir, record, True)
    trainer = _trainer(campaign, gen, sources, run_dir, beta)
    trainer.run(resume=True, max_steps=plan.total)
    del trainer
    gc.collect()
    metrics = last_val(run_dir)
    if metrics is None:
        raise UsageError(f"{run_dir}: no validation metrics")
    return metrics


def beta_grid_runs(
    context: AppContext,
    scorer: str | None,
    allow_provisional: bool,
    deviation: str | None,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    """One short run per ``finetune.train.beta_grid`` value; the beta chosen on the val pairs
    is written to the campaign's beta file."""
    campaign = _campaign(context, scorer, allow_provisional, deviation, err=err)
    cfg, store = campaign.cfg, campaign.store
    plan = grid_schedule(cfg, len(campaign.pairs))
    require_cuda(campaign.settings)
    data, val, _ = pair_sources(store, campaign.scorer, campaign.settings.runtime.device)
    if len(val) == 0:
        raise UsageError("the beta grid needs val pairs (finetune.prompts.val_fraction > 0)")
    gen = load_timing(campaign.settings).generation
    results = {}
    for value in cfg.train.beta_grid:
        metrics = _grid_run(campaign, gen, (data, val), plan, value)
        results[value] = metrics
        echo(f"beta {value:g}: implicit_acc {metrics.implicit_acc:.3f} kl {metrics.kl_proxy:.4g}")
    chosen = choose_beta(results, cfg.train.beta_kl_factor)
    payload = {
        "chosen": chosen,
        "steps": plan.total,
        "pairs_sha256": pairs_sha256(campaign.pairs),
        "results": {f"{b:g}": vars(m) for b, m in results.items()},
    }
    write_text_atomic(store.beta_path(), json.dumps(payload, indent=2, sort_keys=True))
    echo(f"chosen beta: {chosen:g} ({store.beta_path()})")


@dataclass(frozen=True)
class _DevBaseline:
    """The dev prompts, the active scorer and the base model's scores on them."""

    prompts: list[PromptRecord]
    scorer: AggregateScorer
    base: list[CandidateScore]


def _dev_baseline(
    context: AppContext, store: CampaignStore, scorer: str | None, *, echo: Echo
) -> _DevBaseline:
    active = active_scorer(context, scorer)
    dev = campaign_prompts(store, "dev")
    generate_candidates(candidate_generator(context, store, None), dev, None, None, echo=echo)
    base = score_prompts(context, store, BASE_TAG, dev, active, echo=echo)
    check_current(store, BASE_TAG, base, active, "dev")
    return _DevBaseline(dev, active, base)


def _score_checkpoint(
    context: AppContext,
    store: CampaignStore,
    step: int,
    adapter: Path,
    baseline: _DevBaseline,
    select: SelectSettings,
    *,
    echo: Echo,
) -> CheckpointResult:
    dev, active = baseline.prompts, baseline.scorer
    generator = candidate_generator(context, store, adapter)
    generate_candidates(generator, dev, None, None, echo=echo)
    scores = score_prompts(context, store, generator.model_tag, dev, active, echo=echo)
    check_current(store, generator.model_tag, scores, active, "dev")
    result = evaluate_checkpoint(
        step, str(adapter), generator.model_tag, scores, baseline.base, select, rescorer(active)
    )
    del generator
    gc.collect()
    return result


def _report_selection(selection: Selection, path: Path, *, echo: Echo, err: Echo) -> None:
    for result in selection.checkpoints:
        flag = " suspected_hacking" if result.suspected_hacking else ""
        echo(f"step {result.step}: paired diff {result.paired_diff} ci {result.ci}{flag}")
    chosen = selection.selected
    echo(f"selected: {chosen.adapter if chosen else 'none'} ({path})")
    if not selection.beats_base:
        err("warning: no checkpoint beats the base on dev (paired diff <= 0 or its CI holds 0)")


def select_checkpoint(
    context: AppContext,
    scorer: str | None,
    run_id: str,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    """Scores dev for spaced checkpoints of the run against the base and records the pick."""
    cfg = load_finetune(context.settings)
    store = CampaignStore.from_settings(context.settings)
    found = dict(checkpoints(store.train_dir(run_id)))
    if not found:
        raise UsageError(f"{store.train_dir(run_id)} has no checkpoints: run 'finetune train'")
    require_cuda(context.settings)
    baseline = _dev_baseline(context, store, scorer, echo=echo)
    results = [
        _score_checkpoint(
            context, store, step, found[step] / ADAPTER_DIR, baseline, cfg.select, echo=echo
        )
        for step in spaced(list(found), cfg.select.checkpoints)
    ]
    chosen = choose(results)
    selection = Selection(
        run_id=run_id,
        scorer=baseline.scorer.name,
        thresholds=baseline.scorer.thresholds(),
        selected=chosen,
        checkpoints=results,
        beats_base=beats_base(chosen),
    )
    path = store.select_dir(run_id) / SELECTED_FILE
    write_record(path, selection)
    _report_selection(selection, path, echo=echo, err=err)
