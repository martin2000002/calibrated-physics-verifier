"""The Flow-DPO runs of a campaign from the ``AppContext``: training, the beta grid and the
checkpoint selection on dev."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass

from physverify.app.context import AppContext
from physverify.app.finetune import (
    active_scorer,
    candidate_generator,
    require_cuda,
    require_stop_rule,
    score_prompts,
)
from physverify.app.services import scorer_for
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import UsageError
from physverify.core.output import Echo
from physverify.finetune.candidate_scoring import rescorer
from physverify.finetune.candidates import generate_candidates
from physverify.finetune.dpo import ShiftedFlowNoise
from physverify.finetune.layout import SELECTED_FILE, CampaignStore
from physverify.finetune.pairs_report import check_current, load_pairs, require_calibrated
from physverify.finetune.prompts import campaign_prompts
from physverify.finetune.records import BASE_TAG, write_record
from physverify.finetune.schedule import checkpoints
from physverify.finetune.selection import (
    Selection,
    beats_base,
    choose,
    evaluate_checkpoint,
    spaced,
)
from physverify.finetune.settings import load_finetune
from physverify.finetune.stop_rule import pairs_sha256
from physverify.finetune.trainer import ADAPTER_DIR, DpoTrainer, choose_beta, last_val
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


def train_lora(
    context: AppContext,
    request: TrainRequest,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    settings = context.settings
    cfg = load_finetune(settings)
    store = CampaignStore.from_settings(settings)
    name = request.scorer or cfg.scorer
    pairs_ = load_pairs(store, name)
    require_calibrated(pairs_, request.allow_provisional)
    require_stop_rule(context, cfg, store, scorer_for(context, name), request.deviation, err=err)
    plan = train_schedule(cfg, len(pairs_), request.max_steps)
    run_dir = store.train_dir(request.run_id)
    if not request.resume and not request.force and checkpoints(run_dir):
        raise UsageError(f"{run_dir} holds checkpoints: resume, use a new --run-id or --force")
    require_cuda(settings)
    data, val, _ = pair_sources(store, name, settings.runtime.device)
    chosen, source = run_beta(store, cfg, request.beta, pairs_sha256(pairs_))
    gen = load_timing(settings).generation
    commit = current_commit(settings.repo_root)
    record = run_record(
        cfg, gen, commit, request.run_id, pairs_, len(val), chosen, source, plan, request.max_steps
    )
    start_run(run_dir, record, request.resume)
    noise = ShiftedFlowNoise(gen.flow_shift, cfg.train.seed)
    model = adapter_model(gen, settings.paths, cfg, settings.runtime.device)
    trainer = DpoTrainer(model, data, cfg.train, noise, run_dir, val=val, beta=chosen)
    report = run_trainer(trainer, request.resume, plan.total, request.force)
    echo(f"run: {run_dir} (beta {chosen:g} from {source})")
    echo(
        f"steps {report.resumed_from} -> {report.steps} (warmup {plan.warmup}, checkpoint every "
        f"{plan.checkpoint_every}); last checkpoint {report.last_checkpoint}"
    )
    if report.val is not None:
        echo(f"val implicit_acc {report.val.implicit_acc:.3f} kl_proxy {report.val.kl_proxy:.4g}")


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
    settings = context.settings
    cfg = load_finetune(settings)
    store = CampaignStore.from_settings(settings)
    name = scorer or cfg.scorer
    pairs_ = load_pairs(store, name)
    require_calibrated(pairs_, allow_provisional)
    require_stop_rule(context, cfg, store, scorer_for(context, name), deviation, err=err)
    plan = grid_schedule(cfg, len(pairs_))
    require_cuda(settings)
    data, val, _ = pair_sources(store, name, settings.runtime.device)
    if len(val) == 0:
        raise UsageError("the beta grid needs val pairs (finetune.prompts.val_fraction > 0)")
    gen = load_timing(settings).generation
    results = {}
    for value in cfg.train.beta_grid:
        run_id = f"beta-{value:g}"
        run_dir = store.train_dir(run_id)
        commit = current_commit(settings.repo_root)
        record = run_record(
            cfg, gen, commit, run_id, pairs_, len(val), value, "beta-grid", plan, plan.total
        )
        start_run(run_dir, record, True)
        noise = ShiftedFlowNoise(gen.flow_shift, cfg.train.seed)
        model = adapter_model(gen, settings.paths, cfg, settings.runtime.device)
        trainer = DpoTrainer(model, data, cfg.train, noise, run_dir, val=val, beta=value)
        trainer.run(resume=True, max_steps=plan.total)
        metrics = last_val(run_dir)
        if metrics is None:
            raise UsageError(f"{run_dir}: no validation metrics")
        results[value] = metrics
        echo(f"beta {value:g}: implicit_acc {metrics.implicit_acc:.3f} kl {metrics.kl_proxy:.4g}")
        del trainer
        gc.collect()
    chosen = choose_beta(results, cfg.train.beta_kl_factor)
    payload = {
        "chosen": chosen,
        "steps": plan.total,
        "pairs_sha256": pairs_sha256(pairs_),
        "results": {f"{b:g}": vars(m) for b, m in results.items()},
    }
    write_text_atomic(store.beta_path(), json.dumps(payload, indent=2, sort_keys=True))
    echo(f"chosen beta: {chosen:g} ({store.beta_path()})")


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
    active = active_scorer(context, scorer)
    dev = campaign_prompts(store, "dev")
    generate_candidates(candidate_generator(context, store, None), dev, None, None, echo=echo)
    base = score_prompts(context, store, BASE_TAG, dev, active, echo=echo)
    check_current(store, BASE_TAG, base, active, "dev")
    results = []
    for step in spaced(list(found), cfg.select.checkpoints):
        adapter = found[step] / ADAPTER_DIR
        generator = candidate_generator(context, store, adapter)
        generate_candidates(generator, dev, None, None, echo=echo)
        scores = score_prompts(context, store, generator.model_tag, dev, active, echo=echo)
        check_current(store, generator.model_tag, scores, active, "dev")
        results.append(
            evaluate_checkpoint(
                step,
                str(adapter),
                generator.model_tag,
                scores,
                base,
                cfg.select,
                rescorer(active),
            )
        )
        del generator
        gc.collect()
    chosen = choose(results)
    selection = Selection(
        run_id=run_id,
        scorer=active.name,
        thresholds=active.thresholds(),
        selected=chosen,
        checkpoints=results,
        beats_base=beats_base(chosen),
    )
    path = store.select_dir(run_id) / SELECTED_FILE
    write_record(path, selection)
    for result in results:
        flag = " suspected_hacking" if result.suspected_hacking else ""
        echo(f"step {result.step}: paired diff {result.paired_diff} ci {result.ci}{flag}")
    echo(f"selected: {chosen.adapter if chosen else 'none'} ({path})")
    if not selection.beats_base:
        err("warning: no checkpoint beats the base on dev (paired diff <= 0 or its CI holds 0)")
