"""``physverify finetune``: preference pairs from the verifier and Flow-DPO LoRA (ADR 0012, 3.10).

Every step resumes: ``generate`` skips complete candidates, ``score`` the ones with a current
score (same scorer stamp, same candidate ``config_hash``), ``train`` starts from the last
checkpoint. The GPU subcommands exit 2 without ``--device cuda`` and a CUDA device,
before they write anything.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Final

import typer

from physverify.cli.commands._finetune_deps import candidate_generator, require_cuda, scoring_deps
from physverify.cli.commands.timing import _timing_directory
from physverify.cli.context import AppContext, app_context, heavy_command
from physverify.core.errors import EXIT_ERROR, UsageError
from physverify.diagnosis.atomic_io import write_text_atomic
from physverify.evaluation.arms import EVAL_SPLIT
from physverify.finetune.budget import (
    PAIR_MEASURE,
    FinetuneBudgetQuery,
    budget_lines,
    train_prompts,
)
from physverify.finetune.dpo import ShiftedFlowNoise
from physverify.finetune.generate import CandidateGenerator
from physverify.finetune.pairs import build_pairs
from physverify.finetune.pilot import fit_and_val, pilot_prompts
from physverify.finetune.prompts import build_prompt_set, sizing_labels
from physverify.finetune.records import (
    BASE_TAG,
    CandidateScore,
    PairRecord,
    PairsReport,
    PromptRecord,
    read_record,
    read_records,
    write_record,
    write_records,
)
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scoring import build_scorer, rescorer, score_candidates, stale_scores
from physverify.finetune.seeds import parse_shard
from physverify.finetune.selection import (
    CheckpointResult,
    Selection,
    choose,
    evaluate_checkpoint,
    spaced,
)
from physverify.finetune.settings import (
    SPLITS,
    FinetuneSettings,
    PromptSplit,
    TrainSettings,
    generator_settings,
    load_finetune,
)
from physverify.finetune.store import (
    PAIRS_FILE,
    REPORT_FILE,
    RUN_FILE,
    SELECTED_FILE,
    VAL_PAIRS_FILE,
    VAL_REPORT_FILE,
    CampaignStore,
)
from physverify.finetune.trainer import (
    ADAPTER_DIR,
    DpoTrainer,
    ExistingRunError,
    Schedule,
    ScheduleError,
    StoredPairSource,
    beta_grid_steps,
    checkpoints,
    choose_beta,
    last_val,
    require_checkpoints,
    schedule,
    stored_pairs,
)
from physverify.finetune.wan_lora import load_wan_adapter_model
from physverify.runtime.runs import current_commit
from physverify.timing.budget import BudgetError, cost_from_results
from physverify.timing.generation import snapshot_dir, snapshot_problem
from physverify.timing.report import load_results, load_samples
from physverify.timing.settings import load_timing
from physverify.timing.stats import across_medians, stat_rows

NAME = "finetune"
OWNER = "A11"
IMPLEMENTED = True
DEFAULT_RUN: Final = "main"
SIGMA_LEVELS: Final = (0.1, 0.25, 0.5, 0.75, 0.9)
PILOT_PROMPTS: Final = 20
RESUME_IGNORED: Final = frozenset({"commit", "beta_source"})
SHOWN_STALE: Final = 5
BASE_MODEL_FIELDS: Final = ("model_ref", "revision", "sha256", "dtype")

app = typer.Typer(no_args_is_help=True, help="Preference pairs and Flow-DPO LoRA of Wan2.1.")
prompts_app = typer.Typer(no_args_is_help=True, help="The prompt set (prompts.jsonl).")
app.add_typer(prompts_app, name="prompts")

SplitOption = Annotated[str, typer.Option(help="train or dev (eval belongs to `h3`).")]
ScorerOption = Annotated[
    str | None, typer.Option(help="calibrated or provisional (default: finetune.scorer).")
]
ShardOption = Annotated[str | None, typer.Option(help="Shard i/n of the candidates.")]
RunIdOption = Annotated[str, typer.Option(help="Training run id.")]


def _cfg(context: AppContext) -> FinetuneSettings:
    return load_finetune(context.settings)


def _store(context: AppContext) -> CampaignStore:
    return CampaignStore.from_settings(context.settings)


def parse_split(split: str) -> PromptSplit:
    for known in SPLITS:
        if split == known:
            return known
    raise UsageError(f"--split must be one of {list(SPLITS)}, got {split!r}")


def campaign_split(split: str, h3_step: str) -> PromptSplit:
    """A train or dev split; the eval candidates and scores are H3's (ADR 0013), whose
    generation stamp and pre-registration a finetune run would break."""
    parsed = parse_split(split)
    if parsed == EVAL_SPLIT:
        raise UsageError(
            f"the eval split belongs to H3: run `physverify h3 {h3_step} --arm base|ft`"
        )
    return parsed


def campaign_prompts(store: CampaignStore, split: PromptSplit | None = None) -> list[PromptRecord]:
    if not store.prompts_path.is_file():
        raise UsageError(f"{store.prompts_path} does not exist: run 'finetune prompts build'")
    prompts = store.prompts()
    return [p for p in prompts if split is None or p.split == split]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@prompts_app.command("build", help="Write prompts.jsonl (or with --check, verify it).")
def prompts_build(
    check: Annotated[bool, typer.Option(help="Only verify the existing file.")] = False,
) -> None:
    context = app_context()
    store = _store(context)
    records = build_prompt_set(_cfg(context).prompts, sizing_labels(context.settings))
    text = "".join(record.model_dump_json() + "\n" for record in records)
    path = store.prompts_path
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    if check:
        if current != text:
            typer.echo(f"{path} differs from the configuration", err=True)
            raise typer.Exit(code=EXIT_ERROR)
    elif current != text:
        if current is not None and store.model_tags():
            raise UsageError(
                f"{path} differs from the configuration and candidates exist: "
                "use a new finetune.campaign"
            )
        write_records(path, records)
    typer.echo(f"prompts: {path} sha256 {_sha256(path)}")
    for split in SPLITS:
        typer.echo(f"  {split}: {sum(r.split == split for r in records)}")


def _generate(
    generator: CandidateGenerator,
    prompts: Sequence[PromptRecord],
    shard: str | None,
    limit: int | None,
) -> int:
    plan = generator.plan(prompts, parse_shard(shard))
    plan = plan if limit is None else plan[:limit]
    report = generator.run(plan, {p.prompt_id: p for p in prompts})
    typer.echo(f"{generator.model_tag}: generated {len(report.generated)} candidates")
    return len(report.generated)


@app.command("generate", help="Generate the candidates of a split (resumes).")
@heavy_command
def generate(
    split: SplitOption,
    adapter: Annotated[Path | None, typer.Option(help="LoRA adapter directory.")] = None,
    shard: ShardOption = None,
    limit: Annotated[int | None, typer.Option(min=1, help="At most N candidates.")] = None,
    pilot: Annotated[
        int | None,
        typer.Option(min=1, help="Only N fit prompts of train, spread across the families."),
    ] = None,
) -> None:
    which = campaign_split(split, "generate")
    if pilot is not None and which != "train":
        raise UsageError("--pilot draws train prompts: use it with --split train")
    context = app_context()
    require_cuda(context.settings)
    store = _store(context)
    prompts = campaign_prompts(store, which)
    if pilot is not None:
        prompts = pilot_prompts(prompts, pilot)
    _generate(candidate_generator(context, store, adapter), prompts, shard, limit)


def _scorer(context: AppContext, name: str | None) -> AggregateScorer:
    scorer = build_scorer(name or _cfg(context).scorer, context.settings)
    reason = scorer.unavailable()
    if reason is not None:
        raise UsageError(f"scorer {scorer.name} is unavailable: {reason}")
    return scorer


def _score(
    context: AppContext,
    store: CampaignStore,
    model_tag: str,
    prompts: Sequence[PromptRecord],
    scorer: AggregateScorer,
    shard: str | None = None,
) -> list[CandidateScore]:
    scores = score_candidates(
        store, model_tag, prompts, scorer, scoring_deps(context), parse_shard(shard)
    )
    wanted = {p.prompt_id for p in prompts}
    selected = [s for s in scores if s.prompt_id in wanted]
    abstained = sum(s.score is None for s in selected)
    typer.echo(f"{model_tag}/{scorer.name}: {len(selected)} scored, {abstained} abstained")
    return selected


@app.command("score", help="Extract, verify and score the candidates of a model tag (resumes).")
@heavy_command
def score(
    split: SplitOption,
    model_tag: Annotated[str, typer.Option(help="base or lora-<hash>.")] = BASE_TAG,
    scorer: ScorerOption = None,
    shard: ShardOption = None,
) -> None:
    which = campaign_split(split, "score")
    context = app_context()
    require_cuda(context.settings)
    store = _store(context)
    prompts = campaign_prompts(store, which)
    _score(context, store, model_tag, prompts, _scorer(context, scorer), shard)


def _check_current(
    store: CampaignStore,
    model_tag: str,
    scores: Sequence[CandidateScore],
    scorer: AggregateScorer,
    split: PromptSplit,
) -> None:
    """Refuses scores of another scorer stamp or of another candidate config_hash."""
    stale = stale_scores(scores, scorer, store.candidates(model_tag))
    if stale:
        raise UsageError(
            f"{len(stale)} {model_tag} scores are stale (another {scorer.name} version or "
            f"thresholds, or another candidate configuration; e.g. {stale[:SHOWN_STALE]}): run "
            f"'finetune score --split {split} --model-tag {model_tag}' to score them again"
        )


def _pairs_of(
    prompts: Sequence[PromptRecord],
    scores: Sequence[CandidateScore],
    context: AppContext,
    scorer: AggregateScorer,
) -> tuple[list[PairRecord], PairsReport | None]:
    wanted = {p.prompt_id for p in prompts}
    selected = [s for s in scores if s.prompt_id in wanted]
    if not selected:
        return [], None
    return build_pairs(prompts, selected, _cfg(context).pairs, rescorer(scorer))


def _report_lines(label: str, report: PairsReport) -> list[str]:
    return [
        f"{label}: {report.pairs} pairs; {report.yield_per_scored_prompt:.3f} per scored prompt "
        f"({report.scored_prompts} scored), {report.yield_per_prompt:.3f} per prompt "
        f"({report.prompts}); {report.eligible}/{report.candidates} candidates eligible",
        f"  dropped: {json.dumps(report.dropped, sort_keys=True)}",
        f"  rejected: {json.dumps(report.rejected, sort_keys=True)}",
    ]


@app.command("pairs", help="Winner-loser pairs of the train prompts (val prompts kept apart).")
def pairs(scorer: ScorerOption = None) -> None:
    context = app_context()
    store = _store(context)
    active = _scorer(context, scorer)
    train = campaign_prompts(store, "train")
    train_ids = {p.prompt_id for p in train}
    scores = [s for s in store.scores(BASE_TAG, active.name) if s.prompt_id in train_ids]
    if not scores:
        raise UsageError(
            f"no {active.name} scores of {BASE_TAG}: run 'finetune score --split train'"
        )
    _check_current(store, BASE_TAG, scores, active, "train")
    fit, held = fit_and_val(train)
    directory = store.pairs_dir(active.name)
    outputs = (
        (fit, PAIRS_FILE, REPORT_FILE, "train"),
        (held, VAL_PAIRS_FILE, VAL_REPORT_FILE, "val"),
    )
    for prompts, pairs_file, report_file, label in outputs:
        records, report = _pairs_of(prompts, scores, context, active)
        write_records(directory / pairs_file, records)
        if report is not None:
            write_record(directory / report_file, report)
            for line in _report_lines(label, report):
                typer.echo(line)
    typer.echo(f"pairs: {directory}")


def load_pairs(store: CampaignStore, scorer: str, name: str = PAIRS_FILE) -> list[PairRecord]:
    path = store.pairs_dir(scorer) / name
    if not path.is_file():
        raise UsageError(f"{path} does not exist: run 'finetune pairs'")
    return read_records(path, PairRecord)


def require_calibrated(pairs: Sequence[PairRecord], allow_provisional: bool) -> None:
    if not pairs:
        raise UsageError("there are no training pairs")
    if not allow_provisional and any(not p.calibrated for p in pairs):
        raise UsageError(
            "the pairs come from an uncalibrated scorer; pass --allow-provisional to train on them"
        )


def _sources(
    context: AppContext, store: CampaignStore, scorer: str
) -> tuple[StoredPairSource, StoredPairSource, list[PairRecord]]:
    candidates = store.candidates(BASE_TAG)
    train = load_pairs(store, scorer)
    val = load_pairs(store, scorer, VAL_PAIRS_FILE)
    device = context.settings.runtime.device
    try:
        sources = (
            StoredPairSource(stored_pairs(train, candidates), device),
            StoredPairSource(stored_pairs(val, candidates), device),
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return sources[0], sources[1], train


def _model(context: AppContext) -> Any:
    settings = context.settings
    gen = generator_settings(settings)
    problem = snapshot_problem(gen, settings.paths)
    if problem is not None:
        raise UsageError(f"Wan2.1 weights: {problem}")
    cfg = _cfg(context)
    return load_wan_adapter_model(
        snapshot_dir(gen, settings.paths),
        gen,
        cfg.lora,
        settings.runtime.device,
        cfg.train.seed,
        cfg.train.gradient_checkpointing,
    )


def _pairs_hash(pairs: Sequence[PairRecord]) -> str:
    text = "".join(p.model_dump_json() + "\n" for p in pairs)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _beta(
    store: CampaignStore, cfg: FinetuneSettings, beta: float | None, pairs_sha256: str
) -> tuple[float, str]:
    """``--beta``, else the beta grid's choice on these very pairs, else the configured beta."""
    if beta is not None:
        return beta, "--beta"
    path = store.beta_path()
    if path.is_file():
        grid = json.loads(path.read_text(encoding="utf-8"))
        if grid.get("pairs_sha256") != pairs_sha256:
            raise UsageError(
                f"{path} was chosen on other pairs: run 'finetune beta-grid' again or pass --beta"
            )
        return float(grid["chosen"]), str(path)
    return cfg.train.beta, "finetune.train.beta"


def run_record(
    context: AppContext,
    run_id: str,
    pairs: Sequence[PairRecord],
    n_val: int,
    beta: float,
    beta_source: str,
    plan: Schedule,
    max_steps: int | None,
) -> dict[str, Any]:
    cfg = _cfg(context)
    gen = generator_settings(context.settings)
    sigma = ShiftedFlowNoise(gen.flow_shift).quantiles(SIGMA_LEVELS)
    return {
        "run_id": run_id,
        **pairs[0].model_dump(include={"scorer", "scorer_version", "calibrated", "thresholds"}),
        "pairs_sha256": _pairs_hash(pairs),
        "n_pairs": len(pairs),
        "n_val_pairs": n_val,
        "beta": beta,
        "beta_source": beta_source,
        "max_steps": max_steps,
        "steps": plan.total,
        "warmup_steps": plan.warmup,
        "checkpoint_every": plan.checkpoint_every,
        "flow_shift": gen.flow_shift,
        "base_model": gen.model_dump(mode="json", include=set(BASE_MODEL_FIELDS)),
        "sigma_quantiles": sigma,
        "lora": cfg.lora.model_dump(mode="json"),
        "train": cfg.train.model_dump(mode="json"),
        "commit": current_commit(context.settings.repo_root),
    }


def _check_resume(path: Path, record: dict[str, Any]) -> None:
    """Every field of run.json but the commit and the beta's source must be unchanged: the pairs,
    the base model, the whole TrainSettings and LoRA dumps and the resolved schedule."""
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    keys = (previous.keys() | record.keys()) - RESUME_IGNORED
    changed = sorted(k for k in keys if previous.get(k) != record.get(k))
    if changed:
        raise UsageError(f"{path}: {changed} changed since this run started; use a new --run-id")


def _planned(cfg: TrainSettings, n_pairs: int, max_steps: int | None = None) -> Schedule:
    try:
        return schedule(cfg, n_pairs, max_steps)
    except ScheduleError as exc:
        raise UsageError(str(exc)) from exc


def train_schedule(cfg: FinetuneSettings, n_pairs: int, max_steps: int | None) -> Schedule:
    """The main run's schedule, refused (exit 2) when select would have too few checkpoints."""
    plan = _planned(cfg.train, n_pairs, max_steps)
    try:
        require_checkpoints(plan, cfg.train.keep_last, cfg.select.checkpoints)
    except ScheduleError as exc:
        raise UsageError(str(exc)) from exc
    return plan


def grid_schedule(cfg: FinetuneSettings, n_pairs: int) -> Schedule:
    """One beta-grid run: ``beta_grid_fraction`` of the main run's steps."""
    main = _planned(cfg.train, n_pairs)
    try:
        steps = beta_grid_steps(cfg.train, main.total)
    except ScheduleError as exc:
        raise UsageError(str(exc)) from exc
    return _planned(cfg.train, n_pairs, steps)


def _start(run_dir: Path, record: dict[str, Any], resume: bool) -> None:
    if resume:
        _check_resume(run_dir / RUN_FILE, record)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(run_dir / RUN_FILE, json.dumps(record, indent=2, sort_keys=True))


def _run(trainer: DpoTrainer, resume: bool, max_steps: int | None, force: bool) -> Any:
    try:
        return trainer.run(resume=resume, max_steps=max_steps, force=force)
    except ExistingRunError as exc:
        raise UsageError(f"{exc} (--force)") from exc


@app.command("train", help="Flow-DPO LoRA on the stored pairs (resumes from the last checkpoint).")
@heavy_command
def train(
    scorer: ScorerOption = None,
    run_id: RunIdOption = DEFAULT_RUN,
    resume: Annotated[bool, typer.Option(help="Resume from the last checkpoint.")] = True,
    force: Annotated[
        bool, typer.Option(help="With --no-resume, delete the run's existing checkpoints.")
    ] = False,
    max_steps: Annotated[int | None, typer.Option(min=1, help="Stop after N steps.")] = None,
    beta: Annotated[
        float | None, typer.Option(help="DPO beta (default: beta-grid or config).")
    ] = None,
    allow_provisional: Annotated[
        bool, typer.Option(help="Train on pairs of an uncalibrated scorer.")
    ] = False,
) -> None:
    context = app_context()
    cfg = _cfg(context)
    store = _store(context)
    name = scorer or cfg.scorer
    pairs_ = load_pairs(store, name)
    require_calibrated(pairs_, allow_provisional)
    plan = train_schedule(cfg, len(pairs_), max_steps)
    run_dir = store.train_dir(run_id)
    if not resume and not force and checkpoints(run_dir):
        raise UsageError(f"{run_dir} holds checkpoints: resume, use a new --run-id or --force")
    require_cuda(context.settings)
    data, val, _ = _sources(context, store, name)
    chosen, source = _beta(store, cfg, beta, _pairs_hash(pairs_))
    record = run_record(context, run_id, pairs_, len(val), chosen, source, plan, max_steps)
    _start(run_dir, record, resume)
    gen = generator_settings(context.settings)
    noise = ShiftedFlowNoise(gen.flow_shift, cfg.train.seed)
    trainer = DpoTrainer(_model(context), data, cfg.train, noise, run_dir, val=val, beta=chosen)
    report = _run(trainer, resume, plan.total, force)
    typer.echo(f"run: {run_dir} (beta {chosen:g} from {source})")
    typer.echo(
        f"steps {report.resumed_from} -> {report.steps} (warmup {plan.warmup}, checkpoint every "
        f"{plan.checkpoint_every}); last checkpoint {report.last_checkpoint}"
    )
    if report.val is not None:
        typer.echo(
            f"val implicit_acc {report.val.implicit_acc:.3f} kl_proxy {report.val.kl_proxy:.4g}"
        )


@app.command("beta-grid", help="Short runs over finetune.train.beta_grid, chosen on val pairs.")
@heavy_command
def beta_grid(
    scorer: ScorerOption = None,
    allow_provisional: Annotated[
        bool, typer.Option(help="Train on pairs of an uncalibrated scorer.")
    ] = False,
) -> None:
    context = app_context()
    cfg = _cfg(context)
    store = _store(context)
    name = scorer or cfg.scorer
    pairs_ = load_pairs(store, name)
    require_calibrated(pairs_, allow_provisional)
    plan = grid_schedule(cfg, len(pairs_))
    require_cuda(context.settings)
    data, val, _ = _sources(context, store, name)
    if len(val) == 0:
        raise UsageError("the beta grid needs val pairs (finetune.prompts.val_fraction > 0)")
    gen = generator_settings(context.settings)
    results = {}
    for value in cfg.train.beta_grid:
        run_id = f"beta-{value:g}"
        run_dir = store.train_dir(run_id)
        record = run_record(context, run_id, pairs_, len(val), value, "beta-grid", plan, plan.total)
        _start(run_dir, record, True)
        noise = ShiftedFlowNoise(gen.flow_shift, cfg.train.seed)
        trainer = DpoTrainer(_model(context), data, cfg.train, noise, run_dir, val=val, beta=value)
        trainer.run(resume=True, max_steps=plan.total)
        metrics = last_val(run_dir)
        if metrics is None:
            raise UsageError(f"{run_dir}: no validation metrics")
        results[value] = metrics
        typer.echo(
            f"beta {value:g}: implicit_acc {metrics.implicit_acc:.3f} kl {metrics.kl_proxy:.4g}"
        )
        del trainer
        gc.collect()
    chosen = choose_beta(results, cfg.train.beta_kl_factor)
    payload = {
        "chosen": chosen,
        "steps": plan.total,
        "pairs_sha256": _pairs_hash(pairs_),
        "results": {f"{b:g}": vars(m) for b, m in results.items()},
    }
    write_text_atomic(store.beta_path(), json.dumps(payload, indent=2, sort_keys=True))
    typer.echo(f"chosen beta: {chosen:g} ({store.beta_path()})")


@app.command("select", help="Score dev for spaced checkpoints against the base; pick one.")
@heavy_command
def select(scorer: ScorerOption = None, run_id: RunIdOption = DEFAULT_RUN) -> None:
    context = app_context()
    cfg = _cfg(context)
    store = _store(context)
    found = dict(checkpoints(store.train_dir(run_id)))
    if not found:
        raise UsageError(f"{store.train_dir(run_id)} has no checkpoints: run 'finetune train'")
    require_cuda(context.settings)
    active = _scorer(context, scorer)
    dev = campaign_prompts(store, "dev")
    _generate(candidate_generator(context, store, None), dev, None, None)
    base = _score(context, store, BASE_TAG, dev, active)
    _check_current(store, BASE_TAG, base, active, "dev")
    results = []
    for step in spaced(list(found), cfg.select.checkpoints):
        adapter = found[step] / ADAPTER_DIR
        generator = candidate_generator(context, store, adapter)
        _generate(generator, dev, None, None)
        scores = _score(context, store, generator.model_tag, dev, active)
        _check_current(store, generator.model_tag, scores, active, "dev")
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
    selection = Selection(
        run_id=run_id,
        scorer=active.name,
        thresholds=active.thresholds(),
        selected=choose(results),
        checkpoints=results,
    )
    path = store.select_dir(run_id) / SELECTED_FILE
    write_record(path, selection)
    for result in results:
        flag = " suspected_hacking" if result.suspected_hacking else ""
        typer.echo(f"step {result.step}: paired diff {result.paired_diff} ci {result.ci}{flag}")
    chosen = selection.selected
    typer.echo(f"selected: {chosen.adapter if chosen else 'none'} ({path})")
    if not beats_base(chosen):
        typer.echo(
            "warning: no checkpoint beats the base on dev (paired diff <= 0 or its CI holds 0)",
            err=True,
        )


def beats_base(chosen: CheckpointResult | None) -> bool:
    """The chosen checkpoint's paired difference with the base is positive and its CI excludes 0."""
    if chosen is None or chosen.paired_diff is None or chosen.paired_diff <= 0:
        return False
    return chosen.ci is not None and chosen.ci[0] > 0


def _yield(store: CampaignStore, scorer: str, given: float | None) -> tuple[float, bool]:
    path = store.pairs_dir(scorer) / REPORT_FILE
    if given is not None:
        return given, False
    if path.is_file():
        return read_record(path, PairsReport).yield_per_scored_prompt, True
    raise UsageError(f"no measured pair yield ({path} missing): pass --pair-yield")


@app.command("budget", help="Training prompts that fit in a GPU budget (valid timing runs only).")
def budget(
    timing_run: Annotated[Path, typer.Option(help="Timing run directory with dpo_step.")],
    gpu_hours: Annotated[float, typer.Option(help="GPU hours available (H).")],
    pair_yield: Annotated[
        float | None, typer.Option(help="Pairs per prompt (default: measured in report.json).")
    ] = None,
    extract_batch: Annotated[
        int | None, typer.Option(help="Clips per extraction process (B_x).")
    ] = None,
    generator_only: Annotated[
        bool, typer.Option(help="Cost = generation only (extraction runs elsewhere).")
    ] = False,
    pilot_prompts: Annotated[int, typer.Option(min=0, help="Pilot prompts still to run.")] = (
        PILOT_PROMPTS
    ),
    scorer: ScorerOption = None,
) -> None:
    context = app_context()
    settings = context.settings
    cfg = _cfg(context)
    timing = load_timing(settings)
    directory = _timing_directory(timing_run)
    rows = stat_rows(load_samples(directory), timing.max_iqr_rel)
    cost = cost_from_results(load_results(directory), rows, timing.total_measures, 1)
    pair_s = across_medians(rows).get(PAIR_MEASURE)
    if pair_s is None:
        raise BudgetError(f"the timing run has no {PAIR_MEASURE}: run the dpo_step target")
    y, measured = _yield(_store(context), scorer or cfg.scorer, pair_yield)
    query = FinetuneBudgetQuery(
        gpu_hours=gpu_hours,
        k=cfg.candidates.k_train,
        k_eval_base=cfg.candidates.k_eval,
        k_eval_ft=cfg.candidates.k_eval_finetuned,
        n_eval=cfg.prompts.n_eval + len(cfg.prompts.fixed_eval),
        n_dev=cfg.prompts.n_dev,
        k_dev=cfg.candidates.k_dev,
        dev_checkpoints=cfg.select.checkpoints,
        epochs=cfg.train.epochs,
        pair_yield=y,
        yield_measured=measured,
        n_pilot=pilot_prompts,
        n_beta=len(cfg.train.beta_grid),
        beta_fraction=cfg.train.beta_grid_fraction,
    )
    cost_s = cost.generator_only() if generator_only else cost.sequential(extract_batch)
    for line in budget_lines(train_prompts(cost, cost_s, pair_s, query)):
        typer.echo(line)


def status_lines(store: CampaignStore) -> list[str]:
    if not store.prompts_path.is_file():
        return [f"campaign {store.campaign}: no prompts.jsonl yet"]
    prompts = store.prompts()
    lines = [f"campaign {store.campaign}: {len(prompts)} prompts at {store.root}"]
    for tag in store.model_tags():
        records = store.candidates(tag)
        by_split = {s: sum(r.split == s for r in records) for s in SPLITS}
        lines.append(f"  {tag}: candidates {json.dumps(by_split)}")
        scores_root = store.root / "scores" / tag
        for directory in sorted(scores_root.iterdir()) if scores_root.is_dir() else []:
            scores = store.scores(tag, directory.name)
            abstained = sum(s.score is None for s in scores)
            lines.append(f"    {directory.name}: {len(scores)} scored, {abstained} abstained")
    pairs_root = store.root / "pairs"
    for directory in sorted(pairs_root.iterdir()) if pairs_root.is_dir() else []:
        report = directory / REPORT_FILE
        if report.is_file():
            parsed = read_record(report, PairsReport)
            lines.append(
                f"  pairs/{directory.name}: {parsed.pairs} "
                f"({parsed.yield_per_scored_prompt:.3f}/scored prompt, "
                f"{parsed.yield_per_prompt:.3f}/prompt)"
            )
    for run in store.train_runs():
        found = checkpoints(store.train_dir(run))
        last = found[-1][0] if found else 0
        selected = store.select_dir(run) / SELECTED_FILE
        mark = ", selected" if selected.is_file() else ""
        lines.append(f"  train/{run}: last checkpoint step {last}{mark}")
    return lines


@app.command("status", help="Candidates, scores, pairs and training steps of the campaign.")
def status() -> None:
    for line in status_lines(_store(app_context())):
        typer.echo(line)
