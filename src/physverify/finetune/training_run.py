"""One Flow-DPO training run on disk: its ``run.json`` record, resume check, schedules, stored
pair sources, adapter model and beta."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from physverify.config.models import PathsSettings
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import UsageError
from physverify.finetune.dpo import SIGMA_LEVELS, ShiftedFlowNoise
from physverify.finetune.layout import VAL_PAIRS_FILE, CampaignStore
from physverify.finetune.pairs_report import load_pairs
from physverify.finetune.records import BASE_TAG, PairRecord
from physverify.finetune.schedule import (
    Schedule,
    ScheduleError,
    beta_grid_steps,
    require_checkpoints,
    schedule,
)
from physverify.finetune.settings import FinetuneSettings, TrainSettings
from physverify.finetune.stop_rule import pairs_sha256
from physverify.finetune.stored_pairs import StoredPairSource, stored_pairs
from physverify.finetune.trainer import DpoTrainer, ExistingRunError
from physverify.finetune.wan_lora import load_wan_adapter_model
from physverify.generation.settings import GenerationSettings
from physverify.generation.wan21 import snapshot_dir, snapshot_problem
from physverify.runtime.runs import RUN_FILE

RESUME_IGNORED: Final = frozenset({"commit", "beta_source"})


BASE_MODEL_FIELDS: Final = ("model_ref", "revision", "sha256", "dtype")


def pair_sources(
    store: CampaignStore, scorer: str, device: str
) -> tuple[StoredPairSource, StoredPairSource, list[PairRecord]]:
    candidates = store.candidates(BASE_TAG)
    train = load_pairs(store, scorer)
    val = load_pairs(store, scorer, VAL_PAIRS_FILE)
    try:
        sources = (
            StoredPairSource(stored_pairs(train, candidates), device),
            StoredPairSource(stored_pairs(val, candidates), device),
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return sources[0], sources[1], train


def adapter_model(
    gen: GenerationSettings, paths: PathsSettings, cfg: FinetuneSettings, device: str
) -> Any:
    problem = snapshot_problem(gen, paths)
    if problem is not None:
        raise UsageError(f"Wan2.1 weights: {problem}")
    return load_wan_adapter_model(
        snapshot_dir(gen, paths),
        gen,
        cfg.lora,
        device,
        cfg.train.seed,
        cfg.train.gradient_checkpointing,
    )


def run_beta(
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
    cfg: FinetuneSettings,
    gen: GenerationSettings,
    commit: str,
    run_id: str,
    pairs: Sequence[PairRecord],
    n_val: int,
    beta: float,
    beta_source: str,
    plan: Schedule,
    max_steps: int | None,
) -> dict[str, Any]:
    sigma = ShiftedFlowNoise(gen.flow_shift).quantiles(SIGMA_LEVELS)
    return {
        "run_id": run_id,
        **pairs[0].model_dump(include={"scorer", "scorer_version", "calibrated", "thresholds"}),
        "pairs_sha256": pairs_sha256(pairs),
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
        "commit": commit,
    }


def check_resume(path: Path, record: dict[str, Any]) -> None:
    """Every field of run.json but the commit and the beta's source must be unchanged: the pairs,
    the base model, the whole TrainSettings and LoRA dumps and the resolved schedule. A run.json
    without ``base_model`` predates its tracking and is invalidated: it can never be resumed."""
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    if "base_model" not in previous:
        raise UsageError(
            f"{path} predates base-model tracking, so it cannot be checked against the current "
            "base model and must be restarted: train with a new --run-id or with --no-resume "
            f"--force; for a beta-grid run, delete {path.parent}"
        )
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


def start_run(run_dir: Path, record: dict[str, Any], resume: bool) -> None:
    if resume:
        check_resume(run_dir / RUN_FILE, record)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(run_dir / RUN_FILE, json.dumps(record, indent=2, sort_keys=True))


def run_trainer(trainer: DpoTrainer, resume: bool, max_steps: int | None, force: bool) -> Any:
    try:
        return trainer.run(resume=resume, max_steps=max_steps, force=force)
    except ExistingRunError as exc:
        raise UsageError(f"{exc} (--force)") from exc
