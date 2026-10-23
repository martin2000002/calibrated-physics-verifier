"""Offline Flow-DPO trainer over stored latent pairs (ADR 0012, 3.8 and 4.4).

The model is injected (``AdapterModel``): the Wan transformer with a LoRA on the H200, a tiny
torch module in the tests. Determinism comes from identifiers, not from global RNG state: the data
order of an epoch is a seeded permutation and the noise of micro-step m is keyed by m, so a run
resumed from a checkpoint reproduces the uninterrupted one exactly.
"""

from __future__ import annotations

import csv
import importlib
import math
import re
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from physverify.finetune.dpo import (
    DpoStats,
    ShiftedFlowNoise,
    _torch,
    flow_dpo_loss,
    noised,
    per_sample_error,
)
from physverify.finetune.records import CandidateRecord, PairRecord
from physverify.finetune.settings import TrainSettings

STEP_PREFIX: Final = "step-"
STEP_PATTERN: Final = re.compile(r"^step-(\d+)$")
ADAPTER_DIR: Final = "adapter"
STATE_FILE: Final = "trainer.pt"
METRICS_FILE: Final = "metrics.csv"
VAL_FILE: Final = "val.csv"
VAL_KEY_OFFSET: Final = 1 << 40


class AdapterModel(Protocol):
    def predict(self, x_t: Any, t: Any, cond: Any) -> Any: ...

    def reference(self) -> AbstractContextManager[None]: ...

    def trainable_parameters(self) -> list[Any]: ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...


@dataclass(frozen=True)
class PairBatch:
    """One pair: final latents of winner and loser, the shared text condition, its weight."""

    pair_id: str
    x0_w: Any
    x0_l: Any
    cond: Any
    weight: float = 1.0


class PairBatchSource(Protocol):
    def __len__(self) -> int: ...

    def get(self, index: int) -> PairBatch: ...


def pair_loss(
    model: AdapterModel, batch: PairBatch, noise: ShiftedFlowNoise, key: int, beta: float
) -> tuple[Any, DpoStats]:
    """Learner with gradient and reference (adapter off) without, on one shared ``sigma, eps``."""
    torch = _torch()
    draw = noise.draw(key)
    eps = noise.epsilon(draw, batch.x0_w)
    x_w, v_w, t = noised(batch.x0_w, eps, draw.sigma)
    x_l, v_l, _ = noised(batch.x0_l, eps, draw.sigma)
    x_t = torch.cat([x_w, x_l])
    t2 = torch.cat([t, t])
    cond = torch.cat([batch.cond, batch.cond])
    target = torch.cat([v_w, v_l])
    err = per_sample_error(model.predict(x_t, t2, cond), target)
    with torch.no_grad(), model.reference():
        err_ref = per_sample_error(model.predict(x_t, t2, cond), target)
    n = batch.x0_w.shape[0]
    weight = torch.full((n,), batch.weight, device=err.device)
    return flow_dpo_loss(err[:n], err_ref[:n], err[n:], err_ref[n:], beta, weight)


def data_order(seed: int, epoch: int, n: int) -> np.ndarray:
    return np.random.default_rng([seed, epoch]).permutation(n)


def total_steps(cfg: TrainSettings, n_pairs: int, max_steps: int | None = None) -> int:
    if max_steps is not None:
        return max_steps
    if cfg.max_steps is not None:
        return cfg.max_steps
    return max(1, math.ceil(cfg.epochs * n_pairs / cfg.grad_accum))


class ScheduleError(ValueError):
    """The schedule cannot work for this many optimizer steps."""


@dataclass(frozen=True)
class Schedule:
    """Optimizer steps of one run; warmup and checkpoint spacing are fractions of ``total``."""

    total: int
    warmup: int
    checkpoint_every: int

    def checkpoint_steps(self) -> list[int]:
        return sorted(
            set(range(self.checkpoint_every, self.total + 1, self.checkpoint_every)) | {self.total}
        )


def schedule(cfg: TrainSettings, n_pairs: int, max_steps: int | None = None) -> Schedule:
    total = total_steps(cfg, n_pairs, max_steps)
    warmup = math.floor(cfg.warmup_fraction * total)
    if warmup >= total:
        raise ScheduleError(f"warmup of {warmup} steps does not end within {total} steps")
    every = max(1, round(cfg.checkpoint_fraction * total))
    return Schedule(total, warmup, every)


def require_checkpoints(plan: Schedule, keep_last: int | None, needed: int) -> None:
    """``select`` compares ``needed`` checkpoints; fewer kept ones would make it a no-op."""
    have = len(kept(plan.checkpoint_steps(), keep_last))
    if have < needed:
        raise ScheduleError(
            f"{plan.total} steps keep {have} checkpoints (every {plan.checkpoint_every}), "
            f"fewer than finetune.select.checkpoints = {needed}: lower checkpoint_fraction "
            "or raise epochs"
        )


def beta_grid_steps(cfg: TrainSettings, main_total: int) -> int:
    steps = max(1, round(cfg.beta_grid_fraction * main_total))
    if steps > main_total:
        raise ScheduleError(f"beta-grid runs of {steps} steps exceed the main run ({main_total})")
    return steps


@dataclass(frozen=True)
class StepMetrics:
    step: int
    loss: float
    implicit_acc: float
    reward_margin: float
    kl_proxy: float
    grad_norm: float
    lr: float
    pair_s: float


@dataclass(frozen=True)
class ValMetrics:
    step: int
    implicit_acc: float
    kl_proxy: float
    n_pairs: int


@dataclass(frozen=True)
class TrainReport:
    steps: int
    resumed_from: int
    last_checkpoint: Path | None
    val: ValMetrics | None


def checkpoint_name(step: int) -> str:
    return f"{STEP_PREFIX}{step:06d}"


def checkpoints(directory: Path) -> list[tuple[int, Path]]:
    if not directory.is_dir():
        return []
    found = []
    for path in directory.iterdir():
        match = STEP_PATTERN.match(path.name)
        if match and (path / STATE_FILE).is_file():
            found.append((int(match.group(1)), path))
    return sorted(found)


def kept(steps: Sequence[int], keep_last: int | None) -> set[int]:
    """Every checkpoint, or the last ``keep_last``."""
    return set(steps) if keep_last is None else set(sorted(steps)[-keep_last:])


class CsvSink:
    """Appends dataclass rows to a CSV; on resume, rows after the resumed step are dropped."""

    def __init__(self, path: Path, row_type: type[Any]) -> None:
        self._path = path
        self._columns = [f.name for f in fields(row_type)]

    def truncate_after(self, step: int) -> None:
        if not self._path.is_file():
            return
        with self._path.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if int(r["step"]) <= step]
        self._write(rows, "w")

    def append(self, row: Any) -> None:
        self._write([asdict(row)], "a")

    def _write(self, rows: Iterable[Mapping[str, Any]], mode: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new = mode == "w" or not self._path.is_file()
        with self._path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns)
            if new:
                writer.writeheader()
            writer.writerows(rows)


class ExistingRunError(ValueError):
    """A fresh start was asked for in a run directory that already holds checkpoints."""


class DpoTrainer:
    def __init__(
        self,
        model: AdapterModel,
        data: PairBatchSource,
        cfg: TrainSettings,
        noise: ShiftedFlowNoise,
        run_dir: Path,
        *,
        val: PairBatchSource | None = None,
        beta: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
        on_step: Callable[[int], None] | None = None,
    ) -> None:
        if len(data) == 0:
            raise ValueError("no training pairs")
        self._model = model
        self._data = data
        self._cfg = cfg
        self._noise = noise
        self._dir = run_dir
        self._val = val
        self._beta = cfg.beta if beta is None else beta
        self._clock = clock
        self._on_step = on_step
        self._metrics = CsvSink(run_dir / METRICS_FILE, StepMetrics)
        self._val_sink = CsvSink(run_dir / VAL_FILE, ValMetrics)

    def run(
        self, resume: bool = True, max_steps: int | None = None, force: bool = False
    ) -> TrainReport:
        """``resume=False`` refuses a directory holding checkpoints unless ``force`` clears it."""
        torch = _torch()
        plan = schedule(self._cfg, len(self._data), max_steps)
        params = self._model.trainable_parameters()
        optimizer = torch.optim.AdamW(params, lr=self._cfg.lr, weight_decay=self._cfg.weight_decay)
        warmup = plan.warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / (warmup + 1)) if warmup else 1.0
        )
        start = self._resume(optimizer, scheduler) if resume else self._fresh(force)
        marks = set(plan.checkpoint_steps())
        last: Path | None = None
        val: ValMetrics | None = None
        for step in range(start, plan.total):
            if self._on_step is not None:
                self._on_step(step)
            self._metrics.append(self._step(step, params, optimizer, scheduler))
            done = step + 1
            if done in marks:
                last = self._checkpoint(done, optimizer, scheduler)
                val = self.validate(done)
        return TrainReport(plan.total, start, last, val)

    def validate(self, step: int) -> ValMetrics | None:
        """Validation pairs on fixed ``(sigma, eps)`` per pair, never trained on."""
        if self._val is None or len(self._val) == 0:
            return None
        torch = _torch()
        stats = []
        with torch.no_grad():
            for index in range(len(self._val)):
                batch = self._val.get(index)
                _, s = pair_loss(
                    self._model, batch, self._noise, VAL_KEY_OFFSET + index, self._beta
                )
                stats.append(s)
        metrics = ValMetrics(
            step,
            float(np.mean([s.implicit_acc for s in stats])),
            float(np.mean([s.kl_proxy for s in stats])),
            len(stats),
        )
        self._val_sink.append(metrics)
        return metrics

    def _step(self, step: int, params: list[Any], optimizer: Any, scheduler: Any) -> StepMetrics:
        torch = _torch()
        optimizer.zero_grad(set_to_none=True)
        accum = self._cfg.grad_accum
        stats: list[DpoStats] = []
        start = self._clock()
        for j in range(accum):
            micro = step * accum + j
            loss, s = pair_loss(self._model, self._batch(micro), self._noise, micro, self._beta)
            (loss / accum).backward()
            stats.append(s)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self._cfg.grad_clip)
        lr = float(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
        return StepMetrics(
            step=step + 1,
            loss=float(np.mean([s.loss for s in stats])),
            implicit_acc=float(np.mean([s.implicit_acc for s in stats])),
            reward_margin=float(np.mean([s.reward_margin for s in stats])),
            kl_proxy=float(np.mean([s.kl_proxy for s in stats])),
            grad_norm=float(grad_norm),
            lr=lr,
            pair_s=(self._clock() - start) / accum,
        )

    def _batch(self, micro: int) -> PairBatch:
        n = len(self._data)
        epoch, position = divmod(micro, n)
        return self._data.get(int(data_order(self._cfg.seed, epoch, n)[position]))

    def _checkpoint(self, step: int, optimizer: Any, scheduler: Any) -> Path:
        torch = _torch()
        final = self._dir / checkpoint_name(step)
        partial = self._dir / f".{final.name}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        self._model.save(partial / ADAPTER_DIR)
        state = {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, partial / STATE_FILE)
        shutil.rmtree(final, ignore_errors=True)
        partial.rename(final)
        self._prune()
        return final

    def _prune(self) -> None:
        found = checkpoints(self._dir)
        keep = kept([s for s, _ in found], self._cfg.keep_last)
        for step, path in found:
            if step not in keep:
                shutil.rmtree(path)

    def _fresh(self, force: bool = False) -> int:
        found = checkpoints(self._dir)
        if found and not force:
            raise ExistingRunError(
                f"{self._dir} holds {len(found)} checkpoints; resume, use a new run id, "
                "or pass force to delete them"
            )
        for _, path in found:
            shutil.rmtree(path)
        self._metrics.truncate_after(0)
        self._val_sink.truncate_after(0)
        return 0

    def _resume(self, optimizer: Any, scheduler: Any) -> int:
        found = checkpoints(self._dir)
        if not found:
            return self._fresh()
        step, path = found[-1]
        torch = _torch()
        self._model.load(path / ADAPTER_DIR)
        state = torch.load(path / STATE_FILE, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        self._metrics.truncate_after(step)
        self._val_sink.truncate_after(step)
        return int(state["step"])


def choose_beta(results: Mapping[float, ValMetrics], kl_factor: float) -> float:
    """Highest validation ``implicit_acc`` among the betas whose ``|kl_proxy|`` stays within
    ``kl_factor`` times the smallest; ties go to the smaller beta."""
    if not results:
        raise ValueError("no beta results")
    floor = min(abs(m.kl_proxy) for m in results.values())
    allowed = [b for b, m in results.items() if abs(m.kl_proxy) <= kl_factor * floor + 1e-12]
    return max(sorted(allowed), key=lambda b: results[b].implicit_acc)


def last_val(run_dir: Path) -> ValMetrics | None:
    """The last validation row of a run (also of one that was already complete)."""
    path = run_dir / VAL_FILE
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    row = rows[-1]
    return ValMetrics(
        int(row["step"]), float(row["implicit_acc"]), float(row["kl_proxy"]), int(row["n_pairs"])
    )


TensorLoader = Callable[[Path], Mapping[str, Any]]


def load_safetensors(path: Path) -> Mapping[str, Any]:
    load_file = importlib.import_module("safetensors.torch").load_file
    loaded: Mapping[str, Any] = load_file(str(path))
    return loaded


@dataclass(frozen=True)
class StoredPair:
    pair_id: str
    latents_w: Path
    latents_l: Path
    embeds: Path
    weight: float


class StoredPairSource:
    """Pairs read from the campaign store on demand (latents float32, embeds as generated)."""

    def __init__(
        self,
        pairs: Sequence[StoredPair],
        device: str,
        dtype: Any = None,
        loader: TensorLoader = load_safetensors,
    ) -> None:
        self._pairs = list(pairs)
        self._device = device
        self._dtype = dtype
        self._load = loader

    def __len__(self) -> int:
        return len(self._pairs)

    def get(self, index: int) -> PairBatch:
        pair = self._pairs[index]
        return PairBatch(
            pair_id=pair.pair_id,
            x0_w=self._tensor(pair.latents_w, "latents"),
            x0_l=self._tensor(pair.latents_l, "latents"),
            cond=self._tensor(pair.embeds, "prompt_embeds"),
            weight=pair.weight,
        )

    def _tensor(self, path: Path, key: str) -> Any:
        tensor = self._load(path)[key].to(self._device)
        return tensor if self._dtype is None else tensor.to(self._dtype)


def stored_pairs(
    pairs: Sequence[PairRecord], candidates: Sequence[CandidateRecord]
) -> list[StoredPair]:
    """The latent files of each pair; the embeddings are the prompt's (shared by both)."""
    by_id = {c.candidate_id: c for c in candidates}
    stored = []
    for pair in pairs:
        winner, loser = by_id.get(pair.winner), by_id.get(pair.loser)
        if winner is None or loser is None:
            raise ValueError(f"{pair.pair_id}: a member has no candidate record")
        if winner.latents_path is None or loser.latents_path is None:
            raise ValueError(f"{pair.pair_id}: latents were not stored (store_latents = false)")
        stored.append(
            StoredPair(
                pair.pair_id,
                Path(winner.latents_path),
                Path(loser.latents_path),
                Path(winner.embeds_path),
                pair.weight,
            )
        )
    return stored
