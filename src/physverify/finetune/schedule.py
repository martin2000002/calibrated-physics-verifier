"""Optimizer-step schedule, data order and checkpoint naming of a Flow-DPO run."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from physverify.finetune.settings import TrainSettings

STEP_PREFIX: Final = "step-"
STEP_PATTERN: Final = re.compile(r"^step-(\d+)$")
STATE_FILE: Final = "trainer.pt"


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
