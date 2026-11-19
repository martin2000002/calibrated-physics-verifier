from __future__ import annotations

import csv
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from physverify.config.models import Settings
from physverify.finetune.dpo import ShiftedFlowNoise
from physverify.finetune.records import CandidateRecord, PairRecord
from physverify.finetune.schedule import (
    Schedule,
    ScheduleError,
    beta_grid_steps,
    checkpoints,
    kept,
    require_checkpoints,
    schedule,
    total_steps,
)
from physverify.finetune.settings import TrainSettings, load_finetune
from physverify.finetune.stored_pairs import StoredPair, StoredPairSource, stored_pairs
from physverify.finetune.trainer import (
    METRICS_FILE,
    DpoTrainer,
    ExistingRunError,
    PairBatch,
    ValMetrics,
    choose_beta,
    last_val,
    pair_loss,
)

SHAPE = (1, 2, 1, 2, 2)
NUMEL = 8


class TinyAdapter:
    """A frozen zero base plus a LoRA-like low-rank bias ``A @ B`` (B starts at 0)."""

    def __init__(self, rank: int = 2) -> None:
        generator = torch.Generator().manual_seed(0)
        self.a = torch.randn(NUMEL, rank, generator=generator).requires_grad_()
        self.b = torch.zeros(rank, 1).requires_grad_()
        self.enabled = True
        self.inputs: list[torch.Tensor] = []

    def predict(self, x_t: Any, t: Any, cond: Any) -> Any:
        self.inputs.append(x_t.detach().clone())
        base = torch.zeros_like(x_t)
        if not self.enabled:
            return base
        return base + (self.a @ self.b).reshape(SHAPE[1:])

    @contextmanager
    def reference(self) -> Iterator[None]:
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True

    def trainable_parameters(self) -> list[Any]:
        return [self.a, self.b]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True)
        torch.save({"a": self.a.detach(), "b": self.b.detach()}, directory / "lora.pt")

    def load(self, directory: Path) -> None:
        state = torch.load(directory / "lora.pt")
        with torch.no_grad():
            self.a.copy_(state["a"])
            self.b.copy_(state["b"])


class Pairs:
    def __init__(self, n: int, offset: int = 0) -> None:
        generator = torch.Generator().manual_seed(100 + offset)
        self.items = [
            PairBatch(
                f"pair{offset + i}",
                torch.randn(SHAPE, generator=generator) + 1.0,
                torch.randn(SHAPE, generator=generator) - 1.0,
                torch.zeros(1, 3, 4),
            )
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def get(self, index: int) -> PairBatch:
        return self.items[index]


class StopError(Exception):
    pass


@pytest.fixture
def cfg(settings: Settings) -> TrainSettings:
    return load_finetune(settings).train.model_copy(
        update={
            "lr": 0.05,
            "grad_accum": 2,
            "warmup_fraction": 0.1,
            "checkpoint_fraction": 0.17,
            "keep_last": 3,
            "beta": 10.0,
        }
    )


def _trainer(cfg: TrainSettings, run_dir: Path, model: TinyAdapter, **kw: Any) -> DpoTrainer:
    return DpoTrainer(
        model, Pairs(4), cfg, ShiftedFlowNoise(8.0, 0), run_dir, val=Pairs(3, 50), **kw
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_thirty_steps_learn_the_preference(cfg: TrainSettings, tmp_path: Path) -> None:
    model = TinyAdapter()
    report = _trainer(cfg, tmp_path / "run", model).run(max_steps=30)
    assert report.steps == 30 and report.resumed_from == 0
    assert report.val is not None and report.val.implicit_acc >= 0.9
    rows = _rows(tmp_path / "run" / METRICS_FILE)
    assert [int(r["step"]) for r in rows] == list(range(1, 31))
    assert float(rows[0]["loss"]) == pytest.approx(0.693147, abs=1e-5)
    assert sum(float(r["implicit_acc"]) for r in rows[-5:]) / 5 >= 0.9
    assert [s for s, _ in checkpoints(tmp_path / "run")] == [20, 25, 30]
    assert last_val(tmp_path / "run") == report.val


def test_a_resumed_run_equals_the_uninterrupted_one(cfg: TrainSettings, tmp_path: Path) -> None:
    whole = TinyAdapter()
    _trainer(cfg, tmp_path / "whole", whole).run(max_steps=30)

    def stop_at(step: int) -> None:
        if step == 17:
            raise StopError

    with pytest.raises(StopError):
        _trainer(cfg, tmp_path / "cut", TinyAdapter(), on_step=stop_at).run(max_steps=30)
    resumed = TinyAdapter()
    report = _trainer(cfg, tmp_path / "cut", resumed).run(max_steps=30)
    assert report.resumed_from == 15
    assert torch.equal(whole.a, resumed.a) and torch.equal(whole.b, resumed.b)
    cut_rows = _rows(tmp_path / "cut" / METRICS_FILE)
    assert [int(r["step"]) for r in cut_rows] == list(range(1, 31))
    whole_rows = _rows(tmp_path / "whole" / METRICS_FILE)
    assert [r["loss"] for r in cut_rows] == [r["loss"] for r in whole_rows]


def test_winner_and_loser_share_sigma_and_noise() -> None:
    model = TinyAdapter()
    batch = Pairs(1).get(0)
    noise = ShiftedFlowNoise(8.0)
    pair_loss(model, batch, noise, key=3, beta=1.0)
    sigma = noise.draw(3).sigma
    x_t = model.inputs[0]
    eps_w = (x_t[0] - (1 - sigma) * batch.x0_w[0]) / sigma
    eps_l = (x_t[1] - (1 - sigma) * batch.x0_l[0]) / sigma
    assert torch.allclose(eps_w, eps_l, atol=1e-5)
    assert torch.equal(model.inputs[0], model.inputs[1])


def test_total_steps_and_kept_checkpoints(cfg: TrainSettings) -> None:
    epochs = cfg.model_copy(update={"max_steps": None, "epochs": 2.0, "grad_accum": 8})
    assert total_steps(epochs, 100) == 25
    assert total_steps(epochs, 100, max_steps=7) == 7
    assert kept([100, 200, 300, 400, 500], keep_last=2) == {400, 500}
    assert kept([100, 200, 300], keep_last=None) == {100, 200, 300}


def test_the_schedule_scales_with_the_pair_count(settings: Settings) -> None:
    cfg = load_finetune(settings).train
    for n_pairs, total, warmup, every, count in [(54, 41, 4, 8, 6), (180, 135, 13, 27, 5)]:
        plan = schedule(cfg, n_pairs)
        assert (plan.total, plan.warmup, plan.checkpoint_every) == (total, warmup, every)
        assert len(plan.checkpoint_steps()) == count and plan.checkpoint_steps()[-1] == total
        require_checkpoints(plan, cfg.keep_last, load_finetune(settings).select.checkpoints)
        grid = beta_grid_steps(cfg, plan.total)
        assert len(cfg.beta_grid) * grid <= plan.total + len(cfg.beta_grid)
    assert Schedule(7, 0, 3).checkpoint_steps() == [3, 6, 7]


def test_an_unworkable_schedule_is_refused(cfg: TrainSettings) -> None:
    long_warmup = cfg.model_construct(**{**cfg.model_dump(), "warmup_fraction": 1.0})
    with pytest.raises(ScheduleError, match="warmup"):
        schedule(long_warmup, 1, max_steps=200)
    few = schedule(cfg.model_copy(update={"checkpoint_fraction": 1.0}), 4, max_steps=30)
    with pytest.raises(ScheduleError, match="fewer than"):
        require_checkpoints(few, keep_last=None, needed=3)
    with pytest.raises(ScheduleError, match="fewer than"):
        require_checkpoints(schedule(cfg, 4, max_steps=30), keep_last=1, needed=3)
    loose = cfg.model_construct(**{**cfg.model_dump(), "beta_grid_fraction": 2.0})
    with pytest.raises(ScheduleError, match="exceed"):
        beta_grid_steps(loose, 10)


def test_a_fresh_start_refuses_existing_checkpoints_unless_forced(
    cfg: TrainSettings, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    _trainer(cfg, run_dir, TinyAdapter()).run(max_steps=10)
    assert checkpoints(run_dir)
    with pytest.raises(ExistingRunError, match="checkpoints"):
        _trainer(cfg, run_dir, TinyAdapter()).run(resume=False, max_steps=10)
    report = _trainer(cfg, run_dir, TinyAdapter()).run(resume=False, max_steps=5, force=True)
    assert report.resumed_from == 0
    assert [s for s, _ in checkpoints(run_dir)] == [3, 4, 5]
    assert [int(r["step"]) for r in _rows(run_dir / METRICS_FILE)] == list(range(1, 6))


def test_choose_beta_prefers_accuracy_within_the_kl_budget() -> None:
    results = {
        100.0: ValMetrics(10, 0.6, 0.010, 5),
        500.0: ValMetrics(10, 0.8, -0.015, 5),
        2500.0: ValMetrics(10, 0.95, 0.050, 5),
        5000.0: ValMetrics(10, 0.8, 0.012, 5),
    }
    assert choose_beta(results, kl_factor=2.0) == 500.0
    assert choose_beta(results, kl_factor=10.0) == 2500.0
    with pytest.raises(ValueError, match="no beta results"):
        choose_beta({}, 2.0)


def _candidate(cid: str, latents: str | None) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=cid,
        prompt_id="p",
        split="train",
        k=int(cid[-2:]),
        seed=1,
        model_tag="base",
        video_path=f"{cid}.mp4",
        latents_path=latents,
        embeds_path="embeds/p.safetensors",
        config_hash="h",
    )


def _pair() -> PairRecord:
    return PairRecord(
        scorer="calibrated",
        scorer_version="1",
        calibrated=True,
        thresholds="t",
        pair_id="p__k00__vs__p__k01",
        prompt_id="p",
        winner="p__k00",
        loser="p__k01",
        score_w=0.9,
        score_l=0.5,
        margin=0.4,
        weight=1.0,
        common_checks=["R1"],
    )


def test_stored_pairs_point_at_the_latents_and_the_prompt_embeds() -> None:
    candidates = [_candidate("p__k00", "w.safetensors"), _candidate("p__k01", "l.safetensors")]
    [stored] = stored_pairs([_pair()], candidates)
    assert stored == StoredPair(
        _pair().pair_id,
        Path("w.safetensors"),
        Path("l.safetensors"),
        Path("embeds/p.safetensors"),
        1.0,
    )
    with pytest.raises(ValueError, match="no candidate"):
        stored_pairs([_pair()], candidates[:1])
    with pytest.raises(ValueError, match="store_latents"):
        stored_pairs([_pair()], [candidates[0], _candidate("p__k01", None)])
    tensors = {
        "w.safetensors": {"latents": torch.ones(SHAPE)},
        "l.safetensors": {"latents": torch.zeros(SHAPE)},
        "embeds/p.safetensors": {"prompt_embeds": torch.ones(1, 3, 4)},
    }
    source = StoredPairSource([stored], "cpu", torch.float64, loader=lambda p: tensors[str(p)])
    batch = source.get(0)
    assert len(source) == 1 and batch.x0_w.dtype == torch.float64
    assert torch.equal(batch.x0_l, torch.zeros(SHAPE, dtype=torch.float64))


def test_no_val_rows_means_no_last_val(tmp_path: Path) -> None:
    assert last_val(tmp_path) is None
