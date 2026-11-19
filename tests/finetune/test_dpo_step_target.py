from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from physverify.config.models import Settings
from physverify.timing._base import TimingContext
from physverify.timing.settings import load_timing
from physverify.timing.targets.dpo_step import CALLS, DpoStep, latent_shape


class ScaleModel:
    def __init__(self) -> None:
        self.w = torch.ones(1, requires_grad=True)
        self.enabled = True

    def predict(self, x_t: Any, t: Any, cond: Any) -> Any:
        return x_t * self.w if self.enabled else x_t

    @contextmanager
    def reference(self) -> Iterator[None]:
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True

    def trainable_parameters(self) -> list[Any]:
        return [self.w]

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


@pytest.fixture
def ctx(make_settings: Callable[..., Settings], tmp_path: Path) -> TimingContext:
    settings = make_settings(
        "timing.generation.num_frames=5",
        "timing.generation.height=16",
        "timing.generation.width=16",
        profile="precise",
    )
    return TimingContext(settings, "cpu", 1, tmp_path)


def _services() -> Any:
    memory = SimpleNamespace(reset=lambda: None, peak_mb=lambda: 12.0)
    return SimpleNamespace(
        clock=Clock(), sync=lambda device: None, memory=lambda d: memory, empty_cache=lambda: None
    )


def test_one_call_is_one_pair_step(ctx: TimingContext) -> None:
    model = ScaleModel()
    target = DpoStep(_services(), model_factory=lambda c: model, find_module=lambda m: True)
    clips = target.own_clips(ctx)
    assert len(clips) == CALLS
    setup = target.prepare(clips, ctx)
    assert setup == {"dpo.load_s": 0.5}
    measures = target.run(clips[0], ctx)
    assert measures == {"dpo.pair_s": 0.5, "dpo.cuda_peak_mb": 12.0}
    assert model.w.grad is None
    described = target.describe(ctx)
    assert described["latent_shape"] == "1x16x2x2x2"
    assert "sigma.q50" in described
    target.close()


def test_missing_libraries_make_the_target_unavailable(ctx: TimingContext) -> None:
    target = DpoStep(_services(), find_module=lambda m: m != "peft")
    assert "peft" in (target.unavailable(ctx) or "")


def test_latent_shape_follows_the_vae_strides(ctx: TimingContext) -> None:
    gen = load_timing(ctx.settings).generation.model_copy(
        update={"num_frames": 81, "height": 480, "width": 832}
    )
    assert latent_shape(gen) == (1, 16, 21, 60, 104)
