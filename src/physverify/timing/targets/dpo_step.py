"""Seconds per preference pair of the Flow-DPO step on the H200 (ADR 0012, 3.9 and Juicio 10).

One call is one pair: learner forward + backward and reference forward (adapter off) for winner
and loser, on synthetic latents with the generator's shape. It reuses ``finetune.dpo`` and
``finetune.wan_lora``; nothing of the step is reimplemented here. Measures: ``dpo.pair_s`` and
``dpo.cuda_peak_mb``; setup ``dpo.load_s``.
"""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar, Final

from physverify.core.errors import TimingError
from physverify.finetune.dpo import SIGMA_LEVELS, ShiftedFlowNoise
from physverify.finetune.settings import load_finetune
from physverify.finetune.trainer import AdapterModel, PairBatch, pair_loss
from physverify.finetune.wan_lora import load_wan_adapter_model
from physverify.generation.settings import TEMPORAL_STRIDE, GenerationSettings
from physverify.generation.wan21 import snapshot_dir, snapshot_problem
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.timing._base import TimingContext
from physverify.timing.hooks import TargetHooks
from physverify.timing.settings import load_timing
from physverify.timing.target_services import TimingServices
from physverify.timing.targets.wan21_generate import module_found

LATENT_CHANNELS: Final = 16
VAE_SPATIAL: Final = 8
TEXT_TOKENS: Final = 512
TEXT_DIM: Final = 4096
CALLS: Final = 5
DATASET: Final = "synthetic_pair"

ModelFactory = Callable[[TimingContext], AdapterModel]


def latent_shape(gen: GenerationSettings) -> tuple[int, int, int, int, int]:
    frames = (gen.num_frames - 1) // TEMPORAL_STRIDE + 1
    return (1, LATENT_CHANNELS, frames, gen.height // VAE_SPATIAL, gen.width // VAE_SPATIAL)


def default_model(ctx: TimingContext) -> AdapterModel:
    gen = load_timing(ctx.settings).generation
    cfg = load_finetune(ctx.settings)
    snapshot = snapshot_dir(gen, ctx.settings.paths)
    return load_wan_adapter_model(
        snapshot, gen, cfg.lora, ctx.device, cfg.train.seed, cfg.train.gradient_checkpointing
    )


def pair_clip(index: int, gen: GenerationSettings) -> ClipManifest:
    return ClipManifest(
        dataset=DATASET,
        clip_id=f"pair-{index:02d}",
        split=Split.EVAL,
        video_path=DATASET,
        n_frames=gen.num_frames,
        width=gen.width,
        height=gen.height,
        container_fps=float(gen.fps),
        seconds_per_frame=1.0 / gen.fps,
        time_source=TimeSource.CONTAINER_FPS,
        object_queries=(),
    )


class DpoStep(TargetHooks):
    name: ClassVar[str] = "dpo_step"
    requires_gpu: ClassVar[bool] = True

    def __init__(
        self,
        services: TimingServices,
        *,
        model_factory: ModelFactory = default_model,
        find_module: Callable[[str], bool] = module_found,
    ) -> None:
        self._services = services
        self._factory = model_factory
        self._find = find_module
        self._model: AdapterModel | None = None
        self._batch: PairBatch | None = None
        self._noise: ShiftedFlowNoise | None = None
        self._beta = 1.0

    def unavailable(self, ctx: TimingContext) -> str | None:
        missing = [m for m in ("diffusers", "peft") if not self._find(m)]
        if missing:
            return f"{', '.join(missing)} not installed: sync the 'generation' dependency group"
        return snapshot_problem(load_timing(ctx.settings).generation, ctx.settings.paths)

    def own_clips(self, ctx: TimingContext) -> Sequence[ClipManifest]:
        gen = load_timing(ctx.settings).generation
        return [pair_clip(i, gen) for i in range(CALLS)]

    def prepare(self, clips: Sequence[ClipManifest], ctx: TimingContext) -> Mapping[str, float]:
        torch = importlib.import_module("torch")
        gen = load_timing(ctx.settings).generation
        cfg = load_finetune(ctx.settings)
        services = self._services
        start = services.clock()
        self._model = self._factory(ctx)
        services.sync(ctx.device)
        load_s = services.clock() - start
        generator = torch.Generator(device="cpu").manual_seed(gen.seed)
        shape = latent_shape(gen)
        x0_w = torch.randn(shape, generator=generator).to(ctx.device)
        x0_l = torch.randn(shape, generator=generator).to(ctx.device)
        cond = torch.randn((1, TEXT_TOKENS, TEXT_DIM), generator=generator).to(ctx.device)
        self._batch = PairBatch("synthetic", x0_w, x0_l, cond)
        self._noise = ShiftedFlowNoise(gen.flow_shift, cfg.train.seed)
        self._beta = cfg.train.beta
        return {"dpo.load_s": load_s}

    def run(self, clip: ClipManifest, ctx: TimingContext) -> Mapping[str, float]:
        if self._model is None or self._batch is None or self._noise is None:
            raise TimingError(f"{self.name}: run() before prepare()")
        services = self._services
        memory = services.memory(ctx.device)
        memory.reset()
        key = int(clip.clip_id.rsplit("-", 1)[-1])
        services.sync(ctx.device)
        start = services.clock()
        loss, _ = pair_loss(self._model, self._batch, self._noise, key, self._beta)
        loss.backward()
        services.sync(ctx.device)
        measures = {"dpo.pair_s": services.clock() - start}
        for parameter in self._model.trainable_parameters():
            parameter.grad = None
        peak = memory.peak_mb()
        if peak is not None:
            measures["dpo.cuda_peak_mb"] = peak
        return measures

    def describe(self, ctx: TimingContext) -> Mapping[str, str]:
        gen = load_timing(ctx.settings).generation
        cfg = load_finetune(ctx.settings)
        sigma = ShiftedFlowNoise(gen.flow_shift).quantiles(SIGMA_LEVELS)
        return {
            "latent_shape": "x".join(map(str, latent_shape(gen))),
            "lora": f"r={cfg.lora.rank},alpha={cfg.lora.alpha}",
            "gradient_checkpointing": str(cfg.train.gradient_checkpointing),
            "flow_shift": str(gen.flow_shift),
            **{f"sigma.{name}": f"{value:.4f}" for name, value in sigma.items()},
        }

    def close(self) -> None:
        self._model = None
        self._batch = None
        gc.collect()
        self._services.empty_cache()


PLUGINS = (DpoStep,)
