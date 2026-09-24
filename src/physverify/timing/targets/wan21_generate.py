"""Per-video cost of the thesis generator, Wan2.1-T2V-1.3B, on the H200 (ADR 0008, D7-D9).

Each configured prompt is one clip. A call generates it once per batch size B and reports, as
``generate.b{B}.*``: ``wall_s`` (device sync to device sync), ``per_video_s = wall_s / B``,
``first_step_s``, ``step_s``, ``decode_s``, ``cpu_s`` and ``cuda_peak_mb``. The B = 1 video is
written as an mp4 into the ``local_video`` subset, whose manifest is rewritten after every video,
so the extraction and verification targets time exactly these clips. Running out of memory at
B > 1 drops that batch size for the rest of the run (noted in ``describe``); at B = 1 it fails.
"""

from __future__ import annotations

import gc
import importlib.util
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
from numpy.typing import NDArray

from physverify.config.models import Settings
from physverify.datasets.local_video import (
    MANIFEST_FILE,
    ClipEntry,
    LocalManifest,
    LocalVideoDataset,
    to_queries,
)
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.synth.store import atomic_write_bytes
from physverify.timing._base import TimingContext
from physverify.timing.generation import (
    StepTimer,
    call_kwargs,
    is_out_of_memory,
    load_wan_pipeline,
    snapshot_dir,
    snapshot_problem,
    step_measures,
    torch_generator,
)
from physverify.timing.hooks import TargetHooks
from physverify.timing.mp4 import RGB_CHANNELS, VIDEO_SUFFIX, to_uint8, write_mp4
from physverify.timing.records import (
    GENERATE_LOAD_CUDA_MB,
    GENERATE_LOAD_S,
    TimingError,
    generate_measure,
)
from physverify.timing.services import TimingServices
from physverify.timing.settings import GenerationSettings, PromptSpec, load_timing

BATCH_NDIM: Final = 5  # (videos, frames, height, width, channels)
MISSING_DIFFUSERS: Final = "diffusers is not installed: sync the 'generation' dependency group"

PipeFactory = Callable[[Path, GenerationSettings, str], Any]
GeneratorFactory = Callable[[str, int], Any]


def module_found(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def subset_dir(settings: Settings, gen: GenerationSettings) -> Path:
    return settings.paths.raw_dir / LocalVideoDataset.name / gen.subset


def prompt_clip(prompt: PromptSpec, directory: Path, gen: GenerationSettings) -> ClipManifest:
    return ClipManifest(
        dataset=LocalVideoDataset.name,
        clip_id=prompt.id,
        split=Split.EVAL,
        video_path=str((directory / f"{prompt.id}{VIDEO_SUFFIX}").resolve()),
        n_frames=gen.num_frames,
        width=gen.width,
        height=gen.height,
        container_fps=float(gen.fps),
        seconds_per_frame=1.0 / gen.fps,
        time_source=TimeSource.CONTAINER_FPS,
        object_queries=to_queries(prompt.queries),
        prompt=prompt.text,
    )


def batch_frames(output: Any, batch_size: int) -> NDArray[Any]:
    frames = np.asarray(output.frames)
    if frames.ndim != BATCH_NDIM or len(frames) != batch_size or frames.shape[-1] != RGB_CHANNELS:
        raise TimingError(
            f"expected {batch_size} RGB videos (B, T, H, W, 3) from the pipeline, "
            f"got {frames.shape}"
        )
    return frames


class Wan21Generate(TargetHooks):
    name: ClassVar[str] = "wan21_generate"
    requires_gpu: ClassVar[bool] = True

    def __init__(
        self,
        services: TimingServices,
        *,
        pipe_factory: PipeFactory = load_wan_pipeline,
        generator_factory: GeneratorFactory = torch_generator,
        find_module: Callable[[str], bool] = module_found,
    ) -> None:
        self._services = services
        self._pipe_factory = pipe_factory
        self._generator_factory = generator_factory
        self._find_module = find_module
        self._pipe: Any | None = None
        self._gen: GenerationSettings | None = None
        self._prompts: dict[str, tuple[int, PromptSpec]] = {}
        self._dropped: set[int] = set()
        self._notes: dict[str, str] = {}
        self._written: dict[str, ClipEntry] = {}

    def unavailable(self, ctx: TimingContext) -> str | None:
        if not self._find_module("diffusers"):
            return MISSING_DIFFUSERS
        return snapshot_problem(load_timing(ctx.settings).generation, ctx.settings.paths)

    def own_clips(self, ctx: TimingContext) -> Sequence[ClipManifest]:
        gen = load_timing(ctx.settings).generation
        directory = subset_dir(ctx.settings, gen)
        return [prompt_clip(prompt, directory, gen) for prompt in gen.prompts]

    def prepare(self, clips: Sequence[ClipManifest], ctx: TimingContext) -> Mapping[str, float]:
        """Loads the pipeline (timed) and removes the manifest of an earlier run."""
        gen = load_timing(ctx.settings).generation
        self._gen = gen
        self._prompts = {prompt.id: (index, prompt) for index, prompt in enumerate(gen.prompts)}
        self._dropped.clear()
        self._notes.clear()
        self._written.clear()
        (subset_dir(ctx.settings, gen) / MANIFEST_FILE).unlink(missing_ok=True)
        services = self._services
        memory = services.memory(ctx.device)
        memory.reset()
        start = services.clock()
        self._pipe = self._pipe_factory(snapshot_dir(gen, ctx.settings.paths), gen, ctx.device)
        services.sync(ctx.device)
        measures = {GENERATE_LOAD_S: services.clock() - start}
        peak = memory.peak_mb()
        if peak is not None:
            measures[GENERATE_LOAD_CUDA_MB] = peak
        return measures

    def run(self, clip: ClipManifest, ctx: TimingContext) -> Mapping[str, float]:
        if self._pipe is None or self._gen is None:
            raise TimingError(f"{self.name}: run() before prepare()")
        if clip.clip_id not in self._prompts:
            raise TimingError(f"{self.name}: {clip.clip_id!r} is not a configured prompt")
        index, prompt = self._prompts[clip.clip_id]
        measures: dict[str, float] = {}
        for batch_size in self._gen.batch_sizes:
            if batch_size in self._dropped:
                continue
            result = self._attempt(prompt, self._gen.seed + index, batch_size, ctx)
            if result is None:
                continue
            timed, frames = result
            measures.update(timed)
            if batch_size == 1:
                measures[generate_measure(1, "write_s")] = self._write(clip, prompt, frames[0])
            del result, frames
        return measures

    def describe(self, ctx: TimingContext) -> Mapping[str, str]:
        gen = load_timing(ctx.settings).generation
        return {
            "model": gen.model_ref,
            "revision": gen.revision,
            "dtype": gen.dtype,
            "vae_dtype": gen.vae_dtype,
            "resolution": f"{gen.width}x{gen.height}",
            "num_frames": str(gen.num_frames),
            "fps": str(gen.fps),
            "num_inference_steps": str(gen.num_inference_steps),
            "guidance_scale": str(gen.guidance_scale),
            "flow_shift": str(gen.flow_shift),
            "batch_sizes": ",".join(map(str, gen.batch_sizes)),
            "subset": f"{LocalVideoDataset.name}/{gen.subset}",
            **self._notes,
        }

    def close(self) -> None:
        self._pipe = None
        gc.collect()
        self._services.empty_cache()

    def _attempt(
        self, prompt: PromptSpec, seed: int, batch_size: int, ctx: TimingContext
    ) -> tuple[dict[str, float], NDArray[Any]] | None:
        """One generation; None when a batch size above 1 runs out of memory (then dropped)."""
        try:
            return self._generate(prompt, seed, batch_size, ctx)
        except Exception as exc:
            if batch_size == 1 or not is_out_of_memory(exc):
                raise
            self._drop(batch_size, exc)
        # Outside the handler, so the traceback no longer pins the failed call's tensors.
        gc.collect()
        self._services.empty_cache()
        return None

    def _generate(
        self, prompt: PromptSpec, seed: int, batch_size: int, ctx: TimingContext
    ) -> tuple[dict[str, float], NDArray[Any]]:
        assert self._pipe is not None and self._gen is not None
        services = self._services
        sync = partial(services.sync, ctx.device)
        timer = StepTimer(sync, services.clock)
        generator = self._generator_factory(ctx.device, seed)
        kwargs = call_kwargs(prompt.text, self._gen, batch_size)
        memory = services.memory(ctx.device)
        memory.reset()
        sync()
        cpu_start, start = services.cpu_clock(), services.clock()
        output = self._pipe(**kwargs, generator=generator, callback_on_step_end=timer)
        sync()
        end, cpu_end = services.clock(), services.cpu_clock()
        frames = batch_frames(output, batch_size)
        wall = end - start
        leaves = {"wall_s": wall, "per_video_s": wall / batch_size, "cpu_s": cpu_end - cpu_start}
        leaves.update(step_measures(start, timer.marks, end))
        peak = memory.peak_mb()
        if peak is not None:
            leaves["cuda_peak_mb"] = peak
        return {generate_measure(batch_size, leaf): v for leaf, v in leaves.items()}, frames

    def _drop(self, batch_size: int, exc: BaseException) -> None:
        self._dropped.add(batch_size)
        text = str(exc).strip()
        reason = text.splitlines()[0] if text else type(exc).__name__
        self._notes[f"oom.b{batch_size}"] = f"out of memory, batch size dropped: {reason}"

    def _write(self, clip: ClipManifest, prompt: PromptSpec, frames: NDArray[Any]) -> float:
        """Writes the mp4, then the manifest listing every video of this run; returns seconds."""
        clock = self._services.clock
        start = clock()
        video = Path(clip.video_path)
        write_mp4(to_uint8(frames), video, clip.container_fps)
        self._written.setdefault(prompt.id, ClipEntry(video=video.name, queries=prompt.queries))
        manifest = LocalManifest(clips=list(self._written.values()))
        payload = manifest.model_dump_json(exclude_none=True, indent=2).encode("utf-8")
        atomic_write_bytes(video.parent / MANIFEST_FILE, payload)
        return clock() - start


PLUGINS = (Wan21Generate,)
