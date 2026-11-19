"""wan21_generate and its helpers with a fake pipeline: no torch, no diffusers, no GPU."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from physverify.config.models import Settings
from physverify.core.digests import file_sha256
from physverify.core.errors import TimingError
from physverify.core.local_manifest import MANIFEST_FILE, to_queries
from physverify.datasets.local_video import LocalVideoDataset
from physverify.extraction._base import WeightKind
from physverify.generation.frames import to_uint8
from physverify.generation.settings import GenerationSettings
from physverify.generation.wan21 import (
    FETCH_HINT,
    StepTimer,
    call_kwargs,
    is_out_of_memory,
    load_wan_pipeline,
    snapshot_dir,
    step_measures,
    torch_generator,
    wan_weight_spec,
)
from physverify.schema.clip import ClipManifest, TimeSource
from physverify.timing._base import TimingContext
from physverify.timing.records import (
    GENERATE_LOAD_CUDA_MB,
    GENERATE_LOAD_S,
    check_measures,
    generate_measure,
)
from physverify.timing.settings import load_timing
from physverify.timing.target_services import TimingServices
from physverify.timing.targets.wan21_generate import MISSING_DIFFUSERS, Wan21Generate

from .conftest import FakeClock, FakeCudaMemory

HEIGHT, WIDTH, FRAMES, STEPS = 32, 48, 5, 4
ENCODE_S, STEP_S, DECODE_S, LOAD_S = 1.0, 0.5, 2.0, 3.0
SMALL = (
    f"timing.generation.height={HEIGHT}",
    f"timing.generation.width={WIDTH}",
    f"timing.generation.num_frames={FRAMES}",
    f"timing.generation.num_inference_steps={STEPS}",
)
MODEL_INDEX = {
    "_class_name": "WanPipeline",
    "_diffusers_version": "0.33.0",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "unused": [None, None],
}
TEXT_SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


class OutOfMemoryError(RuntimeError):
    """Stands in for ``torch.OutOfMemoryError`` (matched by class name)."""


class FakePipe:
    """Advances the clock like a pipeline: text encoding, steps (callback after each), decode."""

    def __init__(self, clock: FakeClock, fail_at: dict[int, Exception] | None = None) -> None:
        self.clock = clock
        self.fail_at = fail_at or {}
        self.calls: list[dict[str, Any]] = []
        self.frames_shape: tuple[int, ...] | None = None

    def __call__(self, *, callback_on_step_end: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        batch = kwargs["num_videos_per_prompt"]
        if batch in self.fail_at:
            raise self.fail_at[batch]
        self.clock.advance(ENCODE_S)
        for step in range(kwargs["num_inference_steps"]):
            self.clock.advance(STEP_S)
            assert callback_on_step_end(self, step, 1000 - step, {"latents": step}) == {
                "latents": step
            }
        self.clock.advance(DECODE_S)
        shape = self.frames_shape or (
            batch,
            kwargs["num_frames"],
            kwargs["height"],
            kwargs["width"],
            3,
        )
        return SimpleNamespace(frames=_gradient(shape))


def _gradient(shape: tuple[int, ...]) -> NDArray[np.float32]:
    frames = np.zeros(shape, dtype=np.float32)
    if len(shape) == 5:
        for t in range(shape[1]):
            frames[:, t] = (t + 1) / (shape[1] + 1)
    return frames


class Counter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *args: Any) -> None:
        self.count += 1


def _write_snapshot(directory: Path, *, skip: tuple[str, ...] = ()) -> Path:
    """A diffusers-shaped snapshot of empty files; returns its ``model_index.json``."""
    files = {
        "scheduler/scheduler_config.json": "{}",
        "tokenizer/tokenizer_config.json": "{}",
        "text_encoder/config.json": "{}",
        "text_encoder/model.safetensors.index.json": json.dumps(
            {"weight_map": {"a": TEXT_SHARDS[0], "b": TEXT_SHARDS[1], "c": TEXT_SHARDS[1]}}
        ),
        f"text_encoder/{TEXT_SHARDS[0]}": "",
        f"text_encoder/{TEXT_SHARDS[1]}": "",
        "transformer/config.json": "{}",
        "transformer/diffusion_pytorch_model.safetensors": "",
        "vae/config.json": "{}",
        "vae/diffusion_pytorch_model.safetensors": "",
    }
    for name, text in files.items():
        if name in skip:
            continue
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index = directory / "model_index.json"
    index.write_text(json.dumps(MODEL_INDEX), encoding="utf-8")
    return index


@pytest.fixture
def base_settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(*SMALL, f"paths.hf_home={tmp_path / 'hf'}")


@pytest.fixture
def settings(make_settings: Callable[..., Settings], base_settings: Settings) -> Settings:
    """Pinned to a complete fake snapshot."""
    gen = load_timing(base_settings).generation
    index = _write_snapshot(snapshot_dir(gen, base_settings.paths))
    return make_settings(
        *SMALL,
        f"paths.hf_home={base_settings.paths.hf_home}",
        f"timing.generation.sha256={file_sha256(index)}",
    )


def _gen(settings: Settings) -> GenerationSettings:
    return load_timing(settings).generation


def _ctx(settings: Settings, tmp_path: Path) -> TimingContext:
    return TimingContext(settings, "cuda", 4, tmp_path / "run")


class Rig:
    """A target over a FakePipe, with the fakes a test inspects."""

    def __init__(
        self,
        make_services: Callable[..., TimingServices],
        fail_at: dict[int, Exception] | None = None,
        find_module: Callable[[str], bool] = lambda name: True,
    ) -> None:
        self.clock = FakeClock()
        self.memory = FakeCudaMemory(512.0, 2048.0, 4096.0)
        self.pipe = FakePipe(self.clock, fail_at)
        self.syncs = Counter()
        self.empty_cache = Counter()
        self.loads: list[tuple[Path, GenerationSettings, str]] = []
        self.seeds: list[tuple[str, int]] = []
        services = make_services(
            clock=self.clock,
            memory=lambda device: self.memory,
            sync=self.syncs,
            empty_cache=self.empty_cache,
        )
        self.target = Wan21Generate(
            services,
            pipe_factory=self._load,
            generator_factory=self._generator,
            find_module=find_module,
        )

    def _load(self, directory: Path, gen: GenerationSettings, device: str) -> FakePipe:
        self.loads.append((directory, gen, device))
        self.clock.advance(LOAD_S)
        return self.pipe

    def _generator(self, device: str, seed: int) -> str:
        self.seeds.append((device, seed))
        return f"generator-{seed}"


@pytest.fixture
def rig(make_services: Callable[..., TimingServices]) -> Rig:
    return Rig(make_services)


def _prepared(rig: Rig, ctx: TimingContext) -> list[ClipManifest]:
    clips = list(rig.target.own_clips(ctx))
    rig.target.prepare(clips, ctx)
    return clips


def test_unavailable_without_diffusers(
    make_services: Callable[..., TimingServices], settings: Settings, tmp_path: Path
) -> None:
    rig = Rig(make_services, find_module=lambda name: name != "diffusers")
    assert rig.target.unavailable(_ctx(settings, tmp_path)) == MISSING_DIFFUSERS


def test_available_with_a_complete_pinned_snapshot(
    rig: Rig, settings: Settings, tmp_path: Path
) -> None:
    assert rig.target.unavailable(_ctx(settings, tmp_path)) is None


def test_unavailable_without_snapshot(rig: Rig, base_settings: Settings, tmp_path: Path) -> None:
    reason = rig.target.unavailable(_ctx(base_settings, tmp_path))
    assert reason is not None
    assert "not found" in reason and FETCH_HINT in reason


def test_unavailable_with_another_model_index(
    rig: Rig, base_settings: Settings, tmp_path: Path
) -> None:
    _write_snapshot(snapshot_dir(_gen(base_settings), base_settings.paths))
    reason = rig.target.unavailable(_ctx(base_settings, tmp_path))
    assert reason is not None
    assert "does not match the pinned sha256" in reason


@pytest.mark.parametrize(
    ("skip", "missing"),
    [
        ((f"text_encoder/{TEXT_SHARDS[1]}",), f"text_encoder/{TEXT_SHARDS[1]}"),
        (("vae/diffusion_pytorch_model.safetensors",), "vae/*.safetensors"),
        (("tokenizer/tokenizer_config.json",), "tokenizer/"),
    ],
)
def test_unavailable_with_an_incomplete_snapshot(
    rig: Rig,
    make_settings: Callable[..., Settings],
    base_settings: Settings,
    tmp_path: Path,
    skip: tuple[str, ...],
    missing: str,
) -> None:
    index = _write_snapshot(snapshot_dir(_gen(base_settings), base_settings.paths), skip=skip)
    settings = make_settings(
        *SMALL,
        f"paths.hf_home={base_settings.paths.hf_home}",
        f"timing.generation.sha256={file_sha256(index)}",
    )
    reason = rig.target.unavailable(_ctx(settings, tmp_path))
    assert reason is not None
    assert reason.startswith("incomplete Wan2.1 snapshot (1 missing: ")
    assert missing in reason and FETCH_HINT in reason


def test_weight_spec_is_the_pinned_revision(settings: Settings) -> None:
    gen = _gen(settings)
    spec = wan_weight_spec(gen)
    assert (spec.kind, spec.ref, spec.revision, spec.sha256) == (
        WeightKind.HF,
        gen.model_ref,
        gen.revision,
        gen.sha256,
    )


def test_own_clips_are_the_configured_prompts(rig: Rig, settings: Settings, tmp_path: Path) -> None:
    gen = _gen(settings)
    clips = rig.target.own_clips(_ctx(settings, tmp_path))
    directory = settings.paths.raw_dir / LocalVideoDataset.name / gen.subset
    assert [clip.clip_id for clip in clips] == [prompt.id for prompt in gen.prompts]
    for clip, prompt in zip(clips, gen.prompts, strict=True):
        assert clip.dataset == LocalVideoDataset.name
        assert clip.video_path == str((directory / f"{prompt.id}.mp4").resolve())
        assert (clip.n_frames, clip.width, clip.height) == (FRAMES, WIDTH, HEIGHT)
        assert clip.container_fps == gen.fps and clip.seconds_per_frame == 1.0 / gen.fps
        assert clip.time_source is TimeSource.CONTAINER_FPS
        assert clip.object_queries == to_queries(prompt.queries)
        assert clip.prompt == prompt.text


def test_prepare_times_the_load_and_removes_a_stale_manifest(
    rig: Rig, settings: Settings, tmp_path: Path
) -> None:
    ctx = _ctx(settings, tmp_path)
    clips = list(rig.target.own_clips(ctx))
    stale = Path(clips[0].video_path).parent / MANIFEST_FILE
    stale.parent.mkdir(parents=True)
    stale.write_text('{"clips": []}', encoding="utf-8")
    measures = rig.target.prepare(clips, ctx)
    assert measures == {GENERATE_LOAD_S: LOAD_S, GENERATE_LOAD_CUDA_MB: 512.0}
    assert rig.loads == [(snapshot_dir(_gen(settings), settings.paths), _gen(settings), "cuda")]
    assert not stale.exists()
    assert rig.memory.resets == 1


def test_run_measures_every_batch_size(rig: Rig, settings: Settings, tmp_path: Path) -> None:
    ctx = _ctx(settings, tmp_path)
    clips = _prepared(rig, ctx)
    gen = _gen(settings)
    measures = rig.target.run(clips[1], ctx)
    check_measures(measures)
    wall = ENCODE_S + STEPS * STEP_S + DECODE_S
    for batch, peak in ((1, 2048.0), (4, 4096.0)):
        expected = {
            "wall_s": wall,
            "per_video_s": wall / batch,
            "first_step_s": ENCODE_S + STEP_S,
            "step_s": STEP_S,
            "decode_s": DECODE_S,
            "cpu_s": 0.0,
            "cuda_peak_mb": peak,
        }
        for leaf, value in expected.items():
            assert measures[generate_measure(batch, leaf)] == pytest.approx(value)
    assert measures[generate_measure(1, "write_s")] == 0.0
    assert rig.pipe.calls == [
        {**call_kwargs(gen.prompts[1].text, gen, batch), "generator": f"generator-{gen.seed + 1}"}
        for batch in gen.batch_sizes
    ]
    assert rig.seeds == [("cuda", gen.seed + 1)] * len(gen.batch_sizes)
    assert rig.syncs.count == 1 + len(gen.batch_sizes) * (2 + STEPS)


def test_call_kwargs_follow_the_configuration(settings: Settings) -> None:
    gen = _gen(settings)
    kwargs = call_kwargs("a ball falls", gen, 4)
    assert kwargs == {
        "prompt": "a ball falls",
        "negative_prompt": gen.negative_prompt,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": gen.guidance_scale,
        "num_videos_per_prompt": 4,
        "output_type": "np",
    }


def test_generated_videos_form_a_local_video_subset(
    rig: Rig, settings: Settings, tmp_path: Path
) -> None:
    ctx = _ctx(settings, tmp_path)
    clips = _prepared(rig, ctx)
    for clip in (clips[2], clips[0], clips[2]):
        rig.target.run(clip, ctx)
    listed = LocalVideoDataset(settings).clips(_gen(settings).subset)
    assert [clip.clip_id for clip in listed] == [clips[2].clip_id, clips[0].clip_id]
    for got, own in zip(listed, (clips[2], clips[0]), strict=True):
        assert got.video_path == own.video_path
        assert (got.n_frames, got.width, got.height) == (FRAMES, WIDTH, HEIGHT)
        assert got.container_fps == pytest.approx(own.container_fps)
        assert got.object_queries == own.object_queries
    assert not list(Path(clips[0].video_path).parent.glob(".*"))


def test_out_of_memory_above_batch_one_drops_that_batch_size(
    make_services: Callable[..., TimingServices], settings: Settings, tmp_path: Path
) -> None:
    rig = Rig(make_services, fail_at={4: OutOfMemoryError("CUDA out of memory. Tried 2 GiB")})
    ctx = _ctx(settings, tmp_path)
    clips = _prepared(rig, ctx)
    first = rig.target.run(clips[0], ctx)
    second = rig.target.run(clips[1], ctx)
    for measures in (first, second):
        assert generate_measure(1, "wall_s") in measures
        assert not [name for name in measures if name.startswith("generate.b4.")]
    assert [call["num_videos_per_prompt"] for call in rig.pipe.calls] == [1, 4, 1]
    assert rig.empty_cache.count == 1
    note = rig.target.describe(ctx)["oom.b4"]
    assert note.startswith("out of memory") and "Tried 2 GiB" in note


@pytest.mark.parametrize(
    ("fail_at", "error"),
    [
        ({1: OutOfMemoryError("CUDA out of memory")}, OutOfMemoryError),
        ({4: ValueError("bad prompt")}, ValueError),
    ],
)
def test_other_failures_propagate(
    make_services: Callable[..., TimingServices],
    settings: Settings,
    tmp_path: Path,
    fail_at: dict[int, Exception],
    error: type[Exception],
) -> None:
    rig = Rig(make_services, fail_at=fail_at)
    ctx = _ctx(settings, tmp_path)
    clips = _prepared(rig, ctx)
    with pytest.raises(error):
        rig.target.run(clips[0], ctx)
    assert rig.empty_cache.count == 0


def test_frames_of_the_wrong_shape_fail(rig: Rig, settings: Settings, tmp_path: Path) -> None:
    ctx = _ctx(settings, tmp_path)
    clips = _prepared(rig, ctx)
    rig.pipe.frames_shape = (1, 3, FRAMES, HEIGHT, WIDTH)
    with pytest.raises(TimingError, match="RGB videos"):
        rig.target.run(clips[0], ctx)


def test_run_needs_prepare_and_a_configured_prompt(
    rig: Rig, settings: Settings, tmp_path: Path, clip_factory: Callable[..., ClipManifest]
) -> None:
    ctx = _ctx(settings, tmp_path)
    clips = list(rig.target.own_clips(ctx))
    with pytest.raises(TimingError, match="before prepare"):
        rig.target.run(clips[0], ctx)
    rig.target.prepare(clips, ctx)
    with pytest.raises(TimingError, match="not a configured prompt"):
        rig.target.run(clip_factory("other"), ctx)


def test_describe_and_close(rig: Rig, settings: Settings, tmp_path: Path) -> None:
    ctx = _ctx(settings, tmp_path)
    clips = _prepared(rig, ctx)
    gen = _gen(settings)
    described = rig.target.describe(ctx)
    assert described["model"] == gen.model_ref and described["revision"] == gen.revision
    assert described["resolution"] == f"{WIDTH}x{HEIGHT}"
    assert described["batch_sizes"] == ",".join(map(str, gen.batch_sizes))
    assert described["subset"] == f"local_video/{gen.subset}"
    rig.target.close()
    assert rig.empty_cache.count == 1
    with pytest.raises(TimingError, match="before prepare"):
        rig.target.run(clips[0], ctx)


def test_step_timer_syncs_and_marks_every_step() -> None:
    clock, syncs = FakeClock(), Counter()
    timer = StepTimer(syncs, clock)
    tensors = {"latents": 1}
    for _ in range(3):
        clock.advance(2.0)
        assert timer(None, 0, 0, tensors) is tensors
    assert timer.marks == (2.0, 4.0, 6.0)
    assert syncs.count == 3


def test_step_measures() -> None:
    assert step_measures(1.0, [2.0, 2.5, 3.5, 4.0], 7.0) == {
        "first_step_s": 1.0,
        "decode_s": 3.0,
        "step_s": 0.5,
    }
    assert step_measures(0.0, [1.0], 2.0) == {"first_step_s": 1.0, "decode_s": 1.0}
    with pytest.raises(TimingError):
        step_measures(0.0, [], 1.0)


def test_is_out_of_memory() -> None:
    assert is_out_of_memory(OutOfMemoryError("anything"))
    assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert not is_out_of_memory(RuntimeError("shape mismatch"))


def test_to_uint8() -> None:
    raw = np.array([[0, 7, 255]], dtype=np.uint8)
    assert to_uint8(raw) is raw
    scaled = to_uint8(np.array([-0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32))
    assert scaled.dtype == np.uint8
    assert scaled.tolist() == [0, 0, 128, 255, 255]
    with pytest.raises(TimingError):
        to_uint8(np.zeros(3, dtype=np.int16))  # type: ignore[arg-type]


class FakeModel:
    def __init__(self) -> None:
        self.scheduler: Any = SimpleNamespace(config={"name": "flow"})
        self.device: str | None = None

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self


class FakeLoader:
    """``from_pretrained`` that records its call and whether the hub was offline."""

    def __init__(self, loaded: list[tuple[str, Path, dict[str, Any], bool]], name: str) -> None:
        self._loaded = loaded
        self._name = name

    def from_pretrained(self, directory: Path, **kwargs: Any) -> FakeModel:
        from huggingface_hub import constants

        self._loaded.append((self._name, directory, kwargs, constants.HF_HUB_OFFLINE))
        return FakeModel()


@pytest.fixture
def fake_libraries(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path, dict[str, Any], bool]]:
    loaded: list[tuple[str, Path, dict[str, Any], bool]] = []
    diffusers = ModuleType("diffusers")
    diffusers.AutoencoderKLWan = FakeLoader(loaded, "vae")  # type: ignore[attr-defined]
    diffusers.WanPipeline = FakeLoader(loaded, "pipe")  # type: ignore[attr-defined]
    diffusers.UniPCMultistepScheduler = SimpleNamespace(  # type: ignore[attr-defined]
        from_config=lambda config, **kwargs: ("unipc", config, kwargs)
    )
    torch = ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "fp32"  # type: ignore[attr-defined]
    torch.Generator = lambda device: SimpleNamespace(  # type: ignore[attr-defined]
        manual_seed=lambda seed: (device, seed)
    )
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return loaded


def test_load_wan_pipeline_is_offline_and_configured(
    fake_libraries: list[tuple[str, Path, dict[str, Any], bool]],
    settings: Settings,
    tmp_path: Path,
) -> None:
    gen = _gen(settings)
    pipe = load_wan_pipeline(tmp_path, gen, "cuda")
    (vae_name, vae_dir, vae_kwargs, vae_offline), (pipe_name, pipe_dir, pipe_kwargs, offline) = (
        fake_libraries
    )
    assert (vae_name, vae_dir, vae_offline, vae_kwargs) == (
        "vae",
        tmp_path,
        True,
        {"subfolder": "vae", "torch_dtype": "fp32"},
    )
    assert (pipe_name, pipe_dir, offline, pipe_kwargs["torch_dtype"]) == (
        "pipe",
        tmp_path,
        True,
        "bf16",
    )
    assert isinstance(pipe_kwargs["vae"], FakeModel)
    assert pipe.scheduler == ("unipc", {"name": "flow"}, {"flow_shift": gen.flow_shift})
    assert pipe.device == "cuda"


def test_torch_generator_is_seeded(
    fake_libraries: list[tuple[str, Path, dict[str, Any], bool]],
) -> None:
    assert torch_generator("cuda", 7) == ("cuda", 7)
