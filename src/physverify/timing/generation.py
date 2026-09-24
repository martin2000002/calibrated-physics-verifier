"""The thesis generator (Wan2.1-T2V-1.3B via diffusers) for the H200 benchmark (ADR 0008, D7-D9).

``diffusers`` and ``torch`` are imported with importlib inside functions, so mypy and the plugin
registry never need them; nothing here runs off the H200.
"""

from __future__ import annotations

import importlib
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from physverify.config.models import PathsSettings
from physverify.extraction._base import WeightKind, WeightSpec
from physverify.extraction.weights import hub_offline, local_path, sha256_of, weight_file
from physverify.timing.records import TimingError
from physverify.timing.settings import GenerationSettings

SHARD_INDEX_SUFFIX: Final = ".safetensors.index.json"
WEIGHTS_SUFFIX: Final = ".safetensors"
MODEL_CONFIG: Final = "config.json"
MAX_LISTED_MISSING: Final = 3
FETCH_HINT: Final = "run `physverify timing fetch` (it resumes a cut download)"


def wan_weight_spec(gen: GenerationSettings) -> WeightSpec:
    """The pinned snapshot, identified by its ``model_index.json`` (revision and sha256)."""
    return WeightSpec(
        kind=WeightKind.HF, ref=gen.model_ref, revision=gen.revision, sha256=gen.sha256
    )


def snapshot_dir(gen: GenerationSettings, paths: PathsSettings) -> Path:
    return local_path(wan_weight_spec(gen), paths)


def components(model_index: Mapping[str, Any]) -> list[str]:
    """Sub-folders of a diffusers snapshot: public keys of ``model_index.json`` naming a class."""
    return [
        key
        for key, value in model_index.items()
        if not key.startswith("_") and isinstance(value, list) and value and value[-1] is not None
    ]


def _shards(index: Path) -> list[str]:
    weight_map: Mapping[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    return sorted(set(weight_map.values()))


def component_missing(snapshot: Path, name: str) -> list[str]:
    """Missing files of one component: every shard of its ``*.safetensors.index.json`` or, for an
    unsharded model (a ``config.json``), its weights file.

    A hub download writes each snapshot entry only once the file is complete, so an entry that
    exists is whole."""
    folder = snapshot / name
    if not folder.is_dir() or not any(folder.iterdir()):
        return [f"{name}/"]
    indexes = sorted(folder.glob(f"*{SHARD_INDEX_SUFFIX}"))
    if indexes:
        return [
            f"{name}/{shard}"
            for index in indexes
            for shard in _shards(index)
            if not (folder / shard).is_file()
        ]
    weights = [path for path in folder.glob(f"*{WEIGHTS_SUFFIX}") if path.is_file()]
    if (folder / MODEL_CONFIG).is_file() and not weights:
        return [f"{name}/*{WEIGHTS_SUFFIX}"]
    return []


def missing_files(snapshot: Path, model_index: Path) -> list[str]:
    index = json.loads(model_index.read_text(encoding="utf-8"))
    return [path for name in components(index) for path in component_missing(snapshot, name)]


def snapshot_problem(gen: GenerationSettings, paths: PathsSettings) -> str | None:
    """Why the pinned snapshot cannot be loaded (absent, wrong pin, incomplete), or None."""
    spec = wan_weight_spec(gen)
    index = weight_file(spec, paths)
    if not index.is_file():
        return f"Wan2.1 weights not found in {local_path(spec, paths)}: {FETCH_HINT}"
    if sha256_of(index) != spec.sha256:
        return f"{index} does not match the pinned sha256: {FETCH_HINT}"
    missing = missing_files(local_path(spec, paths), index)
    if missing:
        listed = ", ".join(missing[:MAX_LISTED_MISSING])
        return f"incomplete Wan2.1 snapshot ({len(missing)} missing: {listed}): {FETCH_HINT}"
    return None


class StepTimer:
    """``callback_on_step_end`` of a diffusers pipeline: syncs the device and marks the end of
    every denoising step. Returns the tensors it was given, unchanged."""

    def __init__(
        self, sync: Callable[[], None], clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self._sync = sync
        self._clock = clock
        self._marks: list[float] = []

    def __call__(
        self, pipe: Any, step: int, timestep: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        self._sync()
        self._marks.append(self._clock())
        return kwargs

    @property
    def marks(self) -> tuple[float, ...]:
        return tuple(self._marks)


def step_measures(t0: float, marks: Sequence[float], t_end: float) -> dict[str, float]:
    """``first_step_s`` = text encoding plus the first step; ``step_s`` = median of the later step
    durations (needs two marks); ``decode_s`` = VAE decode and post-processing."""
    if not marks:
        raise TimingError("the pipeline reported no denoising step")
    measures = {"first_step_s": marks[0] - t0, "decode_s": t_end - marks[-1]}
    if len(marks) > 1:
        measures["step_s"] = statistics.median(b - a for a, b in pairwise(marks))
    return measures


def call_kwargs(prompt: str, gen: GenerationSettings, batch_size: int) -> dict[str, Any]:
    """Arguments of ``WanPipeline.__call__`` (besides generator and callback); frames come back as
    float arrays ``(B, T, H, W, 3)`` in [0, 1]."""
    return {
        "prompt": prompt,
        "negative_prompt": gen.negative_prompt,
        "height": gen.height,
        "width": gen.width,
        "num_frames": gen.num_frames,
        "num_inference_steps": gen.num_inference_steps,
        "guidance_scale": gen.guidance_scale,
        "num_videos_per_prompt": batch_size,
        "output_type": "np",
    }


def torch_generator(device: str, seed: int) -> Any:
    torch = importlib.import_module("torch")
    return torch.Generator(device=device).manual_seed(seed)


def load_wan_pipeline(directory: Path, gen: GenerationSettings, device: str) -> Any:
    """WanPipeline from the local snapshot (hub offline), VAE in ``vae_dtype``, UniPC scheduler
    with the configured ``flow_shift``, moved to ``device``."""
    diffusers = importlib.import_module("diffusers")
    torch = importlib.import_module("torch")
    with hub_offline():
        vae = diffusers.AutoencoderKLWan.from_pretrained(
            directory, subfolder="vae", torch_dtype=getattr(torch, gen.vae_dtype)
        )
        pipe = diffusers.WanPipeline.from_pretrained(
            directory, vae=vae, torch_dtype=getattr(torch, gen.dtype)
        )
    pipe.scheduler = diffusers.UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=gen.flow_shift
    )
    return pipe.to(device)


def is_out_of_memory(exc: BaseException) -> bool:
    """torch's OutOfMemoryError (any module) or an older "CUDA out of memory" RuntimeError."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    return "OutOfMemoryError" in names or "out of memory" in str(exc).lower()
