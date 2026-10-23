"""Video Depth Anything Small: temporally consistent affine-invariant disparity.

The model code is a pinned checkout (Apache-2.0) imported from ``paths.vendor_dir``; it imports a
top-level ``utils`` namespace package, so a foreign ``utils`` already loaded is a clear error.
The encoder and the checkpoint are class attributes, so a larger model is a subclass with the
same inference path. The plugin modules import it from here.
"""

from __future__ import annotations

import functools
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.core.errors import PluginError
from physverify.core.resample import resize_depth
from physverify.extraction._base import StageContext, WeightKind, WeightSpec
from physverify.extraction.providers._common import (
    LazyModel,
    ProviderParams,
    expect_params,
    set_torch_threads,
)
from physverify.extraction.weights import local_path, vendored_import_path, weight_file
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import DepthKind, DepthMap
from physverify.schema.stages import Stage

CODE = WeightSpec(
    WeightKind.GIT,
    "https://github.com/DepthAnything/Video-Depth-Anything.git",
    "4f5ae23172ba60fd7bc11ef671cca678842c7072",
)
CHECKPOINT = WeightSpec(
    WeightKind.URL,
    "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/"
    "256875362cff76724b920335dfb4b29dd611f66e/video_depth_anything_vits.pth",
    "256875362cff76724b920335dfb4b29dd611f66e",
    "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
)
VENDORED_TOP_LEVEL = "utils"
VENDORED_MODULE = "video_depth_anything.video_depth"


@dataclass(frozen=True)
class Encoder:
    """Constructor arguments of the vendored ``VideoDepthAnything`` for one ViT size (the
    ``model_configs`` of the checkout's run.py)."""

    name: str
    features: int
    out_channels: tuple[int, int, int, int]

    def kwargs(self) -> dict[str, Any]:
        return {
            "encoder": self.name,
            "features": self.features,
            "out_channels": list(self.out_channels),
        }


SMALL_ENCODER = Encoder("vits", 64, (48, 96, 192, 384))


class Params(ProviderParams):
    input_size: int = Field(default=518, ge=14, multiple_of=14)


class VideoDepthAnythingSmall:
    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = "vda_small"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = (CODE, CHECKPOINT)
    Params: ClassVar[type[BaseModel]] = Params
    encoder: ClassVar[Encoder] = SMALL_ENCODER

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model: LazyModel[Any] = LazyModel(self.loader())

    @classmethod
    def checkpoint(cls) -> WeightSpec:
        """The model checkpoint: the entry of ``weights`` after the code (kept in one place)."""
        (checkpoint,) = (w for w in cls.weights if w != CODE)
        return checkpoint

    def loader(self) -> Callable[[StageContext], Any]:
        return functools.partial(
            load_model,
            checkpoint=self.checkpoint(),
            provider=f"{self.stage.value}/{self.name}",
            encoder=self.encoder,
        )

    def depth_kind(self) -> DepthKind:
        return DepthKind.AFFINE_DISPARITY

    def run(self, ctx: StageContext) -> DepthMap:
        s = ctx.sampling
        frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
        set_torch_threads(ctx.threads)
        model = self._model.get(ctx)
        depths, _ = model.infer_video_depth(
            frames,
            1.0 / (s.seconds_per_frame * s.step),
            input_size=self.params.input_size,
            device=ctx.device,
            fp32=ctx.device == "cpu",
        )
        return to_depth_map(s, np.asarray(depths, dtype=np.float32), self.depth_kind())


def to_depth_map(
    sampling: FrameSampling,
    disparity: NDArray[np.float32],
    kind: DepthKind = DepthKind.AFFINE_DISPARITY,
) -> DepthMap:
    valid = np.isfinite(disparity)
    if disparity.shape[1:] != (sampling.height, sampling.width):
        disparity, valid = resize_depth(disparity, valid, sampling.size)
    return DepthMap(
        sampling=sampling,
        depth=np.where(valid, disparity, 0.0).astype(np.float32),
        kind=kind,
        valid=valid,
    )


def module_locations(module: ModuleType) -> tuple[Path, ...]:
    """Where a module was loaded from: its file, else every portion of a namespace package."""
    file = getattr(module, "__file__", None)
    if file:
        return (Path(file),)
    spec = getattr(module, "__spec__", None)
    portions = getattr(module, "__path__", None) or getattr(
        spec, "submodule_search_locations", None
    )
    return tuple(Path(portion) for portion in portions or ())


def check_vendored_namespace(root: Path, provider: str = "depth/vda_small") -> None:
    """R6: the checkout's top-level ``utils`` must not be shadowed by another package."""
    loaded = sys.modules.get(VENDORED_TOP_LEVEL)
    if loaded is None:
        return
    locations = module_locations(loaded)
    base = root.resolve()
    if not locations or not all(path.resolve().is_relative_to(base) for path in locations):
        found = ", ".join(str(path) for path in locations) or "?"
        raise PluginError(
            f"{provider}: a top-level {VENDORED_TOP_LEVEL!r} package from {found} "
            f"is already imported and would shadow the one in {root}"
        )


def load_model(
    ctx: StageContext,
    checkpoint: WeightSpec = CHECKPOINT,
    metric: bool = False,
    provider: str = "depth/vda_small",
    encoder: Encoder = SMALL_ENCODER,
) -> Any:
    """The vendored model; ``metric`` selects the metric head, which skips the per-window
    scale-and-shift alignment of the relative one (video_depth.py:135)."""
    import torch

    root = local_path(CODE, ctx.paths)
    check_vendored_namespace(root, provider)
    with vendored_import_path(root):
        module = importlib.import_module(VENDORED_MODULE)
    model = module.VideoDepthAnything(**encoder.kwargs(), metric=metric)
    state = torch.load(
        weight_file(checkpoint, ctx.paths), map_location=ctx.device, weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model.to(ctx.device).eval()
