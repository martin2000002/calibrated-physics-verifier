"""Metric Video Depth Anything Small.

Its output is depth in the units of its training scenes (VKITTI, IRS); on other footage that
scale is an estimate with no measured error, so the map is stored as relative depth, never
as metres.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ClassVar

from physverify.extraction._base import StageContext, WeightKind, WeightSpec
from physverify.extraction.providers.depth_vda_small import (
    CODE,
    VideoDepthAnythingSmall,
    load_model,
)
from physverify.schema.perception import DepthKind

METRIC_CHECKPOINT = WeightSpec(
    WeightKind.URL,
    "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/"
    "273d090f2ce17df50c2872d82c8322c45da5b4dd/metric_video_depth_anything_vits.pth",
    "273d090f2ce17df50c2872d82c8322c45da5b4dd",
    "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33",
)


class VideoDepthAnythingMetricSmall(VideoDepthAnythingSmall):
    name: ClassVar[str] = "vda_metric_small"
    version: ClassVar[str] = "1"
    weights: ClassVar[tuple[WeightSpec, ...]] = (CODE, METRIC_CHECKPOINT)

    def loader(self) -> Callable[[StageContext], Any]:
        return functools.partial(
            load_model, checkpoint=self.checkpoint(), metric=True, provider="depth/vda_metric_small"
        )

    def depth_kind(self) -> DepthKind:
        return DepthKind.RELATIVE_DEPTH


PLUGINS = (VideoDepthAnythingMetricSmall,)
