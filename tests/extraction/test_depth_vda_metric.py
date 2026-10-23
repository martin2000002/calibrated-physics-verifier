"""depth/vda_metric_small: metric head, own checkpoint, stored as relative depth (ADR 0014 §2.2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from support import make_context, run_provider
from synthetic_clip import SyntheticClip

from physverify.extraction.providers import depth_vda_metric_small, depth_vda_small
from physverify.extraction.providers.depth_vda_metric_small import (
    METRIC_CHECKPOINT,
    VideoDepthAnythingMetricSmall,
)
from physverify.extraction.providers.depth_vda_small import (
    CHECKPOINT,
    CODE,
    SMALL_ENCODER,
    VideoDepthAnythingSmall,
)
from physverify.schema.perception import DepthKind, DepthMap


class ConstantDepthModel:
    def infer_video_depth(
        self, frames: NDArray[np.uint8], fps: float, **_: Any
    ) -> tuple[NDArray[np.float32], float]:
        return np.full(frames.shape[:3], 2.5, dtype=np.float32), fps


def fake_loads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def load(ctx: Any, **kwargs: Any) -> ConstantDepthModel:
        calls.append(kwargs)
        return ConstantDepthModel()

    for module in (depth_vda_small, depth_vda_metric_small):
        monkeypatch.setattr(module, "load_model", load)
    monkeypatch.setattr(depth_vda_small, "set_torch_threads", lambda threads: None)
    return calls


def test_metric_head_and_checkpoint_are_requested(
    fixture_clip: SyntheticClip, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_loads(monkeypatch)
    out = run_provider(VideoDepthAnythingMetricSmall, make_context(fixture_clip))
    assert calls == [
        {"checkpoint": METRIC_CHECKPOINT, "metric": True, "provider": "depth/vda_metric_small"}
    ]
    assert isinstance(out, DepthMap)
    assert out.kind is DepthKind.RELATIVE_DEPTH
    assert VideoDepthAnythingMetricSmall.weights == (CODE, METRIC_CHECKPOINT)


def test_the_relative_model_keeps_its_defaults(
    fixture_clip: SyntheticClip, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_loads(monkeypatch)
    out = run_provider(VideoDepthAnythingSmall, make_context(fixture_clip))
    assert calls == [
        {"checkpoint": CHECKPOINT, "provider": "depth/vda_small", "encoder": SMALL_ENCODER}
    ]
    assert isinstance(out, DepthMap)
    assert out.kind is DepthKind.AFFINE_DISPARITY
    assert VideoDepthAnythingSmall.weights == (CODE, CHECKPOINT)
