"""depth/moge2 with a fake point-map model injected through ``loader()`` (no torch, no weights)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray
from support import make_context, run_provider
from synthetic_clip import SyntheticClip

from physverify.core.errors import PluginError
from physverify.extraction._base import StageContext, WeightKind
from physverify.extraction.providers import depth_moge2
from physverify.extraction.providers._moge2 import (
    CHECKPOINT,
    FrameEstimate,
    PointMapModel,
    check_load_warnings,
)
from physverify.extraction.providers.depth_moge2 import MoGe2Depth
from physverify.schema.perception import DepthKind, DepthMap, horizontal_fov_deg

DEPTH_M = 3.25
FREE_FOCAL_REL = (0.80, 0.90, 1.00)


@dataclass
class FakeMoGe:
    """Metric z constant except one masked pixel; a free focal cycles through FREE_FOCAL_REL
    frame after frame, across batches."""

    calls: list[tuple[int, float | None]] = field(default_factory=list)
    free_frames: int = 0

    def estimate(self, frames: NDArray[np.uint8], fov_x_deg: float | None) -> FrameEstimate:
        self.calls.append((frames.shape[0], fov_x_deg))
        n, h, w = frames.shape[:3]
        depth = np.full((n, h, w), DEPTH_M, dtype=np.float32)
        depth[:, 0, 0] = np.inf
        if fov_x_deg is None:
            cycle = np.arange(self.free_frames, self.free_frames + n) % len(FREE_FOCAL_REL)
            focal = np.array(FREE_FOCAL_REL)[cycle]
            self.free_frames += n
        else:
            focal = np.full(n, 0.5 / math.tan(math.radians(fov_x_deg) / 2.0))
        return FrameEstimate(depth=depth, focal_rel=focal)


def faked(model: PointMapModel) -> type[MoGe2Depth]:
    class Faked(MoGe2Depth):
        def loader(self) -> Callable[[StageContext], PointMapModel]:
            return lambda ctx: model

    return Faked


@pytest.fixture(autouse=True)
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(depth_moge2, "set_torch_threads", lambda threads: None)


def run(clip: SyntheticClip, model: FakeMoGe, **params: int) -> DepthMap:
    out = run_provider(faked(model), make_context(clip), **params)
    assert isinstance(out, DepthMap)
    return out


def test_metric_z_with_the_shared_focal_in_processing_pixels(fixture_clip: SyntheticClip) -> None:
    out = run(fixture_clip, FakeMoGe())
    s = out.sampling
    assert out.kind is DepthKind.METRIC_Z
    assert out.depth.shape == (s.n_frames, s.height, s.width)
    assert not out.valid[:, 0, 0].any()
    assert out.valid[:, 1:, 1:].all()
    np.testing.assert_allclose(out.depth[out.valid], DEPTH_M)
    assert out.focal_px == pytest.approx(np.median(FREE_FOCAL_REL) * s.width)
    assert math.isnan(out.focal_sigma_px)


def test_the_probe_pass_fixes_the_field_of_view(fixture_clip: SyntheticClip) -> None:
    model = FakeMoGe()
    out = run(fixture_clip, model, shared_focal_frames=3, batch_size=2)
    probe = [call for call in model.calls if call[1] is None]
    fixed = [call for call in model.calls if call[1] is not None]
    assert sum(n for n, _ in probe) == 3
    assert sum(n for n, _ in fixed) == out.sampling.n_frames
    assert all(n <= 2 for n, _ in model.calls)
    expected_fov = horizontal_fov_deg(float(np.median(np.resize(np.array(FREE_FOCAL_REL), 3))), 1.0)
    assert all(fov == pytest.approx(expected_fov) for _, fov in fixed)


def test_zero_probe_frames_keeps_the_per_frame_focal(fixture_clip: SyntheticClip) -> None:
    model = FakeMoGe()
    out = run(fixture_clip, model, shared_focal_frames=0)
    assert all(fov is None for _, fov in model.calls)
    focal = np.resize(np.array(FREE_FOCAL_REL), out.sampling.n_frames)
    assert out.focal_px == pytest.approx(float(np.median(focal)) * out.sampling.width)


def test_a_checkpoint_that_does_not_match_the_model_is_refused() -> None:
    check_load_warnings(["torch.meshgrid: in an upcoming release"], "depth/moge2")
    with pytest.raises(PluginError, match="depth/moge2"):
        check_load_warnings(
            ["3 parameter(s) are absent from the checkpoint and keep ..."], "depth/moge2"
        )
    with pytest.raises(PluginError, match="depth/moge2"):
        check_load_warnings(
            ["2 parameter(s) in the checkpoint have no counterpart in the model"], "depth/moge2"
        )


def test_weights_are_pinned() -> None:
    assert CHECKPOINT.kind is WeightKind.HF
    assert len(CHECKPOINT.sha256) == 64
    assert all(spec.revision for spec in MoGe2Depth.weights)
    assert [spec.kind for spec in MoGe2Depth.weights] == [
        WeightKind.GIT,
        WeightKind.GIT,
        WeightKind.HF,
    ]
