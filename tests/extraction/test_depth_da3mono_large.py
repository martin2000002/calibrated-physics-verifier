"""depth/da3mono_large with a fake relative-depth model injected through ``loader()`` (no torch,
no weights): the stored map is the reciprocal of the model's depth, as affine disparity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from support import make_context, run_provider, sampling_of
from synthetic_clip import SyntheticClip

from physverify.core.errors import PluginError
from physverify.extraction._base import StageContext, WeightKind
from physverify.extraction.cache import stage_key
from physverify.extraction.plan import StagePlan
from physverify.extraction.providers import _da3, depth_da3mono_large
from physverify.extraction.providers._da3 import (
    CODE,
    DepthModel,
    reduce_frames,
    to_disparity,
    unprefixed,
)
from physverify.extraction.providers.depth_da3mono_large import CHECKPOINT, DA3MonoLarge, Params
from physverify.extraction.providers.depth_vda_large import VideoDepthAnythingLarge
from physverify.schema.error_profile import EntryFilter
from physverify.schema.perception import DepthKind, DepthMap
from physverify.schema.stages import Stage

DEPTH = 4.0


@dataclass
class FakeDA3:
    """Depth DEPTH everywhere except pixel (0, 0): zero, below the valid range."""

    shapes: list[tuple[int, ...]] = field(default_factory=list)

    def relative_depth(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        self.shapes.append(frame.shape)
        depth = np.full(frame.shape[:2], DEPTH, dtype=np.float32)
        depth[0, 0] = 0.0
        return depth


def faked(model: DepthModel) -> type[DA3MonoLarge]:
    class Faked(DA3MonoLarge):
        def loader(self) -> Callable[[StageContext], DepthModel]:
            return lambda ctx: model

    return Faked


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
    calls: dict[str, list[int]] = {"torch": [], "cv2": []}
    monkeypatch.setattr(depth_da3mono_large, "set_torch_threads", calls["torch"].append)
    monkeypatch.setattr(depth_da3mono_large, "set_cv2_threads", calls["cv2"].append)
    return calls


def run(clip: SyntheticClip, model: FakeDA3, **params: Any) -> DepthMap:
    out = run_provider(faked(model), make_context(clip), **params)
    assert isinstance(out, DepthMap)
    return out


def test_affine_disparity_of_one_model_call_per_frame(
    fixture_clip: SyntheticClip, threads: dict[str, list[int]]
) -> None:
    model = FakeDA3()
    out = run(fixture_clip, model)
    s = out.sampling
    assert out.kind is DepthKind.AFFINE_DISPARITY
    assert out.depth.shape == (s.n_frames, s.height, s.width)
    assert model.shapes == [(s.height, s.width, 3)] * s.n_frames
    assert not out.valid[:, 0, 0].any()
    assert out.valid.sum() == s.n_frames * (s.height * s.width - 1)
    np.testing.assert_allclose(out.depth[out.valid], 1.0 / DEPTH)


def test_the_injected_thread_count_bounds_torch_and_opencv(
    fixture_clip: SyntheticClip, threads: dict[str, list[int]]
) -> None:
    ctx = make_context(fixture_clip)
    run_provider(faked(FakeDA3()), ctx)
    assert threads == {"torch": [ctx.threads], "cv2": [ctx.threads]}


def test_set_cv2_threads_sets_the_opencv_pool() -> None:
    before = cv2.getNumThreads()
    try:
        _da3.set_cv2_threads(1)
        assert cv2.getNumThreads() == 1
    finally:
        cv2.setNumThreads(before)


def test_frames_above_the_input_side_are_reduced_before_the_model(
    fixture_clip: SyntheticClip, threads: dict[str, list[int]]
) -> None:
    model = FakeDA3()
    s = sampling_of(fixture_clip.manifest)
    side = max(s.width, s.height) // 2
    out = run(fixture_clip, model, max_input_side=side)
    assert max(model.shapes[0][:2]) == side
    assert out.depth.shape == (s.n_frames, s.height, s.width)


def test_reduce_frames_keeps_small_frames_and_area_downsamples_large_ones() -> None:
    small = np.zeros((2, 256, 256, 3), dtype=np.uint8)
    assert reduce_frames(small, 512) is small
    large = np.full((2, 540, 960, 3), 7, dtype=np.uint8)
    reduced = reduce_frames(large, 512)
    assert reduced.shape == (2, 288, 512, 3)
    assert (reduced == 7).all()


def test_disparity_is_the_reciprocal_of_valid_depth_only() -> None:
    depth = np.array([[2.0, 0.0, -1.0, np.inf, np.nan, 1e-7]], dtype=np.float32)
    disparity = to_disparity(depth)
    assert disparity[0, 0] == pytest.approx(0.5)
    assert np.isnan(disparity[0, 1:]).all()
    assert disparity.dtype == np.float32


def test_process_res_is_a_multiple_of_the_patch_size() -> None:
    assert Params().process_res % 14 == 0
    with pytest.raises(ValueError, match="multiple"):
        Params(process_res=500)


def test_checkpoint_keys_lose_their_model_prefix_and_strays_are_refused() -> None:
    assert unprefixed({"model.head.w": 1, "model.backbone.b": 2}, "depth/x") == {
        "head.w": 1,
        "backbone.b": 2,
    }
    with pytest.raises(PluginError, match="depth/x"):
        unprefixed({"model.head.w": 1, "aux.w": 2}, "depth/x")


def test_code_and_checkpoint_are_pinned() -> None:
    assert [spec.kind for spec in DA3MonoLarge.weights] == [WeightKind.GIT, WeightKind.HF]
    assert DA3MonoLarge.weights == (CODE, CHECKPOINT)
    assert len(CODE.revision) == 40
    assert len(CHECKPOINT.revision) == 40
    assert len(CHECKPOINT.sha256) == 64


def key_of(clip: SyntheticClip, provider: type[Any], params: Any, device: str = "cuda") -> str:
    return stage_key(
        StagePlan(Stage.DEPTH, provider, params),
        clip_id=clip.manifest.clip_id,
        object_queries=clip.manifest.object_queries,
        sampling=sampling_of(clip.manifest),
        upstream={},
        plan={Stage.DEPTH: provider.name},
        entry_filter=EntryFilter(),
        gt_version=None,
        error_profile=None,
        device=device,
        dataset=None,
    )


def test_the_cache_key_follows_params_and_device_and_differs_from_vda_large(
    fixture_clip: SyntheticClip,
) -> None:
    base = key_of(fixture_clip, DA3MonoLarge, Params())
    assert base == key_of(fixture_clip, DA3MonoLarge, Params())
    assert base != key_of(fixture_clip, DA3MonoLarge, Params(process_res=518))
    assert base != key_of(fixture_clip, DA3MonoLarge, Params(max_input_side=960))
    assert base != key_of(fixture_clip, DA3MonoLarge, Params(), device="cpu")
    vda = VideoDepthAnythingLarge
    assert base != key_of(fixture_clip, vda, vda.Params())
