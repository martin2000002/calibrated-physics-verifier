"""depth/vda_large is vda_small with the ViT-L encoder and checkpoint; vda_small keeps its
version, its model and its cache keys, so the two depth providers can be compared side by side."""

from __future__ import annotations

import pytest
from support import make_context, run_provider, sampling_of
from synthetic_clip import SyntheticClip
from test_depth_vda_metric import fake_loads

from physverify.extraction._base import WeightKind
from physverify.extraction.cache import stage_key
from physverify.extraction.plan import StagePlan
from physverify.extraction.providers.depth_vda_large import (
    LARGE_CHECKPOINT,
    LARGE_ENCODER,
    VideoDepthAnythingLarge,
)
from physverify.extraction.providers.depth_vda_small import (
    CHECKPOINT,
    CODE,
    SMALL_ENCODER,
    VideoDepthAnythingSmall,
)
from physverify.schema.error_profile import EntryFilter
from physverify.schema.perception import DepthKind, DepthMap
from physverify.schema.stages import Stage

KEYS_BEFORE_IT3 = {
    (280, "cpu"): "70f6c003ff50af82ae87a48aa6a2c81030761e6a0c859278fc99fd8dba78cf1c",
    (280, "cuda"): "8a1c35b03c5393f34ee9b155aed58dfdeeb34075bbd750b7e1bd2b0652630491",
    (518, "cpu"): "cef8fcd5f4036349655f0830c50e481d21148400a4fded20d1877346cd85412d",
    (518, "cuda"): "dabe5eafcce44da9fb7471e63cd523077c67655fda15ed6d9a2b1250f24acabe",
}
"""vda_small keys of the fixture clip written by 2a762ad (input size, device)."""


def test_the_large_model_requests_its_encoder_and_checkpoint(
    fixture_clip: SyntheticClip, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_loads(monkeypatch)
    out = run_provider(VideoDepthAnythingLarge, make_context(fixture_clip))
    assert calls == [
        {"checkpoint": LARGE_CHECKPOINT, "provider": "depth/vda_large", "encoder": LARGE_ENCODER}
    ]
    assert isinstance(out, DepthMap) and out.kind is DepthKind.AFFINE_DISPARITY
    assert VideoDepthAnythingLarge.weights == (CODE, LARGE_CHECKPOINT)
    assert VideoDepthAnythingLarge.checkpoint() == LARGE_CHECKPOINT
    assert VideoDepthAnythingSmall.checkpoint() == CHECKPOINT
    assert (VideoDepthAnythingLarge.version, VideoDepthAnythingLarge.heavy) == ("1", True)


def test_the_encoders_are_those_of_the_pinned_checkout() -> None:
    """run.py of Video-Depth-Anything 4f5ae23, model_configs (lines 45-49)."""
    before_it3 = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    assert SMALL_ENCODER.kwargs() == before_it3
    vitl = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
    assert LARGE_ENCODER.kwargs() == vitl


def test_the_large_checkpoint_is_pinned_by_revision_and_sha256() -> None:
    assert LARGE_CHECKPOINT.kind is WeightKind.URL
    revision = "7aafbcb5c6af0bac741aad2b6471894fb4761afa"
    assert f"/resolve/{revision}/video_depth_anything_vitl.pth" in LARGE_CHECKPOINT.ref
    assert LARGE_CHECKPOINT.revision == revision
    assert LARGE_CHECKPOINT.sha256 == (
        "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa"
    )
    assert LARGE_CHECKPOINT != CHECKPOINT


@pytest.mark.parametrize(("size", "device"), sorted(KEYS_BEFORE_IT3))
def test_vda_small_keeps_its_keys(fixture_clip: SyntheticClip, size: int, device: str) -> None:
    assert VideoDepthAnythingSmall.version == "1"
    step = StagePlan(
        Stage.DEPTH, VideoDepthAnythingSmall, VideoDepthAnythingSmall.Params(input_size=size)
    )
    key = stage_key(
        step,
        clip_id=fixture_clip.manifest.clip_id,
        object_queries=fixture_clip.manifest.object_queries,
        sampling=sampling_of(fixture_clip.manifest),
        upstream={},
        plan={Stage.DEPTH: "vda_small"},
        entry_filter=EntryFilter(),
        gt_version=None,
        error_profile=None,
        device=device,
    )
    assert key == KEYS_BEFORE_IT3[(size, device)]
