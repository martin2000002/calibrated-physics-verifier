"""The real heavy providers: one model load per stage over many clips, and the vendored
package shadowing check (VDA's ``utils``, DA3's ``depth_anything_3``) that must pass on every
load. Loaders and inference are faked."""

from __future__ import annotations

import importlib
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import DISPARITY_A, affine_disparity, fixture_truth
from numpy.typing import NDArray
from settings_factory import SettingsFactory
from support import plan_overrides, run_pipeline
from synthetic_clip import SyntheticFixtureDataset
from wiring import resolve_plan

from physverify.core.errors import PluginError
from physverify.extraction import pipeline
from physverify.extraction._base import StageProvider
from physverify.extraction.providers import (
    _da3,
    _grounding_dino,
    _sam_video,
    _vda,
    track_edgetam,
    track_sam2_1_hiera_tiny,
)
from physverify.extraction.providers._grounding_dino import Params, RawDetections
from physverify.extraction.providers._vda import VENDORED_TOP_LEVEL, check_vendored_package
from physverify.schema.clip import FrameSampling, ObjectQuery, resolve_sampling
from physverify.schema.perception import Detections
from physverify.schema.stages import STAGE_ORDER, Stage

Providers = Mapping[str, type[StageProvider]]
TWIN = "fixture_twin"
VENDORED_PACKAGES = [VENDORED_TOP_LEVEL, _da3.VENDORED_PACKAGE]


@pytest.fixture
def clean_vendored() -> Iterator[None]:
    """Hides any loaded vendored package during the test and restores it afterwards."""

    def owned() -> list[str]:
        return [
            name
            for name in sys.modules
            if any(name == top or name.startswith(f"{top}.") for top in VENDORED_PACKAGES)
        ]

    saved = {name: sys.modules.pop(name) for name in owned()}
    yield
    for name in owned():
        del sys.modules[name]
    sys.modules.update(saved)


def vendored_package(root: Path, name: str, *, regular: bool) -> Path:
    package = root / name
    package.mkdir(parents=True)
    (package / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    if regular:
        (package / "__init__.py").write_text("", encoding="utf-8")
    return root


@pytest.mark.usefixtures("clean_vendored")
@pytest.mark.parametrize("package", VENDORED_PACKAGES)
@pytest.mark.parametrize("regular", [False, True])
def test_the_vendored_package_passes_the_check_on_every_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str, regular: bool
) -> None:
    root = vendored_package(tmp_path / "vendor", package, regular=regular)
    check_vendored_package(root, package, "depth/x")
    monkeypatch.syspath_prepend(str(root))
    module = importlib.import_module(package)
    assert (getattr(module, "__file__", None) is None) is not regular
    check_vendored_package(root, package, "depth/x")
    check_vendored_package(root, package, "depth/x")


@pytest.mark.usefixtures("clean_vendored")
@pytest.mark.parametrize("package", VENDORED_PACKAGES)
@pytest.mark.parametrize("regular", [False, True])
def test_a_foreign_vendored_package_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str, regular: bool
) -> None:
    foreign = vendored_package(tmp_path / "elsewhere", package, regular=regular)
    monkeypatch.syspath_prepend(str(foreign))
    importlib.import_module(package)
    root = vendored_package(tmp_path / "vendor", package, regular=regular)
    with pytest.raises(PluginError, match=rf"depth/x: a top-level '{package}' .* shadow"):
        check_vendored_package(root, package, "depth/x")


class TruthDepthModel:
    """``infer_video_depth`` returning the fixture's affine disparity (NaN where invalid)."""

    def __init__(self, sampling: FrameSampling) -> None:
        self._sampling = sampling

    def infer_video_depth(
        self, frames: NDArray[np.uint8], fps: float, **_: Any
    ) -> tuple[NDArray[np.float32], float]:
        disparity = affine_disparity(fixture_truth().depth(self._sampling), DISPARITY_A, 0.0)
        assert disparity.depth.shape == frames.shape[:3]
        return np.where(disparity.valid, disparity.depth, np.nan).astype(np.float32), fps


def fake_heavy_models(monkeypatch: pytest.MonkeyPatch, sampling: FrameSampling) -> Counter[Stage]:
    """Counting loaders and truth-backed inference for DETECT, TRACK and DEPTH."""
    loads: Counter[Stage] = Counter()
    truth_boxes = fixture_truth().detections(sampling)
    truth_masks = fixture_truth().masks(sampling)

    def loader(stage: Stage, model: Any) -> Any:
        def load(*_: Any, **__: Any) -> Any:
            loads[stage] += 1
            return model

        return load

    def detect(
        model: Any,
        processor: Any,
        frame: NDArray[np.uint8],
        queries: tuple[ObjectQuery, ...],
        params: Params,
        device: str,
    ) -> RawDetections:
        return RawDetections(
            boxes=truth_boxes.boxes_xyxy,
            scores=truth_boxes.scores,
            texts=tuple(str(label) for label in truth_boxes.labels),
        )

    def track_boxes(
        model: Any, processor: Any, frames: Any, detections: Detections, *_: Any
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        labels = truth_masks.labels.tolist()
        rows = [labels.index(str(label)) for label in detections.labels.tolist()]
        return truth_masks.masks[:, rows], truth_masks.scores[:, rows]

    pairs = (object(), object())
    monkeypatch.setattr(_grounding_dino, "load_detector", loader(Stage.DETECT, pairs))
    monkeypatch.setattr(_grounding_dino, "detect", detect)
    monkeypatch.setattr(_sam_video, "load_video_model", loader(Stage.TRACK, pairs))
    monkeypatch.setattr(_sam_video, "track_boxes", track_boxes)
    depth_model = TruthDepthModel(sampling)
    monkeypatch.setattr(_vda, "load_model", loader(Stage.DEPTH, depth_model))
    for module in (
        _grounding_dino,
        track_edgetam,
        track_sam2_1_hiera_tiny,
        _vda,
    ):
        monkeypatch.setattr(module, "set_torch_threads", lambda threads: None)
    monkeypatch.setattr(pipeline, "missing_weights", lambda providers, paths: [])
    return loads


@pytest.mark.parametrize("tracker", ["edgetam", "sam2_1_hiera_tiny"])
def test_each_heavy_model_loads_once_per_stage_over_two_clips(
    make_settings: SettingsFactory,
    providers: Providers,
    fixture_dataset: SyntheticFixtureDataset,
    monkeypatch: pytest.MonkeyPatch,
    tracker: str,
) -> None:
    names = plan_overrides(detect="grounding_dino_tiny", track=tracker, depth="vda_small")
    settings = make_settings(*names)
    clip = fixture_dataset.clips()[0]
    plan = resolve_plan(settings, providers, dataset=fixture_dataset.name)
    loads = fake_heavy_models(monkeypatch, resolve_sampling(clip, plan.sampling))
    clips = [clip, replace(clip, clip_id=TWIN)]
    report = run_pipeline(
        settings, providers, fixture_dataset, gt_stages=(Stage.CAMERA,), clips=clips
    )
    assert report.failures == ()
    for clip_id in (clip.clip_id, TWIN):
        assert [e.stage for e in report.events if e.clip_id == clip_id] == list(STAGE_ORDER)
    assert loads == {Stage.DETECT: 1, Stage.TRACK: 1, Stage.DEPTH: 1}
    computed = Counter(e.stage for e in report.events if not e.cached)
    assert all(computed[stage] == 2 for stage in loads)
